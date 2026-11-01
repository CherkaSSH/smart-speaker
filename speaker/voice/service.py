"""The voice half: microphone -> ASR -> wake phrase -> brain -> TTS -> speaker.

States, as on Yandex Station:
  idle       waiting for the wake phrase
  listening  taking a follow-up without the wake phrase
  thinking   the brain is working on it; new speech replaces the request
  speaking   playing the answer; new speech or a stop word interrupts
  ringing    a timer or alarm is going off; new speech or a stop word silences it
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from difflib import SequenceMatcher

import aiohttp
import numpy as np
from num2words import num2words

from ..config import Config
from ..lang import detect
from . import sounds
from .asr import ASR, Final
from .audio import Mic, Player
from .echo import EchoCanceller, Reference
from .gate import Gate
from .music import MusicPlayer
from .tts import TTS, normalize
from .wake import WakeMatcher, phonetic, words

log = logging.getLogger("voice")

# Our own voice reaching the mic is ignored when the transcript overlaps playback
# and shares at least this fraction of words with what we said.
ECHO_WORD_OVERLAP = 0.5
ECHO_WORD_SIMILARITY = 0.75  # a heard word this close to one we said counts as our own voice
ECHO_RESIDUE_WORDS = 2  # after removing our voice, this few leftover words are echo noise too
CONVERSATION_STATES = {"listening", "speaking", "thinking", "ringing"}
RING_SECONDS = 60
ECHO_TAIL = 0.5  # seconds after our last sound that the mic may still hear it (buffers, room echo)
TALK_HOLD = 2.5  # seconds the music stays ducked after the user was last heard over it


class Voice:
    def __init__(self, cfg: Config, loop: asyncio.AbstractEventLoop):
        self.cfg = cfg
        w, v = cfg["wake"], cfg["voice"]
        self.follow_up_seconds = float(v["follow_up_seconds"])
        self.listen_seconds = float(v["listen_seconds"])
        self.brain_url = v["brain_url"]
        self.wake = WakeMatcher(w["phrases"], float(w["threshold"]))
        self.stop_words = {" ".join(words(s)) for s in w["stop_words"]}
        # One record of everything we play, so the mic can subtract it (echo cancellation).
        self.reference = Reference() if cfg["audio"]["echo_cancellation"] else None
        self.player = Player(cfg["audio"], loop, self.reference)
        self.music = MusicPlayer(cfg["audio"], cfg["music"], loop, self.on_music_event, self.reference)
        self.talk_until = 0.0  # the user is talking over the music until then: keep it ducked
        self._music_start: asyncio.Task | None = None
        a = cfg["asr"]
        self.asr = ASR("main", a, a["language"], w["phrases"], float(w["boost"]), self.on_partial, self.on_main_final)
        self.asr_alt = ASR("auto", a, "auto", [], 0, None, self.on_alt_final) if a["detect_english"] else None
        self.streams = [self.asr] + ([self.asr_alt] if self.asr_alt else [])
        self.tts = TTS(cfg, cfg["tts"])
        self.gate = Gate(cfg, self.on_small_wake, lambda: self.player.last_sound + ECHO_TAIL,
                         self.music.level, self.on_user_talking) if cfg["gate"]["enabled"] else None
        self.recognition_active = self.gate is None
        if self.gate:
            for stream in self.streams:
                stream.active = False
        self.mic = Mic(cfg["audio"], loop,
                       [self.gate.audio] if self.gate else [s.audio for s in self.streams],
                       self.streams if self.gate else [],
                       EchoCanceller(self.reference, cfg["audio"]["echo_model"]) if self.reference else None)
        self.acks = {lang: self.tts.synth(text) for lang, text in cfg["character"]["ack"].items()}
        self.ack_texts = cfg["character"]["ack"]

        self.state = "idle"
        self.turn_id: str | None = None
        self.listen_until = 0.0
        self.heard_at = 0.0  # when the user stopped speaking, for latency logs
        self.first_audio_pending = False
        self.speech: asyncio.Queue[tuple[str | None, str | None, bool]] = asyncio.Queue()
        self.spoken: list[tuple[float, float, list[str]]] = []  # (start, end, phonetic words) of our recent speech
        self.play_end = 0.0  # estimated end of everything queued for playback
        self.ring_task: asyncio.Task | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.alt_finals: list[Final] = []  # recent finals of the language-detecting stream
        self.ignore_before = 0.0  # audio at or before this time already belongs to an older turn
        self.handled: list[tuple[float, float]] = []  # time spans of utterances already acted on
        self.recognition_started = 0.0

    # Helpers -------------------------------------------------------------
    def _set_streams_active(self, active: bool) -> None:
        """Large ASR on or off. The English stream's text is only used in a
        conversation, so it stays off while the gate's wake is being verified."""
        if self.gate is None:
            return
        if active != self.recognition_active:
            self.recognition_active = active
            self.gate.active = active
            log.info("large ASR %s", "active" if active else "idle")
            if not active and self.asr.partial:
                log.info("dropped unfinished %r", self.asr.partial)
            if not active:
                self.gate.reset()
            self._update_duck()
        wanted = [(self.asr, active)]
        if self.asr_alt:
            wanted.append((self.asr_alt, active and self.state in CONVERSATION_STATES))
        for stream, on in wanted:
            if on and not stream.active:
                stream.activate(self.gate.preroll())
            elif not on and stream.active:
                stream.deactivate()

    async def on_small_wake(self) -> None:
        if self.state == "idle":
            self.recognition_started = time.monotonic()
            self._set_streams_active(True)

    def set_state(self, state: str, why: str = "") -> None:
        if state != self.state:
            log.info("%s -> %s%s", self.state, state, f" ({why})" if why else "")
            self.state = state
            self._set_streams_active(state != "idle")
            self._update_duck()

    def _update_duck(self) -> None:
        """Music is quiet whenever we listen to or talk with the user."""
        self.music.duck(self.state != "idle" or self.recognition_active or self.player.playing
                        or time.monotonic() < self.talk_until)

    def on_user_talking(self) -> None:
        if self.music.state == "playing" and time.monotonic() >= self.talk_until:
            log.info("user talking over the music: ducking")
        self.talk_until = time.monotonic() + TALK_HOLD
        self._update_duck()

    def play(self, audio: np.ndarray, rate: int, text: str = "") -> None:
        if not len(audio):
            return
        now = time.monotonic()
        start = max(now, self.play_end)
        self.play_end = start + len(audio) / rate
        if text:
            # Match the words as they were spoken: "77 855" was said as "семьдесят семь
            # восемьсот пятьдесят пять", which is what the mic transcript will contain.
            said = words(text) + words(normalize(text, detect(text)))
            self.spoken.append((start, self.play_end, [phonetic(x) for x in said]))
            self.spoken = [s for s in self.spoken if s[1] > now - 30]
        self.player.play(audio, rate)

    def without_echo(self, f: Final) -> str:
        """The user's part of a transcript.

        If the transcript overlaps our playback and mostly repeats what we said,
        it is our own voice coming back through the mic; whatever words remain
        (a "хватит" said over the ring) are the user's.
        """
        tokens = words(f.text)
        lang = detect(f.text)
        # ASR may write digits although TTS spoke words ("5" vs "пять"). Keep
        # one group per original token so echo removal can still rebuild text.
        forms = [[phonetic(w) for w in (words(num2words(int(t), lang=lang)) if t.isdigit() else [t])]
                 for t in tokens]
        heard = [w for group in forms for w in group]
        said = {w for start, end, ws in self.spoken
                if f.start < end + 0.5 and f.end > start - 0.3 for w in ws}
        if self.music.state == "playing":
            said |= self.music.lyric_words  # the singer is not the user

        def echoed(w: str) -> bool:
            # ASR garbles our voice a little: "долла" for "доллара", "десятое" for "десятая".
            return w in said or len(w) >= 4 and any(SequenceMatcher(None, w, s).ratio() >= ECHO_WORD_SIMILARITY
                                                    for s in said if abs(len(s) - len(w)) <= 3)

        is_echo = [all(echoed(w) for w in group) for group in forms]
        if not heard or not said or sum(is_echo) / len(forms) < ECHO_WORD_OVERLAP:
            return f.text
        keep = [not e for e in is_echo]
        # Unless we said the wake phrase ourselves, a wake phrase in the mix is the user's.
        m = self.wake.find(f.text)
        if m and not self.wake.find(" ".join(said)):
            keep[m.start:m.end] = [True] * (m.end - m.start)
        return " ".join(t for t, k in zip(tokens, keep) if k)

    def is_stop(self, text: str) -> bool:
        """Only stop words ("стоп", "хватит!"), allowing ASR slips like "стопэ"."""
        ws = [w for w in words(text) if not w.isdigit()]  # echo leftovers: "10 хватит"
        if not ws:
            return False
        if " ".join(ws) in self.stop_words:
            return True
        singles = [phonetic(s) for s in self.stop_words if " " not in s]
        return all(any(phonetic(w).startswith(s) or SequenceMatcher(None, phonetic(w), s).ratio() >= 0.8
                       for s in singles) for w in ws)


    async def send(self, msg: dict) -> None:
        if self.ws is not None and not self.ws.closed:
            await self.ws.send_str(json.dumps(msg, ensure_ascii=False))

    async def interrupt(self, why: str) -> None:
        """Silence everything and drop the current answer."""
        if self.ring_task and not self.ring_task.done():
            self.ring_task.cancel()
        was_busy = self.state in ("thinking", "speaking")
        self.turn_id = None
        self.player.stop()
        self.play_end = 0.0
        if was_busy:
            await self.send({"type": "cancel"})
        log.info("interrupted: %s", why)

    def listen(self, seconds: float, why: str) -> None:
        self.listen_until = max(time.monotonic(), self.play_end) + seconds
        self.set_state("listening", why)

    # ASR events ------------------------------------------------------------
    async def on_partial(self, text: str) -> None:
        if self.state == "idle" and self.recognition_active:
            # A real command can run past the false-wake timeout. Keep the
            # large stream alive for as long as Nemotron hears speech.
            self.recognition_started = time.monotonic()
        if self.state == "listening":
            # The user is talking: don't time out mid-sentence.
            self.listen_until = max(self.listen_until, time.monotonic() + 2.5)
        # Busy-state interruption waits for a final: partials are too often our
        # own TTS echo, and finals let without_echo remove the playback first.

    def _overlaps(self, f: Final, spans: list[tuple[float, float]]) -> bool:
        return any(s < f.end + 0.3 and e > f.start - 0.3 for s, e in spans)

    def _during_playback(self, f: Final) -> bool:
        """English text heard while we spoke can't be checked against our Russian words:
        the English stream turns our own voice into "two thousand and twenty six"."""
        return self._overlaps(f, [(s, e) for s, e, _ in self.spoken])

    async def on_main_final(self, f: Final) -> None:
        if self.state in CONVERSATION_STATES and self.asr_alt is not None and not self.wake.find(f.text):
            # The language-detecting stream finishes the same utterance about now.
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline and not self._alt_for(f):
                await asyncio.sleep(0.03)
            alt = self._alt_for(f)
            if alt and detect(alt.text) == "en" and not self.wake.find(alt.text) and not self._during_playback(alt):
                log.info("English %r (the Russian stream heard %r)", alt.text, f.text)
                f = alt
        if self._overlaps(f, self.handled):
            return
        self.handled = [s for s in self.handled if s[1] > time.monotonic() - 30] + [(f.start, f.end)]
        await self.on_final(f)

    def _alt_for(self, f: Final) -> Final | None:
        hits = [g for g in self.alt_finals if self._overlaps(g, [(f.start, f.end)])]
        return hits[-1] if hits else None

    async def on_alt_final(self, f: Final) -> None:
        self.alt_finals = [g for g in self.alt_finals if g.end > time.monotonic() - 30] + [f]
        # The wake phrase is the Russian stream's job (this one garbles it into "E dalba jok").
        if self.state not in CONVERSATION_STATES or detect(f.text) != "en" or self.wake.find(f.text) \
                or self._during_playback(f):
            return
        if self.asr.partial or self.asr.finishing:
            return  # the Russian stream is still on this utterance and will pick the English text
        # The Russian stream heard nothing at all; give it a moment to be sure.
        await asyncio.sleep(0.3)
        if self.asr.partial or self.asr.finishing or self._overlaps(f, self.handled):
            return
        self.handled.append((f.start, f.end))
        await self.on_final(f)

    async def on_final(self, f: Final) -> None:
        now = time.monotonic()
        if f.end <= self.ignore_before + 0.05:
            log.info("ignored stale audio %r", f.text)
            return
        self.ignore_before = f.end
        log.info("heard %r (ended %.2fs ago)", f.text, now - f.end)
        text = self.without_echo(f)
        echo = text != f.text
        if echo:
            log.info("our own voice removed; remaining speech: %r", text)
        if not text:
            return
        woke, rest = self.wake.strip(text)
        if woke:
            await self.interrupt("wake phrase")
            if self.is_stop(rest):
                self.set_state("idle", "stop")
            elif len(re.sub(r"\W", "", rest)) < 2:
                lang = "en" if detect(text) == "en" else "ru"
                ack = self.acks.get(lang, next(iter(self.acks.values())))
                self.play(ack, self.tts.rate, self.ack_texts.get(lang, ""))
                self.listen(self.listen_seconds, "wake phrase alone")
            else:
                await self.utterance(rest, f.end)
            return
        if self.state in ("ringing", "speaking", "thinking") and len(words(text)) <= 4 \
                and any(self.is_stop(w) for w in words(text)):
            # Short and holding a stop word: "дойми хватит" is a misheard ring plus "хватит".
            await self.interrupt("stop word")
            self.set_state("idle", "stop word")
            return
        if self.is_stop(text) and self.state != "idle":
            await self.interrupt("stop word")
            self.set_state("idle", "stop word")
            return
        if echo and len(words(text)) <= ECHO_RESIDUE_WORDS:
            # Mostly our own voice; one or two leftovers are misheard echo ("кокаин" for
            # "KuCoin"), not a command. Stop words and the wake phrase were handled above.
            log.info("ignored echo leftovers %r", text)
            return
        if self.state in CONVERSATION_STATES and (not echo or self.state != "listening"):
            if self.state != "listening":
                # A real user word left after removing our voice is a barge-in, even
                # without the wake phrase. Pure echo has no words left and returns above.
                await self.interrupt("user speech")
            await self.utterance(text, f.end)
            return
        log.info("ignored: no wake phrase")
        if self.state == "idle":
            self._set_streams_active(False)  # the small gate opened, but Nemotron rejected it

    async def utterance(self, text: str, ended: float) -> None:
        if self.ws is None or self.ws.closed:
            log.error("brain is not connected")
            self.play(self.tts.synth("Мозг не отвечает."), self.tts.rate, "Мозг не отвечает.")
            self.set_state("idle", "no brain")
            return
        self.turn_id = uuid.uuid4().hex[:8]
        self.heard_at = ended
        self.first_audio_pending = True
        self.set_state("thinking", text)
        await self.send({"type": "utterance", "id": self.turn_id, "text": text, "lang": detect(text)})

    # Brain events ------------------------------------------------------------
    async def on_brain(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "volume":
            self.player.set_volume(int(msg["level"]))
            self.music.set_volume(int(msg["level"]))
            log.info("volume %s", msg["level"])
        elif kind == "music":
            await self.on_music_command(msg)
        elif kind == "say" and msg.get("id") == self.turn_id:
            await self.speech.put((msg["id"], msg["text"], False))
        elif kind == "turn_end" and msg.get("id") == self.turn_id:
            await self.speech.put((msg["id"], None, bool(msg.get("expect_reply"))))
        elif kind == "ring":
            await self.interrupt("ring")
            self.ring_task = asyncio.create_task(self.ring(msg["text"]))

    async def on_music_command(self, msg: dict) -> None:
        action = msg["action"]
        if self._music_start and not self._music_start.done():
            self._music_start.cancel()  # a newer command wins over a track still waiting to start
        if action == "play":
            self._music_start = asyncio.create_task(self._start_music(msg["track"]))
        elif action == "pause":
            self.music.pause()
        elif action == "resume":
            self.music.resume()
        elif action == "stop":
            self.music.stop()
        self._update_duck()

    async def _start_music(self, track: dict) -> None:
        # Let the answer ("Включаю ...") finish before the song starts.
        deadline = time.monotonic() + 15
        await asyncio.sleep(0.3)
        while (self.turn_id is not None or self.player.playing) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        self.music.play(track)
        self._update_duck()

    async def on_music_event(self, event: str, track: dict, detail: str) -> None:
        log.info("music %s: %s %s", event, track.get("title"), detail[:120])
        await self.send({"type": "music_event", "event": event, "id": track["id"], "detail": detail})

    async def speaker(self) -> None:
        """Synthesize answer sentences in order, ahead of playback."""
        while True:
            turn, text, expect_reply = await self.speech.get()
            if turn != self.turn_id:
                continue
            if text is None:  # end of the answer: wait for it to finish playing
                await self.player.wait_idle()
                if turn != self.turn_id:
                    continue
                self.turn_id = None
                if expect_reply:
                    self.play(sounds.listen_start(), sounds.RATE)
                    self.listen(self.follow_up_seconds, "answer was a question")
                else:
                    self.set_state("idle", "answered")
                continue
            t0 = time.monotonic()
            audio = await asyncio.to_thread(self.tts.synth, text)
            if turn != self.turn_id:
                continue
            self.set_state("speaking")
            self.play(audio, self.tts.rate, text)
            if self.first_audio_pending:
                self.first_audio_pending = False
                log.info("first audio %.2fs after the user stopped speaking (TTS %.2fs)",
                         time.monotonic() - self.heard_at, time.monotonic() - t0)
            log.info("say: %s", text)

    async def ring(self, text: str) -> None:
        self.set_state("ringing", text)
        audio = await asyncio.to_thread(self.tts.synth, text)
        try:
            deadline = time.monotonic() + RING_SECONDS
            while time.monotonic() < deadline:
                self.play(sounds.ring(), sounds.RATE)
                self.play(audio, self.tts.rate, text)
                await self.player.wait_idle()
            self.set_state("idle", "ring timed out")
        except asyncio.CancelledError:
            self.player.stop()
            if self.state == "ringing":
                self.set_state("idle", "ring stopped")
            raise

    async def watchdog(self) -> None:
        """Close the listening window when nobody speaks."""
        while True:
            await asyncio.sleep(0.2)
            self._update_duck()  # speech playback and the talk hold end on their own
            if self.state == "listening" and time.monotonic() > self.listen_until and not self.player.playing:
                self.play(sounds.listen_end(), sounds.RATE)
                self.set_state("idle", "nobody spoke")
            elif self.state == "idle" and self.recognition_active \
                    and time.monotonic() > self.recognition_started + 4:
                self._set_streams_active(False)

    async def brain_link(self) -> None:
        async with aiohttp.ClientSession() as http:
            while True:
                try:
                    async with http.ws_connect(self.brain_url, heartbeat=20) as ws:
                        self.ws = ws
                        log.info("connected to brain")
                        await ws.send_str(json.dumps({"type": "hello"}))
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self.on_brain(json.loads(msg.data))
                except (aiohttp.ClientError, ConnectionError) as e:
                    log.warning("brain connection: %s", e)
                self.ws = None
                await asyncio.sleep(1)

    async def run(self) -> None:
        self.mic.start()
        log.info("ready; say %r", self.cfg["wake"]["phrases"][0])
        try:
            streams = [self.asr.run()] + ([self.asr_alt.run()] if self.asr_alt else [])
            gate = [self.gate.run()] if self.gate else []
            await asyncio.gather(*streams, *gate, self.brain_link(), self.speaker(), self.watchdog())
        finally:
            self.mic.stop()
            self.player.close()
            self.music.close()


async def run(cfg: Config) -> None:
    voice = Voice(cfg, asyncio.get_running_loop())
    await voice.run()
