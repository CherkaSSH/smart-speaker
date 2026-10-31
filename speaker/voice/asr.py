"""Client for nemo-speech's realtime transcription WebSocket.

Audio streams continuously and each utterance gets its own server session, so
every utterance starts from a clean state. Measured on loopback recordings of
eight wake-phrase commands (see README):

- Each session starts with PREROLL seconds of silence. Without it the
  streaming model dropped or garbled the first words in 2 of 8 ("Поставь
  таймер ..." with no wake phrase); with it, 8 of 8.
- Audio must not arrive faster than real time at the start of speech: the
  first 1.5 s at 2x found 5 of 8 wake phrases, at 1.5x 7 of 8, at 1.25x 8 of
  8. Backlog is sent at MAX_SPEED at most.
- Language "auto" hears "Эй" as English "Hey" and writes the wake phrase in
  Latin ("Ei dalba jop"), so the wake stream is pinned to Russian; a second
  stream in "auto" catches English commands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable

import aiohttp

log = logging.getLogger("voice.asr")
RATE = 16000
PREROLL = 2.5  # seconds of silence opening each session
MAX_SPEED = 1.2  # fastest catch-up pace, x real time
CARRY_AFTER = 0.3  # audio from this long after an utterance's last word moves to the next session


@dataclass
class Final:
    text: str
    start: float  # monotonic time the first word began (best effort)
    end: float  # monotonic time the last word ended
    stream: str  # which ASR stream heard it


class ASR:
    def __init__(self, name: str, cfg: dict, language: str, boost_phrases: list[str], boost: float,
                 on_partial: Callable[[str], Awaitable[None]] | None, on_final: Callable[[Final], Awaitable[None]]):
        self.name = name
        self.url = cfg["url"]
        self.endpoint_ms = int(cfg["endpoint_ms"])
        self.language = language
        self.boost_phrases = boost_phrases
        self.boost = boost
        self.on_partial = on_partial
        self.on_final = on_final
        self.partial = ""  # text heard so far in the current utterance
        self.finishing = 0  # finals still being handled
        self.active = True  # Mic sends audio only while this is true
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._carry: list[tuple[float, bytes]] = []  # audio for the next session
        self.audio: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue(maxsize=600)

    def activate(self, preroll: list[tuple[float, bytes]]) -> None:
        while not self.audio.empty():
            self.audio.get_nowait()
        for item in preroll:
            self.audio.put_nowait(item)
        self.active = True

    def deactivate(self) -> None:
        self.active = False
        self._carry = []
        while not self.audio.empty():
            self.audio.get_nowait()
        if self._ws is not None:
            asyncio.create_task(self._ws.close())  # discard the old utterance/session state

    def _session(self) -> dict:
        s = {"sample_rate": RATE, "language": self.language, "endpointing_ms": self.endpoint_ms,
             "word_timestamps": True}
        if self.boost > 0 and self.boost_phrases:
            s["speech_contexts"] = [{"phrases": self.boost_phrases, "boost": self.boost}]
        return s

    async def run(self) -> None:
        async with aiohttp.ClientSession() as http:
            while True:
                try:
                    await self._one_session(http)
                except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError) as e:
                    log.warning("%s: ASR connection failed (%s); retrying", self.name, e)
                    await asyncio.sleep(1)

    async def _one_session(self, http: aiohttp.ClientSession) -> None:
        async with http.ws_connect(self.url, heartbeat=20) as ws:
            self._ws = ws
            await ws.receive_json(timeout=5)  # session.created
            await ws.send_json({"type": "session.update", "session": self._session()})
            await ws.send_bytes(bytes(int(PREROLL * RATE) * 2))
            base: list[float] = []  # monotonic time of session audio t=0 (the start of the preroll)
            self.partial = ""
            recent: deque[tuple[float, bytes]] = deque(maxlen=30)  # last 3 s sent, for carry-over

            async def pump() -> None:
                started, sent = time.monotonic(), 0.0
                carry, self._carry = self._carry, []
                while True:
                    captured, pcm = carry.pop(0) if carry else await self.audio.get()
                    recent.append((captured, pcm))  # before any await: a cancelled send still carries over
                    if not base:
                        base.append(captured - PREROLL)
                    # Never faster than MAX_SPEED x real time (see the module docstring).
                    ahead = started + sent / MAX_SPEED - time.monotonic()
                    if ahead > 0:
                        await asyncio.sleep(ahead)
                    await ws.send_bytes(pcm)
                    sent += len(pcm) / 2 / RATE
                    # The server caps audio per session (512 MiB); start over during silence.
                    if sent > 600 and not self.partial:
                        await ws.close()
                        return

            pumper = asyncio.create_task(pump())
            try:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    ev = json.loads(msg.data)
                    kind = ev.get("type", "")
                    if kind.endswith(".delta"):
                        if ev.get("delta"):
                            self.partial += ev["delta"]
                            if self.on_partial:
                                await self.on_partial(self.partial)
                    elif kind.endswith(".completed"):
                        text = (ev.get("transcript") or "").strip()
                        t0 = base[0] if base else 0.0
                        words = ev.get("words") or []
                        if words:
                            start, end = t0 + float(words[0]["start"]), t0 + float(words[-1]["end"])
                        else:
                            end = t0 + float(ev.get("audio_processed") or 0) - self.endpoint_ms / 1000
                            start = end - len(text) * 0.06
                        # Audio this session got after the utterance may already hold the
                        # start of the next one: hand it to the next session.
                        pumper.cancel()
                        self._carry = [c for c in recent if c[0] > end + CARRY_AFTER]
                        if text:
                            self.finishing += 1
                            asyncio.create_task(self._deliver(Final(text, start, end, self.name)))
                        return  # next utterance, next session
                    elif kind == "error":
                        log.error("%s: ASR error: %s", self.name, ev)
                        return
            finally:
                if self._ws is ws:
                    self._ws = None
                self.partial = ""
                pumper.cancel()

    async def _deliver(self, f: Final) -> None:
        # Runs beside the next session so that one starts listening at once.
        try:
            await self.on_final(f)
        except Exception:
            log.exception("%s: handling %r failed", self.name, f.text)
        finally:
            self.finishing -= 1
