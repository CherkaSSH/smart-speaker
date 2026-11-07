"""The thinking half: conversation, tools, timers. Talks to voice clients over a WebSocket.

Protocol (JSON text frames):
  voice -> brain  {"type": "hello"}
                  {"type": "utterance", "id": str, "text": str, "lang": "ru"|"en",
                   "speaker": str}             the recognised voice's name; optional, absent = unknown
                  {"type": "cancel"}            stop the current answer (barge-in, "стоп")
                  {"type": "listening", "id": str}   a follow-up window opened after turn or speak `id`
                  {"type": "music_position", "id": str, "provider": str, "position": float}
                                                seconds into the current track (books resume, lyrics)
                  {"type": "voices", "names": [str]}  the enrolled voice profiles (also sent after
                                                enroll and forget_voice), a reply to brain's request
                  {"type": "enroll_result", "id": str, "name": str, "ok": bool, "error": str}
                                                a voice enrollment finished (the service speaks it too)
  brain -> voice  {"type": "volume", "level": 0..10, "user": bool}
                  {"type": "night", "on": bool|null}   night mode forced on/off, null = by the clock
                  {"type": "say", "id": str, "text": str, "lang": str}
                  {"type": "turn_end", "id": str, "expect_reply": bool}
                  {"type": "voices"}            which voice profiles exist: the voice service answers
                  {"type": "enroll", "id": str, "name": str, "lang": str}
                                                record a voice profile: the service says three prompts,
                                                takes the answers, averages their embeddings and stores
                                                them; the profile never leaves the machine
                  {"type": "forget_voice", "name": str}   delete a voice profile
                  {"type": "ring", "text": str, "lang": str}   a timer or alarm went off; one that went off
                                           while no voice service was connected comes when one connects
                  {"type": "speak", "id": str, "sentences": [str], "lang": str, "queue": bool,
                   "expect_reply": bool, "loud": bool}   said on the speaker's own initiative («Включить мою
                                           волну?», an agent's message); with `queue` it waits until the speaker
                                           is free, otherwise it is dropped when busy; a reply comes as an
                                           utterance. `loud` ignores the night volume: it is an alarm

Coding-agent threads (omp sessions) connect to /omp; see agents.py and omp/speaker.ts.

HTTP: POST /ask {"text", "lang"?, "speak"?, "big"?, "speaker"?} answers in JSON,
for testing without audio; "speak": true also sends it to the voice service.

Text chat (Telegram, Discord) runs beside the voice: see chat.py. Chat messages go through `answer` in
their own tasks, with their own `speaker` and `pending` (see `chat_scope`), never through `start_turn`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from aiohttp import WSMsgType, web

from ..config import Config
from ..lang import detect as detect_lang
from . import intents, speech
from .agents import Agents
from .chat import Chat
from .escalate import RemoteLLM
from .history import History, format_fact
from .llm import LLM
from .memory import Memory
from .music import MY_WAVE, Music, asks_for_music
from .news import News
from .rates import Rates
from .shopping import Shopping
from .timers import Item, Scheduler
from .tools import BIG_MODEL_SCHEMA, SCHEMAS, WEB_SEARCH_SCHEMA, Tools
from .text import SentenceSplitter
from .websearch import WebSearch

log = logging.getLogger("brain")

# Who is talking and which conversation a question belongs to. A voice turn and each chat user run in their
# own task, so each sees its own values; "" is the voice. The voice is trusted fully ("admin"); a chat user
# may be a plain "user", who cannot reach the coding agents (see `answer`).
_speaker_var: ContextVar[str | None] = ContextVar("speaker", default=None)
_scope_var: ContextVar[str] = ContextVar("scope", default="")
_role_var: ContextVar[str] = ContextVar("role", default="admin")

CONFIRM_SECONDS = 30  # how long a "should I ...?" question waits for its yes/no
MISSED_RING_SECONDS = 90  # a ring kept this briefly (a voice restart) still rings normally, without "missed"
MAX_MISSED_RINGS = 20
_YES = {"да", "ага", "угу", "давай", "конечно", "спроси", "спрашивай", "можно", "хорошо", "ладно", "пусть",
        "yes", "yeah", "yep", "sure", "ok", "okay", "please", "go"}
_NO = {"нет", "не", "неа", "no", "nope", "don't", "dont"}


def yes_no(text: str) -> bool | None:
    """True/False for a short yes or no ("да", "не надо", "ладно, не надо"); None for anything else."""
    words = re.findall(r"[\w']+", text.casefold().replace("ё", "е"))
    if not words or len(words) > 5:
        return None
    if any(w in _NO for w in words):
        return False
    if any(w in _YES for w in words):
        return True
    return None


def ring_text(kind: str, label: str, seconds: int, lang: str, who: str = "") -> str:
    """What the speaker says when a timer, alarm or reminder rings.

    `who` is the recognised person who set it (see voice profiles): a reminder or
    alarm for a known voice is addressed to them by name.
    """
    ru = lang == "ru"
    if label:
        if ru:
            return f"{who}, напоминаю: {label}." if who else f"Напоминаю: {label}."
        return f"{who}, reminder: {label}." if who else f"Reminder: {label}."
    if kind == "alarm":
        if ru:
            return f"{who}, подъём, будильник!" if who else "Подъём, будильник!"
        return f"{who}, wake up, that's your alarm!" if who else "Wake up, that's your alarm!"
    return f"Таймер на {speech.duration(seconds, 'ru', accusative=True)} всё." if ru else \
        f"Your {speech.duration(seconds, 'en')} timer is done."


def missed_ring_text(kind: str, label: str, seconds: int, due: float, lang: str) -> str:
    """A ring no voice service heard when it was due: say what was missed."""
    ru = lang == "ru"
    at = dt.datetime.fromtimestamp(due)
    if label:
        return f"Пропущенное напоминание: {label}." if ru else f"A missed reminder: {label}."
    if kind == "alarm":
        return f"Пропущенный будильник на {speech.clock(at.hour, at.minute, 'ru')}." if ru else \
            f"A missed alarm for {speech.clock(at.hour, at.minute, 'en')}."
    return f"Пропущенный таймер на {speech.duration(seconds, 'ru', accusative=True)}." if ru else \
        f"A missed {speech.duration(seconds, 'en')} timer."


class NeedsConfirmation(Exception):
    """Raised from a tool call that must wait for the user's yes/no."""

    def __init__(self, kind: str, args: dict):
        super().__init__(kind)
        self.kind = kind  # "big" or "search"
        self.kind_tool = "ask_big_model" if kind == "big" else "web_search"
        self.tool_args = args  # not `args`: Exception turns that into a tuple


@dataclass
class Pending:
    kind: str  # "big", "search", "music" or "reply"
    messages: list[dict]  # the conversation up to and including the paused tool call
    call_id: str
    args: dict
    text: str  # the user's original request
    lang: str
    search: asyncio.Task | None = None  # web search already running while the user answers
    expires: float = field(default_factory=lambda: time.monotonic() + CONFIRM_SECONDS)
    # "reply": takes the user's next words as they are (None when they never came) and returns what to say
    reply: Callable[[str | None], Awaitable[str | None]] | None = None

    def cancel(self) -> None:
        if self.search:
            self.search.cancel()
        if self.reply:
            asyncio.ensure_future(self.reply(None))


@dataclass
class Turn:
    """One utterance and what came out of it, for the history and «повтори»."""
    text: str
    lang: str
    speaker: str | None = None
    said: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)  # compact tool calls, for follow-ups


RULES = """
You are the voice of a home smart speaker, like Yandex Station's Alice. Everything you write is spoken aloud.
- Answer in the reply language given in the square brackets (Russian or English).
- Be brief: one or two short sentences. Longer only when asked (a story, a recipe, a joke).
- Plain speech only: no markdown, lists, emoji, links or URLs.
- Never think out loud. Work things out silently or with a tool (calculate also counts days to a date), then say only the result.
- Use the tools for timers, alarms, volume, weather, arithmetic and facts. Never say you did something without calling its tool.
- For questions about real people, places, books, history or science, call lookup instead of guessing.
- Save durable personal facts with save_memory. Questions about the user, their pets, family, things or preferences are answered from saved facts or search_memory, never lookup.
- When ask_big_model is available, use it for requests beyond your reasoning or knowledge ability; the user is asked to confirm first.
- For current events, news, prices, schedules or anything else Wikipedia wouldn't have, call web_search with an English objective and English queries, then answer from the results in the reply language.
- Each user message starts with the current date, time, reply language and, when known, who is speaking, in square brackets. Don't mention the date or time unless asked. You may address that person by name.
- The conversation may include the last few exchanges from the same person, with their dates. Use them for follow-ups like «а завтра?» or «а он?»; [facts: ...] in an earlier user message lists the tool calls behind its answer. Square-bracket notes are machine data: never read them aloud. Ignore the history when the new request is unrelated.
- Translate short phrases directly: for «как по-английски …», «переведи на …», «как будет … по-немецки» say only the translation in one short sentence, without defining or explaining the word and without tools.
- Use play_music only when the user asks to play music: a song, an artist, "some music" or a radio station. Stories, fairy tales and poems you tell yourself. music_control pauses, resumes, skips (next, previous) or stops it.
- You can't control other devices (lights, TV); say so briefly if asked.
"""

# Said while a slow tool runs, unless the model already said something first.
PREFACES = {
    "web_search": {"ru": "Сейчас поищу.", "en": "Let me look that up."},
    "lookup": {"ru": "Сейчас гляну.", "en": "Let me check."},
    "get_weather": {"ru": "Смотрю погоду.", "en": "Checking the weather."},
    "ask_big_model": {"ru": "Спрошу большую модель.", "en": "Asking the big model."},
}

_RU_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


class Brain:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        b = cfg["brain"]
        self.state_path = cfg.file(b["state_file"])
        self.fast_commands = bool(b["fast_commands"])
        self.volume = int(cfg["audio"]["volume"])
        self.missed: list[dict] = []  # rings that went off while no voice service was connected
        self.night_override: bool | None = None  # «включи/выключи ночной режим»; None = follow the clock
        self.scheduler = Scheduler(self._fire, self._save)
        self.tools = Tools(self)
        self.llm = LLM(cfg["llm"])
        self.system = cfg["character"]["prompt"].strip() + "\n" + RULES
        # The remote model gets no tool definitions: it must not try to call any.
        self.remote_system = self.system + "- In this conversation you have no tools: answer directly from what you know " \
                                           "and from any information given in the message.\n"
        r = cfg["remote_llm"]
        self.memory = Memory(cfg["memory"], cfg.file(cfg["memory"]["state_file"]))
        self.remote = RemoteLLM(r) if r["enabled"] else None
        self.big_triggers = r["triggers"]
        b = cfg["brain"]
        host = "127.0.0.1" if b["host"] in ("0.0.0.0", "::", "") else b["host"]
        self.agents = Agents(self, cfg["omp"], f"ws://{host}:{b['port']}/omp")
        self.web = WebSearch(cfg["web_search"])
        self.music = Music(self, cfg["music"], cfg.file(cfg["music"]["library"]))
        self.news = News(cfg["news"])
        self.rates = Rates()
        self.shopping = Shopping(cfg.file(cfg["shopping"]["state_file"]))
        self.chat = Chat(self, cfg)  # Telegram and Discord; started by serve()
        self.confirm_big = bool(r["confirm"])
        self.offer_big_for_search = bool(cfg["web_search"]["offer_big_model"])
        self.schemas = SCHEMAS + ([WEB_SEARCH_SCHEMA] if self.web.enabled else []) \
            + ([BIG_MODEL_SCHEMA] if self._big_ready else [])
        self._pendings: dict[str, Pending] = {}  # a question the speaker asked and awaits an answer to, per conversation
        self.awaiting: dict[str, Pending] = {}  # questions sent to the voice service, not yet asked aloud
        self.clients: set[web.WebSocketResponse] = set()
        self.turn: asyncio.Task | None = None
        self.voice_names: list[str] = []  # enrolled voice profiles, as the voice service last listed them
        self.history = History(int(b["history_minutes"]))
        self.last_answer: dict[str | None, str] = {}  # for «повтори», per speaker
        self._load()

    @property
    def _big_ready(self) -> bool:
        return bool(self.remote and self.remote.configured)

    @property
    def speaker(self) -> str | None:
        """Who is talking in the current turn (a recognised voice or a chat user); None if unknown."""
        return _speaker_var.get()

    @speaker.setter
    def speaker(self, name: str | None) -> None:
        _speaker_var.set(name)

    @property
    def pending(self) -> Pending | None:
        """The question awaiting an answer in the current conversation (the voice's, or a chat user's)."""
        return self._pendings.get(_scope_var.get())

    @pending.setter
    def pending(self, p: Pending | None) -> None:
        if p is None:
            self._pendings.pop(_scope_var.get(), None)
        else:
            self._pendings[_scope_var.get()] = p

    @property
    def in_chat(self) -> bool:
        """True while answering a chat message, False for the voice."""
        return _scope_var.get() != ""

    @property
    def role(self) -> str:
        """"admin" for the voice and chat admins, "user" for plain chat users (no coding agents)."""
        return _role_var.get()

    @contextmanager
    def chat_scope(self, scope: str, speaker: str | None, role: str = "admin"):
        """Run `answer` for a chat user: their own speaker name, pending question and role, apart from
        the voice's and everyone else's, so concurrent conversations cannot answer each other's questions
        and a "user" never gets what only an "admin" may do."""
        tokens = _scope_var.set(scope), _speaker_var.set(speaker), _role_var.set(role)
        try:
            yield
        finally:
            _role_var.reset(tokens[2])
            _speaker_var.reset(tokens[1])
            _scope_var.reset(tokens[0])

    # State -------------------------------------------------------------
    def _load(self) -> None:
        if self.state_path.exists():
            d = json.loads(self.state_path.read_text())
            self.volume = d.get("volume", self.volume)
            self.missed = d.get("rings", [])  # rings no voice service heard yet
            self.night_override = d.get("night")
            self.scheduler.restore(d.get("items", []))

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"volume": self.volume, "items": self.scheduler.dump(),
                                   "rings": self.missed, "night": self.night_override}, ensure_ascii=False))
        tmp.replace(self.state_path)

    async def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except ConnectionError:
                self.clients.discard(ws)

    async def set_volume(self, level: int, *, user: bool = True) -> None:
        self.volume = max(0, min(10, int(level)))
        self._save()
        await self.broadcast({"type": "volume", "level": self.volume, "user": user})

    async def set_night(self, on: bool | None) -> None:
        """Night mode: True/False from «включи/выключи ночной режим», None back to the clock."""
        self.night_override = on
        self._save()
        await self.broadcast({"type": "night", "on": on})

    async def _send_ring(self, text: str, lang: str) -> bool:
        """Ring on every connected voice client; False when there is nobody to hear it."""
        data = json.dumps({"type": "ring", "text": text, "lang": lang}, ensure_ascii=False)
        sent = False
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
                sent = True
            except ConnectionError:
                self.clients.discard(ws)
        return sent

    def _keep_ring(self, item: Item, due: float) -> None:
        """Remember a ring no voice client heard: it comes when one connects, even after a restart.
        `due` is when it went off (a repeating alarm already has its next due by now)."""
        # One entry per alarm: a daily alarm that fires again while no voice is there
        # replaces its older entry instead of piling up.
        key = (item.kind, item.label, item.seconds, dt.datetime.fromtimestamp(due).strftime("%H:%M"))
        entry = {"kind": item.kind, "label": item.label, "seconds": item.seconds, "due": due,
                 "lang": item.lang, "who": item.who}
        self.missed = [m for m in self.missed
                       if (m["kind"], m["label"], m["seconds"],
                           dt.datetime.fromtimestamp(m["due"]).strftime("%H:%M")) != key] + [entry]
        self.missed = self.missed[-MAX_MISSED_RINGS:]
        self._save()
        self.chat.notify_soon("rings", missed_ring_text(item.kind, item.label, item.seconds, due, item.lang))

    async def _deliver_missed(self) -> None:
        """A voice service just connected: ring what went off while none was there."""
        if not self.missed:
            return
        now = time.time()
        texts = [ring_text(m["kind"], m["label"], m["seconds"], m["lang"], m.get("who") or "")
                 if now - m["due"] <= MISSED_RING_SECONDS
                 else missed_ring_text(m["kind"], m["label"], m["seconds"], m["due"], m["lang"]) for m in self.missed]
        text = " ".join(texts)
        if not await self._send_ring(text, self.missed[-1]["lang"]):
            return  # nobody heard it after all: keep them for the next connection
        log.info("ring missed with no voice connected: %s", text)
        self.missed = []
        self._save()

    async def _fire(self, item: Item) -> None:
        if item.kind == "sleep":
            log.info("sleep timer: stopping the music")
            await self.music.control("stop")
            return
        due = item.due
        if item.repeat:  # a repeating alarm goes off again at its next occurrence
            self.scheduler.again(item)
        text = ring_text(item.kind, item.label, item.seconds, item.lang, item.who or "")
        if item.kind == "alarm" and item.music is not None:
            if not self.clients:
                log.info("alarm with music with no voice connected: kept until one connects: %s", text)
                self._keep_ring(item, due)
                return
            # The music is the alarm: a short line (which the voice finishes first), then the songs.
            # Both ignore the night volume: the alarm must wake the house.
            log.info("alarm with music %r: %s", item.music, text)
            await self.speak(text, item.lang, queue=True, loud=True)
            await self.music.play(item.music, False, item.lang, alarm=True)
            return
        if not await self._send_ring(text, item.lang):
            log.info("ring with no voice connected: kept until one connects: %s", text)
            self._keep_ring(item, due)
            return
        log.info("ring: %s", text)

    def expect_reply(self, lang: str, reply: Callable[[str | None], Awaitable[str | None]]) -> Pending:
        """A question whose answer is the user's next words, whatever they are."""
        return Pending("reply", [], "", {}, "", lang, reply=reply)

    async def speak(self, text: str, lang: str, *, queue: bool, pending: Pending | None = None,
                    loud: bool = False) -> None:
        """Say something on the speaker's own initiative. With `queue`, it waits until the speaker is
        free; otherwise it is dropped if the user is busy with it. With `pending`, the speaker listens
        afterwards and the reply resolves it, like the answers to the speaker's other questions.
        `loud` skips the night volume: an alarm says this."""
        sid = uuid.uuid4().hex[:8]
        now = time.monotonic()
        for old, p in list(self.awaiting.items()):  # never said: the voice service dropped or lost them
            if p.expires < now - 3600:
                del self.awaiting[old]
                p.cancel()
        if pending:
            self.awaiting[sid] = pending
        splitter = SentenceSplitter()
        sentences = splitter.feed(text) + splitter.flush()
        log.info("speaking%s: %s", " (queued)" if queue else "", text)
        await self.broadcast({"type": "speak", "id": sid, "sentences": sentences, "lang": lang, "queue": queue,
                              "expect_reply": pending is not None, "loud": loud})

    def _listening(self, sid: str) -> None:
        """The voice service said question `sid` and now listens: its answer is the next utterance."""
        p = self.awaiting.pop(sid, None)
        if p is None:
            return
        if self.pending:
            self.pending.cancel()
        p.expires = time.monotonic() + CONFIRM_SECONDS
        self.pending = p

    async def offer(self, kind: str, question: str, lang: str) -> None:
        """Ask on the speaker's own initiative (the songs the user asked for ended), unless the user
        is busy with it; the next utterance's yes or no resolves it."""
        await self.speak(question, lang, queue=False, pending=Pending(kind, [], "", {}, question, lang))

    # Voice profiles ------------------------------------------------------
    async def start_voice_enroll(self, name: str, lang: str) -> None:
        """Have the voice service record a profile of `name`: it says the prompts and takes the answers."""
        await self.broadcast({"type": "enroll", "id": uuid.uuid4().hex[:8], "name": name, "lang": lang})

    async def forget_voice(self, name: str) -> None:
        await self.broadcast({"type": "forget_voice", "name": name})
        self.voice_names = [n for n in self.voice_names if n.casefold() != name.casefold()]


    # Turns -------------------------------------------------------------
    def _stamp(self, lang: str, speaker: str | None = None, when: dt.datetime | None = None) -> str:
        now = when or dt.datetime.now()
        reply = "отвечай по-русски" if lang == "ru" else "reply in English"
        who = f"; говорит {speaker}" if speaker else ""
        return f"[{now:%Y-%m-%d %H:%M}, {_RU_WEEKDAYS[now.weekday()]}; {reply}{who}]"

    def _history_messages(self, speaker: str | None) -> list[dict]:
        """Earlier exchanges as real chat turns; the model sees the same stamps the user did."""
        messages = []
        for exchange in self.history.recent(speaker):
            facts = f" [facts: {'; '.join(exchange.facts)}]" if exchange.facts else ""
            messages.append({"role": "user", "content":
                             f"{self._stamp(exchange.lang, speaker, exchange.at)} {exchange.text}{facts}"})
            messages.append({"role": "assistant", "content": exchange.answer})
        return messages

    def _remember(self, turn: Turn) -> None:
        said = " ".join(s.strip() for s in turn.said if s.strip()).strip()
        if not said:
            return
        self.last_answer[turn.speaker] = said
        self.history.add(turn.speaker, turn.lang, turn.text, said, turn.facts)

    @staticmethod
    def _trigger(text: str, triggers: list[str]) -> str | None:
        folded = text.casefold()
        for trigger in sorted(triggers, key=len, reverse=True):
            if folded.startswith(trigger.casefold()):
                tail = text[len(trigger):]
                if not tail or not tail[0].isalnum():
                    return tail.lstrip(" \t\r\n,:;—-")
        return None

    async def answer(self, text: str, lang: str, big: bool = False, speaker: str | None = None):
        """Yield the spoken answer to one utterance, sentence by sentence."""
        turn = Turn(text, lang, speaker)
        complete = False
        try:
            async for sentence in self._answer(turn, big):
                turn.said.append(sentence)
                yield sentence
            complete = True
        finally:
            if complete:  # a cancelled turn is not part of the conversation
                self._remember(turn)

    async def _answer(self, turn: Turn, big: bool):
        text, lang = turn.text, turn.lang
        pending, self.pending = self.pending, None
        if pending:
            live = not big and time.monotonic() < pending.expires
            if pending.reply and live:
                log.info("reply to the speaker's question: %r", text)
                said = await pending.reply(text)
                if said:
                    yield said
                return
            choice = yes_no(text) if live and not pending.reply else None
            if choice is not None:
                log.info("confirmation %s: %s", pending.kind, "yes" if choice else "no")
                async for sentence in self._resolve(pending, choice, turn.facts, turn.speaker):
                    yield sentence
                return
            pending.cancel()  # the user moved on to something else
        prompt = self._trigger(text, self.big_triggers)
        if big or prompt is not None:
            prompt = text if prompt is None else prompt
            if not prompt:
                yield "Что передать?" if lang == "ru" else "What should I pass along?"
            elif not self.remote or not self.remote.configured:
                yield "Большая модель не настроена." if lang == "ru" else "The big model is not configured."
            else:
                messages = [{"role": "system", "content": self.remote_system}]
                messages += self._history_messages(turn.speaker)
                messages.append({"role": "user", "content": f"{self._stamp(lang, turn.speaker)} {prompt}"})
                async for sentence in self.remote.respond(messages):
                    yield sentence
            return
        if _role_var.get() == "admin":
            reply = await self.agents.handle(text, lang)
        else:  # the agents have shell access: only an admin reaches them, never a plain chat user
            reply = self.agents.refuse(text, lang)
        if reply is not None:
            log.info("agent command: %r -> %r", text, reply)
            yield reply
            return
        if self.fast_commands:
            reply = await intents.handle(self, text, lang)
            if reply is not None:
                log.info("fast command: %r -> %r", text, reply)
                if reply:  # "" = done silently (pausing music)
                    yield reply
                return
        # The last few exchanges ride along for follow-ups; the history is short
        # and per speaker, so background speech and old topics don't colour a new
        # request. Timers, alarms, volume and saved memories persist regardless.
        recalled = await self.memory.recall(text, turn.speaker)
        if recalled:
            log.info("recalled memories: %s", recalled)
        context = f" [Saved facts about the user: {' '.join(recalled)}]" if recalled else ""
        messages = [{"role": "system", "content": self.system}]
        messages += self._history_messages(turn.speaker)
        messages.append({"role": "user", "content": f"{self._stamp(lang, turn.speaker)}{context} {text}"})
        async for sentence in self._llm(messages, text, lang, self.schemas, confirm=True, facts=turn.facts):
            yield sentence

    async def _llm(self, messages: list[dict], text: str, lang: str, schemas: list[dict], confirm: bool,
                   facts: list[str] | None = None):
        """Run LFM3 with tools. With `confirm`, the big model (and web search when the big
        model could answer it) first asks the user; the turn ends on that question."""
        def needs_confirm(name: str) -> bool:
            return confirm and (name == "ask_big_model" and self.confirm_big
                                or name == "web_search" and self.offer_big_for_search and self._big_ready)

        async def run_tool(name: str, args: dict):
            if needs_confirm(name):
                raise NeedsConfirmation("big" if name == "ask_big_model" else "search", args)
            if name == "play_music" and not asks_for_music(text):
                return {"error": "the user didn't ask for music; answer in words"}
            result = await self.tools.call(name, args, lang)
            if facts is not None:
                facts.append(format_fact(name, args, result))  # enough context for a follow-up
            return result

        def preface(name: str) -> str | None:
            # A tool that first asks the user says its own question instead.
            return None if needs_confirm(name) else PREFACES.get(name, {}).get(lang)

        try:
            async for sentence in self.llm.respond(messages, schemas, run_tool, lang, preface=preface):
                yield sentence
        except NeedsConfirmation as c:
            log.info("paused %s(%s) for the user's confirmation", c.kind_tool, c.tool_args)
            assistant = next(m for m in reversed(messages) if m.get("tool_calls"))
            call = next(tc for tc in assistant["tool_calls"] if tc["function"]["name"] == c.kind_tool)
            search = None
            if c.kind == "search":  # search while the user answers: both answers need the results
                search = asyncio.create_task(self.tools.call("web_search", c.tool_args, lang))
            self.pending = Pending(c.kind, messages, call["id"], c.tool_args, text, lang, search)
            ru = lang == "ru"
            if c.kind == "big":
                yield "Это сложновато. Спросить большую модель?" if ru else "That's a tough one. Should I ask the big model?"
            else:
                yield "Ищу в интернете. Пусть ответит большая модель?" if ru else \
                    "I'm searching the web. Should the big model answer?"

    async def _resolve(self, p: "Pending", yes: bool, facts: list[str] | None = None,
                       speaker: str | None = None):
        ru = p.lang == "ru"
        if p.kind == "music":
            track = await self.music.keep_playing() if yes else None
            if not yes:
                yield "Ладно." if ru else "Okay."
            elif track is None:
                yield "Не получилось включить." if ru else "I couldn't play anything."
            elif track.station == MY_WAVE:
                yield "Включаю мою волну." if ru else "Playing My Wave."
            else:
                yield "Включаю музыку." if ru else "Playing some music."
            return
        if p.kind == "big" and yes:
            messages = [{"role": "system", "content": self.remote_system},
                        {"role": "user", "content": f"{self._stamp(p.lang, speaker)} "
                                                    f"{p.args.get('question') or p.text}"}]
            async for sentence in self.remote.respond(messages):
                yield sentence
            return
        if p.kind == "search":
            if not p.search.done():
                yield "Ищу." if ru else "Searching."
            result = await p.search
            if yes:
                found = json.dumps(result, ensure_ascii=False)
                messages = [{"role": "system", "content": self.remote_system},
                            {"role": "user", "content": f"{self._stamp(p.lang, speaker)} {p.text}\n\nWeb search "
                                                        f"results (English; answer in the reply language):\n{found}"}]
                async for sentence in self.remote.respond(messages):
                    yield sentence
                return
        else:  # big model declined: LFM3 answers on its own
            result = {"declined": "Answer this yourself as well as you can; don't mention the big model."}
        p.messages.append({"role": "tool", "tool_call_id": p.call_id, "content": json.dumps(result, ensure_ascii=False)})
        local = [s for s in self.schemas if s is not BIG_MODEL_SCHEMA]
        async for sentence in self._llm(p.messages, p.text, p.lang, local, confirm=False, facts=facts):
            yield sentence

    async def _run_turn(self, turn_id: str, text: str, lang: str, big: bool = False,
                        speaker: str | None = None) -> None:
        self.speaker = speaker
        said = []
        t0 = time.monotonic()
        try:
            try:
                async for sentence in self.answer(text, lang, big=big, speaker=speaker):
                    if not said:
                        log.info("first sentence after %.2fs", time.monotonic() - t0)
                    said.append(sentence)
                    await self.broadcast({"type": "say", "id": turn_id, "text": sentence, "lang": lang})
            except asyncio.CancelledError:
                log.info("turn %s cancelled", turn_id)
                raise
            except Exception:
                log.exception("turn failed")
                said.append("Что-то я сломался." if lang == "ru" else "Something broke.")
                await self.broadcast({"type": "say", "id": turn_id, "text": said[-1], "lang": lang})
            await self.broadcast({"type": "turn_end", "id": turn_id,
                                  "expect_reply": bool(said) and said[-1].rstrip().endswith("?")})
        finally:
            if self.turn is asyncio.current_task():
                self.speaker = None

    def start_turn(self, turn_id: str, text: str, lang: str, big: bool = False,
                   speaker: str | None = None) -> None:
        self.cancel_turn()
        log.info("utterance %s [%s]%s: %s", turn_id, lang, f" {speaker}" if speaker else "", text)
        self.turn = asyncio.create_task(self._run_turn(turn_id, text, lang, big, speaker))

    def cancel_turn(self) -> None:
        if self.turn and not self.turn.done():
            self.turn.cancel()

    # HTTP --------------------------------------------------------------
    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        log.info("voice connected from %s", request.remote)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                d = json.loads(msg.data)
                kind = d.get("type")
                if kind == "hello":
                    await ws.send_str(json.dumps({"type": "volume", "level": self.volume}))
                    await ws.send_str(json.dumps({"type": "night", "on": self.night_override}))
                    await ws.send_str(json.dumps({"type": "voices"}))  # ask which profiles exist
                    await self._deliver_missed()
                elif kind == "utterance":
                    self.start_turn(d.get("id") or uuid.uuid4().hex[:8], d["text"], d.get("lang", "ru"),
                                    speaker=d.get("speaker") or None)
                elif kind == "cancel":
                    self.cancel_turn()
                elif kind == "music_event":
                    await self.music.on_event(d["event"], d["id"], d.get("detail", ""))
                elif kind == "music_position":
                    await self.music.on_position(d["id"], d.get("provider", ""), float(d.get("position") or 0.0))
                elif kind == "listening":
                    self._listening(d.get("id", ""))
                elif kind == "voices":
                    self.voice_names = [str(n) for n in d.get("names") or []]
                    log.info("enrolled voices: %s", ", ".join(self.voice_names) or "none")
                elif kind == "enroll_result":
                    log.info("voice enrollment %r: %s", d.get("name"),
                             "done" if d.get("ok") else d.get("error") or "failed")
        finally:
            self.clients.discard(ws)
            log.info("voice disconnected")
        return ws

    async def ask_handler(self, request: web.Request) -> web.Response:
        d = await request.json()
        text = d["text"]
        lang = d.get("lang") or detect_lang(text)
        big = bool(d.get("big"))
        speaker = d.get("speaker") or None
        if d.get("speak"):
            turn_id = uuid.uuid4().hex[:8]
            self.start_turn(turn_id, text, lang, big, speaker)
            await self.turn
            return web.json_response({"id": turn_id})
        t0 = time.monotonic()
        out, first = [], None
        previous, self.speaker = self.speaker, speaker
        try:
            async for s in self.answer(text, lang, big=big, speaker=speaker):
                first = first or time.monotonic() - t0
                out.append(s)
        finally:
            self.speaker = previous
        return web.json_response({"text": " ".join(out), "sentences": out,
                                  "first_sentence_s": round(first or 0, 3),
                                  "total_s": round(time.monotonic() - t0, 3)})



async def serve(cfg: Config) -> None:
    brain = Brain(cfg)
    app = web.Application()
    app.add_routes([web.get("/ws", brain.ws_handler), web.post("/ask", brain.ask_handler),
                    web.get("/omp", brain.agents.ws_handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    b = cfg["brain"]
    await web.TCPSite(runner, b["host"], int(b["port"])).start()
    log.info("brain listening on %s:%s", b["host"], b["port"])
    brain.chat.start()
    try:
        await brain.scheduler.run()
    finally:
        await brain.chat.close()
        await brain.llm.close()
        await brain.memory.close()
        await runner.cleanup()
