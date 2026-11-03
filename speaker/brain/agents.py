"""Coding-agent threads: omp sessions you talk to through the speaker.

Every omp session that loads omp/speaker.ts connects to the brain's /omp WebSocket
(see that file for the protocol). Sessions you opened yourself are found that way;
«запусти агента …» opens a new one as a tab of a Tern session (the terminal's daemon keeps
it running; it shows in your Tern window) and gives it the task. Only main sessions connect:
subagents, advisors and print-mode runs never become threads.
Threads are numbered from one, in the order they connect, so they can be named by voice:
«скажи второму агенту …», «спроси агента три …». Without a number the request goes to
the thread you last talked with, or to the only one.

Agents talk back with their speaker_say / speaker_ask tools; those, side-question
answers and "finished" notices are said as soon as the speaker is free.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import WSMsgType, web

from ..lang import ru_plural
from . import speech
from .text import clean

if TYPE_CHECKING:
    from .server import Brain

log = logging.getLogger("brain.agents")

START_TIMEOUT = 60  # seconds for a new thread to start and connect
DONE_CHARS = 300  # how much of a final answer is read out when the agent didn't speak itself

_ORDINAL_STEMS = {"перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9}
_NUMBERS = {"один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}

_TAIL = r"\b[\s,.:;!?—-]*(?P<rest>.*)$"
_COMMANDS = [(kind, re.compile(rf"^(?:{head}){_TAIL}", re.I | re.S)) for kind, head in [
    ("start", r"(?:запусти|создай|заведи|начни)\s+(?:ещ[её]\s+)?(?:одного\s+)?(?:нового\s+)?агента"),
    ("start", r"нов(?:ый|ого)\s+агент(?:а)?"),
    ("tell", r"(?:скажи|передай|напиши|ответь)\s+(?:(?P<ord>\w+)\s+)?агенту"),
    ("ask", r"спроси\s+(?:(?P<ord>\w+)\s+)?агента"),
    ("list", r"(?:какие|сколько)\s+(?:есть\s+|сейчас\s+)?агент\w*|что\s+(?:с\s+агентами|делают\s+агенты)"
             r"|список\s+агентов"),
    ("start", r"(?:start|launch|spin\s+up)\s+(?:a\s+|an\s+|another\s+)?(?:new\s+)?agent"),
    ("tell", r"tell\s+(?:the\s+)?(?:(?P<ord>\w+)\s+)?agent"),
    ("ask", r"ask\s+(?:the\s+)?(?:(?P<ord>\w+)\s+)?agent"),
    ("list", r"list\s+(?:the\s+)?agents|what\s+are\s+the\s+agents\s+doing|which\s+agents"),
]]


def number(word: str | None) -> int | None:
    """1 for «первому», «один», "1" or "first"; None for anything else."""
    if not word:
        return None
    w = word.casefold().replace("ё", "е")
    if w.isdigit():
        return int(w)
    if w in _NUMBERS:
        return _NUMBERS[w]
    return next((n for stem, n in _ORDINAL_STEMS.items() if w.startswith(stem)), None)


def parse(text: str) -> tuple[str, int | None, str] | None:
    """(kind, thread number or None, the rest) for an agent command; None for anything else."""
    t = text.strip().lstrip(",.!?:;—- ")
    for kind, rx in _COMMANDS:
        m = rx.match(t)
        if not m:
            continue
        rest = m["rest"].strip()
        n = number(m.groupdict().get("ord"))
        if n is None and kind in ("tell", "ask"):  # «агенту два, …», «агента номер три …»
            words = rest.split(maxsplit=2)
            skip = 1 if words and words[0].casefold() in ("номер", "number") else 0
            if len(words) > skip and (n := number(words[skip].strip(",.:;!?"))) is not None:
                rest = " ".join(words[skip + 1:]).lstrip(",.:;!?—- ")
        return kind, n, rest
    return None


async def _tern(*args: str) -> tuple[bool, str]:
    """Run a `tern` command against the Tern daemon: (succeeded, its error output)."""
    proc = await asyncio.create_subprocess_exec("tern", *args, stdout=asyncio.subprocess.DEVNULL,
                                                stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    return proc.returncode == 0, err.decode(errors="replace").strip()


def _short(text: str, limit: int = DONE_CHARS) -> str:
    """The first sentences of `text` that fit in `limit` characters."""
    text = clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > 0 else cut.rsplit(" ", 1)[0] + "…"


@dataclass
class Thread:
    ws: web.WebSocketResponse
    number: int
    session: str = ""
    name: str = ""
    cwd: str = ""
    busy: bool = False
    spawned: bool = False  # started by the speaker
    lang: str = "ru"  # the language the user last spoke to it in

    @property
    def title(self) -> str:
        """The session's title (omp prefixes it with an icon from a private-use font), else its folder."""
        name = re.sub(r"[\ue000-\uf8ff\U000f0000-\U0010ffff]", "", self.name).strip()
        return (name or Path(self.cwd).name or "omp")[:60]


class Agents:
    def __init__(self, brain: "Brain", cfg: dict, url: str):
        self.brain = brain
        self.enabled = bool(cfg["enabled"])
        self.cwd = brain.cfg.file(cfg["cwd"])
        self.command = list(cfg["command"])
        self.session = cfg["tern_session"]  # the Tern session new threads open in, as tabs
        self.url = url  # where the threads we start connect back to
        self.threads: list[Thread] = []
        self.focus: Thread | None = None
        self.starting: dict[str, tuple[int, str, str]] = {}  # token -> (number, task, lang)
        self._tasks: set[asyncio.Task] = set()

    # Voice commands ------------------------------------------------------
    async def handle(self, text: str, lang: str) -> str | None:
        """The spoken reply to an agent command, or None when `text` isn't one."""
        if not self.enabled:
            return None
        cmd = parse(text)
        if cmd is None:
            return None
        kind, n, rest = cmd
        ru = lang == "ru"
        log.info("agent command %s #%s: %r", kind, n, rest)
        if kind == "list":
            return self._describe(lang)
        if kind == "start":
            if not rest:
                async def task(answer: str | None) -> str | None:
                    return await self.start(answer, lang) if answer else None
                self.brain.pending = self.brain.expect_reply(lang, task)
                return "Что сделать агенту?" if ru else "What should the agent do?"
            return await self.start(rest, lang)
        if not self.threads:
            return "Агентов нет." if ru else "There are no agents."
        thread = self._pick(n)
        if thread is None:
            if n is None:
                return "Агентов несколько: назовите номер." if ru else "There are several agents: say a number."
            return f"Агента номер {speech.ru_number(n)} нет." if ru else f"There is no agent {n}."
        thread.lang, self.focus = lang, thread
        if rest:
            return await self._deliver(thread, kind, rest)

        async def deliver(answer: str | None) -> str | None:
            return await self._deliver(thread, kind, answer) if answer else None
        self.brain.pending = self.brain.expect_reply(lang, deliver)
        if kind == "tell":
            return "Что передать?" if ru else "What should I pass along?"
        return "Что спросить?" if ru else "What should I ask?"

    async def _deliver(self, t: Thread, kind: str, text: str) -> str:
        """Pass what the user said to a thread: as a message ("tell") or a side question ("ask")."""
        ru = t.lang == "ru"
        if kind == "tell":
            await self._send(t, {"type": "prompt", "text": text})
            return "Передаю." if ru else "Passing it on."
        await self._send(t, {"type": "btw", "id": uuid.uuid4().hex[:8], "text": text})
        return "Спрашиваю." if ru else "Asking."

    def _pick(self, n: int | None) -> Thread | None:
        """Thread number `n`; without a number, the one last talked with or the only one."""
        if n is not None:
            return next((t for t in self.threads if t.number == n), None)
        if self.focus in self.threads:
            return self.focus
        return self.threads[0] if len(self.threads) == 1 else None

    def _describe(self, lang: str) -> str:
        ru = lang == "ru"
        if not self.threads:
            return "Агентов нет." if ru else "There are no agents."
        n = len(self.threads)
        parts = [f"Сейчас {speech.ru_number(n)} {ru_plural(n, 'агент', 'агента', 'агентов')}." if ru else
                 f"{n} agent{'s' if n > 1 else ''}."]
        for t in sorted(self.threads, key=lambda t: t.number):
            state = ("работает" if t.busy else "свободен") if ru else ("working" if t.busy else "idle")
            parts.append(f"{self._label(t, lang, always=True)}: {t.title}, {state}.")
        return " ".join(parts)

    def _label(self, t: Thread, lang: str, always: bool = False) -> str:
        if len(self.threads) > 1 or always:
            return f"Агент {speech.ru_number(t.number)}" if lang == "ru" else f"Agent {t.number}"
        return "Агент" if lang == "ru" else "The agent"

    def _free_number(self) -> int:
        used = {t.number for t in self.threads} | {n for n, _, _ in self.starting.values()}
        return next(n for n in range(1, len(used) + 2) if n not in used)

    # Starting threads ----------------------------------------------------
    async def start(self, task: str, lang: str) -> str:
        ru = lang == "ru"
        n = self._free_number()
        token = uuid.uuid4().hex[:8]
        launch = ["--cwd", str(self.cwd), "--", "env", f"SPEAKER_THREAD={token}", f"SPEAKER_BRAIN_URL={self.url}",
                  *self.command]
        # A tab in the session; the first thread creates the session.
        ok, err = await _tern("new", "tab", self.session, *launch)
        if not ok and "no session is called" in err:
            ok, err = await _tern("new", "session", self.session, *launch)
        if not ok:
            log.error("tern failed: %s", err)
            return "Не получилось запустить агента." if ru else "I couldn't start an agent."
        log.info("started thread %d in Tern session %s: %s", n, self.session, task)
        self.starting[token] = (n, task, lang)
        watch = asyncio.create_task(self._watch_start(token))
        self._tasks.add(watch)
        watch.add_done_callback(self._tasks.discard)
        if not self.threads:
            return "Запускаю агента." if ru else "Starting an agent."
        return f"Запускаю агента номер {speech.ru_number(n)}." if ru else f"Starting agent {n}."

    async def _watch_start(self, token: str) -> None:
        await asyncio.sleep(START_TIMEOUT)
        started = self.starting.pop(token, None)
        if started:
            n, _, lang = started
            log.error("thread %d never connected", n)
            await self.brain.speak("Агент не запустился." if lang == "ru" else "The agent didn't start.", lang,
                                   queue=True)

    # Messages from threads ----------------------------------------------
    async def _send(self, t: Thread, msg: dict) -> None:
        try:
            await t.ws.send_str(json.dumps(msg, ensure_ascii=False))
        except ConnectionError:
            log.warning("thread %d is gone", t.number)

    async def answer_question(self, t: Thread, qid: str, text: str | None) -> str | None:
        """The user's spoken reply to an agent's speaker_ask (None: nobody answered)."""
        await self._send(t, {"type": "answer", "id": qid, "text": text})
        if text is None:
            return None
        self.focus = t
        return "Передаю." if t.lang == "ru" else "Passing it on."

    async def _on_message(self, t: Thread, d: dict) -> None:
        kind = d.get("type")
        ru = t.lang == "ru"
        if kind == "hello":
            t.session, t.name, t.cwd, t.busy = d.get("session", ""), d.get("name", ""), d.get("cwd", ""), \
                bool(d.get("busy"))
            started = self.starting.pop(d.get("thread") or "", None)
            if started:
                t.number, task, t.lang = started
                t.spawned, self.focus = True, t
                await self._send(t, {"type": "prompt", "text": task})
            log.info("thread %d: %s in %s%s", t.number, t.session, t.cwd, " (started here)" if t.spawned else "")
        elif kind == "state":
            t.busy, t.name = bool(d.get("busy")), d.get("name") or t.name
        elif kind == "say":
            self.focus = t
            await self.brain.speak(f"{self._label(t, t.lang)}: {d.get('text', '')}", t.lang, queue=True)
        elif kind == "ask":
            self.focus = t
            qid = d.get("id", "")

            async def reply(text: str | None) -> str | None:
                return await self.answer_question(t, qid, text)
            await self.brain.speak(f"{self._label(t, t.lang)} {'спрашивает' if ru else 'asks'}: {d.get('text', '')}",
                                   t.lang, queue=True, pending=self.brain.expect_reply(t.lang, reply))
        elif kind == "done":
            said = _short(d.get("text", ""))
            head = f"{self._label(t, t.lang)} {'закончил' if ru else 'is done'}."
            await self.brain.speak(f"{head} {said}" if said else head, t.lang, queue=True)
        elif kind == "btw_reply":
            if d.get("error") or not d.get("text"):
                log.warning("side question failed: %s", d.get("error"))
                text = f"{self._label(t, t.lang)} {'не ответил' if ru else 'did not answer'}."
            else:
                text = f"{self._label(t, t.lang)} {'отвечает' if ru else 'says'}: {d['text']}"
            await self.brain.speak(text, t.lang, queue=True)

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        t = Thread(ws, self._free_number())
        self.threads.append(t)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._on_message(t, json.loads(msg.data))
        finally:
            self.threads.remove(t)
            log.info("thread %d disconnected", t.number)
        return ws
