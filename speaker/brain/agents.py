"""Coding-agent threads: omp sessions you talk to through the speaker.

Every omp session that loads omp/speaker.ts connects to the brain's /omp WebSocket
(see that file for the protocol). Sessions you opened yourself are found that way;
«запусти агента …» opens a new one as a tab of a Tern session (the terminal's daemon keeps
it running; it shows in your Tern window) and gives it the task. Only main sessions connect:
subagents, advisors and print-mode runs never become threads.
Threads are numbered from one, in the order they connect, so they can be named by voice:
«скажи второму агенту …», «спроси агента три …»; or by name: the tag of the session's title
(«агенту алекса» for "ALEXA: …") or its folder. Without either, the request goes to the thread
you last talked with, or to the only one. A thread that has closed is reopened (omp --resume in
a new Tern tab) and gets the request once it connects; named ones are found among omp's
session files, so any past session can be picked up again.

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
from .music import score
from .text import clean

if TYPE_CHECKING:
    from .server import Brain

log = logging.getLogger("brain.agents")

START_TIMEOUT = 60  # seconds for a new thread to start and connect
DONE_CHARS = 300  # how much of a final answer is read out when the agent didn't speak itself
NAME_MATCH = 0.85  # how closely spoken words must match a session's name (music's fuzzy score)
RECENT_SESSIONS = 15  # closed sessions a name is looked for among, most recently used first

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


_ICONS = re.compile(r"[\ue000-\uf8ff\U000f0000-\U0010ffff]")  # omp puts a nerd-font icon before titles


def _names(title: str, cwd: str) -> list[str]:
    """What a session can be called by voice: its title's tag («ALEXA» in "ALEXA: Connect threads…") and its folder."""
    names = []
    tag = re.match(r"\s*([^\W_]+)\s*:", _ICONS.sub("", title))
    if tag:
        names.append(tag[1])
    folder = Path(cwd).name.replace("-", " ").replace("_", " ")
    if folder:
        names.append(folder)
    return names


@dataclass
class Saved:
    """A session that isn't connected (closed, or open in an omp without this extension): enough to reopen it."""
    session: str
    file: str
    cwd: str
    name: str
    mtime: float = 0.0


@dataclass
class Thread:
    ws: web.WebSocketResponse
    number: int
    session: str = ""
    file: str = ""
    name: str = ""
    cwd: str = ""
    busy: bool = False
    spawned: bool = False  # started by the speaker
    lang: str = "ru"  # the language the user last spoke to it in

    @property
    def title(self) -> str:
        """The session's title, else its folder."""
        return (_ICONS.sub("", self.name).strip() or Path(self.cwd).name or "omp")[:60]


def saved_sessions(root: Path, limit: int) -> list[Saved]:
    """The `limit` most recently used omp sessions on disk, newest first. A session file starts with
    its title record and header."""
    files = []
    for f in root.glob("*/*.jsonl"):
        try:
            files.append((f.stat().st_mtime, f))
        except OSError:
            continue
    out = []
    for mtime, f in sorted(files, reverse=True):
        if len(out) == limit:
            break
        try:
            with f.open(encoding="utf-8") as fh:
                head = [json.loads(line) for line, _ in zip(fh, range(3)) if line.strip()]
        except (OSError, ValueError):
            continue
        header = next((r for r in head if r.get("type") == "session"), None)
        if header is None:
            continue
        title = next((r.get("title", "") for r in head if r.get("type") == "title"), header.get("title", ""))
        out.append(Saved(header.get("id", ""), str(f), header.get("cwd", ""), title or "", mtime))
    return out


class Agents:
    def __init__(self, brain: "Brain", cfg: dict, url: str):
        self.brain = brain
        self.enabled = bool(cfg["enabled"])
        self.cwd = brain.cfg.file(cfg["cwd"])
        self.command = list(cfg["command"])
        self.session = cfg["tern_session"]  # the Tern session new threads open in, as tabs
        self.sessions = brain.cfg.file(cfg["sessions"])  # omp's session files, to reopen closed threads
        self.url = url  # where the threads we start connect back to
        self.threads: list[Thread] = []
        self.focus: Thread | Saved | None = None  # the thread last talked with; Saved once it closed
        self.starting: dict[str, tuple[int, dict, str]] = {}  # token -> (number, first message, lang)
        self._tasks: set[asyncio.Task] = set()

    # Voice commands ------------------------------------------------------
    def refuse(self, text: str, lang: str) -> str | None:
        """For a chat user without the admin role: what to say to an agent command instead of running it
        (None when `text` isn't one). Nothing here touches a thread."""
        if not self.enabled or parse(text) is None:
            return None
        log.warning("agent command refused for a chat user without the admin role: %r", text)
        return "С агентами может говорить только админ." if lang == "ru" else "Only an admin can talk to the agents."

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
        target, rest = self._target(n, rest)
        if target is None:
            if n is not None:
                return f"Агента номер {speech.ru_number(n)} нет." if ru else f"There is no agent {n}."
            if not self.threads:
                return "Открытых агентов нет: назовите агента по имени." if ru else "No agents are open: name one."
            return "Агентов несколько: назовите номер или имя." if ru else \
                "There are several agents: say a number or a name."
        if isinstance(target, Thread):
            target.lang = lang
        self.focus = target
        if rest:
            return await self._deliver(target, kind, rest, lang)

        async def deliver(answer: str | None) -> str | None:
            return await self._deliver(target, kind, answer, lang) if answer else None
        self.brain.pending = self.brain.expect_reply(lang, deliver)
        if kind == "tell":
            return "Что передать?" if ru else "What should I pass along?"
        return "Что спросить?" if ru else "What should I ask?"

    async def _deliver(self, t: Thread | Saved, kind: str, text: str, lang: str) -> str:
        """Pass what the user said to a thread: as a message ("tell") or a side question ("ask").
        A closed thread is reopened first and gets it once it connects."""
        ru = lang == "ru"
        msg = {"type": "prompt", "text": text} if kind == "tell" else \
            {"type": "btw", "id": uuid.uuid4().hex[:8], "text": text}
        if isinstance(t, Thread):
            await self._send(t, msg)
            if kind == "tell":
                return "Передаю." if ru else "Passing it on."
            return "Спрашиваю." if ru else "Asking."
        cwd = t.cwd if Path(t.cwd).is_dir() else str(self.cwd)
        if await self._launch(cwd, [*self.command, "--resume", t.file], msg, lang) is None:
            return "Не получилось открыть агента." if ru else "I couldn't reopen the agent."
        log.info("reopening %s (%s)", t.session, t.name)
        if kind == "tell":
            return "Открываю агента и передаю." if ru else "Reopening the agent and passing it on."
        return "Открываю агента и спрашиваю." if ru else "Reopening the agent and asking."

    def _target(self, n: int | None, rest: str) -> tuple[Thread | Saved | None, str]:
        """Who a request is for, and the request without the name that picked it: thread number `n`;
        a thread named at the start of `rest` («агенту алекса, …»), open or closed; the thread last
        talked with, even if it was closed since; or the only open one."""
        if n is not None:
            return next((t for t in self.threads if t.number == n), None), rest
        named = self._by_name(rest)
        if named:
            return named
        if isinstance(self.focus, Saved) or self.focus in self.threads:
            return self.focus, rest
        return (self.threads[0] if len(self.threads) == 1 else None), rest

    def _by_name(self, rest: str) -> tuple[Thread | Saved, str] | None:
        """The thread whose name (title tag or folder) the first one or two words of `rest` are.
        Open threads win ties, then the most recently used session."""
        words = rest.split()
        live = {t.session for t in self.threads}
        candidates: list[tuple[Thread | Saved, list[str]]] = [(t, _names(t.name, t.cwd)) for t in self.threads]
        candidates += [(s, _names(s.name, s.cwd)) for s in saved_sessions(self.sessions, RECENT_SESSIONS)
                       if s.session not in live]
        best, best_score = None, NAME_MATCH
        for k in (2, 1):
            spoken = " ".join(words[:k]).strip(",.:;!?—- ")
            if len(words) < k or len(re.sub(r"\W", "", spoken)) < 4:
                continue
            for target, names in candidates:
                s = score(spoken, *names) if names else 0.0
                if s > best_score:
                    best, best_score = (target, " ".join(words[k:]).lstrip(",.:;!?—- ")), s
        return best

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
        n = await self._launch(str(self.cwd), self.command, {"type": "prompt", "text": task}, lang)
        if n is None:
            return "Не получилось запустить агента." if ru else "I couldn't start an agent."
        log.info("started thread %d: %s", n, task)
        if not self.threads:
            return "Запускаю агента." if ru else "Starting an agent."
        return f"Запускаю агента номер {speech.ru_number(n)}." if ru else f"Starting agent {n}."

    async def _launch(self, cwd: str, command: list[str], first: dict, lang: str) -> int | None:
        """Run omp `command` as a new tab of the Tern session; once it connects it gets message `first`.
        Returns the number it will have, or None when Tern failed."""
        n = self._free_number()
        token = uuid.uuid4().hex[:8]
        launch = ["--cwd", cwd, "--", "env", f"SPEAKER_THREAD={token}", f"SPEAKER_BRAIN_URL={self.url}", *command]
        ok, err = await _tern("new", "tab", self.session, *launch)
        if not ok and "no session is called" in err:  # the first thread creates the session
            ok, err = await _tern("new", "session", self.session, *launch)
        if not ok:
            log.error("tern failed: %s", err)
            return None
        self.starting[token] = (n, first, lang)
        watch = asyncio.create_task(self._watch_start(token))
        self._tasks.add(watch)
        watch.add_done_callback(self._tasks.discard)
        return n

    async def _watch_start(self, token: str) -> None:
        await asyncio.sleep(START_TIMEOUT)
        started = self.starting.pop(token, None)
        if started:
            n, _, lang = started
            log.error("thread %d never connected", n)
            await self._speak("Агент не запустился." if lang == "ru" else "The agent didn't start.", lang)

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

    async def _speak(self, text: str, lang: str, pending=None) -> None:
        """Say an agent's message on the speaker; with no voice service connected it goes to the notify chat."""
        await self.brain.speak(text, lang, queue=True, pending=pending)
        self.brain.chat.notify_soon("agents", text)

    async def _on_message(self, t: Thread, d: dict) -> None:
        kind = d.get("type")
        ru = t.lang == "ru"
        if kind == "hello":
            t.session, t.file, t.name, t.cwd, t.busy = d.get("session", ""), d.get("file", ""), d.get("name", ""), \
                d.get("cwd", ""), bool(d.get("busy"))
            if isinstance(self.focus, Saved) and self.focus.session == t.session:
                self.focus = t  # the closed thread you last talked with is back
            started = self.starting.pop(d.get("thread") or "", None)
            if started:
                t.number, first, t.lang = started
                t.spawned, self.focus = True, t
                await self._send(t, first)
            log.info("thread %d: %s in %s%s", t.number, t.session, t.cwd, " (started here)" if t.spawned else "")
        elif kind == "state":
            t.busy, t.name = bool(d.get("busy")), d.get("name") or t.name
        elif kind == "say":
            self.focus = t
            await self._speak(f"{self._label(t, t.lang)}: {d.get('text', '')}", t.lang)
        elif kind == "ask":
            self.focus = t
            qid = d.get("id", "")

            async def reply(text: str | None) -> str | None:
                return await self.answer_question(t, qid, text)
            await self._speak(f"{self._label(t, t.lang)} {'спрашивает' if ru else 'asks'}: {d.get('text', '')}",
                              t.lang, pending=self.brain.expect_reply(t.lang, reply))
        elif kind == "done":
            said = _short(d.get("text", ""))
            head = f"{self._label(t, t.lang)} {'закончил' if ru else 'is done'}."
            await self._speak(f"{head} {said}" if said else head, t.lang)
        elif kind == "btw_reply":
            if d.get("error") or not d.get("text"):
                log.warning("side question failed: %s", d.get("error"))
                text = f"{self._label(t, t.lang)} {'не ответил' if ru else 'did not answer'}."
            else:
                text = f"{self._label(t, t.lang)} {'отвечает' if ru else 'says'}: {d['text']}"
            await self._speak(text, t.lang)

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
            if self.focus is t and t.file:  # closed: «скажи агенту …» reopens it
                self.focus = Saved(t.session, t.file, t.cwd, t.name)
        return ws
