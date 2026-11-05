"""Text chat with the speaker: the shared core of the Telegram and Discord bots.

A message from an allowed user is answered exactly like a spoken request (`Brain.answer`, so music,
timers, weather, memory and the coding agents all work) but as text, without going through the voice
turn: it never cancels or talks over what the speaker is saying. Each user maps to a speaker name
(`users` in the config: the same names as the voice profiles, so history and memories are shared with
that person's voice), otherwise their chat name.

Also here: «/say» and «скажи на колонке …» (the speaker says the message aloud), the voice commands
«напиши Мише в телеграм …» (speaker -> chat), and notifications (missed rings, agents' messages) to a
configured chat. Security: only ids in a service's allowlist get anything; strangers get one refusal
per hour with their id (so the owner can add it) and the bot never listens on a port: Telegram long
polling and the Discord gateway only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ..config import Config
from ..lang import detect as detect_lang
from .music import score

if TYPE_CHECKING:
    from .server import Brain

log = logging.getLogger("brain.chat")

ANSWER_TIMEOUT = 300  # seconds a chat message may take (a coding agent, the big model, a web search)
SEND_TIMEOUT = 30  # seconds for a message out of a voice command
CHAT_CONFIRM_SECONDS = 600  # how long a "should I ...?" question to a chat user waits for their yes/no
REFUSE_EVERY = 3600  # a stranger gets one refusal (and one warning in the log) per this many seconds
NAME_MATCH = 0.8  # how closely a spoken name must match a contact (music's fuzzy score)
MAX_STRANGERS = 500  # strangers remembered for the refusal rate limit

Reply = Callable[[str], Awaitable[None]]

_COMMAND = re.compile(r"^[/!]([a-zA-Zа-яА-Я]+)(?:@\w+)?(?:\s+(.*))?$", re.S)
_SAY_NATURAL = re.compile(
    r"^(?:пожалуйста[, ]+)?(?:скажи(?:те)?|передай(?:те)?|произнеси|озвучь|say|tell)\s+(?:на|в|по|on|to|over)\s+"
    r"(?:колонк\w*|динамик\w*|(?:the\s+)?speaker)[\s,:—–-]*(?:что\b|that\b)?[\s,:—–-]*(?P<msg>.*)$", re.I | re.S)
_SERVICE = r"(?P<svc>телеграм\w*|телег\w*|тг|telegram|tg|дискорд\w*|дискорт\w*|discord)"
_WRITE = re.compile(
    r"^(?:пожалуйста[, ]+)?(?:напиши(?:те)?|отправь(?:те)?|пошли(?:те)?|скинь(?:те)?|кинь(?:те)?|черкани|сообщи|"
    r"передай(?:те)?|send|write|text|message|tell)\s+"
    r"(?:(?:сообщение|смс|a message|message)\s+)?(?:to\s+)?"
    r"(?:(?P<who>\S+(?:\s+\S+){0,2}?)\s+)?"
    r"(?:в|во|через|по|на|in|on|via|over)\s+" + _SERVICE + r"\b[\s,:—–-]*(?:что\b|that\b)?[\s,:—–-]*(?P<msg>.*)$",
    re.I | re.S)
_ME = {"мне", "себе", "me", "myself"}

START = {
    "ru": "Привет! Напиши мне то же, что сказал бы колонке: «включи радио», «таймер на пять минут», «какая погода». "
          "Чтобы колонка сказала что-то вслух: /say текст или «скажи на колонке …».",
    "en": "Hi! Write what you would say to the speaker: \"play some radio\", \"set a timer for five minutes\", "
          "\"what's the weather\". To make the speaker say something aloud: /say text or \"say on the speaker …\".",
}


def refusal(user_id: int) -> str:
    """What a stranger is told: their own id is what the owner needs to let them in."""
    return (f"Извини, я тебя не знаю. Твой id: {user_id}; владелец может добавить его в allowed_users.\n"
            f"Sorry, I don't know you. Your id is {user_id}; the owner can add it to allowed_users.")


def split_message(text: str, limit: int) -> list[str]:
    """Pieces of at most `limit` characters, cut at a line break or a space when there is one."""
    text = text.strip()
    pieces = []
    while len(text) > limit:
        cut = max(text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        if cut < limit // 2:
            cut = limit
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return pieces + [text] if text else pieces


def name_score(query: str, name: str) -> float:
    """How well a spoken name matches a contact; «Мише» and «Маше» are the dative of «Миша» and «Маша»."""
    q, n = (s.casefold().replace("ё", "е").strip(" .,!?") for s in (query, name))
    best = score(q, n)
    if len(q) > 3 and len(n) > 3:
        best = max(best, score(q[:-1], n[:-1]))
    return best


class Service:
    """One chat platform's config: the allowlist, contacts and notification settings. The subclasses talk
    to the platform (`run`, `send`, `close`)."""

    name = ""  # "telegram"
    title = ""  # as spoken in Russian: «в телеграме»

    def __init__(self, chat: "Chat", cfg: Config, scfg: dict):
        self.chat, self.cfg, self.scfg = chat, cfg, scfg
        self.enabled = bool(scfg["enabled"])
        self.names: dict[int, str] = {int(k): str(v) for k, v in scfg["users"].items()}
        self.allowed: set[int] = {int(x) for x in scfg["allowed_users"]} | set(self.names)
        self.contacts: dict[str, str] = {str(k): str(v) for k, v in scfg["contacts"].items()}
        self.default_contact: str = scfg["default_contact"]
        self.notify_to: str = str(scfg["notify_to"]).strip()
        self.notify_kinds: set[str] = set(scfg["notify"])
        self.notify_always: bool = bool(scfg["notify_always"])

    def token(self) -> str:
        """$token_env, if set, wins over the owner-only file at token_file."""
        token = os.environ.get(self.scfg["token_env"], "").strip()
        if not token and self.scfg.get("token_file"):
            path = self.cfg.file(self.scfg["token_file"])
            if path.exists():
                token = path.read_text().strip()
        return token

    def book(self) -> dict[str, str]:
        """Who can be written to: the people in `users` (their id is the chat) plus `contacts`."""
        book = {name: str(uid) for uid, name in self.names.items()}
        book.update(self.contacts)
        return book

    def person(self, user_id: int, display: str) -> str:
        return self.names.get(user_id) or display

    async def run(self) -> None:  # pragma: no cover - platform
        raise NotImplementedError

    async def send(self, target: str, text: str) -> None:  # pragma: no cover - platform
        raise NotImplementedError

    async def close(self) -> None:
        pass


class Chat:
    def __init__(self, brain: "Brain", cfg: Config):
        from .discord_bot import DiscordBot
        from .telegram import TelegramBot
        self.brain, self.cfg = brain, cfg
        self.services: dict[str, Service] = {}
        for cls in (TelegramBot, DiscordBot):
            if cls.name not in cfg.data:  # a config from before the chat bots: they are off
                continue
            svc = cls(self, cfg, cfg[cls.name])
            if svc.enabled:
                self.services[svc.name] = svc
        self._tasks: set[asyncio.Task] = set()
        self._locks: dict[tuple[str, int], asyncio.Lock] = {}
        self._refused: dict[tuple[str, int], float] = {}

    # Lifecycle -----------------------------------------------------------
    def start(self) -> None:
        for svc in self.services.values():
            log.info("%s: enabled for %d allowed user(s)", svc.name, len(svc.allowed))
            if not svc.allowed:
                log.warning("%s: allowed_users and users are empty, so everyone is refused", svc.name)
            self._spawn(self._supervise(svc))

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for svc in self.services.values():
            try:
                await svc.close()
            except Exception:
                log.exception("%s: closing failed", svc.name)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _supervise(self, svc: Service) -> None:
        """Keep a platform's connection running; whatever it raises, the brain carries on."""
        delay = 5.0
        while True:
            started = time.monotonic()
            try:
                await svc.run()
                return  # it gave up on purpose (no token, bad token)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("%s: connection failed; retrying in %.0fs", svc.name, delay)
            delay = 5.0 if time.monotonic() - started > 300 else min(delay * 2, 300.0)
            await asyncio.sleep(delay)

    def spawn(self, coro) -> None:
        """Run a handler beside the platform's receive loop, so one slow answer does not block the next."""
        self._spawn(coro)

    # Incoming messages ---------------------------------------------------
    async def admit(self, svc: Service, user_id: int, display: str, reply: Reply) -> bool:
        """True for an allowed user. A stranger is logged with their id and refused politely, once an hour."""
        if user_id in svc.allowed:
            return True
        key = (svc.name, user_id)
        now = time.monotonic()
        if now - self._refused.get(key, -REFUSE_EVERY) >= REFUSE_EVERY:
            if len(self._refused) >= MAX_STRANGERS:
                self._refused = {k: v for k, v in self._refused.items() if now - v < REFUSE_EVERY}
            self._refused[key] = now
            log.warning("%s: refused user id %s (%.40r); to allow them add the id to allowed_users", svc.name,
                        user_id, display)
            try:
                await reply(refusal(user_id))
            except Exception as e:
                log.warning("%s: could not send the refusal: %s", svc.name, e)
        else:
            log.debug("%s: ignoring user id %s (refused recently)", svc.name, user_id)
        return False

    async def handle(self, svc: Service, user_id: int, display: str, text: str, reply: Reply,
                     typing: Callable[[], Awaitable[None]] | None = None) -> None:
        """One chat message from `user_id`: refuse, run a command, or answer it like a spoken request."""
        if not await self.admit(svc, user_id, display, reply):
            return
        name = svc.person(user_id, display)
        async with self._locks.setdefault((svc.name, user_id), asyncio.Lock()):  # one answer at a time, in order
            try:
                if typing:
                    await typing()
                out = await asyncio.wait_for(self._respond(svc, user_id, name, text), ANSWER_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning("%s: no answer for %s in %ds: %r", svc.name, name, ANSWER_TIMEOUT, text)
                out = "Слишком долго, не дождалась ответа. / That took too long."
            except Exception:
                log.exception("%s: chat turn failed", svc.name)
                out = "Что-то я сломалась. / Something broke."
        try:
            await reply(out)
        except Exception as e:
            log.warning("%s: could not send the answer: %s", svc.name, e)

    async def _respond(self, svc: Service, user_id: int, name: str, text: str) -> str:
        text = text.strip()
        lang = detect_lang(text)
        m = _COMMAND.match(text)
        if m:
            word, arg = m.group(1).lower(), (m.group(2) or "").strip()
            if word in ("say", "сказать", "скажи"):
                return await self.say(name, arg)
            if word in ("start", "help", "помощь"):
                return START["ru"] + "\n" + START["en"]
        m = _SAY_NATURAL.match(text)
        if m:
            return await self.say(name, m.group("msg"))
        log.info("%s: %s [%s]: %s", svc.name, name, lang, text)
        out = []
        with self.brain.chat_scope(f"{svc.name}:{user_id}", name):
            async for sentence in self.brain.answer(text, lang, speaker=name):
                out.append(sentence)
            if self.brain.pending:  # «Спросить большую модель?»: a person typing needs longer than 30 s to answer
                self.brain.pending.expires = time.monotonic() + CHAT_CONFIRM_SECONDS
        said = " ".join(s.strip() for s in out if s.strip())
        log.info("%s: -> %s", svc.name, said)
        return said or ("Готово." if lang == "ru" else "Done.")

    async def say(self, name: str, msg: str) -> str:
        """/say: the speaker says «<name> пишет: …» as soon as it is free."""
        msg = msg.strip()
        if not msg:
            return "Что сказать? /say текст\nWhat should I say? /say text"
        if not self.brain.clients:
            return "Колонка сейчас не на связи, говорить некому.\nThe speaker isn't connected right now."
        lang = detect_lang(msg)
        await self.brain.speak(f"{name} пишет: {msg}" if lang == "ru" else f"{name} writes: {msg}", lang, queue=True)
        return "Передано на колонку." if lang == "ru" else "Passed to the speaker."

    # Speaker -> chat -----------------------------------------------------
    async def command(self, text: str, lang: str) -> str | None:
        """«напиши Мише в телеграм, что я скоро буду»: send the message and say how it went (None: not for us)."""
        m = _WRITE.match(text.strip().rstrip("."))
        if not m:
            return None
        ru = lang == "ru"
        word = m.group("svc").casefold()
        kind = "discord" if word.startswith(("дискор", "discord")) else "telegram"
        svc = self.services.get(kind)
        title = {"telegram": ("телеграм", "Telegram"), "discord": ("дискорд", "Discord")}[kind][0 if ru else 1]
        if svc is None:
            return f"{title.capitalize()} не настроен." if ru else f"{title} isn't set up."
        book = svc.book()
        who, msg = (m.group("who") or "").strip(), (m.group("msg") or "").strip()
        if not who:
            who, msg = self._leading_contact(book, msg)
        if who.casefold() in _ME:
            me = self.brain.speaker
            if not me or not any(c.casefold() == me.casefold() for c in book):
                return f"Я не знаю, как тебе писать в {title}е." if ru else f"I don't know how to message you on {title}."
            who = me
        if not who:
            who = svc.default_contact
            if not who and len(book) == 1:
                who = next(iter(book))
        known = ", ".join(book) or ("никого" if ru else "nobody")
        if not who:
            return (f"Кому писать? В {title}е я знаю: {known}." if ru else
                    f"Who should I write to? On {title} I know: {known}.") if book else \
                (f"В {title}е у меня нет контактов." if ru else f"I have no contacts on {title}.")
        best = max(book, key=lambda c: name_score(who, c), default=None)
        if best is None or name_score(who, best) < NAME_MATCH:
            return (f"Нет такого контакта: {who}. В {title}е я знаю: {known}." if ru else
                    f"No such contact: {who}. On {title} I know: {known}.")
        if not msg:
            return "Что написать?" if ru else "What should I write?"
        sender = self.brain.speaker
        signed = (f"{sender} (с колонки): {msg}" if ru else f"{sender} (from the speaker): {msg}") if sender else \
            (f"С колонки: {msg}" if ru else f"From the speaker: {msg}")
        try:
            await asyncio.wait_for(svc.send(book[best], signed), SEND_TIMEOUT)
        except Exception as e:
            log.warning("%s: could not send to %s: %s", svc.name, best, e)
            return "Не получилось отправить." if ru else "I couldn't send it."
        log.info("%s: sent to %s: %s", svc.name, best, signed)
        return f"Отправлено в {title}: {best}." if ru else f"Sent to {best} on {title}."

    @staticmethod
    def _leading_contact(book: dict[str, str], msg: str) -> tuple[str, str]:
        """«напиши в телеграм Мише, что я скоро буду»: the name that opens the message, if it is a contact."""
        m = re.match(r"^(\S+?)(?P<sep>\s*[,:]\s*|\s+)(?:что\b\s*|that\b\s*)?(.+)$", msg, re.S)
        if not m:
            return "", msg
        first, sep = m.group(1), m.group("sep")
        needed = NAME_MATCH if "," in sep or ":" in sep else 0.85
        if book and max(name_score(first, c) for c in book) >= needed:
            return first, m.group(3).strip()
        return "", msg

    # Notifications -------------------------------------------------------
    def notify_soon(self, kind: str, text: str) -> None:
        """Forward `text` to the notify chat of every service that wants `kind` ("rings", "agents")."""
        if any(svc.notify_to and kind in svc.notify_kinds for svc in self.services.values()):
            self._spawn(self.notify(kind, text))

    async def notify(self, kind: str, text: str) -> None:
        for svc in self.services.values():
            if not svc.notify_to or kind not in svc.notify_kinds:
                continue
            if self.brain.clients and not svc.notify_always:
                continue  # the speaker said it itself
            try:
                await asyncio.wait_for(svc.send(svc.notify_to, text), SEND_TIMEOUT)
                log.info("%s: notified (%s): %s", svc.name, kind, text)
            except Exception as e:
                log.warning("%s: notification failed: %s", svc.name, e)
