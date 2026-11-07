"""Text chat with the speaker: the shared core of the Telegram and Discord bots.

A message from a known user is answered exactly like a spoken request (`Brain.answer`, so music,
timers, weather, memory and, for admins, the coding agents all work) but as text, without going
through the voice turn: it never cancels or talks over what the speaker is saying. Each user maps to
a speaker name (the same names as the voice profiles, so history and memories are shared with that
person's voice), otherwise their chat name.

Who is known: the ids in the config (`users`, `allowed_users`: admins) and the people registered at
runtime (chat_users.py). A stranger registers with a six-digit code the speaker says aloud («код для
телеграма»), which proves they stand next to it, or asks: their first message goes to the admins
(and to the speaker, if it is on) to approve or deny. Admins can also talk to the coding agents (shell
access) and manage users; plain users can't. The bots never listen on a port: Telegram long polling
and the Discord gateway only.

Also here: «/say» and «скажи на колонке …» (the speaker says the message aloud), the voice commands
«напиши Мише в телеграм …» (speaker -> chat), and notifications (missed rings, agents' messages) to a
configured chat.
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
from .chat_users import Registry
from .music import score

if TYPE_CHECKING:
    from .server import Brain

log = logging.getLogger("brain.chat")

ANSWER_TIMEOUT = 300  # seconds a chat message may take (a coding agent, the big model, a web search)
SEND_TIMEOUT = 30  # seconds for a message out of a voice command
CHAT_CONFIRM_SECONDS = 600  # how long a "should I ...?" question to a chat user waits for their yes/no
NAME_MATCH = 0.8  # how closely a spoken name must match a contact (music's fuzzy score)
MAX_NAME = 40  # characters of a chosen name

Reply = Callable[[str], Awaitable[None]]

_COMMAND = re.compile(r"^[/!]([a-zA-Zа-яА-Я]+)(?:@\w+)?(?:\s+(.*))?$", re.S)
# A registration: the six digits alone, or after /register (/start, /reg), optionally followed by a name.
_CODE_MESSAGE = re.compile(r"^\s*(?:[/!](?:register|reg|start|регистрация)(?:@\w+)?\s+)?((?:\d[\s-]*){6})"
                           r"(?:[\s,:]+(?P<name>\D.*))?\s*$", re.I | re.S)
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
# «код для телеграма», «дай код для дискорда», «код для чата» (either bot), "code for telegram".
_PAIR = re.compile(
    r"^(?:(?:дай|скажи|назови|сделай|создай|придумай|нужен|give|tell)\s+)?(?:мне\s+|me\s+)?(?:(?:новый|a|the|new)\s+)?"
    r"(?:код|code)\w*\s+(?:для|в|на|for|to)\s+(?:" + _SERVICE + r"|чат\w*|бот\w*|chat|bots?)\b", re.I)
# «кто есть в телеграме?», «кто в дискорде», "who is on telegram"
_WHO = re.compile(r"^(?:кто\s+(?:есть\s+|сейчас\s+)?(?:в|во)|who(?:'s|\s+is)?\s+(?:on|in))\s+" + _SERVICE + r"\b", re.I)
# «удали Пашу из телеграма», "remove Pasha from discord"
_REMOVE = re.compile(r"^(?:удали|убери|выгони|исключи|отключи|remove|kick)\s+(?P<who>.+?)\s+(?:из|from)\s+"
                     + _SERVICE + r"\b", re.I)
_ME = {"мне", "себе", "me", "myself"}
_TITLES = {"telegram": ("телеграм", "Telegram"), "discord": ("дискорд", "Discord")}

START = {
    "ru": "Привет! Напиши мне то же, что сказал бы колонке: «включи радио», «таймер на пять минут», «какая погода». "
          "Чтобы колонка сказала что-то вслух: {p}say текст или «скажи на колонке …». "
          "Своё имя для колонки: {p}name Имя.",
    "en": "Hi! Write what you would say to the speaker: \"play some radio\", \"set a timer for five minutes\", "
          "\"what's the weather\". To make the speaker say something aloud: {p}say text or \"say on the speaker …\". "
          "Your name for the speaker: {p}name Name.",
}
ADMIN_HELP = ("Админу: {p}users — кто есть и кто просится; {p}approve ID / {p}deny ID — ответить на запрос; "
              "{p}revoke ID или имя — убрать; {p}admin ID / {p}user ID — дать или снять права админа. "
              "Новым людям колонка даёт код: «код для {title}а».")


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


def _kind(word: str) -> str:
    return "discord" if word.casefold().startswith(("дискор", "discord")) else "telegram"


class Service:
    """One chat platform's config: owners, contacts and notification settings, plus its registered users.
    The subclasses talk to the platform (`run`, `send`, `close`, `ask_admin`)."""

    name = ""  # "telegram"
    title = ""  # as spoken in Russian: «телеграм»
    prefix = "/"  # how commands start: /say, !say

    def __init__(self, chat: "Chat", cfg: Config, scfg: dict):
        self.chat, self.cfg, self.scfg = chat, cfg, scfg
        self.enabled = bool(scfg["enabled"])
        self.names: dict[int, str] = {int(k): str(v) for k, v in scfg["users"].items()}
        # Ids in the config are the owners: admins that only the config can remove.
        self.owners: set[int] = {int(x) for x in scfg["allowed_users"]} | set(self.names)
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

    def member(self, uid: int) -> tuple[str | None, str] | None:
        """(speaker name or None, role) of someone who may use the bot; None for a stranger."""
        if uid in self.owners:
            return self.names.get(uid), "admin"
        user = self.chat.registry.get(self.name, uid)
        return (user.name, user.role) if user else None

    def admins(self) -> set[int]:
        return self.owners | {i for i, u in self.chat.registry.users(self.name).items() if u.role == "admin"}

    def target(self, uid: int) -> str:
        """Where a direct message to user `uid` goes (`send`'s target)."""
        return str(uid)

    def book(self) -> dict[str, str]:
        """Who can be written to: the people in `users`, the registered users and `contacts`."""
        book = {name: self.target(uid) for uid, name in self.names.items()}
        for uid, user in self.chat.registry.users(self.name).items():
            book.setdefault(user.name, self.target(uid))
        book.update(self.contacts)
        return book

    async def ask_admin(self, admin: int, uid: int, display: str) -> None:
        """Ask one admin to let a stranger in (Telegram overrides it with buttons)."""
        p = self.prefix
        await self.send(self.target(admin), f"{display} (id {uid}) просит доступ к боту. "
                                            f"Пустить: {p}approve {uid} · отказать: {p}deny {uid}")

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
        c = cfg.data.get("chat", {})
        self.registry = Registry(cfg.file(c.get("users_file", "state/chat_users.json")),
                                 code_seconds=60 * float(c.get("code_minutes", 10)))
        self.services: dict[str, Service] = {}
        for cls in (TelegramBot, DiscordBot):
            if cls.name not in cfg.data:  # a config from before the chat bots: they are off
                continue
            svc = cls(self, cfg, cfg[cls.name])
            if svc.enabled:
                self.services[svc.name] = svc
        self._tasks: set[asyncio.Task] = set()
        self._locks: dict[tuple[str, int], asyncio.Lock] = {}

    # Lifecycle -----------------------------------------------------------
    def start(self) -> None:
        for svc in self.services.values():
            known = len(svc.owners | set(self.registry.users(svc.name)))
            log.info("%s: enabled, %d known user(s)", svc.name, known)
            if not known:
                log.warning("%s: nobody is registered yet: say «код для %sа» to the speaker and send the code "
                            "to the bot", svc.name, svc.title)
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

    @staticmethod
    async def _reply(svc: Service, reply: Reply, text: str) -> None:
        try:
            await reply(text)
        except Exception as e:
            log.warning("%s: could not reply: %s", svc.name, e)

    # Strangers: registration and access requests -------------------------
    async def admit(self, svc: Service, uid: int, display: str, reply: Reply) -> bool:
        """True for a known user; a stranger goes through `stranger` (a request) instead."""
        if svc.member(uid):
            return True
        await self.stranger(svc, uid, display, "", reply)
        return False

    async def stranger(self, svc: Service, uid: int, display: str, text: str, reply: Reply) -> None:
        """A message from someone unknown: a pairing code registers them; otherwise their first message
        asks the admins (and the speaker) to let them in, and later ones are ignored until that's decided."""
        reg, p = self.registry, svc.prefix
        m = _CODE_MESSAGE.match(text)
        if m:
            if not reg.may_try(svc.name, uid):
                log.warning("%s: id %s keeps sending wrong codes; ignored", svc.name, uid)
                return
            code = reg.redeem(svc.name, uid, re.sub(r"\D", "", m.group(1)))
            if code is None:
                log.warning("%s: wrong or expired code from id %s (%.40r)", svc.name, uid, display)
                await self._reply(svc, reply, "Код не подошёл: он неверный, уже использован или устарел. "
                                              f"Попроси у колонки новый: «код для {svc.title}а».")
                return
            name = ((m.group("name") or "").strip() or code.name or display)[:MAX_NAME]
            role = "user" if svc.admins() else "admin"  # the first person on a platform runs it
            reg.add(svc.name, uid, name, role, "code")
            log.info("%s: %s (id %s) registered with a code as %s", svc.name, name, uid, role)
            text = (f"Готово, {name}, теперь я тебя знаю." + (" Ты админ: " + ADMIN_HELP.format(p=p, title=svc.title)
                                                             if role == "admin" else "")
                    + "\n" + START["ru"].format(p=p))
            await self._reply(svc, reply, text)
            return
        if reg.request(svc.name, uid) is not None:
            log.debug("%s: ignoring id %s: their request is pending or denied", svc.name, uid)
            return
        if not reg.open_request(svc.name, uid, display[:MAX_NAME]):
            log.warning("%s: too many open requests; ignoring id %s", svc.name, uid)
            return
        log.warning("%s: %s (id %s) asks for access", svc.name, display, uid)
        await self._reply(svc, reply,
                          "Я тебя пока не знаю. Запрос отправлен владельцу.\n"
                          f"Быстрее: скажи колонке «код для {svc.title}а» и пришли сюда шесть цифр, которые она назовёт.\n"
                          "I don't know you yet; the owner has been asked. Or ask the speaker for "
                          f"\"code for {_TITLES[svc.name][1]}\" and send me the six digits.")
        await self._ask_admins(svc, uid, display)

    async def _ask_admins(self, svc: Service, uid: int, display: str) -> None:
        for admin in svc.admins():
            try:
                await asyncio.wait_for(svc.ask_admin(admin, uid, display), SEND_TIMEOUT)
            except Exception as e:
                log.warning("%s: could not ask admin %s: %s", svc.name, admin, e)
        if not self.brain.clients:
            return
        from .server import yes_no

        async def decide(answer: str | None) -> str | None:
            choice = yes_no(answer) if answer else None
            if choice is None:
                return None if answer is None else f"Не поняла. Решите в {svc.title}е: {svc.prefix}approve {uid}."
            if choice:
                return f"Пустила: {display} теперь может писать боту." if await self.approve(svc, uid, "голосом") \
                    else "Запрос уже не действует."
            self.registry.deny(svc.name, uid)
            return "Ладно, не пускаю."
        await self.brain.speak(f"{display} просит доступ к {svc.title}у. Пустить?", "ru", queue=True,
                               pending=self.brain.expect_reply("ru", decide))

    async def approve(self, svc: Service, uid: int, by: str) -> str | None:
        """Let a requester in (a plain user, or the admin if the bot has none yet); their name, or None when
        there is no request."""
        req = self.registry.request(svc.name, uid)
        if req is None or svc.member(uid):
            return None
        role = "user" if svc.admins() else "admin"
        self.registry.add(svc.name, uid, req.name, role, by)
        log.info("%s: %s (id %s) approved by %s as %s", svc.name, req.name, uid, by, role)
        welcome = "Тебя пустили! / You're in.\n" + START["ru"].format(p=svc.prefix)
        if role == "admin":
            welcome += "\nТы админ: " + ADMIN_HELP.format(p=svc.prefix, title=svc.title)
        try:
            await asyncio.wait_for(svc.send(svc.target(uid), welcome), SEND_TIMEOUT)
        except Exception as e:
            log.warning("%s: could not tell %s: %s", svc.name, uid, e)
        return req.name

    def deny(self, svc: Service, uid: int, by: str) -> str | None:
        req = self.registry.request(svc.name, uid)
        if req is None or req.status != "pending":
            return None
        self.registry.deny(svc.name, uid)
        log.info("%s: %s (id %s) denied by %s", svc.name, req.name, uid, by)
        return req.name

    def _find(self, svc: Service, who: str) -> int | None:
        """A registered user by id or (fuzzy) name."""
        who = who.strip()
        users = self.registry.users(svc.name)
        if who.isdigit():
            return int(who) if int(who) in users else None
        best = max(users, key=lambda i: name_score(who, users[i].name), default=None)
        return best if best is not None and name_score(who, users[best].name) >= NAME_MATCH else None

    def roster(self, svc: Service, ids: bool) -> str:
        """Who is in a bot, for a spoken answer (names) or an admin in chat (with ids and requests)."""
        people = [f"{svc.names.get(i) or i} ({i}, владелец)" if ids else svc.names.get(i) or "владелец"
                  for i in sorted(svc.owners)]
        for uid, user in self.registry.users(svc.name).items():
            people.append(f"{user.name} ({uid}, {'админ' if user.role == 'admin' else 'пользователь'})" if ids
                          else user.name)
        text = (", ".join(people) if people else "никого") + "."
        if ids:
            waiting = self.registry.pending(svc.name)
            if waiting:
                text += "\nПросятся: " + ", ".join(f"{r.name} ({i})" for i, r in waiting.items()) + \
                        f". Ответ: {svc.prefix}approve ID / {svc.prefix}deny ID."
        return text

    @staticmethod
    def who(svc: Service, uid: int) -> str:
        """How a known user is named in the log and in «approved by»."""
        member = svc.member(uid)
        return (member[0] if member else None) or str(uid)

    async def manage(self, svc: Service, uid: int, role: str, word: str, arg: str) -> str | None:
        """Chat commands about who may use the bot; None when `word` isn't one of them."""
        reg, p = self.registry, svc.prefix
        if word in ("register", "reg"):
            return "Ты уже зарегистрирован."
        if word in ("name", "имя"):
            if not arg:
                return f"Как тебя называть? {p}name Имя"
            if reg.get(svc.name, uid) is None:
                return "Твоё имя задано в config.toml (users), там его и меняй."
            reg.update(svc.name, uid, name=arg[:MAX_NAME])
            return f"Теперь ты {arg[:MAX_NAME]}. Под этим именем колонка помнит тебя и твой голос."
        if word in ("leave", "выйти"):
            if reg.remove(svc.name, uid) is None:
                return "Тебя добавили в config.toml: убрать можно только там."
            return "Готово, я тебя забыла. Вернуться: код от колонки."
        if word not in ("users", "approve", "deny", "revoke", "admin", "user", "code"):
            return None
        if role != "admin":
            return "Это может только админ."
        if word == "users":
            return "В боте: " + self.roster(svc, ids=True)
        if word == "code":
            code = reg.issue_code(svc.name)
            minutes = round(reg.code_seconds / 60)
            return f"Код для нового человека: {code}. Он действует {minutes} мин, один раз."
        if not arg:
            return f"Кого? {p}{word} ID" + (" или имя" if word in ("revoke", "admin", "user") else "")
        if word in ("approve", "deny"):
            if not arg.isdigit():
                return f"Нужен id из запроса: {p}{word} ID. Запросы: {p}users"
            by = self.who(svc, uid)
            done = await self.approve(svc, int(arg), by) if word == "approve" else self.deny(svc, int(arg), by)
            if done is None:
                return "Такого запроса нет (или он уже решён)."
            return f"Пустила: {done}." if word == "approve" else f"Отказала: {done}."
        target = self._find(svc, arg)
        if target is None:
            return "Не нашла такого зарегистрированного. Список: " + p + "users"
        if word == "revoke":
            if target == uid:
                return f"Себя — через {p}leave."
            user = reg.remove(svc.name, target)
            log.info("%s: %s (id %s) removed by %s", svc.name, user.name, target, uid)
            return f"Убрала: {user.name}."
        new_role = "admin" if word == "admin" else "user"
        reg.update(svc.name, target, role=new_role)
        return f"{reg.get(svc.name, target).name} теперь {'админ' if new_role == 'admin' else 'обычный пользователь'}."

    # Incoming messages ---------------------------------------------------
    async def handle(self, svc: Service, user_id: int, display: str, text: str, reply: Reply,
                     typing: Callable[[], Awaitable[None]] | None = None) -> None:
        """One chat message from `user_id`: registration, a command, or an answer like a spoken request."""
        member = svc.member(user_id)
        if member is None:
            await self.stranger(svc, user_id, display, text, reply)
            return
        name, role = member[0] or display, member[1]
        async with self._locks.setdefault((svc.name, user_id), asyncio.Lock()):  # one answer at a time, in order
            try:
                if typing:
                    await typing()
                out = await asyncio.wait_for(self._respond(svc, user_id, name, role, text), ANSWER_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning("%s: no answer for %s in %ds: %r", svc.name, name, ANSWER_TIMEOUT, text)
                out = "Слишком долго, не дождалась ответа. / That took too long."
            except Exception:
                log.exception("%s: chat turn failed", svc.name)
                out = "Что-то я сломалась. / Something broke."
        await self._reply(svc, reply, out)

    async def _respond(self, svc: Service, user_id: int, name: str, role: str, text: str) -> str:
        text = text.strip()
        lang = detect_lang(text)
        m = _COMMAND.match(text)
        if m:
            word, arg = m.group(1).lower(), (m.group(2) or "").strip()
            if word in ("say", "сказать", "скажи"):
                return await self.say(name, arg)
            if word in ("start", "help", "помощь"):
                out = START["ru"].format(p=svc.prefix) + "\n" + START["en"].format(p=svc.prefix)
                return out + ("\n" + ADMIN_HELP.format(p=svc.prefix, title=svc.title) if role == "admin" else "")
            managed = await self.manage(svc, user_id, role, word, arg)
            if managed is not None:
                return managed
        m = _SAY_NATURAL.match(text)
        if m:
            return await self.say(name, m.group("msg"))
        log.info("%s: %s [%s, %s]: %s", svc.name, name, role, lang, text)
        out = []
        with self.brain.chat_scope(f"{svc.name}:{user_id}", name, role):
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

    # Voice commands --------------------------------------------------------
    async def command(self, text: str, lang: str) -> str | None:
        """Voice (or chat) commands about the bots; None when `text` isn't one."""
        text = text.strip().rstrip(".?!")
        for rx, handler in ((_PAIR, self._pair), (_WHO, self._who), (_REMOVE, self._remove), (_WRITE, self._write)):
            m = rx.match(text)
            if m:
                return await handler(m, lang == "ru")
        return None

    def _service(self, m: re.Match, ru: bool) -> tuple[Service | None, str]:
        kind = _kind(m.group("svc"))
        title = _TITLES[kind][0 if ru else 1]
        return self.services.get(kind), title

    def _trusted(self) -> bool:
        """Codes and removals come from the voice (someone at the speaker) or from a chat admin."""
        return not self.brain.in_chat or self.brain.role == "admin"

    async def _pair(self, m: re.Match, ru: bool) -> str:
        """«код для телеграма»: a six-digit code to register with, said aloud."""
        if not self._trusted():
            return "Код может дать только колонка или админ." if ru else "Only the speaker or an admin can give a code."
        if m.group("svc"):
            svc, title = self._service(m, ru)
            if svc is None:
                return f"{title.capitalize()} не настроен." if ru else f"{title} isn't set up."
            platform, where = svc.name, (f"боту в {title}е" if ru else f"to the {title} bot")
        else:
            if not self.services:
                return "Чат-боты не настроены." if ru else "No chat bots are set up."
            platform, where = None, ("боту" if ru else "to the bot")
        code = self.registry.issue_code(platform, self.brain.speaker)
        spaced = " ".join(code)
        minutes = round(self.registry.code_seconds / 60)
        log.info("pairing code issued for %s", platform or "any bot")
        if ru:
            return f"Код: {spaced}. Повторяю: {spaced}. Пришли его {where} в течение {minutes} минут."
        return f"The code is {spaced}. Again: {spaced}. Send it {where} within {minutes} minutes."

    async def _who(self, m: re.Match, ru: bool) -> str:
        svc, title = self._service(m, ru)
        if svc is None:
            return f"{title.capitalize()} не настроен." if ru else f"{title} isn't set up."
        return (f"В {title}е: " if ru else f"On {title}: ") + self.roster(svc, ids=False)

    async def _remove(self, m: re.Match, ru: bool) -> str:
        svc, title = self._service(m, ru)
        if svc is None:
            return f"{title.capitalize()} не настроен." if ru else f"{title} isn't set up."
        if not self._trusted():
            return "Это может только админ." if ru else "Only an admin can do that."
        uid = self._find(svc, m.group("who"))
        if uid is None:
            return (f"В {title}е нет такого: {m.group('who')}." if ru else f"No one called {m.group('who')} on {title}.")
        user = self.registry.remove(svc.name, uid)
        log.info("%s: %s (id %s) removed by voice", svc.name, user.name, uid)
        return f"Убрала из {title}а: {user.name}." if ru else f"Removed {user.name} from {title}."

    async def _write(self, m: re.Match, ru: bool) -> str:
        """«напиши Мише в телеграм, что я скоро буду»: send the message and say how it went."""
        svc, title = self._service(m, ru)
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
