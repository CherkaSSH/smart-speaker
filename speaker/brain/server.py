"""The thinking half: conversation, tools, timers. Talks to voice clients over a WebSocket.

Protocol (JSON text frames):
  voice -> brain  {"type": "hello"}
                  {"type": "utterance", "id": str, "text": str, "lang": "ru"|"en"}
                  {"type": "cancel"}            stop the current answer (barge-in, "стоп")
  brain -> voice  {"type": "volume", "level": 0..10}
                  {"type": "say", "id": str, "text": str, "lang": str}
                  {"type": "turn_end", "id": str, "expect_reply": bool}
                  {"type": "ring", "text": str, "lang": str}   a timer or alarm went off
                  {"type": "ask", "text": str, "lang": str}    a question of the speaker's own («Включить
                                                               мою волну?»); the reply comes as an utterance

HTTP: POST /ask {"text", "lang"?, "speak"?, "big"?, "harness"?} answers in JSON,
for testing without audio; "speak": true also sends it to the voice service.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from aiohttp import WSMsgType, web

from ..config import Config
from ..lang import detect as detect_lang
from . import intents, speech
from .escalate import Harness, RemoteLLM
from .llm import LLM
from .memory import Memory
from .music import MY_WAVE, Music, asks_for_music
from .timers import Item, Scheduler
from .tools import BIG_MODEL_SCHEMA, SCHEMAS, WEB_SEARCH_SCHEMA, Tools
from .websearch import WebSearch

log = logging.getLogger("brain")

CONFIRM_SECONDS = 30  # how long a "should I ...?" question waits for its yes/no
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


class NeedsConfirmation(Exception):
    """Raised from a tool call that must wait for the user's yes/no."""

    def __init__(self, kind: str, args: dict):
        super().__init__(kind)
        self.kind = kind  # "big" or "search"
        self.kind_tool = "ask_big_model" if kind == "big" else "web_search"
        self.tool_args = args  # not `args`: Exception turns that into a tuple


@dataclass
class Pending:
    kind: str  # "big", "search" or "music"
    messages: list[dict]  # the conversation up to and including the paused tool call
    call_id: str
    args: dict
    text: str  # the user's original request
    lang: str
    search: asyncio.Task | None = None  # web search already running while the user answers
    expires: float = field(default_factory=lambda: time.monotonic() + CONFIRM_SECONDS)

    def cancel(self) -> None:
        if self.search:
            self.search.cancel()


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
- Each user message starts with the current date, time and reply language in square brackets. Don't mention the date or time unless asked.
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
        self.scheduler = Scheduler(self._fire, self._save)
        self.tools = Tools(self)
        self.llm = LLM(cfg["llm"])
        self.system = cfg["character"]["prompt"].strip() + "\n" + RULES
        # The remote model gets no tool definitions: it must not try to call any.
        self.remote_system = self.system + "- In this conversation you have no tools: answer directly from what you know " \
                                           "and from any information given in the message.\n"
        r, h = cfg["remote_llm"], cfg["harness"]
        self.memory = Memory(cfg["memory"], cfg.file(cfg["memory"]["state_file"]))
        self.remote = RemoteLLM(r) if r["enabled"] else None
        self.harness = Harness(h, str(cfg.file(h["cwd"]))) if h["enabled"] else None
        self.big_triggers = r["triggers"]
        self.harness_triggers = h["triggers"]
        self.web = WebSearch(cfg["web_search"])
        self.music = Music(self, cfg["music"], cfg.file(cfg["music"]["library"]))
        self.confirm_big = bool(r["confirm"])
        self.offer_big_for_search = bool(cfg["web_search"]["offer_big_model"])
        self.schemas = SCHEMAS + ([WEB_SEARCH_SCHEMA] if self.web.enabled else []) \
            + ([BIG_MODEL_SCHEMA] if self._big_ready else [])
        self.pending: Pending | None = None  # a question the speaker asked and awaits yes/no for
        self.clients: set[web.WebSocketResponse] = set()
        self.turn: asyncio.Task | None = None
        self._load()

    @property
    def _big_ready(self) -> bool:
        return bool(self.remote and self.remote.configured)

    # State -------------------------------------------------------------
    def _load(self) -> None:
        if self.state_path.exists():
            d = json.loads(self.state_path.read_text())
            self.volume = d.get("volume", self.volume)
            self.scheduler.restore(d.get("items", []))

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"volume": self.volume, "items": self.scheduler.dump()}, ensure_ascii=False))
        tmp.replace(self.state_path)

    async def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except ConnectionError:
                self.clients.discard(ws)

    async def set_volume(self, level: int) -> None:
        self.volume = max(0, min(10, int(level)))
        self._save()
        await self.broadcast({"type": "volume", "level": self.volume})

    async def _fire(self, item: Item) -> None:
        ru = item.lang == "ru"
        if item.label:
            text = f"Напоминаю: {item.label}." if ru else f"Reminder: {item.label}."
        elif item.kind == "alarm":
            text = "Подъём, будильник!" if ru else "Wake up, that's your alarm!"
        else:
            text = f"Таймер на {speech.duration(item.seconds, 'ru', accusative=True)} всё." if ru else \
                f"Your {speech.duration(item.seconds, 'en')} timer is done."
        log.info("ring: %s", text)
        await self.broadcast({"type": "ring", "text": text, "lang": item.lang})

    async def offer(self, kind: str, question: str, lang: str) -> None:
        """Ask on the speaker's own initiative (the songs the user asked for ended); the next
        utterance's yes or no resolves it, like the speaker's other questions."""
        if self.pending:
            self.pending.cancel()
        self.pending = Pending(kind, [], "", {}, question, lang)
        log.info("asking: %s", question)
        await self.broadcast({"type": "ask", "text": question, "lang": lang})

    # Turns -------------------------------------------------------------
    def _stamp(self, lang: str) -> str:
        now = dt.datetime.now()
        reply = "отвечай по-русски" if lang == "ru" else "reply in English"
        return f"[{now:%Y-%m-%d %H:%M}, {_RU_WEEKDAYS[now.weekday()]}; {reply}]"

    @staticmethod
    def _trigger(text: str, triggers: list[str]) -> str | None:
        folded = text.casefold()
        for trigger in sorted(triggers, key=len, reverse=True):
            if folded.startswith(trigger.casefold()):
                tail = text[len(trigger):]
                if not tail or not tail[0].isalnum():
                    return tail.lstrip(" \t\r\n,:;—-")
        return None

    async def answer(self, text: str, lang: str, big: bool = False, harness: bool = False):
        """Yield the spoken answer to one utterance, sentence by sentence."""
        pending, self.pending = self.pending, None
        if pending:
            choice = yes_no(text) if not (big or harness) and time.monotonic() < pending.expires else None
            if choice is not None:
                log.info("confirmation %s: %s", pending.kind, "yes" if choice else "no")
                async for sentence in self._resolve(pending, choice):
                    yield sentence
                return
            pending.cancel()  # the user moved on to something else
        if big and harness:
            raise ValueError("big and harness are mutually exclusive")
        mode = "big" if big else "harness" if harness else ""
        prompt = text
        if mode:
            triggers = self.big_triggers if big else self.harness_triggers
            stripped = self._trigger(text, triggers)
            if stripped is not None:
                prompt = stripped
        else:
            for candidate, triggers in (("big", self.big_triggers), ("harness", self.harness_triggers)):
                stripped = self._trigger(text, triggers)
                if stripped is not None:
                    mode, prompt = candidate, stripped
                    break
        if mode:
            if not prompt:
                yield "Что передать?" if lang == "ru" else "What should I pass along?"
            elif mode == "big":
                if not self.remote or not self.remote.configured:
                    yield "Большая модель не настроена." if lang == "ru" else "The big model is not configured."
                else:
                    messages = [{"role": "system", "content": self.remote_system},
                                {"role": "user", "content": f"{self._stamp(lang)} {prompt}"}]
                    async for sentence in self.remote.respond(messages):
                        yield sentence
            else:
                if not self.harness or not self.harness.configured:
                    yield "Внешний агент не настроен." if lang == "ru" else "The external harness is not configured."
                else:
                    async for sentence in self.harness.respond(prompt):
                        yield sentence
            return
        if self.fast_commands:
            reply = await intents.handle(self, text, lang)
            if reply is not None:
                log.info("fast command: %r -> %r", text, reply)
                if reply:  # "" = done silently (pausing music)
                    yield reply
                return
        # No conversation history: background speech and old topics must never
        # influence this request. Only timers, alarms, volume and explicitly
        # saved memories persist; memories closely related to this request ride along.
        recalled = await self.memory.recall(text)
        if recalled:
            log.info("recalled memories: %s", recalled)
        context = f" [Saved facts about the user: {' '.join(recalled)}]" if recalled else ""
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"{self._stamp(lang)}{context} {text}"}]
        async for sentence in self._llm(messages, text, lang, self.schemas, confirm=True):
            yield sentence

    async def _llm(self, messages: list[dict], text: str, lang: str, schemas: list[dict], confirm: bool):
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
            return await self.tools.call(name, args, lang)

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

    async def _resolve(self, p: "Pending", yes: bool):
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
                        {"role": "user", "content": f"{self._stamp(p.lang)} {p.args.get('question') or p.text}"}]
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
                            {"role": "user", "content": f"{self._stamp(p.lang)} {p.text}\n\nWeb search results "
                                                        f"(English; answer in the reply language):\n{found}"}]
                async for sentence in self.remote.respond(messages):
                    yield sentence
                return
        else:  # big model declined: LFM3 answers on its own
            result = {"declined": "Answer this yourself as well as you can; don't mention the big model."}
        p.messages.append({"role": "tool", "tool_call_id": p.call_id, "content": json.dumps(result, ensure_ascii=False)})
        local = [s for s in self.schemas if s is not BIG_MODEL_SCHEMA]
        async for sentence in self._llm(p.messages, p.text, p.lang, local, confirm=False):
            yield sentence

    async def _run_turn(self, turn_id: str, text: str, lang: str,
                        big: bool = False, harness: bool = False) -> None:
        said = []
        t0 = time.monotonic()
        try:
            async for sentence in self.answer(text, lang, big=big, harness=harness):
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

    def start_turn(self, turn_id: str, text: str, lang: str,
                   big: bool = False, harness: bool = False) -> None:
        self.cancel_turn()
        log.info("utterance %s [%s]: %s", turn_id, lang, text)
        self.turn = asyncio.create_task(self._run_turn(turn_id, text, lang, big, harness))

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
                elif kind == "utterance":
                    self.start_turn(d.get("id") or uuid.uuid4().hex[:8], d["text"], d.get("lang", "ru"))
                elif kind == "cancel":
                    self.cancel_turn()
                elif kind == "music_event":
                    await self.music.on_event(d["event"], d["id"], d.get("detail", ""))
        finally:
            self.clients.discard(ws)
            log.info("voice disconnected")
        return ws

    async def ask_handler(self, request: web.Request) -> web.Response:
        d = await request.json()
        text = d["text"]
        lang = d.get("lang") or detect_lang(text)
        big, harness = bool(d.get("big")), bool(d.get("harness"))
        if big and harness:
            return web.json_response({"error": "big and harness are mutually exclusive"}, status=400)
        if d.get("speak"):
            turn_id = uuid.uuid4().hex[:8]
            self.start_turn(turn_id, text, lang, big, harness)
            await self.turn
            return web.json_response({"id": turn_id})
        t0 = time.monotonic()
        out, first = [], None
        async for s in self.answer(text, lang, big=big, harness=harness):
            first = first or time.monotonic() - t0
            out.append(s)
        return web.json_response({"text": " ".join(out), "sentences": out,
                                  "first_sentence_s": round(first or 0, 3),
                                  "total_s": round(time.monotonic() - t0, 3)})



async def serve(cfg: Config) -> None:
    brain = Brain(cfg)
    app = web.Application()
    app.add_routes([web.get("/ws", brain.ws_handler), web.post("/ask", brain.ask_handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    b = cfg["brain"]
    await web.TCPSite(runner, b["host"], int(b["port"])).start()
    log.info("brain listening on %s:%s", b["host"], b["port"])
    try:
        await brain.scheduler.run()
    finally:
        await brain.llm.close()
        await brain.memory.close()
        await runner.cleanup()
