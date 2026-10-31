"""The thinking half: conversation, tools, timers. Talks to voice clients over a WebSocket.

Protocol (JSON text frames):
  voice -> brain  {"type": "hello"}
                  {"type": "utterance", "id": str, "text": str, "lang": "ru"|"en"}
                  {"type": "cancel"}            stop the current answer (barge-in, "стоп")
  brain -> voice  {"type": "volume", "level": 0..10}
                  {"type": "say", "id": str, "text": str, "lang": str}
                  {"type": "turn_end", "id": str, "expect_reply": bool}
                  {"type": "ring", "text": str, "lang": str}   a timer or alarm went off

HTTP: POST /ask {"text", "lang"?, "speak"?, "big"?, "harness"?} answers in JSON,
for testing without audio; "speak": true also sends it to the voice service.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import time
import uuid

from aiohttp import WSMsgType, web

from ..config import Config
from ..lang import detect as detect_lang
from . import intents, speech
from .escalate import Harness, RemoteLLM
from .llm import LLM
from .memory import Memory
from .timers import Item, Scheduler
from .tools import BIG_MODEL_SCHEMA, SCHEMAS, Tools

log = logging.getLogger("brain")

RULES = """
You are the voice of a home smart speaker, like Yandex Station's Alice. Everything you write is spoken aloud.
- Answer in the reply language given in the square brackets (Russian or English).
- Be brief: one or two short sentences. Longer only when asked (a story, a recipe, a joke).
- Plain speech only: no markdown, lists, emoji, links or URLs.
- Never think out loud. Work things out silently or with a tool (calculate also counts days to a date), then say only the result.
- Use the tools for timers, alarms, volume, weather, arithmetic and facts. Never say you did something without calling its tool.
- For questions about real people, places, books, history or science, call lookup instead of guessing.
- Save durable personal facts with save_memory. Questions about the user, their pets, family, things or preferences are answered from saved facts or search_memory, never lookup.
- When ask_big_model is available, use it automatically for requests beyond your reasoning or knowledge ability.
- Each user message starts with the current date, time and reply language in square brackets. Don't mention the date or time unless asked.
- You can't play music or control other devices; say so briefly if asked.
"""

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
        r, h = cfg["remote_llm"], cfg["harness"]
        self.memory = Memory(cfg["memory"], cfg.file(cfg["memory"]["state_file"]))
        self.remote = RemoteLLM(r) if r["enabled"] else None
        self.harness = Harness(h, str(cfg.file(h["cwd"]))) if h["enabled"] else None
        self.big_triggers = r["triggers"]
        self.harness_triggers = h["triggers"]
        self.schemas = SCHEMAS + ([BIG_MODEL_SCHEMA] if self.remote and self.remote.configured else [])
        self.clients: set[web.WebSocketResponse] = set()
        self.turn: asyncio.Task | None = None
        self._load()

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
        """Yield the spoken answer to one standalone utterance, sentence by sentence."""
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
                    messages = [{"role": "system", "content": self.system},
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
            if reply:
                log.info("fast command: %r -> %r", text, reply)
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

        async def run_tool(name: str, args: dict):
            return await self.tools.call(name, args, lang)

        async for sentence in self.llm.respond(messages, self.schemas, run_tool, lang):
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
