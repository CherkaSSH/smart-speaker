"""Tools the LLM can call. Each returns a JSON-serialisable result for the model."""

from __future__ import annotations

import ast
import datetime as dt
import math
import operator
import time
from typing import TYPE_CHECKING, Any

import aiohttp

from . import speech

if TYPE_CHECKING:
    from .server import Brain


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []}}}


SCHEMAS = [
    _fn("set_timer", "Start a countdown timer. Also use it for reminders like 'remind me in 10 minutes to ...'.",
        {"seconds": {"type": "integer", "description": "Length in seconds"},
         "label": {"type": "string", "description": "What it is for, e.g. 'пельмени'. Optional."}},
        ["seconds"]),
    _fn("set_alarm", "Set an alarm clock for a time of day.",
        {"time": {"type": "string", "description": "24-hour HH:MM"},
         "date": {"type": "string", "description": "YYYY-MM-DD; omit for the next occurrence of that time"},
         "label": {"type": "string"}},
        ["time"]),
    _fn("list_timers", "List running timers and alarms with the time left.", {}),
    _fn("cancel_timers", "Cancel timers or alarms.",
        {"which": {"type": "string", "description": "'all', a label, or 'timer'/'alarm' for every one of that kind"}},
        ["which"]),
    _fn("set_volume", "Change the speaker volume (0-10).",
        {"level": {"type": "integer", "description": "Absolute level 0-10"},
         "change": {"type": "integer", "description": "Relative change, e.g. 1 or -2"}}),
    _fn("get_weather", "Current weather and forecast.",
        {"city": {"type": "string", "description": "City name; omit for home"},
         "day": {"type": "string", "enum": ["now", "today", "tomorrow", "day_after_tomorrow"]}}),
    _fn("lookup", "Look up facts on Wikipedia: people, places, history, science, books, films. "
        "Use it for any factual question instead of guessing.",
        {"query": {"type": "string", "description": "Short search query, e.g. 'Война и мир роман'"},
         "lang": {"type": "string", "enum": ["ru", "en"]}},
        ["query"]),
    _fn("calculate", "Evaluate arithmetic exactly. Use for any calculation, including dates.",
        {"expression": {"type": "string", "description":
                        "Python-style expression, e.g. '27*4', 'sqrt(2)', or days_until('2027-01-01') for days from today"}},
        ["expression"]),
    _fn("save_memory", "Save durable personal information or a preference when the user asks to remember it, "
        "or when it will clearly be useful in later requests. Do not save transient commands.",
        {"text": {"type": "string", "description": "Self-contained fact to remember"}}, ["text"]),
    _fn("search_memory", "Search saved personal facts semantically. Use when memory would help answer or the user "
        "asks about preferences, past requests, or what you remember.",
        {"query": {"type": "string", "description": "What to find, in natural language"},
         "limit": {"type": "integer", "description": "Maximum results, 1-10; default 5"}}, ["query"]),
]

WEB_SEARCH_SCHEMA = _fn(
    "web_search",
    "Search the web for current or specific information: news, recent events, prices, schedules, opening hours, "
    "or anything Wikipedia (lookup) wouldn't have. Write the objective and queries in English even when the user "
    "speaks Russian; then answer in the reply language.",
    {"objective": {"type": "string", "description": "One sentence in English: what to find out"},
     "queries": {"type": "array", "items": {"type": "string"},
                 "description": "1-3 English keyword queries of 3-6 words"}},
    ["objective", "queries"])

BIG_MODEL_SCHEMA = _fn(
    "ask_big_model",
    "Delegate difficult reasoning, coding, or synthesis to a larger remote model. Use automatically when the request "
    "exceeds your capabilities. Do not use for routine questions, device actions, calculations, or facts available "
    "through other tools.",
    {"question": {"type": "string", "description": "Self-contained question with all necessary context"}}, ["question"])

_WEATHER_RU = {0: "ясно", 1: "в основном ясно", 2: "переменная облачность", 3: "пасмурно", 45: "туман", 48: "изморозь",
               51: "лёгкая морось", 53: "морось", 55: "сильная морось", 56: "ледяная морось", 57: "ледяная морось",
               61: "небольшой дождь", 63: "дождь", 65: "ливень", 66: "ледяной дождь", 67: "ледяной дождь",
               71: "небольшой снег", 73: "снег", 75: "сильный снег", 77: "снежная крупа", 80: "ливни",
               81: "ливни", 82: "сильные ливни", 85: "снегопад", 86: "сильный снегопад", 95: "гроза",
               96: "гроза с градом", 99: "гроза с градом"}
_WEATHER_EN = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
               51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
               61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
               71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "showers", 81: "showers",
               82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
               96: "thunderstorm with hail", 99: "thunderstorm with hail"}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
          "log10": math.log10, "abs": abs, "round": round, "pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _FUNCS and not callable(_FUNCS[node.id]):
        return _FUNCS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "days_until" \
            and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return (dt.date.fromisoformat(node.args[0].value) - dt.date.today()).days
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and callable(_FUNCS.get(node.func.id)):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


class Tools:
    def __init__(self, brain: "Brain"):
        self.brain = brain
        self._geo_cache: dict[str, dict] = {}
        self._forecast_cache: dict[tuple, tuple[float, dict]] = {}
        self.lang = "ru"

    async def call(self, name: str, args: dict, lang: str) -> Any:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        self.lang = lang  # language of the current turn; one turn runs at a time
        try:
            return await fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}
        except TimeoutError:
            return {"error": "the service did not answer in time"}
        except Exception as e:  # tool failures go back to the model, which explains them
            return {"error": str(e) or type(e).__name__}

    # Timers ------------------------------------------------------------
    async def t_set_timer(self, seconds: int, label: str = "") -> dict:
        seconds = int(seconds)
        if seconds <= 0:
            return {"error": "seconds must be positive"}
        item = self.brain.scheduler.add("timer", time.time() + seconds, self.lang, label, seconds)
        return {"ok": True, "timer_id": item.id, "length": speech.duration(seconds, self.lang), "label": item.label}

    async def t_set_alarm(self, time: str, date: str = "", label: str = "") -> dict:
        hh, mm = (int(x) for x in time.strip().split(":")[:2])
        now = dt.datetime.now()
        if date:
            when = dt.datetime.combine(dt.date.fromisoformat(date), dt.time(hh, mm))
        else:
            when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if when <= now:
                when += dt.timedelta(days=1)
        if when <= now:
            return {"error": "that time has passed"}
        item = self.brain.scheduler.add("alarm", when.timestamp(), self.lang, label)
        return {"ok": True, "alarm_id": item.id, "rings_at": when.strftime("%Y-%m-%d %H:%M"),
                "in": speech.duration(int((when - now).total_seconds()) // 60 * 60, self.lang)}

    async def t_list_timers(self) -> dict:
        items = self.brain.scheduler.pending()
        return {"items": [{"kind": i.kind, "label": i.label, "left": speech.duration(i.remaining(), self.lang),
                           "at": dt.datetime.fromtimestamp(i.due).strftime("%H:%M")} for i in items]}

    async def t_cancel_timers(self, which: str = "all") -> dict:
        which = which.strip().lower()
        items = self.brain.scheduler.pending()
        if which in ("timer", "timers", "таймер", "таймеры"):
            items = [i for i in items if i.kind == "timer"]
        elif which in ("alarm", "alarms", "будильник", "будильники"):
            items = [i for i in items if i.kind == "alarm"]
        elif which not in ("all", "все", "всё"):
            items = [i for i in items if which in i.label.lower() or which == str(i.id)]
        self.brain.scheduler.cancel(items)
        return {"cancelled": len(items)}

    # Device ------------------------------------------------------------
    async def t_set_volume(self, level: int | None = None, change: int | None = None) -> dict:
        if level is None and change is None:
            change = 1
        new = int(level) if level is not None else self.brain.volume + int(change)
        await self.brain.set_volume(new)
        return {"volume": self.brain.volume}

    # Internet ----------------------------------------------------------
    async def _get(self, url: str, params: dict, timeout: float = 6) -> dict:
        timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "smart-speaker/0.1"}) as s:
            async with s.get(url, params=params) as r:
                r.raise_for_status()
                return await r.json(content_type=None)

    async def t_get_weather(self, city: str = "", day: str = "now") -> dict:
        city = city or self.brain.cfg["brain"]["home_city"]
        geo = self._geo_cache.get(city)
        if geo is None:
            res = await self._get("https://geocoding-api.open-meteo.com/v1/search",
                                  {"name": city, "count": 1, "language": self.lang, "format": "json"})
            if not res.get("results"):
                return {"error": f"city not found: {city}"}
            geo = self._geo_cache[city] = res["results"][0]
        key = (round(geo["latitude"], 2), round(geo["longitude"], 2))
        cached = self._forecast_cache.get(key)
        if cached and time.time() - cached[0] < 600:
            w = cached[1]
        else:
            w = await self._get("https://api.open-meteo.com/v1/forecast", {
                "latitude": geo["latitude"], "longitude": geo["longitude"], "timezone": "auto", "forecast_days": 3,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
                "wind_speed_unit": "ms"})
            self._forecast_cache[key] = (time.time(), w)
        names = _WEATHER_RU if self.lang == "ru" else _WEATHER_EN
        out: dict[str, Any] = {"city": geo.get("name", city)}
        if day in ("now", "today", ""):
            c = w["current"]
            out["now"] = {"temperature_c": round(c["temperature_2m"]), "feels_like_c": round(c["apparent_temperature"]),
                          "sky": names.get(c["weather_code"], ""), "wind_m_s": round(c["wind_speed_10m"])}
        idx = {"now": 0, "today": 0, "tomorrow": 1, "day_after_tomorrow": 2}.get(day, 0)
        d = w["daily"]
        out[day or "today"] = {"min_c": round(d["temperature_2m_min"][idx]), "max_c": round(d["temperature_2m_max"][idx]),
                               "sky": names.get(d["weather_code"][idx], ""),
                               "precipitation_chance_percent": d["precipitation_probability_max"][idx]}
        return out

    async def t_lookup(self, query: str, lang: str = "") -> dict:
        first = lang if lang in ("ru", "en") else self.lang
        # ru.wikipedia is sometimes unreachable from Russian networks: fall back to English.
        for wl in dict.fromkeys([first, "en"]):
            try:
                res = await self._get(f"https://{wl}.wikipedia.org/w/api.php", {
                    "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 1,
                    "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1, "format": "json"}, timeout=3.5)
            except (aiohttp.ClientError, TimeoutError):
                continue
            pages = list(res.get("query", {}).get("pages", {}).values())
            if pages:
                summary = pages[0].get("extract", "").replace("\u0301", "")[:700]  # drop stress marks
                return {"title": pages[0].get("title", ""), "summary": summary}
        return {"error": "nothing found"}

    async def t_calculate(self, expression: str) -> dict:
        expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
        value = _eval(ast.parse(expr, mode="eval"))
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        elif isinstance(value, float):
            value = round(value, 4)
        return {"result": value}

    # Memory and escalation ----------------------------------------------
    async def t_save_memory(self, text: str) -> dict:
        return await self.brain.memory.save(text)

    async def t_search_memory(self, query: str, limit: int = 5) -> dict:
        return {"items": await self.brain.memory.search(query, limit)}

    async def t_ask_big_model(self, question: str) -> dict:
        if not self.brain.remote or not self.brain.remote.configured:
            return {"error": "the big model is not configured"}
        messages = [{"role": "system", "content": self.brain.remote_system},
                    {"role": "user", "content": f"{self.brain._stamp(self.lang)} {question}"}]
        answer = [sentence async for sentence in self.brain.remote.respond(messages)]
        return {"answer": " ".join(answer)}

    async def t_web_search(self, objective: str = "", queries: list[str] | str | None = None, query: str = "") -> dict:
        # Small models mix up the argument names; take whatever text they gave.
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in (queries or []) + [query] if q]
        objective = objective or (queries[0] if queries else "")
        if not objective:
            return {"error": "give an objective or a query"}
        return {"results": await self.brain.web.search(objective, queries),
                "note": "Excerpts are in English; answer in the reply language."}
