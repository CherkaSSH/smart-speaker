"""The morning show: greeting, time, weather, what is still due today, a few headlines, music."""

from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING

from ..lang import ru_plural
from . import speech

if TYPE_CHECKING:
    from .server import Brain

log = logging.getLogger("brain.morning")


def _degrees(value: int, lang: str) -> str:
    if lang != "ru":
        return f"{value} degrees"
    return f"{speech.ru_number(value)} {ru_plural(abs(value), 'градус', 'градуса', 'градусов')}"


def _weather(w: dict, lang: str) -> str:
    if not w or w.get("error"):
        return ""
    city = w.get("city", "")
    now, today = w.get("now"), w.get("today")
    if lang == "ru":
        text = f"Сейчас {_degrees(round(now['temperature_c']), 'ru')}, {now['sky']}." if now else ""
        if today:
            precip = int(today.get("precipitation_chance_percent") or 0)
            text += f" Днём ожидается {_degrees(round(today['max_c']), 'ru')}, {today['sky']}"
            text += f", осадки — {speech.ru_number(precip)} {ru_plural(precip, 'процент', 'процента', 'процентов')}."
        return text
    text = f"It's {now['temperature_c']} degrees and {now['sky']} in {city}." if now else ""
    if today:
        text += f" Today {today['min_c']} to {today['max_c']} degrees, {today['sky']}, " \
                f"{int(today.get('precipitation_chance_percent') or 0)}% chance of precipitation."
    return text


def _due_today(brain: "Brain", now: dt.datetime, lang: str) -> str:
    ru = lang == "ru"
    items = [i for i in brain.scheduler.pending()
             if dt.datetime.fromtimestamp(i.due).date() == now.date() and i.due > now.timestamp()]
    said = []
    for item in items:
        at = dt.datetime.fromtimestamp(item.due)
        clock = speech.clock(at.hour, at.minute, lang)
        if item.kind == "alarm":
            said.append(f"будильник на {clock}" if ru else f"an alarm at {clock}")
        elif item.label:
            said.append(f"напоминание «{item.label}» в {clock}" if ru else f"a reminder to {item.label} at {clock}")
        else:
            said.append(f"таймер в {clock}" if ru else f"a timer at {clock}")
    if not said:
        return ""
    if ru:
        return "На сегодня ещё " + ", и ".join(said) + "."
    return "Still today: " + ", ".join(said) + "."


async def show(brain: "Brain", lang: str) -> str:
    ru = lang == "ru"
    now = dt.datetime.now()
    if now.hour < 12:
        hello = "Доброе утро!" if ru else "Good morning!"
    elif now.hour < 18:
        hello = "Добрый день!" if ru else "Good afternoon!"
    else:
        hello = "Добрый вечер!" if ru else "Good evening!"
    parts = [hello, f"Сейчас {speech.clock(now.hour, now.minute, 'ru')}." if ru else
             f"It's {speech.clock(now.hour, now.minute, 'en')}."]
    try:
        weather = await brain.tools.call("get_weather", {"day": "today"}, lang)
    except Exception:
        log.warning("weather for the morning show failed", exc_info=True)
        weather = {}
    said = _weather(weather if isinstance(weather, dict) else {}, lang)
    if said:
        parts.append(said)
    due = _due_today(brain, now, lang)
    if due:
        parts.append(due)
    titles = await brain.news.headlines(lang, 3)
    if titles:
        headlines = " ".join(t if t.endswith((".", "!", "?", "…")) else t + "." for t in titles)
        parts.append(("В новостях. " if ru else "In the news. ") + headlines)
    track = await brain.music.play("", False, lang) if brain.clients else None
    if track is not None:
        parts.append("Включаю музыку." if ru else "Playing some music.")
    return " ".join(parts)
