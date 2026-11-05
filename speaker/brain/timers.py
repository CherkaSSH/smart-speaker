"""Timers and alarms that survive restarts. Fired items go to a callback."""

from __future__ import annotations

import asyncio
import datetime as dt
import itertools
import re
import time
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_DAY_WORDS = {
    "mon": "mon", "monday": "mon", "пн": "mon", "понедельник": "mon", "понедельникам": "mon",
    "tue": "tue", "tuesday": "tue", "вт": "tue", "вторник": "tue", "вторникам": "tue",
    "wed": "wed", "wednesday": "wed", "ср": "wed", "среда": "wed", "средам": "wed",
    "thu": "thu", "thursday": "thu", "чт": "thu", "четверг": "thu", "четвергам": "thu",
    "fri": "fri", "friday": "fri", "пт": "fri", "пятница": "fri", "пятницам": "fri",
    "sat": "sat", "saturday": "sat", "сб": "sat", "суббота": "sat", "субботам": "sat",
    "sun": "sun", "sunday": "sun", "вс": "sun", "воскресенье": "sun", "воскресеньям": "sun",
}
_GROUPS = {
    "daily": {"daily", "every day", "everyday", "each day", "каждый день", "ежедневно"},
    "weekdays": {"weekdays", "every weekday", "on weekdays", "будни", "по будням", "в будни",
                 "каждый будний день", "рабочие дни", "по рабочим дням"},
    "weekends": {"weekends", "every weekend", "on weekends", "выходные", "по выходным"},
}


def normalize_repeat(value: str | None) -> str:
    """Canonical repeat rule: "daily", "weekdays", "weekends" or "mon,wed,fri"; "" for none.

    The LLM tool gets free text ("weekdays", "mon, wed", "будни"), the fast command
    parser passes its own rules; both end up here.
    """
    text = " ".join(str(value or "").casefold().replace("ё", "е").replace(",", " ").split())
    if not text:
        return ""
    for rule, phrases in _GROUPS.items():
        if text in phrases:
            return rule
    days = {_DAY_WORDS[w] for w in re.split(r"[\s,;/]+", text) if w in _DAY_WORDS}
    if not days:
        return ""
    if days == set(_DAYS[:5]):
        return "weekdays"
    if days == set(_DAYS[5:]):
        return "weekends"
    if len(days) == 7:
        return "daily"
    return ",".join(d for d in _DAYS if d in days)


def repeat_days(repeat: str | None) -> set[int]:
    """Weekday numbers (Monday = 0) a repeat rule covers; empty for a one-off."""
    rule = normalize_repeat(repeat)
    days = {"daily": _DAYS, "weekdays": _DAYS[:5], "weekends": _DAYS[5:]}.get(rule, rule.split(","))
    return {_DAYS.index(d) for d in days if d in _DAYS}


def next_due(due: float, repeat: str | None, now: float | None = None) -> float:
    """The next time an alarm due at `due` repeats, after `now`.

    The clock time is taken from `due` and kept in local wall-clock terms, so
    «в семь» stays seven across daylight saving changes.
    """
    days = repeat_days(repeat)
    if not days:
        return due
    now = time.time() if now is None else now
    at = dt.datetime.fromtimestamp(due)
    start = max(at.date(), dt.datetime.fromtimestamp(now).date())
    for offset in range(8):
        date = start + dt.timedelta(days=offset)
        if date.weekday() in days:
            when = dt.datetime.combine(date, at.time())  # naive local time: timestamp() applies DST
            if when.timestamp() > now:
                return when.timestamp()
    return due + 7 * 86400


@dataclass
class Item:
    id: int
    kind: str  # "timer" | "alarm" | "sleep"
    due: float  # epoch seconds
    lang: str
    label: str = ""
    seconds: int = 0  # original length, timers only
    music: str | None = None  # alarm: what to play after the wake-up line ("" = any music); None = ring only
    repeat: str | None = None  # alarm: "daily", "weekdays", "weekends" or "mon,wed,fri"; None = once
    who: str | None = None  # the recognised person who set it; the ring says their name

    def remaining(self, now: float | None = None) -> int:
        return max(0, int(round(self.due - (now or time.time()))))


class Scheduler:
    def __init__(self, on_fire: Callable[[Item], Awaitable[None]], on_change: Callable[[], None]):
        self.items: dict[int, Item] = {}
        self._ids = itertools.count(1)
        self._on_fire = on_fire
        self._on_change = on_change
        self._wake = asyncio.Event()

    # Persistence -------------------------------------------------------
    def dump(self) -> list[dict]:
        return [asdict(i) for i in self.items.values()]

    def restore(self, items: list[dict]) -> None:
        for d in items:
            item = Item(**d)
            self.items[item.id] = item
        self._ids = itertools.count(max(self.items, default=0) + 1)
        self._wake.set()

    # API ---------------------------------------------------------------
    def add(self, kind: str, due: float, lang: str, label: str = "", seconds: int = 0,
            music: str | None = None, repeat: str | None = None, who: str | None = None) -> Item:
        item = Item(next(self._ids), kind, due, lang, label.strip(), seconds, music,
                    normalize_repeat(repeat) or None, who or None)
        self.items[item.id] = item
        self._changed()
        return item

    def again(self, item: Item, now: float | None = None) -> Item:
        """A repeating alarm back in the queue at its next occurrence, with the same id."""
        item.due = next_due(item.due, item.repeat, now)
        self.items[item.id] = item
        self._changed()
        return item

    def cancel(self, items: list[Item]) -> None:
        for i in items:
            self.items.pop(i.id, None)
        self._changed()

    def pending(self, kind: str | None = None) -> list[Item]:
        return sorted((i for i in self.items.values() if kind in (None, i.kind)), key=lambda i: i.due)

    def _changed(self) -> None:
        self._on_change()
        self._wake.set()

    async def run(self) -> None:
        while True:
            self._wake.clear()
            now = time.time()
            due = [i for i in self.items.values() if i.due <= now]
            for item in sorted(due, key=lambda i: i.due):
                self.items.pop(item.id, None)
                self._on_change()
                await self._on_fire(item)
            upcoming = self.pending()
            timeout = (upcoming[0].due - time.time()) if upcoming else None
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=max(0.0, timeout) if timeout is not None else None)
            except TimeoutError:
                pass
