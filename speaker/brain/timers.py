"""Timers and alarms that survive restarts. Fired items go to a callback."""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable


@dataclass
class Item:
    id: int
    kind: str  # "timer" | "alarm"
    due: float  # epoch seconds
    lang: str
    label: str = ""
    seconds: int = 0  # original length, timers only

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
    def add(self, kind: str, due: float, lang: str, label: str = "", seconds: int = 0) -> Item:
        item = Item(next(self._ids), kind, due, lang, label.strip(), seconds)
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
