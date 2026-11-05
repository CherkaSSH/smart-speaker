"""The last few exchanges, per speaker, so follow-ups like «а завтра?» work.

History is a convenience, not a memory: it expires after `[brain] history_minutes`
of silence and keeps only the last few exchanges, so background speech and old
topics never colour a new request. Durable facts belong in memory.py.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass, field


@dataclass
class Exchange:
    at: dt.datetime  # when the user said it, for the stamp the model sees
    mono: float  # time.monotonic() of the same moment, for expiry
    lang: str
    text: str
    answer: str  # what the speaker said, joined into one string
    facts: list[str] = field(default_factory=list)  # compact tool calls behind the answer


class History:
    """Recent exchanges keyed by the recognised speaker (None = unknown)."""

    def __init__(self, minutes: int, limit: int = 4):
        self.minutes = max(0, int(minutes))
        self.limit = limit
        self._items: dict[str | None, list[Exchange]] = {}

    @property
    def enabled(self) -> bool:
        return self.minutes > 0

    def add(self, speaker: str | None, lang: str, text: str, answer: str,
            facts: list[str] | None = None) -> None:
        if not self.enabled or not text.strip() or not answer.strip():
            return
        now = time.monotonic()
        self._expire(speaker, now)
        items = self._items.setdefault(speaker, [])
        items.append(Exchange(dt.datetime.now(), now, lang, text, answer, list(facts or [])))
        del items[:-self.limit]

    def recent(self, speaker: str | None) -> list[Exchange]:
        if not self.enabled:
            return []
        self._expire(speaker, time.monotonic())
        return list(self._items.get(speaker, []))

    def _expire(self, speaker: str | None, now: float) -> None:
        items = self._items.get(speaker)
        if items and now - items[-1].mono > self.minutes * 60:
            del self._items[speaker]  # silence broke the thread


def format_fact(name: str, args: dict, result: object) -> str:
    """One line for the next turn: which tool ran, with what, and what it found."""
    try:
        found = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        found = str(result)
    if len(found) > 160:
        found = found[:160].rsplit(" ", 1)[0] + "…"
    asked = ", ".join(f"{k}={v}" for k, v in args.items() if v not in ("", None, {}))
    return f"{name}({asked}) -> {found}"
