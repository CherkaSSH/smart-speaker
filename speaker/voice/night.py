"""Night mode: answers, chimes and music at no more than the configured volume.

`[audio] night_start`, `night_end` and `night_volume`; the window may run past
midnight. Alarms and reminders ignore it: the brain says they are loud. The
brain's «включи ночной режим»/«выключи ночной режим» arrives as a
{"type": "night", "on": ...} message and overrides the clock until the brain
says otherwise («ночной режим по расписанию»).
"""

from __future__ import annotations

import datetime as dt


def _minutes(text) -> int | None:
    try:
        hh, mm = (int(x) for x in str(text).strip().split(":")[:2])
    except ValueError:
        return None
    return hh * 60 + mm if 0 <= hh < 24 and 0 <= mm < 60 else None


class Night:
    """The night volume cap. Equal or empty start and end turn it off."""

    def __init__(self, cfg: dict):
        self.start = _minutes(cfg.get("night_start"))
        self.end = _minutes(cfg.get("night_end"))
        volume = cfg.get("night_volume")
        self.volume = int(volume) if isinstance(volume, (int, float)) and not isinstance(volume, bool) else None
        self.enabled = None not in (self.start, self.end, self.volume) and self.start != self.end
        self.override: bool | None = None  # the brain's toggle: True/False, None follows the clock
        self.free_music = False  # the user set a volume at night: music plays at it, not at the cap

    def active(self, now: dt.datetime | None = None) -> bool:
        if self.override is not None:
            return self.override and self.enabled
        if not self.enabled:
            return False
        t = now or dt.datetime.now()
        minute = t.hour * 60 + t.minute
        if self.start < self.end:
            return self.start <= minute < self.end
        return minute >= self.start or minute < self.end  # past midnight

    def level(self, level: int, now: dt.datetime | None = None) -> int:
        """The effective volume: the night cap while the night mode is on."""
        if self.volume is not None and self.active(now):
            return min(int(level), self.volume)
        return int(level)
