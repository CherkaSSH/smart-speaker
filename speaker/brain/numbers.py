"""Spoken numbers and durations in Russian and English ("пять минут", "полтора часа", "2 minutes")."""

from __future__ import annotations

import re

_RU_UNITS = {
    "ноль": 0, "один": 1, "одну": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
    "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_RU_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
            "семьдесят": 70, "восемьдесят": 80, "девяносто": 90}
_RU_HUNDREDS = {"сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
                "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900}
_EN_UNITS = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90}

# Unit stems -> seconds. Matched by prefix so every case form works
# ("минуту", "минуты", "минут").
_UNITS = [
    ("сек", 1), ("мин", 60), ("час", 3600),
    ("sec", 1), ("min", 60), ("hour", 3600), ("hr", 3600),
]
_HALF = {"пол": 0.5, "полминуты": None}  # "полчаса", "пол минуты"
_ONE_AND_HALF = ("полтор",)  # полтора / полторы


def _unit_seconds(word: str) -> int | None:
    for stem, sec in _UNITS:
        if word.startswith(stem):
            return sec
    return None


def parse_number(words: list[str], i: int) -> tuple[float | None, int]:
    """Parse a number starting at words[i]. Returns (value, next index)."""
    total = 0
    seen = False
    last_rank = 4  # 3 hundreds, 2 tens, 1 units: each word must be a smaller rank ("семь тридцать" is two numbers)
    j = i
    while j < len(words):
        w = words[j]
        if re.fullmatch(r"\d+([.,]\d+)?", w):
            if seen:
                break
            return float(w.replace(",", ".")), j + 1
        v = (_RU_HUNDREDS.get(w) or _RU_TENS.get(w) or _EN_TENS.get(w))
        rank = 3 if w in _RU_HUNDREDS else 2
        if v is None:
            v = _RU_UNITS.get(w, _EN_UNITS.get(w))
            rank = 1
        if v is None:
            if w in ("and", "и") and seen:
                j += 1
                continue
            break
        # "a"/"an" count as one only right before a unit ("a minute").
        if w in ("a", "an") and not (j + 1 < len(words) and _unit_seconds(words[j + 1])):
            break
        if rank >= last_rank or (rank == 1 and v >= 10 and last_rank == 2):
            break
        total += v
        last_rank = rank
        seen = True
        j += 1
    return (float(total) if seen else None), j


def parse_duration(text: str) -> int | None:
    """Seconds in a spoken duration, or None if there is no duration."""
    words = re.findall(r"[\w.,]+", text.lower().replace("ё", "е"))
    words = [w.strip(".,") for w in words]
    seconds = 0.0
    found = False
    i = 0
    while i < len(words):
        w = words[i]
        # полчаса / полминуты / пол часа
        if w.startswith("пол") and not w.startswith(_ONE_AND_HALF) and len(w) > 3 and _unit_seconds(w[3:]):
            seconds += 0.5 * _unit_seconds(w[3:])
            found = True
            i += 1
            continue
        if w == "пол" and i + 1 < len(words) and _unit_seconds(words[i + 1]):
            seconds += 0.5 * _unit_seconds(words[i + 1])
            found = True
            i += 2
            continue
        if w.startswith(_ONE_AND_HALF) and i + 1 < len(words) and _unit_seconds(words[i + 1]):
            seconds += 1.5 * _unit_seconds(words[i + 1])
            found = True
            i += 2
            continue
        if w == "half" and i + 2 < len(words) and words[i + 1] in ("a", "an") and _unit_seconds(words[i + 2]):
            seconds += 0.5 * _unit_seconds(words[i + 2])
            found = True
            i += 3
            continue
        value, j = parse_number(words, i)
        if value is not None and j < len(words) and _unit_seconds(words[j]):
            unit = _unit_seconds(words[j])
            # "5 с половиной минут", "two and a half minutes"
            seconds += value * unit
            found = True
            i = j + 1
            continue
        if value is not None and j + 2 < len(words) and words[j] == "с" and words[j + 1].startswith("половин") \
                and _unit_seconds(words[j + 2]):
            seconds += (value + 0.5) * _unit_seconds(words[j + 2])
            found = True
            i = j + 3
            continue
        # A bare unit means one of it: "на минуту", "на час", "for an hour" is handled above.
        unit = _unit_seconds(w)
        if unit and i > 0 and words[i - 1] in ("на", "через", "for", "in"):
            seconds += unit
            found = True
        i += 1
    return int(round(seconds)) if found and seconds > 0 else None
