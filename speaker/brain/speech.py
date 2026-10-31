"""Render durations and times as words, with Russian plural and gender agreement."""

from __future__ import annotations

from num2words import num2words

from ..lang import ru_plural


def ru_number(n: int, feminine: bool = False) -> str:
    words = num2words(n, lang="ru")
    if feminine:
        if words.endswith("один"):
            words = words[: -len("один")] + "одна"
        elif words.endswith("два"):
            words = words[: -len("два")] + "две"
    return words


_RU_UNITS = [
    (3600, False, ("час", "часа", "часов")),
    (60, True, ("минута", "минуты", "минут")),
    (1, True, ("секунда", "секунды", "секунд")),
]
_EN_UNITS = [(3600, "hour"), (60, "minute"), (1, "second")]


def duration(seconds: int, lang: str, accusative: bool = False) -> str:
    """'пять минут', 'одну минуту тридцать секунд', 'two minutes'.

    `accusative` gives the form after "на"/"через": "на одну минуту".
    """
    seconds = max(0, int(seconds))
    parts = []
    if lang == "ru":
        for size, fem, forms in _RU_UNITS:
            n, seconds = divmod(seconds, size)
            if n:
                word = ru_number(n, fem)
                form = ru_plural(n, *forms)
                if accusative and fem and form.endswith("а") and word.endswith("одна"):
                    word, form = word[:-1] + "у", form[:-1] + "у"
                parts.append(f"{word} {form}")
        return " ".join(parts) or "ноль секунд"
    for size, unit in _EN_UNITS:
        n, seconds = divmod(seconds, size)
        if n:
            parts.append(f"{num2words(n)} {unit}{'' if n == 1 else 's'}")
    return " and ".join(parts) or "zero seconds"


def clock(hour: int, minute: int, lang: str) -> str:
    """'семь часов пятнадцать минут', 'seven fifteen'."""
    if lang == "ru":
        h = f"{ru_number(hour)} {ru_plural(hour, 'час', 'часа', 'часов')}"
        if minute == 0:
            return h + " ровно"
        return f"{h} {ru_number(minute, True)} {ru_plural(minute, 'минута', 'минуты', 'минут')}"
    h12 = hour % 12 or 12
    suffix = "a m" if hour < 12 else "p m"
    if minute == 0:
        return f"{num2words(h12)} o'clock {suffix}"
    m = num2words(minute) if minute >= 10 else f"oh {num2words(minute)}"
    return f"{num2words(h12)} {m} {suffix}"
