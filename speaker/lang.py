"""Language helpers shared by the brain and the voice."""

from __future__ import annotations

import re


def ru_plural(n: int, one: str, few: str, many: str) -> str:
    """Russian noun form for a count: 1 минута, 2 минуты, 5 минут."""
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def detect(text: str) -> str:
    """'ru' if the text is mostly Cyrillic, else 'en'."""
    cyr = len(re.findall(r"[а-яёА-ЯЁ]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    return "ru" if cyr >= lat else "en"
