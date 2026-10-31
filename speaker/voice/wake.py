"""Find the wake phrase in a transcript, tolerating ASR spelling slips.

"Пей долбоеб", "эй долбаеб" and "Эй, долбаёб!" all match "эй долбаёб".
Words are compared on a rough phonetic form (unstressed о/а and е/и/э merge,
ё is е, voiced consonants devoice) with a similarity threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_PHONETIC = str.maketrans({"ё": "е", "о": "а", "э": "е", "и": "е", "ы": "е", "й": "", "ь": "", "ъ": "",
                           "б": "п", "в": "ф", "г": "к", "д": "т", "ж": "ш", "з": "с"})
# ASR sometimes writes a Russian name in Latin letters ("Dolbayob").
_LATIN = [("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"),
          ("ye", "е"), ("ts", "ц")] + list(zip("abvgdezijklmnoprstufhcy", "абвгдезийклмнопрстуфхцы"))
# One-word phrases need a closer match: short words collide easily ("толпа"
# scores 0.83 against "долбаёб"; a slurred "долбаёбен" scores 0.875).
SINGLE_WORD_MIN = 0.85


def words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def phonetic(word: str) -> str:
    if word.isascii():
        for lat, cyr in _LATIN:
            word = word.replace(lat, cyr)
    return word.translate(_PHONETIC)


@dataclass
class Match:
    start: int  # word index where the phrase starts
    end: int  # word index after the phrase
    score: float


class WakeMatcher:
    def __init__(self, phrases: list[str], threshold: float):
        # Longest first, so "эй долбаёб" wins over its tail "долбаёб".
        self.phrases = sorted(([phonetic(w) for w in words(p)] for p in phrases), key=len, reverse=True)
        self.threshold = threshold

    def find(self, text: str) -> Match | None:
        """The best-matching wake phrase; among equally good ones, the last.

        The command follows the latest call, and earlier ones may be our own
        voice or a repeat ("эй долбаёб... эй долбаёб, стоп").
        """
        ws = [phonetic(w) for w in words(text)]
        found: list[Match] = []
        for phrase in self.phrases:
            target = " ".join(phrase)
            n = len(phrase)
            threshold = max(self.threshold, SINGLE_WORD_MIN) if n == 1 else self.threshold
            for size in {max(1, n - 1), n, n + 1}:
                for i in range(0, len(ws) - size + 1):
                    score = SequenceMatcher(None, " ".join(ws[i:i + size]), target).ratio()
                    if score >= threshold:
                        found.append(Match(i, i + size, score))
        if not found:
            return None
        top = max(m.score for m in found)
        # Latest end wins; at the same end the longer span ("эй долбаёб" over "долбаёб").
        return max((m for m in found if m.score >= top - 0.05), key=lambda m: (m.end, -m.start))

    def strip(self, text: str) -> tuple[bool, str]:
        """(woken, the command after the wake phrase)."""
        m = self.find(text)
        if m is None:
            return False, text
        # Cut the original text after the phrase's last word, keeping case and punctuation.
        tokens = list(re.finditer(r"[\w']+", text))
        rest = text[tokens[m.end - 1].end():]
        return True, rest.strip(" ,.!?;:—-")
