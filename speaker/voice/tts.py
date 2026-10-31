"""Silero text-to-speech (Russian and English) with spoken-form text normalisation."""

from __future__ import annotations

import logging
import re
import threading
import time
import warnings

import numpy as np
import torch
from num2words import num2words

from ..lang import detect, ru_plural

log = logging.getLogger("voice.tts")

_LATIN_TO_CYR = [("shch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("th", "з"), ("ph", "ф"),
                 ("oo", "у"), ("ee", "и"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"), ("ts", "ц"), ("ck", "к"),
                 ("qu", "кв"), ("w", "в"), ("x", "кс"), ("q", "к"), ("j", "дж")] + \
    list(zip("abcdefghiklmnoprstuvyz", "абкдефгхиклмнопрстувиз"))

# Units after a number: Russian forms for 1 / 2-4 / 5+, English singular / plural.
_RU_UNITS = {"°C": ("градус", "градуса", "градусов"), "°": ("градус", "градуса", "градусов"),
             "%": ("процент", "процента", "процентов"), "м/с": ("метр в секунду", "метра в секунду", "метров в секунду"),
             "км/ч": ("километр в час", "километра в час", "километров в час")}
_EN_UNITS = {"°C": ("degree", "degrees"), "°": ("degree", "degrees"), "%": ("percent", "percent"),
             "m/s": ("meter per second", "meters per second"), "km/h": ("kilometer per hour", "kilometers per hour")}
_NUMBER = r"[-−]?\d+(?:[.,]\d+)?"


def _translit(word: str) -> str:
    w = word.lower()
    for lat, cyr in _LATIN_TO_CYR:
        w = w.replace(lat, cyr)
    return w


def _number_words(raw: str, lang: str) -> str:
    neg = raw.startswith(("-", "−"))
    raw = raw.lstrip("-−").replace(",", ".")
    try:
        words = num2words(float(raw) if "." in raw else int(raw), lang=lang)
    except (ValueError, NotImplementedError, OverflowError):
        return raw
    return ("минус " if lang == "ru" else "minus ") + words if neg else words


def normalize(text: str, lang: str) -> str:
    """Digits, units and symbols to words; Latin words spelled in Cyrillic for the Russian voice."""
    units = _RU_UNITS if lang == "ru" else _EN_UNITS

    def with_unit(m: re.Match) -> str:
        raw, forms = m.group(1), units[m.group(2)]
        n = int(float(raw.lstrip("-−").replace(",", ".")))
        if lang == "ru":
            form = ru_plural(n, *forms) if "." not in raw and "," not in raw else forms[1]
        else:
            form = forms[0] if n == 1 else forms[1]
        return f" {_number_words(raw, lang)} {form} "

    unit_re = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
    text = re.sub(rf"(?<!\w)({_NUMBER})\s*({unit_re})", with_unit, text)
    # 07:30 -> 7 30, 07:00 -> 7
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: f"{int(m.group(1))} {'' if m.group(2) == '00' else int(m.group(2))}", text)
    text = re.sub(rf"(?<!\w){_NUMBER}", lambda m: f" {_number_words(m.group(0), lang)} ", text)
    text = text.replace("&", " и " if lang == "ru" else " and ").replace("+", " плюс " if lang == "ru" else " plus ")
    if lang == "ru":
        text = re.sub(r"[A-Za-z]+", lambda m: _translit(m.group(0)), text)
    text = re.sub(r"[«»\"“”„()\[\]{}<>*_#]", "", text)
    return re.sub(r"\s+([,.!?…])", r"\1", re.sub(r"\s+", " ", text)).strip()


class TTS:
    def __init__(self, cfg, tts_cfg: dict):
        torch.set_num_threads(int(tts_cfg["threads"]))
        self.rate = int(tts_cfg["sample_rate"])
        self.speakers = {"ru": tts_cfg["ru_speaker"], "en": tts_cfg["en_speaker"]}
        self.models = {}
        self._lock = threading.Lock()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Silero's packaged code has an old regex escape
            for lang in ("ru", "en"):
                t = time.monotonic()
                imp = torch.package.PackageImporter(str(cfg.file(tts_cfg[f"{lang}_model"])))
                self.models[lang] = imp.load_pickle("tts_models", "model")
                log.info("loaded %s TTS in %.1fs", lang, time.monotonic() - t)
        self.synth("Привет.")  # warm up: the first call is several times slower
        self.synth("Hello.")

    def synth(self, text: str) -> np.ndarray:
        """Float32 mono audio at `self.rate`; empty if nothing speakable.

        The voice follows the sentence's own script, so an English title inside a
        Russian answer, or an English answer, gets the right speaker.
        """
        lang = detect(text)
        clean = normalize(text, lang)
        if not re.search(r"\w", clean):
            return np.zeros(0, np.float32)
        with self._lock, torch.inference_mode():
            try:
                audio = self.models[lang].apply_tts(text=clean, speaker=self.speakers[lang], sample_rate=self.rate)
            except Exception:  # Silero rejects some inputs (only symbols, too long); skip the sentence
                log.exception("TTS failed for %r", clean)
                return np.zeros(0, np.float32)
        return audio.numpy().astype(np.float32)
