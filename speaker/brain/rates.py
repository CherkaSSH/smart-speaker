"""Central Bank of Russia exchange rates, cached for an hour."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET

import aiohttp

from ..lang import ru_plural
from . import speech

log = logging.getLogger("brain.rates")

JSON_URL = "https://www.cbr-xml-daily.ru/daily_json.js"
XML_URL = "https://www.cbr.ru/scripts/XML_daily.asp"

# Spoken names -> ISO code. A stem matches any case form ("долларов", "долларом").
_CURRENCIES = {
    "USD": ("доллар", "бакс", "usd", "dollar", "buck"),
    "EUR": ("евро", "euro", "eur"),
    "CNY": ("юан", "yuan", "cny", "женьминьби"),
    "GBP": ("фунт", "gbp", "pound", "стерлинг"),
    "JPY": ("йен", "иен", "jpy", "yen"),
    "KZT": ("тенге", "kzt", "tenge"),
    "TRY": ("лир", "try", "lira"),
    "UAH": ("гривн", "uah", "hryvnia"),
    "BYN": ("белорусск", "byn"),
    "CHF": ("франк", "chf", "franc"),
}
_RU_NAMES = {  # nominative, genitive singular, genitive plural
    "USD": ("доллар", "доллара", "долларов"),
    "EUR": ("евро", "евро", "евро"),
    "CNY": ("юань", "юаня", "юаней"),
    "GBP": ("фунт", "фунта", "фунтов"),
    "JPY": ("иена", "иены", "иен"),
    "KZT": ("тенге", "тенге", "тенге"),
    "TRY": ("лира", "лиры", "лир"),
    "UAH": ("гривна", "гривны", "гривен"),
    "BYN": ("белорусский рубль", "белорусского рубля", "белорусских рублей"),
    "CHF": ("франк", "франка", "франков"),
}
_EN_NAMES = {"USD": "dollar", "EUR": "euro", "CNY": "yuan", "GBP": "pound", "JPY": "yen", "KZT": "tenge",
             "TRY": "lira", "UAH": "hryvnia", "BYN": "Belarusian ruble", "CHF": "franc"}
_EN_PLURALS = {"USD": "dollars", "EUR": "euros", "CNY": "yuan", "GBP": "pounds", "JPY": "yen", "KZT": "tenge",
               "TRY": "liras", "UAH": "hryvnias", "BYN": "Belarusian rubles", "CHF": "francs"}


def money(value: float, lang: str) -> str:
    """'81 рубль 20 копеек', '81.20 rubles'."""
    rub = int(value)
    kop = int(round((value - rub) * 100))
    if kop >= 100:
        rub, kop = rub + 1, kop - 100
    if lang == "ru":
        text = f"{speech.ru_number(rub)} {ru_plural(rub, 'рубль', 'рубля', 'рублей')}"
        if kop:
            text += f" {speech.ru_number(kop)} {ru_plural(kop, 'копейка', 'копейки', 'копеек')}"
        return text
    return f"{value:.2f} rubles"


class Rates:
    def __init__(self):
        self._values: dict[str, float] = {}  # rubles per one unit of the currency
        self._at = 0.0

    @staticmethod
    def code(text: str) -> str | None:
        """ISO code for a spoken or written currency name ("доллара", "usd")."""
        t = text.casefold().replace("ё", "е")
        for code, stems in _CURRENCIES.items():
            if any(re.search(rf"\b{re.escape(s)}", t) for s in stems):
                return code
        return None

    async def values(self) -> dict[str, float]:
        if self._values and time.time() - self._at < 3600:
            return self._values
        try:
            found = await self._fetch()
        except Exception:
            if self._values:  # the bank is unreachable right now: today's cached rates still hold
                log.warning("rates refresh failed; using the cached ones", exc_info=True)
                return self._values
            raise
        self._values, self._at = found, time.time()
        return found

    async def _fetch(self) -> dict[str, float]:
        try:
            data = await self._get(JSON_URL)
            out = {}
            for code, v in json.loads(data)["Valute"].items():
                out[code] = float(v["Value"]) / float(v.get("Nominal") or 1)
            return out
        except Exception:
            log.warning("cbr-xml-daily.ru failed, trying cbr.ru", exc_info=True)
        data = await self._get(XML_URL)
        out = {}
        for v in ET.fromstring(data).findall("Valute"):
            code = (v.findtext("CharCode") or "").strip()
            value = float((v.findtext("Value") or "0").replace(",", "."))
            nominal = float(v.findtext("Nominal") or 1)
            if code:
                out[code] = value / nominal
        return out

    async def _get(self, url: str) -> bytes:
        timeout = aiohttp.ClientTimeout(total=8)
        for attempt in range(2):  # a TLS handshake to the bank occasionally resets
            try:
                async with aiohttp.ClientSession(timeout=timeout,
                                                 headers={"User-Agent": "Mozilla/5.0 smart-speaker/0.1"}) as s:
                    async with s.get(url) as r:
                        r.raise_for_status()
                        return await r.read()
            except (aiohttp.ClientError, TimeoutError):
                if attempt == 1:
                    raise
                await asyncio.sleep(0.3)
        raise TimeoutError(url)

    async def say(self, code: str, amount: float | None, lang: str) -> str | None:
        """The spoken answer: the rate itself, or what `amount` of the currency is worth."""
        values = await self.values()
        rate = values.get(code)
        if rate is None:
            return None
        if amount is None and rate < 1:  # quoted like the bank: 100 yen at once
            amount = 100
        total = rate * (amount if amount is not None else 1)
        if lang == "ru":
            name = _RU_NAMES[code][0]
            if amount is None:
                return f"{name[:1].upper() + name[1:]} — {money(rate, 'ru')}."
            words = speech.ru_number(int(amount)) if float(amount).is_integer() else f"{amount:g}"
            plural = ru_plural(int(amount), *_RU_NAMES[code]) if float(amount).is_integer() else _RU_NAMES[code][2]
            said = f"{words} {plural} — это {money(total, 'ru')}."
            return said[:1].upper() + said[1:]
        en = _EN_NAMES[code]
        if amount is None:
            return f"The {en} is {total:.2f} rubles."
        name = en if amount == 1 else _EN_PLURALS[code]
        return f"{amount:g} {name} is {total:.2f} rubles."
