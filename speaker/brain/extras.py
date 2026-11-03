"""Fast commands for the extras: sleep timer, morning show, news, exchange rates, shopping list.

These run before the music commands (a sleep timer stops the music) and before
the LLM, like the timers and the clock in intents.py.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from typing import TYPE_CHECKING

from . import morning, speech
from .numbers import parse_duration, parse_number
from .rates import Rates

if TYPE_CHECKING:
    from .server import Brain

_CANCEL = r"\b(отмени\w*|отключи\w*|выключи\w*|сбрось\w*|cancel|stop|delete|remove|turn off)\b"
_SLEEP = r"\b(таймер\w* сна|sleep timer|выключи\w* (музыку|колонку|его)|останови\w* (музыку|воспроизведение|его)|" \
         r"stop (the )?music|turn (the )?music off)\b"
_MORNING = r"\b(доброе утро|утреннее шоу|morning show|good morning)\b"
_NEWS = r"\b(новост\w*|news|headlines|что нового|what( s|s)? new)\b"
_RATE = r"\b(курс\w*|скольк\w*|стоит|стоимост\w*|переведи\w*|конверт\w*|rate|exchange|convert|price|рубл\w*|rubles?)\b"
_SHOPPING = r"\b(спис\w* покуп\w*|покупк\w*|shopping list|shopping)\b"
_SHOPPING_OUT = r"\b((в|во|к|из|to|on|into|from)\s+)?(спис\w*(\s+покуп\w*)?|покупк\w*|shopping list|" \
                r"the shopping list|my shopping list)\b"
_TARGET = (r"(?:английск\w*|англ\w*|english|немецк\w*|немец\w*|german|французск\w*|француз\w*|french|"
           r"испанск\w*|испан\w*|spanish|итальянск\w*|итальян\w*|italian|китайск\w*|китай\w*|chinese|"
           r"японск\w*|япон\w*|japanese|турецк\w*|турец\w*|turkish|корейск\w*|корей\w*|korean|"
           r"польск\w*|поль\w*|polish|украинск\w*|украин\w*|ukrainian|арабск\w*|араб\w*|arabic)")
_TRANSLATE = [
    rf"(?:скажи |подскажи |скажите )?(?:а )?как (?:будет |сказать |правильно |перевести )?(?:по-?)?({_TARGET})"
    rf"(?: (?:будет|слово|сказать))? (.+)$",
    rf"(?:скажи |подскажи |скажите )?(?:а )?как (?:будет |сказать |правильно |перевести )?(.+?) (?:по-?)({_TARGET})$",
    rf"(?:скажи |подскажи |скажите )?(?:а )?как (?:по-?)?({_TARGET}) (?:будет|слово|сказать) (.+)$",
    rf"(?:переведи|переведите|перевод) (?:на )?({_TARGET})(?: язык)?[ :]*(.+)$",
    rf"(?:переведи|переведите) (.+?) на ({_TARGET})$",
    rf"how (?:do|would|can) (?:you|i) say (.+?) in ({_TARGET})$",
    rf"how to say (.+?) in ({_TARGET})$",
    rf"translate (.+?) (?:in)?to ({_TARGET})$",
    rf"translate (?:in)?to ({_TARGET})[ :]+(.+)$",
    rf"what (?:is|s) (.+?) in ({_TARGET})$",
]
_LANGUAGES = ((("англ", "english"), "English"), (("нем", "german"), "German"), (("франц", "french"), "French"),
              (("испан", "spanish"), "Spanish"), (("итальян", "italian"), "Italian"),
              (("китай", "chinese"), "Chinese"), (("япон", "japanese"), "Japanese"),
              (("турец", "turkish"), "Turkish"), (("корей", "korean"), "Korean"),
              (("поль", "polish"), "Polish"), (("украин", "ukrainian"), "Ukrainian"),
              (("араб", "arabic"), "Arabic"))


def _translation(t: str) -> tuple[str, str] | None:
    """(text to translate, target language) for a translation request, else None."""
    for pattern in _TRANSLATE:
        m = re.fullmatch(pattern, t)
        if m:
            groups = [g for g in m.groups() if g]
            if len(groups) < 2:
                continue
            first, second = groups[0], groups[1]
            target = _language(first) or _language(second)
            phrase = second if _language(first) else first
            if target and phrase.strip():
                return phrase.strip(" ,.:;"), target
    return None


def _language(word: str) -> str | None:
    w = word.casefold()
    for stems, name in _LANGUAGES:
        if w.startswith(stems):
            return name
    return None


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _joined(names: list[str], lang: str, quoted: bool = False) -> str:
    names = [f"«{n}»" if quoted else n for n in names]
    if len(names) == 1:
        return names[0]
    last = " и " if lang == "ru" else " and "
    return ", ".join(names[:-1]) + last + names[-1]


def _split_items(rest: str) -> list[str]:
    rest = re.sub(_SHOPPING_OUT, " ", rest)
    rest = re.sub(r"\b(мне|пожалуйста|ещ[её]|please|the|my|some)\b", " ", rest)
    out = []
    for piece in re.split(r"\s*(?:[,;]|\bи\b|\band\b)\s*", rest):
        piece = piece.strip(" -")
        if piece and piece not in out:
            out.append(piece)
    return out


def _currency_amount(text: str, words: list[str], code: str) -> float | None:
    """The number said right before the currency: «100 долларов» -> 100."""
    index = next((i for i, w in enumerate(words) if Rates.code(w) == code), None)
    if index is None:
        return None
    for i in range(index - 1, -1, -1):
        value, end = parse_number(words, i)
        if value is not None and end == index:
            return value
    return None


async def _sleep(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    if _has(t, r"\b(таймер\w* сна|sleep timer)\b") and _has(t, _CANCEL):
        items = brain.scheduler.pending("sleep")
        if not items:
            return "Таймера сна и так нет." if ru else "There is no sleep timer."
        brain.scheduler.cancel(items)
        return "Отменил таймер сна." if ru else "Cancelled the sleep timer."
    if not _has(t, _SLEEP):
        return None
    seconds = parse_duration(t)
    if not seconds:
        return None  # «выключи музыку» without a delay: the music commands answer it
    brain.scheduler.cancel(brain.scheduler.pending("sleep"))  # a new sleep timer replaces the old one
    brain.scheduler.add("sleep", time.time() + seconds, lang, "", seconds)
    if ru:
        return f"Хорошо, выключу музыку через {speech.duration(seconds, 'ru', accusative=True)}."
    return f"I'll stop the music in {speech.duration(seconds, 'en')}."


async def _translate(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    parsed = _translation(t)
    if parsed is None:
        return None
    phrase, target = parsed
    messages = [{"role": "system", "content": f"Translate the user's text into {target}. Reply with the "
                                              "translation only: no explanations, no quotes, no questions."},
                {"role": "user", "content": phrase}]
    said = " ".join([sentence async for sentence in brain.llm.respond(messages, [], None, "en", max_rounds=1)])
    if not said.strip():
        return None
    return ("Перевод: " if ru else "Translation: ") + said.strip()


async def _morning(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    if _has(t, _MORNING):
        return await morning.show(brain, lang)
    return None


async def _news(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    if not _has(t, _NEWS):
        return None
    if not brain.news.enabled:
        return "Ленты новостей не настроены." if ru else "No news feeds are configured."
    titles = await brain.news.headlines(lang)
    if not titles:
        return "Новостей сейчас не нашёл." if ru else "I couldn't read the news right now."
    headlines = " ".join(title if title.endswith((".", "!", "?", "…")) else title + "." for title in titles)
    return ("Главные новости. " if ru else "Here are the top headlines. ") + headlines


async def _rates(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    code = Rates.code(t)
    if code is None:
        return None
    if not (_has(t, _RATE) or len(words) <= 2):
        return None
    amount = _currency_amount(t, words, code)
    try:
        said = await brain.rates.say(code, amount, lang)
    except Exception:
        return "Не получилось узнать курс." if ru else "I couldn't get the exchange rate."
    if said is None:
        return "Такой валюты у Центробанка нет." if ru else "The Central Bank doesn't quote that currency."
    return said


async def _shopping(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    if not _has(t, _SHOPPING):
        return None
    shopping = brain.shopping
    if _has(t, r"\b(очист\w*|clear|remove all|удали\w* (вс[её]|все)|убери\w* (вс[её]|все))\b"):
        if shopping.clear():
            return "Очистил список покупок." if ru else "Cleared the shopping list."
        return "Список покупок и так пуст." if ru else "The shopping list is already empty."
    m = re.search(r"\b(удали\w*|убери\w*|вычеркни\w*|remove|delete)\b(.*)$", t)
    if m:
        query = re.sub(_SHOPPING_OUT, " ", m.group(2)).strip(" -")
        if not query:
            return None
        removed = shopping.remove(query)
        if not removed:
            return f"Не нашёл «{query}» в списке покупок." if ru else f"I couldn't find {query} on the shopping list."
        return ("Удалил: " if ru else "Removed: ") + _joined(removed, lang) + "."
    m = re.search(r"\b(добав\w*|запиши\w*|внеси\w*|add|put)\b(.*)$", t)
    if m:
        items = _split_items(m.group(2))
        if not items:
            return None
        added = shopping.add(items)
        if not added:
            return "Это уже в списке покупок." if ru else "That's already on the shopping list."
        return ("Добавил: " if ru else "Added: ") + _joined(added, lang) + "."
    items = shopping.items()
    if not items:
        return "Список покупок пуст." if ru else "The shopping list is empty."
    return ("В списке покупок: " if ru else "On the shopping list: ") + _joined(items, lang) + "."


async def handle(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    """A fast answer for an extra command, or None when it is not one."""
    for handler in (_sleep, _translate, _morning, _news, _rates, _shopping):
        reply = await handler(brain, t, words, lang, ru)
        if reply is not None:
            return reply
    return None
