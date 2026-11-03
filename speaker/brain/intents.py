"""Instant answers for everyday speaker commands, without the LLM.

Like Yandex Station's own command grammar: timers, volume and the clock must
work every time and at once. Anything not matched here goes to the LLM, which
has the same abilities as tools.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from typing import TYPE_CHECKING

from . import books, extras, speech
from .music import MY_WAVE
from .numbers import parse_duration, parse_number

if TYPE_CHECKING:
    from .server import Brain

_RU_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
_RU_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
              "октября", "ноября", "декабря"]


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s:.,-]", " ", text)
    text = re.sub(r"(?<!\d)[.,](?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


_CANCEL = r"\b(отмени\w*|отключи\w*|выключи\w*|удали\w*|сбрось\w*|останови\w*|убери\w*|cancel|stop|delete|remove|turn off)\b"
_TIMER = r"\b(таймер\w*|будильник\w*|timer\w*|alarm\w*)\b"


def parse_clock(t: str) -> tuple[int, int] | None:
    """Time of day in "в 7:30", "на семь тридцать", "в семь вечера", "at 6 pm"."""
    m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", t)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
    else:
        words = t.split()
        hh = mm = None
        for i, w in enumerate(words):
            if w in ("в", "на", "at", "for") and i + 1 < len(words):
                h, j = parse_number(words, i + 1)
                if h is None or h > 24 or (j < len(words) and words[j].startswith(("мин", "сек", "min", "sec"))):
                    continue
                hh, mm = int(h), 0
                if j < len(words) and words[j].startswith("час"):
                    j += 1
                mins, _ = parse_number(words, j)
                if mins is not None and mins < 60:
                    mm = int(mins)
                break
        if hh is None:
            return None
    if _has(t, r"\b(вечера|дня|pm|p m)\b") and hh < 12:
        hh += 12
    elif _has(t, r"\b(ночи|утра|am|a m)\b") and hh == 12:
        hh = 0
    if hh > 23 or mm > 59:
        return None
    return hh, mm


def alarm_music(t: str) -> str | None:
    """Music to wake up to: «под Кино» -> "кино", «с музыкой» -> "" (any music), else None."""
    m = re.search(r"\b(?:под|с)\s+(.+)$", t)
    if m:
        tail = m.group(1).strip(" ,.")
        tail = re.sub(r"^(музыку|музыка|музыкой|music)\b\s*", "", tail).strip(" ,.")
        if tail in ("утра", "вечера", "дня", "ночи", "утром", "вечером"):
            return None
        return tail
    return "" if _has(t, r"\b(музык\w*|music)\b") else None


async def handle(brain: "Brain", text: str, lang: str) -> str | None:
    t = normalize(text)
    words = t.split()
    ru = lang == "ru"

    # Volume ------------------------------------------------------------
    if len(words) <= 5:
        m = re.search(r"\b(громкость|звук|volume)\b(?: \w+)?(?: на| to)? ([\w]+)$", t)
        level = None
        if m:
            level, _ = parse_number([m.group(2)], 0)
            if level is None and m.group(2) in ("максимум", "максимальн", "max", "maximum"):
                level = 10
        if level is not None:
            await brain.set_volume(int(level))
            return f"Громкость {speech.ru_number(brain.volume, True)}." if ru else f"Volume {speech.num2words(brain.volume)}."
        louder = _has(t, r"\b(громче|погромче|louder|volume up)\b")
        quieter = _has(t, r"\b(тише|потише|quieter|softer|volume down)\b")
        if louder != quieter:
            step = 3 if _has(t, r"\b(намного|гораздо|сильно|much|a lot)\b") else 1
            before = brain.volume
            await brain.set_volume(before + (step if louder else -step))
            if brain.volume == before:
                if ru:
                    return "Куда уж громче, и так на максимуме." if louder else "Тише уже некуда."
                return "Already at maximum." if louder else "Already at minimum."
            return f"Громкость {speech.ru_number(brain.volume, True)}." if ru else f"Volume {speech.num2words(brain.volume)}."

    # Extras: sleep timer, morning show, news, rates, shopping list -------
    reply = await extras.handle(brain, t, words, lang, ru)
    if reply is not None:
        return reply

    # Timers ------------------------------------------------------------
    if _has(t, _TIMER) and _has(t, _CANCEL):
        kind = "alarm" if _has(t, r"\b(будильник\w*|alarm\w*)\b") else "timer"
        items = brain.scheduler.pending(kind)
        if not items:
            return ("Будильников" if kind == "alarm" else "Таймеров") + " и так нет." if ru else f"There are no {kind}s."
        brain.scheduler.cancel(items)
        if ru:
            return "Отменил." if len(items) == 1 else f"Отменил все, {speech.ru_number(len(items))} штук."
        return "Cancelled." if len(items) == 1 else f"Cancelled all {speech.num2words(len(items))}."

    wake_me = _has(t, r"^(разбуди\w*|wake me)\b")
    # Labelled timers and alarms ("для пельменей", "с подписью работа") go to the LLM, which fills the label.
    labelled = _has(t, r"\b(для|чтобы|напомн\w*|подпис\w*|назов\w*|называ\w*|called|named|labell?ed|remind\w*|for the)\b")
    if (_has(t, r"\b(таймер\w*|timer)\b") or wake_me or _has(t, r"^засеки\b")) and not labelled:
        seconds = parse_duration(t)
        if seconds:
            brain.scheduler.add("timer", time.time() + seconds, lang, "", seconds)
            if ru:
                return f"Засёк {speech.duration(seconds, 'ru', accusative=True)}."
            return f"Timer set for {speech.duration(seconds, 'en')}."

    if (_has(t, r"\b(будильник\w*|alarm)\b") or wake_me) and not labelled:
        at = parse_clock(t)
        days = 2 if _has(t, r"\b(послезавтра|day after tomorrow)\b") else 1 if _has(t, r"\b(завтра|tomorrow)\b") else 0
        if at:
            hh, mm = at
            now = dt.datetime.now()
            when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if days:
                when += dt.timedelta(days=days)
            elif when <= now:
                when += dt.timedelta(days=1)
            music = alarm_music(t)
            brain.scheduler.add("alarm", when.timestamp(), lang, music=music)
            offset = (when.date() - now.date()).days
            day = (["сегодня", "завтра", "послезавтра"] if ru else ["today", "tomorrow", "the day after tomorrow"])[offset]
            if music is not None:
                tune = f" Под «{music}»." if music else " С музыкой."
                if ru:
                    return f"Будильник на {day}, {speech.clock(hh, mm, 'ru')}." + tune
                return f"Alarm set for {speech.clock(hh, mm, 'en')} {day}." + tune
            if ru:
                return f"Будильник на {day}, {speech.clock(hh, mm, 'ru')}."
            return f"Alarm set for {speech.clock(hh, mm, 'en')} {day}."

    if _has(t, r"\b(сколько (еще |времени )?осталось|сколько на таймере|how (much time|long) is left|time left)\b"):
        items = brain.scheduler.pending("timer")
        if not items:
            return "Таймеров нет." if ru else "No timers running."
        left = [speech.duration(i.remaining(), lang) for i in items]
        if ru:
            return "Осталось " + ", и ".join(left) + "."
        return " and ".join(left) + " left."

    # Audiobooks ---------------------------------------------------------
    reply = await books.handle(brain, t, words, ru)
    if reply is not None:
        return reply

    # Music -------------------------------------------------------------
    reply = await _music(brain, t, words, ru)
    if reply is not None:
        return reply

    # Clock -------------------------------------------------------------
    now = dt.datetime.now()
    if len(words) <= 6 and _has(t, r"\b(который час|сколько (сейчас )?времени|what time is it|what s the time|whats the time)\b"):
        return f"Сейчас {speech.clock(now.hour, now.minute, 'ru')}." if ru else \
            f"It's {speech.clock(now.hour, now.minute, 'en')}."
    if len(words) <= 6 and _has(t, r"\b(какое (сегодня )?число|какой сегодня день|какая сегодня дата|what s the date|what day is it|what is the date)\b"):
        if ru:
            day = speech.num2words(now.day, lang="ru", to="ordinal")
            day = day[:-2] + ("ье" if day.endswith("ий") else "ое")  # "тридцать первое", "третье"
            return f"Сегодня {_RU_WEEKDAYS[now.weekday()]}, {day} {_RU_MONTHS[now.month - 1]}."
        return f"Today is {now:%A, %B} {speech.num2words(now.day, to='ordinal')}."
    return None


_PLAY = r"^(включи|поставь|сыграй|играй|запусти|play|put on)\b ?(.*)$"
_FILLER = r"\b(мне|пожалуйста|please|какую нибудь|какую-нибудь|что нибудь|что-нибудь|some|the|a|музыку|музыка|music|" \
          r"песню|песня|song|трек|track)\b"
_DEVICES = r"\b(свет\w*|ламп\w*|телевизор\w*|тв|чайник\w*|кондиционер\w*|отоплени\w*|обогрев\w*|пылесос\w*|" \
           r"розетк\w*|light\w*|lamp\w*|tv|television|kettle|heater|vacuum)\b"
_LYRICS = r"\b(какие (там )?слова|какой (там )?текст|текст (песни|трека)|слова (песни|трека)|" \
          r"о чем (поет|поется|песня|трек)|что (там )?поется|lyrics|what are the (words|lyrics)|" \
          r"what s (it|the song|this song) about|what does it say)\b"


async def _music(brain: "Brain", t: str, words: list[str], ru: bool) -> str | None:
    """Music commands. "" means done silently: pausing shouldn't talk over the song."""
    music = brain.music
    short = len(words) <= 5
    active = music.state != "stopped"

    async def control(action: str, nothing: tuple[str, str]) -> str:
        return "" if await music.control(action) else nothing[0 if ru else 1]

    if short and _has(t, r"\b(режим\w* )?(диджея|диджей|dj)\b") \
            and _has(t, r"\b(включ\w*|вруб\w*|включай|on|enable|выключ\w*|отключ\w*|off|disable)\b"):
        music.announce = not _has(t, _CANCEL)
        if ru:
            return "Режим диджея включён." if music.announce else "Режим диджея выключен."
        return "DJ mode on." if music.announce else "DJ mode off."

    if short and _has(t, r"\b(повтор\w*|repeat\w*)\b"):
        if _has(t, _CANCEL) or _has(t, r"\b(не повторяй\w*|без повтора|no repeat)\b"):
            music.repeat = False
            return "Повтор выключен." if ru else "Repeat off."
        cur = music.current
        if cur is None:
            return "Сейчас ничего не играет." if ru else "Nothing is playing."
        if cur.live:
            return "Радио не повторяю." if ru else "I can't repeat a radio stream."
        music.repeat = True
        return f"Повторяю {cur.label('ru')}." if ru else f"Repeating {cur.label('en')}."

    if short and _has(t, r"\b(перемешай\w*|перемешать|shuffle\w*)\b"):
        cur = music.current
        if cur is None:
            return "Сейчас ничего не играет." if ru else "Nothing is playing."
        if cur.live or cur.station:
            return "В волне перемешивать нечего." if ru else "There's nothing to shuffle in a wave."
        music.shuffle()
        return "Перемешал остальное." if ru else "Shuffled the rest."

    dislike = short and (_has(t, r"\b(дизлайк\w*|dislike\w*)\b") or _has(t, r"\bне (нравится|понравил\w*)\b")
                         or _has(t, r"\b(don t|do not) like\b"))
    like = short and not dislike and not _has(t, r"\b(что|чем|какие|какой|какая|what|which|do you)\b") \
        and (_has(t, r"\bлайк\w*\b") or _has(t, r"\bнравится\b") or _has(t, r"\bi like (it|this|that|the song)\b"))
    if like or dislike:
        cur = music.current
        if cur is None:
            return "Сейчас ничего не играет." if ru else "Nothing is playing."
        if cur.provider != "yandex":
            return "Лайки работают только в Яндекс Музыке." if ru else "Likes only work with Yandex Music."
        if dislike:
            if not await music.dislike():
                return "Не получилось." if ru else "That didn't work."
            return "Не нравится — пропускаю." if ru else "Disliked; skipping."
        if not await music.like():
            return "Не получилось." if ru else "That didn't work."
        return "Поставила лайк." if ru else "Liked."

    if _has(t, _LYRICS):
        cur = music.current
        if cur is None:
            return "Сейчас ничего не играет." if ru else "Nothing is playing."
        lines = await music.lyric_lines()
        if not lines:
            return "У этого трека нет текста." if ru else "This track has no lyrics."
        spoken = " ".join(lines)
        if len(spoken) > 260:
            spoken = spoken[:260].rsplit(" ", 1)[0] + "…"
        return f"Вот: «{spoken}»" if ru else f"Here: “{spoken}”"

    if short and _has(t, r"\b(пауз\w*|pause)\b"):
        return await control("pause", ("Ничего не играет.", "Nothing is playing."))
    if short and _has(t, r"\b(продолж\w*|сними (с )?паузу?|играй дальше|resume|continue|unpause)\b"):
        return await control("resume", ("Нечего продолжать.", "Nothing to resume."))
    if short and (_has(t, r"\b(следующ\w*|next|skip)\b") or active and _has(t, r"^(дальше|переключи\w*)$")):
        return await control("next", ("Это была последняя.", "That was the last one."))
    if short and _has(t, r"\b(предыдущ\w*|прошл\w* (песн\w*|трек\w*)|previous)\b"):
        return await control("previous", ("Раньше ничего не было.", "There's nothing before this."))
    if short and (_has(t, r"\b(выключи\w*|останови\w*|стоп|хватит|stop|turn off)\b.*\b(музык\w*|песн\w*|радио|music|song|radio)\b")
                  or active and _has(t, r"^(выключи|выключай|останови|stop it|turn it off)$")):
        return await control("stop", ("Музыка и так не играет.", "No music is playing."))
    if _has(t, r"\b(что (сейчас |это )?(играет|за песня|звучит)|какая (это |сейчас )?песня|what s playing|what is playing|what song)\b"):
        cur = music.current
        if cur is None:
            return "Сейчас ничего не играет." if ru else "Nothing is playing."
        return f"Это {cur.label('ru')}." if ru else f"This is {cur.label('en')}."
    if _has(t, r"\b(волн\w*|радио) по (это\w* |текущ\w* )?(трек\w*|песн\w*)|\bmore like this\b|"
               r"\b(songs|music|something) (like this|similar)\b|\bwave (from|based on) this\b") \
            or active and short and _has(t, r"^(включи|поставь|давай|play) (\w+ )?(похож\w*|similar)\b"):
        cur = music.current
        if cur is None:
            return "Сейчас ничего не играет." if ru else "Nothing is playing."
        if await music.similar() is None:
            return "Похожего не нашёл." if ru else "I couldn't find anything similar."
        return f"Включаю волну по треку {cur.label('ru')}." if ru else f"Playing songs like {cur.label('en')}."
    m = re.search(_PLAY, t)
    if m:
        rest = re.sub(_FILLER, " ", m.group(2))
        radio = _has(rest, r"\b(радио|radio|волн\w*|wave)\b")  # Yandex calls its radio «волна»
        query = re.sub(r"\s+", " ", rest).strip()
        station = re.sub(r"\b(радио|radio)\b", "", query).strip()
        if _has(query, _DEVICES):
            return None  # "включи свет": the LLM explains it can't control devices
        if not brain.clients:
            return "Голосовой модуль не подключён." if ru else "The voice service is not connected."
        lang = "ru" if ru else "en"
        if _has(rest, r"\b(подкаст\w*|podcast\w*)\b"):
            name = re.sub(r"\b(подкаст\w*|podcast\w*)\b", " ", query).strip()
            if not name:
                return "Какой подкаст включить?" if ru else "Which podcast?"
            track = await music.play_podcast(name, lang)
            if track is None:
                return f"Не нашёл подкаст «{name}»." if ru else f"I couldn't find the podcast {name}."
            return (f"Включаю подкаст «{track.album}», с последнего выпуска." if ru
                    else f"Playing the podcast “{track.album}”, latest episode first.")
        if radio:
            track = await music.play(station, True, lang)
        elif query and _has(m.group(2), r"\b(музык\w*|music)\b"):  # «спокойную музыку»: a wave, else a search
            track = await music.play(query, True, lang) or await music.play(query, False, lang)
        else:
            track = await music.play(query, False, lang)
        if track is None:
            if m.group(1) in ("поставь", "put on"):
                return None  # "поставь напоминание ...": not necessarily music
            return f"Не нашёл «{query}»." if ru else f"I couldn't find {query}."
        if track.station == MY_WAVE and query:
            return "Включаю мою волну." if ru else "Playing My Wave."
        if track.station and track.station != MY_WAVE:
            return f"Включаю волну {track.station}." if ru else f"Playing the {track.station} wave."
        if radio:
            return f"Включаю радио {track.title}." if ru else f"Playing {track.title} radio."
        if not query:
            return "Включаю музыку." if ru else "Playing some music."
        return f"Включаю {track.label('ru')}." if ru else f"Playing {track.label('en')}."
    return None
