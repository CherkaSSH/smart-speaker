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

from . import books, extras, help, speech
from .music import MY_WAVE
from .numbers import parse_duration, parse_number
from .timers import next_due, normalize_repeat, repeat_days

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

# Not names: what a bare answer to «как тебя зовут?» may hold instead.
_NOT_NAMES = {"да", "нет", "не", "ну", "это", "меня", "мое", "моё", "имя", "зовут", "как", "what", "my", "name",
              "who", "i", "am", "is", "the", "ты", "вы", "я", "пожалуйста", "незнаю", "ой", "э", "эм", "ээ", "эээ",
              "хм", "ам", "так", "ладно", "хорошо", "привет", "здравствуй", "стоп", "отмена", "хватит", "yes", "no",
              "hi", "hello", "um", "uh", "ok", "okay", "cancel", "stop"}


def parse_name(text: str) -> str | None:
    """The name in «меня зовут Миша», "my name is Misha", or a bare «Миша» / «я Миша»."""
    m = re.search(r"(?:меня зовут|мо[её] имя|зови меня|my name is|call me)\s+([A-Za-zА-Яа-яЁё-]{2,})", text, re.I)
    if m:
        word = m.group(1)
    else:
        spoken = re.findall(r"[A-Za-zА-Яа-яЁё-]+", text)
        words = [w for w in spoken if w.casefold() not in _NOT_NAMES]
        if len(words) != 1 or len(spoken) > 2:
            return None  # a sentence is not a name
        word = words[0]
    if len(word) < 2 or word.casefold() in _NOT_NAMES:
        return None
    return word[:1].upper() + word[1:]


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
        if tail in ("утра", "вечера", "дня", "ночи", "утром", "вечером") \
                or re.match(r"(\d|по\b|кажд\w*|выходн\w*|будн\w*)", tail):
            return None  # «с понедельника», «с 7 утра»: not a song
        return tail
    return "" if _has(t, r"\b(музык\w*|music)\b") else None


_DAY_STEMS = (("понед", "mon"), ("вторн", "tue"), ("сред", "wed"), ("четверг", "thu"),
              ("пятниц", "fri"), ("суббот", "sat"), ("воскрес", "sun"))
_EN_DAYS = {"monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu",
            "friday": "fri", "saturday": "sat", "sunday": "sun"}
_RU_DAY_EACH = {"mon": "каждый понедельник", "tue": "каждый вторник", "wed": "каждую среду",
                "thu": "каждый четверг", "fri": "каждую пятницу", "sat": "каждую субботу",
                "sun": "каждое воскресенье"}
_RU_DAY_DAT = {"mon": "понедельникам", "tue": "вторникам", "wed": "средам", "thu": "четвергам",
               "fri": "пятницам", "sat": "субботам", "sun": "воскресеньям"}
_EN_DAY = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
           "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}
_DAY_CODES = list(_EN_DAY)  # Monday first: repeat_days() numbers are indexes into this


def recurrence(t: str) -> str | None:
    """Repeat rule of «по будням», «каждый понедельник», "every weekend" — or None."""
    if _has(t, r"\b(по будням|по будничным|в будни|будние дни|по рабочим дням|каждый будний день|"
               r"weekdays?|every weekday|on weekdays?)\b"):
        return "weekdays"
    if _has(t, r"\b(по выходным|в выходные|выходные|weekends?|every weekend|on weekends?)\b"):
        return "weekends"
    if _has(t, r"\b(каждый день|каждое утро|каждые сутки|ежедневно|every day|every morning|daily)\b"):
        return "daily"
    # Several days: «по вторникам и четвергам», "every tuesday and thursday".
    ru_list, en_list = _has(t, r"\bпо\b"), _has(t, r"\b(every|on|for)\b")
    days = [day for stem, day in _DAY_STEMS
            if _has(t, rf"\b(?:по|кажд\w*|every|on|for)\s+{stem}\w*") or (ru_list and _has(t, rf"\b{stem}\w*"))]
    days += [day for word, day in _EN_DAYS.items()
             if _has(t, rf"\b(?:every|on|for)\s+{word}s?\b") or (en_list and _has(t, rf"\b{word}s?\b"))]
    return normalize_repeat(",".join(days)) or None if days else None


def _join_and(parts: list[str], word: str) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" {word} " + parts[-1]


def repeat_text(rule: str | None, lang: str) -> str:
    """«каждый будний день», «по понедельникам и четвергам», "every Monday"."""
    days = sorted(repeat_days(rule))
    if not days:
        return ""
    if days == list(range(5)):
        return "каждый будний день" if lang == "ru" else "every weekday"
    if days == [5, 6]:
        return "по выходным" if lang == "ru" else "on weekends"
    if len(days) == 7:
        return "каждый день" if lang == "ru" else "every day"
    if lang == "ru":
        if len(days) == 1:
            return _RU_DAY_EACH[_DAY_CODES[days[0]]]
        return "по " + _join_and([_RU_DAY_DAT[_DAY_CODES[d]] for d in days], "и")
    return "every " + _join_and([_EN_DAY[_DAY_CODES[d]] for d in days], "and")


def _ru_date(at: dt.datetime) -> str:
    day = speech.num2words(at.day, lang="ru", to="ordinal")
    day = day[:-2] + ("ье" if day.endswith("ий") else "ое")  # "тридцать первое", "третье"
    return f"{day} {_RU_MONTHS[at.month - 1]}"


def _when(item, now: dt.datetime, lang: str) -> str:
    """One line of «какие будильники?»: «каждый будний день в семь часов», «осталось пять минут»."""
    ru = lang == "ru"
    if item.kind == "timer":
        left = speech.duration(item.remaining(), lang)
        return f"осталось {left}" if ru else f"{left} left"
    at = dt.datetime.fromtimestamp(item.due)
    clock = speech.clock(at.hour, at.minute, lang)
    rule = repeat_text(item.repeat, lang)
    if rule:
        when = f"{rule} в {clock}" if ru else f"{rule} at {clock}"
    else:
        days = (at.date() - now.date()).days
        if days == 0:
            when = f"сегодня в {clock}" if ru else f"today at {clock}"
        elif days == 1:
            when = f"завтра в {clock}" if ru else f"tomorrow at {clock}"
        elif ru:
            when = f"{_ru_date(at)} в {clock}"
        else:
            when = f"on {at:%B} {at.day} at {clock}"
    if item.label:
        return f"напоминание «{item.label}», {when}" if ru else f"reminder {item.label}, {when}"
    return when


def _list_timers(brain: "Brain", kind: str | None, lang: str, ru: bool) -> str:
    now = dt.datetime.now()
    alarms = brain.scheduler.pending("alarm") if kind != "timer" else []
    timers = brain.scheduler.pending("timer") if kind != "alarm" else []
    parts = []
    if alarms:
        parts.append(("Будильники: " if ru else "Alarms: ") + "; ".join(_when(i, now, lang) for i in alarms))
    if timers:
        parts.append(("Таймеры: " if ru else "Timers: ") + "; ".join(_when(i, now, lang) for i in timers))
    if parts:
        return ". ".join(parts) + "."
    if ru:
        return "Будильников нет." if kind == "alarm" else "Таймеров нет." if kind == "timer" else \
            "Ничего не заведено."
    return f"No {kind}s." if kind else "Nothing is set."


def _matching_repeat(items, rule: str) -> list:
    """«отмени будильник по будням»: the same rule, or (none matching) any alarm covering its days."""
    same = [i for i in items if i.repeat == rule]
    if same:
        return same
    days = repeat_days(rule)
    return [i for i in items if i.repeat and repeat_days(i.repeat) & days]


def reminder_label(t: str) -> str:
    """«каждый понедельник в 9 напоминай про спортзал» -> "спортзал"; "" when there is none."""
    m = re.search(r"\b(?:напом(?:н|ин)\w*|remind(?:\s+me)?)\b.*?\b(?:про|о|об|about)\s+(.+)$", t)
    if not m:
        return ""
    label = re.sub(r"\b(?:в|на|at)\s+\d.*$", "", m.group(1)).strip(" ,.")
    return re.sub(r"^(кажд\w*|every)\b.*", "", label).strip(" ,.")


def _night_schedule(brain: "Brain", ru: bool) -> str:
    a = brain.cfg["audio"]
    start, end = str(a.get("night_start") or ""), str(a.get("night_end") or "")
    volume = int(a.get("night_volume") or 0)
    if not start or not end or start == end:
        return "Ночной режим выключен." if ru else "Night mode is off."
    if ru:
        return f"Ночной режим с {start} до {end}, громкость в это время — {speech.ru_number(volume, True)}."
    return f"Night mode from {start} to {end}, volume at most {volume}."


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

    # Repeat the last answer, and «что ты умеешь?» -------------------------
    reply = await help.handle(brain, t, words, lang, ru)
    if reply is not None:
        return reply

    # Speaker -> chat: «напиши Мише в телеграм, что я скоро буду» --------------
    reply = await brain.chat.command(text, lang)
    if reply is not None:
        return reply

    # Voice profiles -----------------------------------------------------
    # Who is talking comes from the voice service (`brain.speaker` during a turn).
    speaker = brain.speaker
    if _has(t, r"\b(запомни|запиши|сохрани)\b.*\bголос\w*\b|\bremember my voice\b"):
        name = parse_name(text)

        async def begin(who: str) -> str:
            await brain.start_voice_enroll(who, lang)
            return f"Хорошо, {who}. Скажи три фразы, которые я попрошу." if ru else \
                f"Okay {who}. Say the three phrases I'll ask for."

        async def ask_name(answer: str | None, retry: bool = True) -> str | None:
            who = parse_name(answer or "")
            if who:
                return await begin(who)
            if retry:  # one more try: the next words are the name again
                brain.pending = brain.expect_reply(lang, lambda a: ask_name(a, retry=False))
                return "Не понял имя. Как тебя зовут?" if ru else "I didn't catch the name. What is it?"
            return "Ладно, запишем голос в другой раз." if ru else "Okay, we'll record your voice another time."

        if name is None:
            brain.pending = brain.expect_reply(lang, ask_name)
            return "Как тебя зовут?" if ru else "What's your name?"
        return await begin(name)
    if _has(t, r"\b(забудь|удали|сотри|убери)\b.*\bголос\w*\b|\bforget my voice\b"):
        if not speaker:
            return "Я тебя не узнаю, так что забывать нечего." if ru else "I don't recognise you, so there's nothing to forget."
        await brain.forget_voice(speaker)
        return f"Забыл твой голос, {speaker}." if ru else f"I forgot your voice, {speaker}."
    if len(words) <= 4 and _has(t, r"\b(как меня зовут|кто я|мое имя|who am i|what'?s my name)\b"):
        if speaker:
            return f"Ты {speaker}." if ru else f"You're {speaker}."
        return "Не знаю, кто вы." if ru else "I don't know who you are."
    if _has(t, r"\b(чьи голоса|какие голоса|кого ты знаешь|whose voices)\b"):
        names = list(brain.voice_names)
        if not names:
            return "Пока ничьи: я не знаю ни одного голоса." if ru else "None yet: I don't know any voice."
        return "Я знаю голоса: " + ", ".join(names) + "." if ru else "I know these voices: " + ", ".join(names) + "."

    # Extras: sleep timer, morning show, news, rates, shopping list -------
    reply = await extras.handle(brain, t, words, lang, ru)
    if reply is not None:
        return reply

    # Night mode --------------------------------------------------------
    if len(words) <= 6 and _has(t, r"\b(ночн\w* режим|night mode)\b"):
        if _has(t, r"\b(когда|во сколько|what time|when)\b"):
            return _night_schedule(brain, ru)
        if _has(t, _CANCEL):
            await brain.set_night(False)
            return "Ночной режим выключен." if ru else "Night mode off."
        if _has(t, r"\b(по расписанию|как обычно|обратно|automatic|schedule)\b"):
            await brain.set_night(None)
            return "Ночной режим по расписанию." if ru else "Night mode on its schedule."
        await brain.set_night(True)
        return "Ночной режим включён." if ru else "Night mode on."

    # Timers and alarms -------------------------------------------------
    repeat = recurrence(t)
    if _has(t, _TIMER) and _has(t, _CANCEL):
        kind = "alarm" if _has(t, r"\b(будильник\w*|alarm\w*)\b") else "timer"
        items = brain.scheduler.pending(kind)
        if repeat:
            items = _matching_repeat(items, repeat)
            if not items:
                return "Такого будильника нет." if ru else "There's no such alarm."
        if not items:
            return ("Будильников" if kind == "alarm" else "Таймеров") + " и так нет." if ru else f"There are no {kind}s."
        brain.scheduler.cancel(items)
        if ru:
            n = len(items)
            return "Отменил." if n == 1 else \
                f"Отменил все: {speech.ru_number(n, True)} {speech.ru_plural(n, 'штука', 'штуки', 'штук')}."
        return "Cancelled." if len(items) == 1 else f"Cancelled all {speech.num2words(len(items))}."

    if len(words) <= 6 and _has(t, r"\b(какие (у меня |сейчас |там )?(будильники|таймеры)|"
                                     r"мои (будильники|таймеры)|список (будильников|таймеров)|"
                                     r"(what|which) (alarms|timers)|list (the )?(alarms|timers))\b"):
        asks_alarm = _has(t, r"\b(будильник\w*|alarm\w*)")
        asks_timer = _has(t, r"\b(таймер\w*|timer\w*)")
        return _list_timers(brain, "alarm" if asks_alarm and not asks_timer else
                            "timer" if asks_timer and not asks_alarm else None, lang, ru)

    wake_me = _has(t, r"^(разбуди\w*|буди меня|wake me)\b")
    # Labelled timers and alarms ("для пельменей", "с подписью работа") go to the LLM, which fills the label;
    # repeating ones («каждый понедельник в 9 напоминай про спортзал») are made here.
    labelled = not repeat and _has(
        t, r"\b(для|чтобы|напомн\w*|подпис\w*|назов\w*|называ\w*|called|named|labell?ed|remind\w*|for the)\b")
    if (_has(t, r"\b(таймер\w*|timer)\b") or wake_me or _has(t, r"^засеки\b")) and not labelled:
        seconds = parse_duration(t)
        if seconds:
            brain.scheduler.add("timer", time.time() + seconds, lang, "", seconds, who=speaker)
            if ru:
                return f"Засёк {speech.duration(seconds, 'ru', accusative=True)}."
            return f"Timer set for {speech.duration(seconds, 'en')}."

    reminder = bool(repeat) and _has(t, r"\b(напом(?:н|ин)\w*|remind\w*)\b")
    if (_has(t, r"\b(будильник\w*|alarm)\b") or wake_me or reminder) and not labelled:
        at = parse_clock(t)
        if at:
            hh, mm = at
            now = dt.datetime.now()
            when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if repeat:
                when = dt.datetime.fromtimestamp(next_due(when.timestamp(), repeat, now.timestamp()))
            else:
                days = 2 if _has(t, r"\b(послезавтра|day after tomorrow)\b") \
                    else 1 if _has(t, r"\b(завтра|tomorrow)\b") else 0
                if days:
                    when += dt.timedelta(days=days)
                elif when <= now:
                    when += dt.timedelta(days=1)
            music = alarm_music(t)
            label = reminder_label(t) if reminder else ""
            brain.scheduler.add("alarm", when.timestamp(), lang, label, music=music, repeat=repeat, who=speaker)
            tune = (f" Под «{music}»." if music else " С музыкой.") if music is not None else ""
            if repeat:
                rule = repeat_text(repeat, lang)
                clock = speech.clock(hh, mm, lang)
                if label:
                    if ru:
                        return f"Буду напоминать {rule} в {clock}: {label}."
                    return f"I'll remind you {rule} at {clock}: {label}."
                if ru:
                    return f"Будильник {rule} в {clock}." + tune
                return f"Alarm {rule} at {clock}." + tune
            offset = (when.date() - now.date()).days
            day = (["сегодня", "завтра", "послезавтра"] if ru else ["today", "tomorrow", "the day after tomorrow"])[offset]
            if music is not None:
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
