"""«Повтори» and «что ты умеешь?»: the last answer, and a short spoken overview.

Both are fast commands: they run before the LLM and before the music commands,
so music's own «повтори этот трек» keeps its meaning (the regexes here match
only a bare request to repeat the answer).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import Brain

_REPEAT = re.compile(
    r"^(?:(?:ну|а|эй|слушай|пожалуйста)\s+)*"
    r"(?:"
    r"повтори(?:-ка)?(?:те)?(?:\s+(?:это|то|еще|раз|последн\w*|ответ\w*|пожалуйста|сказанное))*"
    r"|повторить(?:\s+(?:это|еще\s+раз))?"
    r"|что\s+(?:ты\s+)?(?:мне\s+)?(?:сказал[аи]?|говорил[аи]?)(?:\s+(?:еще|раз|последн\w*|только\s+что))*"
    r"|(?:я\s+)?(?:тебя\s+)?не\s+(?:расслышал[аи]?|услышал[аи]?|слышал[аи]?)(?:\s+тебя)?"
    r"|прости\s+(?:что|не\s+расслышал[аи]?)"
    r"|(?:can|could)\s+you\s+(?:repeat|say)\s+(?:that|it)(?:\s+again)?"
    r"|repeat(?:\s+(?:that|it))?(?:\s+again)?"
    r"|say\s+(?:that|it)\s+again"
    r"|what\s+did\s+you\s+say"
    r"|come\s+again"
    r"|i\s+didn\s*t\s+(?:hear|catch)(?:\s+(?:that|you))?"
    r"|pardon"
    r")$"
)

_HELP = re.compile(
    r"^(?:(?:а|ну|эй)\s+)*"
    r"(?:"
    r"что\s+(?:ты\s+)?умеешь(?:\s+делать)?"
    r"|что\s+(?:ты\s+)?можешь(?:\s+делать)?"
    r"|(?:твои\s+)?(?:возможности|умения)"
    r"|помощь"
    r"|какие\s+(?:у\s+тебя\s+)?(?:есть\s+)?(?:команды|умения|возможности)"
    r"|список\s+команд"
    r"|расскажи(?:\s+(?:мне\s+)?(?:что\s+ты\s+умеешь|о\s+своих\s+(?:умениях|возможностях)))"
    r"|what\s+can\s+you\s+do(?:\s+for\s+me)?"
    r"|what\s+are\s+your\s+(?:commands|skills)"
    r"|your\s+(?:commands|skills)"
    r"|help(?:\s+me)?"
    r")$"
)

HELP = {
    "ru": "Вот что я умею. Музыка и книги: «включи Кино», «включи мою волну», «продолжи книгу». "
          "Таймеры, будильники и напоминания: «таймер на пять минут», «разбуди в семь», "
          "«каждый будний день в семь», «напомни через час позвонить маме». Погода, новости, курсы валют и перевод: "
          "«какая погода», «что нового», «как по-английски яблоко». Список покупок: "
          "«добавь молоко в список покупок», «что в списке покупок». Ночью отвечаю тише: "
          "«включи ночной режим». Отвечу на вопросы и поищу "
          "в интернете, а сложное спрошу у большой модели: «кто написал Войну и мир», "
          "«сколько будет два плюс два». Ещё могу передать задачу агенту: «запусти агента», "
          "и повторить ответ: «повтори».",
    "en": "Here is what I can do. Music and books: «play some music», «play my wave», "
          "«continue the book». Timers, alarms and reminders: «set a timer for five minutes», "
          "«wake me at seven», «every weekday at seven», «remind me in an hour to call mom». Weather, news, "
          "exchange rates and translation: «what's the weather», «what's new», «how do you say apple in Russian». "
          "The shopping list: «add milk to the shopping list», «what's on the shopping list». "
          "At night I answer more quietly: «night mode on». I answer questions and search the web, "
          "and I ask the big model for hard ones: "
          "«who wrote War and Peace», «what is two plus two». I can also pass a task to a coding "
          "agent: «start an agent», and repeat my last answer: «repeat that».",
}


async def handle(brain: "Brain", t: str, words: list[str], lang: str, ru: bool) -> str | None:
    if _REPEAT.fullmatch(t):
        last = brain.last_answer.get(brain.speaker)
        if last:
            return last
        return "Я ещё ничего не говорила." if ru else "I haven't said anything yet."
    if _HELP.fullmatch(t):
        return HELP["ru" if ru else "en"]
    return None
