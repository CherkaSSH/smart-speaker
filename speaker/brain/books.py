"""Audiobooks from Yandex Books (the Bookmate API): search, chapters, and where you stopped.

The same Yandex OAuth token as Yandex Music is enough (`speaker yandex-login`);
it goes to Bookmate as the `Auth-Token` header. Search returns a mix of text
and audio books, so only `AudioBook` results play. A chapter's HLS URL is signed
and expires, so `resolve()` fetches the playlist right before the chapter
starts. The voice service reports how many seconds of the current chapter
played; `remember()` keeps chapter and position per book in a small JSON state
file, which is what «продолжи книгу» uses.

This is not a normal music provider: it is never searched for «включи музыку»,
only for explicit book requests (see `handle` below and `Music.play_book`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from yandex_book import YandexBookClientAsync
from yandex_book.exceptions import NetworkError, TimedOutError

from .music import MATCH_THRESHOLD, Track, score, yandex_token

if TYPE_CHECKING:
    from .server import Brain

log = logging.getLogger("brain.books")

API = "https://api.bookmate.yandex.net/api/v5"
RETRIES = 3  # 1 in 4 TLS handshakes to Yandex resets from this host


def split_id(track_id: str) -> tuple[str, int]:
    """Track ids are "book uuid:chapter index" (both halves survive the voice round trip)."""
    uuid, _, chapter = track_id.rpartition(":")
    return uuid, int(chapter) if chapter.isdigit() else 0


def chapter_number(track: Track) -> int:
    """1-based chapter number for replies: «глава 3»."""
    return split_id(track.id)[1] + 1


class Books:
    name = "books"

    def __init__(self, state_file: Path, token_env: str, token_file: Path):
        self.state_file = state_file
        self.token_env = token_env
        self.token_file = token_file
        self._state = self._load()

    # Position state -----------------------------------------------------
    def _load(self) -> dict:
        try:
            state = json.loads(self.state_file.read_text())
            return {"last": state.get("last") or "", "books": state.get("books") or {}}
        except (OSError, ValueError):
            return {"last": "", "books": {}}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False))
        tmp.replace(self.state_file)

    def saved(self, uuid: str) -> tuple[int, float]:
        """The chapter index and seconds into it where this book was last left."""
        book = self._state["books"].get(uuid) or {}
        return int(book.get("chapter") or 0), float(book.get("position") or 0.0)

    def remember(self, uuid: str, chapter: int, position: float = 0.0, title: str = "") -> None:
        books = self._state.setdefault("books", {})
        book = books.setdefault(uuid, {})
        book.update({"chapter": max(0, int(chapter)), "position": round(max(0.0, position), 1),
                     "updated": round(time.time())})
        if title:
            book["title"] = title
        self._state["last"] = uuid
        self._save()

    def forget_position(self, uuid: str, title: str = "") -> None:
        """The last chapter ended: asking for or continuing the book starts it over."""
        self.remember(uuid, 0, 0.0, title)

    # Yandex Books -------------------------------------------------------
    @property
    def token(self) -> str:
        return yandex_token(self.token_env, self.token_file)

    async def _retry(self, call):
        for attempt in range(RETRIES):
            try:
                return await call()
            except (NetworkError, TimedOutError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                if attempt == RETRIES - 1:
                    raise
                log.info("books: %s; retrying", e)
                await asyncio.sleep(0.5)

    async def _playlist(self, uuid: str) -> dict:
        """Chapters with fresh signed URLs: GET /audiobooks/<uuid>/playlists.json."""

        async def get() -> dict:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http:
                async with http.get(f"{API}/audiobooks/{uuid}/playlists.json",
                                    headers={"Auth-Token": self.token}) as r:
                    r.raise_for_status()
                    return await r.json(content_type=None)

        return await self._retry(get)

    @staticmethod
    def _book(item: dict) -> tuple[str, str, str]:
        """(uuid, name, authors) of an AudioBook search hit."""
        book = item.get("book") or {}
        authors = ", ".join(a.get("name", "") for a in book.get("authors") or [] if a.get("name"))
        return book.get("uuid") or "", book.get("name") or "", authors

    async def _best(self, query: str) -> tuple[str, str, str] | None:
        """The AudioBook that best matches the spoken title/author (Yandex's order is not always right)."""
        async with YandexBookClientAsync(self.token) as client:
            data = await self._retry(lambda: client.search(query))
        page = (((data or {}).get("data") or {}).get("search") or {}).get("page") or []
        found = [self._book(item) for item in page if item.get("__typename") == "AudioBook"]
        found = [b for b in found if b[0]]
        if not found:
            return None
        best = max(found, key=lambda b: score(query, b[1], f"{b[1]} {b[2]}", b[2]))
        return best if score(query, best[1], f"{best[1]} {best[2]}", best[2]) >= MATCH_THRESHOLD else None

    async def chapters(self, uuid: str, title: str = "", authors: str = "", limit: int = 10_000) -> list[Track]:
        playlist = await self._playlist(uuid)
        found = []
        for index, _ in enumerate(playlist.get("tracks") or []):
            found.append(Track(self.name, f"{uuid}:{index}", f"{title}, глава {index + 1}", authors, album=title))
        return found[:limit]

    async def search(self, query: str, limit: int = 10_000) -> list[Track]:
        """One track per chapter of the best matching audiobook; "" searches nothing (use resume)."""
        if not query:
            return []
        best = await self._best(query)
        if best is None:
            return []
        uuid, title, authors = best
        return await self.chapters(uuid, title, authors, limit)

    async def resume_tracks(self) -> list[Track] | None:
        """The chapter list of the most recent book, or None when nothing was ever played."""
        uuid = self._state.get("last") or ""
        if not uuid:
            return None
        book = self._state["books"].get(uuid) or {}
        tracks = await self.chapters(uuid, book.get("title", ""))
        return tracks or None

    async def resolve(self, track: Track) -> str:
        uuid, index = split_id(track.id)
        playlist = await self._playlist(uuid)
        chapters = playlist.get("tracks") or []
        if not 0 <= index < len(chapters):
            raise RuntimeError(f"book {uuid} has no chapter {index}")
        chapter = chapters[index]
        offline = chapter.get("offline") or {}
        url = (offline.get("max_bit_rate") or {}).get("url") or (chapter.get("url") or {}).get("m3u8") or ""
        if not url:
            raise RuntimeError(f"book {uuid} chapter {index} has no stream URL")
        return url


# Voice commands ---------------------------------------------------------------
_WHAT = re.compile(r"\b(что (сейчас |это )?(играет|за книга|за аудиокнига|звучит)|какая (это |сейчас )?книга|"
                   r"what('s| is| was) (playing|this book))\b", re.IGNORECASE)
_BOOK_WORD = re.compile(r"\b(аудиокниг\w*|книг\w*|audiobook\w*|books?)\b", re.IGNORECASE)
_START = re.compile(r"^(включи|поставь|запусти|почитай|прочитай|читай|play|read|start)\b(.*)$", re.IGNORECASE)
_RESUME = re.compile(r"\b(продолж\w*|вернись к|continue|resume)\b", re.IGNORECASE)
_FILLER = re.compile(r"^(?:\s*(?:мне|пожалуйста|the|a|an)\b)+|(?:\b(?:мне|пожалуйста|please)\s*)$", re.IGNORECASE)


def _name(track: Track) -> tuple[str, int]:
    """Book title and 1-based chapter of a chapter track."""
    return track.album or track.title.rsplit(", глава", 1)[0], chapter_number(track)


async def handle(brain: "Brain", text: str, words: list[str], ru: bool) -> str | None:
    """Book commands. None = not a book request, "" is not used."""
    music = brain.music
    lang = "ru" if ru else "en"
    t = text
    current = music.current

    if music.books is None:
        if _BOOK_WORD.search(t) and (_START.match(t) or _RESUME.search(t)):
            return "Аудиокниги не настроены." if ru else "Audiobooks are not configured."
        return None

    if _WHAT.search(t) and ("играет" in t or "playing" in t or _BOOK_WORD.search(t)):
        if current is None or current.provider != Books.name:
            return ("Сейчас книгу не читаю." if ru else "I'm not reading a book now.") \
                if _BOOK_WORD.search(t) else None
        title, chapter = _name(current)
        return f"Это аудиокнига «{title}», глава {chapter}." if ru else \
            f"This is the audiobook \"{title}\", chapter {chapter}."

    resume = bool(_RESUME.search(t)) and len(words) <= 6 and not _START.match(t)
    if resume and _BOOK_WORD.search(t):
        if not brain.clients:
            return "Голосовой модуль не подключён." if ru else "The voice service is not connected."
        track = await music.play_book("", lang, resume=True)
        if track is None:
            return "Не помню, где мы остановились." if ru else "I don't remember where we stopped."
        title, chapter = _name(track)
        return f"Продолжаю аудиокнигу «{title}», глава {chapter}." if ru else \
            f"Continuing the audiobook \"{title}\", chapter {chapter}."

    m = _START.match(t)
    if not m:
        return None
    rest = m.group(2)
    book = _BOOK_WORD.search(rest)
    if not book:
        return None
    query = _FILLER.sub(" ", rest[book.end():]).strip(" ,.!?«»\"'—-")
    if not brain.clients:
        return "Голосовой модуль не подключён." if ru else "The voice service is not connected."
    if not query:
        track = await music.play_book("", lang, resume=True)
        if track is None:
            return "Какую книгу включить?" if ru else "Which book should I play?"
    else:
        track = await music.play_book(query, lang)
    if track is None:
        return f"Не нашёл аудиокнигу «{query}»." if ru else f"I couldn't find the audiobook \"{query}\"."
    title, chapter = _name(track)
    return f"Включаю аудиокнигу «{title}», глава {chapter}." if ru else \
        f"Playing the audiobook \"{title}\", chapter {chapter}."
