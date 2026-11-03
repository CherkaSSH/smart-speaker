"""News headlines from RSS feeds, cached for a few minutes."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
import xml.etree.ElementTree as ET

import aiohttp

log = logging.getLogger("brain.news")

_TAG = re.compile(r"<[^>]+>")
_ATOM = "{http://www.w3.org/2005/Atom}"


def _plain(text: str) -> str:
    """A headline as one spoken sentence: no HTML, no entities, no extra spaces."""
    text = html.unescape(_TAG.sub(" ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


class News:
    def __init__(self, cfg: dict):
        self.feeds = {"ru": list(cfg.get("feeds_ru", [])), "en": list(cfg.get("feeds_en", []))}
        self.limit = int(cfg.get("items", 5))
        self.cache_seconds = float(cfg.get("cache_minutes", 10)) * 60
        self._cache: dict[str, tuple[float, list[str]]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.feeds["ru"] or self.feeds["en"])

    async def headlines(self, lang: str, limit: int | None = None) -> list[str]:
        """The latest headlines, a few from each feed in turn; [] when nothing could be read."""
        lang = lang if self.feeds.get(lang) else "en"
        limit = limit or self.limit
        cached = self._cache.get(lang)
        if cached and time.monotonic() - cached[0] < self.cache_seconds:
            return cached[1][:limit]
        found = await asyncio.gather(*(self._titles(url) for url in self.feeds[lang]), return_exceptions=True)
        lists, seen = [], set()
        for result in found:
            if isinstance(result, BaseException):
                log.warning("news feed failed: %s", result)
                continue
            lists.append(result)
        titles: list[str] = []
        for i in range(max((len(x) for x in lists), default=0)):
            for feed in lists:
                if i < len(feed) and feed[i].casefold() not in seen:
                    seen.add(feed[i].casefold())
                    titles.append(feed[i])
                    if len(titles) == limit:
                        self._cache[lang] = (time.monotonic(), titles)
                        return titles
        if titles:
            self._cache[lang] = (time.monotonic(), titles)
            return titles
        if cached:  # every feed failed: old news beats no news
            return cached[1][:limit]
        return []

    async def _titles(self, url: str) -> list[str]:
        timeout = aiohttp.ClientTimeout(total=8)
        data = b""
        for attempt in range(2):  # a TLS handshake to some hosts occasionally resets
            try:
                async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "smart-speaker/0.1"}) as s:
                    async with s.get(url) as r:
                        r.raise_for_status()
                        data = await r.read()
                break
            except (aiohttp.ClientError, TimeoutError) as e:
                if attempt == 1:
                    raise
                log.debug("feed %s failed (%s), retrying", url, e)
        root = ET.fromstring(data)
        items = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
        titles = []
        for item in items:
            title = item.findtext("title") or item.findtext(f"{_ATOM}title") or ""
            title = _plain(title)
            if title:
                titles.append(title)
        return titles
