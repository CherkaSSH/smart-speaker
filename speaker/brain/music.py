"""Music: providers find tracks, the controller keeps the queue, the voice service plays.

A provider turns a request into tracks and a track into something ffmpeg can
play (a file path or a stream URL). Local files and internet radio are built in;
a streaming service is one more provider:

    class YandexMusic:
        name = "yandex"
        async def search(self, query: str, limit: int) -> list[Track]: ...   # call the service's search API
        async def resolve(self, track: Track) -> str: ...                     # fresh (often expiring) stream URL

and list it under [music] providers. Resolving right before playback keeps
expiring stream URLs valid; the voice service needs nothing else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .server import Brain

log = logging.getLogger("brain.music")

AUDIO_EXTENSIONS = {".mp3", ".ogg", ".oga", ".opus", ".flac", ".wav", ".m4a", ".aac", ".wma"}
MATCH_THRESHOLD = 0.6  # minimum fuzzy score for a search hit


@dataclass
class Track:
    provider: str
    id: str
    title: str
    artist: str = ""
    lyrics: str = ""
    live: bool = False  # a radio stream: no end, resume reconnects
    source: str = ""  # filled by the provider's resolve() right before playback
    extra: dict = field(default_factory=dict)

    def label(self, lang: str) -> str:
        """How the speaker names the track aloud: no "(Feodor Chaliapin)" asides."""
        title, artist = (re.sub(r"\s*\([^)]*\)", "", s).strip(" ,") for s in (self.title, self.artist))
        if artist:
            return f"{artist} — {title}" if lang == "en" else f"{title}, {artist}"
        return title


class Provider(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> list[Track]: ...

    async def resolve(self, track: Track) -> str: ...


# Matching ---------------------------------------------------------------------
_CYR_TO_LAT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
               "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
               "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
               "э": "e", "ю": "yu", "я": "ya"}


def _fold(text: str) -> str:
    """Rough Latin sound skeleton, so "май бонни" matches "My Bonnie" and "шайн" matches "Shine"."""
    text = "".join(_CYR_TO_LAT.get(c, c) for c in text.lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    for spelled, sound in (("ck", "k"), ("ph", "f"), ("th", "t"), ("ee", "i"), ("ea", "i"), ("oo", "u"),
                           ("ou", "u"), ("ai", "i"), ("ay", "i"), ("ey", "ei"), ("y", "i"), ("c", "k"), ("w", "v"),
                           ("q", "k"), ("x", "ks")):
        text = text.replace(spelled, sound)
    text = re.sub(r"(?<=[a-z]{3})e\b", "", text)  # silent final e: "shine" sounds like "shin"
    # Voiced and voiceless pairs blur in speech recognition: «газбулат» for «Хас-Булат».
    text = text.translate(str.maketrans("bdgvz", "ptkfs"))
    return re.sub(r"(.)\1+", r"\1", re.sub(r"\s+", " ", text)).strip()


def score(query: str, *fields: str) -> float:
    q = _fold(query)
    if not q:
        return 0.0
    best = 0.0
    for f in fields:
        t = _fold(f)
        if not t:
            continue
        if q in t:
            best = max(best, 0.9 + 0.1 * len(q) / len(t))
            continue
        ratio = SequenceMatcher(None, q, t).ratio()
        # A query naming part of a long title ("ball game") matches by its words;
        # adjacent title words joined catch compounds heard as one word ("hasbulat").
        qw, tw = set(q.split()), t.split()
        tw += [a + b for a, b in zip(tw, tw[1:])]
        word_hits = sum(1 for w in qw if any(SequenceMatcher(None, w, x).ratio() >= 0.8 for x in tw)) / len(qw)
        best = max(best, ratio, 0.85 * word_hits)
    return best


# Providers ----------------------------------------------------------------------
class LocalLibrary:
    """Audio files under a folder. Optional sidecars: NAME.json (title, artist) and NAME.txt (lyrics)."""

    name = "local"

    def __init__(self, folder: Path):
        self.folder = folder
        self._tracks: dict[str, tuple[float, Track]] = {}  # path -> (mtime, track)

    def _scan(self) -> list[Track]:
        if not self.folder.is_dir():
            return []
        found = {}
        for path in sorted(self.folder.rglob("*")):
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            # Sidecars count too: lyrics or a title added later must show up.
            mtime = max(p.stat().st_mtime for p in (path, path.with_suffix(".json"), path.with_suffix(".txt"))
                        if p.exists())
            key = str(path)
            cached = self._tracks.get(key)
            found[key] = cached if cached and cached[0] == mtime else (mtime, self._read(path))
        self._tracks = found
        return [t for _, t in found.values()]

    def _read(self, path: Path) -> Track:
        meta = {}
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
        else:  # embedded tags
            try:
                out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist",
                                      "-of", "json", str(path)], capture_output=True, text=True, timeout=10).stdout
                meta = {k.lower(): v for k, v in json.loads(out).get("format", {}).get("tags", {}).items()}
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
                pass
        lyrics = path.with_suffix(".txt")
        return Track(self.name, str(path), meta.get("title") or path.stem.replace("-", " ").replace("_", " "),
                     meta.get("artist", ""), lyrics.read_text() if lyrics.exists() else "")

    async def search(self, query: str, limit: int) -> list[Track]:
        tracks = await asyncio.to_thread(self._scan)
        if not query:
            return random.sample(tracks, len(tracks))  # "play some music": shuffle everything
        ranked = sorted(((score(query, t.title, f"{t.artist} {t.title}", t.artist), t) for t in tracks),
                        key=lambda x: -x[0])
        return [t for s, t in ranked if s >= MATCH_THRESHOLD][:limit]

    async def resolve(self, track: Track) -> str:
        return track.id

    def titles(self) -> list[str]:
        return [t.label("en") for _, t in self._tracks.values()]


class Radio:
    """Internet radio stations from config: {name = url}. Any HTTP/HLS stream works."""

    name = "radio"

    def __init__(self, stations: dict[str, str]):
        self.stations = stations

    async def search(self, query: str, limit: int) -> list[Track]:
        q = re.sub(r"\b(радио|radio|станци\w*|station)\b", " ", query.lower()).strip()
        ranked = sorted(((score(q, name) if q else 1.0, name) for name in self.stations), key=lambda x: -x[0])
        return [Track(self.name, name, name, live=True) for s, name in ranked if s >= MATCH_THRESHOLD][:limit]

    async def resolve(self, track: Track) -> str:
        return self.stations[track.id]


# Controller -----------------------------------------------------------------------
class Music:
    def __init__(self, brain: "Brain", cfg: dict, library: Path):
        self.brain = brain
        available = {"local": LocalLibrary(library), "radio": Radio(dict(cfg.get("radio", {})))}
        self.providers: list[Provider] = [available[name] for name in cfg["providers"] if name in available]
        self.queue: list[Track] = []
        self.index = 0
        self.state = "stopped"  # mirrors the voice service: stopped, playing, paused

    @property
    def current(self) -> Track | None:
        return self.queue[self.index] if self.state != "stopped" and self.index < len(self.queue) else None

    async def find(self, query: str, radio: bool = False) -> list[Track]:
        tracks: list[Track] = []
        for p in self.providers:
            if radio != (p.name == "radio"):
                continue
            try:
                tracks += await p.search(query, 20)
            except Exception:  # one broken service must not break the others
                log.exception("search failed in %s", p.name)
        return tracks

    async def play(self, query: str = "", radio: bool = False) -> Track | None:
        tracks = await self.find(query, radio)
        if not tracks:
            return None
        if query and not radio:
            # The best match first, then the rest of the library so music keeps going.
            rest = [t for t in await self.find("") if t.id != tracks[0].id]
            tracks = [tracks[0]] + rest
        self.queue, self.index = tracks, 0
        await self._send_current()
        return self.queue[0]

    async def _send_current(self) -> None:
        track = self.queue[self.index]
        provider = next(p for p in self.providers if p.name == track.provider)
        track.source = await provider.resolve(track)
        self.state = "playing"
        log.info("play %s (%d/%d)", track.label("en"), self.index + 1, len(self.queue))
        await self.brain.broadcast({"type": "music", "action": "play", "track": asdict(track)})

    async def control(self, action: str) -> bool:
        """pause, resume, next, previous, stop. False when there is nothing to act on."""
        if not self.queue or self.state == "stopped" and action != "resume":
            return False
        if action in ("next", "previous"):
            step = 1 if action == "next" else -1
            if not 0 <= self.index + step < len(self.queue):
                return False
            self.index += step
            await self._send_current()
            return True
        if action == "resume" and self.state == "stopped":
            await self._send_current()  # replay the last track
            return True
        if action == "pause" and self.state != "playing" or action == "resume" and self.state != "paused":
            return False
        self.state = {"pause": "paused", "resume": "playing", "stop": "stopped"}[action]
        await self.brain.broadcast({"type": "music", "action": action})
        return True

    async def on_event(self, event: str, track_id: str, detail: str) -> None:
        """From the voice service: a track ended or failed; move on."""
        cur = self.current
        if cur is None or cur.id != track_id:
            return
        if event == "error":
            log.warning("could not play %s: %s", cur.label("en"), detail)
        if event in ("ended", "error") and not cur.live and self.index + 1 < len(self.queue):
            self.index += 1
            await self._send_current()
        elif event in ("ended", "error"):
            self.state = "stopped"
