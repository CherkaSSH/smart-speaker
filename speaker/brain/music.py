"""Music: providers find tracks, the controller keeps the queue, the voice service plays.

A provider turns a request into tracks and a track into something ffmpeg can
play (a file path or a stream URL): the local library, internet radio, a
Navidrome (Subsonic) server and Yandex Music. Each implements

    async def search(self, query: str, limit: int) -> list[Track]: ...  # "" = any music, shuffled
    async def resolve(self, track: Track) -> str: ...                    # fresh (often expiring) stream URL

and is listed under [music] providers. Resolving right before playback keeps
expiring stream URLs valid; the voice service needs nothing else. Optional:
radio(query, limit) answers radio requests; wave(station), similar(track) and
feedback(track, event, seconds) are Yandex's endless waves.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import secrets
import subprocess
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import aiohttp
from yandex_music import ClientAsync
from yandex_music.exceptions import NetworkError, TimedOutError
from yandex_music.utils.request_async import Request

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
    album: str = ""
    lyrics: str = ""
    live: bool = False  # a radio stream: no end, resume reconnects
    source: str = ""  # filled by the provider's resolve() right before playback
    station: str = ""  # the wave it came from («Рок», «Моя волна»): more comes from there

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


# Words a request to play music contains. LFM3 sometimes calls play_music for
# «расскажи сказку» or «прочитай стихотворение»; without one of these the tool refuses.
_ASKS_FOR_MUSIC = re.compile(r"\b(включ\w*|постав\w*|сыгра\w*|игра\w*|запуст\w*|послуша\w*|музык\w*|песн\w*|песен\w*|"
                             r"трек\w*|радио|альбом\w*|play\w*|music|song\w*|radio|track\w*|listen\w*)\b",
                             re.IGNORECASE)


def asks_for_music(text: str) -> bool:
    return bool(_ASKS_FOR_MUSIC.search(text))


def rank(query: str, tracks: list[Track], limit: int) -> list[Track]:
    """Fuzzy search over a song list; an empty query ("play some music") shuffles all of it."""
    if not query:
        return random.sample(tracks, len(tracks))
    ranked = sorted(((score(query, t.title, f"{t.artist} {t.title}", t.artist, t.album), t) for t in tracks),
                    key=lambda x: -x[0])
    return [t for s, t in ranked if s >= MATCH_THRESHOLD][:limit]


# Providers ----------------------------------------------------------------------
class LocalLibrary:
    """Audio files under a folder. Optional sidecars: NAME.json (title, artist, album) and NAME.txt (lyrics)."""

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
                out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=title,artist,album",
                                      "-of", "json", str(path)], capture_output=True, text=True, timeout=10).stdout
                meta = {k.lower(): v for k, v in json.loads(out).get("format", {}).get("tags", {}).items()}
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
                pass
        lyrics = path.with_suffix(".txt")
        return Track(self.name, str(path), meta.get("title") or path.stem.replace("-", " ").replace("_", " "),
                     meta.get("artist", ""), meta.get("album", ""), lyrics.read_text() if lyrics.exists() else "")

    async def search(self, query: str, limit: int) -> list[Track]:
        return rank(query, await asyncio.to_thread(self._scan), limit)

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
        return []  # only answers radio requests

    async def radio(self, query: str, limit: int) -> list[Track]:
        q = re.sub(r"\b(радио|radio|станци\w*|station)\b", " ", query.lower()).strip()
        ranked = sorted(((_station_score(q, name, "") if q else 1.0, name) for name in self.stations),
                        key=lambda x: -x[0])
        return [Track(self.name, name, name, live=True) for s, name in ranked if s >= MATCH_THRESHOLD][:limit]

    async def resolve(self, track: Track) -> str:
        return self.stations[track.id]


class Navidrome:
    """A Navidrome (or other Subsonic) server. Its search is word-based, so «май бонни» would not find
    "My Bonnie"; instead the whole song list is fetched (500 songs per request, refreshed after
    CATALOG_TTL) and matched like the local library."""

    name = "navidrome"
    CATALOG_TTL = 600  # seconds
    PAGE = 500

    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/")
        self.user = cfg["user"]
        self.password_env = cfg["password_env"]
        self._songs: list[Track] = []
        self._loaded = float("-inf")
        self._lock = asyncio.Lock()

    def _auth(self) -> dict[str, str]:
        """Subsonic token auth: the password itself never goes over the wire."""
        password = os.environ.get(self.password_env)
        if not password:
            raise RuntimeError(f"set ${self.password_env} to the Navidrome password of {self.user!r}")
        salt = secrets.token_hex(8)
        return {"u": self.user, "t": hashlib.md5((password + salt).encode()).hexdigest(), "s": salt,
                "v": "1.16.1", "c": "smart-speaker"}

    async def _call(self, endpoint: str, **params) -> dict:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http:
            async with http.get(f"{self.url}/rest/{endpoint}", params={**self._auth(), "f": "json", **params}) as r:
                r.raise_for_status()
                body = (await r.json(content_type=None))["subsonic-response"]
        if body["status"] != "ok":
            raise RuntimeError(f"navidrome {endpoint}: {body.get('error', {}).get('message', body)}")
        return body

    async def _catalog(self) -> list[Track]:
        async with self._lock:
            if time.monotonic() - self._loaded > self.CATALOG_TTL:
                songs: list[Track] = []
                while True:  # an empty search3 query lists every song
                    page = (await self._call("search3", query="", artistCount=0, albumCount=0,
                                             songCount=self.PAGE, songOffset=len(songs)))
                    page = page.get("searchResult3", {}).get("song", [])
                    songs += [Track(self.name, s["id"], s.get("title", ""),
                                    "" if s.get("artist") == "[Unknown Artist]" else s.get("artist", ""),
                                    "" if s.get("album") == "[Unknown Album]" else s.get("album", ""))
                              for s in page]
                    if len(page) < self.PAGE:
                        break
                self._songs, self._loaded = songs, time.monotonic()
                log.info("navidrome: %d songs", len(songs))
        return self._songs

    async def search(self, query: str, limit: int) -> list[Track]:
        return rank(query, await self._catalog(), limit)

    async def resolve(self, track: Track) -> str:
        return f"{self.url}/rest/stream?" + "&".join(f"{k}={v}" for k, v in {**self._auth(), "id": track.id}.items())


# Yandex's own waves: "radio" by genre, mood, activity, decade and language. Names repeat across
# types (Техно is a genre and a mix); the first type in this order keeps the name.
STATION_TYPES = ("genre", "mood", "activity", "epoch", "local-language", "personal", "micro-genre", "editorial",
                 "tempo")
STATION_MATCH = 0.8  # stricter than songs: 600 names match almost anything loosely («groove salad» → Грув-метал)
MY_WAVE = "Моя волна"
_DECADES = {"пятидесят": "1950", "шестидесят": "1960", "семидесят": "1970", "восьмидесят": "1980",
            "девяност": "1990", "нулев": "2000", "двухтысячн": "2000"}


def _station_query(query: str) -> str:
    """«радио для тренировки» → «тренировки», «девяностые» and «90-х» → «1990-е» (Yandex's decade names)."""
    q = re.sub(r"\b(радио|radio|станци\w*|station|волн\w*|wave|музык\w*|music|песн\w*|songs?|для|под|на|for)\b",
               " ", query.lower())
    for word, year in _DECADES.items():
        q = re.sub(rf"\b{word}\w*", f"{year}-е", q)
    q = re.sub(r"\b(?:19)?([5-9]0)(?:\s*-?\s*(?:х|е|ых|ые|s)\b|\b)", r"19\1-е", q)
    return re.sub(r"\s+", " ", q).strip()


def _station_score(q: str, name: str, tag: str) -> float:
    """Fuzzy, plus whole words («рэп» → «Русский рэп», not «Рэпкор») and Russian stems (тренировки → Тренируюсь).
    A query under four letters only matches whole words: fuzzily, «ддт» is «Даб»."""
    qw, nw = _fold(q).split(), _fold(name).split()
    if not qw:
        return 0.0
    if all(w in nw for w in qw):
        return 1.0 if len(qw) == len(nw) else 0.98
    if len("".join(qw)) < 4:
        return 0.0
    s = max(score(q, name), score(q, tag.replace("-", " ")))  # the tag is English: «jazz», «workout»
    return max(s, 0.9 * sum(any(n.startswith(w[:5]) for n in nw) for w in qw) / len(qw))


class YandexMusic:
    """Yandex Music. Full tracks need the OAuth token of an account with Yandex Plus (`speaker yandex-login`);
    without one the API only serves 30-second previews and there are no waves.

    Search is Yandex's own, which copes with misheard and transliterated names: its best result decides
    what plays (a track, an artist's popular tracks, an album or a playlist). Any music is My Wave; radio
    requests play a wave by genre, mood, activity, decade or language, and a wave can start from a song.
    Waves come in batches of five; reporting what played (as the app does) keeps the next batches from
    repeating it (7-9 of 20 tracks repeated without) and teaches My Wave from skips."""

    name = "yandex"

    def __init__(self, cfg: dict):
        self.token_env = cfg["token_env"]
        self._client = None
        self._stations: dict[str, tuple[str, str, int]] = {}  # name -> (rotor id, English tag, type order)
        self._seeded: dict[str, str] = {}  # «по треку …» -> rotor id: waves started from a song, not searchable
        self._wave_last: dict[str, str] = {}  # rotor id -> last track handed out: the next batch follows it
        self._batches: dict[str, tuple[str, str]] = {}  # track id -> (rotor id, batch id), for feedback

    async def _api(self):
        if self._client is None:
            token = os.environ.get(self.token_env)
            if not token:
                log.warning("$%s is not set: Yandex Music plays 30-second previews", self.token_env)
            client = ClientAsync(token or None, request=_KeepAliveRequest())
            self._client = await self._retry(client.init)
        return self._client

    @staticmethod
    async def _retry(call):
        """Connections to the API sometimes stall in the TLS handshake and reset (1 in 4 from this
        speaker's network); one retry hides most of that."""
        try:
            return await call()
        except (NetworkError, TimedOutError) as e:
            log.info("yandex: %s; retrying", e)
            return await call()

    def _track(self, t, station: str = "") -> Track:
        return Track(self.name, str(t.track_id), t.title or "", ", ".join(t.artists_name()),
                     t.albums[0].title or "" if t.albums else "", station=station)

    async def stations(self) -> dict[str, tuple[str, str, int]]:
        if not self._stations:
            api = await self._api()
            listed = await self._retry(lambda: api.rotor_stations_list(language="ru"))
            stations = {MY_WAVE: ("user:onyourwave", "my wave", -1)}
            for order, kind in enumerate(STATION_TYPES):
                for s in listed:
                    if s.station.id.type == kind:
                        stations.setdefault(s.station.name, (f"{kind}:{s.station.id.tag}", s.station.id.tag, order))
            self._stations = stations
        return self._stations

    async def wave(self, station: str) -> list[Track]:
        """The next batch of a wave, continuing from the last track handed out."""
        api = await self._api()
        sid = self._seeded.get(station) or (await self.stations())[station][0]
        first = sid not in self._wave_last
        batch = await self._retry(lambda: api.rotor_station_tracks(sid, queue=self._wave_last.get(sid)))
        if first:
            await self._retry(lambda: api.rotor_station_feedback_radio_started(sid, "smart-speaker", batch.batch_id))
        found = [s.track for s in batch.sequence if s.track and s.track.available is not False]
        if found:
            self._wave_last[sid] = found[-1].track_id
        for t in found:
            self._batches[str(t.track_id)] = (sid, batch.batch_id)
        return [self._track(t, station) for t in found]

    async def feedback(self, track: Track, event: str, seconds: float = 0.0) -> None:
        """started, finished or skipped, for a track from a wave."""
        if track.id not in self._batches:
            return
        sid, batch = self._batches[track.id]
        api = await self._api()
        if event == "started":
            await self._retry(lambda: api.rotor_station_feedback_track_started(sid, track.id, batch))
        elif event == "finished":
            await self._retry(lambda: api.rotor_station_feedback_track_finished(sid, track.id, seconds, batch))
        elif event == "skipped":
            await self._retry(lambda: api.rotor_station_feedback_skip(sid, track.id, seconds, batch))

    async def radio(self, query: str, limit: int) -> list[Track]:
        """«радио рок», «спокойное», «для тренировки», «90-х»; «радио» or «моя волна» alone is My Wave."""
        if not (await self._api()).token:
            return []
        stations = await self.stations()
        q = _station_query(query)
        if not q or re.search(r"\bмо\w* волн|\bmy wave", query.lower()):
            return (await self.wave(MY_WAVE))[:limit]
        ranked = max((_station_score(q, n, tag), -order, -len(n), n) for n, (_, tag, order) in stations.items())
        if ranked[0] < STATION_MATCH:
            return []
        return (await self.wave(ranked[-1]))[:limit]

    async def similar(self, seed: Track) -> list[Track]:
        """A wave of songs like `seed`. A song from another service is looked up on Yandex first."""
        api = await self._api()
        if not api.token:
            return []
        if seed.provider == self.name:
            tid = seed.id.split(":")[0]
        else:
            artist = seed.artist.split(",")[0]
            for q in (f"{artist} {seed.title}", seed.title):  # the full credit list finds nothing
                res = await self._retry(lambda: api.search(q, type_="track"))
                if res.tracks and res.tracks.results:
                    break
            else:
                return []
            tid = str(res.tracks.results[0].id)
        name = f"по треку {seed.label('ru')}"
        self._seeded[name] = f"track:{tid}"
        return await self.wave(name)

    async def search(self, query: str, limit: int) -> list[Track]:
        api = await self._api()
        if not query:
            return await self.wave(MY_WAVE) if api.token else []
        if api.token and not re.search(r"\b(песн|трек|song|track)", query.lower()):  # «включи рок»: that wave
            q = _fold(_station_query(query))
            name = next((n for n in await self.stations() if q and _fold(n) == q), None)
            if name:
                return (await self.wave(name))[:limit]
        res = await self._retry(lambda: api.search(query))
        best = res.best
        if best is None or best.result is None:
            found = res.tracks.results[:1] if res.tracks else []
        elif best.type == "track":
            found = [best.result]
        elif best.type == "artist":
            found = (await self._retry(lambda: api.artists_tracks(best.result.id, page_size=limit))).tracks
        elif best.type == "album":
            album = await self._retry(lambda: api.albums_with_tracks(best.result.id))
            found = [t for disc in album.volumes or [] for t in disc]
        elif best.type == "playlist":
            found = [s.track for s in await self._retry(best.result.fetch_tracks_async) if s.track]
        else:  # a podcast, a video, a user: fall back to the best matching song
            found = res.tracks.results[:1] if res.tracks else []
        return [self._track(t) for t in found if t.available is not False][:limit]

    async def resolve(self, track: Track) -> str:
        api = await self._api()
        infos = await self._retry(lambda: api.tracks_download_info(track.id))
        best = max(infos, key=lambda i: (not i.preview, i.bitrate_in_kbps))
        return await self._retry(best.get_direct_link_async)


class _KeepAliveRequest(Request):
    """yandex-music opens a new connection per request; a shared connector keeps them alive, saving a TLS
    handshake (and its chance to stall) on each of the 3-4 requests a track takes."""

    _connector: aiohttp.TCPConnector | None = None

    def _prepare_kwargs(self, kwargs):
        if self._connector is None:
            self._connector = aiohttp.TCPConnector(keepalive_timeout=60)
        return {**super()._prepare_kwargs(kwargs), "connector": self._connector}


async def yandex_login(token_env: str) -> None:
    """OAuth device flow: enter a code on Yandex's page and get a token for $token_env."""
    token = await ClientAsync().device_auth(
        on_code=lambda c: print(f"Open {c.verification_url} and enter the code {c.user_code}", flush=True))
    print(f"export {token_env}={token.access_token}")


# Controller -----------------------------------------------------------------------
class Music:
    def __init__(self, brain: "Brain", cfg: dict, library: Path):
        self.brain = brain
        make = {"local": lambda: LocalLibrary(library), "radio": lambda: Radio(dict(cfg.get("radio", {}))),
                "navidrome": lambda: Navidrome(dict(cfg["navidrome"])), "yandex": lambda: YandexMusic(dict(cfg["yandex"]))}
        for name in cfg["providers"]:
            if name not in make:
                log.warning("unknown music provider %r; known: %s", name, ", ".join(make))
        self.providers: list[Provider] = [make[name]() for name in cfg["providers"] if name in make]
        self.queue: list[Track] = []
        self.index = 0
        self.state = "stopped"  # mirrors the voice service: stopped, playing, paused
        self._playing: tuple[Track, float] | None = None  # the started track and when, for feedback
        self._feedback_tasks: set[asyncio.Task] = set()

    @property
    def current(self) -> Track | None:
        return self.queue[self.index] if self.state != "stopped" and self.index < len(self.queue) else None

    def _provider(self, name: str) -> Provider:
        return next(p for p in self.providers if p.name == name)

    async def find(self, query: str, radio: bool = False) -> list[Track]:
        """Tracks from the first provider, in config order, that has any."""
        for p in self.providers:
            look = getattr(p, "radio", None) if radio else p.search
            if look is None:
                continue
            try:
                tracks = await look(query, 20)
            except Exception:  # one broken service must not break the others
                log.exception("search failed in %s", p.name)
                continue
            if tracks:
                return tracks
        return []

    async def _more(self, after: list[Track]) -> list[Track]:
        """More of the same wave, or any music from the same service, so music keeps going."""
        p, station = self._provider(after[0].provider), after[0].station
        try:
            more = await (p.wave(station) if station else p.search("", 20))
        except Exception:
            log.exception("search failed in %s", p.name)
            return []
        seen = {t.id for t in after}
        return [t for t in more if t.id not in seen]

    async def play(self, query: str = "", radio: bool = False) -> Track | None:
        tracks = await self.find(query, radio)
        if not tracks:
            return None
        if query and not radio and not tracks[0].station:
            tracks += await self._more(tracks)
        return await self._replace_queue(tracks)

    async def similar(self) -> Track | None:
        """A wave of songs like the one playing (Yandex Music finds them; a local song is looked up there)."""
        cur = self.current
        service = next((p for p in self.providers if hasattr(p, "similar")), None)
        if cur is None or service is None:
            return None
        try:
            tracks = await service.similar(cur)
        except Exception:
            log.exception("no wave for %s", cur.label("en"))
            return None
        return await self._replace_queue(tracks) if tracks else None

    async def _replace_queue(self, tracks: list[Track]) -> Track | None:
        self._report(None)
        self.queue, self.index = tracks, 0
        return self.current if await self._send_current() else None

    async def _send_current(self) -> bool:
        """Start queue[index], skipping tracks the service won't hand out (False: none left)."""
        while self.index < len(self.queue):
            track = self.queue[self.index]
            try:
                track.source = await self._provider(track.provider).resolve(track)
            except Exception:
                log.exception("could not get %s from %s", track.label("en"), track.provider)
                self.index += 1
                continue
            self.state = "playing"
            log.info("play %s (%d/%d)", track.label("en"), self.index + 1, len(self.queue))
            await self.brain.broadcast({"type": "music", "action": "play", "track": asdict(track)})
            self._playing = (track, time.monotonic())
            self._feedback(track, "started")
            return True
        self.state = "stopped"
        await self.brain.broadcast({"type": "music", "action": "stop"})
        return False

    def _report(self, how: str | None) -> None:
        """How the started track went: "finished", "skipped", or None (stopped, replaced, failed)."""
        if self._playing is not None:
            track, started = self._playing
            self._playing = None
            if how:
                self._feedback(track, how, time.monotonic() - started)

    def _feedback(self, track: Track, event: str, seconds: float = 0.0) -> None:
        """Tell the service in the background (a wave learns from it); playback doesn't wait."""
        send = getattr(self._provider(track.provider), "feedback", None)
        if send is not None:
            task = asyncio.create_task(send(track, event, seconds))
            self._feedback_tasks.add(task)
            task.add_done_callback(self._feedback_done)

    def _feedback_done(self, task: asyncio.Task) -> None:
        self._feedback_tasks.discard(task)
        if not task.cancelled() and task.exception():
            log.warning("music feedback failed: %s", task.exception())

    async def control(self, action: str) -> bool:
        """pause, resume, next, previous, stop. False when there is nothing to act on."""
        if not self.queue or self.state == "stopped" and action != "resume":
            return False
        if action in ("next", "previous"):
            step = 1 if action == "next" else -1
            cur = self.queue[self.index]
            if step == 1 and self.index + 1 == len(self.queue) and not cur.live:
                self.queue += await self._more([cur])  # skipping the last of a wave batch: the next batch
            if not 0 <= self.index + step < len(self.queue):
                return False
            self._report("skipped")
            self.index += step
            return await self._send_current()
        if action == "resume" and self.state == "stopped":
            return await self._send_current()  # replay the last track
        if action == "pause" and self.state != "playing" or action == "resume" and self.state != "paused":
            return False
        self.state = {"pause": "paused", "resume": "playing", "stop": "stopped"}[action]
        if action == "stop":
            self._report(None)
        await self.brain.broadcast({"type": "music", "action": action})
        return True

    async def on_event(self, event: str, track_id: str, detail: str) -> None:
        """From the voice service: a track ended or failed; move on."""
        cur = self.current
        if cur is None or cur.id != track_id:
            return
        if event == "error":
            log.warning("could not play %s: %s", cur.label("en"), detail)
        self._report("finished" if event == "ended" else None)
        if event == "ended" and not cur.live and self.index + 1 == len(self.queue):
            self.queue += await self._more([cur])  # the queue ran out: more of the same service
        if event in ("ended", "error") and not cur.live and self.index + 1 < len(self.queue):
            self.index += 1
            await self._send_current()
        elif event in ("ended", "error"):
            self.state = "stopped"
