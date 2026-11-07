"""Who may use the chat bots besides the ids in config.toml: registered users, access requests, pairing codes.

Users and requests live in one JSON file (written atomically, owner-only) and take effect at once, with no
restart. Pairing codes live in memory only: a random six-digit number the speaker says aloud, valid for a
few minutes and good for one registration, so whoever sends it has been standing next to the speaker.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("brain.chat.users")

ROLES = ("admin", "user")
MAX_CODES = 10  # codes alive at once; asking for more drops the oldest
MAX_REQUESTS = 50  # open requests per platform: a flood of strangers can't grow the file without bound
CODE_TRIES = 5  # wrong codes a stranger may send per code lifetime before being ignored


@dataclass
class User:
    name: str
    role: str  # "admin" or "user"
    added_at: float
    added_by: str  # "code", "Миша", "voice"


@dataclass
class Request:
    name: str  # what the stranger is called on the platform
    at: float
    status: str  # "pending" (admins were asked) or "denied" (ignored until it expires)


@dataclass
class _Code:
    platform: str | None  # None: any platform
    name: str | None  # the recognised voice that asked for it: the new user is that person
    expires: float


class Registry:
    def __init__(self, path: Path, code_seconds: float = 600, request_seconds: float = 86400,
                 clock: Callable[[], float] = time.time):
        self.path, self.code_seconds, self.request_seconds, self.clock = path, code_seconds, request_seconds, clock
        self._users: dict[str, dict[int, User]] = {}
        self._requests: dict[str, dict[int, Request]] = {}
        self._codes: dict[str, _Code] = {}
        self._tries: dict[tuple[str, int], list[float]] = {}  # wrong codes per stranger
        self._load()

    # Persistence ---------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            d = json.loads(self.path.read_text())
            self._users = {p: {int(i): User(**u) for i, u in us.items()} for p, us in d.get("users", {}).items()}
            self._requests = {p: {int(i): Request(**r) for i, r in rs.items()}
                              for p, rs in d.get("requests", {}).items()}
        except (OSError, ValueError, TypeError) as e:
            # Nobody registered is the safe reading of a broken file; keep it for the owner to look at.
            bad = self.path.with_suffix(".bad")
            log.error("%s is unreadable (%s); starting with nobody registered, the file is kept as %s",
                      self.path, e, bad)
            try:
                self.path.replace(bad)
            except OSError:
                pass
            self._users, self._requests = {}, {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"users": {p: {str(i): asdict(u) for i, u in us.items()} for p, us in self._users.items() if us},
                "requests": {p: {str(i): asdict(r) for i, r in rs.items()} for p, rs in self._requests.items() if rs}}
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # Users ---------------------------------------------------------------
    def users(self, platform: str) -> dict[int, User]:
        return dict(self._users.get(platform, {}))

    def get(self, platform: str, uid: int) -> User | None:
        return self._users.get(platform, {}).get(uid)

    def add(self, platform: str, uid: int, name: str, role: str, by: str) -> User:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, not {role!r}")
        user = User(name, role, self.clock(), by)
        self._users.setdefault(platform, {})[uid] = user
        self._requests.get(platform, {}).pop(uid, None)
        self._save()
        return user

    def remove(self, platform: str, uid: int) -> User | None:
        user = self._users.get(platform, {}).pop(uid, None)
        if user is not None:
            self._save()
        return user

    def update(self, platform: str, uid: int, *, name: str | None = None, role: str | None = None) -> None:
        user = self._users[platform][uid]
        if role is not None and role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, not {role!r}")
        user.name = name if name is not None else user.name
        user.role = role if role is not None else user.role
        self._save()

    # Access requests -----------------------------------------------------
    def _expire_requests(self) -> None:
        cutoff = self.clock() - self.request_seconds
        changed = False
        for rs in self._requests.values():
            for uid in [i for i, r in rs.items() if r.at < cutoff]:
                del rs[uid]
                changed = True
        if changed:
            self._save()

    def request(self, platform: str, uid: int) -> Request | None:
        """The live request of `uid`; one older than a day is forgotten."""
        self._expire_requests()
        return self._requests.get(platform, {}).get(uid)

    def pending(self, platform: str) -> dict[int, Request]:
        self._expire_requests()
        return {i: r for i, r in self._requests.get(platform, {}).items() if r.status == "pending"}

    def open_request(self, platform: str, uid: int, name: str) -> bool:
        """False when too many requests are open already (the stranger is then just ignored)."""
        self._expire_requests()
        rs = self._requests.setdefault(platform, {})
        if len(rs) >= MAX_REQUESTS:
            return False
        rs[uid] = Request(name, self.clock(), "pending")
        self._save()
        return True

    def deny(self, platform: str, uid: int) -> Request | None:
        r = self._requests.get(platform, {}).get(uid)
        if r:
            r.status, r.at = "denied", self.clock()  # ignored for another full request period
            self._save()
        return r

    # Pairing codes -------------------------------------------------------
    def issue_code(self, platform: str | None, name: str | None = None) -> str:
        now = self.clock()
        self._codes = {c: v for c, v in self._codes.items() if v.expires > now}
        while len(self._codes) >= MAX_CODES:
            del self._codes[next(iter(self._codes))]
        code = f"{secrets.randbelow(10 ** 6):06d}"
        while code in self._codes:
            code = f"{secrets.randbelow(10 ** 6):06d}"
        self._codes[code] = _Code(platform, name, now + self.code_seconds)
        return code

    def may_try(self, platform: str, uid: int) -> bool:
        """Whether a stranger still may send a code: CODE_TRIES wrong ones within a code lifetime, then no."""
        cutoff = self.clock() - self.code_seconds
        tries = [t for t in self._tries.get((platform, uid), []) if t > cutoff]
        self._tries[(platform, uid)] = tries
        return len(tries) < CODE_TRIES

    def redeem(self, platform: str, uid: int, code: str) -> _Code | None:
        """The live code for `platform`, used up; None for a wrong, used or expired one (and counted against
        `uid`). Every live code is compared in constant time, so timing doesn't tell how close a guess was."""
        now = self.clock()
        self._codes = {c: v for c, v in self._codes.items() if v.expires > now}
        found = None
        for known, c in self._codes.items():
            if hmac.compare_digest(known.encode(), code.encode()) and c.platform in (None, platform):
                found = known
        if found is None:
            if len(self._tries) > 1000:  # strangers forgotten wholesale rather than growing forever
                self._tries.clear()
            self._tries.setdefault((platform, uid), []).append(now)
            return None
        self._tries.pop((platform, uid), None)
        return self._codes.pop(found)
