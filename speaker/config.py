"""Shared configuration: one TOML file read by every process."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class Config:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.root = self.path.parent
        with open(self.path, "rb") as f:
            self.data = tomllib.load(f)

    def __getitem__(self, section: str) -> dict:
        return self.data[section]

    def file(self, rel: str) -> Path:
        """Resolve a path from the config relative to the config file."""
        p = Path(rel).expanduser()
        return p if p.is_absolute() else self.root / p


def load(path: str | os.PathLike | None = None) -> Config:
    return Config(Path(path or os.environ.get("SPEAKER_CONFIG") or DEFAULT_PATH))
