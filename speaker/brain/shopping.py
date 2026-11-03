"""The shopping list: a JSON file the user edits by voice."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^\w]+", text.casefold().replace("ё", "е")) if len(w) > 1]


def _same(a: str, b: str) -> bool:
    """Words match on their first four letters: "молоко" finds "молока", "хлеб" finds "хлеба"."""
    n = min(len(a), len(b), 4)
    return a[:n] == b[:n]


class Shopping:
    def __init__(self, path: Path):
        self.path = path

    def items(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        if isinstance(data, dict):
            data = data.get("items", [])
        return [str(x) for x in data if str(x).strip()] if isinstance(data, list) else []

    def _save(self, items: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1))
        tmp.replace(self.path)

    def add(self, new: list[str]) -> list[str]:
        """Add names, skipping the ones already there. Returns what was added."""
        items = self.items()
        added = []
        for name in new:
            name = name.strip()
            if name and name.casefold() not in (i.casefold() for i in items):
                items.append(name)
                added.append(name)
        if added:
            self._save(items)
        return added

    def remove(self, query: str) -> list[str]:
        """Remove every item whose words match the query's. Returns what was removed."""
        words = _words(query)
        if not words:
            return []
        items = self.items()
        gone = [i for i in items if words and all(
            any(_same(w, v) for v in _words(i)) for w in words)]
        if not gone and len(words) == 1:  # one word, any item holding it
            gone = [i for i in items if any(_same(words[0], v) for v in _words(i))]
        if gone:
            self._save([i for i in items if i not in gone])
        return gone

    def clear(self) -> int:
        count = len(self.items())
        self._save([])
        return count
