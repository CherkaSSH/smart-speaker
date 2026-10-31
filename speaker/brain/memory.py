"""Persistent personal memory with EmbeddingGemma 2 semantic retrieval.

Embeddings come from a llama-server running the EmbeddingGemma 2 GGUF
(`speaker embed`), so the brain needs no PyTorch. Text and normalized
768-dimensional vectors are stored in SQLite; search is a dot product over
all rows, which is instant at personal-memory scale.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import sqlite3
import time
from pathlib import Path

import aiohttp
import numpy as np

log = logging.getLogger("brain.memory")

# EmbeddingGemma 2 task prefixes (model card: asymmetric search).
DOCUMENT = "title: none | text: "
QUERY = "task: search result | query: "


class Memory:
    def __init__(self, cfg: dict, path: Path):
        self.url = cfg["url"].rstrip("/") + "/embeddings"
        self.path = path
        self.results = int(cfg.get("search_results", 5))
        self.recall_threshold = float(cfg.get("recall_threshold", 0.72))
        self.recall_results = int(cfg.get("recall_results", 3))
        self._session: aiohttp.ClientSession | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            )""")

    async def _embed(self, text: str) -> np.ndarray:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        async with self._session.post(self.url, json={"input": text}) as r:
            if r.status != 200:
                raise RuntimeError(f"embedding server {r.status}: {await r.text()}")
            vector = np.asarray((await r.json())["data"][0]["embedding"], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    async def save(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("memory text is empty")
        embedding = await self._embed(DOCUMENT + text)
        with sqlite3.connect(self.path) as db:
            cursor = db.execute("INSERT INTO memories (text, embedding, created_at) VALUES (?, ?, ?)",
                                (text, embedding.tobytes(), time.time()))
        return {"id": cursor.lastrowid, "text": text}

    async def search(self, query: str, limit: int | None = None) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("memory query is empty")
        limit = max(1, min(10, int(limit or self.results)))
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, text, embedding, created_at FROM memories").fetchall()
        if not rows:
            return []  # nothing to rank: skip the embedding request
        query_embedding = await self._embed(QUERY + query)
        vectors = np.stack([np.frombuffer(row[2], dtype=np.float32) for row in rows])
        scores = vectors @ query_embedding
        order = np.argsort(-scores)[:limit]
        return [{"id": rows[i][0], "text": rows[i][1], "score": round(float(scores[i]), 4),
                 "saved_at": dt.datetime.fromtimestamp(rows[i][3]).isoformat(timespec="seconds")}
                for i in order]

    async def recall(self, text: str) -> list[str]:
        """Memories relevant enough to hand the model without it asking."""
        if self.recall_threshold <= 0:
            return []
        try:
            hits = await self.search(text, self.recall_results)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            log.warning("memory recall skipped: %s", e)  # answering still works without it
            return []
        return [h["text"] for h in hits if h["score"] >= self.recall_threshold]

    async def close(self) -> None:
        if self._session:
            await self._session.close()
