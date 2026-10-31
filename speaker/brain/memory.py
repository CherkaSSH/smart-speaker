"""Persistent personal memory with local EmbeddingGemma 2 semantic retrieval."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("brain.memory")
logging.getLogger("httpx2").setLevel(logging.WARNING)  # one INFO line per Hub request otherwise


class Memory:
    def __init__(self, cfg: dict, path: Path):
        self.model_name = cfg["model"]
        self.path = path
        self.results = int(cfg.get("search_results", 5))
        self.recall_threshold = float(cfg.get("recall_threshold", 0.72))
        self.recall_results = int(cfg.get("recall_results", 3))
        self._model = None
        self._lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            )""")

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            t0 = time.monotonic()
            self._model = SentenceTransformer(
                self.model_name, device="cpu",
                config_kwargs={"vision_config": None, "audio_config": None},
                model_kwargs={"torch_dtype": torch.float32})
            log.info("loaded EmbeddingGemma 2 text encoder in %.1fs", time.monotonic() - t0)
        return self._model

    def _encode(self, text: str, prompt_name: str) -> np.ndarray:
        return self._load().encode(text, prompt_name=prompt_name, normalize_embeddings=True).astype(np.float32)

    def _save(self, text: str) -> dict:
        embedding = self._encode(text, "Document")
        now = time.time()
        with sqlite3.connect(self.path) as db:
            cursor = db.execute("INSERT INTO memories (text, embedding, created_at) VALUES (?, ?, ?)",
                                (text, embedding.tobytes(), now))
        return {"id": cursor.lastrowid, "text": text}

    def _search(self, query: str, limit: int) -> list[dict]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, text, embedding, created_at FROM memories").fetchall()
        if not rows:
            return []
        query_embedding = self._encode(query, "SearchQuery")
        vectors = np.stack([np.frombuffer(row[2], dtype=np.float32) for row in rows])
        scores = vectors @ query_embedding  # both sides are L2-normalized by the model
        ranked = sorted(zip(rows, scores), key=lambda item: float(item[1]), reverse=True)[:limit]
        return [{"id": row[0], "text": row[1], "score": round(float(score), 4),
                 "saved_at": dt.datetime.fromtimestamp(row[3]).isoformat(timespec="seconds")}
                for row, score in ranked]

    async def warm(self) -> None:
        """Load the encoder in the background so the first memory request is fast."""
        async with self._lock:
            await asyncio.to_thread(self._load)


    async def save(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("memory text is empty")
        async with self._lock:
            return await asyncio.to_thread(self._save, text)

    async def search(self, query: str, limit: int | None = None) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("memory query is empty")
        limit = max(1, min(10, int(limit or self.results)))
        async with self._lock:
            return await asyncio.to_thread(self._search, query, limit)

    async def recall(self, text: str) -> list[str]:
        """Memories relevant enough to hand the model without it asking."""
        if self.recall_threshold <= 0:
            return []
        hits = await self.search(text, self.recall_results)
        return [h["text"] for h in hits if h["score"] >= self.recall_threshold]
