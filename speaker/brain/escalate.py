"""Optional escalation to a remote LLM."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator

import aiohttp

from .text import SentenceSplitter

log = logging.getLogger("brain.escalate")


class RemoteLLM:
    """Stream an OpenAI-compatible chat response without involving the small local model."""

    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/") + "/chat/completions"
        self.model = cfg["model"]
        self.key_env = cfg.get("api_key_env", "")
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.temperature = float(cfg.get("temperature", 0.3))
        self.timeout = float(cfg.get("timeout", 180))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.model and (not self.key_env or os.environ.get(self.key_env)))

    async def respond(self, messages: list[dict]) -> AsyncIterator[str]:
        if not self.configured:
            raise RuntimeError(f"remote LLM is not configured; set {self.key_env}")
        headers = {"Authorization": f"Bearer {os.environ[self.key_env]}"} if self.key_env else {}
        body = {"model": self.model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        splitter = SentenceSplitter()
        usage = {}
        t0 = time.monotonic()
        first = None
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=body, headers=headers) as response:
                if response.status != 200:
                    raise RuntimeError(f"remote LLM {response.status}: {await response.text()}")
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    usage = chunk.get("usage") or usage
                    choice = (chunk.get("choices") or [{}])[0]
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        first = first if first is not None else time.monotonic() - t0
                        for sentence in splitter.feed(content):
                            yield sentence
        for sentence in splitter.flush():
            yield sentence
        log.info("remote round: %.2fs, first content %s, usage %s", time.monotonic() - t0,
                 f"{first:.2f}s" if first is not None else "-", usage)
