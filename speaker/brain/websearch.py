"""Web search through Parallel (parallel.ai), for the web_search tool.

With an API key the REST Search API runs in the configured mode (turbo, ~200 ms,
needs English queries; the tool asks LFM3 for English). Without a key the free
hosted MCP server is used (fast mode, lower rate limits). Excerpts are trimmed
so the results stay a short prompt for the speaking model.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time

import aiohttp

log = logging.getLogger("brain.web")

API_URL = "https://api.parallel.ai/v1/search"
MCP_URL = "https://mcp.parallel.ai/mcp?tools=web_search"
MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


class WebSearch:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg["enabled"])
        self.key_env = cfg.get("api_key_env", "")
        self.mode = cfg.get("mode", "turbo")
        self.results = int(cfg.get("results", 5))
        self.excerpt_chars = int(cfg.get("excerpt_chars", 600))
        self.max_per_day = int(cfg.get("max_per_day", 200))
        self.timeout = aiohttp.ClientTimeout(total=float(cfg.get("timeout", 15)))
        self._day, self._count = dt.date.today(), 0
        self._mcp_session: str | None = None

    @property
    def key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""

    async def search(self, objective: str, queries: list[str]) -> list[dict]:
        today = dt.date.today()
        if today != self._day:
            self._day, self._count = today, 0
        if self._count >= self.max_per_day:  # a looping model must not run up a bill
            raise RuntimeError(f"daily web search limit ({self.max_per_day}) reached")
        self._count += 1
        queries = [q for q in queries if q.strip()][:3] or [objective]
        t0 = time.monotonic()
        async with aiohttp.ClientSession(timeout=self.timeout) as http:
            raw = await (self._api(http, objective, queries) if self.key else self._mcp(http, objective, queries))
        out = []
        for r in raw[: self.results]:
            text = " ".join(" ".join(r.get("excerpts") or []).split())
            out.append({"title": r.get("title") or "", "date": r.get("publish_date") or "",
                        "text": text[: self.excerpt_chars]})
        log.info("web search %r %s: %d results in %.2fs (%s)", objective, queries, len(out),
                 time.monotonic() - t0, "api " + self.mode if self.key else "keyless mcp")
        return out

    async def _api(self, http: aiohttp.ClientSession, objective: str, queries: list[str]) -> list[dict]:
        body = {"objective": objective, "search_queries": queries, "mode": self.mode}
        async with http.post(API_URL, json=body, headers={"x-api-key": self.key}) as r:
            if r.status != 200:
                raise RuntimeError(f"search API {r.status}: {(await r.text())[:300]}")
            return (await r.json())["results"]

    async def _mcp(self, http: aiohttp.ClientSession, objective: str, queries: list[str]) -> list[dict]:
        call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "web_search", "arguments": {"objective": objective, "search_queries": queries}}}
        for attempt in range(2):
            if self._mcp_session is None:
                await self._mcp_open(http)
            try:
                result = (await self._rpc(http, call))["result"]
                break
            except (RuntimeError, KeyError):
                self._mcp_session = None  # expired session: open a new one once
                if attempt:
                    raise
        if result.get("isError"):
            raise RuntimeError(f"search failed: {result['content'][0]['text'][:300]}")
        return json.loads(result["content"][0]["text"])["results"]

    async def _mcp_open(self, http: aiohttp.ClientSession) -> None:
        await self._rpc(http, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "smart-speaker", "version": "0.1"}}})
        await self._rpc(http, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def _rpc(self, http: aiohttp.ClientSession, msg: dict) -> dict:
        headers = dict(MCP_HEADERS, **({"Mcp-Session-Id": self._mcp_session} if self._mcp_session else {}))
        async with http.post(MCP_URL, json=msg, headers=headers) as r:
            if r.status >= 400:
                raise RuntimeError(f"search MCP {r.status}: {(await r.text())[:300]}")
            self._mcp_session = r.headers.get("Mcp-Session-Id", self._mcp_session)
            body = await r.text()
            if "event-stream" in r.headers.get("Content-Type", ""):
                data = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
                body = data[-1] if data else ""
            return json.loads(body) if body.strip() else {}
