"""Streaming chat with LFM3 through llama-server's OpenAI API, including tool calls."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import AsyncIterator, Awaitable, Callable

import aiohttp

from .text import SentenceSplitter

log = logging.getLogger("brain.llm")

ToolRunner = Callable[[str, dict], Awaitable[object]]


class LLM:
    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/") + "/chat/completions"
        self.budget = int(cfg["reasoning_budget"])
        self.tool_budget = int(cfg["tool_reasoning_budget"])
        self.max_tokens = int(cfg["max_tokens"])
        self.temperature = float(cfg["temperature"])
        self.session: aiohttp.ClientSession | None = None

    async def _stream(self, messages: list[dict], tools: list[dict], budget: int) -> AsyncIterator[dict]:
        body = {
            "messages": messages, "tools": tools, "stream": True,
            "temperature": self.temperature, "top_k": 50, "repeat_penalty": 1.1,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": budget != 0},
            "reasoning_budget_tokens": budget if budget > 0 else -1,
            "reasoning_budget_message": " Enough thinking, answer now.",
        }
        if self.session is None:
            # llama-server closes idle keep-alive sockets; reusing one fails the next request.
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),
                                                 connector=aiohttp.TCPConnector(force_close=True))
        async with self.session.post(self.url, json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"llama-server {r.status}: {await r.text()}")
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                yield json.loads(data)

    @staticmethod
    def _leak(sentence: str, lang: str) -> bool:
        """English reasoning spilling into a Russian answer once the thinking budget ran out."""
        if lang != "ru":
            return False
        ws = re.findall(r"\w+", sentence)
        latin = sum(1 for w in ws if re.fullmatch(r"[A-Za-z']+", w))
        return latin >= 4 and latin >= 0.6 * len(ws)

    async def respond(self, messages: list[dict], tools: list[dict], run_tool: ToolRunner, lang: str,
                      max_rounds: int = 3, preface: Callable[[str], str | None] | None = None) -> AsyncIterator[str]:
        """Yield speakable sentences. `messages` is extended with the assistant turn(s).

        Text the model writes before a tool call is spoken right away, and it keeps
        talking after the result. LFM3 rarely writes any, so `preface(tool_name)` may
        supply a short line ("Сейчас гляну.") to say while a slow tool runs."""
        for round_no in range(max_rounds):
            splitter = SentenceSplitter()
            content = ""  # what was actually said
            calls: dict[int, dict] = {}
            t0 = time.monotonic()
            first = None
            reasoning = 0
            budget = self.budget if round_no == 0 else self.tool_budget
            async for chunk in self._stream(messages, tools, budget):
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning += 1
                if delta.get("content"):
                    if first is None:
                        first = time.monotonic() - t0
                    sentences = splitter.feed(delta["content"])
                    if any(self._leak(s, lang) for s in sentences):
                        log.info("stopped: reasoning leaked into the answer: %r", sentences)
                        splitter = SentenceSplitter()  # drop the unfinished rest too
                        break
                    for s in sentences:
                        content += s + " "
                        yield s
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
            for s in splitter.flush():
                if not self._leak(s, lang):
                    content += s
                    yield s
            log.info("llm round: %.2fs, reasoning chunks %d, first content %s, tools %s",
                     time.monotonic() - t0, reasoning, f"{first:.2f}s" if first else "-",
                     [c["name"] for c in calls.values()])
            if not calls:
                messages.append({"role": "assistant", "content": content})
                return
            parsed = []
            for c in calls.values():
                try:
                    args = json.loads(c["arguments"]) if c["arguments"].strip() else {}
                except json.JSONDecodeError:
                    args = {}
                parsed.append((c, args))
            if not content.strip() and preface:
                line = preface(parsed[0][0]["name"])
                if line:
                    content = line
                    yield line
            # The LFM3 template wants tool-call arguments as a mapping, not a JSON string.
            messages.append({"role": "assistant", "content": content, "tool_calls": [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": args}} for i, (c, args) in enumerate(parsed)]})
            for i, (c, args) in enumerate(parsed):
                result = await run_tool(c["name"], args)
                log.info("tool %s(%s) -> %s", c["name"], args, result)
                messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                                 "content": json.dumps(result, ensure_ascii=False)})

    async def close(self) -> None:
        if self.session:
            await self.session.close()
