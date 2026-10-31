"""Turn streamed LLM text into clean, speakable sentences."""

from __future__ import annotations

import re

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"»”)]?\s+|\n+")


def clean(text: str) -> str:
    """Strip what a voice can't say: markdown, emoji, stray reasoning tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url)
    text = re.sub(r"[*_`#>|~]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = _EMOJI.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


class SentenceSplitter:
    """Feed streamed tokens, get complete sentences as early as possible."""

    def __init__(self, min_chars: int = 12):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, chunk: str) -> list[str]:
        self.buf += chunk
        out = []
        while True:
            # The model sometimes drafts an answer, closes a thought, then answers
            # again: only what follows the last </think> is the answer.
            if "</think>" in self.buf:
                self.buf = self.buf.rsplit("</think>", 1)[1]
                out.clear()
            m = None
            for cand in _SENTENCE_END.finditer(self.buf):
                if cand.start() >= self.min_chars:
                    m = cand
                    break
            if not m:
                return [s for s in (clean(x) for x in out) if s]
            out.append(self.buf[: m.start()])
            self.buf = self.buf[m.end():]

    def flush(self) -> list[str]:
        rest, self.buf = self.buf, ""
        rest = clean(rest)
        return [rest] if rest else []
