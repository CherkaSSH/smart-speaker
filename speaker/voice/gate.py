"""Cheap always-on speech and wake gate for the large ASR streams.

Silero VAD (~2 MB) skips silence. Vosk small Russian (~50 MB) listens for a
restricted wake grammar. Nemotron receives audio only after this gate opens and
still verifies the wake phrase before the speaker acts on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Awaitable, Callable

import numpy as np
import torch
from vosk import KaldiRecognizer, Model, SetLogLevel

from ..config import Config
from .wake import WakeMatcher

log = logging.getLogger("voice.gate")
RATE = 16000


class Gate:
    def __init__(self, cfg: Config, on_wake: Callable[[], Awaitable[None]], echo_until: Callable[[], float]):
        g = cfg["gate"]
        self.on_wake = on_wake
        # Monotonic time until which the mic may still hear our own playback. The
        # restricted wake grammar would otherwise turn that echo into a false wake.
        self.echo_until = echo_until
        self.threshold = float(g["vad_threshold"])
        # Chunks quieter than this are silence without asking Silero VAD.
        self.energy_floor = float(g["energy_floor"])
        self.audio: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue(maxsize=600)
        self.recent: deque[tuple[float, bytes]] = deque(maxlen=max(1, int(float(g["preroll_seconds"]) * 10)))
        self.active = False  # large ASR is active; the small gate can rest
        # The grammar holds the spellings this small model hears for the wake phrases
        # ("наташка" for "гптшка"); any of them opens the gate, and Nemotron then
        # checks the real wake phrase.
        self.wake = WakeMatcher([p for p in g["grammar"] if p != "[unk]"], float(cfg["wake"]["threshold"]))
        SetLogLevel(-1)
        torch.set_num_threads(1)
        self.vad = torch.jit.load(str(cfg.file(g["vad_model"])))
        self.model = Model(str(cfg.file(g["model"])))
        grammar = json.dumps(g["grammar"], ensure_ascii=False)
        self.rec = KaldiRecognizer(self.model, RATE, grammar)
        self._vad_pending = np.empty(0, np.float32)
        self._quiet = 20
        self._streak = 0

    def preroll(self) -> list[tuple[float, bytes]]:
        return list(self.recent)

    def reset(self) -> None:
        self.rec.Reset()
        self.vad.reset_states()
        self._quiet = 20
        self._streak = 0

    def _is_speech(self, pcm: np.ndarray) -> bool:
        self._vad_pending = np.concatenate([self._vad_pending, pcm])
        probabilities = []
        while len(self._vad_pending) >= 512:
            chunk, self._vad_pending = self._vad_pending[:512], self._vad_pending[512:]
            with torch.no_grad():
                probabilities.append(float(self.vad(torch.from_numpy(chunk), RATE)))
        return bool(probabilities) and max(probabilities) >= self.threshold

    async def run(self) -> None:
        while True:
            item = await self.audio.get()
            self.recent.append(item)
            if self.active:
                continue
            captured, raw = item
            if captured < self.echo_until():
                continue
            pcm = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
            if float(np.sqrt(np.mean(pcm * pcm))) < self.energy_floor:
                speech = False
                self._vad_pending = self._vad_pending[:0]  # VAD windows stay contiguous with real audio
            else:
                speech = self._is_speech(pcm)
            if speech:
                self._quiet = 0
            else:
                self._quiet += 1
            if self._quiet > 5:
                if self._quiet == 20:
                    self.reset()
                continue
            self.rec.AcceptWaveform(raw)
            text = json.loads(self.rec.PartialResult()).get("partial", "")
            self._streak = self._streak + 1 if self.wake.find(text) else 0
            if self._streak >= 2:
                log.info("small wake model heard %r", text)
                self._streak = 0
                await self.on_wake()
