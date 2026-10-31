"""Microphone capture (16 kHz PCM16 for ASR) and interruptible playback."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import soxr

log = logging.getLogger("voice.audio")

ASR_RATE = 16000


def find_device(name: str, kind: str) -> int | None:
    """Index of the first device whose name contains `name` and has `kind` ("input"/"output") channels."""
    if name in ("", "default"):
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0:
            return i
    raise RuntimeError(f"no {kind} audio device matching {name!r}; have: "
                       + ", ".join(d["name"] for d in sd.query_devices()))


def volume_gain(level: int) -> float:
    """0..10 like Yandex Station; 3 dB per step, 10 = full scale."""
    return 0.0 if level <= 0 else 10 ** ((min(level, 10) - 10) * 3 / 20)


class Mic:
    """Pushes (capture_time, pcm16 bytes) chunks of ~100 ms at 16 kHz into asyncio queues."""

    def __init__(self, cfg: dict, loop: asyncio.AbstractEventLoop, queues: list[asyncio.Queue],
                 gated_streams: list | None = None):
        self.rate = int(cfg["sample_rate"])
        self.gain = float(cfg["mic_gain"])
        self.device = find_device(cfg["input_device"], "input")
        info = sd.query_devices(self.device, "input")
        self.channels = min(2, int(info["max_input_channels"]))
        self.queues = queues
        self.gated_streams = gated_streams or []
        self._loop = loop
        self._resampler = soxr.ResampleStream(self.rate, ASR_RATE, 1, dtype="float32")
        self._stream = sd.InputStream(device=self.device, samplerate=self.rate, channels=self.channels,
                                      dtype="float32", blocksize=self.rate // 10, callback=self._callback)

    def _callback(self, data, frames, t, status) -> None:
        captured = time.monotonic() - frames / self.rate
        mono = data.mean(axis=1) * self.gain
        pcm = self._resampler.resample_chunk(mono)
        pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()
        try:
            self._loop.call_soon_threadsafe(self._push, (captured, pcm16))
        except RuntimeError:  # loop closed during shutdown
            pass

    def _push(self, item: tuple[float, bytes]) -> None:
        for q in self.queues:
            if q.full():  # consumer is down: keep the newest audio
                q.get_nowait()
            q.put_nowait(item)
        for stream in self.gated_streams:
            if not stream.active:
                continue
            q = stream.audio
            if q.full():
                q.get_nowait()
            q.put_nowait(item)

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


class Player:
    """Plays queued clips; `stop()` silences instantly. Remembers when it last made sound."""

    def __init__(self, cfg: dict, loop: asyncio.AbstractEventLoop):
        self.rate = int(cfg["sample_rate"])
        self.device = find_device(cfg["output_device"], "output")
        info = sd.query_devices(self.device, "output")
        self.channels = min(2, int(info["max_output_channels"]))
        self.out_channels = [c for c in cfg["output_channels"] if c < self.channels] or [0]
        self.gain = volume_gain(int(cfg["volume"]))
        self.last_sound = 0.0  # monotonic time the last non-silent block was played
        self._clips: deque[np.ndarray] = deque()
        self._pos = 0
        self._lock = threading.Lock()
        self._loop = loop
        self._idle = asyncio.Event()
        self._idle.set()
        self._stream = sd.OutputStream(device=self.device, samplerate=self.rate, channels=self.channels,
                                       dtype="float32", blocksize=self.rate // 50, callback=self._callback)

    def set_volume(self, level: int) -> None:
        self.gain = volume_gain(level)

    def _callback(self, out, frames, t, status) -> None:
        out.fill(0)
        written = 0
        with self._lock:
            while written < frames and self._clips:
                clip = self._clips[0]
                n = min(frames - written, len(clip) - self._pos)
                out[written:written + n, self.out_channels] = (clip[self._pos:self._pos + n] * self.gain)[:, None]
                written += n
                self._pos += n
                if self._pos >= len(clip):
                    self._clips.popleft()
                    self._pos = 0
            empty = not self._clips
        if written:
            self.last_sound = time.monotonic() + written / self.rate
        if empty and not self._idle.is_set():
            self._loop.call_soon_threadsafe(self._mark_idle)

    def _mark_idle(self) -> None:
        # Re-checked on the loop: a clip queued since the callback must keep us busy.
        with self._lock:
            if not self._clips:
                self._idle.set()

    def play(self, audio: np.ndarray, rate: int) -> None:
        if rate != self.rate:
            audio = soxr.resample(audio, rate, self.rate)
        with self._lock:
            self._clips.append(audio.astype(np.float32))
            self._idle.clear()

    def stop(self) -> None:
        with self._lock:
            self._clips.clear()
            self._pos = 0
        self._idle.set()

    @property
    def playing(self) -> bool:
        return not self._idle.is_set()

    async def wait_idle(self) -> None:
        await self._idle.wait()

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
