"""Acoustic echo cancellation: subtract what the speaker plays from what the mic hears.

Every output path (speech, music) adds its blocks to one Reference, keyed by
when they reach the speaker. The mic runs each chunk through WebRTC's AEC3 (the
echo canceller in Chrome) against the matching stretch of the reference, so the
wake gate and speech recognition hear the user rather than our own voice or the
song. AEC3 estimates the remaining delay itself; the timestamps only need to be
stable (see audio.TimedResampler), not exact.
"""

from __future__ import annotations

import threading

import numpy as np
from pywebrtc_audio import AudioProcessor

RATE = 16000
SECONDS = 10  # reference history kept


class Reference:
    """Everything being played, mixed, as mono 16 kHz on a monotonic-time axis."""

    def __init__(self):
        self.size = RATE * SECONDS
        self.ring = np.zeros(self.size, np.float32)
        self.origin: float | None = None
        self.cleared_to = 0  # sample index up to which the ring holds current (or zeroed) data
        self.lock = threading.Lock()

    def _index(self, t: float) -> int:
        if self.origin is None:
            self.origin = t
        return int(round((t - self.origin) * RATE))

    def _clear_to(self, end: int) -> None:
        if end <= self.cleared_to:
            return
        if end - self.cleared_to >= self.size:
            self.ring[:] = 0
        else:
            self.ring[np.arange(self.cleared_to, end) % self.size] = 0
        self.cleared_to = end

    def add(self, mono: np.ndarray, play_time: float) -> None:
        """Mono 16 kHz audio (see audio.TimedResampler) that reaches the speaker from play_time on."""
        n = len(mono)
        if not n:
            return
        with self.lock:
            i0 = self._index(play_time)
            self._clear_to(i0 + n)
            self.ring[np.arange(i0, i0 + n) % self.size] += mono

    def take(self, start: float, n: int) -> np.ndarray:
        """n samples of what the speaker played from `start`; silence where nothing was."""
        with self.lock:
            if self.origin is None:
                return np.zeros(n, np.float32)
            i0 = self._index(start)
            idx = np.arange(i0, i0 + n)
            out = self.ring[idx % self.size].copy()
            out[(idx >= self.cleared_to) | (idx < self.cleared_to - self.size)] = 0
            return out


class EchoCanceller:
    def __init__(self, reference: Reference, delay_ms: int = 60):
        self.reference = reference
        self.ap = AudioProcessor(sample_rate=RATE, echo_cancellation=True, stream_delay_ms=delay_ms)

    def process(self, near: np.ndarray, captured: float) -> np.ndarray:
        """Mic audio (float32, 16 kHz) captured from `captured` on, with our own playback removed."""
        return self.ap.process(near.astype(np.float32), self.reference.take(captured, len(near)))
