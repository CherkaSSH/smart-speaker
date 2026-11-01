"""Acoustic echo cancellation: subtract what the speaker plays from what the mic hears.

Every output path (speech, music) adds its blocks to one Reference, keyed by
when they reach the speaker. The mic runs each chunk through LocalVQE, a small
neural echo canceller (adaptive filter plus a GTCRN mask trained on double talk),
against the matching stretch of the reference, so the wake gate and speech
recognition hear the user rather than our own voice or the song. LocalVQE
estimates the remaining delay itself; the timestamps only need to be stable
(see audio.TimedResampler), not exact.

WebRTC's AEC3 was measured first and dropped: over loud music its echo
suppressor removed the user's voice along with the song, leaving the command
less intelligible than the raw mic.
"""

from __future__ import annotations

import ctypes
import os
import threading

import numpy as np

RATE = 16000
SECONDS = 10  # reference history kept


class Reference:
    """Everything being played, mixed, as mono 16 kHz on a monotonic-time axis,
    plus when each source ("voice": speech and sounds, "music") last made sound."""

    def __init__(self):
        self.size = RATE * SECONDS
        self.ring = np.zeros(self.size, np.float32)
        self.origin: float | None = None
        self.cleared_to = 0  # sample index up to which the ring holds current (or zeroed) data
        self.sounded = {"voice": 0.0, "music": 0.0}  # end of each source's last non-silent block
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

    def add(self, mono: np.ndarray, play_time: float, source: str) -> None:
        """Mono 16 kHz audio (see audio.TimedResampler) that reaches the speaker from play_time on."""
        n = len(mono)
        if not n:
            return
        with self.lock:
            i0 = self._index(play_time)
            self._clear_to(i0 + n)
            self.ring[np.arange(i0, i0 + n) % self.size] += mono
            if mono.any():
                self.sounded[source] = max(self.sounded[source], play_time + n / RATE)

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


_H = ctypes.c_size_t  # LocalVQE's opaque handles
_FP = ctypes.POINTER(ctypes.c_float)


def _load_library() -> ctypes.CDLL:
    path = os.environ.get("LOCALVQE_LIB")
    if not path:
        raise RuntimeError("LOCALVQE_LIB is not set: run inside the flake's dev shell (nix develop), "
                           "or set [audio] echo_cancellation = false")
    lib = ctypes.CDLL(path)
    lib.localvqe_options_new.restype = _H
    lib.localvqe_options_set_model_path.argtypes = [_H, ctypes.c_char_p]
    lib.localvqe_options_set_threads.argtypes = [_H, ctypes.c_int]
    lib.localvqe_options_free.argtypes = [_H]
    lib.localvqe_new_with_options.argtypes = [_H]
    lib.localvqe_new_with_options.restype = _H
    lib.localvqe_hop_length.argtypes = [_H]
    lib.localvqe_process_frame_f32.argtypes = [_H, _FP, _FP, ctypes.c_int, _FP]
    return lib


class EchoCanceller:
    """LocalVQE over the mic stream, delaying it by one 16 ms hop.

    Two models, each for what it does best (measured over the same mic):
    - our own speech: the 200K model let «стоп» through over a long answer in 3 of
      4 tries, the 49K model in 0 of 4;
    - music: the 49K model kept commands over loud rock at volume 7 (15 of 18),
      the 200K model didn't (2 of 12).
    Our speech wins while it plays (it is what barge-in talks over); music alone
    goes to the music model. Only one runs per hop (about 6-9% of a CPU core), and
    only while something plays plus HOLD for the room's echo tail. In between, the
    mic passes through the same delay unchanged, and each model keeps what it
    learned about the room: a fresh one needs about 2.5 s to cancel fully again.
    """

    HOLD = 1.0

    def __init__(self, reference: Reference, voice_model: str, music_model: str):
        self.reference = reference
        self.lib = _load_library()
        self.ctx = {"voice": self._load(voice_model), "music": self._load(music_model)}
        self.hop = self.lib.localvqe_hop_length(self.ctx["voice"])
        self.delay = self.hop / RATE
        self._near = np.zeros(0, np.float32)  # input not yet a full hop
        self._far = np.zeros(0, np.float32)
        self._out = np.zeros(self.hop, np.float32)  # output not yet returned; starts with the delay
        self._hop_out = np.zeros(self.hop, np.float32)

    def _load(self, model: str) -> int:
        opts = self.lib.localvqe_options_new()
        self.lib.localvqe_options_set_model_path(opts, model.encode())
        self.lib.localvqe_options_set_threads(opts, 1)  # as fast as more for these model sizes
        ctx = self.lib.localvqe_new_with_options(opts)
        self.lib.localvqe_options_free(opts)
        if not ctx:
            raise RuntimeError(f"LocalVQE could not load {model}")
        return ctx

    def process(self, near: np.ndarray, captured: float) -> tuple[np.ndarray, float]:
        """Mic audio (16 kHz) captured from `captured` on -> (the same length with our
        own playback removed, the capture time of its first sample)."""
        far = self.reference.take(captured, len(near))
        sounded = self.reference.sounded
        ctx = next((self.ctx[s] for s in ("voice", "music") if captured < sounded[s] + self.HOLD), None)
        self._near = np.concatenate([self._near, near.astype(np.float32)])
        self._far = np.concatenate([self._far, far])
        hops = len(self._near) // self.hop
        done = [self._out]
        for k in range(hops):
            m = np.ascontiguousarray(self._near[k * self.hop:(k + 1) * self.hop])
            if ctx is not None:
                r = np.ascontiguousarray(self._far[k * self.hop:(k + 1) * self.hop])
                self.lib.localvqe_process_frame_f32(ctx, m.ctypes.data_as(_FP), r.ctypes.data_as(_FP),
                                                    self.hop, self._hop_out.ctypes.data_as(_FP))
                done.append(self._hop_out.copy())
            else:
                done.append(m)
        self._near, self._far = self._near[hops * self.hop:], self._far[hops * self.hop:]
        out = np.concatenate(done)
        self._out = out[len(near):]
        return out[:len(near)], captured - self.delay
