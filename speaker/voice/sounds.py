"""Short synthetic sounds: listening chimes and the timer/alarm ring."""

from __future__ import annotations

import numpy as np

RATE = 48000


def _tone(freq: float, seconds: float, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.01)  # 10 ms ramps, no clicks
    return (amp / 1.3 * wave * fade).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds), np.float32)


def listen_start() -> np.ndarray:
    return np.concatenate([_tone(660, 0.07), _tone(990, 0.09)])


def listen_end() -> np.ndarray:
    return np.concatenate([_tone(990, 0.07), _tone(660, 0.09)])


def ring() -> np.ndarray:
    """One ring cycle (~1.4 s): three beeps and a pause."""
    beep = np.concatenate([_tone(1046, 0.14, 0.35), _silence(0.08)])
    return np.concatenate([beep, beep, beep, _silence(0.7)])
