"""Short synthetic sounds: listening chimes, the timer/alarm ring and the music for waiting."""

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


def _bell(freq: float, seconds: float, amp: float) -> np.ndarray:
    """A soft struck note: a few harmonics dying away, like a muted vibraphone."""
    t = np.arange(int(RATE * seconds)) / RATE
    wave = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t) * np.exp(-t / 0.15)
    attack = np.minimum(1, t / 0.008)
    return (amp * wave * attack * np.exp(-t / 0.7)).astype(np.float32)


def hold() -> np.ndarray:
    """Eight seconds of quiet music to loop while a slow answer is on its way: a slow arpeggio in
    A minor pentatonic over a faint pad. Notes ring out before the end, so loops join without a click."""
    seconds, step = 8.0, 0.5
    out = np.zeros(int(RATE * seconds), np.float32)
    notes = [440.0, 523.25, 659.25, 783.99, 880.0, 659.25, 587.33, 523.25,
             440.0, 587.33, 659.25, 880.0, 783.99, 659.25]  # 14 notes, the last at 6.5 s
    for i, f in enumerate(notes):
        start = int(RATE * i * step)
        note = _bell(f, seconds - i * step, 0.09 if i % 4 == 0 else 0.06)
        out[start:start + len(note)] += note
    t = np.arange(len(out)) / RATE
    breath = np.sin(np.pi * t / seconds)  # the pad swells and fades within the loop
    out += (0.02 * breath * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 329.63 * t))).astype(np.float32)
    edge = RATE // 50  # 20 ms ramps at both ends: the loop restarts without a click
    out[:edge] *= np.linspace(0, 1, edge, dtype=np.float32)
    out[-edge:] *= np.linspace(1, 0, edge, dtype=np.float32)
    return out
