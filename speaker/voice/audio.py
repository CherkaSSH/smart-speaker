"""Microphone capture (16 kHz PCM16 for ASR) and interruptible playback."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soxr

if TYPE_CHECKING:
    from .echo import EchoCanceller, Reference
    from .speakers import AudioRing

log = logging.getLogger("voice.audio")

ASR_RATE = 16000
PAUSE_AFTER = 2.0  # seconds of silence before the output stream stops


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


class TimedResampler:
    """One stream's audio as 16 kHz mono, each output chunk dated by sample count.

    An echo canceller needs the mic and the speaker reference aligned to the sample.
    Callback timestamps jitter, and soxr returns output in bursts (490, 979, ...
    samples for 640 expected), so chunks are dated from an anchor plus the number
    of output samples so far. All streams of one sound card share its clock, and
    both paths share soxr's delay, so they stay aligned.

    The anchor is the median of what the last blocks' timestamps imply: each one
    is off by up to ±0.9 ms, and a stream's first blocks can be off by much more
    (the mic's by 470 ms), so one timestamp alone can leave a stream misdated.
    """

    RESYNC = 0.05  # restart from the current block when timestamps jump this far (an xrun, start-up)
    WINDOW = 25

    def __init__(self, in_rate: int):
        self.in_rate = in_rate
        self.stream = soxr.ResampleStream(in_rate, ASR_RATE, 1, dtype="float32") if in_rate != ASR_RATE else None
        self.anchor: float | None = None
        self._implied: deque[float] = deque(maxlen=self.WINDOW)
        self.frames_in = 0
        self.samples_out = 0

    def __call__(self, block: np.ndarray, measured: float) -> tuple[np.ndarray, float]:
        """(16 kHz mono, time of its first sample). `measured` is PortAudio's time of the block."""
        implied = measured - self.frames_in / self.in_rate  # the anchor this block's timestamp implies
        if self.anchor is not None and abs(implied - self.anchor) > self.RESYNC:
            self._implied.clear()
        self._implied.append(implied)
        self.anchor = float(np.median(self._implied))
        mono = block.mean(axis=1) if block.ndim == 2 else block
        self.frames_in += len(mono)
        out = self.stream.resample_chunk(np.ascontiguousarray(mono, np.float32)) if self.stream else mono
        start = self.anchor + self.samples_out / ASR_RATE
        self.samples_out += len(out)
        return out, start


def dac_time(t) -> float:
    """PortAudio's estimate of when an output block reaches the speaker, as monotonic time."""
    now = time.monotonic()
    return now + (t.outputBufferDacTime - t.currentTime) if t.outputBufferDacTime else now


class Mic:
    """Pushes (capture_time, pcm16 bytes) chunks of ~100 ms at 16 kHz into asyncio queues."""

    def __init__(self, cfg: dict, loop: asyncio.AbstractEventLoop, queues: list[asyncio.Queue],
                 gated_streams: list | None = None, echo: "EchoCanceller | None" = None,
                 ring: "AudioRing | None" = None):
        self.rate = int(cfg["input_rate"])
        self.gain = float(cfg["mic_gain"])
        self.device = find_device(cfg["input_device"], "input")
        info = sd.query_devices(self.device, "input")
        self.channels = min(2, int(info["max_input_channels"]))
        self.queues = queues
        self.gated_streams = gated_streams or []
        self._loop = loop
        self.echo = echo  # removes our own speech and music from the mic signal
        self.ring = ring  # keeps the last seconds of what was heard, for speaker recognition
        self._timed = TimedResampler(self.rate)
        self._stream = sd.InputStream(device=self.device, samplerate=self.rate, channels=self.channels,
                                      dtype="float32", blocksize=self.rate // 10, callback=self._callback)

    def _callback(self, data, frames, t, status) -> None:
        now = time.monotonic()
        # PortAudio's ADC time is when the first frame was captured.
        measured = now - (t.currentTime - t.inputBufferAdcTime) if t.inputBufferAdcTime else now - frames / self.rate
        pcm, captured = self._timed(data * self.gain, measured)
        if not len(pcm):
            return
        if self.echo is not None:
            pcm, captured = self.echo.process(pcm, captured)
        pcm16 = (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()
        try:
            self._loop.call_soon_threadsafe(self._push, (captured, pcm16))
        except RuntimeError:  # loop closed during shutdown
            pass

    def _push(self, item: tuple[float, bytes]) -> None:
        if self.ring is not None:
            self.ring.add(*item)  # before the queues: an ASR final may read the span at once
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

    def __init__(self, cfg: dict, loop: asyncio.AbstractEventLoop, reference: "Reference | None" = None,
                 night=None):
        self.reference = reference  # what we play, for echo cancellation
        self.night = night  # caps the volume at night; None = never
        self.rate = int(cfg["sample_rate"])
        self.device = find_device(cfg["output_device"], "output")
        info = sd.query_devices(self.device, "output")
        self.channels = min(2, int(info["max_output_channels"]))
        self.out_channels = [c for c in cfg["output_channels"] if c < self.channels] or [0]
        self.level = int(cfg["volume"])  # what the brain set, before the night cap
        self._set_gains()
        self.last_sound = 0.0  # monotonic time the last non-silent block was played
        self._clips: deque[tuple[np.ndarray, bool]] = deque()  # (audio, loud)
        self._pos = 0
        self._lock = threading.Lock()
        self._loop = loop
        self._idle = asyncio.Event()
        self._idle.set()
        self._pause_timer: asyncio.TimerHandle | None = None
        # Open only while there is something to play: with no client, dmix stops
        # the card, so an idle speaker neither wakes the CPU nor keeps the DAC busy.
        # Opening takes about 1-2 ms.
        self._stream: sd.OutputStream | None = None

    def set_volume(self, level: int) -> None:
        """The volume the brain set; the night mode may cap it."""
        self.level = int(level)
        self._set_gains()

    def refresh_volume(self) -> None:
        """Re-apply the night cap when the window opens or closes."""
        self._set_gains()

    def _set_gains(self) -> None:
        self.loud_gain = volume_gain(self.level)  # alarms and reminders ignore the night cap
        self.gain = volume_gain(self.night.level(self.level) if self.night is not None else self.level)

    def _callback(self, out, frames, t, status) -> None:
        out.fill(0)
        written = 0
        with self._lock:
            while written < frames and self._clips:
                clip, loud = self._clips[0]
                n = min(frames - written, len(clip) - self._pos)
                out[written:written + n, self.out_channels] = \
                    (clip[self._pos:self._pos + n] * (self.loud_gain if loud else self.gain))[:, None]
                written += n
                self._pos += n
                if self._pos >= len(clip):
                    self._clips.popleft()
                    self._pos = 0
            empty = not self._clips
        if written:
            self.last_sound = time.monotonic() + written / self.rate
        if self.reference is not None:
            self.reference.add(*self._timed(out, dac_time(t)), "voice")
        if empty and not self._idle.is_set():
            self._loop.call_soon_threadsafe(self._mark_idle)

    def _mark_idle(self) -> None:
        # Re-checked on the loop: a clip queued since the callback must keep us busy.
        with self._lock:
            if self._clips:
                return
        self._idle.set()
        self._schedule_pause()

    def _schedule_pause(self) -> None:
        if self._pause_timer:
            self._pause_timer.cancel()
        # Short gaps between sentences and clips keep the stream open.
        self._pause_timer = self._loop.call_later(PAUSE_AFTER, self._pause)

    def _pause(self) -> None:
        self._pause_timer = None
        with self._lock:
            if self._clips:
                return
        self._close_stream()

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.abort()  # only silence is left in the buffer
            self._stream.close()
            self._stream = None

    def play(self, audio: np.ndarray, rate: int, loud: bool = False) -> None:
        if rate != self.rate:
            audio = soxr.resample(audio, rate, self.rate)
        with self._lock:
            self._clips.append((audio.astype(np.float32), loud))
            self._idle.clear()
        if self._pause_timer:
            self._pause_timer.cancel()
            self._pause_timer = None
        if self._stream is None:
            self._timed = TimedResampler(self.rate)  # a new stream counts its own samples
            self._stream = sd.OutputStream(device=self.device, samplerate=self.rate, channels=self.channels,
                                           dtype="float32", blocksize=self.rate // 50, callback=self._callback)
            self._stream.start()

    def stop(self) -> None:
        with self._lock:
            self._clips.clear()
            self._pos = 0
        self._idle.set()
        self._schedule_pause()

    @property
    def playing(self) -> bool:
        return not self._idle.is_set()

    async def wait_idle(self) -> None:
        await self._idle.wait()

    def close(self) -> None:
        if self._pause_timer:
            self._pause_timer.cancel()
        self._close_stream()
