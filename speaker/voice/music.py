"""Music playback: one track at a time from a file or a stream URL, through ffmpeg.

The brain picks tracks and keeps the queue; this plays them, pauses and resumes,
ducks under speech, and reports when a track ends. Any source ffmpeg can open
works (local files, HTTP/HLS streams), which is what streaming services hand out.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Awaitable, Callable

import numpy as np
import sounddevice as sd

from .audio import TimedResampler, dac_time, find_device, volume_gain
from .wake import phonetic, words

log = logging.getLogger("voice.music")

CHUNK_FRAMES = 4096  # ~85 ms at 48 kHz per ffmpeg read
BUFFER_CHUNKS = 24  # ~2 s decoded ahead
DUCK_RAMP = 0.15  # seconds to fade between full and ducked volume


class MusicPlayer:
    def __init__(self, audio_cfg: dict, music_cfg: dict, loop: asyncio.AbstractEventLoop,
                 on_event: Callable[[str, dict, str], Awaitable[None]], reference=None):
        self.reference = reference  # what we play, for echo cancellation
        self.rate = int(audio_cfg["sample_rate"])
        self.device = find_device(audio_cfg["output_device"], "output")
        self.channels = min(2, int(sd.query_devices(self.device, "output")["max_output_channels"]))
        self.gain = volume_gain(int(audio_cfg["volume"]))
        self.duck_gain = float(music_cfg["duck_volume"])
        self._duck_target = self._duck_now = 1.0
        self._loop = loop
        self._on_event = on_event
        self.track: dict | None = None
        self.state = "stopped"  # stopped, playing, paused
        self.lyric_words: set[str] = set()  # phonetic words of the current song, for echo removal
        self.levels: deque[tuple[float, float]] = deque(maxlen=200)  # (time, output RMS) of recent blocks
        self._proc: subprocess.Popen | None = None
        self._chunks: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=BUFFER_CHUNKS)
        self._rest = np.zeros((0, 2), np.float32)
        self._eof = False
        self._error = ""
        self._stream: sd.OutputStream | None = None
        self._generation = 0  # ignores callbacks from a replaced track

    # Control ---------------------------------------------------------------
    def play(self, track: dict) -> None:
        self.stop(report=False)
        self.track = track
        self.lyric_words = {phonetic(w) for w in words(track.get("lyrics") or "")}
        self._start_decoder()
        self.state = "playing"
        self._open()
        log.info("playing %s — %s", track.get("artist") or "?", track.get("title") or track.get("source"))

    def pause(self) -> None:
        if self.state == "playing":
            self.state = "paused"
            self._close()  # the device idles; ffmpeg stalls on its full pipe
            log.info("paused")

    def resume(self) -> None:
        if self.state != "paused" or self.track is None:
            return
        if self.track.get("live"):  # a live stream must not replay what was buffered
            self._start_decoder()
        self.state = "playing"
        self._open()
        log.info("resumed")

    def stop(self, report: bool = True) -> None:
        if self.state == "stopped":
            return
        self._close()
        self._kill_decoder()
        self.state = "stopped"
        if report:
            log.info("stopped")

    def set_volume(self, level: int) -> None:
        self.gain = volume_gain(level)

    def duck(self, on: bool) -> None:
        self._duck_target = self.duck_gain if on else 1.0

    def level(self, start: float, end: float) -> float | None:
        """Mean output RMS over a time window, or None if nothing was playing."""
        vals = [rms for t, rms in list(self.levels) if start <= t <= end]
        return float(np.mean(vals)) if vals and self.state == "playing" else None

    # Decoding --------------------------------------------------------------
    def _start_decoder(self) -> None:
        self._kill_decoder()
        src = self.track["source"]
        net = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"] \
            if "://" in src else []
        self._proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "error", *net, "-i", src, "-vn", "-f", "f32le",
             "-ac", "2", "-ar", str(self.rate), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        self._chunks = queue.Queue(maxsize=BUFFER_CHUNKS)
        self._rest = np.zeros((0, 2), np.float32)
        self._eof = False
        self._error = ""
        threading.Thread(target=self._read, args=(self._proc, self._chunks, self._generation), daemon=True).start()

    def _read(self, proc: subprocess.Popen, chunks: queue.Queue, generation: int) -> None:
        size = CHUNK_FRAMES * 2 * 4
        while True:
            data = proc.stdout.read(size)
            if not data:
                break
            chunk = np.frombuffer(data[: len(data) // 8 * 8], np.float32).reshape(-1, 2)
            while generation == self._generation:
                try:
                    chunks.put(chunk, timeout=0.5)
                    break
                except queue.Full:
                    continue  # paused or ahead: wait
            if generation != self._generation:
                return
        err = proc.stderr.read().decode(errors="replace").strip()
        code = proc.wait()
        if generation == self._generation:
            if code not in (0, -9, -15) and err:
                log.warning("ffmpeg: %s", err[-300:])
                self._error = err[-200:]
            chunks.put(None)

    def _kill_decoder(self) -> None:
        self._generation += 1
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
        self._proc = None

    # Output ----------------------------------------------------------------
    def _open(self) -> None:
        if self._stream is None:
            self._timed = TimedResampler(self.rate)  # a new stream counts its own samples
            self._stream = sd.OutputStream(device=self.device, samplerate=self.rate, channels=self.channels,
                                           dtype="float32", blocksize=self.rate // 25, callback=self._callback)
            self._stream.start()

    def _close(self) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None

    def _callback(self, out, frames, t, status) -> None:
        buf = self._rest
        while len(buf) < frames:
            try:
                chunk = self._chunks.get_nowait()
            except queue.Empty:
                break  # network or disk is behind: play what we have, pad with silence
            if chunk is None:
                self._eof = True
                break
            buf = np.concatenate([buf, chunk]) if len(buf) else chunk
        n = min(frames, len(buf))
        block = np.zeros((frames, 2), np.float32)
        block[:n] = buf[:n]
        self._rest = buf[n:]
        # Ramp toward the duck target within the block (no clicks).
        step = 1.0 / max(1.0, DUCK_RAMP * self.rate)
        target = self._duck_target
        ramp = np.clip(self._duck_now + np.sign(target - self._duck_now) * step * np.arange(1, frames + 1),
                       min(self._duck_now, target), max(self._duck_now, target))
        self._duck_now = float(ramp[-1])
        block *= (ramp * self.gain)[:, None]
        if self.channels == 2:
            out[:] = block
        else:
            out[:, 0] = block.mean(axis=1)
        self.levels.append((time.monotonic(), float(np.sqrt(np.mean(block * block)))))
        if self.reference is not None:
            self.reference.add(*self._timed(block, dac_time(t)))
        if self._eof and n == 0 and not len(self._rest):
            self._loop.call_soon_threadsafe(self._finished, self._generation)

    def _finished(self, generation: int) -> None:
        if generation != self._generation or self.state != "playing":
            return
        self._close()
        self._kill_decoder()
        self.state = "stopped"
        self._report("error" if self._error else "ended", self._error)

    def _report(self, event: str, detail: str) -> None:
        if self.track is not None:
            asyncio.ensure_future(self._on_event(event, self.track, detail))

    def close(self) -> None:
        self._close()
        self._kill_decoder()
