"""Who is talking: local speaker embeddings and enrolled voice profiles.

The embedding model is 3D-Speaker CAM++ (192 numbers per voice), an ONNX file
run by sherpa-onnx on the CPU: 12 ms for a 2 s utterance on 4 threads here.
`sherpa-onnx` computes the fbank front end itself, so the vectors match how the
model was trained. Nothing leaves the machine: the profiles are cosine
comparisons against vectors kept in `[speakers] state_file`.

The [voice] service keeps the last seconds of microphone audio (`AudioRing`) so
that the exact span the ASR reported for an utterance can be embedded after the
fact. Recognition is only attempted once somebody is enrolled.

Enrollment (driven from the brain through the `enroll` protocol message) has
the speaker read three short prompts; their embeddings are averaged into one
profile. Averaging several takes is more stable than one, and the prompts give
the person some seconds of speech.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config

log = logging.getLogger("voice.speakers")
RATE = 16000
PAD_SECONDS = 0.2  # the ASR word times are approximate: keep a bit of room around them
ENROLL_PROMPTS = 3


@dataclass
class Enrollment:
    """One enrollment in progress. The voice service owns the dialogue; this is its state."""
    lang: str = "ru"
    aborted: str = ""  # why it stopped early ("stop" command, too long...), "" while running
    prompt_end: float = 0.0  # playback end of the prompt being answered: earlier audio is our own voice
    started: float = field(default_factory=time.monotonic)


def texts(lang: str) -> dict:
    """What the speaker says while enrolling a voice (spoken by the voice service itself)."""
    if lang == "en":
        return {
            "prompts": ["Say: the weather is nice today.", "Say: I like listening to music in the evening.",
                        "Say: tomorrow will be a good day."],
            "repeat": "I didn't catch that. Once more:",
            "done": "Done. I'll recognise your voice now.",
            "stopped": "Okay, I stopped recording your voice.",
            "timeout": "I can't hear you. Let's try another time.",
            "unavailable": "Voice recognition is not set up.",
        }
    return {
        "prompts": ["Скажи: сегодня хорошая погода.", "Скажи: я люблю слушать музыку по вечерам.",
                    "Скажи: завтра будет хороший день."],
        "repeat": "Не расслышал. Ещё раз:",
        "done": "Готово. Теперь я узнаю твой голос.",
        "stopped": "Ладно, отменяю запись голоса.",
        "timeout": "Тебя не слышно. Давай попробуем позже.",
        "unavailable": "Распознавание голоса не настроено.",
    }


class AudioRing:
    """The last seconds of 16 kHz mic audio, addressed by capture time.

    `Mic` hands each ~100 ms block over with the monotonic time of its first
    sample; ASR finals say when their words started and ended, so the ring can
    return exactly that span. Audio here is after echo cancellation: our own
    answers and music are already gone.
    """

    def __init__(self, seconds: float = 20.0, rate: int = RATE):
        self.rate = rate
        self.seconds = seconds
        self._chunks: deque[tuple[float, np.ndarray]] = deque()
        self._held = 0.0

    def add(self, captured: float, pcm16: bytes) -> None:
        if not pcm16:
            return
        self._chunks.append((captured, np.frombuffer(pcm16, np.int16)))
        self._held += len(pcm16) / 2 / self.rate
        while self._held > self.seconds and len(self._chunks) > 1:
            _, old = self._chunks.popleft()
            self._held -= len(old) / self.rate

    def audio(self, start: float, end: float, pad: float = PAD_SECONDS) -> np.ndarray:
        """Float32 16 kHz mono for [start, end], or an empty array if the audio is gone."""
        start, end = start - pad, end + pad
        if end <= start or not self._chunks:
            return np.zeros(0, np.float32)
        parts = []
        for captured, pcm in self._chunks:
            if captured + len(pcm) / self.rate <= start:
                continue
            if captured >= end:
                break  # chunks arrive in order
            a = max(0, int(round((start - captured) * self.rate)))
            b = min(len(pcm), int(round((end - captured) * self.rate)))
            if b > a:
                parts.append(pcm[a:b])
        if not parts:
            return np.zeros(0, np.float32)
        return np.concatenate(parts).astype(np.float32) / 32768.0


class VoiceProfiles:
    """Enrolled voices: unit-normalized embeddings by name, matched by cosine similarity."""

    def __init__(self, cfg: Config):
        s = cfg["speakers"]
        self.threshold = float(s["threshold"])
        self.min_seconds = float(s["min_seconds"])
        self.model = cfg.file(s["model"])
        self.state_file: Path = cfg.file(s["state_file"])
        self.threads = int(s["threads"])
        self.enabled = bool(s["enabled"])
        self._lock = threading.Lock()
        self._extractor = None
        self.voices: dict[str, np.ndarray] = {}
        if not self.enabled:
            log.info("speaker recognition disabled")
            return
        self._load()
        log.info("voice profiles: %s", ", ".join(self.names()) or "none enrolled")

    # Model and storage ---------------------------------------------------
    def _make_extractor(self):
        if self._extractor is not None:
            return self._extractor
        if not self.model.exists():
            log.warning("speaker model %s is missing: no voice recognition "
                        "(download note: README, Voice profiles)", self.model)
            return None
        try:
            import sherpa_onnx
            conf = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.model), num_threads=self.threads, provider="cpu")
            if not conf.validate():
                log.warning("speaker model %s is not usable", self.model)
                return None
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(conf)
        except Exception:  # a missing sherpa-onnx, a corrupt file: the speaker still works otherwise
            log.exception("speaker model failed to load")
            return None
        return self._extractor

    @property
    def available(self) -> bool:
        return self.enabled and self._make_extractor() is not None

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            d = json.loads(self.state_file.read_text())
            self.voices = {str(name): np.asarray(vec, np.float32) for name, vec in d.get("voices", {}).items()}
        except (OSError, ValueError, TypeError):
            log.exception("voice profiles %s are unreadable; starting empty", self.state_file)

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        d = {"version": 1,
             "voices": {name: [round(float(x), 6) for x in vec] for name, vec in self.voices.items()}}
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        tmp.replace(self.state_file)

    # Embeddings ----------------------------------------------------------
    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Unit-normalized embedding of a clip, or None when it is too short or the model is absent."""
        if len(audio) < self.min_seconds * RATE:
            return None
        extractor = self._make_extractor()
        if extractor is None:
            return None
        with self._lock:  # one inference at a time: the extractor's streams are not thread-safe
            stream = extractor.create_stream()
            stream.accept_waveform(RATE, np.ascontiguousarray(audio, np.float32))
            stream.input_finished()
            vector = np.asarray(extractor.compute(stream), np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        return vector / norm

    def identify(self, audio: np.ndarray) -> tuple[str, float] | None:
        """(name, similarity) of the closest enrolled voice, if it is similar enough."""
        if not self.voices:
            return None
        vector = self.embed(audio)
        if vector is None:
            return None
        name, score = max(((n, float(vector @ v)) for n, v in self.voices.items()), key=lambda x: x[1])
        if score < self.threshold:
            log.info("unknown voice (closest %s at %.2f)", name, score)
            return None
        log.info("voice %s (%.2f)", name, score)
        return name, score

    def enroll(self, name: str, vectors: list[np.ndarray]) -> bool:
        """Average embeddings of the person's takes into (or replace) the profile of `name`."""
        if not vectors:
            return False
        mean = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-6:
            return False
        for old in [n for n in self.voices if n.casefold() == name.casefold()]:
            del self.voices[old]  # «миша» re-enrolling «Миша» replaces, not duplicates
        self.voices[name] = (mean / norm).astype(np.float32)
        self.save()
        log.info("enrolled voice %s from %d takes", name, len(vectors))
        return True

    def forget(self, name: str) -> bool:
        for key in [n for n in self.voices if n.casefold() == name.casefold().strip()]:
            del self.voices[key]
            self.save()
            log.info("forgot voice %s", key)
            return True
        return False

    def names(self) -> list[str]:
        return sorted(self.voices, key=str.casefold)
