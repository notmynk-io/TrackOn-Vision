"""Audio analysis utilities for interrogation analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import threading

import numpy as np

try:  # Optional import so module can be unit-tested without hardware
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled at runtime
    sd = None  # type: ignore


Number = float


def sanitize_numeric(value):
    """Convert numpy scalar types to native Python types for JSON serialization."""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass
class AudioBlockFeatures:
    energy: Number = 0.0
    zero_cross_rate: Number = 0.0
    pitch_hz: Number = 0.0
    pitch_jitter: Number = 0.0
    spectral_centroid: Number = 0.0
    voice_activity: bool = False
    voice_tremor: Number = 0.0
    speech_rate_per_min: Number = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def as_dict(self) -> Dict[str, Number]:
        base = {
            "energy": sanitize_numeric(self.energy),
            "zero_cross_rate": sanitize_numeric(self.zero_cross_rate),
            "pitch_hz": sanitize_numeric(self.pitch_hz),
            "pitch_jitter": sanitize_numeric(self.pitch_jitter),
            "spectral_centroid": sanitize_numeric(self.spectral_centroid),
            "voice_activity": bool(self.voice_activity),
            "voice_tremor": sanitize_numeric(self.voice_tremor),
            "speech_rate_per_min": sanitize_numeric(self.speech_rate_per_min),
            "timestamp": self.timestamp,
        }
        return base


class AudioAnalyzer:
    """Streaming audio analyzer extracting stress-related voice features."""

    def __init__(self, sample_rate: int = 16_000, block_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.block_size = int(sample_rate * block_duration)
        self.stream: Optional[sd.InputStream] = None  # type: ignore[name-defined]
        self.lock = threading.Lock()
        self.total_blocks = 0
        self.speech_events = 0
        self.last_pitch = 0.0

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Begin streaming audio if hardware/permissions allow."""
        if sd is None:
            return False
        if self.stream is not None:
            return True
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                callback=self._callback,
            )
            self.stream.start()
            return True
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(f"Audio analyzer error: {exc}")
            self.stream = None
            return False

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # pragma: no cover
                pass
            finally:
                self.stream = None

    def reset(self) -> None:
        with self.lock:
            self.total_blocks = 0
            self.speech_events = 0
            self.last_pitch = 0.0

    def is_running(self) -> bool:
        return self.stream is not None

    # ------------------------------------------------------------------
    # Sounddevice callback
    # ------------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):  # pragma: no cover - realtime
        try:
            audio = np.squeeze(indata).astype(np.float32)
            features = self._extract_features(audio)

            with self.lock:
                self.total_blocks += 1
                if features.voice_activity:
                    self.speech_events += 1
                duration_minutes = max(self.total_blocks * (self.block_duration / 60.0), 1e-6)
                features.speech_rate_per_min = round(self.speech_events / duration_minutes, 2)

            self._emit(features)
        except Exception as exc:
            print(f"Audio processing error: {exc}")

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def _extract_features(self, audio: np.ndarray) -> AudioBlockFeatures:
        if audio.size == 0:
            return AudioBlockFeatures()

        audio = audio - np.mean(audio)
        energy = float(np.sqrt(np.mean(audio ** 2)))

        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
        zero_cross_rate = float((zero_crossings / 2.0) / max(audio.size, 1))

        pitch = self._estimate_pitch(audio)
        pitch_jitter = float(abs(pitch - self.last_pitch)) if self.last_pitch else 0.0
        self.last_pitch = pitch

        magnitude = np.abs(np.fft.rfft(audio))
        if magnitude.sum() > 0:
            freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.sample_rate)
            spectral_centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
        else:
            spectral_centroid = 0.0

        voice_activity = bool(energy > 0.01)
        voice_tremor = float(np.std(audio))

        return AudioBlockFeatures(
            energy=energy,
            zero_cross_rate=zero_cross_rate,
            pitch_hz=pitch,
            pitch_jitter=pitch_jitter,
            spectral_centroid=spectral_centroid,
            voice_activity=voice_activity,
            voice_tremor=voice_tremor,
        )

    def _estimate_pitch(self, audio: np.ndarray) -> Number:
        min_freq = 75
        max_freq = 400
        min_lag = int(self.sample_rate / max_freq)
        max_lag = int(self.sample_rate / min_freq)

        if max_lag >= len(audio) or min_lag <= 0:
            return 0.0

        autocorr = np.correlate(audio, audio, mode="full")
        autocorr = autocorr[len(audio) - 1 :]
        autocorr[:min_lag] = 0
        search_region = autocorr[min_lag:max_lag]
        if search_region.size == 0:
            return 0.0
        peak_idx = int(np.argmax(search_region)) + min_lag
        peak_value = autocorr[peak_idx]
        if peak_value <= 0:
            return 0.0
        return float(self.sample_rate / peak_idx)

    # ------------------------------------------------------------------
    # Override this hook to integrate with the main app later
    # ------------------------------------------------------------------
    def _emit(self, features: AudioBlockFeatures) -> None:
        """Hook for streaming frameworks – override when integrating."""
        pass


__all__ = ["AudioAnalyzer", "AudioBlockFeatures", "sanitize_numeric"]
