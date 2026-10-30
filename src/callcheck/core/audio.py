"""PCM audio primitives for the call core.

Audio inside the core is PCM16 little-endian mono carried as ``bytes`` in 20 ms
frames. This module provides frame sizing, synthetic tones and silence, level
measurement, a streaming resampler, an energy-based voice activity detector,
and a stereo WAV writer used to record both sides of a call.
"""

import math
import wave
from pathlib import Path
from typing import Literal

import numpy as np
import soxr

FRAME_MS = 20
_SAMPLE_WIDTH = 2
_FULL_SCALE = 32768.0
_FLOOR_DBFS = -120.0


def frame_bytes(rate: int) -> int:
    """Return the byte size of one ``FRAME_MS`` PCM16 mono frame at ``rate`` Hz."""
    return rate * FRAME_MS // 1000 * _SAMPLE_WIDTH


def silence(rate: int, ms: int = FRAME_MS) -> bytes:
    """Return ``ms`` milliseconds of PCM16 mono silence at ``rate`` Hz."""
    return b"\x00" * (rate * ms // 1000 * _SAMPLE_WIDTH)


def tone(freq_hz: float, ms: int, rate: int, dbfs: float = -20.0) -> bytes:
    """Return a sine tone of ``ms`` milliseconds whose RMS level is ``dbfs``."""
    count = rate * ms // 1000
    peak = _FULL_SCALE * 10 ** (dbfs / 20) * math.sqrt(2)
    t = np.arange(count) / rate
    samples = np.round(peak * np.sin(2 * math.pi * freq_hz * t))
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def rms_dbfs(pcm: bytes) -> float:
    """Return the RMS level of ``pcm`` in dBFS, or -120.0 for empty or silent audio."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return _FLOOR_DBFS
    rms = math.sqrt(float(np.mean(samples**2)))
    if rms == 0.0:
        return _FLOOR_DBFS
    return 20 * math.log10(rms / _FULL_SCALE)


class Resampler:
    """Stateful PCM16 mono resampler that converts a stream chunk by chunk."""

    def __init__(self, in_rate: int, out_rate: int) -> None:
        """Create a resampler from ``in_rate`` Hz to ``out_rate`` Hz."""
        self._stream = soxr.ResampleStream(in_rate, out_rate, num_channels=1, dtype="int16")

    def process(self, pcm: bytes) -> bytes:
        """Resample one chunk of PCM16 bytes; raise ``ValueError`` on odd length."""
        if len(pcm) % _SAMPLE_WIDTH:
            raise ValueError("odd byte length")
        chunk = np.frombuffer(pcm, dtype="<i2")
        return self._stream.resample_chunk(chunk).tobytes()

    def flush(self) -> bytes:
        """Emit any audio still buffered inside the resampler at end of stream."""
        empty = np.zeros(0, dtype="<i2")
        return self._stream.resample_chunk(empty, last=True).tobytes()


class EnergyVad:
    """Frame-level voice activity detector with start and end hysteresis."""

    def __init__(
        self,
        *,
        start_dbfs: float = -40.0,
        end_dbfs: float = -45.0,
        start_frames: int = 3,
        end_frames: int = 25,
    ) -> None:
        """Configure thresholds in dBFS and the consecutive frame counts to switch state."""
        self._start_dbfs = start_dbfs
        self._end_dbfs = end_dbfs
        self._start_frames = start_frames
        self._end_frames = end_frames
        self._run = 0
        self.speaking = False

    def feed(self, pcm: bytes) -> Literal["start", "end"] | None:
        """Consume one frame; return ``"start"`` or ``"end"`` on a transition, else ``None``."""
        level = rms_dbfs(pcm)
        if self.speaking:
            self._run = self._run + 1 if level < self._end_dbfs else 0
            if self._run >= self._end_frames:
                self.speaking = False
                self._run = 0
                return "end"
        else:
            self._run = self._run + 1 if level > self._start_dbfs else 0
            if self._run >= self._start_frames:
                self.speaking = True
                self._run = 0
                return "start"
        return None


class StereoWavWriter:
    """Writes two mono PCM16 streams to one stereo WAV, left and right channels."""

    def __init__(self, path: Path, rate: int = 24000) -> None:
        """Open ``path`` for writing a 16-bit stereo WAV at ``rate`` Hz."""
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(_SAMPLE_WIDTH)
        self._wav.setframerate(rate)

    def write(self, left: bytes, right: bytes) -> None:
        """Append equal-length mono buffers as stereo; raise ``ValueError`` if lengths differ."""
        if len(left) != len(right):
            raise ValueError("left and right lengths differ")
        pair = np.empty((len(left) // _SAMPLE_WIDTH, 2), dtype="<i2")
        pair[:, 0] = np.frombuffer(left, dtype="<i2")
        pair[:, 1] = np.frombuffer(right, dtype="<i2")
        self._wav.writeframes(pair.tobytes())

    def close(self) -> None:
        """Finalise the WAV header and close the file."""
        self._wav.close()
