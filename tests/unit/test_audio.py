"""Unit tests for PCM frame helpers, resampling, energy VAD, and the stereo WAV writer."""

import wave

import pytest

from callcheck.core.audio import (
    EnergyVad,
    Resampler,
    StereoWavWriter,
    frame_bytes,
    rms_dbfs,
    silence,
    tone,
)


def test_frame_bytes_and_silence():
    assert frame_bytes(16000) == 640 and frame_bytes(24000) == 960
    assert silence(16000) == b"\x00" * 640


def test_tone_level_and_silence_floor():
    assert -21.0 < rms_dbfs(tone(440, 100, 16000)) < -19.0
    assert rms_dbfs(silence(16000)) == -120.0


def test_resampler_length_and_odd_input():
    r = Resampler(16000, 24000)
    out = r.process(tone(440, 100, 16000)) + r.flush()
    assert abs(len(out) - frame_bytes(24000) * 5) <= 4 * 2
    with pytest.raises(ValueError):
        Resampler(16000, 24000).process(b"\x00" * 3)


def test_vad_start_after_three_frames_and_end_after_twenty_five():
    v = EnergyVad()
    frames = [tone(440, 20, 16000)] * 3 + [silence(16000)] * 25
    marks = [v.feed(f) for f in frames]
    assert marks[2] == "start" and marks[-1] == "end"
    assert marks[:2] == [None, None] and set(marks[3:-1]) == {None}


def test_stereo_wav_roundtrip(tmp_path):
    w = StereoWavWriter(tmp_path / "x.wav")
    w.write(tone(440, 20, 24000), silence(24000))
    w.close()
    with wave.open(str(tmp_path / "x.wav")) as f:
        assert (f.getnchannels(), f.getframerate(), f.getnframes()) == (2, 24000, 480)
