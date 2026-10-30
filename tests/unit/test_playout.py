"""Unit tests for the playout buffer that paces audio to real time."""

from callcheck.core.playout import PlayoutBuffer


def test_pull_pads_with_silence_and_tracks_pending():
    b = PlayoutBuffer(16000)
    b.push(b"\x01" * 640)
    assert b.pending_ms == 20.0
    assert b.pull(1280) == b"\x01" * 640 + b"\x00" * 640
    assert b.pending_ms == 0.0


def test_push_beyond_cap_drops_oldest():
    b = PlayoutBuffer(16000, max_pending_ms=40)
    b.push(b"\x01" * 640)
    b.push(b"\x02" * 640)
    dropped = b.push(b"\x03" * 640)
    assert dropped == 640 and b.pull(640) == b"\x02" * 640


def test_flush_returns_dropped_bytes():
    b = PlayoutBuffer(16000)
    b.push(b"\x01" * 100)
    assert b.flush() == 100 and b.pending_ms == 0.0
