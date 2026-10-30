"""Playout buffer that paces audio to real time.

Endpoints deliver audio in bursts; the mixer consumes it one 20 ms frame at a
time. ``PlayoutBuffer`` holds the surplus (PCM16 LE mono), pads underruns with
silence, and caps how much it will hold by discarding the oldest audio.
"""


class PlayoutBuffer:
    """FIFO of PCM16 audio with silence padding on pull and a cap on pending time."""

    def __init__(self, rate: int, max_pending_ms: int = 30000) -> None:
        """Create an empty buffer for audio at ``rate`` Hz holding at most ``max_pending_ms``."""
        self._bytes_per_ms = rate * 2 / 1000
        self._max_bytes = int(max_pending_ms * self._bytes_per_ms)
        self._buf = bytearray()

    @property
    def pending_ms(self) -> float:
        """Milliseconds of audio waiting to be pulled."""
        return len(self._buf) / self._bytes_per_ms

    def push(self, pcm: bytes) -> int:
        """Append audio; return the bytes dropped from the front to respect the cap."""
        self._buf.extend(pcm)
        dropped = max(0, len(self._buf) - self._max_bytes)
        del self._buf[:dropped]
        return dropped

    def pull(self, n: int) -> bytes:
        """Remove and return exactly ``n`` bytes, zero-padded when fewer are pending."""
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out + b"\x00" * (n - len(out))

    def flush(self) -> int:
        """Discard all pending audio; return the number of bytes dropped."""
        dropped = len(self._buf)
        self._buf.clear()
        return dropped
