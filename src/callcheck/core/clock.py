"""Clock abstraction for the call core.

The mixer and conversation loop read time and wait only through a ``Clock``, so
production runs use the monotonic wall clock while tests drive a virtual clock
that advances instantly and deterministically.
"""

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    """Source of time in seconds and a way to wait until a target instant."""

    def now(self) -> float:
        """Return the current time in seconds."""
        ...

    async def sleep_until(self, t: float) -> None:
        """Wait until the clock reaches ``t`` seconds."""
        ...


class MonotonicClock:
    """Real-time clock backed by ``time.monotonic`` and ``asyncio.sleep``."""

    def now(self) -> float:
        """Return the monotonic time in seconds."""
        return time.monotonic()

    async def sleep_until(self, t: float) -> None:
        """Sleep until monotonic time ``t``; return at once if it has passed."""
        await asyncio.sleep(max(0.0, t - self.now()))


class VirtualClock:
    """Deterministic clock that jumps forward instead of waiting."""

    def __init__(self, start: float = 0.0) -> None:
        """Create a clock whose time starts at ``start`` seconds."""
        self._now = start

    def now(self) -> float:
        """Return the current virtual time in seconds."""
        return self._now

    async def sleep_until(self, t: float) -> None:
        """Advance to ``t`` (never backwards), then let other tasks run.

        Yielding to the event loop three times gives tasks woken by this
        advance, and the tasks they wake in turn, a chance to run before the
        caller continues.
        """
        self._now = max(self._now, t)
        for _ in range(3):
            await asyncio.sleep(0)
