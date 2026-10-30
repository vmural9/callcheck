"""Unit tests for the virtual clock used to run calls in simulated time."""

from callcheck.core.clock import VirtualClock


async def test_virtual_clock_advances_to_target():
    c = VirtualClock(start=1.0)
    await c.sleep_until(1.5)
    assert c.now() == 1.5
    await c.sleep_until(1.2)  # never goes backwards
    assert c.now() == 1.5
