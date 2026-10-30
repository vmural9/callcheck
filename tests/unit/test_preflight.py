"""Unit tests for the preflight hearing check, run on the offline fakes."""

from pathlib import Path

from callcheck.adapters.callers.fake import FakeCaller
from callcheck.adapters.fake_channel import FakeChannel
from callcheck.adapters.targets.fake import FakeTarget
from callcheck.core.clock import VirtualClock
from callcheck.core.preflight import PreflightResult, format_preflight, run_preflight

PHRASE = "Hello, can you hear me? Testing one two three."


async def preflight(
    tmp_path: Path, *, behaviours: set[str] = frozenset(), script: list[str] | None = None
) -> PreflightResult:
    """Run a preflight between a fake target and a fake caller speaking ``script``."""
    channel = FakeChannel()
    target = FakeTarget(channel, ["first_name", "phone"], behaviours=behaviours)
    caller = FakeCaller(channel, {}, script=[PHRASE] if script is None else script)
    return await run_preflight(
        target=target, caller=caller, clock=VirtualClock(), run_dir=tmp_path / "pf"
    )


async def test_preflight_passes_with_fakes(tmp_path):
    r = await preflight(tmp_path)
    assert r.passed and "hello" in r.agent_heard_text.lower()
    assert r.heard_by_caller and r.greeting_latency_ms is not None
    out = format_preflight(r)
    assert "heard by agent:   yes" in out
    assert "heard by caller:  yes" in out
    assert "first audio from agent:" in out and "Try another voice" not in out


async def test_preflight_fails_when_agent_never_hears(tmp_path):
    r = await preflight(tmp_path, behaviours={"deaf"})
    assert not r.passed and not r.heard_by_agent and r.agent_heard_text is None
    out = format_preflight(r)
    assert "heard by agent:   no" in out
    assert "did not register the caller's voice" in out


async def test_preflight_needs_two_whole_words(tmp_path):
    r = await preflight(tmp_path, script=["Hello there, anyone?"])
    assert not r.heard_by_agent and r.heard_by_caller


def test_format_reports_a_caller_that_did_not_hear():
    r = PreflightResult(
        heard_by_agent=True,
        agent_heard_text="hello testing",
        heard_by_caller=False,
        caller_heard_text=None,
        greeting_latency_ms=None,
        passed=False,
    )
    out = format_preflight(r)
    assert "heard by caller:  no" in out and "first audio from agent: n/a" in out
    assert out.endswith(
        "The caller did not register the agent's voice. "
        "Check the caller adapter and its voice settings."
    )
