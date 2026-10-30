"""The preflight call: does the agent hear the caller, and does the caller hear the agent?

Some speech models fail to recognise synthetic voices, which would make a whole test run
meaningless. Before a run, ``run_preflight`` places one short call in which the caller
says a fixed test phrase, then reads the recorded events back to see what each side
transcribed of the other. ``format_preflight`` renders the verdict for the terminal.
"""

import re
from pathlib import Path

from pydantic import BaseModel

from callcheck.core.call import Call, CallSettings
from callcheck.core.clock import Clock
from callcheck.core.endpoints import Caller, TargetAgent
from callcheck.core.events import CallEvent
from callcheck.core.recorder import Recorder, read_events

PREFLIGHT_WORDS = ("hello", "hear", "testing", "one", "two", "three")
MIN_WORDS_HEARD = 2
_WORD = re.compile(r"[^\W_]+")
_AGENT_DEAF = (
    "The agent did not register the caller's voice. Try another voice (--caller-voice), "
    "another caller (--caller openai-realtime, --caller pipeline), or check the agent's "
    "VAD settings."
)
_CALLER_DEAF = (
    "The caller did not register the agent's voice. Check the caller adapter and its "
    "voice settings."
)


class PreflightResult(BaseModel):
    """What each side heard of the other on the preflight call, and the overall verdict."""

    heard_by_agent: bool
    agent_heard_text: str | None
    heard_by_caller: bool
    caller_heard_text: str | None
    greeting_latency_ms: float | None
    passed: bool
    agent_heard_at_ms: float | None = None
    caller_heard_at_ms: float | None = None


async def run_preflight(
    *, target: TargetAgent, caller: Caller, clock: Clock, run_dir: Path
) -> PreflightResult:
    """Place a short call, record it under ``run_dir`` and report what each side heard."""
    recorder = Recorder(run_dir)
    call = Call(
        caller=caller,
        target=target,
        sink=recorder,
        clock=clock,
        settings=CallSettings(max_seconds=20, stall_s=5),
    )
    outcome = await call.run()
    recorder.close(outcome.as_dict())
    return _judge(read_events(recorder.run_dir))


def format_preflight(r: PreflightResult) -> str:
    """Render the result as three report lines, plus a hint on what to try when it failed."""
    latency = "n/a" if r.greeting_latency_ms is None else f"{r.greeting_latency_ms:.0f} ms"
    lines = [
        _heard_line("heard by agent:  ", r.heard_by_agent, r.agent_heard_text, r.agent_heard_at_ms),
        _heard_line(
            "heard by caller: ", r.heard_by_caller, r.caller_heard_text, r.caller_heard_at_ms
        ),
        f"first audio from agent: {latency}",
    ]
    if not r.passed:
        lines += ["", _AGENT_DEAF if not r.heard_by_agent else _CALLER_DEAF]
    return "\n".join(lines)


def _heard_line(label: str, heard: bool, text: str | None, at_ms: float | None) -> str:
    """Format one "heard by" line: yes with the words and when, or no with empty quotes."""
    if not heard:
        return f'{label} no   ""'
    when = "" if at_ms is None else f"   ({at_ms / 1000:.1f} s)"
    return f'{label} yes   "{text}"{when}'


def _judge(events: list[CallEvent]) -> PreflightResult:
    """Decide from the recorded events what each side heard and how fast the agent spoke."""
    agent_ev = _first_transcript(events, side="agent", role="caller")
    caller_ev = _first_transcript(events, side="caller", role="agent")
    starts = [ev.t_ms for ev in events if ev.side == "agent" and ev.kind == "audio_start"]
    heard_by_agent = any(
        _is_test_phrase(ev.payload.get("text", ""))
        for ev in events
        if _is_transcript(ev, "agent", "caller")
    )
    heard_by_caller = any(
        bool(str(ev.payload.get("text", "")).strip())
        for ev in events
        if _is_transcript(ev, "caller", "agent")
    )
    return PreflightResult(
        heard_by_agent=heard_by_agent,
        agent_heard_text=_text(agent_ev),
        heard_by_caller=heard_by_caller,
        caller_heard_text=_text(caller_ev),
        greeting_latency_ms=min(starts) if starts else None,
        passed=heard_by_agent and heard_by_caller,
        agent_heard_at_ms=agent_ev.t_ms if agent_ev else None,
        caller_heard_at_ms=caller_ev.t_ms if caller_ev else None,
    )


def _is_transcript(ev: CallEvent, side: str, role: str) -> bool:
    """Whether ``ev`` is a transcript reported by ``side`` of what ``role`` said."""
    return ev.kind == "transcript" and ev.side == side and ev.payload.get("role") == role


def _first_transcript(events: list[CallEvent], *, side: str, role: str) -> CallEvent | None:
    """Return the earliest matching transcript event, or None."""
    return next((ev for ev in events if _is_transcript(ev, side, role)), None)


def _text(ev: CallEvent | None) -> str | None:
    """Return a transcript event's text, or None when there is no event."""
    return None if ev is None else str(ev.payload.get("text", ""))


def _is_test_phrase(text: str) -> bool:
    """Whether at least two distinct preflight words appear as whole words in ``text``."""
    words = set(_WORD.findall(str(text).casefold()))
    return len(words.intersection(PREFLIGHT_WORDS)) >= MIN_WORDS_HEARD
