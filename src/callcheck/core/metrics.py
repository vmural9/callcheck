"""Latency and turn metrics for a finished call.

``compute_metrics`` reduces a call's event log to the numbers a founder asks
about first: how long the agent took to greet, how long it took to start
answering each caller turn, and how many turns each side took. ``write_metrics``
does the same for a run directory and saves ``metrics.json`` beside the other
run files.
"""

import json
import statistics
from pathlib import Path

from pydantic import BaseModel

from callcheck.core.events import CallEvent
from callcheck.core.recorder import read_events


class Metrics(BaseModel):
    """Latency and turn counts for one call; times are in milliseconds."""

    greeting_latency_ms: float | None
    ttfa_ms: list[float]
    ttfa_p50_ms: float | None
    ttfa_max_ms: float | None
    duration_ms: float
    caller_turns: int
    agent_turns: int
    cost_usd: None = None


def _greeting_latency(events: list[CallEvent]) -> float | None:
    """Return the time of the first agent audio, or ``None`` if the agent never spoke."""
    starts = [ev.t_ms for ev in events if ev.side == "agent" and ev.kind == "audio_start"]
    return min(starts) if starts else None


def _time_to_first_audio(events: list[CallEvent]) -> list[float]:
    """Measure, for each caller turn, the gap from its end to the agent's next audio.

    A turn is skipped when the caller started speaking again before the agent
    answered, since that gap measures the caller's pause rather than the agent.
    """
    ordered = sorted(events, key=lambda ev: ev.t_ms)
    gaps: list[float] = []
    for i, ev in enumerate(ordered):
        if ev.side != "caller" or ev.kind != "speech_end":
            continue
        for later in ordered[i + 1 :]:
            if later.side == "caller" and later.kind == "speech_start":
                break
            if later.side == "agent" and later.kind == "audio_start":
                gaps.append(later.t_ms - ev.t_ms)
                break
    return gaps


def _turns(events: list[CallEvent], side: str) -> int:
    """Count the turns ``side`` began, as its ``speech_start`` events."""
    return sum(1 for ev in events if ev.side == side and ev.kind == "speech_start")


def compute_metrics(events: list[CallEvent], outcome: dict) -> Metrics:
    """Compute a call's latency and turn metrics from its events and recorded outcome."""
    gaps = _time_to_first_audio(events)
    return Metrics(
        greeting_latency_ms=_greeting_latency(events),
        ttfa_ms=gaps,
        ttfa_p50_ms=statistics.median(gaps) if gaps else None,
        ttfa_max_ms=max(gaps) if gaps else None,
        duration_ms=outcome["duration_ms"],
        caller_turns=_turns(events, "caller"),
        agent_turns=_turns(events, "agent"),
    )


def write_metrics(run_dir: Path) -> Metrics:
    """Compute metrics from ``events.jsonl`` and ``outcome.json`` and write ``metrics.json``."""
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
    metrics = compute_metrics(read_events(run_dir), outcome)
    (run_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics
