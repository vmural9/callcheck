"""Latency and turn metrics for a finished call.

``compute_metrics`` reduces a call's event log to the numbers a founder asks
about first: how long the agent took to greet, how long it took to start
answering each caller turn, and how many turns each side took. ``write_metrics``
does the same for a run directory and saves ``metrics.json`` beside the other
run files.
"""

import json
import math
from pathlib import Path

from pydantic import BaseModel

from callcheck.core.cost import Cost, cost_of
from callcheck.core.events import CallEvent
from callcheck.core.recorder import read_events

Interval = tuple[float, float]


class Metrics(BaseModel):
    """Latency and turn counts for one call; times are in milliseconds."""

    greeting_latency_ms: float | None
    ttfa_ms: list[float]
    ttfa_p50_ms: float | None
    ttfa_p90_ms: float | None = None
    ttfa_p99_ms: float | None = None
    ttfa_max_ms: float | None
    duration_ms: float
    caller_turns: int
    agent_turns: int
    cost: Cost | None = None


def percentile(values: list[float], p: float) -> float | None:
    """Return the nearest-rank ``p``th percentile of ``values``, or ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _greeting_latency(events: list[CallEvent]) -> float | None:
    """Return the time of the first agent audio, or ``None`` if the agent never spoke."""
    starts = [ev.t_ms for ev in events if ev.side == "agent" and ev.kind == "audio_start"]
    return min(starts) if starts else None


def agent_intervals(events: list[CallEvent], horizon: float) -> list[Interval]:
    """Return the spans of agent speech.

    A span opens on the agent's ``audio_start`` or ``speech_start`` (whichever comes
    first) and closes on its ``speech_end`` or ``turn_end``. Bursts of audio inside
    one utterance therefore make one span.
    """
    spans: list[Interval] = []
    start: float | None = None
    for ev in events:
        if ev.side != "agent":
            continue
        if ev.kind in ("audio_start", "speech_start") and start is None:
            start = ev.t_ms
        elif ev.kind in ("speech_end", "turn_end") and start is not None:
            spans.append((start, max(start, ev.t_ms)))
            start = None
    if start is not None:
        spans.append((start, max(start, horizon)))
    return spans


def is_open(spans: list[Interval], t_ms: float) -> bool:
    """Say whether ``t_ms`` lies strictly inside one of ``spans``."""
    return any(start < t_ms < end for start, end in spans)


def ttfa_gaps(events: list[CallEvent]) -> list[tuple[float, float]]:
    """Measure, for each caller turn, the gap from its end to the agent's next audio.

    Returns ``(speech_end_t_ms, gap_ms)`` pairs. A turn is skipped when the caller
    started speaking again before the agent answered, since that gap measures the
    caller's pause rather than the agent. It is also skipped when the agent was already
    speaking as the caller's speech ended (the agent talked over the caller), so there was
    no wait for an answer.
    """
    ordered = sorted(events, key=lambda ev: ev.t_ms)
    agent_spans = agent_intervals(ordered, ordered[-1].t_ms if ordered else 0.0)
    gaps: list[tuple[float, float]] = []
    for i, ev in enumerate(ordered):
        if ev.side != "caller" or ev.kind != "speech_end" or is_open(agent_spans, ev.t_ms):
            continue
        for later in ordered[i + 1 :]:
            if later.side == "caller" and later.kind == "speech_start":
                break
            if later.side == "agent" and later.kind == "audio_start":
                gaps.append((ev.t_ms, later.t_ms - ev.t_ms))
                break
    return gaps


def _time_to_first_audio(events: list[CallEvent]) -> list[float]:
    """Return just the gap lengths of ``ttfa_gaps``, in milliseconds."""
    return [gap for _, gap in ttfa_gaps(events)]


def _turns(events: list[CallEvent], side: str) -> int:
    """Count the turns ``side`` began, as its ``speech_start`` events."""
    return sum(1 for ev in events if ev.side == side and ev.kind == "speech_start")


def compute_metrics(events: list[CallEvent], outcome: dict) -> Metrics:
    """Compute a call's latency and turn metrics from its events and recorded outcome."""
    gaps = _time_to_first_audio(events)
    return Metrics(
        greeting_latency_ms=_greeting_latency(events),
        ttfa_ms=gaps,
        ttfa_p50_ms=percentile(gaps, 50),
        ttfa_p90_ms=percentile(gaps, 90),
        ttfa_p99_ms=percentile(gaps, 99),
        ttfa_max_ms=max(gaps) if gaps else None,
        duration_ms=outcome["duration_ms"],
        caller_turns=_turns(events, "caller"),
        agent_turns=_turns(events, "agent"),
    )


def write_metrics(
    run_dir: Path,
    pricing: dict | None = None,
    *,
    agent_model: str = "nova-sonic",
    caller_model: str = "nova-sonic",
) -> Metrics:
    """Compute metrics from the run's files and write ``metrics.json``.

    With ``pricing``, the call is also priced from its ``usage`` events and the
    usage listed in ``judge.json`` (if the judge has run).
    """
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
    events = read_events(run_dir)
    metrics = compute_metrics(events, outcome)
    if pricing is not None:
        judge_path = run_dir / "judge.json"
        judge_usage = []
        if judge_path.exists():
            judge_usage = json.loads(judge_path.read_text(encoding="utf-8")).get("usage", [])
        metrics.cost = cost_of(
            events, judge_usage, pricing, agent_model=agent_model, caller_model=caller_model
        )
    (run_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    return metrics
