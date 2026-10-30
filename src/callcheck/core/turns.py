"""Offline turn-taking detector.

``detect_incidents`` reads a finished call's event log and labels the moments
where the conversation went wrong: the agent not answering, the caller not
being heard, silences, interruptions, talking over each other, slow answers
and transport failures. Each ``Incident`` says who is to blame, how sure the
rule is, and whether it ended the call. ``write_incidents`` saves them as
``incidents.json`` beside the other run files.

Timestamps are milliseconds; the ``CallConfig`` timers are seconds.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from callcheck.core.events import CallEvent
from callcheck.core.metrics import ttfa_gaps
from callcheck.core.scenario import CallConfig

Label = Literal[
    "no_response",
    "caller_not_heard",
    "dead_end",
    "caller_stall",
    "agent_interrupts",
    "turn_edge",
    "caller_interrupts",
    "double_talk",
    "slow_response",
    "transport_error",
]
Attribution = Literal["agent", "simulator", "unclear", "expected"]
Confidence = Literal["high", "medium", "low"]

# An agent interruption is a turn edge, not a fault, when the caller paused longer
# than this inside the window just before the agent began.
_EDGE_GAP_MS = 400.0
_EDGE_WINDOW_MS = 2000.0
# How long after an agent turn_end its final transcript may still arrive.
_ASR_LAG_MS = 1500.0

_NUDGE_LABELS: tuple[Label, ...] = ("no_response", "dead_end", "caller_stall", "caller_not_heard")
_NUDGE_ENDINGS = ("max_nudges", "nudge_refused")
_NUDGE_ATTRIBUTION: dict[str, Attribution] = {
    "no_response": "agent",
    "dead_end": "agent",
    "caller_stall": "simulator",
    "caller_not_heard": "simulator",
}

Interval = tuple[float, float]


class Incident(BaseModel):
    """One turn-taking fault, with who caused it and how sure the rule is."""

    t_ms: float
    label: Label
    attribution: Attribution
    confidence: Confidence
    reason: str
    ends_call: bool = False


def _caller_intervals(events: list[CallEvent], horizon: float) -> list[Interval]:
    """Return the spans between the caller's ``speech_start`` and the next ``speech_end``."""
    spans: list[Interval] = []
    start: float | None = None
    for ev in events:
        if ev.side != "caller":
            continue
        if ev.kind == "speech_start" and start is None:
            start = ev.t_ms
        elif ev.kind == "speech_end" and start is not None:
            spans.append((start, max(start, ev.t_ms)))
            start = None
    if start is not None:
        spans.append((start, max(start, horizon)))
    return spans


def _agent_intervals(events: list[CallEvent], horizon: float) -> list[Interval]:
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


def _is_open(spans: list[Interval], t_ms: float) -> bool:
    """Say whether ``t_ms`` lies strictly inside one of ``spans``."""
    return any(start < t_ms < end for start, end in spans)


def _missed_turns(
    events: list[CallEvent], config: CallConfig, agent_spans: list[Interval]
) -> list[Incident]:
    """Find caller turns the agent did not start answering within ``nudge_after_s``."""
    window = config.nudge_after_s * 1000
    caller_starts = [e.t_ms for e in events if e.side == "caller" and e.kind == "speech_start"]
    call_ends = [e.t_ms for e in events if e.kind == "call_end"]
    found: list[Incident] = []
    for ev in events:
        if ev.side != "caller" or ev.kind != "speech_end":
            continue
        deadline = ev.t_ms + window
        if any(ev.t_ms < s < deadline for s in caller_starts):
            continue  # the caller resumed, so this was a pause, not the end of a turn
        if any(ev.t_ms < t < deadline for t in call_ends):
            continue  # the call ended before the agent's window had run out
        if _is_open(agent_spans, ev.t_ms) or any(
            e.side == "agent" and e.kind == "audio_start" and ev.t_ms <= e.t_ms <= deadline
            for e in events
        ):
            continue
        turn_start = max((s for s in caller_starts if s <= ev.t_ms), default=ev.t_ms)
        registered = any(
            e.side == "agent"
            and turn_start <= e.t_ms <= deadline
            and (
                e.kind in ("heard_speech_start", "heard_speech_end")
                or (e.kind == "transcript" and e.payload.get("role") == "caller")
            )
            for e in events
        )
        if registered:
            found.append(
                Incident(
                    t_ms=ev.t_ms,
                    label="no_response",
                    attribution="agent",
                    confidence="high",
                    reason=f"the agent heard the caller but did not speak within "
                    f"{config.nudge_after_s:g} s",
                )
            )
        else:
            found.append(
                Incident(
                    t_ms=ev.t_ms,
                    label="caller_not_heard",
                    attribution="simulator",
                    confidence="high",
                    reason=f"the agent never registered the caller and did not speak within "
                    f"{config.nudge_after_s:g} s",
                )
            )
    return found


def _agent_turn_ends(events: list[CallEvent]) -> list[tuple[float, float]]:
    """Return ``(turn_end_t_ms, previous_turn_end_t_ms)`` for each distinct agent turn end.

    A ``turn_end`` with no agent audio or speech since the previous one repeats it
    and is dropped. The first turn has no predecessor, so its previous time is -inf.
    """
    ends: list[tuple[float, float]] = []
    previous = float("-inf")
    spoke = True
    for ev in events:
        if ev.side != "agent":
            continue
        if ev.kind in ("audio_start", "speech_start"):
            spoke = True
        elif ev.kind == "turn_end" and spoke:
            ends.append((ev.t_ms, previous))
            previous, spoke = ev.t_ms, False
    return ends


def _silences(
    events: list[CallEvent], config: CallConfig, caller_spans: list[Interval], horizon: float
) -> list[Incident]:
    """Find silences of at least ``stall_s`` after an agent ``turn_end``.

    The silence starts when the agent has stopped and any caller speech already
    under way has finished.
    """
    found: list[Incident] = []
    for turn_end, previous_end in _agent_turn_ends(events):
        start = max([turn_end, *(end for s, end in caller_spans if s < turn_end < end)])
        end = min(
            (
                e.t_ms
                for e in events
                if e.t_ms > start
                and (e.kind == "speech_start" or (e.side == "agent" and e.kind == "audio_start"))
                and e.side != "harness"
            ),
            default=horizon,
        )
        if end - start < config.stall_s * 1000:
            continue
        spoken = [
            e
            for e in events
            if e.side == "agent"
            and e.kind == "transcript"
            and e.payload.get("role") == "agent"
            and e.payload.get("final") is True
            and previous_end < e.t_ms <= turn_end + _ASR_LAG_MS
        ]
        silent_s = (end - start) / 1000
        if not spoken:
            found.append(
                Incident(
                    t_ms=start,
                    label="dead_end",
                    attribution="unclear",
                    confidence="low",
                    reason=f"both sides were silent for {silent_s:.1f} s; the agent's last "
                    "words were not transcribed, so whether it asked a question is unknown",
                )
            )
        elif str(spoken[-1].payload.get("text", "")).rstrip().endswith("?"):
            found.append(
                Incident(
                    t_ms=start,
                    label="caller_stall",
                    attribution="simulator",
                    confidence="high",
                    reason=f"the agent asked; the caller was silent for {silent_s:.1f} s",
                )
            )
        else:
            found.append(
                Incident(
                    t_ms=start,
                    label="dead_end",
                    attribution="agent",
                    confidence="medium",
                    reason=f"both sides were silent for {silent_s:.1f} s after the agent finished",
                )
            )
    return found


def _overlaps(
    caller_spans: list[Interval],
    agent_spans: list[Interval],
    config: CallConfig,
    *,
    caller_interrupts_expected: bool,
) -> list[Incident]:
    """Find interruptions and talking over each other."""
    found: list[Incident] = []
    for a_start, _ in agent_spans:
        if not _is_open(caller_spans, a_start):
            continue
        near_edge = any(
            q_start - p_end > _EDGE_GAP_MS
            and p_end >= a_start - _EDGE_WINDOW_MS
            and q_start <= a_start
            for (_, p_end), (q_start, _) in zip(caller_spans, caller_spans[1:], strict=False)
        )
        if near_edge:
            found.append(
                Incident(
                    t_ms=a_start,
                    label="turn_edge",
                    attribution="unclear",
                    confidence="low",
                    reason="the agent began in a pause of more than 400 ms in the caller's speech",
                )
            )
        else:
            found.append(
                Incident(
                    t_ms=a_start,
                    label="agent_interrupts",
                    attribution="agent",
                    confidence="medium",
                    reason="the agent began speaking while the caller was mid-speech",
                )
            )
    for c_start, _ in caller_spans:
        if _is_open(agent_spans, c_start):
            found.append(
                Incident(
                    t_ms=c_start,
                    label="caller_interrupts",
                    attribution="expected" if caller_interrupts_expected else "simulator",
                    confidence="medium",
                    reason="the caller began speaking while the agent was mid-speech",
                )
            )
    for c_start, c_end in caller_spans:
        for a_start, a_end in agent_spans:
            start, end = max(c_start, a_start), min(c_end, a_end)
            if end - start <= config.double_talk_s * 1000:
                continue
            if c_start == a_start:
                who: Attribution = "unclear"
            else:
                who = "simulator" if c_start > a_start else "agent"
            found.append(
                Incident(
                    t_ms=start,
                    label="double_talk",
                    attribution=who,
                    confidence="low" if who == "unclear" else "medium",
                    reason=f"both sides spoke at once for {(end - start) / 1000:.1f} s",
                )
            )
    return found


def _slow_responses(
    events: list[CallEvent], config: CallConfig, agent_spans: list[Interval]
) -> list[Incident]:
    """Find caller turns the agent answered slower than ``slow_response_s``."""
    return [
        Incident(
            t_ms=t_ms,
            label="slow_response",
            attribution="agent",
            confidence="high",
            reason=f"the agent took {gap / 1000:.1f} s to start answering",
        )
        for t_ms, gap in ttfa_gaps(events)
        if gap > config.slow_response_s * 1000 and not _is_open(agent_spans, t_ms)
    ]


def _transport_errors(events: list[CallEvent]) -> list[Incident]:
    """Find harness errors marked unrecoverable."""
    return [
        Incident(
            t_ms=ev.t_ms,
            label="transport_error",
            attribution="simulator",
            confidence="high",
            reason=f"unrecoverable harness error: {ev.payload.get('message', 'unknown')}",
            ends_call=True,
        )
        for ev in events
        if ev.side == "harness" and ev.kind == "error" and ev.payload.get("recoverable") is False
    ]


def _apply_watchdog_endings(events: list[CallEvent], incidents: list[Incident]) -> list[Incident]:
    """Mark the incident behind each nudge-driven ending as the one that ended the call.

    The watchdog gives up after ``max_nudges`` or when the agent refuses a nudge. The
    cause is the latest incident before it carrying the payload's ``label`` (any
    silence label when the payload names none). With no such incident, one is built
    from the payload's ``label``.
    """
    incidents = sorted(incidents, key=lambda i: i.t_ms)
    for ev in events:
        if ev.side != "harness" or ev.kind != "watchdog":
            continue
        if ev.payload.get("reason") not in _NUDGE_ENDINGS:
            continue
        label = ev.payload.get("label")
        wanted = (label,) if label in _NUDGE_LABELS else _NUDGE_LABELS
        causes = [n for n, i in enumerate(incidents) if i.label in wanted and i.t_ms <= ev.t_ms]
        if causes:
            cause = causes[-1]
            incidents[cause] = incidents[cause].model_copy(update={"ends_call": True})
            continue
        known = label in _NUDGE_LABELS
        incidents.append(
            Incident(
                t_ms=ev.t_ms,
                label=label if known else "no_response",
                attribution=_NUDGE_ATTRIBUTION[label] if known else "unclear",
                confidence="medium" if known else "low",
                reason=f"the watchdog ended the call ({ev.payload.get('reason')})",
                ends_call=True,
            )
        )
        incidents.sort(key=lambda i: i.t_ms)
    return incidents


def detect_incidents(
    events: list[CallEvent], config: CallConfig, *, caller_interrupts_expected: bool
) -> list[Incident]:
    """Label the turn-taking faults in a call, sorted by time.

    ``caller_interrupts_expected`` is true for a persona that interrupts on purpose,
    so its interruptions are attributed to "expected" rather than the simulator.
    """
    ordered = sorted(events, key=lambda ev: ev.t_ms)
    horizon = ordered[-1].t_ms if ordered else 0.0
    caller_spans = _caller_intervals(ordered, horizon)
    agent_spans = _agent_intervals(ordered, horizon)
    incidents = [
        *_missed_turns(ordered, config, agent_spans),
        *_silences(ordered, config, caller_spans, horizon),
        *_overlaps(
            caller_spans,
            agent_spans,
            config,
            caller_interrupts_expected=caller_interrupts_expected,
        ),
        *_slow_responses(ordered, config, agent_spans),
        *_transport_errors(ordered),
    ]
    return _apply_watchdog_endings(ordered, incidents)


def write_incidents(run_dir: Path, incidents: list[Incident]) -> None:
    """Write ``incidents.json`` (a JSON list) into ``run_dir``."""
    rows = [incident.model_dump(mode="json") for incident in incidents]
    (run_dir / "incidents.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def decisive_incident(incidents: list[Incident]) -> Incident | None:
    """Return the last incident that ended the call, or ``None`` if none did."""
    for incident in reversed(incidents):
        if incident.ends_call:
            return incident
    return None
