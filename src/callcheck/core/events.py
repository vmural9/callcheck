"""The event vocabulary of a call.

Every component of the harness (caller, target, mixer, watchdog) reports what
happens as a ``CallEvent``. This module defines that record, the closed set of
event kinds, its one-line JSON form for the event log, and the ``Sink`` protocol
that receives events and audio for persistence.
"""

import json
from dataclasses import dataclass
from typing import Literal, Protocol

Side = Literal["caller", "agent", "harness"]

KINDS: frozenset[str] = frozenset(
    {
        "audio_out",
        "audio_start",
        "speech_start",
        "speech_end",
        "transcript",
        "turn_start",
        "turn_end",
        "heard_speech_start",
        "heard_speech_end",
        "interrupted",
        "tool_call",
        "tool_result",
        "result",
        "usage",
        "nudge",
        "done",
        "error",
        "closed",
        "watchdog",
        "call_start",
        "call_end",
    }
)


@dataclass(frozen=True)
class CallEvent:
    """One thing that happened on a call, stamped in milliseconds from call start."""

    t_ms: float
    side: Side
    kind: str
    payload: dict


def event_to_json(ev: CallEvent) -> str:
    """Serialise an event to a single JSON line, leaving out the raw ``pcm`` audio."""
    payload = {k: v for k, v in ev.payload.items() if k != "pcm"}
    return json.dumps({"t_ms": ev.t_ms, "side": ev.side, "kind": ev.kind, "payload": payload})


def endpoint_event(side: Side, kind: str, **payload: object) -> CallEvent:
    """Build an unstamped event (``t_ms`` 0.0); the ``Call`` restamps it on arrival."""
    return CallEvent(0.0, side, kind, dict(payload))


class Sink(Protocol):
    """Destination for a call's events, stereo audio and final outcome."""

    def record(self, ev: CallEvent) -> None:
        """Persist one event."""
        ...

    def write_audio(self, left: bytes, right: bytes) -> None:
        """Append one frame of caller (left) and agent (right) PCM16 audio."""
        ...

    def close(self, outcome: dict) -> None:
        """Finish the recording with the call's outcome."""
        ...
