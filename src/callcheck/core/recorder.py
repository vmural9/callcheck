"""The run recorder: a call written to disk.

``Recorder`` is the ``Sink`` the call mixer writes to. It turns a call into a run
directory holding the event log, the stereo recording, the outcome and a merged
transcript. ``build_transcript`` reconciles what each side said with what the
other side heard, and ``render_transcript_md`` lays the result out for a reader.
"""

import json
import math
from pathlib import Path

from callcheck.core.audio import StereoWavWriter
from callcheck.core.events import CallEvent, event_to_json

_HEARD_LOOKBACK_MS = 2000
_INLINE_KINDS = frozenset({"nudge", "watchdog", "interrupted", "error"})


def _unique_dir(base: Path) -> Path:
    """Create ``base``, or ``base-2``, ``base-3``, ... if taken, and return the new directory."""
    candidate = base
    n = 1
    while True:
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            n += 1
            candidate = base.with_name(f"{base.name}-{n}")
        else:
            return candidate


class Recorder:
    """Persists a call's events, stereo audio, outcome and transcript under one run directory."""

    def __init__(self, run_dir: Path, rate: int = 24000) -> None:
        """Create a fresh run directory (suffixed ``-2``, ``-3`` if taken) and open its logs."""
        self.run_dir = _unique_dir(run_dir)
        self._events: list[CallEvent] = []
        self._log = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
        self._wav = StereoWavWriter(self.run_dir / "call.wav", rate)
        self._closed = False

    def record(self, ev: CallEvent) -> None:
        """Append the event to the log at once, so a crash mid-call leaves a usable record."""
        self._events.append(ev)
        self._log.write(event_to_json(ev) + "\n")
        self._log.flush()

    def write_audio(self, left: bytes, right: bytes) -> None:
        """Append one frame of caller (left) and agent (right) audio to the recording."""
        self._wav.write(left, right)

    def close(self, outcome: dict) -> None:
        """Finish the recording and write the outcome and transcripts; later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        self._log.close()
        self._wav.close()
        (self.run_dir / "outcome.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        transcript = build_transcript(self._events)
        lines = (json.dumps(u) for u in transcript)
        (self.run_dir / "transcript.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
        (self.run_dir / "transcript.md").write_text(
            render_transcript_md(transcript, self._events), encoding="utf-8"
        )


def read_events(run_dir: Path) -> list[CallEvent]:
    """Read ``events.jsonl`` in ``run_dir`` back into events, reversing ``event_to_json``."""
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    rows = (json.loads(line) for line in text.splitlines() if line.strip())
    return [CallEvent(row["t_ms"], row["side"], row["kind"], row["payload"]) for row in rows]


def _final_transcripts(events: list[CallEvent]) -> list[CallEvent]:
    """Return the final ``transcript`` events in time order."""
    found = [ev for ev in events if ev.kind == "transcript" and ev.payload.get("final", True)]
    return sorted(found, key=lambda ev: ev.t_ms)


def _utterance(ev: CallEvent, side: str, heard_as: str | None, source: str) -> dict:
    """Build one transcript entry."""
    return {
        "t_ms": ev.t_ms,
        "side": side,
        "text": ev.payload.get("text", ""),
        "final": True,
        "heard_as": heard_as,
        "source": source,
    }


def build_transcript(events: list[CallEvent]) -> list[dict]:
    """Merge each side's own transcripts with what the other side heard.

    A side's own final transcripts are its canonical utterances, each carrying as
    ``heard_as`` what the other side transcribed of it. A side with no own
    transcripts is represented by what the other side heard instead.
    """
    finals = _final_transcripts(events)
    out: list[dict] = []
    for side, other in (("caller", "agent"), ("agent", "caller")):
        own = [ev for ev in finals if ev.side == side and ev.payload.get("role") == side]
        heard = [ev for ev in finals if ev.side == other and ev.payload.get("role") == side]
        if not own:
            out.extend(_utterance(ev, side, None, "heard") for ev in heard)
            continue
        for i, ev in enumerate(own):
            end = own[i + 1].t_ms if i + 1 < len(own) else math.inf
            window = [
                h.payload.get("text", "")
                for h in heard
                if ev.t_ms - _HEARD_LOOKBACK_MS <= h.t_ms < end
            ]
            out.append(_utterance(ev, side, " ".join(window) if window else None, "self"))
    return sorted(out, key=lambda u: u["t_ms"])


def _stamp(t_ms: float) -> str:
    """Format milliseconds as ``[mm:ss.mmm]``."""
    total = int(t_ms)
    minutes, rest = divmod(total, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"[{minutes:02d}:{seconds:02d}.{millis:03d}]"


def render_transcript_md(transcript: list[dict], events: list[CallEvent]) -> str:
    """Render the transcript as time-sorted lines, with nudges, stalls and errors inline."""
    rows: list[tuple[float, str]] = [
        (u["t_ms"], f"{_stamp(u['t_ms'])} {u['side'].upper()}: {u['text']}") for u in transcript
    ]
    for ev in events:
        if ev.kind in _INLINE_KINDS:
            detail = (
                ev.payload.get("reason")
                or ev.payload.get("message")
                or ev.payload.get("instruction")
                or ""
            )
            rows.append((ev.t_ms, f"{_stamp(ev.t_ms)} -- {ev.kind}: {detail}"))
    rows.sort(key=lambda row: row[0])
    return "".join(f"{line}\n" for _, line in rows)
