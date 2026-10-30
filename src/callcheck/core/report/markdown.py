"""The single-run Markdown report.

``render_run_report`` turns a graded run directory (judge result, metrics and
outcome) into the ``report.md`` a founder reads first: the verdict and why, each
field checked, conversation quality, latency, cost, and links to the evidence
files. The output is GitHub-flavoured Markdown so it reads cleanly when pasted
into an issue or pull request.
"""

import json
from pathlib import Path

from callcheck.core.judge.schema import JudgeResult
from callcheck.core.metrics import Metrics

_FILES = ("transcript.md", "call.wav", "judge.json", "events.jsonl")


def _cell(value: object) -> str:
    """Format a value for a table cell, escaping pipes and flattening line breaks."""
    text = "" if value is None else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def _ms(value: float | None) -> str:
    """Format milliseconds with no decimals, or ``n/a`` when there is no value."""
    return "n/a" if value is None else f"{round(value)} ms"


def _verdict_section(judge: JudgeResult, outcome: dict) -> list[str]:
    """Render the verdict, failure types, summary and the reason the call ended."""
    failures = ", ".join(judge.failure_types) if judge.failure_types else "none"
    return [
        "## Verdict",
        "",
        f"- Verdict: **{judge.verdict}**",
        f"- Failure types: {failures}",
        f"- Summary: {_cell(judge.summary)}",
        f"- Call ended: {outcome.get('reason', 'unknown')}",
    ]


def _fields_section(judge: JudgeResult) -> list[str]:
    """Render one table row per field the judge compared."""
    lines = [
        "## Fields",
        "",
        "| field | expected | captured | verdict | method | note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for f in judge.fields:
        cells = (f.name, f.expected, f.captured, f.verdict, f.method, f.note)
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return lines


def _quality_section(judge: JudgeResult) -> list[str]:
    """Render the five quality scores, or why they could not be determined."""
    lines = ["## Conversation quality", ""]
    if judge.quality is None:
        lines.append(f"undetermined: {'; '.join(judge.undetermined_reasons)}")
        return lines
    lines += ["| dimension | score | reason |", "| --- | --- | --- |"]
    for name, item in judge.quality:
        lines.append(f"| {name} | {item.score} | {_cell(item.reason)} |")
    return lines


def _latency_section(metrics: Metrics) -> list[str]:
    """Render greeting latency, time to first audio and how many turns were measured."""
    return [
        "## Latency",
        "",
        f"- Greeting latency: {_ms(metrics.greeting_latency_ms)}",
        f"- TTFA p50: {_ms(metrics.ttfa_p50_ms)}",
        f"- TTFA max: {_ms(metrics.ttfa_max_ms)}",
        f"- Agent turns measured: {len(metrics.ttfa_ms)}",
    ]


def render_run_report(run_dir: Path) -> str:
    """Render ``run_dir``'s judge result, metrics and outcome as a Markdown report."""
    judge = JudgeResult.model_validate_json((run_dir / "judge.json").read_text(encoding="utf-8"))
    metrics = Metrics.model_validate_json((run_dir / "metrics.json").read_text(encoding="utf-8"))
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
    sections = [
        ["# callcheck run report"],
        _verdict_section(judge, outcome),
        _fields_section(judge),
        _quality_section(judge),
        _latency_section(metrics),
        ["## Cost", "", "n/a until M2"],
        ["## Files", "", *(f"- [{name}]({name})" for name in _FILES)],
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def write_run_report(run_dir: Path) -> Path:
    """Write ``report.md`` into ``run_dir`` and return its path."""
    path = run_dir / "report.md"
    path.write_text(render_run_report(run_dir), encoding="utf-8")
    return path
