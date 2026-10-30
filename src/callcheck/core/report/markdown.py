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
from callcheck.core.report.format import PARTIAL_NOTE, UNVERIFIED_NOTE, cell, ms, usd

_FILES = ("transcript.md", "call.wav", "judge.json", "events.jsonl")


def _verdict_section(judge: JudgeResult, outcome: dict) -> list[str]:
    """Render the verdict, failure types, summary and the reason the call ended."""
    failures = ", ".join(judge.failure_types) if judge.failure_types else "none"
    lines = [
        "## Verdict",
        "",
        f"- Verdict: **{judge.verdict}**",
        f"- Failure types: {failures}",
        f"- Summary: {cell(judge.summary)}",
        f"- Call ended: {outcome.get('reason', 'unknown')}",
    ]
    if outcome.get("error"):
        lines.append(f"- Call error: {cell(outcome['error'])}")
    return lines


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
        lines.append("| " + " | ".join(cell(c) for c in cells) + " |")
    return lines


def _quality_section(judge: JudgeResult) -> list[str]:
    """Render the five quality scores, or why they could not be determined."""
    lines = ["## Conversation quality", ""]
    if judge.quality is None:
        lines.append(f"undetermined: {'; '.join(judge.undetermined_reasons)}")
        return lines
    lines += ["| dimension | score | reason |", "| --- | --- | --- |"]
    for name, item in judge.quality:
        lines.append(f"| {name} | {item.score} | {cell(item.reason)} |")
    return lines


def _latency_section(metrics: Metrics) -> list[str]:
    """Render greeting latency, time to first audio and how many turns were measured."""
    return [
        "## Latency",
        "",
        f"- Greeting latency: {ms(metrics.greeting_latency_ms)}",
        f"- TTFA p50: {ms(metrics.ttfa_p50_ms)}",
        f"- TTFA max: {ms(metrics.ttfa_max_ms)}",
        f"- Agent turns measured: {len(metrics.ttfa_ms)}",
    ]


def _incidents_section(judge: JudgeResult) -> list[str]:
    """Render the turn-taking incidents the detector found, or ``none``."""
    lines = ["## Turn-taking incidents", ""]
    if not judge.incidents:
        return [*lines, "none"]
    lines += [
        "| time | label | attribution | confidence | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for inc in judge.incidents:
        cells = (f"{inc.t_ms / 1000:.1f} s", inc.label, inc.attribution, inc.confidence, inc.reason)
        lines.append("| " + " | ".join(cell(c) for c in cells) + " |")
    return lines


def _cost_section(metrics: Metrics) -> list[str]:
    """Render what the call cost per party, with footnotes for partial or unverified prices."""
    lines = ["## Cost", ""]
    cost = metrics.cost
    if cost is None:
        return [*lines, "n/a"]
    lines += ["| party | cost | note |", "| --- | --- | --- |"]
    for name in ("agent", "caller", "judge"):
        line = getattr(cost, name)
        lines.append(f"| {name} | {usd(line.usd, line.partial)} | {cell(line.note)} |")
    lines.append(f"| total | {usd(cost.total_usd, cost.partial)} | |")
    notes = []
    if cost.partial or any(getattr(cost, n).partial for n in ("agent", "caller", "judge")):
        notes.append(PARTIAL_NOTE)
    if any(getattr(cost, n).unverified for n in ("agent", "caller", "judge")):
        notes.append(UNVERIFIED_NOTE)
    return [*lines, "", *notes] if notes else lines


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
        _incidents_section(judge),
        _cost_section(metrics),
        ["## Files", "", *(f"- [{name}]({name})" for name in _FILES)],
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def write_run_report(run_dir: Path) -> Path:
    """Write ``report.md`` into ``run_dir`` and return its path."""
    path = run_dir / "report.md"
    path.write_text(render_run_report(run_dir), encoding="utf-8")
    return path
