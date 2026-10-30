"""Render a ``BatchSummary`` as the Markdown ``report.md`` (sections in spec order)."""

from callcheck.core.report.aggregate_model import BatchSummary
from callcheck.core.report.format import PARTIAL_NOTE, UNVERIFIED_NOTE, cell, ms, usd


def _link(s: BatchSummary, name: str | None) -> str:
    """Link a run's transcript if it has one, otherwise its directory."""
    if name is None:
        return "n/a"
    return f"[{name}]({name}/transcript.md)" if name in s.transcripts else f"[{name}]({name}/)"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _rate(matched: int, graded: int) -> str:
    if not graded:
        return "n/a"
    percent = (200 * matched + graded) // (2 * graded)  # half-up, not banker's rounding
    return f"{matched}/{graded} ({percent}%)"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _table(header: tuple[str, ...], rows: list[tuple], raw_col: int | None = None) -> list[str]:
    """Render a table; column ``raw_col`` holds ready-made Markdown (a link) and is not escaped."""
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for row in rows:
        cells = [str(c) if i == raw_col else cell(c) for i, c in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _summary(s: BatchSummary) -> list[str]:
    c = s.counts
    graded = c["pass"] + c["fail"]
    rate = _rate(c["pass"], graded)
    lines = [
        "## Summary",
        "",
        f"- Scenario: {cell(s.scenario) or 'n/a'}",
        f"- Target: {cell(s.target) or 'n/a'}",
        f"- Caller: {cell(s.caller) or 'n/a'}",
        f"- Judge: {cell(s.judge) or 'n/a'}",
        f"- Bugs: {cell(', '.join(s.bugs)) or 'none'}",
        f"- Runs: {s.n}",
        f"- Pass: {c['pass']}, fail: {c['fail']}, undetermined: {c['undetermined']}, "
        f"artifact: {c['artifact']}, errored: {c['errored']}",
        f"- Pass rate: {rate} (pass / (pass + fail); undetermined and artifact are not counted)",
        f"- Total cost: {usd(s.cost.total_usd, s.cost.total_partial)}",
        f"- Wall time: {_duration(s.wall_seconds)}",
    ]
    errored = [r for r in s.runs if r.verdict == "errored"]
    if errored:
        lines += ["", "Errored runs (no judge result, not counted in the pass rate):", ""]
        lines += [f"- {_link(s, r.name)}: {cell(r.error)}" for r in errored]
    return lines


def _failures(s: BatchSummary) -> list[str]:
    lines = ["## Failures by type", ""]
    if not s.failures:
        return [*lines, "none"]
    rows = [(f.type, f.count, _link(s, f.example)) for f in s.failures]
    return lines + _table(("type", "runs", "example"), rows, 2)


def _fields(s: BatchSummary) -> list[str]:
    lines = ["## Field accuracy", ""]
    if not s.fields:
        return [*lines, "n/a"]
    rows = [(f.name, _rate(f.matched, f.graded), f.most_common_wrong or "-") for f in s.fields]
    return lines + _table(("field", "match rate", "most common wrong value"), rows)


def _quality(s: BatchSummary) -> list[str]:
    lines = ["## Conversation quality", ""]
    if not s.quality:
        return [*lines, "n/a"]
    rows = [
        (q.dimension, _num(q.mean), f"{_link(s, q.lowest_run)} ({q.lowest_score})")
        for q in s.quality
    ]
    return lines + _table(("dimension", "mean", "lowest run"), rows, 2)


def _latency(s: BatchSummary) -> list[str]:
    lat = s.latency
    return [
        "## Latency",
        "",
        f"- Turns measured: {lat.turns}",
        f"- TTFA p50: {ms(lat.p50_ms)}",
        f"- TTFA p90: {ms(lat.p90_ms)}",
        f"- TTFA p99: {ms(lat.p99_ms)}",
        f"- TTFA max: {ms(lat.max_ms)} (slowest turn: {_link(s, lat.slowest_run)})",
    ]


def _incidents(s: BatchSummary) -> list[str]:
    lines = ["## Turn-taking incidents", ""]
    if not s.incidents:
        return [*lines, "none"]
    rows = [(i.label, i.attribution, i.count) for i in s.incidents]
    return lines + _table(("label", "attribution", "count"), rows)


def _worst(s: BatchSummary) -> list[str]:
    lines = ["## Worst three runs", ""]
    if not s.worst:
        return [*lines, "none"]
    rows = [(r.verdict, ", ".join(r.failure_types) or "none", _link(s, r.name)) for r in s.worst]
    return lines + _table(("verdict", "failure types", "run"), rows, 2)


def _cost(s: BatchSummary) -> list[str]:
    lines = ["## Cost", ""]
    rows = [(r.party, usd(r.total_usd, r.partial), usd(r.mean_usd, r.partial)) for r in s.cost.rows]
    rows.append(("total", usd(s.cost.total_usd, s.cost.total_partial), "-"))
    lines += _table(("party", "total", "mean per run"), rows)
    notes = []
    if s.cost.total_partial or any(r.partial for r in s.cost.rows):
        notes.append(PARTIAL_NOTE)
    if s.cost.unverified:
        notes.append(UNVERIFIED_NOTE)
    if s.cost.retried_attempts:
        n = s.cost.retried_attempts
        notes.append(f"Includes {n} retried attempt{'' if n == 1 else 's'}.")
    return [*lines, "", *notes] if notes else lines


def render_batch_report(summary: BatchSummary) -> str:
    """Render ``summary`` as the batch's ``report.md``."""
    sections = [
        ["# callcheck batch report"],
        _summary(summary),
        _failures(summary),
        _fields(summary),
        _quality(summary),
        _latency(summary),
        _incidents(summary),
        _worst(summary),
        _cost(summary),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
