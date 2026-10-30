"""Reduce a batch directory of graded runs to ``report.json`` and ``report.md``.

``summarize_batch`` reads each run's ``judge.json``, ``metrics.json``,
``outcome.json`` and ``incidents.json`` and gathers them into a ``BatchSummary``.
A run directory without a readable ``judge.json`` counts as errored: it is listed
with its error and kept out of the pass rate. ``preflight/`` and ``*.attempt1``
directories (a retried first attempt) are not runs, but their cost is in the batch totals.
Rendering lives in ``aggregate_render``.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ValidationError

from callcheck.core.cost import Cost
from callcheck.core.judge.schema import JudgeResult
from callcheck.core.metrics import Metrics, percentile
from callcheck.core.report.aggregate_model import (
    BatchSummary,
    CostRow,
    CostSummary,
    FailureRow,
    FieldRow,
    IncidentRow,
    LatencySummary,
    QualityRow,
    RunRow,
)
from callcheck.core.report.aggregate_render import render_batch_report
from callcheck.core.turns import Incident

__all__ = ["BatchSummary", "render_batch_report", "summarize_batch", "write_batch_report"]

_PARTIES = ("agent", "caller", "judge")
# Worst first: a failed call, then one nobody could grade, then one the harness spoiled.
_VERDICT_RANK = {"fail": 0, "undetermined": 1, "artifact": 2, "pass": 3}


class _Run(BaseModel):
    """One run directory's files, loaded."""

    name: str
    judge: JudgeResult | None
    metrics: Metrics | None
    incidents: list[Incident]
    error: str | None
    outcome_reason: str | None = None


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_judge(path: Path) -> tuple[JudgeResult | None, str | None]:
    """Return the judge result, or why there is none (a missing file is not a problem)."""
    try:
        return JudgeResult.model_validate_json(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValidationError) as exc:
        return None, f"judge.json unreadable: {' '.join(str(exc).split())}"


def _read_metrics(path: Path) -> Metrics | None:
    try:
        return Metrics.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def _read_incidents(path: Path) -> list[Incident] | None:
    """Return the incidents in ``path``, or ``None`` if it is missing or malformed."""
    raw = _read_json(path)
    if not isinstance(raw, list):
        return None
    try:
        return [Incident.model_validate(item) for item in raw]
    except ValidationError:
        return None


def _load_run(run_dir: Path) -> _Run:
    judge, unreadable = _read_judge(run_dir / "judge.json")
    outcome = _read_json(run_dir / "outcome.json")
    outcome = outcome if isinstance(outcome, dict) else {}
    incidents = _read_incidents(run_dir / "incidents.json")
    error = None
    if judge is None:
        error = unreadable or outcome.get("error") or "no judge.json"
    return _Run(
        name=run_dir.name,
        judge=judge,
        metrics=_read_metrics(run_dir / "metrics.json"),
        incidents=incidents if incidents is not None else (judge.incidents if judge else []),
        error=error,
        outcome_reason=outcome.get("reason"),
    )


def _run_dirs(batch_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in batch_dir.iterdir()
        if p.is_dir() and p.name != "preflight" and not p.name.endswith(".attempt1")
    )


def _quality_mean(run: _Run) -> float | None:
    if run.judge is None or run.judge.quality is None:
        return None
    return mean(item.score for _, item in run.judge.quality)


def _run_row(run: _Run) -> RunRow:
    if run.judge is None:
        return RunRow(name=run.name, verdict="errored", error=run.error)
    return RunRow(
        name=run.name,
        verdict=run.judge.verdict,
        failure_types=run.judge.failure_types,
        quality_mean=_quality_mean(run),
    )


def _failures(judged: list[_Run]) -> list[FailureRow]:
    counts: Counter[str] = Counter()
    example: dict[str, str] = {}
    for run in judged:
        for kind in run.judge.failure_types:
            counts[kind] += 1
            example.setdefault(kind, run.name)
    return [FailureRow(type=k, count=c, example=example[k]) for k, c in counts.most_common()]


def _fields(judged: list[_Run]) -> list[FieldRow]:
    """Tally field matches over judged runs; artifact and crashed runs are left out.

    An artifact call ended because of the simulator, and a crashed call (outcome ``error``)
    ended because of the harness, so what the agent captured in either says nothing about the
    agent. Their failure types still count under "Failures by type".
    """
    judged = [
        run for run in judged if run.judge.verdict != "artifact" and run.outcome_reason != "error"
    ]
    graded: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    wrong: dict[str, Counter[str]] = {}
    for run in judged:
        for f in run.judge.fields:
            wrong.setdefault(f.name, Counter())
            if f.verdict == "undetermined":
                continue
            graded[f.name] += 1
            if f.verdict == "match":
                matched[f.name] += 1
            else:
                wrong[f.name]["(nothing captured)" if f.captured is None else str(f.captured)] += 1
    return [
        FieldRow(
            name=name,
            graded=graded[name],
            matched=matched[name],
            most_common_wrong=counter.most_common(1)[0][0] if counter else None,
        )
        for name, counter in wrong.items()
    ]


def _quality(judged: list[_Run]) -> list[QualityRow]:
    scores: dict[str, list[tuple[int, str]]] = {}
    for run in judged:
        if run.judge.quality is not None:
            for name, item in run.judge.quality:
                scores.setdefault(name, []).append((item.score, run.name))
    return [
        QualityRow(
            dimension=name,
            mean=mean(s for s, _ in pairs),
            lowest_run=min(pairs)[1],
            lowest_score=min(pairs)[0],
        )
        for name, pairs in scores.items()
    ]


def _latency(runs: list[_Run]) -> LatencySummary:
    turns = [(gap, run.name) for run in runs if run.metrics for gap in run.metrics.ttfa_ms]
    gaps = [gap for gap, _ in turns]
    slowest = max(turns, key=lambda t: t[0])[1] if turns else None
    return LatencySummary(
        turns=len(gaps),
        p50_ms=percentile(gaps, 50),
        p90_ms=percentile(gaps, 90),
        p99_ms=percentile(gaps, 99),
        max_ms=max(gaps) if gaps else None,
        slowest_run=slowest,
    )


def _incidents(runs: list[_Run]) -> list[IncidentRow]:
    counts = Counter((i.label, i.attribution) for run in runs for i in run.incidents)
    return [
        IncidentRow(label=label, attribution=who, count=n)
        for (label, who), n in counts.most_common()
    ]


def _worst(rows: list[RunRow]) -> list[RunRow]:
    judged = [r for r in rows if r.verdict != "errored"]
    judged.sort(
        key=lambda r: (
            _VERDICT_RANK[r.verdict],
            -len(r.failure_types),
            r.quality_mean if r.quality_mean is not None else float("inf"),
            r.name,
        )
    )
    return judged[:3]


def _retried_costs(batch_dir: Path) -> list[Cost | None]:
    """Return the cost of each retried first attempt that has metrics (``None`` if unpriced).

    A retried attempt's directory is not a run, but its money was spent.
    """
    metrics = (_read_metrics(p / "metrics.json") for p in sorted(batch_dir.glob("*.attempt1")))
    return [m.cost for m in metrics if m is not None]


def _cost(runs: list[_Run], retried: list[Cost | None]) -> CostSummary:
    """Sum the priced cost lines; a run that lacks one makes that row partial, not n/a.

    Retried attempts add to the totals but not to the per-run means.
    """
    costs: list[Cost | None] = [run.metrics.cost if run.metrics else None for run in runs]
    rows = []
    for party in _PARTIES:
        lines = [getattr(c, party) if c else None for c in costs]
        priced = [line for line in lines if line is not None and line.usd is not None]
        extra = [getattr(c, party) if c else None for c in retried]
        extra_priced = [line for line in extra if line is not None and line.usd is not None]
        run_total = sum(line.usd for line in priced)
        total = run_total + sum(line.usd for line in extra_priced)
        rows.append(
            CostRow(
                party=party,
                total_usd=total if priced or extra_priced else None,
                mean_usd=run_total / len(priced) if priced else None,
                partial=len(priced) < len(lines)
                or len(extra_priced) < len(extra)
                or any(line.partial for line in [*priced, *extra_priced]),
            )
        )
    everything = [*costs, *retried]
    priced_totals = [c.total_usd for c in everything if c is not None and c.total_usd is not None]
    return CostSummary(
        rows=rows,
        total_usd=sum(priced_totals) if priced_totals else None,
        total_partial=len(priced_totals) < len(everything)
        or any(c is not None and c.partial for c in everything),
        unverified=any(
            getattr(c, p).unverified for c in everything if c is not None for p in _PARTIES
        ),
        retried_attempts=len(retried),
    )


def _wall_seconds(batch: dict) -> float | None:
    try:
        start = datetime.fromisoformat(batch["started"])
        end = datetime.fromisoformat(batch["finished"])
        return (end - start).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def summarize_batch(batch_dir: Path) -> BatchSummary:
    """Load every run in ``batch_dir`` and reduce them to a ``BatchSummary``."""
    batch = _read_json(batch_dir / "batch.json")
    batch = batch if isinstance(batch, dict) else {}
    runs = [_load_run(p) for p in _run_dirs(batch_dir)]
    rows = [_run_row(run) for run in runs]
    judged = [run for run in runs if run.judge is not None]
    counts = {v: sum(1 for r in rows if r.verdict == v) for v in (*_VERDICT_RANK, "errored")}
    graded = counts["pass"] + counts["fail"]
    return BatchSummary(
        scenario=batch.get("scenario"),
        target=batch.get("target"),
        caller=batch.get("caller"),
        judge=batch.get("judge"),
        bugs=batch.get("bugs") or [],
        started=batch.get("started"),
        finished=batch.get("finished"),
        wall_seconds=_wall_seconds(batch),
        n=len(runs),
        transcripts=[run.name for run in runs if (batch_dir / run.name / "transcript.md").exists()],
        counts=counts,
        pass_rate=counts["pass"] / graded if graded else None,
        runs=rows,
        failures=_failures(judged),
        fields=_fields(judged),
        quality=_quality(judged),
        latency=_latency(runs),
        incidents=_incidents(runs),
        worst=_worst(rows),
        cost=_cost(runs, _retried_costs(batch_dir)),
    )


def write_batch_report(batch_dir: Path) -> Path:
    """Write ``report.json`` and ``report.md`` into ``batch_dir``; return the Markdown path."""
    summary = summarize_batch(batch_dir)
    (batch_dir / "report.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    path = batch_dir / "report.md"
    path.write_text(render_batch_report(summary), encoding="utf-8")
    return path
