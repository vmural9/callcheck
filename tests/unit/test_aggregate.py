import json

from callcheck.core.cost import Cost, CostLine
from callcheck.core.judge.schema import FieldVerdict, JudgeResult, QualityReport, QualityScore
from callcheck.core.metrics import Metrics
from callcheck.core.report.aggregate import (
    BatchSummary,
    render_batch_report,
    summarize_batch,
    write_batch_report,
)
from callcheck.core.turns import Incident

_BATCH = {
    "scenario": "intake-short",
    "target": "reference-agent",
    "caller": "nova-sonic",
    "judge": "claude-judge",
    "bugs": ["wrong-date"],
    "started": "2026-10-31T10:00:00+00:00",
    "finished": "2026-10-31T10:12:05+00:00",
}


def _quality(**scores: int) -> QualityReport:
    items = {
        name: QualityScore(score=scores.get(name, 4), reason="r")
        for name in ("repetition", "confusion", "correction_handling", "clarity", "closing")
    }
    return QualityReport(**items)


def _field(name="date", verdict="match", captured="1988-07-03") -> FieldVerdict:
    return FieldVerdict(
        name=name,
        expected="1988-07-04",
        captured=captured,
        verdict=verdict,
        method="result",
        note="",
    )


def _cost(agent=0.10, caller=0.05, judge=0.02, **flags) -> Cost:
    def line(usd, unverified=False, partial=False):
        return CostLine(usd=usd, unverified=unverified, partial=partial)

    lines = [
        line(agent, flags.get("agent_unverified", False)),
        line(caller),
        line(judge),
    ]
    known = [x.usd for x in lines if x.usd is not None]
    return Cost(
        agent=lines[0],
        caller=lines[1],
        judge=lines[2],
        total_usd=sum(known) if known else None,
        partial=any(x.usd is None for x in lines),
    )


def make_run(
    batch,
    name,
    verdict="pass",
    failure_types=(),
    fields=None,
    quality="default",
    incidents=(),
    ttfa=(1000.0,),
    cost="default",
):
    run = batch / name
    run.mkdir(parents=True)
    judge = JudgeResult(
        verdict=verdict,
        failure_types=list(failure_types),
        fields=[_field()] if fields is None else fields,
        quality=_quality() if quality == "default" else quality,
        captured_method="result",
        summary="s",
        incidents=list(incidents),
    )
    metrics = Metrics(
        greeting_latency_ms=500,
        ttfa_ms=list(ttfa),
        ttfa_p50_ms=None,
        ttfa_max_ms=None,
        duration_ms=9000,
        caller_turns=1,
        agent_turns=1,
        cost=_cost() if cost == "default" else cost,
    )
    (run / "transcript.md").write_text("transcript", encoding="utf-8")
    (run / "judge.json").write_text(judge.model_dump_json(), encoding="utf-8")
    (run / "metrics.json").write_text(metrics.model_dump_json(), encoding="utf-8")
    (run / "outcome.json").write_text(
        json.dumps({"reason": "caller_done", "duration_ms": 9000}), encoding="utf-8"
    )
    return run


def _batch(tmp_path):
    (tmp_path / "batch.json").write_text(json.dumps(_BATCH), encoding="utf-8")
    return tmp_path


def test_pass_rate_excludes_artifact_and_undetermined(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", "pass")
    make_run(batch, "a-2", "pass")
    make_run(batch, "a-3", "fail", ["wrong_value"])
    make_run(batch, "a-4", "artifact")
    make_run(batch, "a-5", "undetermined")
    summary = summarize_batch(batch)
    assert summary.n == 5
    assert summary.counts == {
        "pass": 2,
        "fail": 1,
        "undetermined": 1,
        "artifact": 1,
        "errored": 0,
    }
    md = render_batch_report(summary)
    assert "2/3 (67%)" in md
    assert "- Wall time: 12m 05s" in md
    assert "- Target: reference-agent" in md and "- Bugs: wrong-date" in md


def test_failures_by_type_counts_and_example(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", "fail", ["wrong_value", "no_greeting"])
    make_run(batch, "a-2", "fail", ["wrong_value"])
    make_run(batch, "a-3", "pass")
    md = render_batch_report(summarize_batch(batch))
    assert "| wrong_value | 2 | [a-1](a-1/transcript.md) |" in md
    assert "| no_greeting | 1 | [a-1](a-1/transcript.md) |" in md


def test_field_accuracy_most_common_wrong(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", "fail", fields=[_field(verdict="mismatch", captured="1988-07-03")])
    make_run(batch, "a-2", "fail", fields=[_field(verdict="mismatch", captured="1988-07-03")])
    make_run(batch, "a-3", "fail", fields=[_field(verdict="mismatch", captured="1990-01-01")])
    make_run(batch, "a-4", "pass", fields=[_field(verdict="match", captured="1988-07-04")])
    make_run(batch, "a-5", "undetermined", fields=[_field(verdict="undetermined", captured=None)])
    md = render_batch_report(summarize_batch(batch))
    assert "| date | 1/4 (25%) | 1988-07-03 |" in md


def test_latency_percentiles_across_runs(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", ttfa=[100, 200, 300])
    make_run(batch, "a-2", ttfa=[400, 5000], verdict="fail", failure_types=["slow"])
    summary = summarize_batch(batch)
    assert summary.latency.turns == 5
    assert summary.latency.p50_ms == 300
    assert summary.latency.p90_ms == 5000
    assert summary.latency.max_ms == 5000
    assert summary.latency.slowest_run == "a-2"
    md = render_batch_report(summary)
    assert "[a-2](a-2/transcript.md)" in md and "5000 ms" in md


def test_worst_three_order(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "p-1", "pass")
    make_run(batch, "p-2", "artifact")
    make_run(batch, "p-3", "undetermined")
    make_run(batch, "p-4", "fail", ["x"], quality=_quality(clarity=0))
    make_run(batch, "p-5", "fail", ["x", "y"])
    make_run(batch, "p-6", "fail", ["x"], quality=_quality(clarity=2))
    summary = summarize_batch(batch)
    assert [r.name for r in summary.worst] == ["p-5", "p-4", "p-6"]
    summary_without_fails = summarize_batch(_two_clean(tmp_path / "other"))
    assert [r.name for r in summary_without_fails.worst] == ["b-2", "b-1"]


def _two_clean(path):
    path.mkdir()
    make_run(path, "b-1", "pass")
    make_run(path, "b-2", "undetermined")
    return path


def test_errored_run_listed_not_counted(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", "pass")
    crashed = batch / "a-2"
    crashed.mkdir()
    (crashed / "outcome.json").write_text(
        json.dumps({"reason": "error", "error": "stream closed", "duration_ms": 10}),
        encoding="utf-8",
    )
    (batch / "a-3").mkdir()
    (batch / "preflight").mkdir()
    (batch / "a-1.attempt1").mkdir()
    (batch / "report.md").write_text("old", encoding="utf-8")
    summary = summarize_batch(batch)
    assert summary.n == 3 and summary.counts["errored"] == 2 and summary.counts["pass"] == 1
    md = render_batch_report(summary)
    assert "1/1 (100%)" in md
    assert "- [a-2](a-2/): stream closed" in md
    assert "- [a-3](a-3/): no judge.json" in md


def test_zero_graded_runs_renders_na(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", "undetermined", quality=None, fields=[])
    (batch / "a-2").mkdir()
    md = render_batch_report(summarize_batch(batch))
    assert "- Pass rate: n/a" in md
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "- Pass rate: n/a" in render_batch_report(summarize_batch(empty))


def test_cost_na_and_unverified_footnote(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", cost=_cost(agent_unverified=True))
    make_run(batch, "a-2", cost=_cost())
    md = render_batch_report(summarize_batch(batch))
    assert "| agent | $0.20 | $0.10 |" in md
    assert "Prices are unverified" in md
    assert "- Total cost: $0.34" in md

    other = tmp_path / "other"
    other.mkdir()
    make_run(other, "a-1", cost=_cost())
    make_run(other, "a-2", cost=None)
    md = render_batch_report(summarize_batch(other))
    assert "| agent | ≥ $0.10 | ≥ $0.10 |" in md and "- Total cost: ≥ $0.17" in md
    assert "a partial sum" in md and "Prices are unverified" not in md

    nothing = tmp_path / "nothing"
    nothing.mkdir()
    make_run(nothing, "a-1", cost=None)
    md = render_batch_report(summarize_batch(nothing))
    assert "| agent | n/a | n/a |" in md and "- Total cost: n/a" in md

    partial = tmp_path / "partial"
    partial.mkdir()
    make_run(partial, "a-1", cost=_cost(judge=None))
    assert "- Total cost: ≥ $0.15" in render_batch_report(summarize_batch(partial))


def test_incidents_table_counts_label_by_attribution(tmp_path):
    batch = _batch(tmp_path)
    inc = Incident(
        t_ms=1000, label="slow_response", attribution="agent", confidence="high", reason="r"
    )
    make_run(batch, "a-1", incidents=[inc, inc])
    make_run(batch, "a-2", incidents=[inc])
    md = render_batch_report(summarize_batch(batch))
    assert "| slow_response | agent | 3 |" in md


def test_report_json_matches_summary(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", "fail", ["wrong_value"])
    make_run(batch, "a-2", "pass")
    path = write_batch_report(batch)
    assert path == batch / "report.md"
    summary = summarize_batch(batch)
    on_disk = BatchSummary.model_validate_json((batch / "report.json").read_text(encoding="utf-8"))
    assert on_disk == summary
    assert path.read_text(encoding="utf-8") == render_batch_report(summary)
    write_batch_report(batch)  # a report left in the batch dir is not a run
    assert summarize_batch(batch).n == 2


def test_incidents_json_takes_precedence_over_judge_incidents(tmp_path):
    batch = _batch(tmp_path)
    in_judge = Incident(
        t_ms=1, label="slow_response", attribution="agent", confidence="high", reason="r"
    )
    on_disk = Incident(
        t_ms=2, label="double_talk", attribution="simulator", confidence="low", reason="r"
    )
    run = make_run(batch, "a-1", incidents=[in_judge])
    (run / "incidents.json").write_text(json.dumps([on_disk.model_dump()]), encoding="utf-8")
    summary = summarize_batch(batch)
    assert [(i.label, i.attribution) for i in summary.incidents] == [("double_talk", "simulator")]


def test_malformed_incidents_json_falls_back_to_judge_incidents(tmp_path):
    batch = _batch(tmp_path)
    in_judge = Incident(
        t_ms=1, label="slow_response", attribution="agent", confidence="high", reason="r"
    )
    run = make_run(batch, "a-1", incidents=[in_judge])
    (run / "incidents.json").write_text(json.dumps([{"label": "nonsense"}]), encoding="utf-8")
    other = make_run(batch, "a-2", incidents=[in_judge])
    (other / "incidents.json").write_text("{not json", encoding="utf-8")
    summary = summarize_batch(batch)
    assert [(i.label, i.count) for i in summary.incidents] == [("slow_response", 2)]


def test_corrupt_judge_json_is_errored_with_the_reason(tmp_path):
    batch = _batch(tmp_path)
    run = make_run(batch, "a-1")
    (run / "judge.json").write_text("{broken", encoding="utf-8")
    summary = summarize_batch(batch)
    assert summary.counts["errored"] == 1
    assert "judge.json unreadable:" in (summary.runs[0].error or "")
    assert "- [a-1](a-1/transcript.md): judge.json unreadable:" in render_batch_report(summary)


def test_percentages_round_half_up(tmp_path):
    from callcheck.core.report.aggregate_render import _rate

    assert _rate(1, 8) == "1/8 (13%)" and _rate(1, 200) == "1/200 (1%)"
    assert _rate(1, 3) == "1/3 (33%)" and _rate(0, 0) == "n/a"


def test_artifact_runs_leave_field_accuracy_but_keep_failure_types(tmp_path):
    batch = _batch(tmp_path)
    make_run(batch, "a-1", "pass", fields=[_field(verdict="match")])
    make_run(
        batch,
        "a-2",
        "artifact",
        ["no_response"],
        fields=[_field(verdict="missing", captured=None)],
    )
    md = render_batch_report(summarize_batch(batch))
    assert "| date | 1/1 (100%) | - |" in md
    assert "| no_response | 1 | [a-2](a-2/transcript.md) |" in md
