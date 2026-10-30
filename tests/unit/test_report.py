import json

import pytest

from callcheck.core.judge.schema import FieldVerdict, JudgeResult, QualityReport, QualityScore
from callcheck.core.metrics import Metrics
from callcheck.core.report.markdown import render_run_report, write_run_report


def _quality() -> QualityReport:
    score = QualityScore(score=4, reason="Fine.")
    return QualityReport(
        repetition=score, confusion=score, correction_handling=score, clarity=score, closing=score
    )


def _write(run_dir, judge: JudgeResult, metrics: Metrics) -> None:
    (run_dir / "judge.json").write_text(judge.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "outcome.json").write_text(
        json.dumps({"reason": "caller_done", "duration_ms": 9000}), encoding="utf-8"
    )


def _judge(quality: QualityReport | None) -> JudgeResult:
    return JudgeResult(
        verdict="fail" if quality else "undetermined",
        failure_types=["wrong_value"],
        fields=[
            FieldVerdict(
                name="phone",
                expected="555-0100",
                captured="555-0199",
                verdict="mismatch",
                method="result",
                note="digits differ",
            )
        ],
        quality=quality,
        captured_method="result",
        summary="The agent booked with the wrong phone number.",
        undetermined_reasons=[] if quality else ["judge model unavailable", "no transcript"],
    )


_METRICS = Metrics(
    greeting_latency_ms=900.4,
    ttfa_ms=[1200, 800],
    ttfa_p50_ms=1000,
    ttfa_max_ms=1200,
    duration_ms=9000,
    caller_turns=1,
    agent_turns=2,
)


@pytest.fixture
def run_dir_with_judge_and_metrics(tmp_path):
    _write(tmp_path, _judge(_quality()), _METRICS)
    return tmp_path


def test_report_renders_verdict_fields_and_links(run_dir_with_judge_and_metrics):
    md = render_run_report(run_dir_with_judge_and_metrics)
    assert "## Verdict" in md and "| phone |" in md and "transcript.md" in md
    assert "n/a until M2" in md
    assert "[call.wav](call.wav)" in md and "[events.jsonl](events.jsonl)" in md
    assert "| clarity | 4 | Fine. |" in md
    assert "- Greeting latency: 900 ms" in md and "- TTFA p50: 1000 ms" in md
    assert "- Agent turns measured: 2" in md


def test_report_shows_undetermined_quality_and_missing_latency(tmp_path):
    empty = Metrics(
        greeting_latency_ms=None,
        ttfa_ms=[],
        ttfa_p50_ms=None,
        ttfa_max_ms=None,
        duration_ms=10,
        caller_turns=0,
        agent_turns=0,
    )
    _write(tmp_path, _judge(None), empty)
    md = render_run_report(tmp_path)
    assert "undetermined: judge model unavailable; no transcript" in md
    assert "- Greeting latency: n/a" in md and "- TTFA p50: n/a" in md


def test_write_run_report_writes_report_md(run_dir_with_judge_and_metrics):
    path = write_run_report(run_dir_with_judge_and_metrics)
    assert path.name == "report.md"
    assert path.read_text(encoding="utf-8") == render_run_report(run_dir_with_judge_and_metrics)
