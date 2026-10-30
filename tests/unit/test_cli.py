"""Unit tests for the command line, run on the offline fakes with a virtual clock."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import callcheck.cli as cli
from callcheck.cli import app
from callcheck.core.clock import VirtualClock
from callcheck.core.preflight import PreflightResult

SCENARIO = str(Path(__file__).parents[2] / "scenarios" / "intake_short.yaml")
FAKES = ["--target", "fake", "--caller", "fake", "--judge", "fake"]
RUN_FILES = {
    "call.wav",
    "events.jsonl",
    "transcript.jsonl",
    "transcript.md",
    "plan.json",
    "scenario.json",
    "judge.json",
    "metrics.json",
    "report.md",
    "outcome.json",
}


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    """Keep the CLI off the network, off the real .env and out of real time."""
    monkeypatch.setenv("AWS_PROFILE", "")  # set, so a repo .env cannot supply one
    monkeypatch.setenv("CALLCHECK_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cli, "MonotonicClock", VirtualClock)


def invoke(*args: str):
    return CliRunner().invoke(app, list(args))


def run_dirs(tmp_path) -> list[Path]:
    return sorted(p.parent for p in (tmp_path / "runs").rglob("judge.json"))


def test_run_with_fakes_produces_full_run_dir(tmp_path):
    res = invoke("run", SCENARIO, *FAKES, "--persona", "self_correcting", "--skip-preflight")
    assert res.exit_code == 0, res.output
    (run,) = run_dirs(tmp_path)
    assert {p.name for p in run.iterdir()} >= RUN_FILES
    assert run.name == "self_correcting-1" and run.parent.name.endswith("-intake-short")
    assert "report.md" in res.output and "verdict" in res.output.lower()


def test_run_writes_the_plan_the_seed_chose(tmp_path):
    invoke(
        "run", SCENARIO, *FAKES, "--persona", "self_correcting", "--seed", "7", "--skip-preflight"
    )
    (run,) = run_dirs(tmp_path)
    assert run.name == "self_correcting-7"
    assert '"seed":7' in (run / "plan.json").read_text().replace(" ", "")


def test_run_n_makes_one_dir_per_seed(tmp_path):
    res = invoke("run", SCENARIO, *FAKES, "-n", "2", "--seed", "5", "--skip-preflight")
    assert res.exit_code == 0, res.output
    assert [p.name for p in run_dirs(tmp_path)] == ["baseline-5", "baseline-6"]


def test_run_runs_preflight_first_into_the_batch_dir(tmp_path):
    res = invoke("run", SCENARIO, *FAKES)
    assert res.exit_code == 0, res.output
    (run,) = run_dirs(tmp_path)
    assert (run.parent / "preflight" / "events.jsonl").exists()
    assert "heard by agent:   yes" in res.output


def test_run_aborts_with_code_2_when_preflight_fails(tmp_path, monkeypatch):
    async def deaf_agent(**_):
        return PreflightResult(
            heard_by_agent=False,
            agent_heard_text=None,
            heard_by_caller=True,
            caller_heard_text="hi",
            greeting_latency_ms=400.0,
            passed=False,
        )

    monkeypatch.setattr(cli, "run_preflight", deaf_agent)
    res = invoke("run", SCENARIO, *FAKES)
    assert res.exit_code == 2
    assert "did not register the caller's voice" in res.output
    assert run_dirs(tmp_path) == []


def test_preflight_with_fakes_passes():
    res = invoke("preflight", "--target", "fake", "--caller", "fake")
    assert res.exit_code == 0, res.output
    assert "heard by agent:   yes" in res.output


def test_unknown_bug_is_a_usage_error():
    res = invoke("run", SCENARIO, *FAKES, "--bugs", "nonsense", "--skip-preflight")
    assert res.exit_code == 2
    assert "nonsense" in res.output and "day_month_swap" in res.output


def test_unknown_target_is_a_usage_error():
    res = invoke("run", SCENARIO, "--target", "nope", "--caller", "fake", "--judge", "fake")
    assert res.exit_code == 2 and "nope" in res.output


def test_fix_bugs_and_bug_list_are_accepted():
    for extra in (["--fix-bugs"], ["--bugs", "strict_order,day_month_swap"]):
        res = invoke("run", SCENARIO, *FAKES, "--skip-preflight", *extra)
        assert res.exit_code == 0, res.output


def test_caller_voice_overrides_the_brief(monkeypatch):
    seen = {}
    real = cli._build_caller

    def spy(name, brief, *args, **kwargs):
        seen["voice"] = dict(brief.voice)
        return real(name, brief, *args, **kwargs)

    monkeypatch.setattr(cli, "_build_caller", spy)
    res = invoke(
        "run",
        SCENARIO,
        *FAKES,
        "--persona",
        "self_correcting",
        "--caller-voice",
        "amy",
        "--skip-preflight",
    )
    assert res.exit_code == 0, res.output
    assert seen["voice"]["nova_sonic"] == "amy"


@pytest.mark.parametrize(
    "args",
    [
        ["preflight", "--target", "nova-sonic", "--caller", "fake"],
        ["run", SCENARIO, "--target", "fake", "--caller", "nova-sonic", "--judge", "fake"],
        ["run", SCENARIO, "--target", "fake", "--caller", "fake", "--judge", "bedrock"],
    ],
)
def test_bedrock_backed_adapters_need_a_profile(args):
    res = invoke(*args)
    assert res.exit_code == 2
    assert "AWS_PROFILE" in res.output


def test_judge_and_report_rerun_from_a_run_dir(tmp_path):
    invoke("run", SCENARIO, *FAKES, "--skip-preflight")
    (run,) = run_dirs(tmp_path)
    (run / "report.md").unlink()
    (run / "judge.json").unlink()
    res = invoke("judge", str(run), "--judge", "fake")
    assert res.exit_code == 0, res.output
    assert (run / "judge.json").exists() and (run / "report.md").exists()
    (run / "report.md").unlink()
    res = invoke("report", str(run))
    assert res.exit_code == 0, res.output
    assert (run / "report.md").exists()


def test_judge_rejects_a_directory_that_is_not_a_run(tmp_path):
    res = invoke("judge", str(tmp_path))
    assert res.exit_code == 2 and "scenario.json" in res.output


def test_adapters_lists_registries():
    out = invoke("adapters").output
    assert "nova-sonic" in out and "fake" in out and "bedrock" in out


def test_help_describes_every_command():
    out = invoke("--help").output
    for command in ("preflight", "run", "judge", "report", "adapters"):
        assert command in out
