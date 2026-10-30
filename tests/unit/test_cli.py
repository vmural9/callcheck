"""Unit tests for the command line, run on the offline fakes with a virtual clock."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import callcheck.cli as cli
from callcheck.adapters.targets.fake import FakeTarget
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
    res = invoke("run", SCENARIO, *FAKES, "--personas", "self_correcting", "--skip-preflight")
    assert res.exit_code == 0, res.output
    (run,) = run_dirs(tmp_path)
    assert {p.name for p in run.iterdir()} >= RUN_FILES
    assert run.name == "self_correcting-1" and run.parent.name.endswith("-intake-short")
    assert "self_correcting-1" in res.output and "report.md" in res.output
    assert "pass rate" in res.output


def test_run_writes_the_plan_the_seed_chose(tmp_path):
    invoke(
        "run", SCENARIO, *FAKES, "--personas", "self_correcting", "--seed", "7", "--skip-preflight"
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
        "--personas",
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


class BrokenChat:
    """A judge whose every request fails, as with expired credentials."""

    async def complete(self, *args, **kwargs):
        raise RuntimeError("ExpiredToken")


def test_judge_that_cannot_be_built_fails_before_any_call(tmp_path, monkeypatch):
    def refuse(name, settings):
        raise RuntimeError("The config profile (x) could not be found")

    monkeypatch.setattr(cli, "_build_chat", refuse)
    res = invoke("run", SCENARIO, *FAKES)
    assert res.exit_code == 2
    assert "could not be found" in res.output
    assert not (tmp_path / "runs").exists()


def test_judging_failure_keeps_the_recording_and_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_build_chat", lambda name, settings: BrokenChat())
    res = invoke("run", SCENARIO, *FAKES, "-n", "2", "--skip-preflight")
    assert res.exit_code == 1, res.output  # every call is still placed, but the batch is not clean
    assert res.output.count("Judging failed for") == 2
    assert "ExpiredToken" in res.output and "Re-run with: callcheck judge" in res.output
    runs = sorted(p.parent for p in (tmp_path / "runs").rglob("events.jsonl"))
    assert len(runs) == 2
    assert all(not (run / "judge.json").exists() for run in runs)


class CrashingTarget(FakeTarget):
    """Fake target that fails as soon as the call sends it audio."""

    async def send_audio(self, pcm: bytes) -> None:
        raise RuntimeError("model access denied")


def test_call_error_is_printed_judged_undetermined_and_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_build_target",
        lambda name, agent, fields, channel, settings: CrashingTarget(channel, fields),
    )
    res = invoke("run", SCENARIO, *FAKES, "-n", "2", "--skip-preflight")
    assert res.exit_code == 1, res.output
    assert res.output.count("call ended: error - model access denied") == 2
    runs = run_dirs(tmp_path)
    assert len(runs) == 2
    for run in runs:
        assert json.loads((run / "judge.json").read_text())["verdict"] == "undetermined"
        assert "- Call error: model access denied" in (run / "report.md").read_text()


def test_an_agent_that_fails_its_scenario_still_exits_0(tmp_path):
    res = invoke("run", SCENARIO, *FAKES, "--personas", "self_correcting", "--skip-preflight")
    assert res.exit_code == 0, res.output
    assert "call ended: error" not in res.output


def test_judge_command_exits_1_when_judging_fails(tmp_path, monkeypatch):
    invoke("run", SCENARIO, *FAKES, "--skip-preflight")
    (run,) = run_dirs(tmp_path)
    monkeypatch.setattr(cli, "_build_chat", lambda name, settings: BrokenChat())
    res = invoke("judge", str(run), "--judge", "fake")
    assert res.exit_code == 1 and "Judging failed for" in res.output


def test_missing_scenario_file_is_one_line():
    res = invoke("run", "scenarios/nowhere.yaml", *FAKES)
    assert res.exit_code == 2
    assert "Scenario file not found" in res.output and "Traceback" not in res.output


def test_missing_persona_file_is_one_line():
    res = invoke("run", SCENARIO, *FAKES, "--personas", "nobody")
    assert res.exit_code == 2
    assert "Persona file not found" in res.output and "Traceback" not in res.output


def test_field_the_reference_agent_lacks_is_one_line(tmp_path):
    scenario = tmp_path / "odd.yaml"
    scenario.write_text(
        "id: odd\ndescription: d\ncaller_goal: g\n"
        "fields:\n  - {name: shoe_size, type: number, expected: 9}\n"
    )
    personas = tmp_path / "personas"
    personas.mkdir()
    (personas / "baseline.yaml").write_text("id: baseline\nstyle: plain\n")
    res = invoke("run", str(scenario), *FAKES, "--skip-preflight")
    assert res.exit_code == 2
    assert "shoe_size" in res.output and "intake.yaml" in res.output
    assert "Traceback" not in res.output


def test_report_without_judge_json_says_what_to_run(tmp_path):
    invoke("run", SCENARIO, *FAKES, "--skip-preflight")
    (run,) = run_dirs(tmp_path)
    (run / "judge.json").unlink()
    res = invoke("report", str(run))
    assert res.exit_code == 2
    assert "No judge.json in" in res.output and "callcheck judge" in res.output


def batch_dir(tmp_path) -> Path:
    (batch,) = (p for p in (tmp_path / "runs").iterdir() if p.is_dir())
    return batch


def test_run_matrix_with_two_personas_writes_batch_report(tmp_path):
    res = invoke(
        "run",
        SCENARIO,
        *FAKES,
        "-n",
        "4",
        "--personas",
        "baseline,self_correcting",
        "--parallel",
        "2",
        "--skip-preflight",
    )
    assert res.exit_code == 0, res.output
    assert [p.name for p in run_dirs(tmp_path)] == [
        "baseline-1",
        "baseline-3",
        "self_correcting-2",
        "self_correcting-4",
    ]
    batch = batch_dir(tmp_path)
    assert (batch / "report.md").exists() and (batch / "report.json").exists()
    meta = json.loads((batch / "batch.json").read_text())
    assert meta["scenario"] == "intake-short" and meta["target"] == "fake"
    assert meta["caller"] == "fake" and meta["judge"] == "fake"
    assert meta["started"] <= meta["finished"] and isinstance(meta["bugs"], list)
    assert "pass rate" in res.output and str(batch / "report.md") in res.output
    for run in run_dirs(tmp_path):
        assert {"incidents.json", "judge.json", "metrics.json", "report.md"} <= {
            p.name for p in run.iterdir()
        }


def test_nova_parallel_guard():
    args = ["run", SCENARIO, "--target", "nova-sonic", "--caller", "nova-sonic", "--judge", "fake"]
    res = invoke(*args, "--parallel", "9")
    assert res.exit_code == 2
    assert "--force" in res.output and "AWS_PROFILE" not in res.output
    # --force lifts the guard, so the run goes on to the AWS check
    res = invoke(*args, "--parallel", "9", "--force")
    assert res.exit_code == 2 and "AWS_PROFILE" in res.output


def test_unknown_persona_one_line_error(tmp_path):
    res = invoke("run", SCENARIO, *FAKES, "--personas", "baseline,nobody", "--skip-preflight")
    assert res.exit_code == 2
    assert "Persona file not found" in res.output and "nobody.yaml" in res.output
    assert "Traceback" not in res.output and not (tmp_path / "runs").exists()


def test_report_command_detects_batch_dir(tmp_path):
    invoke("run", SCENARIO, *FAKES, "-n", "2", "--skip-preflight")
    batch = batch_dir(tmp_path)
    (batch / "report.md").unlink()
    (batch / "report.json").unlink()
    res = invoke("report", str(batch))
    assert res.exit_code == 0, res.output
    assert (batch / "report.md").exists() and (batch / "report.json").exists()


def test_report_command_rejects_a_directory_with_no_runs(tmp_path):
    res = invoke("report", str(tmp_path))
    assert res.exit_code == 2 and "Traceback" not in res.output


def test_judge_command_writes_incidents(tmp_path):
    invoke("run", SCENARIO, *FAKES, "--skip-preflight")
    (run,) = run_dirs(tmp_path)
    for name in ("incidents.json", "judge.json", "metrics.json", "report.md"):
        (run / name).unlink()
    res = invoke("judge", str(run), "--judge", "fake")
    assert res.exit_code == 0, res.output
    assert isinstance(json.loads((run / "incidents.json").read_text()), list)
    assert (run / "judge.json").exists() and (run / "metrics.json").exists()
    assert (run / "report.md").exists()


def test_metrics_are_written_after_judging_so_they_can_price_the_judge(tmp_path, monkeypatch):
    order = []
    for name in ("write_incidents", "write_metrics", "write_run_report"):
        real = getattr(cli, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            order.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(cli, name, spy)
    real_judge = cli.judge_run

    async def spy_judge(*args, **kwargs):
        order.append("judge_run")
        return await real_judge(*args, **kwargs)

    monkeypatch.setattr(cli, "judge_run", spy_judge)
    res = invoke("run", SCENARIO, *FAKES, "--skip-preflight")
    assert res.exit_code == 0, res.output
    assert order == ["write_incidents", "judge_run", "write_metrics", "write_run_report"]
