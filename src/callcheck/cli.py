"""The ``callcheck`` command line: a thin layer that wires the core to its adapters.

``run`` places simulated calls against a target agent and grades them; ``preflight`` checks
that the two sides can hear each other; ``judge`` and ``report`` redo the last steps on a run
directory that already exists; ``adapters`` lists what can be plugged in. All decisions about
a call live in ``callcheck.core``; this module only parses options, builds the adapters the
options name, and prints the result.
"""

import asyncio
import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from callcheck.adapters.callers import CALLERS, make_caller
from callcheck.adapters.chat import CHAT, make_chat
from callcheck.adapters.fake_channel import FakeChannel
from callcheck.adapters.targets import TARGETS, make_target
from callcheck.config import ConfigError, Settings, load_settings, require_aws
from callcheck.core.brief import build_brief, preflight_brief
from callcheck.core.call import Call, CallSettings
from callcheck.core.chat import ChatModel
from callcheck.core.clock import MonotonicClock
from callcheck.core.endpoints import AgentDefinition, Caller, CallerBrief, TargetAgent
from callcheck.core.judge.judge import judge_run
from callcheck.core.judge.schema import JudgeResult
from callcheck.core.metrics import write_metrics
from callcheck.core.preflight import format_preflight, run_preflight
from callcheck.core.recorder import Recorder
from callcheck.core.report.markdown import write_run_report
from callcheck.core.scenario import (
    Persona,
    Scenario,
    VariationPlan,
    load_persona,
    load_scenario,
    make_plan,
    persona_path,
    spoken_value,
)
from callcheck.reference_agent import BUGS, load_reference_agent

app = typer.Typer(
    help="Evaluate a voice agent: simulated callers, recorded calls, and a judge that shows its "
    "reasons.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

EXIT_USAGE = 2
PREFLIGHT_PHRASE = "Hello, can you hear me? Testing one two three."
PREFLIGHT_FIELDS = ["first_name", "phone"]
VERDICT_STYLE = {"pass": "green", "fail": "red", "undetermined": "yellow", "artifact": "magenta"}

TargetOption = Annotated[str, typer.Option("--target", help="Agent under test; see `adapters`.")]
CallerOption = Annotated[str, typer.Option("--caller", help="Simulated caller; see `adapters`.")]
VoiceOption = Annotated[
    str | None,
    typer.Option("--caller-voice", help="Nova Sonic voice id for the caller, over the persona's."),
]
JudgeOption = Annotated[
    str | None,
    typer.Option("--judge", help="Judge backend; defaults to CALLCHECK_JUDGE (bedrock)."),
]


@dataclass
class _Choices:
    """The adapter names and caller options one `run` was asked for."""

    target: str
    caller: str
    judge: str
    voice: str | None
    bugs: frozenset[str]


def _fail(message: str) -> typer.Exit:
    """Print ``message`` as an error and return the exception that exits with the usage code."""
    console.print(escape(message), style="red", soft_wrap=True)
    return typer.Exit(EXIT_USAGE)


def _check_name(value: str, registry: dict[str, str], option: str) -> str:
    """Return ``value`` if it names an entry of ``registry``, else a usage error naming them."""
    if value not in registry:
        known = ", ".join(sorted(registry))
        raise typer.BadParameter(f"{value!r} is not one of: {known}", param_hint=option)
    return value


def _parse_bugs(csv: str | None, fix_bugs: bool) -> frozenset[str]:
    """Return the planted bugs to switch on: none with ``--fix-bugs``, all unless ``csv`` says."""
    if fix_bugs:
        return frozenset()
    if csv is None:
        return frozenset(BUGS)
    names = [name.strip() for name in csv.split(",") if name.strip()]
    unknown = [name for name in names if name not in BUGS]
    if unknown:
        raise typer.BadParameter(
            f"unknown bug {unknown[0]!r}; known: {', '.join(BUGS)}", param_hint="--bugs"
        )
    return frozenset(names)


def _need_aws(settings: Settings, *, nova_sonic: bool, bedrock_judge: bool = False) -> None:
    """Exit with the usage code unless the AWS profile is set whenever Bedrock will be used."""
    try:
        if nova_sonic or bedrock_judge:
            require_aws(settings)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


def _with_voice(brief: CallerBrief, voice: str | None) -> CallerBrief:
    """Return ``brief`` with its Nova Sonic voice replaced by ``voice``, if one was given."""
    if voice is None:
        return brief
    return dataclasses.replace(brief, voice={**brief.voice, "nova_sonic": voice})


def _build_target(
    name: str, agent: AgentDefinition, fields: list[str], channel: FakeChannel, settings: Settings
) -> TargetAgent:
    """Build the registered target ``name``; fakes talk over ``channel``, Nova over Bedrock."""
    if name == "fake":
        return make_target(name, channel=channel, fields=fields)
    return make_target(
        name,
        agent=agent,
        model_id=settings.nova_sonic_model,
        region=settings.aws_region,
        profile=settings.aws_profile,
    )


def _build_caller(
    name: str, brief: CallerBrief, channel: FakeChannel, settings: Settings, **fake_args: object
) -> Caller:
    """Build the registered caller ``name``; ``fake_args`` configure the fake caller only."""
    if name == "fake":
        return make_caller(name, channel=channel, **fake_args)
    return make_caller(
        name,
        brief=brief,
        model_id=settings.nova_sonic_model,
        region=settings.aws_region,
        profile=settings.aws_profile,
    )


def _build_chat(name: str, settings: Settings) -> ChatModel:
    """Build the registered judge model ``name``."""
    if name == "fake":
        return make_chat(name)
    return make_chat(
        name,
        model_id=settings.judge_model,
        region=settings.aws_region,
        profile=settings.aws_profile,
    )


def _stamp() -> str:
    """Return the current local time as ``YYYYmmdd-HHMMSS`` for naming a batch directory."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _show_verdict(result: JudgeResult, run_dir: Path, report: Path) -> None:
    """Print where a run was written, how it was graded, and where its report is."""
    style = VERDICT_STYLE.get(result.verdict, "white")
    console.print(f"run:     {escape(str(run_dir))}", soft_wrap=True)
    console.print(f"verdict: [bold {style}]{result.verdict}[/] - {escape(result.summary)}")
    console.print(f"report:  {escape(str(report))}", soft_wrap=True)


async def _preflight(
    settings: Settings, choices: _Choices, brief: CallerBrief, run_dir: Path
) -> bool:
    """Place the preflight call, print what each side heard, and report whether it passed."""
    channel = FakeChannel()
    agent = load_reference_agent(PREFLIGHT_FIELDS)
    target = _build_target(choices.target, agent, PREFLIGHT_FIELDS, channel, settings)
    caller = _build_caller(
        choices.caller, brief, channel, settings, answers={}, script=[PREFLIGHT_PHRASE]
    )
    result = await run_preflight(
        target=target, caller=caller, clock=MonotonicClock(), run_dir=run_dir
    )
    console.print(escape(format_preflight(result)), highlight=False, soft_wrap=True)
    return result.passed


def _make_judge(name: str, settings: Settings) -> ChatModel:
    """Build the judge model before anything is paid for; exit with the usage code if it fails."""
    try:
        return _build_chat(name, settings)
    except Exception as exc:
        raise _fail(f"Could not set up the {name} judge: {exc}") from exc


def _call_error(run_dir: Path) -> str | None:
    """Return the error text of a call that ended with reason ``error``, else ``None``."""
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
    if outcome.get("reason") != "error":
        return None
    return outcome.get("error") or "unknown error"


async def _grade(run_dir: Path, scenario: Scenario, chat: ChatModel) -> bool:
    """Compute metrics, judge the run, render the report and print the verdict.

    Returns ``False`` when the run needs attention: judging failed (credentials, throttling, a
    bad reply; reported in one line, the recording stays on disk and ``callcheck judge`` can
    redo it) or the call itself ended in an error, which is printed with the verdict. An agent
    that merely fails its scenario still returns ``True``.
    """
    try:
        write_metrics(run_dir)
        result = await judge_run(run_dir, scenario, chat)
        _show_verdict(result, run_dir, write_run_report(run_dir))
        call_error = _call_error(run_dir)
        if call_error:
            console.print(f"call ended: error - {escape(call_error)}", style="red", soft_wrap=True)
            return False
    except Exception as exc:
        console.print(
            f"Judging failed for {escape(str(run_dir))}: {escape(str(exc))}. "
            f"Re-run with: callcheck judge {escape(str(run_dir))}",
            style="red",
            soft_wrap=True,
        )
        return False
    return True


async def _one_call(
    settings: Settings,
    choices: _Choices,
    scenario: Scenario,
    persona: Persona,
    plan: VariationPlan,
    run_dir: Path,
    chat: ChatModel,
) -> bool:
    """Place one graded call, record it into ``run_dir``, judge it and print the verdict.

    Returns whether the call and its judging went cleanly (see ``_grade``).
    """
    names = [spec.name for spec in scenario.fields]
    brief = _with_voice(build_brief(scenario, persona, plan), choices.voice)
    channel = FakeChannel()
    answers = {spec.name: spoken_value(spec) for spec in scenario.fields}
    corrections = {c.field: c.wrong_value for c in plan.corrections}
    target = _build_target(
        choices.target, load_reference_agent(names, choices.bugs), names, channel, settings
    )
    caller = _build_caller(
        choices.caller, brief, channel, settings, answers=answers, corrections=corrections
    )
    recorder = Recorder(run_dir)
    (recorder.run_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    (recorder.run_dir / "scenario.json").write_text(
        scenario.model_dump_json(indent=2), encoding="utf-8"
    )
    call = Call(
        caller=caller,
        target=target,
        sink=recorder,
        clock=MonotonicClock(),
        settings=CallSettings(max_seconds=scenario.call.max_seconds, stall_s=scenario.call.stall_s),
    )
    await call.run()
    return await _grade(recorder.run_dir, scenario, chat)


async def _run_batch(
    settings: Settings,
    choices: _Choices,
    scenario: Scenario,
    persona: Persona,
    seeds: list[int],
    batch_dir: Path,
    chat: ChatModel,
    preflight: bool,
) -> int:
    """Run the preflight (unless skipped), then one call per seed; return the exit code.

    The code is 1 when any call ended in an error or any judging failed, 2 when the preflight
    failed, and 0 otherwise (an agent that fails its scenario is still a successful run).
    """
    if preflight:
        brief = _with_voice(
            dataclasses.replace(preflight_brief(), voice=dict(persona.voice)), choices.voice
        )
        if not await _preflight(settings, choices, brief, batch_dir / "preflight"):
            console.print("Preflight failed; no calls were placed.", style="red")
            return EXIT_USAGE
    clean = True
    for seed in seeds:
        plan = make_plan(scenario, persona, seed)
        run_dir = batch_dir / f"{persona.id}-{seed}"
        clean &= await _one_call(settings, choices, scenario, persona, plan, run_dir, chat)
    return 0 if clean else 1


def _load_inputs(scenario_path: Path, persona_id: str) -> tuple[Scenario, Persona]:
    """Load the scenario and persona, exiting with one line if a file or a field is wrong."""
    persona_file = persona_path(scenario_path, persona_id)
    for kind, path in (("Scenario", scenario_path), ("Persona", persona_file)):
        if not path.is_file():
            raise _fail(f"{kind} file not found: {path}")
    scenario = load_scenario(scenario_path)
    try:
        load_reference_agent([spec.name for spec in scenario.fields])
    except KeyError as exc:
        raise _fail(
            f"Scenario field {exc.args[0]!r} is not defined by the reference agent's intake.yaml"
        ) from exc
    return scenario, load_persona(persona_file)


@app.command()
def run(
    scenario_path: Annotated[Path, typer.Argument(help="Scenario YAML file to run.")],
    target: TargetOption,
    caller: CallerOption,
    caller_voice: VoiceOption = None,
    persona: Annotated[
        str, typer.Option(help="Persona id: personas/<id>.yaml beside the scenario.")
    ] = "baseline",
    n: Annotated[
        int, typer.Option("-n", min=1, help="Number of calls, with consecutive seeds.")
    ] = 1,
    seed: Annotated[int, typer.Option(help="Seed of the first call; later calls add one.")] = 1,
    judge: JudgeOption = None,
    skip_preflight: Annotated[
        bool, typer.Option("--skip-preflight", help="Do not check first that both sides can hear.")
    ] = False,
    bugs: Annotated[
        str | None,
        typer.Option(help="Comma list of the reference agent's planted bugs; default all."),
    ] = None,
    fix_bugs: Annotated[
        bool, typer.Option("--fix-bugs", help="Plant no bugs in the reference agent.")
    ] = False,
) -> None:
    """Place simulated calls against an agent, record them, and grade each one."""
    settings = load_settings()
    choices = _Choices(
        target=_check_name(target, TARGETS, "--target"),
        caller=_check_name(caller, CALLERS, "--caller"),
        judge=_check_name(judge or settings.judge, CHAT, "--judge"),
        voice=caller_voice,
        bugs=_parse_bugs(bugs, fix_bugs),
    )
    _need_aws(
        settings,
        nova_sonic="nova-sonic" in (choices.target, choices.caller),
        bedrock_judge=choices.judge == "bedrock",
    )
    scenario, persona_def = _load_inputs(scenario_path, persona)
    chat = _make_judge(choices.judge, settings)
    batch_dir = settings.runs_dir / f"{_stamp()}-{scenario.id}"
    seeds = [seed + i for i in range(n)]
    code = asyncio.run(
        _run_batch(
            settings, choices, scenario, persona_def, seeds, batch_dir, chat, not skip_preflight
        )
    )
    raise typer.Exit(code)


@app.command()
def preflight(target: TargetOption, caller: CallerOption, caller_voice: VoiceOption = None) -> None:
    """Check that the agent hears the caller and the caller hears the agent."""
    settings = load_settings()
    choices = _Choices(
        target=_check_name(target, TARGETS, "--target"),
        caller=_check_name(caller, CALLERS, "--caller"),
        judge=settings.judge,
        voice=caller_voice,
        bugs=frozenset(BUGS),
    )
    _need_aws(settings, nova_sonic="nova-sonic" in (choices.target, choices.caller))
    brief = _with_voice(preflight_brief(), caller_voice)
    run_dir = settings.runs_dir / f"{_stamp()}-preflight"
    passed = asyncio.run(_preflight(settings, choices, brief, run_dir))
    raise typer.Exit(0 if passed else EXIT_USAGE)


def _load_run(run_dir: Path) -> Scenario:
    """Read the scenario a run directory was made from, or exit if it is not a run directory."""
    saved = run_dir / "scenario.json"
    if not saved.exists():
        raise _fail(f"{run_dir} has no scenario.json; it is not a run directory made by `run`")
    return Scenario.model_validate_json(saved.read_text(encoding="utf-8"))


@app.command()
def judge(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to grade again.")],
    judge: JudgeOption = None,
) -> None:
    """Grade a recorded run again and rewrite its judge.json and report.md."""
    settings = load_settings()
    scenario = _load_run(run_dir)
    name = _check_name(judge or settings.judge, CHAT, "--judge")
    _need_aws(settings, nova_sonic=False, bedrock_judge=name == "bedrock")
    chat = _make_judge(name, settings)
    if not asyncio.run(_grade(run_dir, scenario, chat)):
        raise typer.Exit(1)


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to render the report for.")],
) -> None:
    """Render report.md again from a run's existing judge.json and metrics.json."""
    _load_run(run_dir)
    if not (run_dir / "judge.json").exists():
        raise _fail(f"No judge.json in {run_dir}; run callcheck judge {run_dir} first.")
    console.print(f"report:  {escape(str(write_run_report(run_dir)))}", soft_wrap=True)


@app.command()
def adapters() -> None:
    """List the targets, callers and judge models that can be selected by name."""
    for kind, registry in (("targets", TARGETS), ("callers", CALLERS), ("judges", CHAT)):
        console.print(f"{kind:<8}{', '.join(sorted(registry))}")
