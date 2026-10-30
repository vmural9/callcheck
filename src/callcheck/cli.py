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
from collections import Counter
from collections.abc import Awaitable, Callable
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
from callcheck.core.cost import load_pricing
from callcheck.core.endpoints import AgentDefinition, Caller, CallerBrief, TargetAgent
from callcheck.core.judge.judge import judge_run
from callcheck.core.judge.schema import JudgeResult
from callcheck.core.metrics import write_metrics
from callcheck.core.monitor import TurnMonitor
from callcheck.core.preflight import format_preflight, run_preflight
from callcheck.core.recorder import Recorder, read_events
from callcheck.core.report.aggregate import write_batch_report
from callcheck.core.report.markdown import write_run_report
from callcheck.core.runner import MAX_NOVA_PARALLEL, RunResult, RunSpec, expand_runs, run_matrix
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
from callcheck.core.turns import detect_incidents, write_incidents
from callcheck.reference_agent import BUGS, load_reference_agent

app = typer.Typer(
    help="Evaluate a voice agent: simulated callers, recorded calls, and a judge that shows its "
    "reasons.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

EXIT_USAGE = 2
RETRY_DELAY_S = 2.0  # pause before a call is placed again, so a throttled endpoint can recover
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


def _find_pricing() -> dict | None:
    """Return the pricing table from the nearest ``pricing.yaml`` above this file, if any."""
    for parent in Path(__file__).resolve().parents:
        table = parent / "pricing.yaml"
        if table.is_file():
            try:
                return load_pricing(table)
            except Exception as exc:
                console.print(
                    f"pricing.yaml unreadable ({escape(str(exc))}); costs will show n/a.",
                    style="yellow",
                    soft_wrap=True,
                )
                return None
    return None


async def _grade(
    run_dir: Path,
    scenario: Scenario,
    chat: ChatModel,
    choices: _Choices,
    pricing: dict | None,
    *,
    interrupts_expected: bool,
) -> JudgeResult | None:
    """Find the turn-taking incidents, judge the run, price it and render its report.

    The order matters: the judge reads ``incidents.json`` and the cost of judging is only known
    once ``judge.json`` is written. Returns ``None`` after printing one line when this fails
    (credentials, throttling, a bad reply); the recording stays on disk and ``callcheck judge``
    can redo it.
    """
    try:
        incidents = detect_incidents(
            read_events(run_dir),
            scenario.call,
            caller_interrupts_expected=interrupts_expected,
        )
        write_incidents(run_dir, incidents)
        result = await judge_run(run_dir, scenario, chat)
        write_metrics(run_dir, pricing, agent_model=choices.target, caller_model=choices.caller)
        write_run_report(run_dir)
    except Exception as exc:
        console.print(
            f"Judging failed for {escape(str(run_dir))}: {escape(str(exc))}. "
            f"Re-run with: callcheck judge {escape(str(run_dir))}",
            style="red",
            soft_wrap=True,
        )
        return None
    return result


def _note_call_error(run_dir: Path, name: str) -> bool:
    """Print the call's error under the run's ``name``, if it ended in one; say whether it did."""
    call_error = _call_error(run_dir)
    if call_error:
        console.print(
            f"{escape(name)}: call ended: error - {escape(call_error)}",
            style="red",
            soft_wrap=True,
        )
    return call_error is not None


def _record_crash(run_dir: Path, error: str) -> None:
    """Make sure ``run_dir`` exists with an ``outcome.json``, so the batch report counts it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    outcome = run_dir / "outcome.json"
    if not outcome.exists():
        body = {"reason": "error", "duration_ms": 0, "error": error}
        outcome.write_text(json.dumps(body, indent=2), encoding="utf-8")


async def _place_call(
    settings: Settings,
    choices: _Choices,
    scenario: Scenario,
    persona: Persona,
    plan: VariationPlan,
    run_dir: Path,
) -> Path:
    """Place one call and record it; return the run directory the recorder made."""
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
    for name, model in (("plan", plan), ("scenario", scenario), ("persona", persona)):
        (recorder.run_dir / f"{name}.json").write_text(model.model_dump_json(indent=2), "utf-8")
    call = Call(
        caller=caller,
        target=target,
        sink=recorder,
        clock=MonotonicClock(),
        settings=CallSettings(max_seconds=scenario.call.max_seconds, stall_s=scenario.call.stall_s),
        monitor=TurnMonitor(scenario.call),
    )
    await call.run()
    return recorder.run_dir


def _make_place(
    settings: Settings,
    choices: _Choices,
    scenario: Scenario,
    personas: dict[str, Persona],
    chat: ChatModel,
    pricing: dict | None,
) -> Callable[[RunSpec, Path], Awaitable[RunResult]]:
    """Return the ``place(spec, run_dir)`` function ``run_matrix`` calls for each run.

    A call that ends in an error is reported as ``"transport_error"`` so the matrix retries it
    once (the mixer cannot tell a transport failure from any other endpoint failure). The
    first attempt of such a call is not judged; the retry is, whatever its outcome, so a call
    that fails twice still ends with a verdict and a report. The retry waits ``RETRY_DELAY_S``.
    Anything else that goes wrong while placing or grading a run is printed against the run,
    recorded as an errored outcome, and returned as ``"error"``; it never ends the batch.
    """
    attempts: Counter[RunSpec] = Counter()

    async def place(spec: RunSpec, run_dir: Path) -> RunResult:
        attempts[spec] += 1
        if attempts[spec] > 1:
            await asyncio.sleep(RETRY_DELAY_S)
        try:
            return await attempt(spec, run_dir)
        except Exception as exc:
            console.print(
                f"{escape(spec.dir_name)}: crashed - {escape(str(exc))}",
                style="red",
                soft_wrap=True,
            )
            _record_crash(run_dir, str(exc))
            return "error"

    async def attempt(spec: RunSpec, run_dir: Path) -> RunResult:
        persona = personas[spec.persona_id]
        plan = make_plan(scenario, persona, spec.seed)
        made = await _place_call(settings, choices, scenario, persona, plan, run_dir)
        if _call_error(made) and attempts[spec] == 1:
            return "transport_error"
        result = await _grade(
            made,
            scenario,
            chat,
            choices,
            pricing,
            interrupts_expected=persona.behaviours.interrupts,
        )
        if result is None:
            return "error"
        style = VERDICT_STYLE.get(result.verdict, "white")
        console.print(f"{escape(spec.dir_name)}: [bold {style}]{result.verdict}[/]")
        return "transport_error" if _note_call_error(made, spec.dir_name) else "ok"

    return place


def _pass_rate_line(batch_dir: Path) -> str:
    """Return the pass rate and the other buckets of a batch, read from its ``report.json``."""
    summary = json.loads((batch_dir / "report.json").read_text(encoding="utf-8"))
    counts = summary["counts"]
    graded = counts["pass"] + counts["fail"]
    rate = summary["pass_rate"]
    head = f"{counts['pass']}/{graded} ({rate:.0%})" if rate is not None else "n/a"
    rest = ", ".join(f"{counts[k]} {k}" for k in ("undetermined", "artifact", "errored"))
    return f"pass rate: {head}; {rest}"


async def _run_batch(
    settings: Settings,
    choices: _Choices,
    scenario: Scenario,
    personas: dict[str, Persona],
    specs: list[RunSpec],
    parallel: int,
    batch_dir: Path,
    chat: ChatModel,
    preflight: bool,
) -> int:
    """Run the preflight (unless skipped), then every call; return the exit code.

    The preflight uses the voice of the first persona. The code is 1 when any call ended in an
    error or any judging failed, 2 when the preflight failed, and 0 otherwise (an agent that
    fails its scenario is still a successful run).
    """
    if preflight:
        first = personas[specs[0].persona_id]
        brief = _with_voice(
            dataclasses.replace(preflight_brief(), voice=dict(first.voice)), choices.voice
        )
        if not await _preflight(settings, choices, brief, batch_dir / "preflight"):
            console.print("Preflight failed; no calls were placed.", style="red")
            return EXIT_USAGE
    batch_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "scenario": scenario.id,
        "target": choices.target,
        "caller": choices.caller,
        "judge": choices.judge,
        "bugs": sorted(choices.bugs),
        "started": datetime.now().astimezone().isoformat(),
        "finished": None,
    }
    meta_path = batch_dir / "batch.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    place = _make_place(settings, choices, scenario, personas, chat, _find_pricing())
    results = await run_matrix(specs, place, batch_dir, parallel)
    meta["finished"] = datetime.now().astimezone().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report_path = write_batch_report(batch_dir)
    console.print(_pass_rate_line(batch_dir))
    console.print(f"report:  {escape(str(report_path))}", soft_wrap=True)
    return 0 if all(result == "ok" for result in results.values()) else 1


def _load_inputs(
    scenario_path: Path, persona_ids: list[str]
) -> tuple[Scenario, dict[str, Persona]]:
    """Load the scenario and every persona, exiting with one line if a file or a field is wrong."""
    files = {pid: persona_path(scenario_path, pid) for pid in persona_ids}
    if not scenario_path.is_file():
        raise _fail(f"Scenario file not found: {scenario_path}")
    for path in files.values():
        if not path.is_file():
            raise _fail(f"Persona file not found: {path}")
    scenario = load_scenario(scenario_path)
    try:
        load_reference_agent([spec.name for spec in scenario.fields])
    except KeyError as exc:
        raise _fail(
            f"Scenario field {exc.args[0]!r} is not defined by the reference agent's intake.yaml"
        ) from exc
    return scenario, {pid: load_persona(path) for pid, path in files.items()}


def _parse_personas(csv: str) -> list[str]:
    """Return the persona ids in ``csv``, or exit if it names none."""
    ids = [name.strip() for name in csv.split(",") if name.strip()]
    if not ids:
        raise _fail("--personas needs at least one persona id")
    return ids


@app.command()
def run(
    scenario_path: Annotated[Path, typer.Argument(help="Scenario YAML file to run.")],
    target: TargetOption,
    caller: CallerOption,
    caller_voice: VoiceOption = None,
    personas: Annotated[
        str,
        typer.Option(
            help="Comma list of persona ids (personas/<id>.yaml beside the scenario); "
            "call i uses the (i mod count)th."
        ),
    ] = "baseline",
    n: Annotated[
        int, typer.Option("-n", min=1, help="Number of calls, with consecutive seeds.")
    ] = 1,
    seed: Annotated[int, typer.Option(help="Seed of the first call; later calls add one.")] = 1,
    parallel: Annotated[
        int, typer.Option("--parallel", min=1, help="How many calls to place at once.")
    ] = 1,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help=f"Allow more than {MAX_NOVA_PARALLEL} parallel Nova Sonic calls."
        ),
    ] = False,
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
    """Place simulated calls against an agent, record them, grade each, and report the batch."""
    settings = load_settings()
    choices = _Choices(
        target=_check_name(target, TARGETS, "--target"),
        caller=_check_name(caller, CALLERS, "--caller"),
        judge=_check_name(judge or settings.judge, CHAT, "--judge"),
        voice=caller_voice,
        bugs=_parse_bugs(bugs, fix_bugs),
    )
    if (
        parallel > MAX_NOVA_PARALLEL
        and not force
        and {choices.target, choices.caller} == {"nova-sonic"}
    ):
        raise _fail(
            f"--parallel {parallel} is above {MAX_NOVA_PARALLEL}, the most Nova Sonic sessions "
            "run at once without throttling; pass --force to go on."
        )
    _need_aws(
        settings,
        nova_sonic="nova-sonic" in (choices.target, choices.caller),
        bedrock_judge=choices.judge == "bedrock",
    )
    persona_ids = _parse_personas(personas)
    scenario, persona_defs = _load_inputs(scenario_path, persona_ids)
    chat = _make_judge(choices.judge, settings)
    batch_dir = settings.runs_dir / f"{_stamp()}-{scenario.id}"
    specs = expand_runs(persona_ids, n, seed)
    code = asyncio.run(
        _run_batch(
            settings,
            choices,
            scenario,
            persona_defs,
            specs,
            parallel,
            batch_dir,
            chat,
            not skip_preflight,
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


def _interrupts_expected(run_dir: Path) -> bool:
    """Say whether the run's persona interrupts on purpose; ``False`` if the run saved none."""
    saved = run_dir / "persona.json"
    if not saved.exists():
        return False
    return Persona.model_validate_json(saved.read_text(encoding="utf-8")).behaviours.interrupts


def _endpoints_of(run_dir: Path) -> tuple[str, str]:
    """Return the target and caller names the run's batch recorded, for pricing it again."""
    try:
        meta = json.loads((run_dir.resolve().parent / "batch.json").read_text(encoding="utf-8"))
        return str(meta["target"]), str(meta["caller"])
    except (OSError, ValueError, KeyError):
        return "nova-sonic", "nova-sonic"


@app.command()
def judge(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to grade again.")],
    judge: JudgeOption = None,
) -> None:
    """Find a recorded run's incidents again, grade it, and rewrite its files and report."""
    settings = load_settings()
    scenario = _load_run(run_dir)
    name = _check_name(judge or settings.judge, CHAT, "--judge")
    _need_aws(settings, nova_sonic=False, bedrock_judge=name == "bedrock")
    chat = _make_judge(name, settings)
    target, caller = _endpoints_of(run_dir)
    choices = _Choices(target, caller, name, voice=None, bugs=frozenset())
    result = asyncio.run(
        _grade(
            run_dir,
            scenario,
            chat,
            choices,
            _find_pricing(),
            interrupts_expected=_interrupts_expected(run_dir),
        )
    )
    if result is None:
        raise typer.Exit(1)
    _show_verdict(result, run_dir, run_dir / "report.md")
    if _note_call_error(run_dir, run_dir.name):
        raise typer.Exit(1)


def _has_runs(directory: Path) -> bool:
    """Say whether ``directory`` holds run directories (each has an ``outcome.json``)."""
    return any((p / "outcome.json").is_file() for p in directory.iterdir() if p.is_dir())


@app.command()
def report(
    directory: Annotated[
        Path, typer.Argument(help="A run directory, or a batch directory of runs.")
    ],
) -> None:
    """Render a run's report.md, or a batch's report.md and report.json, again from its files."""
    if (directory / "judge.json").exists():
        path = write_run_report(directory)
    elif directory.is_dir() and _has_runs(directory):
        path = write_batch_report(directory)
    elif (directory / "outcome.json").is_file():
        raise _fail(f"No judge.json in {directory}; run callcheck judge {directory} first.")
    else:
        raise _fail(f"{directory} is neither a run directory nor a batch directory of runs.")
    console.print(f"report:  {escape(str(path))}", soft_wrap=True)


@app.command()
def adapters() -> None:
    """List the targets, callers and judge models that can be selected by name."""
    for kind, registry in (("targets", TARGETS), ("callers", CALLERS), ("judges", CHAT)):
        console.print(f"{kind:<8}{', '.join(sorted(registry))}")
