"""Unit tests for the Nova Sonic target and caller adapters, over a scripted session."""

from collections.abc import AsyncIterator
from dataclasses import replace

from callcheck.adapters.callers import CALLERS
from callcheck.adapters.callers.nova_sonic import NovaSonicCaller
from callcheck.adapters.targets import TARGETS
from callcheck.adapters.targets.nova_sonic import NovaSonicTarget
from callcheck.core.endpoints import CallerBrief
from callcheck.core.events import CallEvent, endpoint_event
from callcheck.reference_agent.loader import load_reference_agent

CONFIG = {"model_id": "amazon.nova-sonic-v1:0", "region": "us-east-1", "profile": "dev"}


class FakeSession:
    """Stands in for ``NovaSonicSession``: records its arguments and yields scripted events."""

    input_rate = 16000
    output_rate = 24000

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.script: list[CallEvent] = []
        self.started = False
        self.closed = False
        self.texts: list[str] = []
        self.audio: list[bytes] = []

    async def start(self) -> None:
        self.started = True

    async def send_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def events(self) -> AsyncIterator[CallEvent]:
        for ev in self.script:
            yield ev

    async def close(self) -> None:
        self.closed = True


def factory(script: list[CallEvent]) -> tuple[list[FakeSession], type[FakeSession]]:
    """Return a list that collects the built sessions and a session class playing ``script``."""
    built: list[FakeSession] = []

    class Scripted(FakeSession):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.script = script
            built.append(self)

    return built, Scripted


def brief(voice: dict[str, str] | None = None) -> CallerBrief:
    return CallerBrief("p1", "You are Sam.", voice or {})


async def test_target_triggers_greeting_and_emits_result_after_tool():
    agent = load_reference_agent(["first_name"])
    script = [
        endpoint_event("agent", "tool_call", name="save_field", arguments={}),
        endpoint_event("agent", "tool_result", name="save_field", result={"ok": True}),
    ]
    built, session_class = factory(script)
    target = NovaSonicTarget(agent, **CONFIG, session_factory=session_class)
    (fake,) = built

    await agent.tool_executor("save_field", {"name": "first_name", "value": "Ada"})
    await target.start()
    evs = [e async for e in target.events()]

    assert fake.started and fake.texts == ["(call connected)"]
    assert [e.kind for e in evs] == ["tool_call", "tool_result", "result"]
    assert evs[-1].side == "agent"
    assert evs[-1].payload == {"fields": {"first_name": "Ada"}, "submitted": False}
    assert target.result() == {"fields": {"first_name": "Ada"}, "submitted": False}


async def test_target_emits_no_result_event_while_nothing_is_saved():
    agent = load_reference_agent(["first_name"])
    script = [endpoint_event("agent", "tool_result", name="submit_form", result={"ok": False})]
    _, session_class = factory(script)
    target = NovaSonicTarget(agent, **CONFIG, session_factory=session_class)

    assert [e.kind async for e in target.events()] == ["tool_result"]


async def test_target_builds_its_session_from_the_agent_definition():
    agent = load_reference_agent(["first_name"])
    built, session_class = factory([])
    NovaSonicTarget(agent, **CONFIG, session_factory=session_class)
    NovaSonicTarget(
        replace(agent, voice={"nova_sonic": "amy"}),
        **CONFIG,
        session_factory=session_class,
    )

    default, custom = built
    assert default.kwargs == {
        **CONFIG,
        "side": "agent",
        "system_prompt": agent.system_prompt,
        "voice_id": "matthew",
        "tools": agent.tools,
        "tool_executor": agent.tool_executor,
    }
    assert custom.kwargs["voice_id"] == "amy"


async def test_target_delegates_audio_rates_and_close():
    built, session_class = factory([])
    agent = load_reference_agent(["first_name"])
    target = NovaSonicTarget(agent, **CONFIG, session_factory=session_class)
    (fake,) = built

    await target.send_audio(b"\x00\x01")
    await target.close()

    assert fake.audio == [b"\x00\x01"] and fake.closed
    assert (target.input_rate, target.output_rate) == (16000, 24000)


async def test_caller_emits_done_after_end_call_and_nudges():
    script = [
        endpoint_event("caller", "tool_call", name="end_call", arguments={}),
        endpoint_event("caller", "tool_result", name="end_call", result={"ok": True}),
    ]
    built, session_class = factory(script)
    caller = NovaSonicCaller(brief(), **CONFIG, session_factory=session_class)
    (fake,) = built

    await caller.start()
    evs = [e async for e in caller.events()]

    assert [e.kind for e in evs] == ["tool_call", "tool_result", "done"]
    assert evs[-1].side == "caller"
    assert await caller.nudge("Say hello?") is True
    assert fake.texts == ["Say hello?"]


async def test_caller_emits_done_only_once():
    end = [
        endpoint_event("caller", "tool_call", name="end_call", arguments={}),
        endpoint_event("caller", "tool_result", name="end_call", result={"ok": True}),
    ]
    _, session_class = factory(end + end)
    caller = NovaSonicCaller(brief(), **CONFIG, session_factory=session_class)

    kinds = [e.kind async for e in caller.events()]

    assert kinds.count("done") == 1


async def test_caller_ignores_other_tool_results():
    script = [endpoint_event("caller", "tool_result", name="other", result={})]
    _, session_class = factory(script)
    caller = NovaSonicCaller(brief(), **CONFIG, session_factory=session_class)

    assert [e.kind async for e in caller.events()] == ["tool_result"]


async def test_caller_builds_its_session_with_an_end_call_tool():
    built, session_class = factory([])
    NovaSonicCaller(brief(), **CONFIG, session_factory=session_class)
    NovaSonicCaller(brief({"nova_sonic": "amy"}), **CONFIG, session_factory=session_class)

    default, custom = built
    assert default.kwargs["side"] == "caller"
    assert default.kwargs["system_prompt"] == "You are Sam."
    assert default.kwargs["voice_id"] == "tiffany"
    assert custom.kwargs["voice_id"] == "amy"
    (tool,) = default.kwargs["tools"]
    assert tool.name == "end_call" and tool.input_schema == {"type": "object", "properties": {}}
    assert await default.kwargs["tool_executor"]("end_call", {}) == {"ok": True}


async def test_caller_nudge_returns_false_once_closed_or_unstarted():
    built, session_class = factory([])
    caller = NovaSonicCaller(brief(), **CONFIG, session_factory=session_class)
    (fake,) = built

    await caller.close()

    assert await caller.nudge("Too late") is False
    assert fake.texts == []

    class Unstarted(FakeSession):
        async def send_text(self, text: str) -> None:
            raise RuntimeError("NovaSonicSession.start() has not been called")

    assert await NovaSonicCaller(brief(), **CONFIG, session_factory=Unstarted).nudge("x") is False


async def test_caller_nudge_returns_false_after_the_session_reports_closed():
    script = [endpoint_event("caller", "closed")]
    built, session_class = factory(script)
    caller = NovaSonicCaller(brief(), **CONFIG, session_factory=session_class)
    (fake,) = built

    [_ async for _ in caller.events()]

    assert await caller.nudge("Too late") is False
    assert fake.texts == []


def test_registry_resolves_nova_sonic():
    assert "nova-sonic" in TARGETS and "nova-sonic" in CALLERS
