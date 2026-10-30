"""Unit tests for the Call mixer: pacing, recording, barge-in flush, grace, and the watchdog."""

import asyncio
from collections.abc import AsyncIterator

from callcheck.adapters.callers.fake import FakeCaller
from callcheck.adapters.fake_channel import FakeChannel
from callcheck.adapters.targets.fake import FakeTarget
from callcheck.core.audio import FRAME_MS, frame_bytes, rms_dbfs, tone
from callcheck.core.call import Call, CallSettings
from callcheck.core.clock import VirtualClock
from callcheck.core.events import CallEvent, endpoint_event
from callcheck.core.monitor import NUDGE_TEXT, TurnMonitor
from callcheck.core.scenario import CallConfig
from callcheck.core.turns import decisive_incident, detect_incidents

ANSWERS = {"first_name": "Maria", "phone": "415 555 0134"}
FIELDS = ["first_name", "phone"]


class ListSink:
    """In-memory sink that keeps events, counts frames, and counts audible agent frames."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.events: list[CallEvent] = []
        self.frames = 0
        self.agent_audible_frames = 0
        self.first_audible_agent_frame: int | None = None
        self.last_audible_agent_frame: int | None = None
        self.closes = 0
        self.closed_with: dict | None = None

    def record(self, ev: CallEvent) -> None:
        """Keep one event."""
        self.events.append(ev)

    def write_audio(self, left: bytes, right: bytes) -> None:
        """Count one frame, and count it as audible when the agent channel carries sound."""
        if rms_dbfs(right) > -60:
            self.agent_audible_frames += 1
            self.last_audible_agent_frame = self.frames
            if self.first_audible_agent_frame is None:
                self.first_audible_agent_frame = self.frames
        self.frames += 1

    def close(self, outcome: dict) -> None:
        """Keep the outcome and count the close."""
        self.closes += 1
        self.closed_with = outcome


class ScriptedEndpoint:
    """Endpoint that emits a fixed list of events from ``start`` and ignores incoming audio."""

    input_rate = 16000
    output_rate = 24000

    def __init__(self, script: list[CallEvent]) -> None:
        """Remember the events to emit once the call starts."""
        self._script = script
        self._queue: asyncio.Queue[CallEvent] = asyncio.Queue()
        self.closed = False

    async def start(self) -> None:
        """Queue the scripted events."""
        for ev in self._script:
            self._queue.put_nowait(ev)

    async def send_audio(self, pcm: bytes) -> None:
        """Ignore the audio."""

    async def events(self) -> AsyncIterator[CallEvent]:
        """Yield queued events until a ``closed`` event."""
        while True:
            ev = await self._queue.get()
            yield ev
            if ev.kind == "closed":
                return

    async def close(self) -> None:
        """Queue ``closed`` once."""
        if not self.closed:
            self.closed = True
            self._queue.put_nowait(endpoint_event("harness", "closed"))

    def result(self) -> dict | None:
        """Report no result."""
        return None

    async def nudge(self, instruction: str) -> bool:
        """Accept any nudge."""
        return True


class SlowStartEndpoint(ScriptedEndpoint):
    """Scripted endpoint whose connection takes ``connect_s`` seconds on the given clock."""

    def __init__(self, script: list[CallEvent], clock: VirtualClock, connect_s: float) -> None:
        """Remember the clock to advance and how long connecting takes."""
        super().__init__(script)
        self._clock = clock
        self._connect_s = connect_s

    async def start(self) -> None:
        """Spend the connect time, then queue the scripted events."""
        await self._clock.sleep_until(self._clock.now() + self._connect_s)
        await super().start()


class FailingCloseEndpoint(ScriptedEndpoint):
    """Scripted endpoint whose ``close`` raises."""

    async def close(self) -> None:
        """Fail instead of closing."""
        raise RuntimeError("close blew up")


def audio_frames(side: str, ms: int) -> list[CallEvent]:
    """Return ``ms`` of tone as 20 ms ``audio_out`` events on ``side``."""
    pcm = tone(660.0, ms, 24000)
    step = frame_bytes(24000)
    return [
        endpoint_event(side, "audio_out", pcm=pcm[i : i + step]) for i in range(0, len(pcm), step)
    ]


def fake_call(settings: CallSettings, behaviours: set[str] = frozenset()) -> tuple[Call, ListSink]:
    """Build a call between the fake caller and fake target with a fresh sink."""
    channel = FakeChannel()
    sink = ListSink()
    call = Call(
        caller=FakeCaller(channel, ANSWERS),
        target=FakeTarget(channel, FIELDS, behaviours=behaviours),
        sink=sink,
        clock=VirtualClock(),
        settings=settings,
    )
    return call, sink


async def test_full_fake_call_ends_caller_done_and_records_both_channels():
    call, sink = fake_call(CallSettings(max_seconds=60))
    outcome = await call.run()
    assert outcome.reason == "caller_done"
    assert sink.frames > 100 and sink.closed_with["reason"] == "caller_done"
    assert sink.closes == 1
    kinds = [e.kind for e in sink.events]
    assert kinds[0] == "call_start" and kinds[-1] == "call_end"
    assert "speech_start" in kinds and "audio_start" in kinds and "result" in kinds
    assert "audio_out" not in kinds
    assert all(e.t_ms >= 0 for e in sink.events)
    speech = [e for e in sink.events if e.kind in ("speech_start", "speech_end")]
    assert speech and all(isinstance(e.payload["energy"], float) for e in speech)


async def test_speech_end_is_stamped_at_the_first_silent_frame_not_the_vad_decision():
    sink = ListSink()
    await Call(
        caller=ScriptedEndpoint([]),
        target=ScriptedEndpoint(audio_frames("agent", 1000)),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=4),
    ).run()
    [end] = [e for e in sink.events if e.side == "agent" and e.kind == "speech_end"]
    first_silent_ms = (sink.last_audible_agent_frame + 1) * FRAME_MS
    # The VAD decides 25 quiet frames (500 ms) later; the stamp must be the speech boundary.
    assert abs(end.t_ms - first_silent_ms) <= 2 * FRAME_MS
    assert end.t_ms < first_silent_ms + 500 - 5 * FRAME_MS


async def test_stalled_target_ends_with_watchdog_at_twice_stall():
    call, sink = fake_call(CallSettings(stall_s=2, max_seconds=60), {"stall_after:first_name"})
    outcome = await call.run()
    assert outcome.reason == "stalled"
    assert 4000 <= outcome.duration_ms <= 4000 + 3 * 20 + 2500
    assert any(e.kind == "watchdog" for e in sink.events)


async def test_interrupted_event_flushes_speakers_buffer():
    target = ScriptedEndpoint(
        [*audio_frames("agent", 2000), endpoint_event("agent", "interrupted")]
    )
    sink = ListSink()
    outcome = await Call(
        caller=ScriptedEndpoint([]),
        target=target,
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=3),
    ).run()
    assert outcome.reason == "max_duration"
    assert sink.agent_audible_frames < 10


async def test_max_duration_ends_call():
    call, sink = fake_call(CallSettings(stall_s=100, max_seconds=3), {"stall_after:first_name"})
    outcome = await call.run()
    assert outcome.reason == "max_duration"
    assert 3000 <= outcome.duration_ms <= 3000 + 2 * 20


async def test_done_with_queued_agent_audio_respects_grace():
    sink = ListSink()
    outcome = await Call(
        caller=ScriptedEndpoint([endpoint_event("caller", "done")]),
        target=ScriptedEndpoint(audio_frames("agent", 5000)),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(grace_ms=1500, max_seconds=60),
    ).run()
    assert outcome.reason == "caller_done"
    assert 60 <= sink.agent_audible_frames <= 1500 // 20 + 2


async def test_done_ends_early_once_both_playout_buffers_drain():
    sink = ListSink()
    outcome = await Call(
        caller=ScriptedEndpoint([*audio_frames("caller", 200), endpoint_event("caller", "done")]),
        target=ScriptedEndpoint([]),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(grace_ms=1500, max_seconds=60),
    ).run()
    assert outcome.reason == "caller_done"
    assert outcome.duration_ms < 600


async def test_target_closed_ends_the_call():
    sink = ListSink()
    outcome = await Call(
        caller=ScriptedEndpoint([]),
        target=ScriptedEndpoint([endpoint_event("agent", "closed")]),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=60),
    ).run()
    assert outcome.reason == "target_closed"


async def test_unrecoverable_error_event_ends_the_call_with_its_message():
    sink = ListSink()
    outcome = await Call(
        caller=ScriptedEndpoint([]),
        target=ScriptedEndpoint(
            [endpoint_event("agent", "error", message="socket died", recoverable=False)]
        ),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=60),
    ).run()
    assert outcome.reason == "error" and outcome.error == "socket died"
    assert sink.closed_with == {
        "reason": "error",
        "duration_ms": outcome.duration_ms,
        "error": "socket died",
    }


async def test_playout_overflow_is_recorded_as_a_recoverable_error():
    sink = ListSink()
    await Call(
        caller=ScriptedEndpoint([]),
        target=ScriptedEndpoint(audio_frames("agent", 31000)),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=1),
    ).run()
    overflow = [e for e in sink.events if e.kind == "error"]
    assert overflow and overflow[0].payload["message"] == "playout overflow"
    assert overflow[0].payload["dropped"] > 0


async def test_connect_time_is_recorded_as_silence_so_audio_and_events_line_up():
    clock = VirtualClock()
    sink = ListSink()
    await Call(
        caller=ScriptedEndpoint([]),
        target=SlowStartEndpoint(audio_frames("agent", 200), clock, connect_s=1.5),
        sink=sink,
        clock=clock,
        settings=CallSettings(max_seconds=3),
    ).run()
    start = next(e for e in sink.events if e.kind == "audio_start")
    first = sink.first_audible_agent_frame
    assert first is not None and 75 <= first <= 77
    assert abs(start.t_ms - first * 20) <= 20


async def test_a_failing_close_does_not_stop_the_other_close_or_the_sink():
    caller = FailingCloseEndpoint([])
    target = ScriptedEndpoint([endpoint_event("agent", "closed")])
    sink = ListSink()
    outcome = await Call(
        caller=caller,
        target=target,
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=60),
    ).run()
    assert outcome.reason == "target_closed"
    assert target.closed
    assert sink.closes == 1 and sink.closed_with["reason"] == "target_closed"
    failures = [e for e in sink.events if e.kind == "error"]
    assert failures and "close failed" in failures[0].payload["message"]


async def test_long_burst_keeps_the_watchdog_quiet_while_it_plays_out():
    sink = ListSink()
    outcome = await Call(
        caller=ScriptedEndpoint([]),
        target=ScriptedEndpoint(audio_frames("agent", 10000)),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(stall_s=2, max_seconds=60),
    ).run()
    assert outcome.reason == "stalled"
    assert 10000 + 3900 <= outcome.duration_ms <= 10000 + 4000 + 200
    assert sink.agent_audible_frames >= 490


FORM_FIELDS = ["first_name", "last_name", "phone"]
FORM_ANSWERS = {"first_name": "Maria", "last_name": "Lopez", "phone": "415 555 0134"}


def nudging_call(
    target_kwargs: dict, *, nudgeable: bool = True
) -> tuple[Call, ListSink, FakeCaller]:
    """Build a monitored call whose target is the fake form agent with ``target_kwargs``."""
    channel = FakeChannel()
    sink = ListSink()
    caller = FakeCaller(channel, FORM_ANSWERS, nudgeable=nudgeable)
    call = Call(
        caller=caller,
        target=FakeTarget(channel, FORM_FIELDS, **target_kwargs),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=120),
        monitor=TurnMonitor(CallConfig()),
    )
    return call, sink, caller


def of_kind(sink: ListSink, kind: str) -> list[CallEvent]:
    """Return the recorded events of one kind."""
    return [e for e in sink.events if e.kind == kind]


async def test_forced_stall_is_broken_by_nudge():
    call, sink, _ = nudging_call({"go_silent_after": 2})
    outcome = await call.run()
    assert outcome.reason == "caller_done"
    [nudge] = of_kind(sink, "nudge")
    assert nudge.side == "harness" and nudge.payload == {"label": "no_response", "n": 1}
    incidents = detect_incidents(sink.events, CallConfig(), caller_interrupts_expected=False)
    missed = [i for i in incidents if i.label == "no_response"]
    assert len(missed) == 1 and missed[0].attribution == "agent"
    assert not any(e.kind == "watchdog" for e in sink.events)


async def test_nudge_waits_for_nudge_after_s_of_silence():
    call, sink, _ = nudging_call({"go_silent_after": 2})
    await call.run()
    [nudge] = of_kind(sink, "nudge")
    last_answer = [
        e
        for e in sink.events
        if e.side == "caller" and e.kind == "speech_end" and e.t_ms < nudge.t_ms
    ][-1]
    assert 8000 <= nudge.t_ms - last_answer.t_ms <= 8000 + 200


async def test_max_nudges_ends_call():
    call, sink, _ = nudging_call({"go_silent_after": 2, "wake_on_nudge": False})
    outcome = await call.run()
    assert outcome.reason == "stalled"
    assert [e.payload["n"] for e in of_kind(sink, "nudge")] == [1, 2]
    [dog] = of_kind(sink, "watchdog")
    assert dog.payload == {"reason": "max_nudges", "label": "no_response"}
    incidents = detect_incidents(sink.events, CallConfig(), caller_interrupts_expected=False)
    decisive = decisive_incident(incidents)
    assert decisive is not None and decisive.label == "no_response"


async def test_nudge_refused_ends_call():
    call, sink, _ = nudging_call({"go_silent_after": 2}, nudgeable=False)
    outcome = await call.run()
    assert outcome.reason == "stalled"
    assert len(of_kind(sink, "nudge")) == 1
    [dog] = of_kind(sink, "watchdog")
    assert dog.payload == {"reason": "nudge_refused", "label": "no_response"}
    incidents = detect_incidents(sink.events, CallConfig(), caller_interrupts_expected=False)
    decisive = decisive_incident(incidents)
    assert decisive is not None and decisive.label == "no_response"
    assert decisive.attribution == "agent"


async def test_the_nudge_instruction_reaches_the_caller():
    call, _, caller = nudging_call({"go_silent_after": 2})
    heard: list[str] = []
    original = caller.nudge

    async def spy(instruction: str) -> bool:
        heard.append(instruction)
        return await original(instruction)

    caller.nudge = spy
    await call.run()
    assert heard == [NUDGE_TEXT]


class SlowNudgeCaller(FakeCaller):
    """Fake caller whose nudge waits until the mixer has made ``wait_frames`` more frames."""

    def __init__(self, *args, sink: ListSink, wait_frames: int, **kwargs) -> None:
        """Remember the sink to count frames on and how many frames to wait for."""
        super().__init__(*args, **kwargs)
        self._sink = sink
        self._wait_frames = wait_frames
        self.frames_during_nudge: int | None = None

    async def nudge(self, instruction: str) -> bool:
        """Yield to the mixer until enough frames passed, note the count, then speak."""
        before = self._sink.frames
        for _ in range(10 * self._wait_frames):
            if self._sink.frames - before >= self._wait_frames:
                break
            await asyncio.sleep(0)
        self.frames_during_nudge = self._sink.frames - before
        return await super().nudge(instruction)


async def test_a_slow_nudge_does_not_stall_the_mixer():
    channel = FakeChannel()
    sink = ListSink()
    caller = SlowNudgeCaller(channel, FORM_ANSWERS, sink=sink, wait_frames=100)
    outcome = await Call(
        caller=caller,
        target=FakeTarget(channel, FORM_FIELDS, go_silent_after=2),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=120),
        monitor=TurnMonitor(CallConfig()),
    ).run()
    assert outcome.reason == "caller_done"
    assert len(of_kind(sink, "nudge")) == 1
    # The ticker mixed 100 more frames (2 s of call time) while the nudge was still pending.
    assert caller.frames_during_nudge == 100


async def test_a_call_without_a_monitor_never_nudges():
    call, sink = fake_call(CallSettings(stall_s=2, max_seconds=60), {"stall_after:first_name"})
    outcome = await call.run()
    assert outcome.reason == "stalled"
    assert not of_kind(sink, "nudge")


async def test_a_healthy_monitored_call_is_never_nudged():
    channel = FakeChannel()
    sink = ListSink()
    outcome = await Call(
        caller=FakeCaller(channel, FORM_ANSWERS),
        target=FakeTarget(channel, FORM_FIELDS),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=120),
        monitor=TurnMonitor(CallConfig()),
    ).run()
    assert outcome.reason == "caller_done"
    assert not of_kind(sink, "nudge") and not of_kind(sink, "watchdog")


class RaisingNudgeCaller(FakeCaller):
    """Fake caller whose nudge fails."""

    async def nudge(self, instruction: str) -> bool:
        """Raise instead of speaking."""
        raise RuntimeError("nudge boom")


async def test_a_nudge_that_raises_is_recorded_and_ends_the_call_as_refused():
    channel = FakeChannel()
    sink = ListSink()
    outcome = await Call(
        caller=RaisingNudgeCaller(channel, FORM_ANSWERS),
        target=FakeTarget(channel, FORM_FIELDS, go_silent_after=2),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=120),
        monitor=TurnMonitor(CallConfig()),
    ).run()
    assert outcome.reason == "stalled"
    [dog] = of_kind(sink, "watchdog")
    assert dog.payload == {"reason": "nudge_refused", "label": "no_response"}
    [err] = of_kind(sink, "error")
    assert err.payload["recoverable"] is True and "nudge boom" in err.payload["message"]


def agent_turn(text: str) -> list[CallEvent]:
    """Script one agent turn: its words, 1.5 s of audio, then ``turn_end`` right away."""
    return [
        endpoint_event("agent", "turn_start"),
        endpoint_event("agent", "transcript", role="agent", text=text, final=True),
        *audio_frames("agent", 1500),
        endpoint_event("agent", "turn_end"),
    ]


async def run_silent_caller_call(
    target_events: list[CallEvent],
) -> tuple[ListSink, list]:
    """Run a call whose agent says its piece and whose caller never speaks; return its incidents."""
    sink = ListSink()
    await Call(
        caller=ScriptedEndpoint([]),
        target=ScriptedEndpoint(target_events),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=120),
        monitor=TurnMonitor(CallConfig()),
    ).run()
    return sink, detect_incidents(sink.events, CallConfig(), caller_interrupts_expected=False)


async def test_a_statement_then_silence_is_a_dead_end_where_the_monitor_nudged():
    sink, incidents = await run_silent_caller_call(agent_turn("One moment please."))
    [nudge] = of_kind(sink, "nudge")
    assert nudge.payload["label"] == "dead_end"
    [found] = [i for i in incidents if i.label == "dead_end"]
    assert found.attribution == "agent"
    assert abs(nudge.t_ms - (found.t_ms + 6000)) <= 3 * FRAME_MS


async def test_a_question_then_silence_is_a_caller_stall_where_the_monitor_nudged():
    sink, incidents = await run_silent_caller_call(agent_turn("What is your name?"))
    [nudge] = of_kind(sink, "nudge")
    assert nudge.payload["label"] == "caller_stall"
    [found] = [i for i in incidents if i.label == "caller_stall"]
    assert found.attribution == "simulator"
    assert abs(nudge.t_ms - (found.t_ms + 6000)) <= 3 * FRAME_MS


async def test_a_fake_target_question_met_by_a_silent_caller_agrees_with_the_nudge():
    channel = FakeChannel()
    sink = ListSink()
    await Call(
        caller=FakeCaller(channel, FORM_ANSWERS, script=[], nudgeable=False),
        target=FakeTarget(channel, FORM_FIELDS),
        sink=sink,
        clock=VirtualClock(),
        settings=CallSettings(max_seconds=120),
        monitor=TurnMonitor(CallConfig()),
    ).run()
    [nudge] = of_kind(sink, "nudge")
    incidents = detect_incidents(sink.events, CallConfig(), caller_interrupts_expected=False)
    [found] = [i for i in incidents if i.label == "caller_stall"]
    assert nudge.payload["label"] == "caller_stall"
    assert abs(nudge.t_ms - (found.t_ms + 6000)) <= 3 * FRAME_MS
