"""Unit tests for the fake target, the fake caller, their text channel, and the registries."""

import pytest

from callcheck.adapters.callers import CALLERS, make_caller
from callcheck.adapters.callers.fake import FakeCaller
from callcheck.adapters.fake_channel import FakeChannel
from callcheck.adapters.targets import TARGETS, make_target
from callcheck.adapters.targets.fake import FakeTarget
from callcheck.core.audio import Resampler, frame_bytes, silence
from callcheck.core.endpoints import Caller, TargetAgent
from callcheck.core.events import CallEvent

ANSWERS = {"first_name": "Maria", "phone": "415 555 0134"}
CORRECTIONS = {"phone": "415 555 0143"}


async def pump(target: TargetAgent, caller: Caller, max_frames: int) -> list[CallEvent]:
    """Run both fakes for ``max_frames`` 20 ms frames without a mixer; return non-audio events."""
    await target.start()
    await caller.start()
    lanes = [
        (target, caller, Resampler(24000, 16000), bytearray()),
        (caller, target, Resampler(24000, 16000), bytearray()),
    ]
    seen: list[CallEvent] = []
    for _ in range(max_frames):
        for source, sink, resampler, pending in lanes:
            for ev in source.drain():
                if ev.kind == "audio_out":
                    pending += resampler.process(ev.payload["pcm"])
                else:
                    seen.append(ev)
            if len(pending) >= frame_bytes(16000):
                frame = bytes(pending[: frame_bytes(16000)])
                del pending[: frame_bytes(16000)]
            else:
                frame = silence(16000)
            await sink.send_audio(frame)
    return seen


def make_pair(behaviours: set[str] = frozenset()) -> tuple[FakeTarget, FakeCaller]:
    """Build a target and caller joined by one channel, the caller correcting its phone number."""
    channel = FakeChannel()
    target = FakeTarget(channel, ["first_name", "phone"], behaviours=behaviours)
    return target, FakeCaller(channel, ANSWERS, corrections=CORRECTIONS)


async def test_fakes_complete_a_two_field_form():
    target, caller = make_pair()
    events = await pump(target, caller, max_frames=600)
    assert target.result() == {"first_name": "Maria", "phone": "415 555 0134"}
    assert any(e.kind == "done" for e in events if e.side == "caller")
    assert any(e.kind == "result" for e in events if e.side == "agent")


async def test_fake_target_ignore_correction_keeps_first_value():
    target, caller = make_pair({"ignore_correction"})
    await pump(target, caller, max_frames=600)
    assert target.result()["phone"] == "415 555 0143"


async def test_fake_target_stalls_after_field():
    target, caller = make_pair({"stall_after:first_name"})
    events = await pump(target, caller, max_frames=600)
    assert target.result() == {"first_name": "Maria"}
    agent_lines = [
        e.payload["text"] for e in events
        if e.side == "agent" and e.kind == "transcript" and e.payload["role"] == "agent"
    ]
    assert agent_lines and not any("phone" in line for line in agent_lines)
    assert not any(e.kind == "done" for e in events)


async def test_greeting_is_a_transcript_then_20ms_audio_frames():
    target, _ = make_pair()
    await target.start()
    events = target.drain()
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["turn_start", "transcript"] and kinds[-1] == "turn_end"
    assert events[1].payload == {
        "role": "agent",
        "text": "Thanks for calling the clinic. What is your first name?",
        "final": True,
    }
    audio = [e.payload["pcm"] for e in events if e.kind == "audio_out"]
    assert len(audio) == 30 and all(len(p) == frame_bytes(24000) for p in audio)


async def test_close_ends_the_event_stream_with_closed():
    target, _ = make_pair()
    await target.start()
    await target.close()
    kinds = [e.kind async for e in target.events()]
    assert kinds[-1] == "closed" and kinds.count("closed") == 1


async def test_caller_nudge_speaks_hello_and_is_accepted():
    _, caller = make_pair()
    assert await caller.nudge("say something") is True
    spoken = [e.payload["text"] for e in caller.drain() if e.kind == "transcript"]
    assert spoken == ["Hello?"]


def test_registries_build_the_fakes_by_name():
    channel = FakeChannel()
    assert isinstance(make_target("fake", channel=channel, fields=["phone"]), FakeTarget)
    assert isinstance(make_caller("fake", channel=channel, answers={}), FakeCaller)
    assert "fake" in TARGETS and "fake" in CALLERS


def test_unknown_registry_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="fake"):
        make_target("nope")
    with pytest.raises(KeyError, match="fake"):
        make_caller("nope")
