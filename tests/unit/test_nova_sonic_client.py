"""Unit tests for the Nova Sonic session client, driven by a scripted stream.

The scripted server events copy the shapes recorded in ``docs/research/nova-sonic-spike.md``:
output events name their content by ``contentId``, the generation stage sits in
``additionalModelFields`` on ``contentStart`` only, and a barge-in arrives as an ASSISTANT
TEXT block holding ``{ "interrupted" : true }`` that ends with ``stopReason: "INTERRUPTED"``.
"""

import asyncio
import base64
import json
import subprocess
import sys

import pytest

from callcheck.adapters.providers import nova_sonic_client
from callcheck.adapters.providers.nova_sonic_client import NovaSonicSession
from callcheck.core.endpoints import ToolSpec
from callcheck.core.events import CallEvent

IDS = {"completionId": "cmp-1", "promptName": "srv-prompt", "sessionId": "sess-1"}
MARKER = '{ "interrupted" : true }'


def out(name: str, **body: object) -> str:
    """Encode one server event the way the stream delivers it."""
    return json.dumps({"event": {name: {**IDS, **body}}})


def text_start(cid: str, role: str, stage: str) -> str:
    """A TEXT ``contentStart`` with its generation stage."""
    return out(
        "contentStart",
        additionalModelFields=json.dumps({"generationStage": stage}),
        contentId=cid,
        role=role,
        textOutputConfiguration={"mediaType": "text/plain"},
        type="TEXT",
    )


def text_output(cid: str, role: str, text: str) -> str:
    """A ``textOutput`` inside content ``cid``."""
    return out("textOutput", content=text, contentId=cid, role=role)


def content_end(cid: str, kind: str, stop: str) -> str:
    """A ``contentEnd`` for content ``cid`` of type ``kind``."""
    return out("contentEnd", contentId=cid, stopReason=stop, type=kind)


def text_block(cid: str, role: str, stage: str, text: str, stop: str) -> list[str]:
    """A whole TEXT content: start, one text output, end."""
    return [
        text_start(cid, role, stage),
        text_output(cid, role, text),
        content_end(cid, "TEXT", stop),
    ]


def audio_start(cid: str) -> str:
    """An ASSISTANT AUDIO ``contentStart`` (no stage on audio)."""
    return out(
        "contentStart",
        audioOutputConfiguration={
            "channelCount": 1,
            "encoding": "base64",
            "mediaType": "audio/lpcm",
            "sampleRateHertz": 24000,
            "sampleSizeBits": 16,
        },
        contentId=cid,
        role="ASSISTANT",
        type="AUDIO",
    )


def audio_output(cid: str, pcm: bytes) -> str:
    """One ``audioOutput`` chunk of 24 kHz PCM16."""
    return out(
        "audioOutput", content=base64.b64encode(pcm).decode(), contentId=cid, role="ASSISTANT"
    )


def tool_block(cid: str, name: str, arguments: dict, tool_use_id: str) -> list[str]:
    """A TOOL content carrying one ``toolUse``, ended with ``TOOL_USE``."""
    return [
        out(
            "contentStart",
            contentId=cid,
            role="TOOL",
            toolUseOutputConfiguration={"mediaType": "application/json"},
            type="TOOL",
        ),
        out(
            "toolUse",
            content=json.dumps(arguments),
            contentId=cid,
            role="TOOL",
            toolName=name,
            toolUseId=tool_use_id,
        ),
        content_end(cid, "TOOL", "TOOL_USE"),
    ]


def spoken_reply(tag: str, text: str, chunks: list[bytes]) -> list[str]:
    """An uninterrupted reply: SPECULATIVE text, its audio, then the FINAL text."""
    return [
        out("completionStart"),
        *text_block(f"{tag}-spec", "ASSISTANT", "SPECULATIVE", text, "PARTIAL_TURN"),
        audio_start(f"{tag}-audio"),
        *(audio_output(f"{tag}-audio", chunk) for chunk in chunks),
        content_end(f"{tag}-audio", "AUDIO", "END_TURN"),
        *text_block(f"{tag}-final", "ASSISTANT", "FINAL", text, "END_TURN"),
    ]


USAGE = {
    "details": {
        "delta": {
            "input": {"speechTokens": 0, "textTokens": 22},
            "output": {"speechTokens": 0, "textTokens": 0},
        },
        "total": {
            "input": {"speechTokens": 0, "textTokens": 22},
            "output": {"speechTokens": 0, "textTokens": 0},
        },
    },
    "totalInputTokens": 22,
    "totalOutputTokens": 0,
    "totalTokens": 22,
}
COMPLETION_END = out("completionEnd", stopReason="END_TURN")


class ScriptedStream:
    """Stand-in for the bidirectional stream: replays scripted server events, records sends.

    A ``None`` in the script ends the stream, as the SDK's ``receive()`` does; an exception
    in the script is raised from ``receive()``. Once the script runs out the stream stays
    open, like the real server, until the client closes its input; it then sends
    ``completionEnd`` and ends, unless ``ignore_close`` makes it hang instead.
    """

    def __init__(self, script: list[str | Exception | None], *, ignore_close: bool = False):
        """Queue the scripted server events."""
        self._script = list(script)
        self._ignore_close = ignore_close
        self._input_closed = asyncio.Event()
        self._ending = [COMPLETION_END, None]
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, event_json: str) -> None:
        """Record one client event."""
        self.sent.append(json.loads(event_json)["event"])

    async def receive(self) -> str | None:
        """Return the next scripted server event."""
        await asyncio.sleep(0)
        if self._script:
            item = self._script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        await self._input_closed.wait()
        if self._ignore_close:
            await asyncio.Event().wait()
        return self._ending.pop(0)

    async def close(self) -> None:
        """Mark the client's input as finished."""
        self.closed = True
        self._input_closed.set()

    def names(self) -> list[str]:
        """Return the name of every client event sent, in order."""
        return [next(iter(ev)) for ev in self.sent]

    def bodies(self, name: str) -> list[dict]:
        """Return the bodies of the client events called ``name``."""
        return [ev[name] for ev in self.sent if name in ev]


def session(
    stream: ScriptedStream,
    *,
    side: str = "agent",
    tools: list[ToolSpec] | None = None,
    executor=None,
) -> NovaSonicSession:
    """Build a session over ``stream``."""
    return NovaSonicSession(
        side=side,
        model_id="m",
        region="r",
        profile=None,
        system_prompt="You are a receptionist.",
        voice_id="matthew",
        tools=tools or [],
        tool_executor=executor,
        stream_factory=lambda: stream,
    )


async def run_to_end(s: NovaSonicSession) -> list[CallEvent]:
    """Start ``s`` and collect its events until it closes."""
    await s.start()
    return [ev async for ev in s.events()]


def kinds(events: list[CallEvent]) -> list[str]:
    """Return the kind of each event."""
    return [ev.kind for ev in events]


async def test_start_sends_the_open_sequence_in_order():
    stream = ScriptedStream([])
    s = session(stream)
    await s.start()
    await s.close()
    assert stream.names()[:6] == [
        "sessionStart",
        "promptStart",
        "contentStart",
        "textInput",
        "contentEnd",
        "contentStart",
    ]
    prompt = stream.bodies("promptStart")[0]
    assert prompt["audioOutputConfiguration"]["sampleRateHertz"] == 24000
    assert prompt["audioOutputConfiguration"]["voiceId"] == "matthew"
    assert "toolConfiguration" not in prompt
    system, audio = stream.bodies("contentStart")[:2]
    assert (system["role"], system["type"], system["interactive"]) == ("SYSTEM", "TEXT", False)
    assert stream.bodies("textInput")[0]["content"] == "You are a receptionist."
    assert (audio["role"], audio["type"], audio["interactive"]) == ("USER", "AUDIO", True)
    assert audio["audioInputConfiguration"]["sampleRateHertz"] == 16000
    assert system["contentName"] != audio["contentName"]
    names = {
        body["promptName"] for ev in stream.sent for body in ev.values() if "promptName" in body
    }
    assert names == {prompt["promptName"]}


async def test_prompt_start_carries_tools_with_the_schema_as_a_json_string():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    stream = ScriptedStream([])
    s = session(stream, tools=[ToolSpec("save_field", "Save one field.", schema)])
    await s.start()
    await s.close()
    prompt = stream.bodies("promptStart")[0]
    assert prompt["toolUseOutputConfiguration"] == {"mediaType": "application/json"}
    [tool] = prompt["toolConfiguration"]["tools"]
    assert tool["toolSpec"]["name"] == "save_field"
    assert tool["toolSpec"]["description"] == "Save one field."
    assert json.loads(tool["toolSpec"]["inputSchema"]["json"]) == schema


async def test_send_audio_base64_encodes_into_the_open_audio_content():
    stream = ScriptedStream([])
    s = session(stream)
    await s.start()
    await s.send_audio(b"\x00" * 640)
    audio_name = stream.bodies("contentStart")[1]["contentName"]
    assert stream.sent[-1] == {
        "audioInput": {
            "promptName": stream.bodies("promptStart")[0]["promptName"],
            "contentName": audio_name,
            "content": base64.b64encode(b"\x00" * 640).decode(),
        }
    }
    await s.close()


async def test_send_text_sends_a_complete_interactive_user_turn():
    stream = ScriptedStream([])
    s = session(stream)
    await s.start()
    before = len(stream.sent)
    await s.send_text("(call connected)")
    start, text, end = stream.sent[before:]
    name = start["contentStart"]["contentName"]
    assert start["contentStart"]["role"] == "USER"
    assert start["contentStart"]["type"] == "TEXT"
    assert start["contentStart"]["interactive"] is True
    assert text["textInput"] == {
        "promptName": start["contentStart"]["promptName"],
        "contentName": name,
        "content": "(call connected)",
    }
    assert end["contentEnd"]["contentName"] == name
    assert name not in {body["contentName"] for body in stream.bodies("contentStart")[:2]}
    await s.close()


async def test_spoken_reply_maps_to_turn_transcripts_and_audio():
    chunks = [b"\x01\x02" * 1920, b"\x03\x04" * 960]
    stream = ScriptedStream([*spoken_reply("r1", "Hello, how can I help?", chunks), None])
    events = await run_to_end(session(stream))
    assert kinds(events) == [
        "turn_start",
        "transcript",
        "audio_out",
        "audio_out",
        "turn_end",
        "transcript",
        "closed",
    ]
    speculative, final = (ev.payload for ev in events if ev.kind == "transcript")
    assert speculative == {"role": "agent", "text": "Hello, how can I help?", "final": False}
    assert final == {"role": "agent", "text": "Hello, how can I help?", "final": True}
    assert [ev.payload["pcm"] for ev in events if ev.kind == "audio_out"] == chunks
    assert {ev.side for ev in events} == {"agent"}


async def test_multi_part_reply_is_one_turn():
    script = [
        *text_block("a", "ASSISTANT", "SPECULATIVE", "One.", "PARTIAL_TURN"),
        audio_start("a-audio"),
        audio_output("a-audio", b"\x00\x01"),
        content_end("a-audio", "AUDIO", "PARTIAL_TURN"),
        *text_block("b", "ASSISTANT", "SPECULATIVE", "Two.", "PARTIAL_TURN"),
        audio_start("b-audio"),
        audio_output("b-audio", b"\x00\x02"),
        content_end("b-audio", "AUDIO", "END_TURN"),
        None,
    ]
    events = await run_to_end(session(ScriptedStream(script)))
    assert kinds(events).count("turn_start") == 1
    assert kinds(events).count("turn_end") == 1
    assert kinds(events)[-2:] == ["turn_end", "closed"]


async def test_tool_call_runs_the_executor_and_sends_the_result_back():
    calls = []

    async def executor(name: str, arguments: dict) -> dict:
        calls.append((name, arguments))
        return {"ok": True}

    script = [
        *tool_block("t1", "save_field", {"name": "first_name", "value": "Maria"}, "tool-7"),
        *spoken_reply("r1", "I have saved your first name.", [b"\x00\x00"]),
        None,
    ]
    stream = ScriptedStream(script)
    s = session(stream, tools=[ToolSpec("save_field", "d", {})], executor=executor)
    events = await run_to_end(s)
    assert calls == [("save_field", {"name": "first_name", "value": "Maria"})]
    assert kinds(events)[:4] == ["tool_call", "tool_result", "turn_start", "transcript"]
    assert events[0].payload == {
        "name": "save_field",
        "arguments": {"name": "first_name", "value": "Maria"},
    }
    assert events[1].payload == {"name": "save_field", "result": {"ok": True}}
    start, result, end = stream.sent[-3:]
    assert start["contentStart"]["role"] == "TOOL"
    assert start["contentStart"]["type"] == "TOOL"
    assert start["contentStart"]["interactive"] is False
    assert start["contentStart"]["toolResultInputConfiguration"]["toolUseId"] == "tool-7"
    name = start["contentStart"]["contentName"]
    assert result["toolResult"]["contentName"] == name
    assert json.loads(result["toolResult"]["content"]) == {"ok": True}
    assert end["contentEnd"]["contentName"] == name


async def test_failing_tool_reports_a_recoverable_error_and_still_answers_the_model():
    async def executor(name: str, arguments: dict) -> dict:
        raise ValueError("disk full")

    stream = ScriptedStream([*tool_block("t1", "save_field", {}, "tool-1"), None])
    s = session(stream, tools=[ToolSpec("save_field", "d", {})], executor=executor)
    events = await run_to_end(s)
    assert kinds(events) == ["tool_call", "error", "tool_result", "closed"]
    assert events[1].payload["recoverable"] is True
    assert "disk full" in events[1].payload["message"]
    assert events[2].payload["result"] == {"ok": False, "error": "disk full"}
    [sent] = stream.bodies("toolResult")
    assert json.loads(sent["content"]) == {"ok": False, "error": "disk full"}


async def test_tool_call_without_an_executor_answers_with_an_error():
    stream = ScriptedStream([*tool_block("t1", "save_field", {"name": "x"}, "tool-1"), None])
    events = await run_to_end(session(stream, tools=[ToolSpec("save_field", "d", {})]))
    assert kinds(events) == ["tool_call", "error", "tool_result", "closed"]
    assert events[2].payload["result"]["ok"] is False
    assert len(stream.bodies("toolResult")) == 1


async def test_malformed_tool_arguments_are_refused_without_calling_the_executor():
    calls = []

    async def executor(name: str, arguments: dict) -> dict:
        calls.append(arguments)
        return {"ok": True}

    [start, use, end] = tool_block("t1", "save_field", {}, "tool-1")
    use = out(
        "toolUse",
        content="not json",
        contentId="t1",
        role="TOOL",
        toolName="save_field",
        toolUseId="tool-1",
    )
    stream = ScriptedStream([start, use, end, None])
    s = session(stream, tools=[ToolSpec("save_field", "d", {})], executor=executor)
    events = await run_to_end(s)
    assert calls == []
    assert events[0].payload == {"name": "save_field", "arguments": {"raw": "not json"}}
    assert events[2].payload["result"] == {"ok": False, "error": "arguments are not a JSON object"}


async def test_audio_waiting_on_the_lock_is_dropped_once_closing_starts():
    stream = ScriptedStream([])
    s = session(stream)
    await s.start()
    late_frame = asyncio.create_task(s.send_audio(b"\x00" * 640))
    async with s._send_lock:
        closing = asyncio.create_task(s.close())
        await asyncio.sleep(0)
        assert not late_frame.done()
    await asyncio.gather(late_frame, closing)
    assert stream.names()[-3:] == ["contentEnd", "promptEnd", "sessionEnd"]
    assert "audioInput" not in stream.names()


async def test_interrupted_reply_ends_the_turn_and_drops_the_marker_text():
    script = [
        *text_block("a", "ASSISTANT", "SPECULATIVE", "One. Two. Three.", "PARTIAL_TURN"),
        audio_start("a-audio"),
        audio_output("a-audio", b"\x00\x01"),
        out("userSpeechStart", inputAudioOffsetMs=5280),
        content_end("a-audio", "AUDIO", "PARTIAL_TURN"),
        *text_block("m", "ASSISTANT", "FINAL", MARKER, "INTERRUPTED"),
        *text_block("f", "ASSISTANT", "FINAL", "One. Two.", "PARTIAL_TURN"),
        out("userSpeechEnd", inputAudioDetectionOffsetMs=8120, inputAudioOffsetMs=7900),
        *text_block("u", "USER", "FINAL", "hello? can you hear me?", "PARTIAL_TURN"),
        *spoken_reply("r2", "Yes, I can hear you.", [b"\x00\x02"]),
        None,
    ]
    events = await run_to_end(session(ScriptedStream(script)))
    assert kinds(events)[:6] == [
        "turn_start",
        "transcript",
        "audio_out",
        "heard_speech_start",
        "interrupted",
        "turn_end",
    ]
    texts = [ev.payload["text"] for ev in events if ev.kind == "transcript"]
    assert not any("interrupted" in text for text in texts)
    assert "One. Two." in texts and "hello? can you hear me?" in texts
    assert kinds(events).count("turn_start") == kinds(events).count("turn_end") == 2
    assert events[3].payload == {"input_offset_ms": 5280}
    [heard_end] = [ev for ev in events if ev.kind == "heard_speech_end"]
    assert heard_end.payload == {"input_offset_ms": 7900, "detection_offset_ms": 8120}


async def test_interrupted_marker_is_recognised_by_its_content_alone():
    script = [
        *text_block("a", "ASSISTANT", "SPECULATIVE", "Counting.", "PARTIAL_TURN"),
        *text_block("m", "ASSISTANT", "FINAL", MARKER, "END_TURN"),
        None,
    ]
    events = await run_to_end(session(ScriptedStream(script)))
    assert kinds(events) == ["turn_start", "transcript", "interrupted", "turn_end", "closed"]


@pytest.mark.parametrize(("side", "other"), [("agent", "caller"), ("caller", "agent")])
async def test_user_text_is_the_other_side_and_assistant_text_is_this_side(side, other):
    script = [
        *text_block("u", "USER", "FINAL", "my name is Maria", "PARTIAL_TURN"),
        *text_block("a", "ASSISTANT", "FINAL", "Thank you, Maria.", "END_TURN"),
        None,
    ]
    events = await run_to_end(session(ScriptedStream(script), side=side))
    heard, said = (ev for ev in events if ev.kind == "transcript")
    assert heard.payload == {"role": other, "text": "my name is Maria", "final": True}
    assert said.payload == {"role": side, "text": "Thank you, Maria.", "final": True}
    assert {ev.side for ev in events} == {side}


async def test_usage_event_carries_the_provider_usage_payload():
    events = await run_to_end(session(ScriptedStream([out("usageEvent", **USAGE), None])))
    assert kinds(events) == ["usage", "closed"]
    assert events[0].payload == {**IDS, **USAGE}


async def test_stream_failure_emits_an_unrecoverable_error_then_closed():
    failure = RuntimeError("Timed out waiting for audio bytes or interactive content.")
    events = await run_to_end(session(ScriptedStream([out("usageEvent", **USAGE), failure])))
    assert kinds(events) == ["usage", "error", "closed"]
    assert events[1].payload["recoverable"] is False
    assert "Timed out waiting" in events[1].payload["message"]


async def test_stream_failure_mid_reply_still_ends_the_turn():
    script = [*text_block("a", "ASSISTANT", "SPECULATIVE", "Hi.", "PARTIAL_TURN"), None]
    events = await run_to_end(session(ScriptedStream(script)))
    assert kinds(events) == ["turn_start", "transcript", "turn_end", "closed"]


async def test_closed_is_emitted_once_on_stream_end_and_double_close():
    stream = ScriptedStream([None])
    s = session(stream)
    assert kinds(await run_to_end(s)) == ["closed"]
    await s.close()
    await s.close()
    assert [ev async for ev in s.events()] == []
    assert "promptEnd" not in stream.names()


async def test_close_sends_the_end_sequence_and_waits_for_the_server_to_finish():
    stream = ScriptedStream([])
    s = session(stream)
    await s.start()
    await s.close()
    await s.close()
    assert stream.names()[-3:] == ["contentEnd", "promptEnd", "sessionEnd"]
    audio_name = stream.bodies("contentStart")[1]["contentName"]
    assert stream.sent[-3]["contentEnd"]["contentName"] == audio_name
    assert stream.closed
    assert kinds([ev async for ev in s.events()]) == ["closed"]
    sent = len(stream.sent)
    await s.send_audio(b"\x00" * 640)
    await s.send_text("too late")
    assert len(stream.sent) == sent


async def test_close_gives_up_on_a_server_that_never_ends(monkeypatch):
    monkeypatch.setattr(nova_sonic_client, "_CLOSE_WAIT_S", 0.05)
    stream = ScriptedStream([], ignore_close=True)
    s = session(stream)
    await s.start()
    await s.close()
    assert kinds([ev async for ev in s.events()]) == ["closed"]


async def test_close_before_start_only_emits_closed():
    stream = ScriptedStream([])
    s = session(stream)
    await s.close()
    assert kinds([ev async for ev in s.events()]) == ["closed"]
    assert stream.sent == []


def test_harness_is_not_a_valid_side():
    with pytest.raises(ValueError, match="side"):
        session(ScriptedStream([]), side="harness")


def test_importing_the_client_does_not_load_the_provider_sdk():
    code = (
        "import sys, callcheck.adapters.providers.nova_sonic_client\n"
        "loaded = [m for m in ('aws_sdk_bedrock_runtime', 'boto3', 'awscrt') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
