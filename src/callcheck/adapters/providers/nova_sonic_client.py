"""One Amazon Nova Sonic speech-to-speech session, reported as ``CallEvent`` records.

Nova Sonic runs over Bedrock's ``InvokeModelWithBidirectionalStream``: every message in
either direction is a JSON object ``{"event": {<name>: {...}}}``. The client opens with
``sessionStart``, ``promptStart`` (voice, output format, tools), a SYSTEM text content
holding the system prompt, and one USER AUDIO content that stays open for the whole
session and carries base64 16 kHz PCM16 frames; text turns and tool results are sent as
further contents while it is open, and the session ends with that content's
``contentEnd``, ``promptEnd``, ``sessionEnd`` and closing the input. The server names its
contents by ``contentId`` and announces each with a ``contentStart`` giving its role, type
and, for text, a SPECULATIVE or FINAL stage: assistant text arrives once as SPECULATIVE
before the audio it describes and again as FINAL after it, a reply's AUDIO contents end
with ``PARTIAL_TURN`` until the last ends with ``END_TURN``, and a barge-in shows up as a
FINAL assistant TEXT content holding ``{ "interrupted" : true }`` that ends with
``INTERRUPTED``, after which no more audio of that reply follows. ``completionEnd`` is not
a turn boundary. The model answers a text turn only while input audio is flowing, so the
code driving this class keeps sending frames (silence included), as the call mixer does every
20 ms; that also keeps the server's 55 s input-idle timeout from ending the session. The
wire facts behind all of this are in ``docs/research/nova-sonic-spike.md``.
"""

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from callcheck.core.endpoints import ToolExecutor, ToolSpec
from callcheck.core.events import CallEvent, Side, endpoint_event

INPUT_RATE = 16000
OUTPUT_RATE = 24000

_INFERENCE = {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}
_ENDPOINTING = "MEDIUM"
_CLOSE_WAIT_S = 2.0
_OTHER_SIDE: dict[str, Side] = {"agent": "caller", "caller": "agent"}
_TEXT_FORMAT = {"mediaType": "text/plain"}


class SonicStream(Protocol):
    """The open bidirectional stream, reduced to JSON text in and out."""

    async def send(self, event_json: str) -> None:
        """Send one client event, already encoded as ``{"event": {...}}`` JSON."""
        ...

    async def receive(self) -> str | None:
        """Return the next server event as JSON text, or ``None`` once the stream has ended."""
        ...

    async def close(self) -> None:
        """Close the client's input side; the server then finishes and ends its output."""
        ...


StreamFactory = Callable[[], SonicStream]


@dataclass
class _Content:
    """A server content block being received: what it is and what it has carried so far."""

    role: str
    type: str
    stage: str
    texts: list[str] = field(default_factory=list)
    tool_use: dict | None = None


def _pcm_format(rate: int) -> dict:
    """Return the LPCM 16-bit mono base64 format block for ``rate`` Hz."""
    return {
        "mediaType": "audio/lpcm",
        "sampleRateHertz": rate,
        "sampleSizeBits": 16,
        "channelCount": 1,
        "audioType": "SPEECH",
        "encoding": "base64",
    }


def _tool_configuration(tools: list[ToolSpec]) -> dict:
    """Describe ``tools`` for ``promptStart``; the input schema travels as a JSON string."""
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": json.dumps(tool.input_schema)},
                }
            }
            for tool in tools
        ]
    }


def _stage(content_start: dict) -> str:
    """Return a content's generation stage; contents without one (audio, tools) are FINAL."""
    fields = json.loads(content_start.get("additionalModelFields") or "{}")
    return fields.get("generationStage", "FINAL")


def _json_object(text: str) -> dict | None:
    """Parse ``text`` as a JSON object, or return ``None`` if it is not one."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _is_interruption_marker(text: str) -> bool:
    """Tell whether assistant text is the ``{ "interrupted" : true }`` barge-in marker."""
    value = _json_object(text)
    return value is not None and value.get("interrupted") is True


class NovaSonicSession:
    """A Nova Sonic session for one side of a call: audio and text in, ``CallEvent``s out.

    ``start`` opens the stream and begins reading it in the background; ``events`` yields
    what was read, mapped onto the harness's event kinds, and ends after ``closed``.
    Tool calls are answered inline through ``tool_executor`` before reading resumes.
    """

    input_rate = INPUT_RATE
    output_rate = OUTPUT_RATE

    def __init__(
        self,
        *,
        side: Side,
        model_id: str,
        region: str,
        profile: str | None,
        system_prompt: str,
        voice_id: str,
        tools: list[ToolSpec],
        tool_executor: ToolExecutor | None,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        """Configure the session; nothing is opened until ``start``.

        ``stream_factory`` replaces the Bedrock stream (tests); without it ``start``
        connects to ``model_id`` in ``region`` with credentials from ``profile``.
        """
        if side not in _OTHER_SIDE:
            raise ValueError(f"side must be 'agent' or 'caller', not {side!r}")
        self._side: Side = side
        self._other: Side = _OTHER_SIDE[side]
        self._model_id = model_id
        self._region = region
        self._profile = profile
        self._system_prompt = system_prompt
        self._voice_id = voice_id
        self._tools = tools
        self._tool_executor = tool_executor
        self._stream_factory = stream_factory
        self._prompt_name = str(uuid.uuid4())
        self._audio_name = str(uuid.uuid4())
        self._stream: SonicStream | None = None
        self._reader: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._queue: asyncio.Queue[CallEvent] = asyncio.Queue()
        self._contents: dict[str, _Content] = {}
        self._in_turn = False
        self._closing = False
        self._finished = False

    async def start(self) -> None:
        """Open the stream, send the opening sequence, and start reading server events."""
        if self._stream_factory is not None:
            self._stream = self._stream_factory()
        else:
            self._stream = await _open_bedrock_stream(self._model_id, self._region, self._profile)
        await self._send(self._opening_events(), always=True)
        self._reader = asyncio.create_task(self._read())

    async def send_audio(self, pcm: bytes) -> None:
        """Send one frame of 16 kHz PCM16 into the open audio content; dropped once closing."""
        if self._accepts_input():
            content = base64.b64encode(pcm).decode()
            await self._send(
                [self._event("audioInput", contentName=self._audio_name, content=content)]
            )

    async def send_text(self, text: str) -> None:
        """Send ``text`` as a complete USER text turn; the model answers once audio flows."""
        if self._accepts_input():
            await self._send(self._text_content("USER", text, interactive=True))

    async def events(self) -> AsyncIterator[CallEvent]:
        """Yield events as they arrive until ``closed`` has been yielded."""
        while not (self._finished and self._queue.empty()):
            ev = await self._queue.get()
            yield ev
            if ev.kind == "closed":
                return

    async def close(self) -> None:
        """End the session and release the stream on every path; idempotent.

        While the server is still sending, the end sequence goes first (best effort) and
        the server is given a moment to finish. The stream is closed exactly once, whether
        the session ended cleanly, failed, or never finished starting.
        """
        if self._closing:
            return
        self._closing = True
        if self._stream is not None:
            live = self._reader is not None and not self._reader.done()
            if live:
                try:
                    await self._send(self._closing_events(), always=True)
                except Exception as exc:
                    self._emit(
                        "error", message=f"Nova Sonic end sequence failed: {exc}", recoverable=True
                    )
            try:
                await self._stream.close()
            except Exception as exc:
                self._emit(
                    "error", message=f"Nova Sonic stream close failed: {exc}", recoverable=True
                )
            if live:
                await self._await_reader()
        self._finish()

    def _accepts_input(self) -> bool:
        """Tell whether input may still be sent; raise if the session was never started."""
        if self._stream is None and not self._closing:
            raise RuntimeError("NovaSonicSession.start() has not been called")
        return not (self._closing or self._finished)

    async def _await_reader(self) -> None:
        """Let the reader drain the server's final events, cancelling it after the wait."""
        assert self._reader is not None
        try:
            await asyncio.wait_for(asyncio.shield(self._reader), _CLOSE_WAIT_S)
        except TimeoutError:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)

    # Client events

    def _event(self, name: str, **body: object) -> dict:
        """Build a client event scoped to this session's prompt."""
        return {name: {"promptName": self._prompt_name, **body}}

    def _opening_events(self) -> list[dict]:
        """Return the session, prompt, system prompt and audio-content opening events."""
        prompt: dict[str, Any] = {
            "textOutputConfiguration": _TEXT_FORMAT,
            "audioOutputConfiguration": {
                **_pcm_format(OUTPUT_RATE),
                "voiceId": self._voice_id,
            },
        }
        if self._tools:
            prompt["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt["toolConfiguration"] = _tool_configuration(self._tools)
        return [
            {
                "sessionStart": {
                    "inferenceConfiguration": _INFERENCE,
                    "turnDetectionConfiguration": {"endpointingSensitivity": _ENDPOINTING},
                }
            },
            self._event("promptStart", **prompt),
            *self._text_content("SYSTEM", self._system_prompt, interactive=False),
            self._event(
                "contentStart",
                contentName=self._audio_name,
                type="AUDIO",
                interactive=True,
                role="USER",
                audioInputConfiguration=_pcm_format(INPUT_RATE),
            ),
        ]

    def _text_content(self, role: str, text: str, *, interactive: bool) -> list[dict]:
        """Return a whole text content: its start, the text, and its end."""
        name = str(uuid.uuid4())
        return [
            self._event(
                "contentStart",
                contentName=name,
                type="TEXT",
                interactive=interactive,
                role=role,
                textInputConfiguration=_TEXT_FORMAT,
            ),
            self._event("textInput", contentName=name, content=text),
            self._event("contentEnd", contentName=name),
        ]

    def _tool_result_content(self, tool_use_id: str, result: dict) -> list[dict]:
        """Return the content that answers tool call ``tool_use_id`` with ``result``."""
        name = str(uuid.uuid4())
        return [
            self._event(
                "contentStart",
                contentName=name,
                interactive=False,
                type="TOOL",
                role="TOOL",
                toolResultInputConfiguration={
                    "toolUseId": tool_use_id,
                    "type": "TEXT",
                    "textInputConfiguration": _TEXT_FORMAT,
                },
            ),
            self._event("toolResult", contentName=name, content=json.dumps(result)),
            self._event("contentEnd", contentName=name),
        ]

    def _closing_events(self) -> list[dict]:
        """Return the events that close the audio content, the prompt and the session."""
        return [
            self._event("contentEnd", contentName=self._audio_name),
            self._event("promptEnd"),
            {"sessionEnd": {}},
        ]

    async def _send(self, events: list[dict], *, always: bool = False) -> None:
        """Send ``events`` back to back, so concurrent senders never split a content.

        Unless ``always``, the events are dropped once the session is closing or over;
        that is checked under the lock, so nothing slips in after the closing sequence.
        """
        assert self._stream is not None
        async with self._send_lock:
            if not always and (self._closing or self._finished):
                return
            for event in events:
                await self._stream.send(json.dumps({"event": event}))

    # Server events

    async def _read(self) -> None:
        """Map server events until the stream ends or fails, then emit ``closed``."""
        assert self._stream is not None
        try:
            while (raw := await self._stream.receive()) is not None:
                for name, body in json.loads(raw).get("event", {}).items():
                    await self._dispatch(name, body)
        except Exception as exc:
            self._emit("error", message=f"Nova Sonic stream failed: {exc}", recoverable=False)
        self._finish()

    async def _dispatch(self, name: str, body: dict) -> None:
        """Route one server event; events with no harness meaning are ignored."""
        if name == "contentStart":
            self._on_content_start(body)
        elif name == "textOutput":
            if content := self._contents.get(body["contentId"]):
                content.texts.append(body["content"])
        elif name == "audioOutput":
            self._emit("audio_out", pcm=base64.b64decode(body["content"]))
        elif name == "toolUse":
            if content := self._contents.get(body["contentId"]):
                content.tool_use = body
        elif name == "contentEnd":
            await self._on_content_end(body)
        elif name == "userSpeechStart":
            self._emit("heard_speech_start", input_offset_ms=body.get("inputAudioOffsetMs"))
        elif name == "userSpeechEnd":
            self._emit(
                "heard_speech_end",
                input_offset_ms=body.get("inputAudioOffsetMs"),
                detection_offset_ms=body.get("inputAudioDetectionOffsetMs"),
            )
        elif name == "usageEvent":
            self._emit("usage", **body)

    def _on_content_start(self, body: dict) -> None:
        """Track a new content; speculative text or audio from the assistant opens a turn."""
        content = _Content(role=body.get("role", ""), type=body.get("type", ""), stage=_stage(body))
        self._contents[body["contentId"]] = content
        speaks = content.type == "AUDIO" or content.stage == "SPECULATIVE"
        if content.role == "ASSISTANT" and speaks and not self._in_turn:
            self._in_turn = True
            self._emit("turn_start")

    async def _on_content_end(self, body: dict) -> None:
        """Finish a content: emit its text, close the turn, or answer its tool call."""
        content = self._contents.pop(body["contentId"], None)
        if content is None:
            return
        stop_reason = body.get("stopReason")
        if content.type == "TEXT":
            self._finish_text(content, stop_reason)
        elif content.type == "AUDIO" and stop_reason != "PARTIAL_TURN":
            self._end_turn()
        elif content.type == "TOOL" and content.tool_use is not None:
            await self._answer_tool(content.tool_use)

    def _finish_text(self, content: _Content, stop_reason: str | None) -> None:
        """Emit a finished text content as a transcript, or as a barge-in if it is the marker."""
        text = "".join(content.texts).strip()
        if content.role == "ASSISTANT" and (
            stop_reason == "INTERRUPTED" or _is_interruption_marker(text)
        ):
            self._emit("interrupted")
            self._end_turn()
            return
        if text:
            role = self._side if content.role == "ASSISTANT" else self._other
            self._emit("transcript", role=role, text=text, final=content.stage != "SPECULATIVE")

    async def _answer_tool(self, tool_use: dict) -> None:
        """Run the tool the model called and send its result back; a failure is the result."""
        name = tool_use.get("toolName", "")
        raw = tool_use.get("content") or "{}"
        arguments = _json_object(raw)
        self._emit(
            "tool_call", name=name, arguments={"raw": raw} if arguments is None else arguments
        )
        try:
            result = await self._run_tool(name, arguments)
        except Exception as exc:
            self._emit("error", message=f"tool {name} failed: {exc}", recoverable=True)
            result = {"ok": False, "error": str(exc)}
        await self._send(self._tool_result_content(tool_use["toolUseId"], result))
        self._emit("tool_result", name=name, result=result)

    async def _run_tool(self, name: str, arguments: dict | None) -> dict:
        """Call the executor, refusing arguments that are not a JSON object."""
        if arguments is None:
            raise ValueError("arguments are not a JSON object")
        if self._tool_executor is None:
            raise RuntimeError("no tool executor is configured")
        return await self._tool_executor(name, arguments)

    # Event queue

    def _emit(self, kind: str, **payload: object) -> None:
        """Queue an event on this session's side; nothing follows ``closed``."""
        if not self._finished:
            self._queue.put_nowait(endpoint_event(self._side, kind, **payload))

    def _end_turn(self) -> None:
        """Emit ``turn_end`` if the assistant is mid-reply."""
        if self._in_turn:
            self._in_turn = False
            self._emit("turn_end")

    def _finish(self) -> None:
        """Close any open turn and emit ``closed``, once."""
        if not self._finished:
            self._end_turn()
            self._emit("closed")
            self._finished = True


class _BedrockStream:
    """``SonicStream`` over the Bedrock runtime SDK's duplex event stream."""

    def __init__(self, stream: Any, chunk: Callable[[bytes], Any]) -> None:
        """Wrap an open ``DuplexEventStream``; ``chunk`` wraps raw bytes as an input event."""
        self._stream = stream
        self._chunk = chunk
        self._receiver: Any = None

    async def send(self, event_json: str) -> None:
        """Send one JSON event as an input chunk."""
        await self._stream.input_stream.send(self._chunk(event_json.encode()))

    async def receive(self) -> str | None:
        """Return the next output event's JSON, or ``None`` at the end of the stream."""
        if self._receiver is None:
            _, self._receiver = await self._stream.await_output()
        while (item := await self._receiver.receive()) is not None:
            data = getattr(item.value, "bytes_", None)
            if data:
                return data.decode()
        return None

    async def close(self) -> None:
        """Close the input side of the stream."""
        await self._stream.input_stream.close()


def _frozen_credentials(profile: str | None) -> Any:
    """Resolve AWS credentials through boto3, which the smithy client cannot do itself."""
    import boto3

    credentials = boto3.Session(profile_name=profile).get_credentials()
    if credentials is None:
        raise RuntimeError("no AWS credentials found for Nova Sonic")
    return credentials.get_frozen_credentials()


async def _open_bedrock_stream(model_id: str, region: str, profile: str | None) -> SonicStream:
    """Open the bidirectional stream to ``model_id``; the SDK is imported only here.

    The SDK's default HTTP transport cannot stream both ways, so the CRT transport from
    the ``bedrock`` extra is required.
    """
    from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
    from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
    from aws_sdk_bedrock_runtime.models import (
        BidirectionalInputPayloadPart,
        InvokeModelWithBidirectionalStreamInputChunk,
        InvokeModelWithBidirectionalStreamOperationInput,
    )
    from smithy_http.aio.crt import AWSCRTHTTPClient

    credentials = await asyncio.to_thread(_frozen_credentials, profile)
    config = await AsyncBedrockRuntimeConfig.resolve(
        region=region,
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_session_token=credentials.token,
        transport=AWSCRTHTTPClient(),
    )
    client = AsyncBedrockRuntimeClient(config=config)
    stream = await client.invoke_model_with_bidirectional_stream(
        InvokeModelWithBidirectionalStreamOperationInput(model_id=model_id)
    )

    def chunk(data: bytes) -> Any:
        """Wrap encoded event JSON as an input chunk."""
        return InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=data)
        )

    return _BedrockStream(stream, chunk)
