"""Throwaway spike: observe the wire behaviour of Amazon Nova Sonic's bidirectional stream.

Subcommands: probe, tool, hears, bargein, model. Each writes spikes/out/<subcommand>.jsonl.
Credentials: boto3 session from AWS_PROFILE, frozen credentials handed to the smithy config.
"""

import asyncio
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
import numpy as np
import soxr
from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
from smithy_http.aio.crt import AWSCRTHTTPClient
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationInput,
)

OUT = Path(__file__).parent / "out"
MODEL_IDS = ["amazon.nova-2-5-sonic", "amazon.nova-2-sonic-v1:0", "amazon.nova-sonic-v1:0"]
IN_RATE, OUT_RATE = 16000, 24000
FRAME = 512  # samples per 32 ms input frame at 16 kHz
FRAME_S = FRAME / IN_RATE


def cid(body: dict) -> str:
    """Output events name their content block by contentId; input events use contentName."""
    return body.get("contentId") or body.get("contentName") or "?"


def tone(seconds: float, hz: float = 440.0) -> bytes:
    t = np.arange(int(IN_RATE * seconds)) / IN_RATE
    return (0.3 * 32767 * np.sin(2 * np.pi * hz * t)).astype("<i2").tobytes()


def silence(seconds: float) -> bytes:
    return bytes(2 * int(IN_RATE * seconds))


def frames(pcm: bytes):
    step = FRAME * 2
    for i in range(0, len(pcm), step):
        chunk = pcm[i : i + step]
        yield chunk + bytes(step - len(chunk))


class Session:
    def __init__(self, label: str, model_id: str, rec: "Recorder"):
        self.label, self.model_id, self.rec = label, model_id, rec
        self.prompt = str(uuid.uuid4())
        self.t0 = time.monotonic()
        self.events: list[dict] = []  # received, decoded
        self.audio: dict[str, bytearray] = {}  # contentName -> 24 kHz PCM
        self.content_meta: dict[str, dict] = {}
        self.audio_name = None
        self.frames_sent = 0
        self.ended = False
        self.on_event = None

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    async def connect(self):
        self.client = AsyncBedrockRuntimeClient(config=await make_config())
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.rx_task = asyncio.create_task(self._recv())

    async def send(self, event: dict, log: bool = True):
        raw = json.dumps({"event": event})
        if log:
            self.rec.write(self.label, "sent", self.now(), {"event": event})
        await self.stream.input_stream.send(
            InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=raw.encode())
            )
        )

    async def _recv(self):
        try:
            _, rx = await self.stream.await_output()
            while True:
                r = await rx.receive()
                if r is None:
                    self.rec.write(self.label, "stream_end", self.now(), None)
                    self.ended = True
                    return
                v = r.value
                raw = getattr(v, "bytes_", None)
                if not raw:
                    self.rec.write(
                        self.label, "recv_nonbytes", self.now(), {"type": type(r).__name__, "repr": repr(v)}
                    )
                    continue
                msg = json.loads(raw.decode())
                ev = msg.get("event", {})
                shown = msg
                if "audioOutput" in ev:
                    c = ev["audioOutput"].get("content", "")
                    shown = {"event": {"audioOutput": {**ev["audioOutput"], "content": f"<b64 {len(c)} chars>"}}}
                self.events.append({"t": self.now(), "event": ev})
                self.rec.write(self.label, "recv", self.now(), shown)
                self._track(ev)
                if self.on_event:
                    self.on_event(ev)
        except Exception as e:  # noqa: BLE001
            self.rec.write(self.label, "recv_error", self.now(), {"type": type(e).__name__, "repr": repr(e)})
            self.ended = True

    def _track(self, ev: dict):
        if "contentStart" in ev:
            cs = ev["contentStart"]
            self.content_meta[cid(cs)] = cs
        if "audioOutput" in ev:
            ao = ev["audioOutput"]
            self.audio.setdefault(cid(ao), bytearray()).extend(base64.b64decode(ao["content"]))

    def seen(self, key: str) -> list[dict]:
        return [e for e in self.events if key in e["event"]]

    def error(self) -> bool:
        return self.ended and not self.events

    # ---- protocol helpers -------------------------------------------------------------
    async def setup(self, system: str, tools: list | None = None, voice: str = "matthew",
                    sensitivity: str = "MEDIUM"):
        await self.send({"sessionStart": {
            "inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7},
            "turnDetectionConfiguration": {"endpointingSensitivity": sensitivity},
        }})
        ps = {
            "promptName": self.prompt,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": OUT_RATE, "sampleSizeBits": 16,
                "channelCount": 1, "voiceId": voice, "encoding": "base64", "audioType": "SPEECH",
            },
        }
        if tools:
            ps["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            ps["toolConfiguration"] = {"tools": tools}
        await self.send({"promptStart": ps})
        await self.text_block("SYSTEM", system, interactive=False)

    async def text_block(self, role: str, text: str, interactive: bool):
        name = str(uuid.uuid4())
        await self.send({"contentStart": {
            "promptName": self.prompt, "contentName": name, "type": "TEXT",
            "interactive": interactive, "role": role,
            "textInputConfiguration": {"mediaType": "text/plain"},
        }})
        await self.send({"textInput": {"promptName": self.prompt, "contentName": name, "content": text}})
        await self.send({"contentEnd": {"promptName": self.prompt, "contentName": name}})

    async def text_turn(self, text: str):
        await self.text_block("USER", text, interactive=True)

    async def audio_start(self):
        self.audio_name = str(uuid.uuid4())
        await self.send({"contentStart": {
            "promptName": self.prompt, "contentName": self.audio_name, "type": "AUDIO",
            "interactive": True, "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": IN_RATE, "sampleSizeBits": 16,
                "channelCount": 1, "audioType": "SPEECH", "encoding": "base64",
            },
        }})

    async def pace(self, pcm: bytes):
        for fr in frames(pcm):
            await self.send({"audioInput": {
                "promptName": self.prompt, "contentName": self.audio_name,
                "content": base64.b64encode(fr).decode(),
            }}, log=False)
            self.frames_sent += 1
            await asyncio.sleep(FRAME_S)

    async def silence_until(self, pred, timeout: float):
        """Stream silence frames at real time until pred() or timeout; return whether pred held."""
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.ended:
            if pred():
                return True
            await self.pace(silence(FRAME_S))
        return pred()

    async def audio_end(self):
        self.rec.write(self.label, "note", self.now(), {"audioInputFramesSent": self.frames_sent})
        await self.send({"contentEnd": {"promptName": self.prompt, "contentName": self.audio_name}})
        self.audio_name = None

    async def close(self, wait: float = 4.0):
        try:
            if self.audio_name:
                await self.audio_end()
            await self.send({"promptEnd": {"promptName": self.prompt}})
            await self.send({"sessionEnd": {}})
            await self.stream.input_stream.close()
        except Exception as e:  # noqa: BLE001
            self.rec.write(self.label, "close_error", self.now(), {"type": type(e).__name__, "repr": repr(e)})
        try:
            await asyncio.wait_for(asyncio.shield(self.rx_task), wait)
        except (TimeoutError, asyncio.TimeoutError):
            self.rec.write(self.label, "note", self.now(), {"receiver_still_open_after_close_s": wait})
            self.rx_task.cancel()


class Recorder:
    def __init__(self, name: str):
        OUT.mkdir(exist_ok=True)
        self.f = open(OUT / f"{name}.jsonl", "w")

    def write(self, label, kind, t, body):
        line = {"session": label, "t": t, "kind": kind, "body": body}
        self.f.write(json.dumps(line) + "\n")
        self.f.flush()
        if kind != "recv" or "audioOutput" not in (body or {}).get("event", {}):
            print(f"[{label} {t:7.3f}] {kind} {json.dumps(body)[:300]}")


async def make_config() -> AsyncBedrockRuntimeConfig:
    creds = boto3.Session(profile_name=os.environ["AWS_PROFILE"]).get_credentials().get_frozen_credentials()
    return await AsyncBedrockRuntimeConfig.resolve(
        region=os.environ["AWS_REGION"],
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        aws_session_token=creds.token,
        transport=AWSCRTHTTPClient(),
    )


async def open_session(rec: Recorder, label: str, system: str, tools=None, try_ids=None,
                       sensitivity="MEDIUM") -> Session:
    """Open a session; fall through the model ids until one is not rejected."""
    cache = OUT / "model_id.txt"
    ids = try_ids or ([cache.read_text().strip()] if cache.exists() else MODEL_IDS)
    for mid in ids:
        s = Session(label, mid, rec)
        rec.write(label, "try_model", 0.0, {"modelId": mid})
        try:
            await s.connect()
            await s.setup(system, tools, sensitivity=sensitivity)
            await s.audio_start()
            await asyncio.sleep(3.0)  # a rejected model id surfaces as an error here
        except Exception as e:  # noqa: BLE001
            rec.write(label, "open_error", s.now(), {"modelId": mid, "type": type(e).__name__, "repr": repr(e)})
            continue
        if s.ended and not s.events:
            rec.write(label, "rejected", s.now(), {"modelId": mid})
            continue
        cache.write_text(mid)
        rec.write(label, "accepted", s.now(), {"modelId": mid})
        return s
    raise SystemExit("no model id accepted")


# ---- subcommands ----------------------------------------------------------------------
async def probe():
    rec = Recorder("probe")
    s = await open_session(rec, "A", "You are a friendly phone assistant. Keep replies to one short sentence.")
    rec.write("A", "marker", s.now(), {"step": "text turn '(call connected)'"})
    await s.text_turn("(call connected)")
    spoke = await s.silence_until(lambda: bool(s.seen("audioOutput")), 6)
    await s.silence_until(lambda: bool(s.seen("completionEnd")), 6)
    rec.write("A", "marker", s.now(), {"step": "model spoke first after text turn", "value": spoke})
    n_before = len(s.events)
    rec.write("A", "marker", s.now(), {"step": "tone 2 s + silence 3 s"})
    await s.pace(tone(2.0))
    await s.pace(silence(3.0))
    await s.silence_until(lambda: False, 3)
    rec.write("A", "marker", s.now(), {"step": "events after tone", "n": len(s.events) - n_before})
    rec.write("A", "marker", s.now(), {"step": "idle 60 s: sending nothing"})
    t_idle = time.monotonic()
    while time.monotonic() - t_idle < 60 and not s.ended:
        await asyncio.sleep(0.5)
    rec.write("A", "marker", s.now(), {"step": "idle over", "stream_ended": s.ended})
    rec.write("A", "marker", s.now(), {"step": "client promptEnd/sessionEnd"})
    await s.close()


SAVE_TOOL = {"toolSpec": {
    "name": "save_field",
    "description": "Save one field the caller has told us.",
    "inputSchema": {"json": json.dumps({
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
        "required": ["name", "value"],
    })},
}}


async def tool():
    rec = Recorder("tool")
    sysp = ("You are a phone intake assistant. When you learn the caller's first name, call the "
            "save_field tool with name 'first_name' and the value, then confirm in one short sentence.")
    s = await open_session(rec, "A", sysp, tools=[SAVE_TOOL])
    answered = {"v": False}

    def on_event(ev):
        if "contentEnd" in ev and ev["contentEnd"].get("type") == "TOOL" and not answered["v"]:
            answered["v"] = True
            asyncio.get_running_loop().create_task(reply())

    tool_ev = {}

    async def reply():
        tu = s.seen("toolUse")[0]["event"]["toolUse"]
        tool_ev.update(tu)
        cname = str(uuid.uuid4())
        await s.send({"contentStart": {
            "promptName": s.prompt, "contentName": cname, "interactive": False, "type": "TOOL",
            "role": "TOOL", "toolResultInputConfiguration": {
                "toolUseId": tu["toolUseId"], "type": "TEXT",
                "textInputConfiguration": {"mediaType": "text/plain"}},
        }})
        await s.send({"toolResult": {"promptName": s.prompt, "contentName": cname,
                                     "content": json.dumps({"saved": True})}})
        await s.send({"contentEnd": {"promptName": s.prompt, "contentName": cname}})
        rec.write("A", "marker", s.now(), {"step": "toolResult sent"})

    s.on_event = on_event
    await s.text_turn("my first name is Maria")
    await s.silence_until(lambda: answered["v"], 15)
    n = len(s.events)
    await s.silence_until(lambda: len(s.events) > n + 2 and bool(s.seen("completionEnd")), 12)
    await s.silence_until(lambda: False, 3)
    await s.close()


async def synth(rec: Recorder, text: str) -> bytes:
    """Ask a second session, by text turn, to say `text`; return its 16 kHz PCM."""
    s = await open_session(rec, "B-tts", "You are a voice reader. When the user sends a text message "
                           "that starts with 'Say:', speak exactly the words after it and nothing else.")
    await s.text_turn(f"Say: {text}")
    await s.silence_until(lambda: bool(s.seen("audioOutput")), 10)
    await s.silence_until(lambda: bool(s.seen("completionEnd")), 10)
    await s.close()
    if not s.audio:
        raise SystemExit("B produced no audio")
    first = next(iter(s.audio))
    rec.write("B-tts", "note", s.now(), {"audio_contents": {k: len(v) for k, v in s.audio.items()},
                                          "used": first})
    pcm24 = np.frombuffer(bytes(s.audio[first]), dtype="<i2")
    pcm16 = soxr.resample(pcm24, OUT_RATE, IN_RATE).astype("<i2").tobytes()
    (OUT / "hello16k.pcm").write_bytes(pcm16)
    return pcm16


async def hears():
    rec = Recorder("hears")
    sentence = "hello, can you hear me, testing one two three"
    pcm16 = await synth(rec, sentence)
    s = await open_session(rec, "A-listener", "You are a phone assistant. Reply with one short sentence.")
    rec.write("A-listener", "marker", s.now(), {"step": f"feeding {len(pcm16) / 2 / IN_RATE:.2f} s of speech + 2 s silence"})
    await s.pace(pcm16)
    await s.pace(silence(2.0))
    await s.silence_until(lambda: False, 6)
    await s.close()
    users = [e["event"]["textOutput"]["content"] for e in s.seen("textOutput")
             if s.content_meta.get(cid(e["event"]["textOutput"]), {}).get("role") == "USER"]
    rec.write("A-listener", "note", s.now(), {"user_transcripts": users})
    print("USER TRANSCRIPTS:", users)


async def bargein():
    rec = Recorder("bargein")
    pcm_path = OUT / "hello16k.pcm"
    speech = pcm_path.read_bytes() if pcm_path.exists() else tone(2.0)
    top = os.environ.get("BARGE_COUNT", "30")
    delay = float(os.environ.get("BARGE_DELAY", "3"))
    s = await open_session(rec, "A", "You are a phone assistant. When asked to count, count aloud slowly "
                           f"from one to {top}, one number per sentence.")
    await s.text_turn(f"Please count out loud from one to {top}.")
    await s.silence_until(lambda: len(s.seen("audioOutput")) >= 1, 10)
    await s.silence_until(lambda: False, delay)
    rec.write("A", "marker", s.now(), {"step": "barge-in: sending user speech while model speaks",
                                       "audio_events_so_far": len(s.seen("audioOutput"))})
    await s.pace(speech)
    await s.pace(silence(2.0))
    await s.silence_until(lambda: bool(s.seen("userSpeechStart")), 25)
    await s.silence_until(lambda: False, 6)
    await s.close()


async def model():
    """Try one model id (env NOVA_MODEL) with a text turn; session count 1."""
    rec = Recorder("model")
    s = await open_session(rec, "A", "You are a friendly phone assistant. One short sentence.",
                           try_ids=[os.environ["NOVA_MODEL"]])
    await s.text_turn("(call connected)")
    await s.silence_until(lambda: bool(s.seen("audioOutput")), 6)
    await s.silence_until(lambda: False, 3)
    await s.close()


COMMANDS = {"probe": probe, "tool": tool, "hears": hears, "bargein": bargein, "model": model}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: nova_sonic_spike.py {{{'|'.join(COMMANDS)}}}")
    asyncio.run(COMMANDS[sys.argv[1]]())
