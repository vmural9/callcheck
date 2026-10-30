"""Live check: a Nova Sonic session greets after a text turn, with audio and a transcript.

Silence frames are streamed throughout, as the call mixer does: the model does not answer a
text turn until input audio is flowing.
"""

import asyncio
import os

import pytest

from callcheck.adapters.providers.nova_sonic_client import NovaSonicSession


async def stream_silence(session: NovaSonicSession) -> None:
    """Send 20 ms silence frames at real-time pace until cancelled."""
    frame = bytes(2 * session.input_rate // 50)
    while True:
        await session.send_audio(frame)
        await asyncio.sleep(0.02)


@pytest.mark.live
async def test_live_text_turn_produces_audio_and_transcript():
    s = NovaSonicSession(
        side="agent",
        model_id=os.environ.get("CALLCHECK_NOVA_SONIC_MODEL", "amazon.nova-2-5-sonic"),
        region=os.environ["AWS_REGION"],
        profile=os.environ.get("AWS_PROFILE"),
        system_prompt="You are a friendly receptionist. Greet the caller in one short sentence.",
        voice_id="matthew",
        tools=[],
        tool_executor=None,
    )
    await s.start()
    got: set[str] = set()
    silence = asyncio.create_task(stream_silence(s))
    try:
        await s.send_text("(call connected)")
        async with asyncio.timeout(20):
            async for ev in s.events():
                got.add(ev.kind)
                if {"audio_out", "transcript"} <= got or ev.kind == "error":
                    break
    finally:
        silence.cancel()
        await asyncio.gather(silence, return_exceptions=True)
        await s.close()
    assert {"turn_start", "audio_out", "transcript"} <= got
