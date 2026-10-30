"""BedrockChat builds the Converse request and parses tool-use replies without AWS."""

import asyncio

from callcheck.adapters.chat.bedrock import BedrockChat
from callcheck.core.chat import Message


class FakeBoto:
    """Stands in for a bedrock-runtime client: records the request, returns a canned reply."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.kwargs: dict = {}

    def converse(self, **kwargs) -> dict:
        self.kwargs = kwargs
        return self.response


def test_emit_tool_returns_json():
    fake = FakeBoto(
        response={
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "emit",
                                "input": {"verdict": "match", "reason": "ok"},
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
    )
    r = asyncio.run(
        BedrockChat(model_id="m", region="us-west-2", client=fake).complete(
            [Message("system", "s"), Message("user", "u")], json_schema={"type": "object"}
        )
    )
    assert r.json == {"verdict": "match", "reason": "ok"}
    assert r.usage == {"input_tokens": 10, "output_tokens": 5}
    assert fake.kwargs["toolConfig"]["toolChoice"] == {"auto": {}}
    assert fake.kwargs["system"] == [
        {"text": "s"},
        {"text": "Answer only by calling the emit tool with your answer as its input."},
    ]
    assert fake.kwargs["messages"] == [{"role": "user", "content": [{"text": "u"}]}]


def test_plain_text_reply_has_no_tool_config():
    fake = FakeBoto(
        response={
            "output": {"message": {"content": [{"text": "hel"}, {"text": "lo"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 2},
        }
    )
    r = asyncio.run(
        BedrockChat(model_id="m", region="us-west-2", client=fake).complete([Message("user", "hi")])
    )
    assert r.text == "hello" and r.json is None
    assert "toolConfig" not in fake.kwargs and "system" not in fake.kwargs
    assert fake.kwargs["inferenceConfig"] == {"maxTokens": 4096}


def _text_reply(text: str) -> FakeBoto:
    return FakeBoto(
        response={
            "output": {"message": {"content": [{"text": text}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
    )


def _ask_with_schema(fake: FakeBoto):
    chat = BedrockChat(model_id="m", region="us-west-2", client=fake)
    return asyncio.run(chat.complete([Message("user", "u")], json_schema={"type": "object"}))


def test_text_json_fallback_strips_code_fence():
    r = _ask_with_schema(_text_reply('```json\n{"verdict": "match"}\n```'))
    assert r.json == {"verdict": "match"}
    assert _ask_with_schema(_text_reply('{"a": 1}')).json == {"a": 1}


def test_prose_reply_gives_no_json_and_system_prompt_is_created():
    fake = _text_reply("I think it matches.")
    r = _ask_with_schema(fake)
    assert r.json is None and r.text == "I think it matches."
    assert fake.kwargs["system"] == [
        {"text": "Answer only by calling the emit tool with your answer as its input."}
    ]


def test_temperature_is_sent_only_when_given():
    fake = _text_reply("x")
    chat = BedrockChat(model_id="m", region="us-west-2", client=fake)
    asyncio.run(chat.complete([Message("user", "u")], temperature=0.3))
    assert fake.kwargs["inferenceConfig"] == {"maxTokens": 4096, "temperature": 0.3}


def test_model_id_is_exposed_for_usage_pricing():
    chat = BedrockChat(
        model_id="us.anthropic.claude-sonnet-5-5", region="us-west-2", client=object()
    )
    assert chat.model_id == "us.anthropic.claude-sonnet-5-5"
