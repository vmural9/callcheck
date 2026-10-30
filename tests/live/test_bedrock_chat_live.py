"""Live check: Claude on Bedrock returns schema-shaped JSON through an auto-choice tool call."""

import os

import pytest

from callcheck.adapters.chat.bedrock import BedrockChat
from callcheck.core.chat import Message
from callcheck.core.judge.schema import FreeTextJudgement


@pytest.mark.live
async def test_live_sonnet_returns_schema_json():
    chat = BedrockChat(
        model_id=os.environ.get("CALLCHECK_JUDGE_MODEL", "us.anthropic.claude-sonnet-5-5"),
        region=os.environ["AWS_REGION"],
        profile=os.environ.get("AWS_PROFILE"),
    )
    r = await chat.complete(
        [Message("user", "Return verdict match with reason 'test'.")],
        json_schema=FreeTextJudgement.model_json_schema(),
    )
    assert FreeTextJudgement.model_validate(r.json).verdict == "match"
