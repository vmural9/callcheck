"""Unit tests for the scriptless mode of FakeChat, the offline judge behind ``--judge fake``."""

import pytest

from callcheck.adapters.chat.fake import FakeChat
from callcheck.core.chat import Message
from callcheck.core.judge.schema import ExtractedFields, FreeTextJudgement, QualityReport

ASK = [Message("user", "grade this")]


async def ask(model) -> dict:
    reply = await FakeChat().complete(ASK, json_schema=model.model_json_schema())
    return model.model_validate(reply.json).model_dump()


async def test_extraction_finds_nothing():
    assert (await ask(ExtractedFields))["fields"] == {}


async def test_free_text_matches():
    assert (await ask(FreeTextJudgement))["verdict"] == "match"


async def test_every_quality_scores_five():
    report = await ask(QualityReport)
    assert {part["score"] for part in report.values()} == {5}


async def test_unknown_schema_fails_loudly():
    with pytest.raises(AssertionError, match="Nope"):
        await FakeChat().complete(ASK, json_schema={"title": "Nope"})
