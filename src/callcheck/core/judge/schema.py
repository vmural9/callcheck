"""Pydantic schemas for what the judge produces.

Field verdicts come from deterministic comparators, quality scores and free-text judgements
from the judge model; ``JudgeResult`` gathers them into the single record a run is graded by.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from callcheck.core.turns import Incident

Verdict = Literal["match", "near_miss", "mismatch", "missing", "undetermined"]
RunVerdict = Literal["pass", "fail", "undetermined", "artifact"]
Method = Literal["result", "transcript"]


class FieldVerdict(BaseModel):
    """The outcome of comparing one captured field with its expected value."""

    name: str
    expected: Any
    captured: Any | None
    verdict: Verdict
    method: Method
    note: str


class QualityScore(BaseModel):
    """A 0 to 5 score for one conversational quality, with the turns that justify it."""

    score: int = Field(ge=0, le=5)
    reason: str
    evidence: list[int] = []


class QualityReport(BaseModel):
    """The five conversational qualities the judge model scores."""

    repetition: QualityScore
    confusion: QualityScore
    correction_handling: QualityScore
    clarity: QualityScore
    closing: QualityScore


class FreeTextJudgement(BaseModel):
    """The judge model's ruling on whether free text conveys the expected meaning."""

    verdict: Literal["match", "mismatch", "undetermined"]
    reason: str


class ExtractedFields(BaseModel):
    """Field values the judge model extracted from a transcript, ``None`` when absent."""

    fields: dict[str, str | None]


class JudgeResult(BaseModel):
    """The complete grade for one run."""

    verdict: RunVerdict
    failure_types: list[str]
    fields: list[FieldVerdict]
    quality: QualityReport | None
    captured_method: Method
    summary: str
    undetermined_reasons: list[str] = []
    incidents: list[Incident] = []
    usage: list[dict] = []
