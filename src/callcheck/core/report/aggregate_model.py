"""The data a batch report is made of: ``BatchSummary`` and the rows inside it."""

from typing import Literal

from pydantic import BaseModel

RunStatus = Literal["pass", "fail", "undetermined", "artifact", "errored"]


class RunRow(BaseModel):
    """One run's outcome; ``errored`` runs have no judge result and carry an ``error``."""

    name: str
    verdict: RunStatus
    failure_types: list[str] = []
    quality_mean: float | None = None
    error: str | None = None


class FailureRow(BaseModel):
    """A failure type, how many runs showed it and the first such run."""

    type: str
    count: int
    example: str


class FieldRow(BaseModel):
    """How often a field matched among the runs that graded it."""

    name: str
    graded: int
    matched: int
    most_common_wrong: str | None = None


class QualityRow(BaseModel):
    """The mean score of one quality dimension and the run that scored lowest."""

    dimension: str
    mean: float | None
    lowest_run: str | None = None
    lowest_score: int | None = None


class LatencySummary(BaseModel):
    """Time to first audio over every measured turn of every run, in milliseconds."""

    turns: int
    p50_ms: float | None
    p90_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    slowest_run: str | None = None


class IncidentRow(BaseModel):
    """How many incidents carried one label and attribution."""

    label: str
    attribution: str
    count: int


class CostRow(BaseModel):
    """One party's cost over the batch; ``None`` when any run lacks it."""

    party: str
    total_usd: float | None
    mean_usd: float | None
    partial: bool = False


class CostSummary(BaseModel):
    """Cost per party, the batch total and whether any price is unverified."""

    rows: list[CostRow]
    total_usd: float | None
    total_partial: bool = False
    unverified: bool = False
    retried_attempts: int = 0


class BatchSummary(BaseModel):
    """Everything ``report.md`` shows about a batch; it is also written as ``report.json``."""

    scenario: str | None = None
    target: str | None = None
    caller: str | None = None
    judge: str | None = None
    bugs: list[str] = []
    started: str | None = None
    finished: str | None = None
    wall_seconds: float | None = None
    n: int
    transcripts: list[str] = []
    counts: dict[str, int]
    pass_rate: float | None
    runs: list[RunRow]
    failures: list[FailureRow]
    fields: list[FieldRow]
    quality: list[QualityRow]
    latency: LatencySummary
    incidents: list[IncidentRow]
    worst: list[RunRow]
    cost: CostSummary
