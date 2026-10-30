"""Run a matrix of calls in parallel, retrying a transport failure once."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_NOVA_PARALLEL = 8

RunResult = Literal["ok", "transport_error", "error"]


@dataclass(frozen=True)
class RunSpec:
    persona_id: str
    seed: int

    @property
    def dir_name(self) -> str:
        return f"{self.persona_id}-{self.seed}"


def expand_runs(personas: list[str], n: int, seed: int) -> list[RunSpec]:
    """Run i gets personas[i % len(personas)] and seed ``seed + i``."""
    return [RunSpec(personas[i % len(personas)], seed + i) for i in range(n)]


async def run_matrix(
    specs: list[RunSpec],
    place: Callable[[RunSpec, Path], Awaitable[RunResult]],
    batch_dir: Path,
    parallel: int,
) -> dict[RunSpec, RunResult]:
    if parallel < 1:
        raise ValueError("parallel must be at least 1")
    sem = asyncio.Semaphore(parallel)

    async def one(spec: RunSpec) -> RunResult:
        run_dir = batch_dir / spec.dir_name
        async with sem:
            try:
                result = await place(spec, run_dir)
                if result == "transport_error":
                    if run_dir.exists():
                        run_dir.rename(run_dir.with_name(f"{spec.dir_name}.attempt1"))
                    result = await place(spec, run_dir)
                return result
            except Exception:
                return "error"

    results = await asyncio.gather(*(one(s) for s in specs))
    return dict(zip(specs, results, strict=True))
