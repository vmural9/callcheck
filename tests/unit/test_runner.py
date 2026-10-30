from __future__ import annotations

import asyncio
from pathlib import Path

from callcheck.core.runner import RunResult, RunSpec, expand_runs, run_matrix


def test_expand_round_robin() -> None:
    specs = expand_runs(["a", "b"], 5, 10)
    assert [s.dir_name for s in specs] == ["a-10", "b-11", "a-12", "b-13", "a-14"]


def test_parallelism_is_bounded(tmp_path: Path) -> None:
    active = 0
    peak = 0

    async def place(spec: RunSpec, run_dir: Path) -> RunResult:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        for _ in range(5):
            await asyncio.sleep(0)
        active -= 1
        return "ok"

    specs = expand_runs(["a"], 7, 0)
    results = asyncio.run(run_matrix(specs, place, tmp_path, 3))
    assert peak == 3
    assert set(results.values()) == {"ok"}
    assert len(results) == 7


def test_parallel_below_one_rejected(tmp_path: Path) -> None:
    async def place(spec: RunSpec, run_dir: Path) -> RunResult:
        return "ok"

    try:
        asyncio.run(run_matrix([], place, tmp_path, 0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_transport_error_retried_once_and_attempt_kept(tmp_path: Path) -> None:
    calls: list[Path] = []

    async def place(spec: RunSpec, run_dir: Path) -> RunResult:
        calls.append(run_dir)
        run_dir.mkdir()
        if len(calls) == 1:
            (run_dir / "partial.txt").write_text("first")
            return "transport_error"
        (run_dir / "final.txt").write_text("second")
        return "ok"

    spec = RunSpec("a", 1)
    results = asyncio.run(run_matrix([spec], place, tmp_path, 1))
    assert results == {spec: "ok"}
    assert calls == [tmp_path / "a-1", tmp_path / "a-1"]
    assert (tmp_path / "a-1.attempt1" / "partial.txt").read_text() == "first"
    assert (tmp_path / "a-1" / "final.txt").read_text() == "second"


def test_transport_error_twice_is_transport_error(tmp_path: Path) -> None:
    n = 0

    async def place(spec: RunSpec, run_dir: Path) -> RunResult:
        nonlocal n
        n += 1
        return "transport_error"

    spec = RunSpec("a", 1)
    results = asyncio.run(run_matrix([spec], place, tmp_path, 1))
    assert results == {spec: "transport_error"}
    assert n == 2


def test_exception_does_not_cancel_others(tmp_path: Path) -> None:
    async def place(spec: RunSpec, run_dir: Path) -> RunResult:
        await asyncio.sleep(0)
        if spec.seed == 1:
            raise RuntimeError("boom")
        return "ok"

    specs = expand_runs(["a"], 4, 0)
    results = asyncio.run(run_matrix(specs, place, tmp_path, 2))
    assert results[RunSpec("a", 1)] == "error"
    assert [results[s] for s in specs if s.seed != 1] == ["ok", "ok", "ok"]


def test_dirs_are_distinct(tmp_path: Path) -> None:
    seen: list[Path] = []

    async def place(spec: RunSpec, run_dir: Path) -> RunResult:
        seen.append(run_dir)
        return "ok"

    specs = expand_runs(["a", "b"], 6, 3)
    asyncio.run(run_matrix(specs, place, tmp_path, 4))
    assert len(set(seen)) == 6
    assert all(p.parent == tmp_path for p in seen)
