"""What a call cost, priced from provider usage and a checked pricing table.

The agent and caller endpoints report cumulative token usage in ``usage``
events; the judge reports per-model token counts in ``judge.json``. ``cost_of``
prices each against ``pricing.yaml``. A price that is missing is never guessed:
an unknown model gives an "n/a" line, and a unit whose rate is ``null`` is left
out of the sum and marks the line (and the total) as partial.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from callcheck.core.events import CallEvent

_UNITS = ("speech_in", "speech_out", "text_in", "text_out")


class CostLine(BaseModel):
    """One party's cost; ``usd`` is ``None`` when it cannot be priced at all."""

    usd: float | None
    unverified: bool
    partial: bool = False
    note: str = ""


class Cost(BaseModel):
    """Cost of a call by party; ``total_usd`` sums the lines that could be priced."""

    agent: CostLine
    caller: CostLine
    judge: CostLine
    total_usd: float | None
    partial: bool = False


def load_pricing(path: Path) -> dict:
    """Read the pricing table from ``path``."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def nova_usage(events: list[CallEvent], side: str) -> dict[str, int] | None:
    """Sum the final cumulative token totals of each session ``side`` reported.

    Returns ``None`` when ``side`` sent no usage events.
    """
    last: dict[str, dict] = {}
    for ev in sorted(events, key=lambda e: e.t_ms):
        if ev.side == side and ev.kind == "usage":
            last[str(ev.payload.get("sessionId"))] = ev.payload.get("details", {}).get("total", {})
    if not last:
        return None
    sums = dict.fromkeys(_UNITS, 0)
    for total in last.values():
        for direction, short in (("input", "in"), ("output", "out")):
            for kind in ("speech", "text"):
                sums[f"{kind}_{short}"] += int(total.get(direction, {}).get(f"{kind}Tokens", 0))
    return sums


def _price(usage: dict[str, int], entry: dict) -> CostLine:
    """Price ``usage`` at ``entry``'s rates, skipping units whose rate is null."""
    rates = entry.get("per_1k_tokens", {})
    usd = 0.0
    unpriced = []
    for unit, tokens in usage.items():
        if not tokens:
            continue
        rate = rates.get(unit)
        if rate is None:
            unpriced.append(unit)
        else:
            usd += tokens / 1000 * rate
    note = f"partial: no price for {', '.join(unpriced)}" if unpriced else ""
    return CostLine(
        usd=usd,
        unverified=bool(entry.get("unverified", False)),
        partial=bool(unpriced),
        note=note,
    )


def _na(key: str, why: str = "") -> CostLine:
    return CostLine(usd=None, unverified=False, note=why or f"n/a: no price for {key}")


def _endpoint_line(events: list[CallEvent], side: str, model: str, models: dict) -> CostLine:
    usage = nova_usage(events, side)
    if usage is None:
        return _na(model, f"n/a: no usage reported by {side}")
    entry = models.get(model)
    if entry is None:
        return _na(model)
    return _price(usage, entry)


def _judge_line(judge_usage: list[dict], models: dict) -> CostLine:
    if not judge_usage:
        return _na("judge", "n/a: no judge usage")
    lines, missing = [], []
    for item in judge_usage:
        key = f"bedrock:{item['model']}"
        entry = models.get(key)
        if entry is None:
            missing.append(key)
            continue
        usage = {"text_in": item.get("input_tokens", 0), "text_out": item.get("output_tokens", 0)}
        lines.append(_price(usage, entry))
    if not lines:
        return _na(", ".join(missing))
    notes = [line.note for line in lines if line.note]
    notes += [f"n/a: no price for {key}" for key in missing]
    return CostLine(
        usd=sum(line.usd or 0.0 for line in lines),
        unverified=any(line.unverified for line in lines),
        partial=bool(missing) or any(line.partial for line in lines),
        note="; ".join(notes),
    )


def cost_of(
    events: list[CallEvent],
    judge_usage: list[dict],
    pricing: dict,
    *,
    agent_model: str,
    caller_model: str,
) -> Cost:
    """Price a call's agent, caller and judge usage against ``pricing``."""
    models = pricing.get("models", {})
    agent = _endpoint_line(events, "agent", agent_model, models)
    caller = _endpoint_line(events, "caller", caller_model, models)
    judge = _judge_line(judge_usage, models)
    lines = [agent, caller, judge]
    known = [line.usd for line in lines if line.usd is not None]
    return Cost(
        agent=agent,
        caller=caller,
        judge=judge,
        total_usd=sum(known) if known else None,
        partial=any(line.usd is None or line.partial for line in lines),
    )
