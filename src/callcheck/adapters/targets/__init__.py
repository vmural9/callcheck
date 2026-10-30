"""Registry of target agents (the agent under test) selectable by name."""

from typing import Any

from callcheck.adapters.registry import build
from callcheck.core.endpoints import TargetAgent

TARGETS: dict[str, str] = {"fake": "callcheck.adapters.targets.fake:FakeTarget"}


def make_target(name: str, **kwargs: Any) -> TargetAgent:
    """Instantiate the registered target ``name`` with ``kwargs``; ``KeyError`` if unknown."""
    return build(TARGETS, "target", name, **kwargs)
