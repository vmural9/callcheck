"""Registry of simulated callers selectable by name."""

from typing import Any

from callcheck.adapters.registry import build
from callcheck.core.endpoints import Caller

CALLERS: dict[str, str] = {"fake": "callcheck.adapters.callers.fake:FakeCaller"}


def make_caller(name: str, **kwargs: Any) -> Caller:
    """Instantiate the registered caller ``name`` with ``kwargs``; ``KeyError`` if unknown."""
    return build(CALLERS, "caller", name, **kwargs)
