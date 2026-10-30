"""Shared pytest configuration: live tests run only when CALLCHECK_LIVE=1."""

import os

import pytest


@pytest.fixture(scope="session")
def live_enabled() -> bool:
    """Report whether live provider tests are enabled for this session."""
    return os.environ.get("CALLCHECK_LIVE") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``live`` unless CALLCHECK_LIVE=1 is set."""
    if os.environ.get("CALLCHECK_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="set CALLCHECK_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
