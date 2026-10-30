"""Guards the architecture: ``callcheck.core`` must not import providers or adapters."""

import ast
import pathlib
from collections.abc import Iterator

FORBIDDEN_IN_CORE = (
    "callcheck.adapters",
    "boto3",
    "botocore",
    "aws_sdk_bedrock_runtime",
    "smithy_aws_core",
    "openai",
    "anthropic",
    "elevenlabs",
)
CORE = pathlib.Path(__file__).parents[2] / "src" / "callcheck" / "core"


def imported_modules(path: pathlib.Path) -> Iterator[str]:
    """Yield the dotted name of every module imported by the file at ``path``."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_core_has_no_provider_imports() -> None:
    """No file under ``core`` imports an adapter or a provider SDK."""
    files = list(CORE.rglob("*.py"))
    assert files, "core package is empty"
    for file in files:
        for module in imported_modules(file):
            assert not module.startswith(FORBIDDEN_IN_CORE), f"{file.name} imports {module}"
