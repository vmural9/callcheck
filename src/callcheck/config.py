"""Settings read from the environment and an optional ``.env`` file.

Everything the command line needs to reach a provider lives in ``Settings``. Values already
present in the environment win over the ``.env`` file, so a shell export or a CI variable
can always override what a developer keeps on disk.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEFAULT_REGION = "us-west-2"
DEFAULT_NOVA_SONIC_MODEL = "amazon.nova-2-5-sonic"
DEFAULT_JUDGE = "bedrock"
DEFAULT_JUDGE_MODEL = "us.anthropic.claude-sonnet-5-5"
DEFAULT_RUNS_DIR = "./runs"


class ConfigError(Exception):
    """Raised when a setting a command needs is missing."""


@dataclass
class Settings:
    """The provider and storage settings for one invocation of the command line."""

    aws_profile: str | None
    aws_region: str
    nova_sonic_model: str
    judge: str
    judge_model: str
    runs_dir: Path


def _env(name: str, default: str) -> str:
    """Return the variable ``name``, or ``default`` when it is unset or empty."""
    return os.environ.get(name) or default


def load_settings(env_file: Path | None = None) -> Settings:
    """Read the settings from the environment, after loading ``env_file`` (default: ``.env``).

    The default file is the nearest ``.env`` at or above the working directory. Variables that
    are already set are never overwritten by the file.
    """
    load_dotenv(env_file if env_file is not None else find_dotenv(usecwd=True), override=False)
    return Settings(
        aws_profile=os.environ.get("AWS_PROFILE") or None,
        aws_region=_env("AWS_REGION", DEFAULT_REGION),
        nova_sonic_model=_env("CALLCHECK_NOVA_SONIC_MODEL", DEFAULT_NOVA_SONIC_MODEL),
        judge=_env("CALLCHECK_JUDGE", DEFAULT_JUDGE),
        judge_model=_env("CALLCHECK_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        runs_dir=Path(_env("CALLCHECK_RUNS_DIR", DEFAULT_RUNS_DIR)),
    )


def require_aws(settings: Settings) -> None:
    """Raise ``ConfigError`` unless an AWS profile is set.

    Called before any Bedrock-backed adapter is built, so a missing profile fails at once and
    never falls through to whichever account the default credentials happen to point at.
    """
    if settings.aws_profile is None:
        raise ConfigError("AWS_PROFILE is not set; put it in .env or the environment")
