"""Unit tests for settings loading and the AWS profile requirement."""

from pathlib import Path

import pytest

from callcheck.config import ConfigError, Settings, load_settings, require_aws

VARIABLES = (
    "AWS_PROFILE",
    "AWS_REGION",
    "CALLCHECK_NOVA_SONIC_MODEL",
    "CALLCHECK_JUDGE",
    "CALLCHECK_JUDGE_MODEL",
    "CALLCHECK_RUNS_DIR",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Start every test with none of the settings variables set."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_nothing_is_set(tmp_path):
    s = load_settings(tmp_path / "missing.env")
    assert s == Settings(
        aws_profile=None,
        aws_region="us-west-2",
        nova_sonic_model="amazon.nova-2-5-sonic",
        judge="bedrock",
        judge_model="us.anthropic.claude-sonnet-5-5",
        runs_dir=Path("./runs"),
    )


def test_environment_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "demo")
    monkeypatch.setenv("CALLCHECK_JUDGE", "fake")
    monkeypatch.setenv("CALLCHECK_RUNS_DIR", str(tmp_path))
    s = load_settings(tmp_path / "missing.env")
    assert (s.aws_profile, s.judge, s.runs_dir) == ("demo", "fake", tmp_path)


def test_env_file_fills_gaps_but_never_overrides_the_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("AWS_PROFILE=from-file\nAWS_REGION=eu-west-1\n")
    monkeypatch.setenv("AWS_PROFILE", "from-shell")
    s = load_settings(env_file)
    assert (s.aws_profile, s.aws_region) == ("from-shell", "eu-west-1")
    monkeypatch.delenv("AWS_REGION")  # load_dotenv put it into os.environ


def test_empty_values_count_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "")
    monkeypatch.setenv("AWS_REGION", "")
    s = load_settings(tmp_path / "missing.env")
    assert s.aws_profile is None and s.aws_region == "us-west-2"


def test_require_aws_names_the_variable(tmp_path):
    s = load_settings(tmp_path / "missing.env")
    with pytest.raises(ConfigError, match="AWS_PROFILE"):
        require_aws(s)


def test_require_aws_accepts_a_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "demo")
    require_aws(load_settings(tmp_path / "missing.env"))
