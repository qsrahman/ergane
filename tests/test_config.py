from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import make_config
from muagent.config import load_config


def test_tools_default_to_read_only():
    assert make_config().tools == "read"


@pytest.mark.parametrize("level", ["off", "read", "write", "execute"])
def test_the_tools_setting_accepts_each_access_level(level):
    assert make_config(TOOLS=level).tools == level


def test_an_unknown_tools_level_is_a_config_error():
    with pytest.raises(ValidationError, match="TOOLS"):
        make_config(TOOLS="all")


def test_invalid_reasoning_effort_is_a_config_error():
    with pytest.raises(ValidationError, match="REASONING_EFFORT"):
        make_config(REASONING_EFFORT="extreme")


class TestLoadConfig:
    """isolate_config makes tmp_path both the working directory and XDG_CONFIG_HOME."""

    @staticmethod
    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_reads_the_user_config_file(self, tmp_path: Path):
        self.write(
            tmp_path / "muagent" / ".env",
            "OPENAI_API_KEY=user\nOPENAI_MODEL=user-model\n",
        )

        assert load_config().model == "user-model"

    def test_project_env_overrides_the_user_file(self, tmp_path: Path):
        self.write(
            tmp_path / "muagent" / ".env",
            "OPENAI_API_KEY=user\nOPENAI_MODEL=user-model\n",
        )
        self.write(tmp_path / ".env", "OPENAI_MODEL=project-model\n")

        config = load_config()

        assert config.model == "project-model"
        assert config.api_key.get_secret_value() == "user"

    def test_environment_overrides_both_files(self, tmp_path: Path, monkeypatch):
        self.write(tmp_path / "muagent" / ".env", "OPENAI_API_KEY=user\n")
        self.write(tmp_path / ".env", "OPENAI_MODEL=project-model\n")
        monkeypatch.setenv("OPENAI_MODEL", "env-model")

        assert load_config().model == "env-model"

    def test_ignores_keys_that_are_not_settings(self, tmp_path: Path):
        self.write(tmp_path / ".env", "OPENAI_API_KEY=x\nSOME_OTHER_TOOL=1\n")

        assert load_config().api_key.get_secret_value() == "x"

    def test_reads_the_default_location_when_xdg_config_home_is_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        home = tmp_path / "home"
        self.write(
            home / ".config" / "muagent" / ".env",
            "OPENAI_API_KEY=user\nOPENAI_MODEL=default-location\n",
        )
        monkeypatch.delenv("XDG_CONFIG_HOME")
        monkeypatch.setenv("HOME", str(home))

        assert load_config().model == "default-location"

    def test_a_relative_xdg_config_home_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The XDG spec says relative paths are invalid, so ~/.config is used instead.
        home = tmp_path / "home"
        self.write(
            home / ".config" / "muagent" / ".env",
            "OPENAI_API_KEY=user\nOPENAI_MODEL=from-home\n",
        )
        self.write(  # what honoring the relative path would have read
            tmp_path / "relative" / "muagent" / ".env",
            "OPENAI_API_KEY=wrong\nOPENAI_MODEL=from-relative\n",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative")
        monkeypatch.setenv("HOME", str(home))

        assert load_config().model == "from-home"


def test_the_api_key_is_not_revealed_when_the_config_is_printed():
    config = make_config(OPENAI_API_KEY="sk-secret-123")

    for text in (repr(config), str(config), str(config.model_dump())):
        assert "sk-secret-123" not in text
    assert config.api_key.get_secret_value() == "sk-secret-123"


def test_an_empty_api_key_is_a_config_error():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        make_config(OPENAI_API_KEY="")
