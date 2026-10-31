from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import make_config
from muagent.config import load_config


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
        assert config.api_key == "user"

    def test_environment_overrides_both_files(self, tmp_path: Path, monkeypatch):
        self.write(tmp_path / "muagent" / ".env", "OPENAI_API_KEY=user\n")
        self.write(tmp_path / ".env", "OPENAI_MODEL=project-model\n")
        monkeypatch.setenv("OPENAI_MODEL", "env-model")

        assert load_config().model == "env-model"

    def test_ignores_keys_that_are_not_settings(self, tmp_path: Path):
        self.write(tmp_path / ".env", "OPENAI_API_KEY=x\nSOME_OTHER_TOOL=1\n")

        assert load_config().api_key == "x"
