"""Settings: the Config model and where its values are loaded from."""

import os
from pathlib import Path
from typing import get_args

from openai.types.shared import ReasoningEffort
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from muagent.tools import Level


class Config(BaseSettings):
    """Configuration for the AI agent."""

    # Config files may hold keys that aren't settings; ignore them, don't fail.
    model_config = SettingsConfigDict(extra="ignore")
    model: str = Field(
        min_length=1,
        default="gpt-5.4-mini",
        validation_alias="OPENAI_MODEL",
    )
    api_key: SecretStr = Field(  # a SecretStr stays hidden if the config is printed
        min_length=1,
        validation_alias="OPENAI_API_KEY",
    )
    base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    max_context_tokens: int = Field(
        default=64 * 1024,
        gt=0,
        validation_alias="MAX_CONTEXT_TOKENS",
    )
    streaming: bool = Field(
        default=True,
        validation_alias="STREAMING",
    )
    max_turns: int = Field(
        default=25,
        gt=0,
        validation_alias="MAX_TURNS",
    )
    # Which tools the model may use: none, or up to this much access.
    tools: Level = Field(
        default="read",
        validation_alias="TOOLS",
    )
    # None sends no reasoning setting, so the model's default applies.
    # Non-reasoning models reject any effort with HTTP 400.
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        validation_alias="REASONING_EFFORT",
    )


def user_env_file() -> Path:
    """Return the user-level settings file, `$XDG_CONFIG_HOME/muagent/.env`.

    An unset, empty or relative XDG_CONFIG_HOME is ignored, as the XDG spec says,
    and `~/.config` is used instead.
    """
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", ""))
    if not config_home.is_absolute():
        config_home = Path.home() / ".config"
    return config_home / "muagent" / ".env"


def load_config() -> Config:
    """Load the settings.

    Environment variables take precedence, then a `.env` in the current directory,
    then the user-level file (see user_env_file()). Missing files are skipped.

    Raises:
        ValidationError: If a setting is missing or invalid.
    """
    # Later files override earlier ones.
    return Config(_env_file=(user_env_file(), ".env"))  # pyright: ignore[reportCallIssue]


# ReasoningEffort is Optional[Literal[...]]; these are the Literal's values.
EFFORTS: tuple[str, ...] = get_args(get_args(ReasoningEffort)[0])
