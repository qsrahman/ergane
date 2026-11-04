"""Settings: the Config model and where its values are loaded from."""

import os
from pathlib import Path
from typing import get_args

from openai.types.shared import ReasoningEffort
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for the AI agent."""

    # A field without an alias is read from the environment variable of the same
    # name, in any case: max_turns from MAX_TURNS.
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
    )
    streaming: bool = Field(
        default=True,
    )
    max_turns: int = Field(
        default=25,
        gt=0,
    )
    # None sends no reasoning setting, so the model's default applies.
    # Non-reasoning models reject any effort with HTTP 400.
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
    )


def _xdg_dir(variable: str, default: str) -> Path:
    """Return the muagent directory under an XDG base directory.

    An unset, empty or relative variable is ignored, as the XDG spec says, and the
    default under the home directory is used instead.
    """
    base = Path(os.environ.get(variable, ""))
    if not base.is_absolute():
        base = Path.home() / default
    return base / "muagent"


def user_config_dir() -> Path:
    """Return the user-level config directory, `$XDG_CONFIG_HOME/muagent`."""
    return _xdg_dir("XDG_CONFIG_HOME", ".config")


def user_state_dir() -> Path:
    """Return the directory for logs, `$XDG_STATE_HOME/muagent`."""
    return _xdg_dir("XDG_STATE_HOME", ".local/state")


def private_state_dir() -> Path:
    """Return `user_state_dir()`, made if needed and closed to other users.

    It holds the prompt history and the server logs, which can contain whatever was
    pasted or logged. A directory that cannot be made or changed is left as it is:
    whoever writes there reports it.
    """
    path = user_state_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError:
        pass
    return path


def user_env_file() -> Path:
    """Return the user-level settings file, `$XDG_CONFIG_HOME/muagent/.env`."""
    return user_config_dir() / ".env"


def load_config() -> Config:
    """Load the settings.

    Environment variables take precedence over the user-level file (see
    user_env_file()), which may be missing. A `.env` in the current directory is not
    read: whoever controls a project could set the base URL, and the API key and
    every prompt would go wherever they chose.

    Raises:
        ValidationError: If a setting is missing or invalid.
    """
    return Config(_env_file=user_env_file())  # pyright: ignore[reportCallIssue]


# ReasoningEffort is Optional[Literal[...]]; these are the Literal's values.
EFFORTS: tuple[str, ...] = get_args(get_args(ReasoningEffort)[0])
