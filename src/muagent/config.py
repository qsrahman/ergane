"""Settings: the Config model and where its values are loaded from.

A value comes from the highest of these that sets it: the command line, the
environment, `config.json`, and the default. The environment has one variable,
`OPENAI_API_KEY`, the one the OpenAI SDK reads; everything else is a flag or a key in
`config.json`, which also lists the MCP servers.

How it is built, and why:

- `Config.settings_customise_sources` is exactly that chain. The flags arrive as
  keyword arguments (pydantic-settings' highest source) by field name; `None` is a
  value (`--reasoning-effort default`). There is no `.env` and no flag for the key (it
  would show in the process list).
- The environment is read for the key only (`_Environment`). A variable for the base
  URL would send the key and every prompt wherever a project's `.envrc` said, one for
  the servers would start a program of its own, and the common names (`MODEL`,
  `BASE_URL`) are exported by other programs. The file is the user's, never one in the
  working directory, and only `--config` names another. `load_config` gives the path
  to a subclass through `json_file`, since it is only known at run time.
- The file's keys are the fields' names (`mcp_servers` is `mcpServers`, as in other
  tools' MCP files). `extra="forbid"`: a key that is not a setting names itself in the
  error, and a file that is not a JSON object is a `ValidationError` too (`_fail`), so
  `main()` has one path for every mistake.
- `api_key` may be `${NAME}`. It is filled in on the value that won, so a placeholder
  the environment overrode is never looked at.
"""

import json
import os
from pathlib import Path
from typing import Any, NoReturn, get_args

from openai.types.shared import ReasoningEffort
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from muagent.mcp_config import McpServerConfig, ServerName, fill_placeholders


def _fail(kind: str, **details: Any) -> NoReturn:
    """Report config.json as invalid the way pydantic reports a bad setting."""
    raise ValidationError.from_exception_data(
        "config.json",
        [{"type": kind, "loc": (), "input": "", **details}],  # pyright: ignore[reportArgumentType]
    ) from None


class _ConfigFile(JsonConfigSettingsSource):
    """The settings in config.json. A file that cannot be read, or is not a JSON
    object, is reported like any other mistake."""

    def _read_file(self, file_path: Any) -> dict[str, Any]:
        try:
            data = super()._read_file(file_path)
        except json.JSONDecodeError as exc:
            _fail("json_invalid", ctx={"error": f"{file_path}: {exc}"})
        except UnicodeDecodeError:
            _fail("value_error", ctx={"error": ValueError(f"{file_path} is not UTF-8")})
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            _fail(
                "value_error",
                ctx={"error": ValueError(f"cannot read {file_path}: {reason}")},
            )
        if not isinstance(data, dict):
            _fail("dict_type", input=data)
        return data


class _Environment(EnvSettingsSource):
    """The environment, for the API key only, as `OPENAI_API_KEY` (not `API_KEY`).

    No other field is even looked at, so a stray `MCPSERVERS=x` or `MAX_TURNS=x` in
    the environment is not an error either.
    """

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        if field_name == "api_key":
            return os.environ.get("OPENAI_API_KEY"), field_name, False
        return None, field_name, False


class Config(BaseSettings):
    """Configuration for the AI agent."""

    # A key that is not a setting is a mistake in config.json, and says so.
    model_config = SettingsConfigDict(extra="forbid", json_file_encoding="utf-8")
    model: str = Field(default="gpt-5.4-mini", min_length=1)
    api_key: SecretStr = Field(min_length=1)  # a SecretStr stays hidden if printed
    base_url: str | None = None
    max_context_tokens: int = Field(default=64 * 1024, gt=0)
    streaming: bool = True
    max_turns: int = Field(default=25, gt=0)
    # When the history (with the instructions and the tools) passes compact_at of
    # max_context_tokens, the oldest turns are summarized, keeping about
    # keep_recent_tokens of the newest verbatim.
    auto_compact: bool = True
    compact_at: float = Field(default=0.75, gt=0, le=1)
    keep_recent_tokens: int = Field(default=15_000, ge=0)
    # None sends no reasoning setting, so the model's default applies.
    # Non-reasoning models reject any effort with HTTP 400.
    reasoning_effort: ReasoningEffort | None = None
    # Only config.json lists servers: they are nested, and no flag or variable
    # could say them. `mcpServers` in the file, as in other tools' MCP files.
    mcp_servers: dict[ServerName, McpServerConfig] = Field(
        default_factory=dict, validation_alias="mcpServers"
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _fill_in_variables(cls, value: object) -> object:
        """Let the key be `${NAME}` in the file, so that it can stay in the
        environment. This looks at the value that won, wherever it came from: a
        placeholder in the file that the environment overrode is never looked at."""
        # An unset variable raises UnsetVariableError, a ValueError: pydantic reports it.
        return fill_placeholders(value, os.environ) if isinstance(value, str) else value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest first: the flags (passed as keyword arguments), the environment (the
        # key), the file. A `.env` file is not read.
        return init_settings, _Environment(settings_cls), _ConfigFile(settings_cls)


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


def config_file(path: Path | None = None) -> Path:
    """The config file in use: the one the user named, or `config.json` in
    `user_config_dir()`, which may not exist."""
    return path or user_config_dir() / "config.json"


def load_config(path: Path | None = None, **flags: object) -> Config:
    """Load the settings: the flags, then the environment, then the file.

    The file is `path` (`--config`; no variable names one, since a project's `.envrc`
    could then point muagent at a file of its own, with servers and a base URL), or
    `config.json` in `user_config_dir()`, which may be missing. It is never looked for in the
    current directory: whoever controls a project could set the base URL, and the
    API key and every prompt would go wherever they chose.

    Args:
        path: The file, if the user named one.
        **flags: Settings given on the command line, by field name. Each wins over
            the environment and the file; None is a value too (the model's default).

    Returns:
        The settings.

    Raises:
        OSError: If the user named a file that does not exist (`FileNotFoundError`) or
            a directory.
        ValidationError: If a setting is missing or invalid, or the file is not a
            JSON object.
    """
    file = config_file(path)

    if path is not None:
        if not file.exists():
            raise FileNotFoundError(f"Config file not found: {file}")
        if file.is_dir():  # pydantic-settings would skip it without a word
            raise IsADirectoryError(f"Config path is a directory, not a file: {file}")

    class Loaded(Config):  # the settings class is told its file by its config
        model_config = SettingsConfigDict(json_file=file)

    return Loaded(**flags)  # pyright: ignore[reportArgumentType]


# ReasoningEffort is Optional[Literal[...]]; these are the Literal's values.
EFFORTS: tuple[str, ...] = get_args(get_args(ReasoningEffort)[0])
