"""Settings: the Config model and where its values are loaded from.

A value comes from the highest of these that sets it: the command line, the
environment, `config.json`, and the default. `config.json` also lists the MCP
servers, which only it can do.

How it is built, and why:

- `Config.settings_customise_sources` is exactly that chain. The flags arrive as
  keyword arguments (pydantic-settings' highest source) by field name; `None` is a
  value (`--reasoning-effort default`). There is no `.env` and no flag for the key (it
  would show in the process list).
- The file is the user's, never one in the working directory: whoever controls a
  project could set `base_url`, and the key and every prompt would go there. For the
  same reason `_Environment` drops `mcpServers` (a server is a program), which
  pydantic-settings would read from a JSON variable. `load_config` gives the path to a
  subclass through `json_file`, since it is only known at run time.
- Three fields have the environment's names as aliases (`OPENAI_*`), so `_ConfigFile`
  renames those keys; the others are read from the variable named like the field, in
  any case. `extra="forbid"`: a key that is not a setting names itself in the error,
  and a file that is not a JSON object is a `ValidationError` too (`_fail`), so
  `main()` has one path for every mistake.
- `api_key` may be `${NAME}`. It is filled in on the value that won, so a placeholder
  the environment overrode is never looked at.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Any, NoReturn, get_args

from openai.types.shared import ReasoningEffort
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from muagent.mcp_client import (
    NAME_PATTERN,
    McpServerConfig,
    UnsetVariableError,
    fill_placeholders,
)


def _fail(kind: str, **details: Any) -> NoReturn:
    """Report config.json as invalid the way pydantic reports a bad setting."""
    raise ValidationError.from_exception_data(
        "config.json",
        [{"type": kind, "loc": (), "input": "", **details}],  # pyright: ignore[reportArgumentType]
    ) from None


class _ConfigFile(JsonConfigSettingsSource):
    """The settings in config.json, under the names a user writes there.

    The environment calls three settings `OPENAI_*`, which is also what pydantic
    knows them by; in the file they are `model`, `base_url` and `api_key`, like the
    others. A file that is not a JSON object is reported like any other mistake.
    """

    def _read_file(self, file_path: Any) -> dict[str, Any]:
        try:
            data = super()._read_file(file_path)
        except json.JSONDecodeError as exc:
            _fail("json_invalid", ctx={"error": f"{file_path}: {exc}"})
        if not isinstance(data, dict):
            _fail("dict_type", input=data)
        renamed = {
            name: str(field.validation_alias)
            for name, field in Config.model_fields.items()
            if field.validation_alias
        }
        return {renamed.get(key, key): value for key, value in data.items()}


class _Environment(EnvSettingsSource):
    """The environment, without the MCP servers.

    A server is a program muagent runs. pydantic-settings would read the field from
    a JSON variable (`MCPSERVERS`), and then a `.envrc` in a project could start
    programs: only the user's own config.json may.
    """

    def __call__(self) -> dict[str, Any]:
        values = super().__call__()
        values.pop("mcpServers", None)  # the key is the alias
        return values


class Config(BaseSettings):
    """Configuration for the AI agent."""

    # A field without an alias is read from the environment variable of the same
    # name, in any case: max_turns from MAX_TURNS. A key that is not a setting is
    # a mistake in config.json, and says so.
    model_config = SettingsConfigDict(extra="forbid", json_file_encoding="utf-8")
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
    # When the history (with the instructions and the tools) passes compact_at of
    # max_context_tokens, the oldest turns are summarized, keeping about
    # keep_recent_tokens of the newest verbatim.
    auto_compact: bool = Field(
        default=True,
    )
    compact_at: float = Field(
        default=0.75,
        gt=0,
        le=1,
    )
    keep_recent_tokens: int = Field(
        default=15_000,
        ge=0,
    )
    # None sends no reasoning setting, so the model's default applies.
    # Non-reasoning models reject any effort with HTTP 400.
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
    )
    # Only config.json lists servers: they are nested, and no flag or variable
    # could say them.
    mcp_servers: dict[Annotated[str, Field(pattern=NAME_PATTERN)], McpServerConfig] = (
        Field(default_factory=dict, validation_alias="mcpServers")
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _fill_in_variables(cls, value: object) -> object:
        """Let the key be `${NAME}` in the file, so that it can stay in the
        environment. This looks at the value that won, wherever it came from: a
        placeholder in the file that the environment overrode is never looked at."""
        if not isinstance(value, str):
            return value
        try:
            return fill_placeholders(value, os.environ)
        except UnsetVariableError as exc:
            raise ValueError(str(exc)) from None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest first: the flags (passed as keyword arguments), the environment,
        # the file. A `.env` file is not read.
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


def load_config(path: Path | None = None, **flags: object) -> Config:
    """Load the settings: the flags, then the environment, then the file.

    The file is `path`, or the one `MUAGENT_CONFIG` names, or `config.json` in
    `user_config_dir()`; the last may be missing. It is never looked for in the
    current directory: whoever controls a project could set the base URL, and the
    API key and every prompt would go wherever they chose.

    Args:
        path: The file, if the user named one.
        **flags: Settings given on the command line, by field name. Each wins over
            the environment and the file; None is a value too (the model's default).

    Returns:
        The settings.

    Raises:
        FileNotFoundError: If the user named a file that does not exist.
        ValidationError: If a setting is missing or invalid, or the file is not a
            JSON object.
    """
    if path is None and os.environ.get("MUAGENT_CONFIG"):
        path = Path(os.environ["MUAGENT_CONFIG"])
    if path is not None and not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    class Loaded(Config):  # the settings class is told its file by its config
        model_config = SettingsConfigDict(
            json_file=path or user_config_dir() / "config.json"
        )

    names = {
        name: str(field.validation_alias or name)
        for name, field in Config.model_fields.items()
    }
    return Loaded(**{names[name]: value for name, value in flags.items()})  # pyright: ignore[reportArgumentType]


# ReasoningEffort is Optional[Literal[...]]; these are the Literal's values.
EFFORTS: tuple[str, ...] = get_args(get_args(ReasoningEffort)[0])
