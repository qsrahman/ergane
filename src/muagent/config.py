"""Settings: the Config model and where its values are loaded from.

A value comes from the highest of these that sets it: the command line, the
environment, `config.json`, and the default. The environment has one variable,
`OPENAI_API_KEY`, the one the OpenAI SDK reads; everything else is a flag or a key in
`config.json`, which also lists the MCP servers.

How it is built, and why:

- `load_config` is exactly that chain, in a few lines: the file's object, the key from
  the environment put over it, the flags over both, and one `Config.model_validate`.
  The flags arrive by field name; `None` is a value (`--reasoning-effort default`).
  There is no `.env` and no flag for the key (it would show in the process list).
- The environment is read for the key only. A variable for the base URL would send the
  key and every prompt wherever a project's `.envrc` said, one for the servers would
  start a program of its own, one naming the file would let a project pick it, and the
  common names (`MODEL`, `BASE_URL`) are exported by other programs. The file is the
  user's, never one in the working directory, and only `--config` names another.
- The file's keys are the fields' names (`mcp_servers` is `mcpServers`, as in other
  tools' MCP files). `extra="forbid"`: a key that is not a setting names itself in the
  error. The file is read as one JSON object by a `TypeAdapter` titled with its path,
  so bad JSON, text that is not UTF-8 and a value that is not an object are
  `ValidationError`s that name the file, like any other mistake, and `main()` has one
  path for them. A file that cannot be opened is an `OSError`.
- A server is a program (`LocalMcpServer`) or an address (`RemoteMcpServer`), told apart
  by `type`, or by having a `url`. A field of the other kind is then an error of its
  own (`Extra inputs are not permitted`), so no check of the combinations is needed.
- `api_key` may be `${NAME}`. It is filled in on the value that won, so a placeholder
  the environment overrode is never looked at.
"""

import os
from pathlib import Path
from typing import Annotated, Any, Literal, get_args
from urllib.parse import urlsplit

from openai.types.shared import ReasoningEffort
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    SecretStr,
    Tag,
    TypeAdapter,
    field_validator,
)

from muagent.mcp_config import ServerName, fill_placeholders

# The hosts that may be reached without TLS: the traffic never leaves the machine.
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def _check_url(url: str) -> None:
    """Refuse a server URL that would send what it carries where it can be read.

    Raises:
        ValueError: If it is not https (http only to this machine), names no host,
            or carries a user name or password, which belong in `headers`.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(
            f"url must be http(s)://host/path, not {parts.scheme or url!r}"
        )
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            "url must be https, except for localhost: headers and tool calls would "
            "cross the network unencrypted"
        )
    if parts.username or parts.password:
        raise ValueError("url must not hold a user name or password: use headers")


class _McpServer(BaseModel):
    """What every server in config.json has, whatever its transport."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True  # False keeps the entry but does not start the server
    # Trust the server's tools to run without asking. Off unless the user says so.
    read_only: bool = False
    # Offer only these tools of the server (by the server's own names). Every tool's
    # definition is sent with every request, so a server with many tools costs a
    # good part of the context budget whether or not the model uses them.
    tools: list[str] | None = Field(default=None, min_length=1)
    # Pass the server's own description of how to use its tools to the model, in its
    # instructions. Off unless the user says so: the text is the server's, and a model
    # can be steered by it (gpt-oss:20b followed an order hidden in it 3 times in 5,
    # framed as notes or not). A call that is not read-only still asks first.
    instructions: bool = False
    # Give the model the pictures a tool returns (a screenshot, say), not just a note that
    # there was one. Off unless the user says so: a picture costs tokens in every later
    # request until it is dropped, and a model without vision, or a compatible server
    # that does not take them in a tool's output, would fail.
    images: bool = False
    # The defaults are Codex's: `npx` fetching a package for the first time takes
    # longer than 20 s, and a tool that builds, browses or searches longer than 60 s.
    # Ctrl-C stops a call that is not worth waiting for.
    startup_timeout: float = Field(default=30, gt=0)  # seconds
    call_timeout: float = Field(default=300, gt=0)  # seconds


class LocalMcpServer(_McpServer):
    """A program to run, which the agent talks to over its standard input and output."""

    type: Literal["local"] = "local"
    command: str = Field(min_length=1)
    args: list[str] = []
    env: dict[str, str] = {}


class RemoteMcpServer(_McpServer):
    """A server to connect to."""

    type: Literal["remote"] = "remote"
    url: str
    headers: dict[str, str] = {}  # e.g. Authorization

    @field_validator("url")
    @classmethod
    def _safe_url(cls, url: str) -> str:
        _check_url(url)
        return url


def _server_kind(entry: Any) -> str | None:
    """Tell which kind of server an entry is: its `type`, else whether it has a `url`.

    None, for an entry that is no object or a `type` that is no text, is reported by
    pydantic as a tag it cannot find.
    """
    if isinstance(entry, dict):
        kind = entry.get("type")
        if kind is None:
            return "remote" if "url" in entry else "local"
        return kind if isinstance(kind, str) else None
    return getattr(entry, "type", None)


McpServerConfig = Annotated[
    Annotated[LocalMcpServer, Tag("local")] | Annotated[RemoteMcpServer, Tag("remote")],
    Discriminator(_server_kind),
]


class Config(BaseModel):
    """Configuration for the AI agent."""

    # A key that is not a setting is a mistake in config.json, and says so.
    model_config = ConfigDict(extra="forbid")

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
            a directory, or a file cannot be read.
        ValidationError: If a setting is missing or invalid, or the file is not a
            JSON object (it is named in the message).
    """
    file = config_file(path)
    data: dict[str, Any] = {}
    if path is not None or file.exists():
        # Titled with the path, so that the error says which file it is about.
        file_object = TypeAdapter(dict[str, Any], config=ConfigDict(title=str(file)))
        data = file_object.validate_json(file.read_bytes())
    if key := os.environ.get("OPENAI_API_KEY"):
        data["api_key"] = key
    return Config.model_validate(data | flags)


# ReasoningEffort is Optional[Literal[...]]; these are the Literal's values.
EFFORTS: tuple[str, ...] = get_args(get_args(ReasoningEffort)[0])
