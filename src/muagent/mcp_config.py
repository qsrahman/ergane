"""How an MCP server is described in config.json, and what may be written in it.

This is the part of the MCP code that `config.py` needs: the model of one server, the
rules for its url, and the `${NAME}` placeholders. Running one is `mcp_server.py`, and all of them `mcp_hub.py`.
"""

import re
from collections.abc import Mapping
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

# OpenAI rejects the whole request, not just the tool, for a name outside this.
NAME_CHARS = "a-zA-Z0-9_-"
NAME_PATTERN = rf"^[{NAME_CHARS}]+$"  # anchored: pydantic searches, not matches


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


class McpServerConfig(BaseModel):
    """One server in config.json: a program to run (`command`) or a `url` to connect to."""

    model_config = ConfigDict(extra="forbid")

    # What kind of server it is. Optional, since command or url says it already, but
    # if given it must agree with them: a mistake is better caught than ignored.
    type: Literal["local", "remote"] | None = None
    enable: bool = True  # False keeps the entry but does not start the server
    command: str | None = Field(default=None, min_length=1)
    args: list[str] = []
    env: dict[str, str] = {}
    url: str | None = None
    headers: dict[str, str] = {}  # e.g. Authorization, for a url
    # Trust the server's tools to run without asking. Off unless the user says so.
    read_only: bool = False
    # Offer only these tools of the server (by the server's own names). Every tool's
    # definition is sent with every request, so a server with many tools costs a
    # good part of the context budget whether or not the model uses them.
    tools: list[str] | None = Field(default=None, min_length=1)
    # The defaults are Codex's: `npx` fetching a package for the first time takes
    # longer than 20 s, and a tool that builds, browses or searches longer than 60 s.
    # Ctrl-C stops a call that is not worth waiting for.
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
    startup_timeout: float = Field(default=30, gt=0)  # seconds
    call_timeout: float = Field(default=300, gt=0)  # seconds

    @model_validator(mode="after")
    def _one_transport(self) -> Self:
        if (self.command is None) == (self.url is None):
            raise ValueError(
                "give either command (a program to run) or url (a server to connect to)"
            )
        kind = "local" if self.url is None else "remote"
        if self.type not in (None, kind):
            raise ValueError(
                f"type is {self.type!r}, but a server with a "
                f"{'command' if self.url is None else 'url'} is {kind!r}"
            )
        if self.url is None:
            if self.headers:
                raise ValueError("headers go with url")
        else:
            if self.args or self.env:
                raise ValueError("args and env go with command")
            _check_url(self.url)
        return self


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class UnsetVariableError(ValueError):
    """A `${NAME}` placeholder names a variable that is not set."""

    def __init__(self, name: str) -> None:
        super().__init__(f"${{{name}}} is not set")
        self.name = name


def fill_placeholders(text: str, environ: Mapping[str, str]) -> str:
    """Fill in the `${NAME}` placeholders of a text from the environment.

    One pass: the text a variable holds is not looked at again. `$NAME` and a `${`
    without a name are plain text.

    Raises:
        UnsetVariableError: If a variable is not set. It names the variable, never a
            value.
    """

    def value(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable not in environ:
            raise UnsetVariableError(variable)
        return environ[variable]

    return _PLACEHOLDER.sub(value, text)
