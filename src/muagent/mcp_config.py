"""What config.json may name for an MCP server, apart from the model of one.

The rules for a server's name and the `${NAME}` placeholders: what `config.py` and the
code that runs the servers (`mcp_server.py`, `mcp_hub.py`) both need.
"""

import re
from collections.abc import Mapping
from typing import Annotated

from pydantic import Field

# OpenAI rejects the whole request, not just the tool, for a name outside this.
NAME_CHARS = "a-zA-Z0-9_-"
NAME_PATTERN = rf"^[{NAME_CHARS}]+$"  # anchored: pydantic searches, not matches
ServerName = Annotated[str, Field(pattern=NAME_PATTERN)]

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
