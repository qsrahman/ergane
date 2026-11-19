"""What an MCP server's tools and results become for the model.

Names for the tools (`server__tool`, with what the API rejects replaced), the text and
the pictures of a result, the check of a call's arguments, and the listing of every page
of a server's tools. All of it is plain functions of what the SDK gives; the connection
is `mcp_client.py`. The `mcp` types are imported where they are used, so that ergane
works without the package as long as no server is configured.
"""

import json
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ergane.tools import Image, invalid_arguments

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp_types import CallToolResult
    from mcp_types import Tool as ToolDefinition

# What OpenAI takes in a tool name: the characters it does not are replaced, and a name
# is cut at the longest it takes (a server's name has the same rule, in `config.py`).
_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_NAME_CHARS = 128

# The kinds of picture the Responses API takes, and the most of one (as base64) that is
# passed on: it is in every request until it is dropped.
_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
_MAX_IMAGE_CHARS = 5_000_000


def tool_name(server: str, tool: str, taken: set[str]) -> str:
    """Name a server's tool for the model, unique among `taken` (which it extends).

    The server's name is a prefix, so servers cannot clash with each other or with
    ergane's own tools; characters the API rejects become underscores, and names
    that collide after that get a number.
    """
    base = _DISALLOWED.sub("_", f"{server}__{tool}")[:_MAX_NAME_CHARS]
    name, number = base, 1
    while name in taken:
        number += 1
        suffix = f"_{number}"
        name = base[: _MAX_NAME_CHARS - len(suffix)] + suffix
    taken.add(name)
    return name


def result_text(result: CallToolResult) -> str:
    """The text a tool result carries for the model.

    Blocks that are not text (an image, say) become a short placeholder. A text file
    the server embedded is text; a result with only structured content is that,
    as JSON.
    """
    from mcp_types import (
        EmbeddedResource,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )

    parts = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, EmbeddedResource):
            resource = block.resource  # a text file the server embedded is text
            parts.append(
                resource.text
                if isinstance(resource, TextResourceContents)
                else f"[resource: {resource.uri}]"
            )
        elif isinstance(block, ResourceLink):
            parts.append(f"[resource_link: {block.uri}]")
        else:
            mime_type = getattr(block, "mime_type", None)
            parts.append(
                f"[{block.type}: {mime_type}]" if mime_type else f"[{block.type}]"
            )
    if not parts and result.structured_content:
        # The spec asks for text next to it, but a server may send only this.
        parts.append(json.dumps(result.structured_content))
    text = "\n".join(parts)
    return f"Error: {text}" if result.is_error else text


def result_images(result: CallToolResult) -> tuple[Image, ...]:
    """The pictures of a result that the model can be given: of a kind the API takes,
    and not so large that one would fill a request."""
    from mcp_types import ImageContent

    return tuple(
        Image(block.data, block.mime_type)
        for block in result.content
        if isinstance(block, ImageContent)
        and block.mime_type in _IMAGE_TYPES
        and len(block.data) <= _MAX_IMAGE_CHARS
    )


def required_keys_check(schema: Mapping[str, Any]) -> Callable[[str], str | None]:
    """Check what a call cannot work without: a JSON object with the required keys.

    Types are left to the server. A JSON Schema check would be stricter than the
    server: it rejects `"4"` for an integer, which servers accept.
    """
    required = schema.get("required", [])

    def check(arguments: str) -> str | None:
        try:
            given = json.loads(arguments)
        except ValueError as exc:
            return invalid_arguments([f"Invalid JSON: {exc}"])
        if not isinstance(given, dict):
            return invalid_arguments(["expected a JSON object"])
        missing = [key for key in required if key not in given]
        if missing:
            return invalid_arguments(
                f"{key}: Missing required argument" for key in missing
            )
        return None

    return check


async def list_tools(session: ClientSession) -> list[ToolDefinition]:
    """Every tool the server lists, across its pages."""
    from mcp_types import PaginatedRequestParams

    found: list[ToolDefinition] = []
    cursor = None
    while True:
        params = PaginatedRequestParams(cursor=cursor) if cursor else None
        page = await session.list_tools(params=params)
        found += page.tools
        cursor = page.next_cursor
        if cursor is None:
            return found
