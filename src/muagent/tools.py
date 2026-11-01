"""Tools the model can call: the @tool decorator, the registry, and how a call is run.

A tool is a plain function. Its name is the function's name, its description is its
docstring, and its parameters are inferred from its type hints, as in FastMCP. A tool
returns text for the model, and any failure is returned as text too, so the model can
read the error and correct its call.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import get_type_hints, overload

from openai.types.responses import FunctionToolParam
from pydantic import TypeAdapter, ValidationError
from pydantic.json_schema import GenerateJsonSchema

# Longer output is cut: it all goes into the next request.
MAX_OUTPUT_CHARS = 20 * 1024


class _NoTitles(GenerateJsonSchema):
    """Leave the 'title' keys that pydantic adds to each property out of the schema."""

    def field_title_should_be_set(self, schema) -> bool:
        return False


@dataclass(frozen=True)
class Tool:
    """A function the model may call, with the schema inferred from its signature."""

    name: str
    description: str  # the function's docstring
    function: Callable[..., str]
    arguments: TypeAdapter[str]  # validates the model's arguments and gives the schema
    read_only: bool  # False if it may change anything outside the program

    @classmethod
    def from_function(
        cls, function: Callable[..., str], read_only: bool = True
    ) -> Tool:
        """Describe a function as a tool, without registering it.

        Args:
            function: A synchronous function that returns text.
            read_only: Whether the tool only reads. True unless declared otherwise:
                a tool that changes anything must say so.

        Returns:
            The tool.

        Raises:
            ValueError: If the function has no docstring.
            TypeError: If the function cannot be called from the model's keyword
                arguments: it must be synchronous, return `str`, and take only
                annotated parameters that can be passed by keyword.
        """
        name = function.__name__
        description = inspect.getdoc(function)
        if not description:
            raise ValueError(
                f"Tool {name} needs a docstring: it is the description the model sees"
            )
        if inspect.iscoroutinefunction(function):
            raise TypeError(f"Tool {name} must not be async")
        hints = get_type_hints(function, include_extras=True)
        if hints.get("return") is not str:
            raise TypeError(f"Tool {name} must be annotated to return str")

        for param_name, param in inspect.signature(function).parameters.items():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise TypeError(
                    f"Tool {name}: {param_name!r} must be passable by keyword "
                    "(no *args, **kwargs or positional-only parameters)"
                )
            if param_name not in hints:
                raise TypeError(
                    f"Tool {name}: parameter {param_name!r} needs a type annotation"
                )

        # Validating JSON against a function's signature calls the function.
        arguments = TypeAdapter[str](function)  # pyright: ignore[reportArgumentType]
        return cls(name, description, function, arguments, read_only)

    def param(self) -> FunctionToolParam:
        """The tool as the Responses API wants it."""
        parameters = self.arguments.json_schema(schema_generator=_NoTitles)
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
            "strict": False,
        }

    def call(self, arguments: str) -> str:
        """Run the tool for a model's call.

        Any failure comes back as text, not as an exception, so that it can be sent
        to the model, which can then correct its call.

        Args:
            arguments: The model's arguments, as the JSON text it produced.

        Returns:
            What the tool returned, cut to MAX_OUTPUT_CHARS, or a description of
            what went wrong.
        """
        try:
            result = self.arguments.validate_json(arguments)
        except ValidationError as exc:
            problems = []
            for error in exc.errors():
                where = ".".join(str(part) for part in error["loc"])
                problems.append(f"{where}: {error['msg']}" if where else error["msg"])
            return "Invalid arguments: " + "; ".join(problems)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {type(exc).__name__}: {exc}"
        if len(result) > MAX_OUTPUT_CHARS:
            cut = len(result) - MAX_OUTPUT_CHARS
            result = f"{result[:MAX_OUTPUT_CHARS]}\n[output truncated: {cut} more characters]"
        return result


TOOLS: dict[str, Tool] = {}


@overload
def tool(function: Callable[..., str], /) -> Tool: ...
@overload
def tool(*, read_only: bool = True) -> Callable[[Callable[..., str]], Tool]: ...
def tool(
    function: Callable[..., str] | None = None, /, *, read_only: bool = True
) -> Tool | Callable[[Callable[..., str]], Tool]:
    """Register a function as a tool: `@tool`, or `@tool(read_only=False)`.

    The decorated name becomes the Tool, so a test calls `add.function(...)` and a
    list of tools is `[add, get_time]`.

    Args:
        function: The function, when used as a bare `@tool`.
        read_only: Whether the tool only reads (default True).

    Returns:
        The Tool, or a decorator that returns it.

    Raises:
        ValueError: If a tool with the same name is already registered, or the
            function has no docstring.
        TypeError: See Tool.from_function.
    """

    def register(function: Callable[..., str]) -> Tool:
        defined = Tool.from_function(function, read_only)
        if defined.name in TOOLS:
            raise ValueError(f"Tool already registered: {defined.name}")
        TOOLS[defined.name] = defined
        return defined

    return register if function is None else register(function)
