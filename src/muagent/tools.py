"""Tools the model can call: the @tool decorator, the registry, and how a call is run.

A tool is a plain function. Its name is the function's name, its description is its
docstring, and its parameters are inferred from its type hints, as in FastMCP. A tool
returns text for the model, and any failure is returned as text too, so the model can
read the error and correct its call.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_type_hints, overload

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic.json_schema import GenerateJsonSchema

Access = Literal["read", "write", "execute"]
Level = Literal["off", "read", "write", "execute"]
LEVELS: tuple[Level, ...] = ("off", "read", "write", "execute")

# Longer output is cut: it all goes into the next request.
MAX_OUTPUT_CHARS = 20_000


def _drop_title(schema: dict[str, Any]) -> None:
    schema.pop("title", None)


class _NoTitles(GenerateJsonSchema):
    """Leave the 'title' keys that pydantic adds out of the schema."""

    def field_title_should_be_set(self, schema) -> bool:
        return False


@dataclass(frozen=True)
class Tool:
    """A function the model may call, with the schema inferred from its signature."""

    name: str
    description: str  # the function's docstring
    function: Callable[..., str]
    args: type[BaseModel]  # validates the model's arguments and gives the schema
    access: Access

    @classmethod
    def from_function(
        cls, function: Callable[..., str], access: Access = "read"
    ) -> Tool:
        """Describe a function as a tool, without registering it.

        Args:
            function: A synchronous function that returns text.
            access: What the tool may do, for choosing which tools to offer.

        Returns:
            The tool.

        Raises:
            ValueError: If the function has no docstring, or access is not one of
                read, write or execute.
            TypeError: If the function cannot be called from the model's keyword
                arguments: it must be synchronous, return `str`, and take only
                annotated parameters that can be passed by keyword and that are
                not named like a pydantic attribute (such as json or copy) or with
                a leading underscore, which pydantic reserves.
        """
        name = function.__name__
        if access not in LEVELS[1:]:
            raise ValueError(
                f"Tool {name}: access must be one of {', '.join(LEVELS[1:])}, "
                f"not {access!r}"
            )
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

        fields: dict[str, Any] = {}
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
            if param_name.startswith("_") or hasattr(BaseModel, param_name):
                raise TypeError(
                    f"Tool {name}: parameter {param_name!r} cannot be used, because "
                    "pydantic reserves that name; rename it"
                )
            default = ... if param.default is param.empty else param.default
            fields[param_name] = (hints[param_name], default)

        args = create_model(
            f"{name}_args",
            __config__=ConfigDict(extra="forbid", json_schema_extra=_drop_title),
            **fields,
        )
        return cls(name, description, function, args, access)

    def param(self) -> FunctionToolParam:
        """The tool as the Responses API wants it."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.args.model_json_schema(schema_generator=_NoTitles),
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
            parsed = self.args.model_validate_json(arguments)
            result = self.function(
                **{name: getattr(parsed, name) for name in self.args.model_fields}
            )
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
def tool(*, access: Access = "read") -> Callable[[Callable[..., str]], Tool]: ...
def tool(
    function: Callable[..., str] | None = None, /, *, access: Access = "read"
) -> Tool | Callable[[Callable[..., str]], Tool]:
    """Register a function as a tool: `@tool`, or `@tool(access="write")`.

    The decorated name becomes the Tool, so a test calls `add.function(...)` and a
    list of tools is `[add, get_time]`.

    Args:
        function: The function, when used as a bare `@tool`.
        access: What the tool may do, for choosing which tools to offer.

    Returns:
        The Tool, or a decorator that returns it.

    Raises:
        ValueError: If a tool with the same name is already registered, or the
            function has no docstring.
        TypeError: See Tool.from_function.
    """

    def register(function: Callable[..., str]) -> Tool:
        defined = Tool.from_function(function, access)
        if defined.name in TOOLS:
            raise ValueError(f"Tool already registered: {defined.name}")
        TOOLS[defined.name] = defined
        return defined

    return register if function is None else register(function)


def select(level: Level) -> list[Tool]:
    """The registered tools whose access is no higher than a level.

    Args:
        level: "off" for none, or the highest access to allow.

    Returns:
        The tools to offer to the model.
    """
    allowed = LEVELS[1 : LEVELS.index(level) + 1]
    return [t for t in TOOLS.values() if t.access in allowed]
