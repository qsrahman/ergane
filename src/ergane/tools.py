"""Tools the model can call: the @tool decorator and how a call is run.

A tool is a plain function. Its name is the function's name, its description is its
docstring, and its parameters are inferred from its type hints, as in FastMCP. A tool
returns text for the model, and any failure is returned as text too, so the model can
read the error and correct its call.

A Tool is what the agent needs of a tool and nothing more: a name, a description, a
JSON schema for the parameters, a way to check the model's arguments and a way to run
them. Tool.from_function builds one from a function; a tool from anywhere else (an
MCP server, say) fills in the same parts.
"""

import functools
import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, get_type_hints, overload

from openai.types.responses import FunctionToolParam
from pydantic import TypeAdapter, ValidationError
from pydantic.json_schema import GenerateJsonSchema

# Longer output is cut: it all goes into the next request.
MAX_OUTPUT_CHARS = 20 * 1024


@dataclass(frozen=True)
class Image:
    """A picture a tool returns for the model to look at."""

    data: str  # base64
    mime_type: str


@dataclass(frozen=True)
class ToolOutput:
    """What a tool returns when it is more than text: the text, and pictures."""

    text: str
    images: tuple[Image, ...] = ()


class _NoTitles(GenerateJsonSchema):
    """Leave the 'title' keys that pydantic adds to each property out of the schema."""

    def field_title_should_be_set(self, schema) -> bool:
        return False


def invalid_arguments(problems: Iterable[str]) -> str:
    """The text that tells the model its arguments were wrong, and how."""
    return "Invalid arguments: " + "; ".join(problems)


def _make_check(checker: TypeAdapter[str]) -> Callable[[str], str | None]:
    """Make a check of the model's arguments out of an adapter that runs nothing."""

    def check(arguments: str) -> str | None:
        try:
            checker.validate_json(arguments)
        except ValidationError as exc:
            problems = []
            for error in exc.errors():
                where = ".".join(str(part) for part in error["loc"])
                problems.append(f"{where}: {error['msg']}" if where else error["msg"])
            return invalid_arguments(problems)
        return None

    return check


@dataclass(frozen=True)
class Tool:
    """A tool the model may call: what it is, how to check a call and how to run it."""

    name: str
    description: str  # the function's docstring
    parameters: dict[str, Any]  # JSON schema of the arguments
    # Checks the model's arguments, given as JSON text, without running the tool:
    # a description of what is wrong with them, or None if they are valid.
    check: Callable[[str], str | None]
    # Arguments as JSON text -> result, which is text or, with pictures, a ToolOutput;
    # may raise.
    runner: Callable[[str], str | ToolOutput]
    read_only: bool  # False if it may change anything outside the program
    # Stops the calls of this tool that are running, if that can be done: called when
    # the user presses Ctrl-C while several calls run together.
    cancel: Callable[[], None] | None = None
    # Arguments as JSON text -> what the call would change, as text for the user, who is
    # asked before a call that is not read-only and may ask to see it. It changes
    # nothing and may raise (the console says so). None: the arguments are all there is
    # to show.
    preview: Callable[[str], str] | None = None

    @classmethod
    def from_function(
        cls,
        function: Callable[..., str],
        read_only: bool = True,
        preview: Callable[..., str] | None = None,
    ) -> Tool:
        """Describe a function as a tool.

        Args:
            function: A synchronous function that returns text.
            read_only: Whether the tool only reads. True unless declared otherwise:
                a tool that changes anything must say so.
            preview: A function with the same parameters as `function`, which returns
                what the call would change, as text for the user to read before
                allowing it (a diff, say). It must change nothing.

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
        parameters = arguments.json_schema(schema_generator=_NoTitles)

        # Same signature, so the same validation, but it does nothing when called.
        @functools.wraps(function)
        def stub(*args, **kwargs) -> str:
            return ""

        checker = TypeAdapter[str](stub)  # pyright: ignore[reportArgumentType]
        return cls(
            name,
            description,
            parameters,
            _make_check(checker),
            arguments.validate_json,
            read_only,
            preview=(
                TypeAdapter[str](preview).validate_json  # pyright: ignore[reportArgumentType]
                if preview
                else None
            ),
        )

    def param(self) -> FunctionToolParam:
        """The tool as the Responses API wants it."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": False,
        }

    def call(self, arguments: str) -> str:
        """The text of `call_output`: what most callers want."""
        return self.call_output(arguments).text

    def call_output(self, arguments: str) -> ToolOutput:
        """Run the tool for a model's call.

        Any failure comes back as text, not as an exception, so that it can be sent
        to the model, which can then correct its call.

        Args:
            arguments: The model's arguments, as the JSON text it produced.

        Returns:
            What the tool returned, its text cut to MAX_OUTPUT_CHARS, or a
            description of what went wrong.
        """
        if problem := self.check(arguments):
            return ToolOutput(problem)
        try:
            result = self.runner(arguments)
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(f"Error: {type(exc).__name__}: {exc}")
        if isinstance(result, str):
            result = ToolOutput(result)
        text = result.text
        if len(text) > MAX_OUTPUT_CHARS:
            cut = len(text) - MAX_OUTPUT_CHARS
            text = (
                f"{text[:MAX_OUTPUT_CHARS]}\n[output truncated: {cut} more characters]"
            )
        return ToolOutput(text, result.images)


@overload
def tool(function: Callable[..., str], /) -> Tool: ...
@overload
def tool(
    *, read_only: bool = True, preview: Callable[..., str] | None = None
) -> Callable[[Callable[..., str]], Tool]: ...
def tool(
    function: Callable[..., str] | None = None,
    /,
    *,
    read_only: bool = True,
    preview: Callable[..., str] | None = None,
) -> Tool | Callable[[Callable[..., str]], Tool]:
    """Make a function a tool: `@tool`, or `@tool(read_only=False)`.

    The decorated name becomes the Tool, so a test calls `add.call('{"a": 1}')` and a
    list of tools is `[add, look]`. Nothing is registered by the decorator: a tool in a
    module of `ergane.builtin` is found by `tools_in`, and any other is listed by
    whoever builds the agent.

    Args:
        function: The function, when used as a bare `@tool`.
        read_only: Whether the tool only reads (default True).
        preview: What the user may ask to see before allowing a call (see
            `Tool.from_function`).

    Returns:
        The Tool, or a decorator that returns it.

    Raises:
        ValueError: If the function has no docstring.
        TypeError: See Tool.from_function.
    """

    def make(function: Callable[..., str]) -> Tool:
        return Tool.from_function(function, read_only, preview)

    return make if function is None else make(function)


def tools_in(
    package: ModuleType, on_notice: Callable[[str], None] | None = None
) -> list[Tool]:
    """The tools a package offers: every `Tool` at the top level of its modules.

    Adding a tool is writing a `@tool` function in a module of the package; nothing
    lists it. A module whose name starts with `_` is not looked at, and a tool one
    module imports from another is offered once. The order is the modules' names, then
    the order of definition, so a request lists the tools the same way every time.

    Args:
        package: A package (`ergane.builtin`, say), not a plain module.
        on_notice: Called with a line about a module that cannot be imported, which is
            then left out so that the others still load. Without it the error is
            raised.

    Returns:
        The tools.

    Raises:
        Exception: Whatever importing a module raises, if there is no `on_notice`.
    """
    found: dict[int, Tool] = {}  # by identity: a Tool holds a dict, so it has no hash
    for info in pkgutil.iter_modules(package.__path__):
        if info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{package.__name__}.{info.name}")
        except Exception as exc:
            if on_notice is None:
                raise
            on_notice(f"Tools in {info.name} not loaded: {type(exc).__name__}: {exc}")
            continue
        for value in vars(module).values():
            if isinstance(value, Tool):
                found[id(value)] = value
    return list(found.values())


def pick_tools(
    tools: Sequence[Tool], wanted: bool | Sequence[str]
) -> tuple[list[Tool], list[str]]:
    """The tools a setting asks for: all of them (True), none (False), or the named ones.

    Args:
        tools: The tools there are.
        wanted: The `builtin_tools` setting.

    Returns:
        The chosen tools, in the order they came, and the names that were asked for but
        that no tool has.
    """
    if isinstance(wanted, bool):
        return (list(tools) if wanted else []), []
    known = {tool.name for tool in tools}
    return (
        [tool for tool in tools if tool.name in wanted],
        [name for name in wanted if name not in known],
    )


def tool_clash(name: str) -> str:
    """The notice for a tool that is left out because another tool has its name."""
    return f"Tool {name} is defined twice; only the first is offered"
