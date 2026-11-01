import inspect
from typing import Annotated, Any

import pytest
from pydantic import Field

from muagent import tools
from muagent.tools import MAX_OUTPUT_CHARS, Tool, select, tool


def add(a: int, b: int = 1) -> str:
    """Add two integers.

    Args:
        a: The first number.
    """
    return str(a + b)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Tool]:
    """An empty registry, so the tools a test defines do not leak into others."""
    empty: dict[str, Tool] = {}
    monkeypatch.setattr(tools, "TOOLS", empty)
    return empty


class TestFromFunction:
    def test_name_and_description_come_from_the_function(self):
        defined = Tool.from_function(add)

        assert defined.name == "add"
        assert defined.description == inspect.getdoc(add)

    def test_the_schema_is_inferred_from_the_signature(self):
        assert Tool.from_function(add).param() == {
            "type": "function",
            "name": "add",
            "description": inspect.getdoc(add),
            "parameters": {
                "additionalProperties": False,
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"default": 1, "type": "integer"},
                },
                "required": ["a"],
                "type": "object",
            },
            "strict": False,
        }

    def test_annotated_limits_and_descriptions_reach_the_schema(self):
        def read(
            path: str,
            max_chars: Annotated[int, Field(le=200, description="Longest text")] = 100,
        ) -> str:
            """Read a file."""
            return path

        parameters: dict[str, Any] = (
            Tool.from_function(read).param()["parameters"] or {}
        )

        assert parameters["properties"]["max_chars"] == {
            "default": 100,
            "description": "Longest text",
            "maximum": 200,
            "type": "integer",
        }

    @pytest.mark.parametrize("access", ["admin", "off", ""])
    def test_an_access_level_that_is_not_read_write_or_execute_is_rejected(
        self, access
    ):
        with pytest.raises(
            ValueError, match="access must be one of read, write, execute"
        ):
            Tool.from_function(add, access=access)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "param", ["json", "copy", "schema", "validate", "model_config", "_secret"]
    )
    def test_a_parameter_name_that_pydantic_reserves_is_rejected(self, param):
        # Without this, json/copy/schema warn at every start-up, model_config can
        # never be passed, and an underscore name is silently dropped.
        def function(**kwargs: str) -> str:
            """Doc."""
            return ""

        # Give it a signature with one parameter of the name under test.
        parameter = inspect.Parameter(
            param, inspect.Parameter.POSITIONAL_OR_KEYWORD, default="x", annotation=str
        )
        function.__signature__ = inspect.Signature([parameter], return_annotation=str)  # pyright: ignore[reportFunctionMemberAccess]
        function.__annotations__ = {param: str, "return": str}

        with pytest.raises(TypeError, match=f"parameter '{param}' cannot be used"):
            Tool.from_function(function)

    @pytest.mark.parametrize(
        ("make", "error", "message"),
        [
            ("no_docstring", ValueError, "needs a docstring"),
            ("returns_int", TypeError, "must be annotated to return str"),
            ("unannotated", TypeError, "needs a type annotation"),
            ("var_args", TypeError, "passable by keyword"),
            ("positional_only", TypeError, "passable by keyword"),
            ("async_tool", TypeError, "must not be async"),
        ],
    )
    def test_a_function_that_cannot_be_a_tool_is_rejected(self, make, error, message):
        def no_docstring(path: str) -> str:
            return path

        def returns_int(path: str) -> int:  # type: ignore[return-value]
            """Doc."""
            return 1

        def unannotated(path) -> str:
            """Doc."""
            return path

        def var_args(*paths: str) -> str:
            """Doc."""
            return ""

        def positional_only(path: str, /) -> str:
            """Doc."""
            return path

        async def async_tool(path: str) -> str:  # type: ignore[return-value]
            """Doc."""
            return path

        function = locals()[make]
        with pytest.raises(error, match=message):
            Tool.from_function(function)


class TestCall:
    def test_runs_the_function_with_the_models_arguments(self):
        assert Tool.from_function(add).call('{"a": 2, "b": 3}') == "5"

    def test_defaults_apply_and_numbers_given_as_text_are_converted(self):
        assert Tool.from_function(add).call('{"a": "4"}') == "5"

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ("{}", "Invalid arguments: a: Field required"),
            ('{"a": "x"}', "Invalid arguments: a: Input should be a valid integer"),
            (
                '{"a": 1, "c": 2}',
                "Invalid arguments: c: Extra inputs are not permitted",
            ),
        ],
    )
    def test_invalid_arguments_come_back_as_text(self, arguments, expected):
        assert Tool.from_function(add).call(arguments).startswith(expected)

    def test_text_that_is_not_json_comes_back_as_text(self):
        result = Tool.from_function(add).call("add them please")

        assert result.startswith("Invalid arguments: Invalid JSON")

    @pytest.mark.parametrize(
        ("raises", "expected"),
        [
            (
                ZeroDivisionError("division by zero"),
                "Error: ZeroDivisionError: division by zero",
            ),
            (
                FileNotFoundError("no such file"),
                "Error: FileNotFoundError: no such file",
            ),
            (
                UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte"),
                "Error: UnicodeDecodeError: ",
            ),
        ],
    )
    def test_any_failure_of_the_function_comes_back_as_text(self, raises, expected):
        def broken() -> str:
            """Fails."""
            raise raises

        assert Tool.from_function(broken).call("{}").startswith(expected)

    def test_ctrl_c_is_not_swallowed(self):
        def interrupted() -> str:
            """Interrupted."""
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            Tool.from_function(interrupted).call("{}")

    def test_long_output_is_cut_and_says_so(self):
        def long() -> str:
            """Long."""
            return "x" * (MAX_OUTPUT_CHARS + 5)

        result = Tool.from_function(long).call("{}")

        assert (
            result == "x" * MAX_OUTPUT_CHARS + "\n[output truncated: 5 more characters]"
        )


class TestRegistry:
    def test_a_bare_decorator_registers_and_returns_the_tool(self, registry):
        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        assert isinstance(ping, Tool)
        assert registry == {"ping": ping}
        assert ping.function() == "pong"

    def test_the_decorator_can_set_the_access_level(self, registry):
        @tool(access="write")
        def save() -> str:
            """Save."""
            return "saved"

        assert registry["save"].access == "write"

    def test_registering_a_name_twice_is_an_error(self, registry):
        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        with pytest.raises(ValueError, match="already registered: ping"):
            tool(ping.function)

    def test_select_offers_the_tools_up_to_a_level(self, registry):
        @tool
        def look() -> str:
            """Look."""
            return ""

        @tool(access="write")
        def save() -> str:
            """Save."""
            return ""

        @tool(access="execute")
        def run() -> str:
            """Run."""
            return ""

        names = {level: [t.name for t in select(level)] for level in tools.LEVELS}

        assert names == {
            "off": [],
            "read": ["look"],
            "write": ["look", "save"],
            "execute": ["look", "save", "run"],
        }
