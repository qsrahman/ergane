import inspect
from typing import Annotated, Any

import pytest
from pydantic import Field

from muagent.tools import MAX_OUTPUT_CHARS, Image, Tool, ToolOutput, tool


def add(a: int, b: int = 1) -> str:
    """Add two integers.

    Args:
        a: The first number.
    """
    return str(a + b)


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

    def test_a_tool_is_read_only_unless_declared_otherwise(self):
        assert Tool.from_function(add).read_only is True
        assert Tool.from_function(add, read_only=False).read_only is False

    def test_parameter_names_that_clash_with_pydantic_attributes_work(self):
        def function(json: str, copy: str = "c", _secret: str = "s") -> str:
            """Doc."""
            return f"{json}{copy}{_secret}"

        defined = Tool.from_function(function)

        assert list(defined.param()["parameters"]["properties"]) == [  # type: ignore[index]
            "json",
            "copy",
            "_secret",
        ]
        assert defined.call('{"json": "j", "_secret": "t"}') == "jct"

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


class TestCheck:
    def test_valid_arguments_have_no_problem(self):
        assert Tool.from_function(add).check('{"a": 1}') is None

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ("{}", "Invalid arguments: a: Missing required argument"),
            ('{"a": "x"}', "Invalid arguments: a: Input should be a valid integer"),
            ('{"a": 1, "c": 2}', "Invalid arguments: c: Unexpected keyword argument"),
            ("add them please", "Invalid arguments: Invalid JSON"),
        ],
    )
    def test_invalid_arguments_are_described(self, arguments, expected):
        problem = Tool.from_function(add).check(arguments)

        assert problem is not None
        assert problem.startswith(expected)

    def test_the_function_is_not_run(self):
        ran: list[str] = []

        def save(path: str) -> str:
            """Save a file."""
            ran.append(path)
            return "saved"

        Tool.from_function(save).check('{"path": "notes.txt"}')

        assert ran == []


class TestCall:
    def test_runs_the_function_with_the_models_arguments(self):
        assert Tool.from_function(add).call('{"a": 2, "b": 3}') == "5"

    def test_defaults_apply_and_numbers_given_as_text_are_converted(self):
        assert Tool.from_function(add).call('{"a": "4"}') == "5"

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ("{}", "Invalid arguments: a: Missing required argument"),
            ('{"a": "x"}', "Invalid arguments: a: Input should be a valid integer"),
            (
                '{"a": 1, "c": 2}',
                "Invalid arguments: c: Unexpected keyword argument",
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


ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


class TestToolFromParts:
    """A tool that is not made from a function, as an MCP server's tool will be."""

    def make(self, runner, check=lambda arguments: None) -> Tool:
        return Tool(
            "remote__echo",
            "Echo some text.",
            ECHO_SCHEMA,
            check,
            runner,
            read_only=False,
        )

    def test_it_is_offered_with_the_schema_it_was_given(self):
        assert self.make(lambda arguments: "").param() == {
            "type": "function",
            "name": "remote__echo",
            "description": "Echo some text.",
            "parameters": ECHO_SCHEMA,
            "strict": False,
        }

    def test_check_is_the_check_it_was_given(self):
        tool = self.make(lambda arguments: "", lambda arguments: f"bad: {arguments}")

        assert tool.check("{}") == "bad: {}"

    def test_a_call_that_fails_the_check_is_not_run(self):
        ran: list[str] = []
        tool = self.make(ran.append, lambda arguments: "Invalid arguments: no text")

        assert tool.call("{}") == "Invalid arguments: no text"
        assert ran == []

    def test_a_call_returns_what_the_runner_returns_cut_to_the_limit(self):
        tool = self.make(lambda arguments: "x" * (MAX_OUTPUT_CHARS + 3))

        result = tool.call('{"text": "hi"}')

        assert result.endswith("\n[output truncated: 3 more characters]")

    def test_a_failure_of_the_runner_comes_back_as_text(self):
        def runner(arguments: str) -> str:
            raise ConnectionError("server went away")

        assert self.make(runner).call('{"text": "hi"}') == (
            "Error: ConnectionError: server went away"
        )


class TestPictures:
    PICTURE = Image("aGk=", "image/png")

    def make(self, runner) -> Tool:
        return Tool("remote__shot", "", ECHO_SCHEMA, lambda a: None, runner, True)

    def test_a_tool_may_return_pictures_with_its_text(self):
        tool = self.make(lambda arguments: ToolOutput("a screenshot", (self.PICTURE,)))

        assert tool.call_output("{}") == ToolOutput("a screenshot", (self.PICTURE,))

    def test_call_is_the_text_of_the_output(self):
        tool = self.make(lambda arguments: ToolOutput("a screenshot", (self.PICTURE,)))

        assert tool.call("{}") == "a screenshot"

    def test_text_is_returned_as_text_without_pictures(self):
        assert self.make(lambda arguments: "plain").call_output("{}") == ToolOutput(
            "plain"
        )

    def test_the_text_is_cut_and_the_pictures_are_kept(self):
        long = "x" * (MAX_OUTPUT_CHARS + 5)
        tool = self.make(lambda arguments: ToolOutput(long, (self.PICTURE,)))

        result = tool.call_output("{}")

        assert result.text.endswith("[output truncated: 5 more characters]")
        assert result.images == (self.PICTURE,)


class TestDecorator:
    def test_a_bare_decorator_returns_the_tool(self):
        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        assert isinstance(ping, Tool)
        assert ping.call("{}") == "pong"

    def test_the_decorator_can_declare_a_tool_not_read_only(self):
        @tool
        def look() -> str:
            """Look."""
            return "seen"

        @tool(read_only=False)
        def save() -> str:
            """Save."""
            return "saved"

        assert look.read_only is True
        assert save.read_only is False
