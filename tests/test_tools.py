import importlib
import inspect
import re
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import Field, ValidationError

from ergane import builtin
from ergane.tools import (
    MAX_OUTPUT_CHARS,
    Image,
    Tool,
    ToolOutput,
    pick_tools,
    tool,
    tool_clash,
    tools_in,
    without_missing_programs,
)


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


class TestAToolThatReturnsPictures:
    def test_str_or_tool_output_is_an_allowed_return_type(self):
        def shot() -> str | ToolOutput:
            """Take a screenshot."""
            return ToolOutput("a screenshot", (Image("aGk=", "image/png"),))

        defined = Tool.from_function(shot)

        assert defined.call_output("{}").images == (Image("aGk=", "image/png"),)

    def test_other_unions_are_not(self):
        def shot() -> str | int:
            """Take a screenshot."""
            return "x"

        with pytest.raises(TypeError, match="must be annotated to return str"):
            Tool.from_function(shot)  # pyright: ignore[reportArgumentType]


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


class TestToolsIn:
    """A tool in a module of a package is offered without being listed anywhere."""

    @pytest.fixture
    def package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Make a package in `tmp_path`: `package({"mod": "source", ...})`."""
        name = f"tools_pkg_{tmp_path.name.replace('-', '_')}"  # new for each test
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(tmp_path))

        def make(modules: dict[str, str]):
            for module, source in modules.items():
                (root / f"{module}.py").write_text(source)
            importlib.invalidate_caches()
            return importlib.import_module(name)

        return make

    @staticmethod
    def defining(*names: str, read_only: bool = True) -> str:
        decorator = "@tool" if read_only else "@tool(read_only=False)"
        body = "".join(
            f'\n{decorator}\ndef {n}() -> str:\n    """Do {n}."""\n    return "{n}"\n'
            for n in names
        )
        return "from ergane.tools import tool\n" + body

    def test_a_tool_in_a_module_is_offered(self, package):
        found = tools_in(package({"files": self.defining("look")}))

        assert [t.name for t in found] == ["look"]
        assert found[0].call("{}") == "look"

    def test_a_tool_that_changes_things_keeps_saying_so(self, package):
        found = tools_in(package({"files": self.defining("save", read_only=False)}))

        assert found[0].read_only is False

    def test_the_order_is_the_modules_names_then_the_order_of_definition(self, package):
        found = tools_in(
            package({"b": self.defining("one", "two"), "a": self.defining("three")})
        )

        assert [t.name for t in found] == ["three", "one", "two"]

    def test_a_module_whose_name_starts_with_an_underscore_is_left_alone(self, package):
        found = tools_in(
            package({"_helpers": self.defining("hidden"), "ok": self.defining("shown")})
        )

        assert [t.name for t in found] == ["shown"]

    def test_a_tool_imported_from_another_module_is_offered_once(self, package):
        name = package({"a": self.defining("look")}).__name__
        found = tools_in(
            package({"a": self.defining("look"), "b": f"from {name}.a import look\n"})
        )

        assert [t.name for t in found] == ["look"]

    def test_a_function_that_is_not_a_tool_is_not_offered(self, package):
        source = 'def helper() -> str:\n    return "x"\n' + self.defining("look")

        assert [t.name for t in tools_in(package({"m": source}))] == ["look"]

    def test_a_module_that_cannot_be_imported_is_an_error_without_a_notice(
        self, package
    ):
        with pytest.raises(ZeroDivisionError):
            tools_in(package({"broken": "1 / 0\n"}))

    def test_with_a_notice_it_is_left_out_and_the_others_load(self, package):
        notices: list[str] = []
        modules = {"a_broken": "1 / 0\n", "ok": self.defining("look")}

        found = tools_in(package(modules), notices.append)

        assert [t.name for t in found] == ["look"]
        assert notices == [
            "Tools in a_broken not loaded: ZeroDivisionError: division by zero"
        ]

    def test_a_syntax_error_is_left_out_too(self, package):
        notices: list[str] = []

        found = tools_in(package({"bad": "def (:\n"}), notices.append)

        assert found == [] and "Tools in bad not loaded: SyntaxError" in notices[0]

    def test_the_built_in_tools_all_have_names_the_api_takes(self):
        found = tools_in(builtin)

        names = [t.name for t in found]
        assert len(set(names)) == len(names)
        assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in names)


def test_a_tool_left_out_for_its_name_is_reported_by_it():
    assert tool_clash("add") == "Tool add is defined twice; only the first is offered"


class TestPickTools:
    @staticmethod
    def three() -> list[Tool]:
        def make(name: str) -> Tool:
            def function() -> str:
                """Say it."""
                return name

            function.__name__ = name
            return Tool.from_function(function)

        return [make("alpha"), make("beta"), make("gamma")]

    def test_true_is_all_of_them(self):
        tools = self.three()

        assert pick_tools(tools, True) == (tools, [])

    def test_false_is_none(self):
        assert pick_tools(self.three(), False) == ([], [])

    def test_names_pick_those_in_the_order_the_tools_came(self):
        tools = self.three()

        chosen, unknown = pick_tools(tools, ["gamma", "alpha"])

        assert [t.name for t in chosen] == ["alpha", "gamma"]
        assert unknown == []

    def test_a_name_that_no_tool_has_is_returned(self):
        chosen, unknown = pick_tools(self.three(), ["beta", "nope", "nada"])

        assert [t.name for t in chosen] == ["beta"]
        assert unknown == ["nope", "nada"]

    def test_an_empty_list_is_none(self):
        assert pick_tools(self.three(), []) == ([], [])


class TestPreview:
    @staticmethod
    def save(path: str, text: str) -> str:
        """Save text."""
        return path

    @staticmethod
    def show(path: str, text: str) -> str:
        return f"{path} would hold {text!r}"

    def test_a_tool_has_none_unless_it_is_given_one(self):
        assert Tool.from_function(self.save, read_only=False).preview is None

    def test_a_preview_is_called_with_the_arguments_the_model_sent(self):
        built = Tool.from_function(self.save, read_only=False, preview=self.show)

        assert built.preview is not None
        assert built.preview('{"path": "a.txt", "text": "hi"}') == (
            "a.txt would hold 'hi'"
        )

    def test_a_preview_checks_the_arguments_like_the_tool_does(self):
        built = Tool.from_function(self.save, read_only=False, preview=self.show)

        assert built.preview is not None
        with pytest.raises(ValidationError):
            built.preview('{"path": "a.txt"}')

    def test_the_decorator_takes_it_too(self):
        @tool(read_only=False, preview=self.show)
        def save(path: str, text: str) -> str:
            """Save text."""
            return path

        assert save.preview is not None and save.preview(
            '{"path": "p", "text": "t"}'
        ) == ("p would hold 't'")

    def test_a_preview_changes_nothing_about_the_definition_the_model_sees(self):
        plain = Tool.from_function(self.save, read_only=False)
        with_preview = Tool.from_function(self.save, read_only=False, preview=self.show)

        assert plain.param() == with_preview.param()


class TestRequiredPrograms:
    def tool_needing(self, program: str | None) -> Tool:
        @tool(requires=program)
        def look() -> str:
            """Look."""
            return ""

        return look

    def test_a_tool_that_needs_nothing_is_always_ready(self):
        plain = self.tool_needing(None)

        assert without_missing_programs([plain]) == ([plain], [])

    def test_a_program_on_the_path_makes_the_tool_ready(self):
        python = self.tool_needing("python3")

        assert without_missing_programs([python]) == ([python], [])

    def test_a_program_that_is_not_installed_leaves_the_tool_out(self):
        missing = self.tool_needing("no-such-program-ergane")
        plain = self.tool_needing(None)

        ready, left_out = without_missing_programs([missing, plain])

        assert ready == [plain] and left_out == [missing]

    def test_the_builtin_search_needs_ripgrep(self):
        found = {t.name: t for t in tools_in(builtin)}

        assert found["search_files"].requires == "rg"
        assert found["read_file"].requires is None
