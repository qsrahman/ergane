import asyncio
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fake_api import TEXT
from muagent.mcp_client import (
    McpServer,
    McpServerError,
    _list_tools,
    load_servers,
    start_servers,
)
from muagent.tools import MAX_OUTPUT_CHARS, Tool

FAKE = str(Path(__file__).parent / "fake_mcp_server.py")


@pytest.fixture(scope="module")
def tools(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Tool]]:
    """The fake server's tools, from one server shared by the tests that only call it."""
    log = tmp_path_factory.mktemp("mcp") / "log"
    server = McpServer("fake", sys.executable, [FAKE], log=log)
    server.start()
    yield {tool.name: tool for tool in server.tools()}
    server.close()


@pytest.fixture
def make_server(tmp_path: Path) -> Iterator[Callable[..., McpServer]]:
    """Start servers for a test (the fake one unless told otherwise); stop them after.

    The server's stderr goes to `tmp_path / "mcp.log"`.
    """
    started: list[McpServer] = []

    def make(*, start: bool = True, **values) -> McpServer:
        values = {
            "name": "fake",
            "command": sys.executable,
            "args": [FAKE],
            "log": tmp_path / "mcp.log",
            **values,
        }
        server = McpServer(**values)
        started.append(server)
        if start:
            server.start()
        return server

    yield make
    for server in started:
        server.close()


class TestTools:
    def test_names_are_prefixed_and_fit_what_the_api_accepts(self, tools):
        assert "fake__add" in tools
        assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", name) for name in tools)

    def test_names_that_collide_once_cleaned_are_numbered(self, tools):
        # The server has "search.docs" and "search_docs".
        assert tools["fake__search_docs"].call('{"query": "x"}') == "dots: x"
        assert tools["fake__search_docs_2"].call('{"query": "x"}') == "underscore: x"

    def test_a_tool_is_offered_with_its_description_and_schema(self, tools):
        offered = tools["fake__add"].param()

        assert offered["description"] == "Add two integers."
        assert offered["parameters"]["required"] == ["a"]  # pyright: ignore[reportOptionalSubscript]

    def test_no_tool_is_read_only_not_even_one_that_says_it_is(self, tools):
        # The server marks "add" and "liar" read-only; that is a claim, not a fact.
        assert not any(tool.read_only for tool in tools.values())

    def test_a_call_runs_on_the_server(self, tools):
        assert tools["fake__add"].call('{"a": 1, "b": 2}') == "3"

    def test_calls_can_run_at_the_same_time(self, tools):
        results: list[str] = []
        threads = [
            threading.Thread(
                target=lambda i=i: results.append(
                    tools["fake__add"].call(f'{{"a": {i}, "b": 0}}')
                )
            )
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(results) == ["0", "1", "2", "3"]

    def test_an_error_from_the_server_comes_back_as_text(self, tools):
        result = tools["fake__boom"].call("{}")

        assert result.startswith("Error: ")
        assert "boom" in result

    def test_a_block_that_is_not_text_becomes_a_placeholder(self, tools):
        assert tools["fake__picture"].call("{}") == "[image: image/png]"

    def test_a_long_result_is_cut(self, tools):
        result = tools["fake__big"].call(f'{{"n": {MAX_OUTPUT_CHARS + 10}}}')

        assert result.endswith("\n[output truncated: 10 more characters]")


class TestArguments:
    """What is checked before a call is sent to the server."""

    @pytest.mark.parametrize(
        ("arguments", "expected"),
        [
            ("not json", "Invalid arguments: Invalid JSON"),
            ("[1, 2]", "Invalid arguments: expected a JSON object"),
            ("{}", "Invalid arguments: a: Missing required argument"),
            ('{"b": 2}', "Invalid arguments: a: Missing required argument"),
        ],
    )
    def test_a_call_that_cannot_work_is_refused_without_asking_the_server(
        self, tools, arguments, expected
    ):
        problem = tools["fake__add"].check(arguments)

        assert problem is not None
        assert problem.startswith(expected)
        assert tools["fake__add"].call(arguments).startswith(expected)

    def test_types_are_left_to_the_server(self, tools):
        # A JSON Schema check would refuse "4"; the server accepts it.
        assert tools["fake__add"].check('{"a": "4"}') is None
        assert tools["fake__add"].call('{"a": "4"}') == "5"

    def test_what_the_server_refuses_comes_back_as_text(self, tools):
        result = tools["fake__add"].call('{"a": "x"}')

        assert result.startswith("Error: ")
        assert "valid integer" in result


class TestServer:
    def test_its_tools_can_be_trusted_as_read_only_by_the_caller(self, make_server):
        server = make_server(read_only=True)

        assert all(tool.read_only for tool in server.tools())

    def test_it_does_not_see_this_programs_environment(self, make_server, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("ASKED_FOR", "1")
        server = make_server(env={"GIVEN": "yes"})
        getenv = {tool.name: tool for tool in server.tools()}["fake__getenv"]

        assert getenv.call('{"name": "OPENAI_API_KEY"}') == "unset"
        assert getenv.call('{"name": "ASKED_FOR"}') == "unset"
        assert getenv.call('{"name": "GIVEN"}') == "yes"

    def test_its_stderr_goes_to_the_log_not_the_terminal(
        self, make_server, tmp_path, capfd
    ):
        server = make_server()
        {tool.name: tool for tool in server.tools()}["fake__chatty"].call("{}")

        assert "fake server log line" in (tmp_path / "mcp.log").read_text()
        assert "fake server log line" not in capfd.readouterr().err

    def test_a_call_that_takes_too_long_gives_up_and_the_server_stays_usable(
        self, make_server
    ):
        server = make_server(call_timeout=0.3)
        by_name = {tool.name: tool for tool in server.tools()}

        result = by_name["fake__slow"].call('{"seconds": 1}')

        assert (
            result
            == "Error: TimeoutError: MCP server 'fake' did not answer within 0.3 s"
        )
        assert by_name["fake__add"].call('{"a": 1, "b": 1}') == "2"

    def test_ctrl_c_stops_waiting_and_the_server_stays_usable(self, make_server):
        server = make_server()
        by_name = {tool.name: tool for tool in server.tools()}
        timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT))

        timer.start()
        with pytest.raises(KeyboardInterrupt):
            by_name["fake__slow"].call('{"seconds": 1}')
        timer.join()

        assert by_name["fake__add"].call('{"a": 2, "b": 2}') == "4"

    def test_a_server_that_exits_mid_call_gives_an_error_not_a_hang(self, make_server):
        server = make_server()
        by_name = {tool.name: tool for tool in server.tools()}

        assert by_name["fake__die"].call("{}").startswith("Error: ")
        assert by_name["fake__add"].call('{"a": 1, "b": 1}').startswith("Error: ")

    def test_a_call_after_close_says_the_server_is_not_running(self, make_server):
        server = make_server()
        add = {tool.name: tool for tool in server.tools()}["fake__add"]

        server.close()

        assert add.call('{"a": 1, "b": 1}') == (
            "Error: McpServerError: MCP server 'fake' is not running"
        )

    def test_close_stops_the_thread_and_can_be_repeated(self, make_server):
        server = make_server(name="closing")
        assert "mcp-closing" in [thread.name for thread in threading.enumerate()]

        server.close()
        server.close()

        assert "mcp-closing" not in [thread.name for thread in threading.enumerate()]

    @pytest.mark.parametrize("name", ["", "a.b", "a b", "é", "a/b"])
    def test_a_name_the_api_would_reject_is_refused(self, make_server, name):
        with pytest.raises(ValueError, match="letters, digits"):
            make_server(name=name, start=False)


class TestStartFailures:
    def test_a_command_that_does_not_exist(self, make_server):
        with pytest.raises(McpServerError, match="failed to start: FileNotFoundError"):
            make_server(command="/no/such/server", args=[])

    def test_a_server_that_exits_at_once(self, make_server):
        with pytest.raises(McpServerError, match="failed to start: MCPError"):
            make_server(args=["-c", "raise SystemExit(3)"])

    def test_a_server_that_never_answers_is_given_up_on(self, make_server):
        started = time.monotonic()
        with pytest.raises(McpServerError, match="did not answer within 0.5 s"):
            make_server(
                name="silent",
                args=["-c", "import time; time.sleep(60)"],
                startup_timeout=0.5,
            )

        # Cancelled at once, not left to the slow fallback in close().
        assert time.monotonic() - started < 6

        # Nothing is left running for it.
        assert "mcp-silent" not in [thread.name for thread in threading.enumerate()]


class TestLog:
    def test_the_log_is_made_where_it_is_asked_and_emptied_by_each_start(
        self, make_server, tmp_path
    ):
        log = tmp_path / "deeper" / "still" / "fake.log"
        log.parent.mkdir(parents=True)
        log.write_text("left over from the last session\n")

        server = make_server(log=log)
        {t.name: t for t in server.tools()}["fake__chatty"].call("{}")

        assert log.read_text() == "fake server log line\n"

    def test_a_log_that_cannot_be_written_is_a_start_failure(self, tmp_path):
        server = McpServer("fake", sys.executable, [FAKE], log=tmp_path)  # a directory

        with pytest.raises(McpServerError, match="cannot write its log"):
            server.start()

    def test_a_missing_mcp_package_is_explained(self, tmp_path, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        server = McpServer("fake", sys.executable, [FAKE], log=tmp_path / "log")

        with pytest.raises(McpServerError, match="needs the mcp package"):
            server.start()


class TestLoadServers:
    def write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "mcp.json"
        path.write_text(text)
        return path

    def test_no_file_means_no_servers(self, tmp_path):
        assert load_servers(tmp_path / "mcp.json") == {}

    def test_a_server_has_a_command_and_optional_arguments_environment_and_trust(
        self, tmp_path
    ):
        path = self.write(
            tmp_path,
            """{"mcpServers": {
                "plain": {"command": "tool"},
                "full": {"command": "uvx", "args": ["a", "b"], "env": {"K": "v"},
                         "read_only": true}
            }}""",
        )

        servers = load_servers(path)

        assert servers["plain"].args == [] and servers["plain"].env == {}
        assert servers["plain"].read_only is False  # trust must be asked for
        assert servers["full"].args == ["a", "b"]
        assert servers["full"].env == {"K": "v"}
        assert servers["full"].read_only is True

    def test_an_empty_object_has_no_servers(self, tmp_path):
        assert load_servers(self.write(tmp_path, "{}")) == {}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("not json", "Invalid JSON"),
            ('{"mcpServers": {"a": {}}}', "mcpServers.a.command"),
            ('{"mcpServers": {"a": {"command": ""}}}', "mcpServers.a.command"),
            ('{"mcpServers": {"a": {"command": "x", "arg": []}}}', "mcpServers.a.arg"),
            ('{"mcpServers": {"a": {"url": "http://x"}}}', "mcpServers.a.url"),
            (
                '{"mcpServers": {"a.b": {"command": "x"}}}',
                "String should match pattern",
            ),
            ('{"servers": {}}', "servers"),
        ],
        ids=["not-json", "no-command", "empty-command", "typo", "http", "name", "key"],
    )
    def test_a_mistake_names_where_it_is(self, tmp_path, text, message):
        with pytest.raises(ValidationError, match=message.replace(".", r"\.")):
            load_servers(self.write(tmp_path, text))


class TestStartServers:
    def fake(self, **values):
        from muagent.mcp_client import McpServerConfig

        return McpServerConfig(command=sys.executable, args=[FAKE], **values)

    def test_every_server_starts_and_its_tools_are_collected(self, tmp_path):
        notices: list[str] = []
        configs = {"one": self.fake(), "two": self.fake(read_only=True)}

        servers, found = start_servers(configs, tmp_path, notices.append)
        try:
            names = {tool.name: tool for tool in found}
            assert "one__add" in names and "two__add" in names
            assert not names["one__add"].read_only
            assert names["two__add"].read_only
            assert notices == [
                "MCP server 'one': 12 tools",
                "MCP server 'two': 12 tools",
            ]
            assert (tmp_path / "one.log").exists()
        finally:
            for server in servers:
                server.close()

    def test_a_server_that_fails_is_reported_and_the_others_still_start(self, tmp_path):
        from muagent.mcp_client import McpServerConfig

        notices: list[str] = []
        configs = {
            "broken": McpServerConfig(command="/no/such/server"),
            "good": self.fake(),
        }

        servers, found = start_servers(configs, tmp_path, notices.append)
        try:
            assert [server.name for server in servers] == ["good"]
            assert all(tool.name.startswith("good__") for tool in found)
            assert len(notices) == 2
            assert notices[0].startswith("MCP server 'broken' failed to start")
            assert notices[1] == "MCP server 'good': 12 tools"
        finally:
            for server in servers:
                server.close()

    def test_nothing_is_left_running_when_the_start_is_interrupted(self, tmp_path):
        def interrupt(notice: str) -> None:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            start_servers(
                {"first": self.fake(), "second": self.fake()}, tmp_path, interrupt
            )

        assert "mcp-first" not in [t.name for t in threading.enumerate()]

    def test_no_servers_is_no_work(self, tmp_path):
        assert start_servers({}, tmp_path, print) == ([], [])


class TestListTools:
    def test_every_page_is_read(self):
        pages = {
            None: SimpleNamespace(tools=["a", "b"], next_cursor="two"),
            "two": SimpleNamespace(tools=["c"], next_cursor=None),
        }

        class Session:
            async def list_tools(self, params=None):
                return pages[params.cursor if params else None]

        assert asyncio.run(_list_tools(Session())) == ["a", "b", "c"]  # pyright: ignore[reportArgumentType]


class TestAgentGate:
    """An MCP tool goes through the agent's confirmation like any other tool."""

    @pytest.fixture
    def read_file(self, tools) -> Tool:
        # The fake Responses API calls a tool named read_file.
        return replace(tools["fake__read_file"], name="read_file")

    def test_it_is_denied_unless_the_user_confirms(
        self, make_agent, fake_server, read_file
    ):
        make_agent(tools=[read_file]).run_turn("tool_call")

        output = fake_server.requests[1]["input"][-1]["output"]
        assert output == "The user denied this call."

    def test_it_runs_when_the_user_confirms(self, make_agent, fake_server, read_file):
        agent = make_agent(tools=[read_file])

        agent.run_turn("tool_call", confirm=lambda tool, arguments: True)

        output = fake_server.requests[1]["input"][-1]["output"]
        assert output == "contents of notes.txt"
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}
