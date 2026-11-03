import asyncio
import contextlib
import gc
import json
import os
import re
import signal
import socket
import subprocess
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
    McpServerConfig,
    McpServerError,
    _expand,
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
    server = McpServer(
        "fake", McpServerConfig(command=sys.executable, args=[FAKE]), log
    )
    server.start()
    yield {tool.name: tool for tool in server.tools()}
    server.close()


@contextlib.contextmanager
def serve_http() -> Iterator[str]:
    """Run the fake server over streamable HTTP on a free port; yield its url."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, FAKE, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(200):
            try:
                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("the fake HTTP server did not start")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        process.wait(10)


@pytest.fixture(scope="module")
def http_url() -> Iterator[str]:
    with serve_http() as url:
        yield url


@pytest.fixture
def make_server(tmp_path: Path) -> Iterator[Callable[..., McpServer]]:
    """Start servers for a test (the fake one unless told otherwise); stop them after.

    The server's stderr goes to `tmp_path / "mcp.log"`.
    """
    started: list[McpServer] = []

    def make(
        *, start: bool = True, name: str = "fake", log: Path | None = None, **values
    ) -> McpServer:
        values = {"command": sys.executable, "args": [FAKE], **values}
        server = McpServer(name, McpServerConfig(**values), log or tmp_path / "mcp.log")
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

        # The error says which server it was.
        assert (
            by_name["fake__die"]
            .call("{}")
            .startswith("Error: McpServerError: MCP server 'fake': ")
        )
        assert (
            by_name["fake__add"]
            .call('{"a": 1, "b": 1}')
            .startswith("Error: McpServerError: MCP server 'fake': ")
        )

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

    @pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="counts open descriptors")
    def test_close_releases_what_the_event_loop_held(self, make_server):
        def open_descriptors() -> int:
            return len(os.listdir("/dev/fd"))

        make_server(name="warmup").close()  # whatever is opened once, like imports
        before = open_descriptors()

        for _ in range(4):
            make_server(name="cycle").close()

        # The servers are still referenced by the fixture, so this is not the
        # garbage collector tidying up: a loop that is not closed keeps three each.
        assert open_descriptors() == before

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
        config = McpServerConfig(command=sys.executable, args=[FAKE])
        server = McpServer("fake", config, tmp_path)  # a directory

        with pytest.raises(McpServerError, match="cannot write its log"):
            server.start()

    def test_a_missing_mcp_package_is_explained(self, tmp_path, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        config = McpServerConfig(command=sys.executable, args=[FAKE])
        server = McpServer("fake", config, tmp_path / "log")

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
        assert servers["plain"].startup_timeout == 20
        assert servers["plain"].call_timeout == 60
        assert servers["full"].args == ["a", "b"]
        assert servers["full"].env == {"K": "v"}
        assert servers["full"].read_only is True

    def test_an_empty_object_has_no_servers(self, tmp_path):
        assert load_servers(self.write(tmp_path, "{}")) == {}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("not json", "Invalid JSON"),
            ('{"mcpServers": {"a": {}}}', "give either command"),
            ('{"mcpServers": {"a": {"command": ""}}}', "mcpServers.a.command"),
            ('{"mcpServers": {"a": {"command": "x", "arg": []}}}', "mcpServers.a.arg"),
            ('{"mcpServers": {"a": {"url": "http://x"}}}', "url must be https"),
            (
                '{"mcpServers": {"a.b": {"command": "x"}}}',
                "String should match pattern",
            ),
            ('{"servers": {}}', "servers"),
            (
                '{"mcpServers": {"a": {"command": "x", "call_timeout": 0}}}',
                "call_timeout",
            ),
            (
                '{"mcpServers": {"a": {"command": "x", "startup_timeout": -1}}}',
                "startup_timeout",
            ),
            (
                '{"mcpServers": {"a": {"command": "x", "call_timeout": "soon"}}}',
                "call_timeout",
            ),
        ],
        ids=[
            "not-json",
            "no-command",
            "empty-command",
            "typo",
            "http",
            "name",
            "key",
            "zero-timeout",
            "negative-timeout",
            "text-timeout",
        ],
    )
    def test_a_mistake_names_where_it_is(self, tmp_path, text, message):
        with pytest.raises(ValidationError, match=message.replace(".", r"\.")):
            load_servers(self.write(tmp_path, text))


class TestUrls:
    def load(self, tmp_path: Path, entry: dict):
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"mcpServers": {"web": entry}}))
        return load_servers(path)["web"]

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/mcp",
            "http://localhost:8000/mcp",
            "http://127.0.0.1:9/mcp",
            "http://[::1]:9/mcp",
        ],
    )
    def test_a_url_may_be_https_or_http_to_this_machine(self, tmp_path, url):
        config = self.load(tmp_path, {"url": url, "headers": {"A": "b"}})

        assert (config.url, config.command) == (url, None)
        assert config.headers == {"A": "b"}

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"url": "http://example.com/mcp"}, "url must be https"),
            ({"url": "http://localhost.example.com/mcp"}, "url must be https"),
            ({"url": "ftp://example.com/mcp"}, "url must be http"),
            ({"url": "example.com/mcp"}, "url must be http"),
            ({"url": "https://"}, "url must be http"),
            ({"url": "https://user:pass@example.com/mcp"}, "user name or password"),
            ({"url": "https://example.com", "command": "x"}, "either command"),
            ({"url": "https://example.com", "args": ["x"]}, "go with command"),
            ({"url": "https://example.com", "env": {"A": "b"}}, "go with command"),
            ({"command": "x", "headers": {"A": "b"}}, "headers go with url"),
            ({}, "either command"),
        ],
    )
    def test_a_server_is_one_thing_and_its_url_must_be_safe(
        self, tmp_path, entry, message
    ):
        with pytest.raises(ValidationError, match=message):
            self.load(tmp_path, entry)


class TestMcpJsonKeys:
    """`type` and `enable`, as other tools' mcp.json files have them."""

    def load(self, tmp_path: Path, entry: dict):
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"mcpServers": {"s": entry}}))
        return load_servers(path)["s"]

    def test_a_server_is_on_by_default_and_can_be_switched_off(self, tmp_path):
        assert self.load(tmp_path, {"command": "x"}).enable is True
        assert self.load(tmp_path, {"command": "x", "enable": False}).enable is False

    @pytest.mark.parametrize(
        "entry",
        [
            {"command": "x", "cwd": "/srv"},
            {"url": "https://example.com", "cwd": "/srv"},
        ],
        ids=["program", "url"],
    )
    def test_there_is_no_working_directory_setting(self, tmp_path, entry):
        # A server starts where muagent does; a file that still has the key is
        # refused, and the message names it.
        with pytest.raises(ValidationError, match=r"mcpServers\.s\.cwd"):
            self.load(tmp_path, entry)

    @pytest.mark.parametrize(
        "entry",
        [
            {"type": "local", "command": "x"},
            {"type": "remote", "url": "https://example.com/mcp"},
        ],
    )
    def test_a_type_that_agrees_is_accepted(self, tmp_path, entry):
        assert self.load(tmp_path, entry).type == entry["type"]

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            (
                {"type": "remote", "command": "x"},
                "type is 'remote', but a server with a command",
            ),
            (
                {"type": "local", "url": "https://example.com"},
                "type is 'local', but a server with a url",
            ),
            ({"type": "stdio", "command": "x"}, "local"),
            ({"command": "x", "enable": "maybe"}, "enable"),
        ],
    )
    def test_a_mistake_is_reported(self, tmp_path, entry, message):
        with pytest.raises(ValidationError, match=message):
            self.load(tmp_path, entry)

    def test_a_file_in_this_dialect_loads(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "docs": {
                            "type": "remote",
                            "url": "https://docs.example.com/mcp",
                            "headers": {"API_KEY": "${DOCS_KEY}"},
                            "enable": True,
                        },
                        "files": {
                            "type": "local",
                            "command": "npx",
                            "args": ["-y", "files-server@1.2.3"],
                            "env": {"TOKEN": "${FILES_TOKEN}"},
                            "enable": False,
                        },
                    }
                }
            )
        )

        servers = load_servers(path)

        assert list(servers) == ["docs", "files"]
        assert servers["files"].env == {"TOKEN": "${FILES_TOKEN}"}  # filled in at start
        assert servers["files"].enable is False


class TestPlaceholders:
    def expand(self, **values):
        config = McpServerConfig(command="x", **values)
        return _expand("s", config, {"A": "1", "B": "two", "NESTED": "${A}"})

    def test_they_are_filled_in_where_a_server_is_given_things(self):
        config = self.expand(
            args=["--n=${A}", "${B}-${A}"],
            env={"K": "v${B}"},
        )

        assert config.args == ["--n=1", "two-1"]
        assert config.env == {"K": "vtwo"}

    def test_headers_too(self):
        config = McpServerConfig(
            url="https://example.com", headers={"H": "Bearer ${B}"}
        )

        assert _expand("s", config, {"B": "tok"}).headers == {"H": "Bearer tok"}

    def test_the_command_and_the_url_are_left_as_they_are(self):
        assert self.expand().command == "x"
        config = McpServerConfig(url="https://example.com/${A}")
        assert _expand("s", config, {"A": "1"}).url == "https://example.com/${A}"

    def test_a_value_is_not_searched_for_placeholders_again(self):
        assert self.expand(env={"K": "${NESTED}"}).env == {"K": "${A}"}

    @pytest.mark.parametrize("text", ["$A", "${A", "${1A}", "${}", "$ {A}", "{A}"])
    def test_anything_else_is_text(self, text):
        assert self.expand(env={"K": text}).env == {"K": text}

    def test_a_variable_that_is_not_set_names_itself_and_where_it_is_used(self):
        with pytest.raises(McpServerError) as exc:
            self.expand(env={"TOKEN": "${MISSING}"})

        assert str(exc.value) == (
            "MCP server 's': env.TOKEN uses ${MISSING}, which is not set"
        )


class TestHttp:
    def make(self, make_server, url, **values) -> McpServer:
        return make_server(name="web", command=None, args=[], url=url, **values)

    def test_the_tools_of_a_server_at_a_url_can_be_called(self, make_server, http_url):
        server = self.make(make_server, http_url)

        by_name = {tool.name: tool for tool in server.tools()}

        assert "web__add" in by_name
        assert by_name["web__add"].call('{"a": 1, "b": 2}') == "3"
        assert not any(tool.read_only for tool in by_name.values())

    def test_headers_are_sent_with_every_request(self, make_server, http_url):
        server = self.make(make_server, http_url, headers={"Authorization": "Bearer t"})
        header = {tool.name: tool for tool in server.tools()}["web__header"]

        assert header.call('{"name": "authorization"}') == "Bearer t"

    def test_no_log_is_made_for_a_server_that_has_no_stderr(
        self, make_server, http_url, tmp_path
    ):
        self.make(make_server, http_url)

        assert not (tmp_path / "mcp.log").exists()

    def test_nothing_listening_is_a_start_failure(self, make_server):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed = probe.getsockname()[1]

        with pytest.raises(McpServerError, match="failed to start: ConnectError"):
            self.make(make_server, f"http://127.0.0.1:{closed}/mcp")

    def test_a_wrong_path_is_a_start_failure(self, make_server, http_url):
        with pytest.raises(McpServerError, match="Not Found"):
            self.make(make_server, http_url.replace("/mcp", "/elsewhere"))

    def test_a_server_that_never_answers_is_given_up_on(self, make_server):
        with socket.socket() as silent:
            silent.bind(("127.0.0.1", 0))
            silent.listen()
            port = silent.getsockname()[1]
            with pytest.raises(McpServerError, match="did not answer within 0.5 s"):
                self.make(
                    make_server, f"http://127.0.0.1:{port}/mcp", startup_timeout=0.5
                )

        assert "mcp-web" not in [thread.name for thread in threading.enumerate()]

    def test_a_server_that_goes_away_gives_an_error_that_names_it(
        self, make_server, caplog
    ):
        with serve_http() as url:
            server = self.make(make_server, url)
            add = {tool.name: tool for tool in server.tools()}["web__add"]
            assert add.call('{"a": 1}') == "2"
        time.sleep(0.5)  # the SDK's background tasks notice

        result = add.call('{"a": 1}')
        server.close()

        assert result.startswith("Error: McpServerError: MCP server 'web'")
        # asyncio logs an error nobody retrieved, with a traceback, when its task is
        # collected, and with no handler that goes to the terminal; the call's error
        # already says what happened.
        gc.collect()
        assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []

    def test_close_stops_the_thread(self, make_server, http_url):
        server = self.make(make_server, http_url)

        server.close()

        assert "mcp-web" not in [thread.name for thread in threading.enumerate()]


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
            assert notices == []  # a server that starts is not announced
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
            assert len(notices) == 1
            assert notices[0].startswith("MCP server 'broken' failed to start")
        finally:
            for server in servers:
                server.close()

    def test_nothing_is_left_running_when_the_start_is_interrupted(self, tmp_path):
        def interrupt(notice: str) -> None:
            raise KeyboardInterrupt

        # The first to be waited for fails and says so, and that is where the
        # interrupt comes; the other two are launched and not yet waited for.
        configs = {
            "broken": McpServerConfig(command="/no/such/server"),
            "first": self.fake(),
            "second": self.fake(),
        }
        with pytest.raises(KeyboardInterrupt):
            start_servers(configs, tmp_path, interrupt)

        names = [t.name for t in threading.enumerate()]
        assert "mcp-first" not in names and "mcp-second" not in names

    def test_the_timeouts_of_a_server_are_used(self, tmp_path):
        servers, found = start_servers(
            {"quick": self.fake(call_timeout=0.3)}, tmp_path, lambda notice: None
        )
        try:
            by_name = {tool.name: tool for tool in found}

            result = by_name["quick__slow"].call('{"seconds": 1}')

            assert result.endswith("did not answer within 0.3 s")
        finally:
            for server in servers:
                server.close()

    def test_a_server_that_is_too_slow_to_start_is_given_up_on(self, tmp_path):
        from muagent.mcp_client import McpServerConfig

        notices: list[str] = []
        silent = McpServerConfig(
            command=sys.executable,
            args=["-c", "import time; time.sleep(60)"],
            startup_timeout=0.5,
        )

        servers, found = start_servers({"silent": silent}, tmp_path, notices.append)

        assert (servers, found) == ([], [])
        assert notices == ["MCP server 'silent' did not answer within 0.5 s"]

    def test_the_servers_start_at_the_same_time(self, tmp_path):
        # Each server waits for the other to have started, and gives up if it does
        # not: only servers that are started together can both come up.
        script = (
            "import pathlib, runpy, sys, time\n"
            "mine, other = map(pathlib.Path, sys.argv[1:3])\n"
            "mine.touch()\n"
            "for _ in range(100):\n"
            "    if other.exists():\n"
            "        break\n"
            "    time.sleep(0.05)\n"
            "else:\n"
            "    sys.exit('the other server never started')\n"
            "sys.argv = ['fake']\n"  # the server takes an argument to mean a port
            f"runpy.run_path({FAKE!r}, run_name='__main__')\n"
        )
        from muagent.mcp_client import McpServerConfig

        a, b = tmp_path / "a", tmp_path / "b"
        configs = {
            "one": McpServerConfig(
                command=sys.executable, args=["-c", script, str(a), str(b)]
            ),
            "two": McpServerConfig(
                command=sys.executable, args=["-c", script, str(b), str(a)]
            ),
        }

        servers, found = start_servers(configs, tmp_path, lambda notice: None)
        try:
            assert [server.name for server in servers] == ["one", "two"]
            assert {tool.name.split("__")[0] for tool in found} == {"one", "two"}
        finally:
            for server in servers:
                server.close()

    def test_close_cancels_a_server_that_is_still_starting(self, make_server):
        server = make_server(
            name="pending", args=["-c", "import time; time.sleep(60)"], start=False
        )
        server.launch()
        started = time.monotonic()

        server.close()

        assert time.monotonic() - started < 6
        assert "mcp-pending" not in [t.name for t in threading.enumerate()]

    def test_a_server_at_a_url_starts_with_its_headers_and_leaves_no_log(
        self, tmp_path, http_url
    ):
        from muagent.mcp_client import McpServerConfig

        configs = {
            "web": McpServerConfig(url=http_url, headers={"X-Test": "yes"}),
            "local": self.fake(),
        }

        servers, found = start_servers(configs, tmp_path, lambda notice: None)
        try:
            by_name = {tool.name: tool for tool in found}
            assert by_name["web__header"].call('{"name": "x-test"}') == "yes"
            assert by_name["local__add"].call('{"a": 1}') == "2"
            assert (tmp_path / "local.log").exists()
            assert not (tmp_path / "web.log").exists()
        finally:
            for server in servers:
                server.close()

    def test_a_server_that_is_switched_off_is_not_started_or_mentioned(self, tmp_path):
        notices: list[str] = []
        configs = {"off": self.fake(enable=False), "on": self.fake()}

        servers, _ = start_servers(configs, tmp_path, notices.append)
        try:
            assert [server.name for server in servers] == ["on"]
            assert notices == []
            assert not (tmp_path / "off.log").exists()
        finally:
            for server in servers:
                server.close()

    def test_placeholders_are_filled_in_from_the_environment(self, tmp_path):
        configs = {"local": self.fake(env={"GIVEN": "${SOURCE}"})}
        environ = {"SOURCE": "from-the-environment"}

        servers, found = start_servers(
            configs, tmp_path, lambda notice: None, environ=environ
        )
        try:
            by_name = {tool.name: tool for tool in found}
            assert by_name["local__getenv"].call('{"name": "GIVEN"}') == (
                "from-the-environment"
            )
            # Only what the file names is passed on, not the variable it came from.
            assert by_name["local__getenv"].call('{"name": "SOURCE"}') == "unset"
        finally:
            for server in servers:
                server.close()

    def test_a_program_starts_in_the_directory_muagent_runs_in(
        self, tmp_path, monkeypatch
    ):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)

        servers, found = start_servers(
            {"local": self.fake()}, tmp_path, lambda notice: None
        )
        try:
            by_name = {tool.name: tool for tool in found}
            result = by_name["local__getcwd"].call("{}")
            assert Path(result).resolve() == project.resolve()
        finally:
            for server in servers:
                server.close()

    def test_headers_are_filled_in_too(self, tmp_path, http_url):
        configs = {
            "web": McpServerConfig(
                url=http_url, headers={"Authorization": "Bearer ${TOKEN}"}
            )
        }

        servers, found = start_servers(
            configs, tmp_path, lambda notice: None, environ={"TOKEN": "s3cret"}
        )
        try:
            header = {tool.name: tool for tool in found}["web__header"]
            assert header.call('{"name": "authorization"}') == "Bearer s3cret"
        finally:
            for server in servers:
                server.close()

    def test_a_server_whose_variable_is_not_set_is_reported_and_skipped(self, tmp_path):
        notices: list[str] = []
        configs = {
            "first": self.fake(),
            "needs": self.fake(env={"K": "${NOT_SET}"}),
            "last": self.fake(),
        }

        servers, _ = start_servers(configs, tmp_path, notices.append, environ={})
        try:
            assert [server.name for server in servers] == ["first", "last"]
            assert notices == [
                "MCP server 'needs': env.K uses ${NOT_SET}, which is not set"
            ]
            assert not (tmp_path / "needs.log").exists()  # it was never launched
        finally:
            for server in servers:
                server.close()

    def test_no_servers_is_no_work(self, tmp_path):
        assert start_servers({}, tmp_path, print) == ([], [])


class TestToolFilter:
    def test_only_the_listed_tools_are_offered(self, make_server):
        server = make_server(tools=["add", "read_file"])

        assert {t.name for t in server.tools()} == {"fake__add", "fake__read_file"}

    def test_without_the_key_every_tool_is_offered(self, make_server):
        assert len(make_server().tools()) > 5

    def test_a_tool_the_server_does_not_have_fails_the_server_and_names_it(
        self, tmp_path
    ):
        notices: list[str] = []
        config = McpServerConfig(
            command=sys.executable, args=[FAKE], tools=["add", "ad"]
        )

        servers, found = start_servers({"s": config}, tmp_path, notices.append)

        assert (servers, found) == ([], [])
        assert len(notices) == 1
        assert "no tool ad " in notices[0] and "add" in notices[0]

    def test_an_empty_list_is_refused(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text('{"mcpServers": {"s": {"command": "x", "tools": []}}}')

        with pytest.raises(ValidationError, match=r"mcpServers\.s\.tools"):
            load_servers(path)


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
