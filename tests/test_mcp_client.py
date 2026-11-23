import asyncio
import gc
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from importlib.metadata import version
from types import SimpleNamespace

import pytest

from ergane.config import LocalMcpServer, RemoteMcpServer
from ergane.mcp_client import McpClient, McpServerError
from ergane.mcp_tools import list_tools
from ergane.tools import MAX_OUTPUT_CHARS, Tool
from fake_api import TEXT
from helpers import FAKE, fake_config, serve_http


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

    def test_the_server_is_told_the_client_is_ergane(self, make_server):
        server = make_server()
        by_name = {tool.name: tool for tool in server.tools()}

        assert by_name["fake__client"].call("{}") == f"ergane {version('ergane')}"

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

    def test_cancel_ends_the_calls_that_are_waiting_and_the_server_stays_usable(
        self, make_server
    ):
        server = make_server()
        by_name = {tool.name: tool for tool in server.tools()}
        results: list[str] = []
        waiting = threading.Thread(
            target=lambda: results.append(by_name["fake__slow"].call('{"seconds": 5}'))
        )
        waiting.start()
        time.sleep(0.5)

        started = time.monotonic()
        assert by_name["fake__slow"].cancel is not None
        by_name["fake__slow"].cancel()  # pyright: ignore[reportOptionalCall]
        waiting.join(5)

        assert time.monotonic() - started < 2
        assert results and results[0].startswith("Error:")
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

    def test_a_server_that_exits_mid_call_gives_an_error_not_a_hang_and_the_next_call_works(
        self, make_server
    ):
        server = make_server()
        by_name = {tool.name: tool for tool in server.tools()}

        # The error says which server it was.
        assert (
            by_name["fake__die"]
            .call("{}")
            .startswith("Error: McpServerError: MCP server 'fake': ")
        )
        # The call that failed made the next one look: the server is started again.
        assert by_name["fake__add"].call('{"a": 1, "b": 1}') == "2"

    def test_a_call_after_close_says_the_server_is_not_running(self, make_server):
        server = make_server()
        add = {tool.name: tool for tool in server.tools()}["fake__add"]

        server.close()

        assert add.call('{"a": 1, "b": 1}') == (
            "Error: McpServerError: MCP server 'fake' is not running"
        )

    def test_a_server_closed_before_it_starts_does_not_start(self, make_server):
        server = make_server(name="early", start=False)
        server.close()

        with pytest.raises(McpServerError, match="closed while it started"):
            server.start()

        assert "mcp-early" not in [t.name for t in threading.enumerate()]

    def test_the_startup_timeout_does_not_end_a_server_that_is_up(self, make_server):
        server = make_server(startup_timeout=1)
        tools = {t.name: t for t in server.tools()}

        time.sleep(1.5)  # past the startup timeout, which only counts until it is up

        assert tools["fake__add"].call('{"a": 1, "b": 2}') == "3"

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
        config = fake_config()
        server = McpClient("fake", config, tmp_path)  # a directory

        with pytest.raises(McpServerError, match="cannot write its log"):
            server.start()

    def test_a_missing_mcp_package_is_explained(self, tmp_path, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        config = fake_config()
        server = McpClient("fake", config, tmp_path / "log")

        with pytest.raises(McpServerError, match="needs the mcp package"):
            server.start()


class TestHttp:
    def make(self, make_server, url, **values) -> McpClient:
        return make_server(name="web", url=url, **values)

    def test_the_tools_of_a_server_at_a_url_can_be_called(self, make_server, http_url):
        server = self.make(make_server, http_url)

        by_name = {tool.name: tool for tool in server.tools()}

        assert "web__add" in by_name
        assert by_name["web__add"].call('{"a": 1, "b": 2}') == "3"
        assert not any(tool.read_only for tool in by_name.values())

    def test_a_call_may_take_longer_than_the_read_timeout_if_call_timeout_allows(
        self, make_server, http_url, monkeypatch
    ):
        # The server says nothing for 1.5 s; the read timeout alone would end the call.
        monkeypatch.setattr("ergane.mcp_client._HTTP_READ_SECONDS", 0.5)
        server = self.make(make_server, http_url, call_timeout=10)
        by_name = {tool.name: tool for tool in server.tools()}

        assert by_name["web__slow"].call('{"seconds": 1.5}') == "done"

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


class TestToolFilter:
    def test_only_the_listed_tools_are_offered(self, make_server):
        server = make_server(tools=["add", "read_file"])

        assert {t.name for t in server.tools()} == {"fake__add", "fake__read_file"}

    def test_without_the_key_every_tool_is_offered(self, make_server):
        assert len(make_server().tools()) > 5

    def test_a_tool_the_server_does_not_have_fails_the_server_and_names_it(
        self, start_hub
    ):
        notices: list[str] = []
        config = LocalMcpServer(
            command=sys.executable, args=[FAKE], tools=["add", "ad"]
        )

        hub, found = start_hub({"s": config}, notices.append)

        assert hub.servers == {} and found == []
        assert len(notices) == 1
        assert "no tool ad " in notices[0] and "add" in notices[0]


class TestListTools:
    def test_every_page_is_read(self):
        pages = {
            None: SimpleNamespace(tools=["a", "b"], next_cursor="two"),
            "two": SimpleNamespace(tools=["c"], next_cursor=None),
        }

        class Session:
            async def list_tools(self, params=None):
                return pages[params.cursor if params else None]

        assert asyncio.run(list_tools(Session())) == ["a", "b", "c"]  # pyright: ignore[reportArgumentType]


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


def block_result(*content, structured=None, is_error=False):
    import mcp_types

    return mcp_types.CallToolResult(
        content=list(content), structured_content=structured, is_error=is_error
    )


class TestPictures:
    """What the model is given of a picture a tool returns: only if the config says so."""

    def test_a_server_whose_config_says_so_gives_the_picture(self, make_server):
        server = make_server(images=True)
        tool = {t.name: t for t in server.tools()}["fake__picture"]

        result = tool.call_output("{}")

        assert result.text == "[image: image/png]"
        assert [(i.mime_type, len(i.data) > 0) for i in result.images] == [
            ("image/png", True)
        ]

    def test_otherwise_it_is_a_note_that_there_was_one(self, make_server):
        server = make_server()
        tool = {t.name: t for t in server.tools()}["fake__picture"]

        assert tool.call_output("{}").images == ()

    def test_only_the_kinds_of_picture_the_api_takes(self):
        import mcp_types

        from ergane.mcp_tools import result_images

        def block(mime: str):
            return mcp_types.ImageContent(type="image", data="aGk=", mime_type=mime)

        result = block_result(block("image/png"), block("image/svg+xml"))

        assert [i.mime_type for i in result_images(result)] == ["image/png"]

    def test_a_picture_too_large_is_left_out(self, monkeypatch):
        import mcp_types

        from ergane.mcp_tools import result_images

        monkeypatch.setattr("ergane.mcp_tools._MAX_IMAGE_CHARS", 4)
        big = mcp_types.ImageContent(type="image", data="x" * 5, mime_type="image/png")
        small = mcp_types.ImageContent(
            type="image", data="x" * 4, mime_type="image/png"
        )

        assert [len(i.data) for i in result_images(block_result(big, small))] == [4]


class TestResultText:
    def test_a_text_file_the_server_embedded_is_text(self):
        import mcp_types

        from ergane.mcp_tools import result_text

        file = mcp_types.EmbeddedResource(
            type="resource",
            resource=mcp_types.TextResourceContents(
                uri="file:///a.txt", text="contents", mime_type="text/plain"
            ),
        )

        assert result_text(block_result(file)) == "contents"

    def test_a_binary_one_and_a_link_are_named(self):
        import mcp_types

        from ergane.mcp_tools import result_text

        blob = mcp_types.EmbeddedResource(
            type="resource",
            resource=mcp_types.BlobResourceContents(uri="file:///a.bin", blob="AAAA"),
        )
        link = mcp_types.ResourceLink(type="resource_link", uri="file:///b", name="b")

        assert result_text(block_result(blob, link)) == (
            "[resource: file:///a.bin]\n[resource_link: file:///b]"
        )

    def test_a_result_with_only_structured_content_is_that_as_json(self):
        from ergane.mcp_tools import result_text

        assert result_text(block_result(structured={"sum": 3})) == '{"sum": 3}'

    def test_structured_content_next_to_text_is_not_repeated(self):
        import mcp_types

        from ergane.mcp_tools import result_text

        text = mcp_types.TextContent(type="text", text="3")

        assert result_text(block_result(text, structured={"sum": 3})) == "3"

    def test_an_error_is_marked(self):
        import mcp_types

        from ergane.mcp_tools import result_text

        text = mcp_types.TextContent(type="text", text="bad")

        assert result_text(block_result(text, is_error=True)) == "Error: bad"


class TestWhatSaysTheConnectionEnded:
    def test_the_sdks_closed_answer_and_closed_streams_do(self):
        import anyio
        from mcp import MCPError
        from mcp_types import CONNECTION_CLOSED

        from ergane.mcp_client import _connection_lost

        assert _connection_lost(MCPError(CONNECTION_CLOSED, "Connection closed"))
        assert _connection_lost(anyio.ClosedResourceError())
        assert _connection_lost(anyio.BrokenResourceError())

    def test_an_error_a_live_server_answers_with_does_not(self):
        from mcp import MCPError
        from mcp_types import INTERNAL_ERROR, METHOD_NOT_FOUND

        from ergane.mcp_client import _connection_lost

        assert not _connection_lost(MCPError(METHOD_NOT_FOUND, "Method not found"))
        assert not _connection_lost(MCPError(INTERNAL_ERROR, "boom"))
        assert not _connection_lost(ValueError("bad"))


class TestConnectionLost:
    """A server whose connection ended is started again by the next call."""

    @pytest.fixture(autouse=True)
    def always_ask_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Not after 10 s of silence, as in use: no test waits that long.
        monkeypatch.setattr("ergane.mcp_client._IDLE_SECONDS", 0)

    def kill(self, server: McpClient, tool: Tool) -> None:
        import signal

        os.kill(int(tool.call("{}")), signal.SIGKILL)
        time.sleep(0.3)

    def test_a_server_that_died_while_idle_works_again_at_the_next_call(
        self, make_server
    ):
        server = make_server()
        tools = {t.name: t for t in server.tools()}
        self.kill(server, tools["fake__pid"])

        assert tools["fake__add"].call('{"a": 1, "b": 2}') == "3"  # no error at all

    def test_a_live_server_that_answers_the_ping_with_an_error_is_not_restarted(
        self, make_server, monkeypatch
    ):
        from mcp import ClientSession, MCPError
        from mcp_types import METHOD_NOT_FOUND

        async def refused(self, *args, **kwargs):
            raise MCPError(METHOD_NOT_FOUND, "Method not found")

        server = make_server(name="alive")
        tools = {t.name: t for t in server.tools()}
        pid = tools["alive__pid"].call("{}")
        monkeypatch.setattr(ClientSession, "send_ping", refused)

        # The connection is fine: the server answers, and its answer is an error.
        assert tools["alive__add"].call('{"a": 1, "b": 2}') == "3"

        assert server._restarts == 0  # pyright: ignore[reportPrivateUsage]
        assert tools["alive__pid"].call("{}") == pid  # the same program

    def test_a_call_that_failed_makes_the_next_one_check_the_connection(
        self, make_server, monkeypatch
    ):
        # No ping for idleness: only the failed call can make the next one look.
        monkeypatch.setattr("ergane.mcp_client._IDLE_SECONDS", 3600)
        server = make_server(name="checked")
        tools = {t.name: t for t in server.tools()}
        self.kill(server, tools["checked__pid"])

        assert tools["checked__add"].call('{"a": 1, "b": 2}').startswith("Error:")
        assert tools["checked__add"].call('{"a": 1, "b": 2}') == "3"

    def test_the_old_connection_is_gone_when_a_server_is_started_again(
        self, make_server
    ):
        server = make_server(name="again")
        tools = {t.name: t for t in server.tools()}
        self.kill(server, tools["again__pid"])

        assert tools["again__add"].call('{"a": 1, "b": 2}') == "3"

        assert [t.name for t in threading.enumerate()].count("mcp-again") == 1

    def test_calls_on_several_threads_at_once_start_a_dead_server_once(
        self, make_server
    ):
        server = make_server()
        tools = {t.name: t for t in server.tools()}
        self.kill(server, tools["fake__pid"])
        results: list[str] = []

        def call() -> None:
            results.append(tools["fake__add"].call('{"a": 1, "b": 2}'))

        threads = [threading.Thread(target=call) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)

        assert results == ["3"] * 6
        assert server._restarts == 1  # pyright: ignore[reportPrivateUsage]

    def test_the_call_that_found_it_dead_fails_and_the_next_works(self, make_server):
        server = make_server()
        tools = {t.name: t for t in server.tools()}

        assert tools["fake__die"].call("{}").startswith("Error:")

        # The call that failed is not repeated: it may have done part of its work.
        assert tools["fake__add"].call('{"a": 1, "b": 2}') == "3"

    def test_it_is_a_new_process_and_the_log_keeps_what_the_old_one_wrote(
        self, make_server, tmp_path
    ):
        server = make_server()
        tools = {t.name: t for t in server.tools()}
        tools["fake__chatty"].call("{}")
        before = tools["fake__pid"].call("{}")
        self.kill(server, tools["fake__pid"])

        after = tools["fake__pid"].call("{}")

        assert after != before
        assert "fake server log line" in (tmp_path / "mcp.log").read_text()

    def test_a_server_that_keeps_dying_is_not_started_for_ever(self, make_server):
        server = make_server()
        tools = {t.name: t for t in server.tools()}
        outputs = []
        for _ in range(8):
            outputs.append(tools["fake__die"].call("{}"))

        assert any("is not started any more" in output for output in outputs)

    def test_a_restart_that_fails_says_so_and_the_server_is_tried_again_later(
        self, make_server, tmp_path
    ):
        # Starts the fake server the first time and exits at once afterwards.
        script = tmp_path / "once.py"
        script.write_text(
            "import os, sys\n"
            "marker = sys.argv[1]\n"
            "if os.path.exists(marker):\n"
            "    sys.exit(1)\n"
            "open(marker, 'w').close()\n"
            f"os.execv(sys.executable, [sys.executable, {FAKE!r}])\n"
        )
        server = make_server(args=[str(script), str(tmp_path / "marker")])
        tools = {t.name: t for t in server.tools()}
        self.kill(server, tools["fake__pid"])

        output = tools["fake__add"].call('{"a": 1, "b": 2}')

        assert output.startswith("Error: McpServerError:")
        assert "restarting it after it stopped" in output

    def test_a_server_that_was_closed_is_not_started_again(self, make_server):
        server = make_server()
        tools = {t.name: t for t in server.tools()}
        server.close()

        assert "is not running" in tools["fake__add"].call('{"a": 1, "b": 2}')
        assert server._portal is None  # pyright: ignore[reportPrivateUsage]

    def test_a_remote_server_that_went_away_and_came_back_works_again(self, tmp_path):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        def run() -> subprocess.Popen:
            process = subprocess.Popen(
                [sys.executable, FAKE, str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(200):
                try:
                    socket.create_connection(("127.0.0.1", port), 0.1).close()
                    return process
                except OSError:
                    time.sleep(0.05)
            raise RuntimeError("the fake HTTP server did not start")

        process = run()
        server = McpClient(
            "web", RemoteMcpServer(url=f"http://127.0.0.1:{port}/mcp"), tmp_path / "l"
        )
        try:
            server.start()
            tools = {t.name: t for t in server.tools()}
            process.terminate()
            process.wait(10)
            assert tools["web__add"].call('{"a": 1, "b": 2}').startswith("Error:")
            process = run()

            assert tools["web__add"].call('{"a": 1, "b": 2}') == "3"
        finally:
            server.close()
            process.terminate()
            process.wait(10)


class TestInstructions:
    """What a server says about using its tools."""

    def test_the_instructions_of_the_server_are_kept(self, make_server):
        assert make_server().instructions == "Dates are YYYY-MM-DD."

    def test_long_instructions_are_cut(self, make_server, monkeypatch):
        monkeypatch.setattr("ergane.mcp_client._MAX_INSTRUCTION_CHARS", 5)

        assert make_server().instructions == "Dates…"
