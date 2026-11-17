import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from helpers import FAKE, fake_config, hanging_config
from muagent.config import LocalMcpServer, McpServerConfig, RemoteMcpServer
from muagent.mcp_client import McpClient, McpServerError
from muagent.mcp_hub import McpHub, _expand, close_servers
from muagent.tools import Tool


class TestPlaceholders:
    def expand(self, **values) -> LocalMcpServer:
        config = LocalMcpServer(command="x", **values)
        expanded = _expand("s", config, {"A": "1", "B": "two", "NESTED": "${A}"})
        assert isinstance(expanded, LocalMcpServer)
        return expanded

    @staticmethod
    def expand_remote(config: RemoteMcpServer, environ) -> RemoteMcpServer:
        expanded = _expand("s", config, environ)
        assert isinstance(expanded, RemoteMcpServer)
        return expanded

    def test_they_are_filled_in_where_a_server_is_given_things(self):
        config = self.expand(
            args=["--n=${A}", "${B}-${A}"],
            env={"K": "v${B}"},
        )

        assert config.args == ["--n=1", "two-1"]
        assert config.env == {"K": "vtwo"}

    def test_headers_too(self):
        config = RemoteMcpServer(
            url="https://example.com", headers={"H": "Bearer ${B}"}
        )

        assert self.expand_remote(config, {"B": "tok"}).headers == {"H": "Bearer tok"}

    def test_the_command_and_the_url_are_left_as_they_are(self):
        assert self.expand().command == "x"
        config = RemoteMcpServer(url="https://example.com/${A}")
        expanded = self.expand_remote(config, {"A": "1"})
        assert expanded.url == "https://example.com/${A}"

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


class TestStart:
    def test_every_server_starts_and_its_tools_are_collected(self, start_hub, tmp_path):
        notices: list[str] = []
        configs = {"one": fake_config(), "two": fake_config(read_only=True)}

        _, found = start_hub(configs, notices.append)

        names = {tool.name: tool for tool in found}
        assert "one__add" in names and "two__add" in names
        assert not names["one__add"].read_only
        assert names["two__add"].read_only
        assert notices == []  # a server that starts is not announced
        assert (tmp_path / "one.log").exists()

    def test_a_server_that_fails_is_reported_and_the_others_still_start(
        self, start_hub
    ):
        notices: list[str] = []
        configs = {
            "broken": LocalMcpServer(command="/no/such/server"),
            "good": fake_config(),
        }

        hub, found = start_hub(configs, notices.append)

        assert list(hub.servers) == ["good"]
        assert all(tool.name.startswith("good__") for tool in found)
        assert len(notices) == 1
        assert notices[0].startswith("MCP server 'broken' failed to start")

    def test_nothing_is_left_running_when_the_start_is_interrupted(self, start_hub):
        def interrupt(notice: str) -> None:
            raise KeyboardInterrupt

        # The interrupt comes with the notice of the server that failed; the other
        # two run by then, and nobody is left to take them.
        configs = {
            "broken": LocalMcpServer(command="/no/such/server"),
            "first": fake_config(),
            "second": fake_config(),
        }
        with pytest.raises(KeyboardInterrupt):
            start_hub(configs, interrupt)

        names = [t.name for t in threading.enumerate()]
        assert "mcp-first" not in names and "mcp-second" not in names

    def test_the_timeouts_of_a_server_are_used(self, start_hub):
        _, found = start_hub({"quick": fake_config(call_timeout=0.3)})
        by_name = {tool.name: tool for tool in found}

        result = by_name["quick__slow"].call('{"seconds": 1}')

        assert result.endswith("did not answer within 0.3 s")

    def test_a_server_that_is_too_slow_to_start_is_given_up_on(self, start_hub):
        notices: list[str] = []
        silent = LocalMcpServer(
            command=sys.executable,
            args=["-c", "import time; time.sleep(60)"],
            startup_timeout=0.5,
        )

        hub, found = start_hub({"silent": silent}, notices.append)

        assert hub.servers == {} and found == []
        assert notices == ["MCP server 'silent' did not answer within 0.5 s"]

    def test_the_servers_start_at_the_same_time(self, start_hub, tmp_path):
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
        a, b = tmp_path / "a", tmp_path / "b"
        configs = {
            "one": LocalMcpServer(
                command=sys.executable, args=["-c", script, str(a), str(b)]
            ),
            "two": LocalMcpServer(
                command=sys.executable, args=["-c", script, str(b), str(a)]
            ),
        }

        hub, found = start_hub(configs)

        assert list(hub.servers) == ["one", "two"]
        assert {tool.name.split("__")[0] for tool in found} == {"one", "two"}

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
        self, start_hub, tmp_path, http_url
    ):
        configs = {
            "web": RemoteMcpServer(url=http_url, headers={"X-Test": "yes"}),
            "local": fake_config(),
        }

        _, found = start_hub(configs)

        by_name = {tool.name: tool for tool in found}
        assert by_name["web__header"].call('{"name": "x-test"}') == "yes"
        assert by_name["local__add"].call('{"a": 1}') == "2"
        assert (tmp_path / "local.log").exists()
        assert not (tmp_path / "web.log").exists()

    def test_a_server_that_is_switched_off_is_not_started_or_mentioned(
        self, start_hub, tmp_path
    ):
        notices: list[str] = []
        configs = {"off": fake_config(enabled=False), "on": fake_config()}

        hub, _ = start_hub(configs, notices.append)

        assert list(hub.servers) == ["on"]
        assert notices == []
        assert not (tmp_path / "off.log").exists()

    def test_placeholders_are_filled_in_from_the_environment(self, start_hub):
        configs = {"local": fake_config(env={"GIVEN": "${SOURCE}"})}

        _, found = start_hub(configs, environ={"SOURCE": "from-the-environment"})

        by_name = {tool.name: tool for tool in found}
        assert by_name["local__getenv"].call('{"name": "GIVEN"}') == (
            "from-the-environment"
        )
        # Only what the file names is passed on, not the variable it came from.
        assert by_name["local__getenv"].call('{"name": "SOURCE"}') == "unset"

    def test_a_program_starts_in_the_directory_muagent_runs_in(
        self, start_hub, tmp_path, monkeypatch
    ):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)

        _, found = start_hub({"local": fake_config()})

        by_name = {tool.name: tool for tool in found}
        result = by_name["local__getcwd"].call("{}")
        assert Path(result).resolve() == project.resolve()

    def test_headers_are_filled_in_too(self, start_hub, http_url):
        configs = {
            "web": RemoteMcpServer(
                url=http_url, headers={"Authorization": "Bearer ${TOKEN}"}
            )
        }

        _, found = start_hub(configs, environ={"TOKEN": "s3cret"})

        header = {tool.name: tool for tool in found}["web__header"]
        assert header.call('{"name": "authorization"}') == "Bearer s3cret"

    def test_a_server_whose_variable_is_not_set_is_reported_and_skipped(
        self, start_hub, tmp_path
    ):
        notices: list[str] = []
        configs = {
            "first": fake_config(),
            "needs": fake_config(env={"K": "${NOT_SET}"}),
            "last": fake_config(),
        }

        hub, _ = start_hub(configs, notices.append, environ={})

        assert list(hub.servers) == ["first", "last"]
        assert notices == [
            "MCP server 'needs': env.K uses ${NOT_SET}, which is not set"
        ]
        assert not (tmp_path / "needs.log").exists()  # it was never launched

    def test_no_servers_is_no_work(self, start_hub):
        hub, found = start_hub({})

        assert hub.servers == {} and found == []


class TestStatuses:
    """What became of each server, for /servers: the notices are shown once."""

    def test_every_server_has_a_status_in_the_order_of_the_file(
        self, start_hub, tmp_path
    ):
        configs = {
            "good": fake_config(tools=["add", "read_file"]),
            "off": fake_config(enabled=False),
            "bad": LocalMcpServer(command="/no/such/server"),
            "unset": fake_config(env={"K": "${NOT_SET_ANYWHERE}"}),
        }
        notices: list[str] = []

        hub, found = start_hub(configs, notices.append, environ={})

        by = {status.name: status for status in hub.statuses}
        assert list(by) == list(configs)
        assert by["good"].state == "running"
        assert by["good"].tools == ("good__add", "good__read_file")
        assert by["good"].hidden == 16 and by["good"].log == tmp_path / "good.log"
        assert [t.name for t in found] == ["good__add", "good__read_file"]
        assert by["off"].state == "off" and by["off"].tools == ()
        assert by["bad"].state == "failed" and by["unset"].state == "failed"
        assert "FileNotFoundError" in by["bad"].detail
        assert "NOT_SET_ANYWHERE" in by["unset"].detail
        # The detail is the notice that was shown, to be shown again.
        assert notices == [by["bad"].detail, by["unset"].detail]

    def test_a_url_server_has_no_log(self, start_hub, http_url):
        hub, _ = start_hub({"web": RemoteMcpServer(url=http_url, tools=["add"])})

        assert hub.statuses[0].state == "running" and hub.statuses[0].log is None


class TestStartingTogether:
    def test_servers_that_never_answer_delay_the_start_by_one_timeout_not_the_sum(
        self, start_hub
    ):
        notices: list[str] = []
        started = time.monotonic()

        hub, found = start_hub(
            {f"hang{i}": hanging_config(startup_timeout=1) for i in range(4)},
            notices.append,
        )

        elapsed = time.monotonic() - started
        assert hub.servers == {} and found == [] and len(notices) == 4
        # One timeout (1 s) and one stop (the SDK waits 2 s for a server that does not
        # leave), together: it was 4 x 3 s when each was waited for in turn.
        assert elapsed < 6

    def test_a_server_that_fails_to_connect_is_left_for_close(self, tmp_path):
        server = McpClient(
            "hang", hanging_config(startup_timeout=1), tmp_path / "hang.log"
        )
        server.launch()
        try:
            with pytest.raises(McpServerError, match="did not answer within 1 s"):
                server.connect()

            assert "mcp-hang" in [t.name for t in threading.enumerate()]
        finally:
            server.close()
        assert "mcp-hang" not in [t.name for t in threading.enumerate()]

    def test_servers_are_closed_together(self, tmp_path):
        servers = [
            McpClient(
                f"slow{i}", hanging_config(startup_timeout=30), tmp_path / f"{i}.log"
            )
            for i in range(3)
        ]
        for server in servers:
            server.launch()
        time.sleep(0.5)  # let each start its process
        started = time.monotonic()

        close_servers(servers)

        # Each takes about 2 s to stop (the process ignores its closed input).
        assert time.monotonic() - started < 4.5
        assert not any(t.name.startswith("mcp-slow") for t in threading.enumerate())


class TestNamesAcrossServers:
    def test_two_servers_cannot_give_the_model_the_same_tool_name(self, start_hub):
        configs = {
            "outer": fake_config(tools=["inner__tool"]),
            "outer__inner": fake_config(tools=["tool"]),
        }

        _, found = start_hub(configs)

        names = [tool.name for tool in found]
        assert len(names) == len(set(names)) == 2
        assert "outer__inner__tool" in names


class TestHub:
    @pytest.fixture
    def hub(self, tmp_path: Path) -> Iterator[McpHub]:
        hub = McpHub({"fake": self.fake()}, tmp_path)
        yield hub
        hub.close()

    @staticmethod
    def fake(**values) -> McpServerConfig:
        values = {"instructions": True, **values}  # off by default, as in the config
        return fake_config(**values)

    @staticmethod
    def names(tools: list[Tool]) -> list[str]:
        return [tool.name for tool in tools]

    def test_it_starts_the_servers_and_knows_what_they_say(self, hub):
        tools = hub.start(lambda notice: None)

        assert "fake__add" in self.names(tools)
        assert [s.state for s in hub.statuses] == ["running"]
        assert hub.instructions() == {"fake": "Dates are YYYY-MM-DD."}

    def test_instructions_are_kept_from_the_model_unless_asked_for(self, tmp_path):
        hub = McpHub({"fake": self.fake(instructions=False)}, tmp_path)
        try:
            hub.start(lambda notice: None)

            assert hub.instructions() == {}
        finally:
            hub.close()

    def test_a_server_that_runs_is_started_again(self, hub):
        tools = hub.start(lambda notice: None)
        by_name = {tool.name: tool for tool in tools}
        first = by_name["fake__pid"].call("{}")

        update = hub.reconnect("fake")

        again = {tool.name: tool for tool in update.tools}
        assert again["fake__pid"].call("{}") != first
        assert set(update.removed) == set(by_name)
        assert update.message == f"MCP server 'fake' is running: {len(tools)} tools"
        assert update.instructions == "Dates are YYYY-MM-DD."

    def test_reconnecting_keeps_what_the_log_said(self, hub, tmp_path):
        tools = hub.start(lambda notice: None)
        {tool.name: tool for tool in tools}["fake__chatty"].call("{}")
        log = tmp_path / "fake.log"
        before = log.read_text()
        assert "fake server log line" in before

        hub.reconnect("fake")

        assert log.read_text().startswith(before)

    def test_a_server_that_failed_can_start_when_the_cause_is_gone(self, tmp_path):
        configs: dict[str, McpServerConfig] = {
            "fake": LocalMcpServer(command="/no/such/server")
        }
        hub = McpHub(configs, tmp_path)
        try:
            assert hub.start(lambda notice: None) == []
            assert hub.statuses[0].state == "failed"
            configs["fake"] = self.fake()  # the program was installed

            update = hub.reconnect("fake")

            assert "fake__add" in self.names(update.tools)
            assert hub.statuses[0].state == "running"
            assert hub.statuses[0].detail == ""
        finally:
            hub.close()

    def test_a_server_that_fails_again_loses_its_tools_and_says_why(self, hub):
        tools = hub.start(lambda notice: None)
        hub.configs = {"fake": LocalMcpServer(command="/no/such/server")}

        update = hub.reconnect("fake")

        assert update.tools == [] and update.instructions is None
        assert set(update.removed) == {tool.name for tool in tools}
        assert "FileNotFoundError" in update.message
        assert hub.statuses[0].state == "failed"
        assert hub.servers == {}

    def test_a_server_that_is_off_stays_off(self, tmp_path):
        hub = McpHub({"fake": self.fake(enabled=False)}, tmp_path)
        try:
            hub.start(lambda notice: None)

            update = hub.reconnect("fake")

            assert update.message == "MCP server 'fake' is off"
            assert hub.statuses[0].state == "off" and hub.servers == {}
        finally:
            hub.close()


class TestStartingInTheBackground:
    @staticmethod
    def fake(**values) -> McpServerConfig:
        values = {"instructions": True, **values}
        return fake_config(**values)

    @pytest.fixture
    def hubs(self, tmp_path: Path) -> Iterator[Callable[..., McpHub]]:
        made: list[McpHub] = []

        def make(configs) -> McpHub:
            hub = McpHub(configs, tmp_path)
            made.append(hub)
            return hub

        yield make
        for hub in made:
            hub.close()

    def test_begin_does_not_wait_and_the_servers_are_starting(self, hubs):
        hub = hubs({"slow": hanging_config(), "off": self.fake(enabled=False)})

        started = time.monotonic()
        hub.begin()

        assert time.monotonic() - started < 1
        assert hub.starting
        assert [(s.name, s.state) for s in hub.statuses] == [
            ("slow", "starting"),
            ("off", "off"),
        ]
        assert hub.take(lambda notice: None) == []  # nothing to take yet

    def test_take_waits_when_asked_and_gives_each_server_once(self, hubs):
        hub = hubs({"fake": self.fake()})
        hub.begin()

        [update] = hub.take(lambda notice: None, wait=True)

        assert update.name == "fake" and update.removed == ()
        assert "fake__add" in [tool.name for tool in update.tools]
        assert update.instructions == "Dates are YYYY-MM-DD."
        assert update.message == ""
        assert [s.state for s in hub.statuses] == ["running"]
        assert hub.take(lambda notice: None, wait=True) == []

    def test_poll_finishes_a_start_every_server_has_answered(self, hubs):
        hub = hubs({"fake": self.fake()})
        hub.begin()
        deadline = time.monotonic() + 10
        while hub.starting and time.monotonic() < deadline:
            time.sleep(0.05)

        hub.poll()

        assert [s.state for s in hub.statuses] == ["running"]

    def test_a_server_that_fails_is_told_when_taken_not_before(self, hubs):
        hub = hubs({"bad": LocalMcpServer(command="/no/such/server")})
        notices: list[str] = []
        hub.begin()

        hub.poll()
        assert notices == []

        assert hub.take(notices.append, wait=True) == []
        assert len(notices) == 1 and "FileNotFoundError" in notices[0]
        assert [s.state for s in hub.statuses] == ["failed"]
        hub.take(notices.append)
        assert len(notices) == 1  # once

    def test_a_server_that_never_answers_ends_as_failed(self, hubs):
        hub = hubs({"slow": hanging_config(startup_timeout=0.5)})
        hub.begin()

        hub.take(lambda notice: None, wait=True)

        assert "did not answer within 0.5 s" in hub.statuses[0].detail

    def test_ctrl_c_while_waiting_leaves_the_start_to_be_waited_for_again(
        self, hubs, monkeypatch
    ):
        hub = hubs({"fake": self.fake()})
        hub.begin()
        startup = hub._startup  # pyright: ignore[reportPrivateUsage]
        assert startup is not None
        real = startup.finish
        calls = []

        def interrupted_once(on_notice):
            calls.append(1)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return real(on_notice)

        monkeypatch.setattr(startup, "finish", interrupted_once)

        with pytest.raises(KeyboardInterrupt):
            hub.take(lambda notice: None, wait=True)
        [update] = hub.take(lambda notice: None, wait=True)

        assert update.name == "fake"

    def test_closing_while_servers_start_stops_them(self, hubs):
        hub = hubs({"slow": hanging_config()})
        hub.begin()

        hub.close()

        assert not hub.starting and hub.servers == {}

    def test_the_servers_list_their_tools_at_the_same_time(self, hubs, monkeypatch):
        spans: list[tuple[float, float]] = []
        real = McpClient.definitions

        def slow(self):
            begun = time.monotonic()
            time.sleep(0.4)
            found = real(self)
            spans.append((begun, time.monotonic()))
            return found

        monkeypatch.setattr(McpClient, "definitions", slow)
        hub = hubs({name: self.fake() for name in ("a", "b", "c")})

        hub.start(lambda notice: None)

        # Each started before any had finished: one after another would not do that.
        assert len(spans) == 3
        assert max(begun for begun, _ in spans) < min(end for _, end in spans)
