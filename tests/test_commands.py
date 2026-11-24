import sys
import time
from pathlib import Path

import pytest

from ergane.commands import COMMANDS, Command, Warn, command
from ergane.config import LocalMcpServer
from ergane.context import as_dicts, estimate_tokens
from ergane.mcp_hub import McpHub, ServerStatus
from ergane.repl import Repl
from ergane.sessions import SessionInfo, SessionStore, ago
from ergane.tools import Tool
from fake_api import TEXT
from helpers import anchor, fake_config, hanging_config, make_context, run_repl


class TestCommandsWithAgent:
    """/retry, /undo, /model and /effort against a real Agent and the fake server."""

    @staticmethod
    def inputs(*lines: str) -> list:
        return [*lines, "/exit"]

    def test_retry_replaces_a_saved_exchange(
        self, monkeypatch, make_agent, fake_server
    ):
        agent = make_agent()

        run_repl(monkeypatch, agent, self.inputs("hi", "/retry"))

        assert len(fake_server.requests) == 2
        assert fake_server.requests[1]["input"] == [{"role": "user", "content": "hi"}]
        assert agent.context.items == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": TEXT},
        ]

    def test_retry_replaces_a_partial_reply_that_was_kept(
        self, monkeypatch, make_agent, fake_server
    ):
        # An incomplete reply with text is saved, so /retry must replace it.
        agent = make_agent()

        run_repl(monkeypatch, agent, self.inputs("incomplete", "/retry"))

        assert fake_server.requests[1]["input"] == [
            {"role": "user", "content": "incomplete"}
        ]
        assert agent.context.items == [
            {"role": "user", "content": "incomplete"},
            {"role": "assistant", "content": TEXT},
        ]

    def test_retry_resends_a_failed_turn(self, monkeypatch, make_agent, fake_server):
        agent = make_agent()

        run_repl(monkeypatch, agent, self.inputs("failed", "/retry"))

        assert [r["input"] for r in fake_server.requests] == [
            [{"role": "user", "content": "failed"}],
            [{"role": "user", "content": "failed"}],
        ]
        assert agent.context.items == []

    def test_retry_with_nothing_to_retry(
        self, monkeypatch, make_agent, fake_server, capsys
    ):
        run_repl(monkeypatch, make_agent(), self.inputs("/retry"))

        assert "[Nothing to retry]" in capsys.readouterr().out
        assert fake_server.requests == []

    def test_undo_removes_the_last_exchange(self, monkeypatch, make_agent, capsys):
        agent = make_agent()

        run_repl(monkeypatch, agent, self.inputs("first", "second", "/undo"))

        assert agent.context.items == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": TEXT},
        ]
        assert "[Last exchange removed]" in capsys.readouterr().out

    def test_undo_with_nothing_to_undo(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), self.inputs("/undo"))

        assert "[Nothing to undo]" in capsys.readouterr().out

    def test_retry_after_undo_does_not_remove_an_older_exchange(
        self, monkeypatch, make_agent, fake_server
    ):
        agent = make_agent()

        run_repl(monkeypatch, agent, self.inputs("first", "second", "/undo", "/retry"))

        assert fake_server.requests[-1]["input"] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": TEXT},
            {"role": "user", "content": "second"},
        ]
        assert len(agent.context.items) == 4

    def test_switching_the_model_leaves_the_servers_count_behind(
        self, monkeypatch, make_agent
    ):
        agent = make_agent()
        anchor(agent, 1000)

        run_repl(monkeypatch, agent, ["/model", "/exit"])
        assert agent.counted_tokens() == 1000  # only showing it changes nothing

        run_repl(monkeypatch, agent, ["/model Other-Model", "/exit"])
        assert agent.counted_tokens() is None
        run_repl(monkeypatch, agent, ["/model gpt-4o", "/exit"])
        assert agent.counted_tokens() == 1000  # the model that counted is back

    def test_model_shows_and_switches_the_model(
        self, monkeypatch, make_agent, fake_server, capsys
    ):
        # The command name is case-insensitive; the model name keeps its case.
        lines = self.inputs("/model", "/MODEL Other-Model", "hi")

        run_repl(monkeypatch, make_agent(), lines)

        out = capsys.readouterr().out
        assert "[Model: gpt-4o]" in out
        assert "[Model: Other-Model]" in out
        assert fake_server.requests[-1]["model"] == "Other-Model"

    def test_effort_sets_and_clears_the_reasoning_effort(
        self, monkeypatch, make_agent, fake_server, capsys
    ):
        lines = self.inputs("/effort", "/effort LOW", "one", "/effort default", "two")

        run_repl(monkeypatch, make_agent(), lines)

        out = capsys.readouterr().out
        assert "[Reasoning effort: model default]" in out
        assert "[Reasoning effort: low]" in out
        first, second = fake_server.requests
        assert first["reasoning"] == {"effort": "low"}
        assert "reasoning" not in second

    def test_effort_rejects_unknown_levels(self, monkeypatch, make_agent, capsys):
        agent = make_agent(REASONING_EFFORT="high")

        run_repl(monkeypatch, agent, self.inputs("/effort extreme"))

        assert "[Unknown effort: extreme. Use none, minimal, low" in (
            capsys.readouterr().out
        )
        assert agent.config.reasoning_effort == "high"

    def test_reset_clears_the_history(
        self, monkeypatch, make_agent, fake_server, capsys
    ):
        agent = make_agent()

        run_repl(monkeypatch, agent, self.inputs("a", "b", " /Reset ", "c"))

        # "c" was sent with an empty history, so only "c" and its reply remain.
        assert fake_server.requests[-1]["input"] == [{"role": "user", "content": "c"}]
        assert agent.context.items == [
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": TEXT},
        ]
        assert "[History reset]" in capsys.readouterr().out

    @pytest.mark.parametrize("is_terminal", [True, False])
    def test_clear_clears_the_screen_only_on_a_terminal_and_keeps_the_history(
        self, monkeypatch, make_agent, capsys, is_terminal
    ):
        # capsys replaces stdout with a capture object, which is never a terminal.
        monkeypatch.setattr(sys.stdout, "isatty", lambda: is_terminal)
        agent = make_agent()

        run_repl(monkeypatch, agent, self.inputs("a", "/clear", "b"))

        assert len(agent.context.items) == 4  # "b" was sent after "a" and its reply
        printed_codes = "\033[2J\033[3J\033[H" in capsys.readouterr().out
        assert printed_codes is is_terminal

    def test_help_without_a_line_editor_lists_only_the_keys_that_work(
        self, monkeypatch, make_agent, capsys
    ):
        run_repl(monkeypatch, make_agent(), self.inputs("/help"))

        out = capsys.readouterr().out
        assert "Alt-Enter" not in out and "Ctrl-C" not in out  # no session here
        assert 'y, a, ?  At "Allow …?"' in out  # the question works anywhere

    def test_help_lists_the_commands_in_order(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), self.inputs("/help"))

        names = [
            line.split()[0]
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("  /")
        ]
        assert names == sorted(COMMANDS)

    def test_help_lists_every_registered_command(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), self.inputs("/help"))

        out = " ".join(capsys.readouterr().out.split())  # a long line wraps
        for name, cmd in COMMANDS.items():
            assert name in out
            assert cmd.description in out


class TestCommandRegistry:
    def test_a_returned_message_is_shown_as_a_notice(
        self, monkeypatch, make_agent, capsys
    ):
        pong = Command(lambda repl, args: f"pong {args}", "test")
        monkeypatch.setitem(COMMANDS, "/ping", pong)

        run_repl(monkeypatch, make_agent(), ["/PING now", "/exit"])

        assert "[pong now]\n\n" in capsys.readouterr().out

    def test_a_handler_that_returns_nothing_prints_no_notice(
        self, monkeypatch, make_agent, capsys
    ):
        monkeypatch.setitem(COMMANDS, "/quiet", Command(lambda repl, args: None, "t"))

        run_repl(monkeypatch, make_agent(), ["/quiet", "/exit"])

        assert "[" not in capsys.readouterr().out

    def test_a_failing_command_is_reported_and_the_session_survives(
        self, monkeypatch, make_agent, capsys
    ):
        def save(repl, args):
            raise OSError("disk full")

        monkeypatch.setitem(COMMANDS, "/save", Command(save, "test"))
        agent = make_agent()

        run_repl(monkeypatch, agent, ["hi", "/save x.md", "/exit"])

        out = capsys.readouterr().out
        assert "[Command error: OSError: disk full]" in out
        assert len(agent.context.items) == 2  # the conversation is still there
        assert out.endswith("Bye.\n")

    def test_ctrl_c_in_a_command_cancels_only_that_command(
        self, monkeypatch, make_agent, capsys
    ):
        def slow(repl, args):
            raise KeyboardInterrupt

        monkeypatch.setitem(COMMANDS, "/slow", Command(slow, "test"))

        run_repl(monkeypatch, make_agent(), ["/slow", "/exit"])

        out = capsys.readouterr().out
        assert "[Cancelled]" in out
        assert out.endswith("Bye.\n")

    def test_registering_a_name_twice_is_an_error(self):
        with pytest.raises(ValueError, match="already registered: /help"):
            command("/help")(lambda repl, args: None)


class TestCommandDefinition:
    @pytest.mark.parametrize(
        "name", ["help", "/Reset", "/two words", "", "/", "/a_b", "/-x", "/1x"]
    )
    def test_an_invalid_name_is_rejected(self, name):
        # Otherwise "help" would hijack a plain prompt, and "/Reset" could never be
        # typed: input is lowercased before the lookup.
        with pytest.raises(ValueError, match="Command names are"):
            command(name)

    def test_a_handler_needs_a_docstring(self, monkeypatch):
        registry: dict = {}
        monkeypatch.setattr("ergane.commands.COMMANDS", registry)

        with pytest.raises(ValueError, match="Command /probe needs a docstring"):
            command("/probe")(lambda repl, args: None)

        assert registry == {}

    def test_the_description_is_the_first_line_of_the_docstring(self, monkeypatch):
        registry: dict = {}
        monkeypatch.setattr("ergane.commands.COMMANDS", registry)

        @command("/probe")
        def probe(repl, args):
            """Do a thing.

            More detail, which /help does not show.
            """

        assert registry["/probe"] == Command(probe, "Do a thing.")

    def test_every_builtin_command_has_a_short_one_line_description(self):
        for name, cmd in COMMANDS.items():
            assert cmd.description, name
            assert len(cmd.description) < 90, name


def look(path: str) -> str:
    """Look at a file."""
    return path


def save(path: str) -> str:
    """Save a file."""
    return path


LOOK = Tool.from_function(look)
SAVE = Tool.from_function(save, read_only=False)


class TestContextCommand:
    def test_it_shows_the_parts_of_the_budget(self, monkeypatch, make_agent, capsys):
        agent = make_agent(tools=[LOOK, SAVE], MAX_CONTEXT_TOKENS=10000)
        instructions, tools, history = agent.context_estimate()
        total = instructions + tools + history

        run_repl(monkeypatch, agent, ["/context", "/exit"])

        out = capsys.readouterr().out
        assert f"  instructions   ~{instructions:,}\n" in out
        # The todo tool is offered with the others, so it is counted (and costs tokens).
        assert f"  tools (3)      ~{tools:,}\n" in out
        assert f"  history        ~{history:,}\n" in out
        assert (
            f"  total          ~{total:,} of 10,000 ({total * 100 // 10000}%)\n" in out
        )
        assert "last request" not in out  # nothing has been sent yet

    def test_the_total_is_the_servers_count_while_it_holds(
        self, monkeypatch, make_agent, capsys
    ):
        agent = make_agent(MAX_CONTEXT_TOKENS=10000)
        anchor(agent, 2500)

        run_repl(monkeypatch, agent, ["/context", "/exit"])

        out = capsys.readouterr().out
        assert "  total          2,500 of 10,000 (25%) (counted by the server)\n" in out

    def test_it_adds_the_count_of_the_server_once_there_is_one(
        self, monkeypatch, make_agent, capsys
    ):
        agent = make_agent()
        agent.last_input_tokens = 12345

        run_repl(monkeypatch, agent, ["/context", "/exit"])

        assert (
            "  last request   12,345 (counted by the server)\n"
            in capsys.readouterr().out
        )

    def test_the_history_grows_and_a_reset_empties_it(
        self, monkeypatch, make_agent, capsys
    ):
        agent = make_agent()
        agent.run_turn("hello")
        assert agent.context_estimate()[2] > 0

        run_repl(monkeypatch, agent, ["/reset", "/context", "/exit"])

        assert "  history        ~0\n" in capsys.readouterr().out


def long_history() -> list:
    items = []
    for i in range(10):
        items.append({"role": "user", "content": f"q{i} " + "x" * 400})
        items.append({"role": "assistant", "content": f"a{i} " + "y" * 400})
    return items


class TestCompactCommands:
    def agent(self, make_agent, **config):
        agent = make_agent(MAX_CONTEXT_TOKENS=4000, KEEP_RECENT_TOKENS=500, **config)
        agent.context.items = long_history()
        return agent

    def test_compact_summarizes_now_and_summary_shows_it(
        self, monkeypatch, make_agent, capsys
    ):
        agent = self.agent(make_agent, AUTO_COMPACT=False)

        run_repl(monkeypatch, agent, ["/compact", "/summary", "/exit"])

        out = capsys.readouterr().out
        assert "[Summarizing earlier messages…]" in out
        assert "[Summarized " in out and "earlier turns" in out
        assert "\nNOTES 8\n" in out  # shown by /summary: the 8 turns summarized
        assert agent.context.summary == "## Goal\nNOTES 8"

    def test_a_focus_goes_to_the_model(self, monkeypatch, make_agent, fake_server):
        agent = self.agent(make_agent, AUTO_COMPACT=False)

        run_repl(monkeypatch, agent, ["/compact the database schema", "/exit"])

        assert (
            "Focus: the database schema"
            in fake_server.requests[0]["input"][0]["content"]
        )

    def test_compact_says_when_there_is_nothing_to_summarize(
        self, monkeypatch, make_agent, capsys
    ):
        run_repl(monkeypatch, make_agent(), ["/compact", "/exit"])

        assert "[Nothing to summarize: the history is short]" in capsys.readouterr().out

    def test_a_summary_that_fails_is_reported_and_changes_nothing(
        self, monkeypatch, make_agent, capsys
    ):
        agent = self.agent(make_agent, AUTO_COMPACT=False)
        agent.context.items[0] = {"role": "user", "content": "SUMMARY_EMPTY"}
        before = list(agent.context.items)

        run_repl(monkeypatch, agent, ["/compact", "/exit"])

        assert "[Could not summarize: the summary is empty]" in capsys.readouterr().out
        assert agent.context.items == before and agent.context.summary is None

    def test_summary_before_there_is_one(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), ["/summary", "/exit"])

        assert "[No summary yet]" in capsys.readouterr().out

    def test_reset_forgets_the_summary(self, monkeypatch, make_agent):
        agent = self.agent(make_agent, AUTO_COMPACT=False)
        agent.context.summary = "NOTES"

        run_repl(monkeypatch, agent, ["/reset", "/exit"])

        assert agent.context.summary is None

    def test_context_says_where_summaries_start(self, monkeypatch, make_agent, capsys):
        agent = make_agent(MAX_CONTEXT_TOKENS=10000)

        run_repl(monkeypatch, agent, ["/context", "/exit"])

        assert "  summarizes at  ~7,500\n" in capsys.readouterr().out

    def test_context_says_when_they_are_off(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(AUTO_COMPACT=False), ["/context", "/exit"])

        assert "off (/compact does it by hand)" in capsys.readouterr().out


class TestServersCommand:
    def servers(self):
        return [
            ServerStatus("web", "running", tools=("web__a", "web__b"), hidden=3),
            ServerStatus(
                "files", "running", tools=("files__x",), log=Path.home() / "f.log"
            ),
            ServerStatus("bad", "failed", "MCP server 'bad' failed to start: X"),
            ServerStatus("old", "off", "enabled is false"),
        ]

    @staticmethod
    def hub_of(tmp_path, statuses) -> McpHub:
        hub = McpHub({}, tmp_path)
        hub.statuses = {s.name: s for s in statuses}
        return hub

    def test_it_says_when_there_are_none(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), ["/servers", "/exit"])

        assert "[No MCP servers]" in capsys.readouterr().out

    def test_every_server_gets_a_line_with_what_became_of_it(
        self, monkeypatch, make_agent, capsys, tmp_path
    ):
        def tool(name: str) -> Tool:
            def function() -> str:
                """Do nothing."""
                return ""

            function.__name__ = name
            return Tool.from_function(function)

        names = ("web__a", "web__b", "files__x")
        agent = make_agent(tools=[tool(name) for name in names])

        hub = self.hub_of(tmp_path, self.servers())
        run_repl(monkeypatch, agent, ["/servers", "/exit"], hub=hub)

        out = capsys.readouterr().out
        assert '  web    running, 2 tools (3 left out by "tools"), ~' in out
        assert "  files  running, 1 tool, ~" in out and ", log ~/f.log" in out
        assert "  bad    failed: MCP server 'bad' failed to start: X" in out
        assert "  old    off: enabled is false" in out

    def test_the_tokens_are_those_of_the_definitions_the_model_gets(
        self, monkeypatch, make_agent, capsys, tmp_path
    ):
        agent = make_agent(tools=[LOOK])
        status = [ServerStatus("s", "running", tools=("look",))]

        hub = self.hub_of(tmp_path, status)
        run_repl(monkeypatch, agent, ["/servers", "/exit"], hub=hub)

        tokens = estimate_tokens(LOOK.param())  # type: ignore[arg-type]
        assert f"~{tokens:,} tokens" in capsys.readouterr().out


class TestToolsCommand:
    def test_it_lists_the_tools_by_whether_they_ask_first(
        self, monkeypatch, make_agent, capsys
    ):
        agent = make_agent(tools=[LOOK, SAVE])

        run_repl(monkeypatch, agent, ["/tools", "/exit"])

        out = capsys.readouterr().out
        assert "  run without asking  todo, look\n  ask first           save\n\n" in out

    def test_a_group_with_no_tools_is_left_out(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(tools=[LOOK]), ["/tools", "/exit"])

        out = capsys.readouterr().out
        assert "  run without asking  todo, look\n" in out
        assert "ask first" not in out

    def test_a_tool_allowed_for_the_session_is_listed_on_its_own(
        self, monkeypatch, make_agent, capsys
    ):
        agent = make_agent(tools=[LOOK, SAVE])
        agent.always_allowed.add("save")

        run_repl(monkeypatch, agent, ["/tools", "/exit"])

        out = capsys.readouterr().out
        assert (
            "  run without asking    todo, look\n  allowed this session  save\n\n"
            in out
        )
        assert "ask first" not in out

    def test_it_says_when_there_are_none(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), ["/tools", "/exit"])

        assert "[Tools: none]" in capsys.readouterr().out


FAKE = str(Path(__file__).parent / "fake_mcp_server.py")


class TestReconnectCommand:
    @pytest.fixture
    def hub(self, tmp_path: Path):
        hub = McpHub(
            {
                "fake": LocalMcpServer(
                    command=sys.executable, args=[FAKE], instructions=True
                )
            },
            tmp_path,
        )
        yield hub
        hub.close()

    def agent_of(self, make_agent, hub: McpHub):
        tools = hub.start(lambda notice: None)
        return make_agent(tools=tools, server_notes=hub.instructions())

    def test_without_servers_it_says_so(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), ["/reconnect", "/exit"])

        assert "[No MCP servers to start]" in capsys.readouterr().out

    def test_an_unknown_name_is_a_warning(self, make_agent, hub):
        repl = Repl(self.agent_of(make_agent, hub), COMMANDS, hub=hub)

        result = COMMANDS["/reconnect"].handler(repl, "nope")

        assert isinstance(result, Warn) and "No MCP server 'nope'" in result

    def test_an_unknown_name_is_reported(self, monkeypatch, make_agent, hub, capsys):
        agent = self.agent_of(make_agent, hub)

        run_repl(monkeypatch, agent, ["/reconnect nope", "/exit"], hub=hub)

        assert "[No MCP server 'nope'; /servers lists them]" in capsys.readouterr().out

    def test_when_no_server_failed_it_says_so(
        self, monkeypatch, make_agent, hub, capsys
    ):
        agent = self.agent_of(make_agent, hub)

        run_repl(monkeypatch, agent, ["/reconnect", "/exit"], hub=hub)

        assert "[No MCP server failed;" in capsys.readouterr().out

    def test_a_server_that_failed_is_started_and_the_agent_gets_its_tools(
        self, monkeypatch, make_agent, tmp_path, capsys
    ):
        configs = {"fake": LocalMcpServer(command="/no/such/server")}
        hub = McpHub(configs, tmp_path)
        try:
            agent = self.agent_of(make_agent, hub)
            assert agent.tools == {} and agent.server_notes == {}
            configs["fake"] = LocalMcpServer(
                command=sys.executable, args=[FAKE], instructions=True
            )

            run_repl(monkeypatch, agent, ["/reconnect", "/exit"], hub=hub)

            assert "fake__add" in agent.tools
            assert agent.server_notes == {"fake": "Dates are YYYY-MM-DD."}
            assert "[MCP server 'fake' is running: " in capsys.readouterr().out
        finally:
            hub.close()

    def test_a_server_that_runs_is_started_again_by_name(
        self, monkeypatch, make_agent, hub, capsys
    ):
        agent = self.agent_of(make_agent, hub)
        before = agent.tools["fake__pid"].call("{}")

        run_repl(monkeypatch, agent, ["/reconnect fake", "/exit"], hub=hub)

        assert agent.tools["fake__pid"].call("{}") != before
        assert "[MCP server 'fake' is running: " in capsys.readouterr().out

    def test_a_server_that_fails_again_takes_its_tools_away(
        self, monkeypatch, make_agent, hub, capsys
    ):
        agent = self.agent_of(make_agent, hub)
        hub.configs = {"fake": LocalMcpServer(command="/no/such/server")}

        run_repl(monkeypatch, agent, ["/reconnect fake", "/exit"], hub=hub)

        assert agent.tools == {} and agent.server_notes == {}
        assert "failed to start: FileNotFoundError" in capsys.readouterr().out


class TestServersStartingInTheBackground:
    FAKE_CONFIG = fake_config()

    @pytest.fixture
    def hubs(self, tmp_path: Path):
        made: list[McpHub] = []

        def make(configs) -> McpHub:
            hub = McpHub(configs, tmp_path)
            made.append(hub)
            return hub

        yield make
        for hub in made:
            hub.close()

    def test_the_first_turn_has_the_tools_though_the_agent_began_without(
        self, monkeypatch, make_agent, fake_server, hubs
    ):
        hub = hubs({"fake": self.FAKE_CONFIG})
        hub.begin()
        agent = make_agent()
        assert agent.tools == {}

        run_repl(monkeypatch, agent, ["hi", "/exit"], hub=hub)

        sent = [tool["name"] for tool in fake_server.requests[0]["tools"]]
        assert "fake__add" in sent

    def test_a_turn_says_it_waits_when_the_servers_are_not_up(
        self, monkeypatch, make_agent, fake_server, hubs, capsys
    ):
        hub = hubs({"slow": hanging_config(startup_timeout=0.5)})
        hub.begin()

        run_repl(monkeypatch, make_agent(), ["hi", "/exit"], hub=hub)

        out = capsys.readouterr().out
        assert "[Waiting for the MCP servers…]" in out
        assert "did not answer within 0.5 s" in out
        assert out.index("Waiting") < out.index("did not answer")
        assert len(fake_server.requests) == 1  # the turn went on without them

    def test_a_server_that_failed_is_told_at_the_first_input(
        self, monkeypatch, make_agent, hubs, capsys
    ):
        hub = hubs({"bad": LocalMcpServer(command="/no/such/server")})
        hub.begin()
        deadline = time.monotonic() + 10
        while hub.starting and time.monotonic() < deadline:
            time.sleep(0.05)  # the user would have taken longer to type

        run_repl(monkeypatch, make_agent(), ["/help", "/exit"], hub=hub)

        assert "[MCP server 'bad' failed to start: FileNotFoundError" in (
            capsys.readouterr().out
        )

    def test_servers_says_starting_for_a_server_that_is_not_up_yet(
        self, monkeypatch, make_agent, hubs, capsys
    ):
        hub = hubs({"slow": hanging_config()})
        hub.begin()

        run_repl(monkeypatch, make_agent(), ["/servers", "/exit"], hub=hub)

        assert "  slow  starting…\n" in capsys.readouterr().out

    def test_reconnect_waits_for_the_start_before_it_counts_what_failed(
        self, monkeypatch, make_agent, hubs
    ):
        configs = {"fake": LocalMcpServer(command="/no/such/server")}
        hub = hubs(configs)
        hub.begin()
        configs["fake"] = self.FAKE_CONFIG  # the program was installed meanwhile
        agent = make_agent()

        run_repl(monkeypatch, agent, ["/reconnect", "/exit"], hub=hub)

        assert "fake__add" in agent.tools


class TestSessionsCommand:
    @pytest.fixture
    def store(self, tmp_path):
        return SessionStore(tmp_path / "state" / "sessions", tmp_path)

    def saved(self, store: SessionStore, *prompts: str) -> list[SessionInfo]:
        """Sessions with one turn each, saved in the order given: the last is newest."""
        return [
            store.save(
                None,
                None,
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"re: {prompt}"},
                ],
                "gpt-4o",
            )
            for prompt in prompts
        ]

    def repl(self, make_agent, store) -> Repl:
        return Repl(make_agent(STREAMING=False), COMMANDS, store=store)

    def test_a_session_that_cannot_be_found_is_a_warning(self, make_agent, store):
        self.saved(store, "first")
        repl = self.repl(make_agent, store)

        result = COMMANDS["/sessions"].handler(repl, "no-such-id")

        assert isinstance(result, Warn)

    def test_the_menu_shows_the_end_of_an_id_and_the_typed_list_all_of_it(
        self, monkeypatch, make_agent, store
    ):
        [info] = self.saved(store, "first")
        repl = self.repl(make_agent, store)
        offered: list[str] = []

        def choose(question, options):
            offered.extend(label for _, label in options)
            return "q"

        repl.console.choose = choose
        COMMANDS["/sessions"].handler(repl, "")

        assert offered[0].endswith(info.id.rsplit("-", 1)[-1])
        assert info.id not in offered[0]

    def test_it_says_when_sessions_are_not_saved(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), ["/sessions", "/exit"])

        assert "[Sessions are not saved]" in capsys.readouterr().out

    def test_it_says_when_there_are_none(self, monkeypatch, make_agent, store, capsys):
        run_repl(monkeypatch, make_agent(), ["/sessions", "/exit"], store=store)

        assert "[No saved sessions for this directory]" in capsys.readouterr().out

    def test_without_a_terminal_it_lists_them_by_number_newest_first(
        self, make_agent, store, clock, capsys
    ):
        older, newer = self.saved(store, "older prompt", "newer prompt")

        self.repl(make_agent, store).handle("/sessions")

        out = capsys.readouterr().out
        assert f"  1  newer prompt  ·  1 turn  ·  {ago(newer)}  ·  {newer.id}\n" in out
        assert f"  2  older prompt  ·  1 turn  ·  {ago(older)}  ·  {older.id}\n" in out
        assert "[Continue one with /sessions NUMBER or /sessions ID]" in out

    def test_a_number_continues_that_session_in_place_of_the_conversation(
        self, make_agent, store, clock, capsys
    ):
        older, _ = self.saved(store, "older prompt", "newer prompt")
        repl = self.repl(make_agent, store)
        repl.agent.context.items = make_context(3)

        repl.handle("/sessions 2")

        assert 'Resumed "older prompt": 1 turn, saved ' in capsys.readouterr().out
        assert [m["content"] for m in as_dicts(repl.agent.context.items)] == [
            "older prompt",
            "re: older prompt",
        ]
        assert repl.saved == older

    def test_an_id_or_the_start_of_one_continues_that_session(
        self, make_agent, store, clock
    ):
        older, _ = self.saved(store, "older prompt", "newer prompt")
        repl = self.repl(make_agent, store)

        repl.handle(f"/sessions {older.id}")
        assert repl.saved == older
        repl.saved = None
        repl.handle(f"/sessions {older.id[:-1]}")  # the start of it

        assert repl.saved == older

    def test_an_id_that_names_nothing_is_said_so_and_nothing_changes(
        self, make_agent, store, clock, capsys
    ):
        self.saved(store, "one")
        repl = self.repl(make_agent, store)
        repl.agent.context.items = make_context(1)

        repl.handle("/sessions nope")
        repl.handle("/sessions 9")  # no ninth: it is looked for as an id

        assert capsys.readouterr().out.count("No saved session") == 2
        assert repl.agent.context.items == make_context(1)

    def test_a_session_that_cannot_be_read_is_said_so(self, make_agent, store, capsys):
        [info] = self.saved(store, "one")
        (store.directory / f"{info.id}.jsonl").write_text(
            (store.directory / f"{info.id}.jsonl").read_text().split("\n")[0] + "\n[]\n"
        )
        repl = self.repl(make_agent, store)

        repl.handle("/sessions 1")

        assert f"[The session {info.id} cannot be read" in capsys.readouterr().out

    def test_at_a_terminal_a_menu_offers_the_newest_first_and_marks_the_current(
        self, make_agent, store, clock
    ):
        older, newer = self.saved(store, "older prompt", "newer prompt")
        repl = self.repl(make_agent, store)
        repl.saved = older
        offered = []

        def choose(question, options):
            offered.extend(options)
            return "q"

        repl.console.choose = choose

        repl.handle("/sessions")

        assert [key for key, _ in offered] == [newer.id, older.id, "q"]
        assert offered[0][1].startswith("newer prompt  ·  1 turn  ·  ")
        assert offered[1][1].endswith(f"{older.id.rsplit('-', 1)[-1]} (current)")
        assert offered[2] == ("q", "Cancel")

    def test_the_choice_in_the_menu_is_continued_and_saved_over_afterwards(
        self, make_agent, store, clock, capsys
    ):
        older, newer = self.saved(store, "older prompt", "newer prompt")
        repl = self.repl(make_agent, store)
        repl.console.choose = lambda question, options: older.id

        repl.handle("/sessions")
        repl.handle("one more")
        repl.save_session()

        assert [s.id for s in store.sessions()] == [older.id, newer.id]
        assert store.load(older.id).info.turns == 2  # the same session, gone on with

    def test_cancelling_changes_nothing(self, make_agent, store, clock, capsys):
        self.saved(store, "one")
        repl = self.repl(make_agent, store)
        repl.agent.context.items = make_context(1)
        repl.console.choose = lambda question, options: "q"

        repl.handle("/sessions")

        assert repl.agent.context.items == make_context(1) and repl.saved is None

    def test_the_menu_offers_the_newest_ten_and_the_list_all_of_them(
        self, make_agent, store, clock, capsys
    ):
        self.saved(store, *[f"prompt {n:02d}" for n in range(12)])
        repl = self.repl(make_agent, store)
        offered = []
        repl.console.choose = lambda question, options: offered.extend(options) or "q"

        repl.handle("/sessions")
        repl.console.choose = None
        repl.handle("/sessions")

        assert len(offered) == 11 and offered[0][1].startswith("prompt 11")
        assert "  12  prompt 00" in capsys.readouterr().out

    def test_reset_makes_the_next_prompt_a_new_session(self, make_agent, store, clock):
        [first] = self.saved(store, "one")
        repl = self.repl(make_agent, store)
        repl.saved = first

        repl.handle("/reset")

        assert repl.saved is None
