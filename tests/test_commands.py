import sys

import pytest

from fake_api import TEXT
from helpers import run_repl
from muagent.commands import COMMANDS, Command, command
from muagent.tools import Tool


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

    def test_help_lists_every_registered_command(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), self.inputs("/help"))

        out = capsys.readouterr().out
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
        monkeypatch.setattr("muagent.commands.COMMANDS", registry)

        with pytest.raises(ValueError, match="Command /probe needs a docstring"):
            command("/probe")(lambda repl, args: None)

        assert registry == {}

    def test_the_description_is_the_first_line_of_the_docstring(self, monkeypatch):
        registry: dict = {}
        monkeypatch.setattr("muagent.commands.COMMANDS", registry)

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


class TestToolsCommand:
    def test_it_lists_the_tools_by_whether_they_ask_first(
        self, monkeypatch, make_agent, capsys
    ):
        agent = make_agent(tools=[LOOK, SAVE])

        run_repl(monkeypatch, agent, ["/tools", "/exit"])

        out = capsys.readouterr().out
        assert "  run without asking  look\n  ask first           save\n\n" in out

    def test_a_group_with_no_tools_is_left_out(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(tools=[LOOK]), ["/tools", "/exit"])

        out = capsys.readouterr().out
        assert "  run without asking  look\n" in out
        assert "ask first" not in out

    def test_it_says_when_there_are_none(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), ["/tools", "/exit"])

        assert "[Tools: none]" in capsys.readouterr().out
