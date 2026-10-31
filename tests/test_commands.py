import pytest

from fake_api import TEXT
from helpers import run_repl
from muagent.commands import COMMANDS, Command, command


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

    def test_help_lists_the_new_commands(self, monkeypatch, make_agent, capsys):
        run_repl(monkeypatch, make_agent(), self.inputs("/help"))

        out = capsys.readouterr().out
        for name in ("/retry", "/undo", "/model", "/effort"):
            assert name in out


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

    def test_registering_a_name_twice_is_an_error(self):
        with pytest.raises(ValueError, match="already registered: /help"):
            command("/help", "again")(lambda repl, args: None)
