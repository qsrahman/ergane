import io
from pathlib import Path

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output.vt100 import Vt100_Output

from ergane.agent import Agent
from ergane.commands import COMMANDS
from ergane.console import Console
from ergane.errors import TurnError
from ergane.mcp_hub import ServerUpdate
from ergane.repl import Repl
from ergane.sessions import SessionStore
from ergane.tools import Tool
from fake_api import REFUSAL, STREAM_ERROR, TEXT
from helpers import make_config, make_context, run_repl


class TestRepl:
    @pytest.fixture
    def repl(self, monkeypatch: pytest.MonkeyPatch):
        """Run a Repl with scripted input; returns the prompts sent to the agent.

        A callable passed as `turn` is called with each prompt and the agent's
        context in place of Agent.run_turn.
        """

        def run(inputs: list, turn=None) -> list[str]:
            agent = Agent(make_config())
            prompts: list[str] = []

            def fake_run_turn(
                prompt: str,
                on_text=None,
                on_notice=None,
                confirm=None,
                on_call=None,
                on_todos=None,
            ) -> None:
                prompts.append(prompt)
                if turn:
                    turn(prompt, agent.context)

            monkeypatch.setattr(agent, "run_turn", fake_run_turn)
            run_repl(monkeypatch, agent, inputs)
            return prompts

        return run

    @pytest.mark.parametrize(
        "stop",
        [
            "/exit",
            "/quit",
            "  /EXIT  ",
            EOFError(),
            [KeyboardInterrupt(), KeyboardInterrupt()],
        ],
    )
    def test_stops_on_exit_command_or_eof(self, repl, capsys, stop):
        assert repl(stop if isinstance(stop, list) else [stop]) == []
        assert capsys.readouterr().out.endswith("Bye.\n")

    def test_end_of_input_quits_at_once_when_it_is_not_a_terminal(self, repl):
        def talk(prompt, context):
            context.items.append({"role": "user", "content": prompt})

        assert repl(["hi", EOFError()], turn=talk) == ["hi"]

    def test_one_ctrl_c_at_the_prompt_does_not_quit(self, repl, capsys):
        assert repl([KeyboardInterrupt(), "hi", "/exit"]) == ["hi"]
        assert "Press Ctrl-C again, or type /exit, to quit" in capsys.readouterr().out

    def test_typing_between_two_ctrl_c_starts_the_count_again(self, repl, capsys):
        inputs = [KeyboardInterrupt(), "hi", KeyboardInterrupt(), "there", "/exit"]

        assert repl(inputs) == ["hi", "there"]
        assert capsys.readouterr().out.count("Press Ctrl-C again") == 2

    def test_strips_input_and_skips_blank_lines(self, repl):
        assert repl(["", "   ", "  hi  ", "/exit"]) == ["hi"]

    def test_ctrl_c_cancels_only_the_current_turn(self, repl, capsys):
        def turn(prompt: str, _context) -> None:
            if prompt == "slow":
                raise KeyboardInterrupt

        assert repl(["slow", "next", "/exit"], turn) == ["slow", "next"]
        assert "[Cancelled]" in capsys.readouterr().out

    def test_reply_ending_in_a_newline_gets_no_extra_blank_line(
        self, monkeypatch, capsys
    ):
        agent = Agent(make_config())

        def turn(prompt: str, on_text, on_notice, confirm, on_call, on_todos) -> None:
            on_text("line\n")

        monkeypatch.setattr(agent, "run_turn", turn)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        assert "\nline\n\nBye." in capsys.readouterr().out

    def test_ctrl_c_mid_reply_puts_the_notice_on_its_own_line(
        self, monkeypatch, capsys
    ):
        agent = Agent(make_config())

        def turn(prompt: str, on_text, on_notice, confirm, on_call, on_todos) -> None:
            on_text("partial")
            raise KeyboardInterrupt

        monkeypatch.setattr(agent, "run_turn", turn)
        run_repl(monkeypatch, agent, ["hi", "/exit"])

        assert "partial\n\n[Cancelled]\n" in capsys.readouterr().out

    def test_turn_errors_are_reported_and_loop_continues(self, repl, capsys):
        def turn(prompt: str, _context) -> None:
            if prompt == "bad":
                raise ValueError("boom")

        assert repl(["bad", "next", "/exit"], turn) == ["bad", "next"]
        assert "Agent error: ValueError: boom" in capsys.readouterr().out

    def test_unknown_command_is_not_sent_to_the_agent(self, repl, capsys):
        assert repl(["/rest", "/exit"]) == []
        assert "[Unknown command: /rest. Type /help" in capsys.readouterr().out

    def test_slash_prompt_with_spaces_goes_to_the_agent(self, repl):
        prompt = "/usr/bin is missing"

        assert repl([prompt, "/exit"]) == [prompt]


def read_file(path: str) -> str:
    """Read a text file (the tool the fake API calls)."""
    return "contents"


@pytest.mark.parametrize("streaming", [True, False])
class TestReplRendering:
    """The Repl against a real Agent and the fake server."""

    def test_prints_the_reply(self, monkeypatch, make_agent, capsys, streaming):
        run_repl(monkeypatch, make_agent(STREAMING=streaming), ["hi", "/exit"])

        assert f"\n{TEXT}\n\nBye." in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("prompt", "notice"),
        [
            ("incomplete", "[Response incomplete: max_output_tokens]"),
            ("failed", "[Response failed: invalid_prompt: boom]"),
            ("empty", "[No text in response]"),
            ("refusal", f"[Refused: {REFUSAL}]"),
        ],
    )
    def test_reports_unsuccessful_responses_on_their_own_line(
        self, monkeypatch, make_agent, capsys, streaming, prompt, notice
    ):
        run_repl(monkeypatch, make_agent(STREAMING=streaming), [prompt, "/exit"])

        assert f"\n{notice}\n" in capsys.readouterr().out

    @staticmethod
    def terminal(monkeypatch, answers):
        """Stand in for input() the way a terminal looks: the prompt is shown, then
        what was typed and the Enter that ends the line."""
        typed = iter(answers)

        def fake_input(prompt: str) -> str:
            line = next(typed)
            print(f"{prompt}{line}")
            return line

        monkeypatch.setattr("builtins.input", fake_input)

    def run(self, agent):
        Repl(agent, COMMANDS).run()

    def test_a_call_and_the_reply_are_each_set_apart_by_a_blank_line(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        agent = make_agent(tools=[Tool.from_function(read_file)], STREAMING=streaming)
        self.terminal(monkeypatch, ["tool_call", "/exit"])

        self.run(agent)

        out = capsys.readouterr().out
        assert out.endswith(
            f"> tool_call\n\n▸ read_file  path=notes.txt\n\n{TEXT}\n\n> /exit\nBye.\n"
        )

    def test_a_call_that_asks_is_one_line_and_nothing_else(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        tool = Tool.from_function(read_file, read_only=False)
        agent = make_agent(tools=[tool], STREAMING=streaming)
        self.terminal(monkeypatch, ["tool_call", "y", "/exit"])

        self.run(agent)

        out = capsys.readouterr().out
        assert out.endswith(
            '> tool_call\n\nAllow read_file {"path": "notes.txt"}? [y/N/a] y\n\n'
            f"{TEXT}\n\n> /exit\nBye.\n"
        )

    def test_the_banner_is_one_line(self, monkeypatch, make_agent, capsys, streaming):
        # The effort and the tools are in the toolbar, which a terminal shows.
        agent = make_agent(STREAMING=streaming, REASONING_EFFORT="high")
        self.terminal(monkeypatch, ["/exit"])

        self.run(agent)

        banner, blank, typed = capsys.readouterr().out.splitlines()[:3]
        # The directory the file tools work in is named, shortened if it is long.
        assert banner.startswith("ergane · gpt-4o · ")
        assert banner.endswith(" · /help for commands")
        assert Path.cwd().name in banner
        assert (blank, typed) == ("", "> /exit")

    def test_an_api_error_is_one_line_with_what_to_check(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        agent = make_agent(STREAMING=streaming)
        monkeypatch.setenv("COLUMNS", "200")
        self.terminal(monkeypatch, ["status_429", "/exit"])

        self.run(agent)

        assert (
            "[API error 429 (code_429): message 429 The account is out of quota, "
            "or rate limited.]"
        ) in capsys.readouterr().out

    def test_a_tool_call_is_shown_as_a_notice_before_the_reply(
        self, monkeypatch, make_agent, capsys, streaming
    ):
        def read_file(path: str) -> str:
            """Read a text file."""
            return "contents"

        agent = make_agent(tools=[Tool.from_function(read_file)], STREAMING=streaming)

        run_repl(monkeypatch, agent, ["tool_call", "/exit"])

        out = capsys.readouterr().out
        shown = "▸ read_file  path=notes.txt"
        assert f"\n{shown}\n\n{TEXT}\n\nBye." in out

    def test_reports_trimmed_context(self, monkeypatch, make_agent, capsys, streaming):
        agent = make_agent(STREAMING=streaming, MAX_CONTEXT_TOKENS=1)
        agent.context.items = make_context(2)

        run_repl(monkeypatch, agent, ["hi", "/exit"])

        out = capsys.readouterr().out
        assert f"Context trimmed: oldest turns dropped\n\n{TEXT}\n" in out


def test_stream_error_notice_follows_the_partial_reply(monkeypatch, make_agent, capsys):
    run_repl(monkeypatch, make_agent(STREAMING=True), ["error_event", "/exit"])

    out = capsys.readouterr().out
    assert f"Hello\n\n[Stream error: server_error: {STREAM_ERROR}]\n" in out


class TestReplConfirm:
    """The Repl hands the console's question to the agent; see test_console."""

    @pytest.fixture
    def run(self, monkeypatch, make_agent):
        """Run "tool_call" against a tool that is not read-only, answering its
        question with `answer`; returns the paths the tool was run with."""

        def run(answer) -> list[str]:
            ran: list[str] = []

            def read_file(path: str) -> str:
                """Read a text file."""
                ran.append(path)
                return "contents"

            agent = make_agent(tools=[Tool.from_function(read_file, read_only=False)])
            run_repl(monkeypatch, agent, ["tool_call", answer, "/exit"])
            return ran

        return run

    @pytest.mark.parametrize("answer", ["y", " YES "])
    def test_yes_lets_the_tool_run(self, run, answer):
        assert run(answer) == ["notes.txt"]

    @pytest.mark.parametrize("answer", ["", "n", "no", "yep", "ok"])
    def test_anything_else_denies_it(self, run, answer):
        assert run(answer) == []

    def test_end_of_input_denies_it(self, run):
        assert run(EOFError()) == []

    def test_ctrl_c_at_the_question_cancels_the_turn(self, run, capsys):
        assert run(KeyboardInterrupt()) == []
        assert "[Cancelled]" in capsys.readouterr().out


def test_a_server_tool_whose_name_is_taken_is_reported(capsys):
    def look() -> str:
        """Look."""
        return "seen"

    mine, theirs = Tool.from_function(look), Tool.from_function(look)
    repl = Repl(Agent(make_config(), tools=[mine]), COMMANDS)

    repl.apply(ServerUpdate("web", (), [theirs], None, ""))

    assert repl.agent.tools["look"] is mine
    assert "[Tool look is defined twice; only the first is offered]" in (
        capsys.readouterr().out
    )


class TestSessions:
    """The conversation is saved after each prompt, and a saved one can be continued."""

    @pytest.fixture
    def store(self, tmp_path) -> SessionStore:
        return SessionStore(tmp_path / "state" / "sessions", tmp_path)

    def test_a_prompt_is_saved_when_it_is_handled(self, monkeypatch, make_agent, store):
        run_repl(
            monkeypatch, make_agent(STREAMING=False), ["hello", "/exit"], store=store
        )

        [info] = store.sessions()
        assert info.title == "hello" and info.turns == 1
        assert [m["content"] for m in store.load(info.id).items] == ["hello", TEXT]

    def test_a_conversation_with_nothing_in_it_saves_nothing(
        self, monkeypatch, make_agent, store
    ):
        run_repl(monkeypatch, make_agent(), ["/help", "/exit"], store=store)

        assert store.sessions() == [] and not store.directory.exists()

    def test_a_reset_that_comes_first_does_not_make_an_empty_session(
        self, monkeypatch, make_agent, store
    ):
        run_repl(monkeypatch, make_agent(), ["/reset", "/exit"], store=store)

        assert store.sessions() == []

    def test_without_a_store_nothing_is_written(
        self, monkeypatch, make_agent, tmp_path
    ):
        run_repl(monkeypatch, make_agent(STREAMING=False), ["hello", "/exit"])

        assert not (tmp_path / "state").exists()

    def test_every_prompt_is_saved_over_the_same_session(
        self, monkeypatch, make_agent, store
    ):
        run_repl(
            monkeypatch,
            make_agent(STREAMING=False),
            ["one", "two", "/exit"],
            store=store,
        )

        [info] = store.sessions()
        assert info.title == "one" and info.turns == 2

    def test_a_command_that_changes_the_history_is_saved_too(
        self, monkeypatch, make_agent, store
    ):
        agent = make_agent(STREAMING=False)

        run_repl(monkeypatch, agent, ["one", "two", "/undo", "/exit"], store=store)

        [info] = store.sessions()
        assert info.turns == 1

    def test_undoing_the_only_turn_leaves_a_session_with_nothing_in_it(
        self, monkeypatch, make_agent, store
    ):
        run_repl(
            monkeypatch,
            make_agent(STREAMING=False),
            ["one", "/undo", "/exit"],
            store=store,
        )

        [info] = store.sessions()
        assert info.turns == 0 and store.load(info.id).items == []

    def test_reset_starts_a_new_session_and_keeps_the_old_one(
        self, monkeypatch, make_agent, store
    ):
        run_repl(
            monkeypatch,
            make_agent(STREAMING=False),
            ["first", "/reset", "second", "/exit"],
            store=store,
        )

        assert sorted(s.title for s in store.sessions()) == ["first", "second"]

    def test_a_save_that_fails_is_told_and_the_conversation_goes_on(
        self, monkeypatch, make_agent, store, capsys
    ):
        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(store, "save", fail)

        run_repl(
            monkeypatch,
            make_agent(STREAMING=False),
            ["one", "two", "/exit"],
            store=store,
        )

        out = capsys.readouterr().out
        assert out.count("[Could not save the session: disk full]") == 2
        assert out.count(TEXT) == 2

    def test_a_resumed_session_is_the_conversation_and_goes_on_being_saved(
        self, monkeypatch, make_agent, fake_server, store, capsys
    ):
        run_repl(
            monkeypatch, make_agent(STREAMING=False), ["hello", "/exit"], store=store
        )
        saved = store.latest()
        capsys.readouterr()

        run_repl(
            monkeypatch,
            make_agent(STREAMING=False),
            ["again", "/exit"],
            store=store,
            resume=saved,
        )

        out = capsys.readouterr().out
        assert 'Resumed "hello": 1 turn, saved just now' in out
        sent = fake_server.requests[-1]["input"]
        assert [m["content"] for m in sent] == ["hello", TEXT, "again"]
        [info] = store.sessions()  # the same session, not another
        assert info.turns == 2 and info.title == "hello"

    def test_a_resumed_session_that_nothing_is_added_to_is_not_written_again(
        self, monkeypatch, make_agent, store
    ):
        run_repl(
            monkeypatch, make_agent(STREAMING=False), ["hello", "/exit"], store=store
        )
        saved = store.latest()
        assert saved is not None
        before = (store.directory / f"{saved.info.id}.jsonl").stat().st_mtime_ns

        run_repl(
            monkeypatch, make_agent(), ["/help", "/exit"], store=store, resume=saved
        )

        assert (store.directory / f"{saved.info.id}.jsonl").stat().st_mtime_ns == before


class TestHowSeriousANoticeIs:
    """A failure or a warning has its own colour, where there is one."""

    YELLOW = "\x1b[0;33m"
    RED = "\x1b[0;31m"

    def run(self, monkeypatch, make_agent, lines, agent=None):
        written = io.StringIO()
        console = Console(
            output=Vt100_Output(written, lambda: Size(24, 100), term="xterm-256color")
        )
        typed = iter(lines)
        monkeypatch.setattr("builtins.input", lambda _prompt: next(typed))
        Repl(agent or make_agent(), COMMANDS, console=console).run()
        return written.getvalue()

    def test_a_mistyped_command_is_a_warning(self, monkeypatch, make_agent):
        shown = self.run(monkeypatch, make_agent, ["/nope", "/exit"])

        assert f"{self.YELLOW}[Unknown command: /nope." in shown

    def test_a_response_that_failed_is_an_error(self, monkeypatch, make_agent):
        shown = self.run(monkeypatch, make_agent, ["failed", "/exit"])

        assert f"{self.RED}[Response failed: invalid_prompt: boom]" in shown

    def test_a_level_a_command_asks_for_is_used(self, monkeypatch, make_agent):
        shown = self.run(monkeypatch, make_agent, ["/effort extreme", "/exit"])

        assert f"{self.YELLOW}[Unknown effort: extreme." in shown

    def test_a_command_that_failed_is_an_error(self, monkeypatch, make_agent):
        def fails(self, notice, focus=None):
            raise TurnError("the summary was empty")

        monkeypatch.setattr(Agent, "compact", fails)

        shown = self.run(monkeypatch, make_agent, ["/compact", "/exit"])

        assert f"{self.RED}[Could not summarize: the summary was empty]" in shown

    def test_the_todo_command_shows_the_checklist_styled(self, monkeypatch, make_agent):
        agent = make_agent()
        agent.todos.tool().call(
            '{"todos": [{"content": "a", "status": "in_progress"},'
            ' {"content": "b", "status": "completed"}]}'
        )

        shown = self.run(monkeypatch, make_agent, ["/todo", "/exit"], agent)

        assert "Todos 1/2 done" in shown
        assert "\x1b[0;1m[>] a" in shown and "\x1b[0;2m[x] b" in shown

    def test_a_routine_notice_is_neither(self, monkeypatch, make_agent):
        shown = self.run(monkeypatch, make_agent, ["/reset", "/exit"])

        assert self.YELLOW not in shown and self.RED not in shown
