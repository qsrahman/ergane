from pathlib import Path

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import default_ui_style, merge_styles

from ergane.agent import Agent
from ergane.commands import COMMANDS
from ergane.mcp_hub import McpHub, ServerStatus
from ergane.repl import Repl
from ergane.terminal import (  # pyright: ignore[reportPrivateUsage]
    _Menu,
    choose,
    terminal_session,
    toolbar,
)
from ergane.tools import Tool
from helpers import anchor, hanging_config, make_config, make_context


class TestToolbar:
    """The line under the prompt: the model, the effort and the context."""

    def toolbar(self, tmp_path, **config) -> str:
        agent = Agent(make_config(**config))
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None
            Repl(agent, COMMANDS, session)
            assert callable(session.bottom_toolbar)
            return fragment_list_to_text(to_formatted_text(session.bottom_toolbar()))

    def test_it_names_the_model_and_the_budget(self, tmp_path):
        text = self.toolbar(tmp_path, model="m1", max_context_tokens=10000)

        assert text.startswith(" m1  ")
        assert text.endswith("% of 10,000")
        assert "~" in text  # nothing has been counted by the server yet
        assert "·" not in text

    def test_it_names_the_effort_when_there_is_one(self, tmp_path):
        assert " m1 · high  " in self.toolbar(
            tmp_path, model="m1", reasoning_effort="high"
        )

    def test_the_count_of_the_server_is_not_marked_as_an_estimate(self):
        agent = Agent(make_config(max_context_tokens=10000))
        anchor(agent, 2500)

        text = fragment_list_to_text(toolbar(agent, McpHub({}, Path())))

        assert text.endswith(" 25% of 10,000") and "~" not in text

    @pytest.mark.parametrize(
        ("counted", "lit", "style"),
        [
            (0, 0, "class:toolbar.meter"),
            (2500, 2, "class:toolbar.meter"),
            (5999, 5, "class:toolbar.meter"),
            (6000, 6, "class:toolbar.warn"),
            (8500, 8, "class:toolbar.bad"),
            (30000, 10, "class:toolbar.bad"),  # over the budget: the meter is full
        ],
    )
    def test_the_meter_fills_and_turns_yellow_then_red(self, counted, lit, style):
        agent = Agent(make_config(max_context_tokens=10000))
        anchor(agent, counted)

        bar = toolbar(agent, McpHub({}, Path()))

        gap = next(i for i, fragment in enumerate(bar) if fragment[1] == "  ")
        assert bar[gap + 1] == (style, "▰" * lit)  # the meter follows the gap
        assert bar[gap + 2] == ("", "▱" * (10 - lit))


class TestToolbarAndServers:
    @pytest.fixture
    def hub(self, tmp_path: Path):
        hub = McpHub({}, tmp_path)
        yield hub
        hub.close()

    def toolbar(self, hub: McpHub) -> str:
        bar = toolbar(Agent(make_config()), hub)
        return fragment_list_to_text(bar)

    def test_it_says_while_servers_start(self, tmp_path):
        hub = McpHub({"slow": hanging_config()}, tmp_path)
        try:
            hub.begin()

            assert " · starting MCP servers  " in self.toolbar(hub)
        finally:
            hub.close()

    def test_it_counts_the_tools_and_the_servers_that_failed(self, hub):
        statuses = [
            ServerStatus("a", "running", tools=("a__x", "a__y")),
            ServerStatus("b", "running", tools=("b__z",)),
            ServerStatus("c", "failed", "boom"),
        ]
        hub.statuses = {s.name: s for s in statuses}

        assert " · 3 tools · 1 MCP failed: /servers  " in self.toolbar(hub)

    def test_it_says_nothing_of_servers_when_there_are_none(self, hub):
        assert "MCP" not in self.toolbar(hub) and "tools" not in self.toolbar(hub)


class TestMenu:
    """The question before a tool runs, as a menu on the session's terminal."""

    OPTIONS = (("n", "Deny"), ("y", "Allow once"), ("a", "Always allow save"))

    def choose(self, tmp_path, keys: str) -> str:
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None
            pipe.send_text(keys)
            return choose(session, "Allow save?", self.OPTIONS)

    @pytest.mark.parametrize(
        ("keys", "chosen"),
        [
            ("\r", "n"),  # Enter takes the first, which is the default
            ("\x1b[B\r", "y"),
            ("\x1b[B\x1b[B\r", "a"),
            ("\x1b[B\x1b[B\x1b[A\r", "y"),
            ("3\r", "a"),
            ("y", "y"),
            ("a", "a"),
            ("n", "n"),
        ],
    )
    def test_arrows_numbers_and_keys_choose(self, tmp_path, keys, chosen):
        assert self.choose(tmp_path, keys) == chosen

    def test_ctrl_c_is_a_keyboard_interrupt_like_at_the_prompt(self, tmp_path):
        with pytest.raises(KeyboardInterrupt):
            self.choose(tmp_path, "\x03")

    def test_the_menu_is_erased_when_it_is_answered(self):
        app = _Menu(message="m", options=self.OPTIONS)._create_application()  # pyright: ignore[reportPrivateUsage]

        assert app.erase_when_done is True

    def test_a_repl_at_a_terminal_asks_with_the_menu_and_one_without_types(
        self, tmp_path
    ):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None

            at_terminal = Repl(Agent(make_config()), COMMANDS, session)

        assert at_terminal.console.choose is not None
        assert Repl(Agent(make_config()), COMMANDS).console.choose is None


class TestTerminalSession:
    """The line editor, driven through a pipe the way prompt_toolkit's docs test it."""

    @staticmethod
    def read(tmp_path, typed: str, conversation: bool = False) -> tuple[list[str], str]:
        """Run a Repl whose session reads `typed`; return the prompts and the history."""
        agent = Agent(make_config())
        if conversation:
            agent.context.items = make_context(1)
        prompts: list[str] = []
        agent.run_turn = lambda prompt, **_: prompts.append(prompt)  # pyright: ignore[reportAttributeAccessIssue]
        history = tmp_path / "state" / "history"
        with create_pipe_input() as pipe:
            pipe.send_text(typed)
            session = terminal_session(
                COMMANDS, history, input=pipe, output=DummyOutput()
            )
            Repl(agent, COMMANDS, session).run()
        return prompts, history.read_text() if history.exists() else ""

    def test_ctrl_c_clears_what_was_typed_and_does_not_quit(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "abc\x03def\r/exit\r")

        assert prompts == ["def"]
        assert "Press Ctrl-C again" not in capsys.readouterr().out

    def test_ctrl_d_quits_at_once_when_there_is_no_conversation(self, tmp_path, capsys):
        self.read(tmp_path, "\x04")

        out = capsys.readouterr().out
        assert "Press Ctrl-D again" not in out and out.endswith("Bye.\n")

    def test_ctrl_d_in_a_conversation_asks_for_a_second_press(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "\x04x\r/exit\r", conversation=True)

        assert prompts == ["x"]
        assert "[Press Ctrl-D again, or type /exit, to quit]" in capsys.readouterr().out

    def test_two_ctrl_d_in_a_row_quit(self, tmp_path, capsys):
        self.read(tmp_path, "\x04\x04", conversation=True)

        assert capsys.readouterr().out.endswith("Bye.\n")

    def test_ctrl_c_on_an_empty_line_asks_for_a_second_press(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "\x03x\r/exit\r")

        assert prompts == ["x"]
        assert "Press Ctrl-C again" in capsys.readouterr().out

    def test_two_ctrl_c_in_a_row_on_an_empty_line_quit(self, tmp_path, capsys):
        prompts, _ = self.read(tmp_path, "\x03\x03")

        assert prompts == [] and capsys.readouterr().out.endswith("Bye.\n")

    def test_help_lists_the_keys_of_the_editor(self, tmp_path, capsys):
        self.read(tmp_path, "/help\r/exit\r")

        out = capsys.readouterr().out
        assert "  Alt-Enter, Ctrl-J  Add a line to the prompt" in out
        assert "Ctrl-C             Clear the line; on an empty line, press twice" in out
        assert '"Allow …?": y allows, a allows the tool for the' in out

    def test_help_lists_the_keys_in_order(self, tmp_path, capsys):
        self.read(tmp_path, "/help\r/exit\r")

        out = capsys.readouterr().out
        keys = [
            "Alt-Enter",
            "Ctrl-C",
            "Ctrl-D",
            "Ctrl-R",
            "Ctrl-X",
            "Tab, Up",
            "y, a, ?",
        ]
        positions = [out.index(f"  {key}") for key in keys]
        assert positions == sorted(positions)

    def test_a_line_added_by_hand_has_a_marker_under_the_prompt(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )

        assert session is not None and callable(session.prompt_continuation)
        marker = session.prompt_continuation

        def shown(width: int, wraps: int) -> str:
            return fragment_list_to_text(to_formatted_text(marker(width, 1, wraps)))

        assert shown(2, 0) == "· "
        assert shown(4, 0) == "  · "  # lined up under the end of the prompt
        assert shown(2, 1) == "  "  # a line that only wraps has none

    def test_the_menu_does_not_keep_the_black_on_grey_of_prompt_toolkit(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )

        assert session is not None and session.style is not None
        style = merge_styles([default_ui_style(), session.style])
        for item in (
            "class:completion-menu.completion",
            "class:completion-menu.meta.completion",
            "class:completion-menu.completion.current",
        ):
            attrs = style.get_attrs_for_style_str(item)
            # Black text on the terminal's own background could not be read on a dark one.
            assert attrs.color != "000000", item
            assert attrs.bgcolor in ("", "default"), item  # not the grey of the default

    def test_a_pasted_block_is_one_prompt(self, tmp_path):
        paste = "\x1b[200~line one\nline two\x1b[201~\r"

        prompts, _ = self.read(tmp_path, paste + "/exit\r")

        assert prompts == ["line one\nline two"]

    def test_prompts_are_kept_in_the_history_file(self, tmp_path):
        _, history = self.read(tmp_path, "first\rsecond\r/exit\r")

        assert "+first" in history and "+second" in history

    def test_the_slash_commands_are_completed(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.completer is not None
            found = session.completer.get_completions(Document("/re"), None)  # pyright: ignore[reportArgumentType]

            assert sorted(c.text for c in found) == ["/reconnect", "/reset", "/retry"]

    def test_the_commands_are_offered_in_order(self, tmp_path):
        # As the menu shows them, not sorted afterwards: they are defined in another order.
        assert self.complete(tmp_path, "/", ordered=True) == sorted(COMMANDS)

    def test_a_command_is_completed_with_what_it_does(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.completer is not None
            found = list(session.completer.get_completions(Document("/und"), None))  # pyright: ignore[reportArgumentType]

        assert [c.text for c in found] == ["/undo"]
        assert (
            fragment_list_to_text(to_formatted_text(found[0].display_meta))
            == COMMANDS["/undo"].description
        )

    def test_the_editor_and_the_hint_are_on(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )

        assert session is not None
        assert session.enable_open_in_editor is True
        assert session.enable_suspend is True
        [(style, hint, *_)] = to_formatted_text(session.placeholder)
        assert "/help" in hint
        # The hint is dim: the class is on the text, and the style makes it dim.
        assert session.style is not None
        assert session.style.get_attrs_for_style_str(style).dim is True

    def test_redirected_input_is_read_with_plain_input(self, tmp_path):
        # Under pytest stdin is not a terminal.
        assert terminal_session(COMMANDS, tmp_path / "state" / "history") is None
        assert not (tmp_path / "state").exists()

    @pytest.mark.parametrize(
        ("typed", "prompt"),
        [
            ("one\x1b\rtwo\r", "one\ntwo"),  # Alt-Enter: Esc, then Enter
            ("one\ntwo\r", "one\ntwo"),  # Ctrl-J, a line feed
            ("one\x1b[13;2utwo\r", "one\ntwo"),  # Shift-Enter, kitty protocol
            ("one\x1b[27;2;13~two\r", "one\ntwo"),  # Shift-Enter, modifyOtherKeys
        ],
        ids=["alt-enter", "ctrl-j", "shift-enter-csi-u", "shift-enter-xterm"],
    )
    def test_a_line_can_be_added_by_hand_and_enter_sends(self, tmp_path, typed, prompt):
        prompts, _ = self.read(tmp_path, typed + "/exit\r")

        assert prompts == [prompt]

    def complete(self, tmp_path, text: str, ordered: bool = False) -> list[str]:
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.completer is not None
            found = session.completer.get_completions(Document(text), None)  # pyright: ignore[reportArgumentType]
            return [c.text for c in found] if ordered else sorted(c.text for c in found)

    def test_the_servers_are_completed_after_reconnect(self, tmp_path):
        hub = McpHub({}, tmp_path)
        statuses = [
            ServerStatus("web", "running"),
            ServerStatus("bad", "failed", "boom"),
            ServerStatus("old", "off", "enabled is false"),
        ]
        hub.statuses = {s.name: s for s in statuses}
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.completer is not None
            Repl(Agent(make_config()), COMMANDS, session, hub=hub)

            def found(text: str) -> list[str]:
                words = session.completer.get_completions(Document(text), None)  # pyright: ignore[reportOptionalMemberAccess, reportArgumentType]
                return [c.text for c in words]

            assert found("/reconnect ") == ["web", "bad"]  # not the one that is off
            assert found("/reconnect b") == ["bad"]

    def test_no_server_is_offered_when_there_are_none(self, tmp_path):
        assert self.complete(tmp_path, "/reconnect ") == []  # no Repl is using it
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.completer is not None
            Repl(Agent(make_config()), COMMANDS, session)  # a Repl without servers

            words = session.completer.get_completions(Document("/reconnect "), None)  # pyright: ignore[reportArgumentType]
            assert list(words) == []

    def test_the_levels_are_completed_after_effort(self, tmp_path):
        assert self.complete(tmp_path, "/effort h") == ["high"]
        assert "default" in self.complete(tmp_path, "/effort ")

    def test_a_command_that_lists_no_arguments_completes_none(self, tmp_path):
        assert self.complete(tmp_path, "/model ") == []
        assert self.complete(tmp_path, "hello") == []

    def test_an_earlier_prompt_is_suggested(self, tmp_path):
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, tmp_path / "history", input=pipe, output=DummyOutput()
            )
            assert session is not None and session.auto_suggest is not None
            session.default_buffer.history.append_string("hello there")

            suggestion = session.auto_suggest.get_suggestion(
                session.default_buffer, Document("hel")
            )

            assert suggestion is not None and suggestion.text == "lo there"

    def test_the_history_is_for_this_user_only(self, tmp_path):
        history = tmp_path / "history"
        with create_pipe_input() as pipe:
            terminal_session(COMMANDS, history, input=pipe, output=DummyOutput())

        assert history.stat().st_mode & 0o777 == 0o600

    def test_a_history_file_that_cannot_be_made_is_not_an_error(self, tmp_path):
        blocked = tmp_path / "a-file"
        blocked.write_text("")
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, blocked / "history", input=pipe, output=DummyOutput()
            )

        assert session is not None  # the prompts are kept in memory instead

    def test_an_answer_to_a_question_is_not_a_prompt(self, tmp_path):
        history = tmp_path / "history"
        agent = Agent(make_config())
        with create_pipe_input() as pipe:
            session = terminal_session(
                COMMANDS, history, input=pipe, output=DummyOutput()
            )
            repl = Repl(agent, COMMANDS, session)
            pipe.send_text("y\r")

            def danger(path: str) -> str:
                """Do something."""
                return ""

            answer = repl.console.confirm(
                Tool.from_function(danger, read_only=False), "{}"
            )

        assert answer is True
        assert history.read_text() == ""  # "y" is not a prompt to recall
