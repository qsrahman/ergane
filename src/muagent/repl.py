"""The input loop: reads prompts, dispatches slash commands, runs turns.

The line editor and its menus are `terminal.py`. With input or output redirected there
is no session and `input()` reads.
"""

from collections.abc import Mapping
from functools import partial
from pathlib import Path

from prompt_toolkit import PromptSession

from muagent.agent import Agent
from muagent.commands import Command
from muagent.console import Console
from muagent.errors import TurnError, describe_error
from muagent.mcp_hub import McpHub, ServerUpdate
from muagent.terminal import PROMPT, CommandCompleter, choose, toolbar
from muagent.tools import tool_clash


class Repl:
    """Interactive terminal loop: reads prompts, runs commands and turns."""

    def __init__(
        self,
        agent: Agent,
        commands: Mapping[str, Command],
        session: PromptSession[str] | None = None,
        console: Console | None = None,
        hub: McpHub | None = None,
    ) -> None:
        """Create a REPL for an agent.

        Args:
            agent: The agent that answers prompts.
            commands: The slash commands, by name.
            session: The line editor that reads the prompts (see
                `terminal_session`); without one, `input()` does.
            console: What prints; one that already printed (the start-up notices)
                keeps the state that separates its output from the next.
            hub: The MCP servers, which `/reconnect` starts again and whose statuses
                `/servers` lists; none if omitted.
        """
        self.agent = agent
        self.console = console or Console()
        # With no servers the log directory is never used.
        self.hub = hub if hub is not None else McpHub({}, Path())
        if session:
            # An answer to "Allow …?" is not a prompt: it is a menu of its own, so it
            # is not kept in the history or completed as a command.
            self.console.choose = partial(choose, session)
            session.bottom_toolbar = partial(toolbar, agent, self.hub)
            if isinstance(session.completer, CommandCompleter):
                session.completer.repl = self
        self.running = False
        self.commands = commands
        self.session = session

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        self.console.write(
            f"muagent · {self.agent.config.model} · /help for commands\n\n"
        )

        self.running = True
        interrupted = False  # the last prompt ended in Ctrl-C or Ctrl-D
        while self.running:
            if self.session:
                # The toolbar is redrawn while servers start, to show when they are up.
                self.session.refresh_interval = 0.5 if self.hub.starting else 0
            try:
                prompt = (
                    self.session.prompt(PROMPT) if self.session else input("> ")
                ).strip()
            except EOFError:
                # Redirected input ends here. At a terminal a stray Ctrl-D must not
                # end a conversation that lives only in memory.
                if interrupted or not self.session or not self.agent.has_conversation:
                    break
                interrupted = True
                self.console.notice("Press Ctrl-D again, or type /exit, to quit")
                continue
            except KeyboardInterrupt:
                # The conversation lives only in memory: one stray Ctrl-C must not
                # end it.
                if interrupted:
                    break
                interrupted = True
                self.console.notice("Press Ctrl-C again, or type /exit, to quit")
                continue
            interrupted = False
            if prompt:
                self.handle(prompt)
        self.console.write("Bye.\n")

    def handle(self, prompt: str) -> None:
        """Run a slash command, or send the prompt to the agent.

        Args:
            prompt: A non-empty line of input.
        """
        # The first word picks the command; the rest is its argument.
        name, _, args = prompt.partition(" ")
        self.attach(wait=False)
        command = self.commands.get(name.lower())
        if command:
            try:
                message = command.handler(self, args.strip())
            except KeyboardInterrupt:
                message = "Cancelled"
            except Exception as exc:  # noqa: BLE001
                message = f"Command error: {type(exc).__name__}: {exc}"
        # A single word starting with "/" is a mistyped command, not a prompt;
        # longer input such as "/usr/bin is missing" still goes to the model.
        elif prompt.startswith("/") and " " not in prompt:
            message = f"Unknown command: {prompt}. Type /help to list commands."
        else:
            self.ask(prompt)
            return
        if message:
            self.console.notice(message)
            self.console.blank()

    def apply(self, update: ServerUpdate) -> None:
        """Give the agent the tools and the notes of a server, and say what the hub
        said about it, if anything.

        Args:
            update: What the hub returned for the server.
        """
        skipped = self.agent.update_server(
            update.removed, update.tools, update.name, update.instructions
        )
        if update.message:
            self.console.notice(update.message)
        for tool in skipped:
            self.console.notice(tool_clash(tool))

    def attach(self, *, wait: bool) -> None:
        """Give the agent the servers that started in the background, and tell of the
        ones that failed.

        This is between inputs, never in the middle of a turn: the tools of a turn
        must not change under it.

        Args:
            wait: Whether to wait for the servers that have not answered yet. A turn
                waits (it needs the tools); a command does not.
        """
        for update in self.hub.take(self.console.notice, wait=wait):
            self.apply(update)

    def ask(self, prompt: str) -> None:
        """Run one turn, printing the reply as it arrives and reporting
        problems without leaving the loop.

        Args:
            prompt: The user's input prompt.
        """
        console = self.console
        console.blank()
        try:
            if self.hub.starting:
                console.notice("Waiting for the MCP servers…")
            self.attach(wait=True)  # the tools are needed now
            self.agent.run_turn(
                prompt,
                on_text=console.write,
                on_notice=console.notice,
                confirm=console.confirm,
                on_call=console.call,
            )
        except TurnError as exc:
            console.notice(str(exc))
        except KeyboardInterrupt:
            console.notice("Cancelled")
        except Exception as exc:  # noqa: BLE001
            console.notice(describe_error(exc))
        console.end_line()
        console.blank()
