"""The interactive terminal loop: reads input, runs commands, shows replies."""

from collections.abc import Mapping

from muagent.agent import Agent, TurnError
from muagent.commands import Command
from muagent.console import Console

try:
    import readline  # noqa: F401
except ImportError:  # not available on Windows
    pass


class Repl:
    """Interactive terminal loop: reads prompts, runs commands and turns."""

    def __init__(self, agent: Agent, commands: Mapping[str, Command]) -> None:
        """Create a REPL for an agent.

        Args:
            agent: The agent that answers prompts.
            commands: The slash commands, by name.
        """
        self.agent = agent
        self.console = Console()
        self.running = False
        self.commands = commands

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        tools = len(self.agent.tools)
        count = f" | {tools} tool{'s' if tools != 1 else ''}" if tools else ""
        self.console.write(
            f"muagent | model: {self.agent.config.model}{count} | /help lists commands\n\n"
        )

        self.running = True
        while self.running:
            try:
                prompt = input("> ").strip()
            except KeyboardInterrupt, EOFError:
                break
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

    def ask(self, prompt: str) -> None:
        """Run one turn, printing the reply as it arrives and reporting
        problems without leaving the loop.

        Args:
            prompt: The user's input prompt.
        """
        console = self.console
        console.blank()
        try:
            self.agent.run_turn(
                prompt,
                on_text=console.write,
                on_notice=console.notice,
                confirm=console.confirm,
            )
        except TurnError as exc:
            console.notice(str(exc))
        except KeyboardInterrupt:
            console.notice("Cancelled")
        except Exception as exc:  # noqa: BLE001
            console.notice(f"Agent error: {type(exc).__name__}: {exc}")
        console.end_line()
        console.blank()
