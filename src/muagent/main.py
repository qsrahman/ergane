import json
import os
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import get_args

from openai import Omit, OpenAI, omit
from openai.types.responses import Response, ResponseInputParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import Reasoning
from pydantic import Field, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import readline  # noqa: F401  # line editing and history for input()
except ImportError:  # not available on Windows
    pass

SYSTEM_PROMPT = """
You are an expert AI coding agent running in the user's terminal.
You help with software engineering: writing, reading, debugging and explaining code.
Operating rules:
- Be concise and direct.
- You cannot read files or run commands yet; ask the user to paste what you need.
- Write math in Unicode (∑ ∫ √ π α β θ ± × ÷ ≤ ≥ ≈ ∈ ∞, x² x₁, a/b for fractions), never LaTeX — this terminal doesn't render `$...$`, `\\frac{}{}`, or other LaTeX/KaTeX syntax; it shows literal backslashes and dollar signs instead.
"""


class Config(BaseSettings):
    """Configuration for the AI agent."""

    # Config files may hold keys that aren't settings; ignore them, don't fail.
    model_config = SettingsConfigDict(extra="ignore")
    model: str = Field(
        min_length=1,
        default="gpt-5.4-mini",
        validation_alias="OPENAI_MODEL",
    )
    api_key: str = Field(
        min_length=1,
        validation_alias="OPENAI_API_KEY",
    )
    base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    max_context_tokens: int = Field(
        default=64 * 1024,
        gt=0,
        validation_alias="MAX_CONTEXT_TOKENS",
    )
    streaming: bool = Field(
        default=True,
        validation_alias="STREAMING",
    )
    max_turns: int = Field(
        default=25,
        gt=0,
        validation_alias="MAX_TURNS",
    )
    # None sends no reasoning setting, so the model's default applies.
    # Non-reasoning models reject any effort with HTTP 400.
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        validation_alias="REASONING_EFFORT",
    )


def load_config() -> Config:
    """Load the settings.

    Environment variables take precedence, then a `.env` in the current
    directory, then the user-level `$XDG_CONFIG_HOME/muagent/.env` (by default
    `~/.config/muagent/.env`). Missing files are skipped.

    Raises:
        ValidationError: If a setting is missing or invalid.
    """
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", ""))
    if not config_home.is_absolute():
        config_home = Path.home() / ".config"
    user_file = config_home / "muagent" / ".env"

    # Later files override earlier ones.
    return Config(_env_file=(user_file, ".env"))  # pyright: ignore[reportCallIssue]


class Context:
    """Conversation history, resent in full each turn because requests use store=False.

    A turn is staged with draft() and only becomes history through commit(), so
    failed requests never leave a user message without a reply.
    """

    def __init__(self, max_tokens: int) -> None:
        """Create an empty conversation.

        Args:
            max_tokens: Token budget for the input sent each turn.
        """
        self.max_tokens = max_tokens
        self.items: ResponseInputParam = []

    def draft(self, prompt: str) -> tuple[ResponseInputParam, bool]:
        """Build the next request's input without changing the history.

        Old turns are dropped from the front, a whole turn at a time, until the
        input fits the token budget. The new user message is always kept.

        Args:
            prompt: The user's new message.

        Returns:
            The history plus the new message, trimmed to the token budget, and
            whether any old turns were dropped.
        """
        items: ResponseInputParam = [*self.items, {"role": "user", "content": prompt}]
        # ~4 bytes of UTF-8 per token on average: tokenizer-agnostic, so it suits any
        # backend, and counting bytes keeps non-Latin text from being undercounted.
        counts = [
            len(json.dumps(item, ensure_ascii=False).encode()) // 4 + 1
            for item in items
        ]
        total = sum(counts)
        user_turn_starts = [
            i
            for i, item in enumerate(items)
            if isinstance(item, dict) and item.get("role") == "user"
        ]

        start = 0
        for turn_start in user_turn_starts[1:]:
            if total <= self.max_tokens:
                break
            total -= sum(counts[start:turn_start])
            start = turn_start
        return items[start:], start > 0

    def commit(self, draft: ResponseInputParam, reply: str) -> None:
        """Make a drafted turn and its reply the new history.

        Args:
            draft: The input returned by draft() for this turn.
            reply: The assistant's reply text.
        """
        self.items = [*draft, {"role": "assistant", "content": reply}]

    def reset(self) -> None:
        """Forget the whole history."""
        self.items = []

    def undo(self) -> bool:
        """Drop the last turn: the last user message and everything after it.

        Returns:
            Whether there was a turn to drop.
        """
        for i in range(len(self.items) - 1, -1, -1):
            item = self.items[i]
            if isinstance(item, dict) and item.get("role") == "user":
                self.items = self.items[:i]
                return True
        return False


class TurnError(Exception):
    """A turn that ended without a final answer; completed steps stay in the history."""


class Agent:
    """Sends conversation turns to the model. Does no terminal I/O of its own."""

    def __init__(self, config: Config, client: OpenAI | None = None) -> None:
        """Create an agent with an empty conversation.

        Args:
            config: The agent configuration.
            client: The API client; built from the config if omitted.
        """
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.context = Context(config.max_context_tokens)

    def run_turn(
        self,
        prompt: str,
        on_text: Callable[[str], None] = lambda text: None,
        on_notice: Callable[[str], None] = lambda text: None,
    ) -> None:
        """Send one user message, passing the reply text to on_text.

        The turn is added to the conversation if the response completes with
        text, or ends incomplete after some text, which the user has seen.

        Args:
            prompt: The user's input prompt.
            on_text: Called with each piece of reply text as it arrives when
                streaming, or once with the whole reply otherwise.
            on_notice: Called with status messages, such as old turns being
                dropped to fit the token budget.

        Raises:
            TurnError: If the response is incomplete, failed, refused or has no
                text, or if a stream ends without a final response.
        """
        draft, trimmed = self.context.draft(prompt)

        if trimmed:
            on_notice("Context trimmed: oldest turns dropped")

        instructions = self._instructions()
        # omit leaves reasoning out of the request; non-reasoning models reject it.
        effort = self.config.reasoning_effort
        reasoning: Reasoning | Omit = omit if effort is None else {"effort": effort}

        if self.config.streaming:
            response = None
            with self.client.responses.stream(
                store=False,
                input=draft,
                model=self.config.model,
                instructions=instructions,
                reasoning=reasoning,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        on_text(event.delta)
                    # Take the final response from the terminal event:
                    # get_final_response() raises unless the stream ends with
                    # response.completed.
                    elif (
                        event.type == "response.completed"
                        or event.type == "response.incomplete"
                        or event.type == "response.failed"
                    ):
                        response = event.response
            if response is None:
                raise TurnError("Stream ended without a final response")
        else:
            response = self.client.responses.create(
                store=False,
                input=draft,
                model=self.config.model,
                instructions=instructions,
                reasoning=reasoning,
            )
            # Show the text before the status check, as streaming does.
            on_text(response.output_text)

        if response.status == "incomplete" and response.output_text:
            # Keep the partial reply the user has already seen.
            self.context.commit(draft, response.output_text)
        self._check(response)
        self.context.commit(draft, response.output_text)

    @staticmethod
    def _check(response: Response) -> None:
        """Make sure a response completed with text.

        Args:
            response: The final response of a turn.

        Raises:
            TurnError: If the response is incomplete, failed, refused or has no text.
        """
        if response.status == "incomplete":
            details = response.incomplete_details
            reason = details.reason if details else "unknown"
            raise TurnError(f"Response incomplete: {reason}")
        if response.status == "failed":
            error = response.error
            message = f"{error.code}: {error.message}" if error else "unknown error"
            raise TurnError(f"Response failed: {message}")
        # status is optional in the SDK; some compatible endpoints omit it.
        if response.status not in (None, "completed"):
            raise TurnError(f"Response not completed: {response.status}")
        # Committing a turn without a reply would leave two user messages in a row.
        if not response.output_text:
            refusal = "".join(
                part.refusal
                for item in response.output
                if item.type == "message"
                for part in item.content
                if part.type == "refusal"
            )
            raise TurnError(f"Refused: {refusal}" if refusal else "No text in response")

    def _instructions(self) -> str:
        """Build the system instructions, including details of the environment."""
        return (
            f"{SYSTEM_PROMPT}\n"
            "Environment:\n"
            f"- Operating system: {platform.platform()}\n"
            f"- Current date: {date.today().isoformat()}\n"
        )


# Control characters that could manipulate the terminal (escape sequences such as
# OSC 52 clipboard writes, cursor movement, BEL, carriage return) and Unicode bidi
# overrides and isolates that can make text display misleadingly. Newline and tab
# are kept. Removing ESC alone defuses an escape sequence, so this works on any
# chunk of a stream, even one that splits a sequence in two.
_UNSAFE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def sanitize(text: str) -> str:
    """Remove characters that could manipulate the terminal or spoof its text.

    Args:
        text: Text from the model or the server.

    Returns:
        The text without control characters (except newline and tab) and without
        bidi overrides and isolates.
    """
    return _UNSAFE.sub("", text)


@dataclass(frozen=True)
class Command:
    """A slash command: what it does and the description /help shows."""

    handler: Handler
    description: str


COMMANDS: dict[str, Command] = {}


class Repl:
    """Interactive terminal loop: reads prompts, runs commands and turns."""

    def __init__(self, agent: Agent) -> None:
        """Create a REPL for an agent.

        Args:
            agent: The agent that answers prompts.
        """
        self.agent = agent
        self.running = False
        self.at_line_start = True  # whether the cursor is at the start of a line
        self.commands = COMMANDS
        self.last_prompt: str | None = None  # the prompt /retry resends
        self.last_saved = False  # whether that prompt's turn is in the history

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        print(f"AI coding agent | model: {self.agent.config.model}")
        print("Type /help to list commands.\n")

        self.running = True
        while self.running:
            try:
                prompt = input("> ").strip()
            except KeyboardInterrupt, EOFError:
                break
            if not prompt:
                continue

            # The first word picks the command; the rest is its argument.
            name, _, args = prompt.partition(" ")
            command = self.commands.get(name.lower())
            if command:
                command.handler(self, args.strip())
            # A single word starting with "/" is a mistyped command, not a
            # prompt; longer input such as "/usr/bin is missing" still goes
            # to the model.
            elif prompt.startswith("/") and " " not in prompt:
                print(f"[Unknown command: {prompt}. Type /help to list commands.]\n")
            else:
                self.ask(prompt)
        print("Bye.")

    def ask(self, prompt: str) -> None:
        """Run one turn, printing the reply as it arrives and reporting
        problems without leaving the loop.

        Args:
            prompt: The user's input prompt.
        """
        print()
        self.last_prompt = prompt
        history = self.agent.context.items
        try:
            self.agent.run_turn(prompt, on_text=self.write, on_notice=self.notice)
        except TurnError as exc:
            self.notice(str(exc))
        except KeyboardInterrupt:
            self.notice("Cancelled")
        except Exception as exc:  # noqa: BLE001
            self.notice(f"Agent error: {type(exc).__name__}: {exc}")
        finally:
            # commit() replaces the list, so a new list means the turn was saved,
            # even when it then raised (an incomplete reply is kept).
            self.last_saved = self.agent.context.items is not history

        if not self.at_line_start:
            self.write("\n")  # end the reply line
        print()

    def write(self, text: str) -> None:
        """Print reply text, tracking whether the cursor is at a line start.

        Args:
            text: A piece of the reply.
        """
        text = sanitize(text)
        print(text, end="", flush=True)
        if text:
            self.at_line_start = text.endswith("\n")

    def notice(self, text: str) -> None:
        """Print a bracketed status line, starting a new line if needed.

        Args:
            text: The status message.
        """
        if not self.at_line_start:
            self.write("\n")
        self.write(f"[{text}]\n")


# A command handler gets the REPL and the text after the command name (or "").
Handler = Callable[[Repl, str], None]


def command(name: str, description: str) -> Callable[[Handler], Handler]:
    """Register a handler as a slash command. Stack it to give a command aliases.

    Args:
        name: The command as typed, lowercase, including the slash.
        description: The one-line description shown by /help.

    Returns:
        A decorator that registers the handler and returns it unchanged.

    Raises:
        ValueError: If the name is already registered.
    """

    def register(handler: Handler) -> Handler:
        if name in COMMANDS:
            raise ValueError(f"Command already registered: {name}")
        COMMANDS[name] = Command(handler, description)
        return handler

    return register


@command("/help", "list commands")
def help_(repl: Repl, args: str) -> None:
    """List the available commands."""
    width = max(len(name) for name in repl.commands)
    for name, cmd in sorted(repl.commands.items()):
        print(f"  {name:<{width}}  {cmd.description}")
    print()


@command("/reset", "clear the conversation history")
def reset(repl: Repl, args: str) -> None:
    """Clear the conversation history."""
    repl.agent.context.reset()
    print("[History reset]\n")


@command("/clear", "clear the screen")
def clear(repl: Repl, args: str) -> None:
    """Clear the terminal screen; the history is kept."""
    # Clear the screen and scrollback, then move the cursor home, like
    # `clear`; not when output is redirected, where the codes are garbage.
    if sys.stdout.isatty():
        print("\033[2J\033[3J\033[H", end="", flush=True)


@command("/exit", "quit")
@command("/quit", "quit")
def stop(repl: Repl, args: str) -> None:
    """Leave the loop after the current command."""
    repl.running = False


@command("/undo", "remove the last exchange from the history")
def undo(repl: Repl, args: str) -> None:
    """Remove the last user message and its reply from the history."""
    if repl.agent.context.undo():
        repl.last_saved = False
        print("[Last exchange removed]\n")
    else:
        print("[Nothing to undo]\n")


@command("/retry", "resend the last prompt, replacing its reply")
def retry(repl: Repl, args: str) -> None:
    """Resend the last prompt, first removing its exchange if it was saved."""
    if repl.last_prompt is None:
        print("[Nothing to retry]\n")
        return
    if repl.last_saved:
        repl.agent.context.undo()
    repl.ask(repl.last_prompt)


@command("/model", "show the model, or switch with /model NAME")
def model(repl: Repl, args: str) -> None:
    """Show the model, or switch to the one named in args."""
    if args:
        repl.agent.config.model = args
    print(f"[Model: {repl.agent.config.model}]\n")


# ReasoningEffort is Optional[Literal[...]]; these are the Literal's values.
EFFORTS: tuple[str, ...] = get_args(get_args(ReasoningEffort)[0])


@command("/effort", "show the reasoning effort, or set it with /effort LEVEL")
def effort(repl: Repl, args: str) -> None:
    """Show the reasoning effort, or set it; "default" sends none at all."""
    config = repl.agent.config
    if args.lower() == "default":
        config.reasoning_effort = None
    elif args:
        try:
            config.reasoning_effort = TypeAdapter(ReasoningEffort).validate_python(
                args.lower()
            )
        except ValidationError:
            print(f"[Unknown effort: {args}. Use {', '.join(EFFORTS)} or default.]\n")
            return
    print(f"[Reasoning effort: {config.reasoning_effort or 'model default'}]\n")


def main() -> None:
    """Entry point for the AI coding agent CLI.

    Raises:
        SystemExit: If the configuration is invalid.
    """
    try:
        config = load_config()
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None

    Repl(Agent(config)).run()


if __name__ == "__main__":
    main()
