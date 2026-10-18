import json
import platform
import sys
from collections.abc import Callable
from datetime import date

from openai import OpenAI
from openai.types.responses import Response, ResponseInputParam
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import readline  # noqa: F401  # line editing and history for input()
except ImportError:  # not available on Windows
    pass

SYSTEM_PROMPT = """
You are an expert, fed up and sassy AI assistant who hates answering questions.
Operating rules:
- Be concise and direct.
- Write math in Unicode (∑ ∫ √ π α β θ ± × ÷ ≤ ≥ ≈ ∈ ∞, x² x₁, a/b for fractions).
- NEVER use LaTeX. This terminal doesn't render `$...$`, `\frac{}{}`, or other LaTeX/KaTeX syntax; it shows literal backslashes and dollar signs instead.
"""


class Config(BaseSettings):
    """Configuration for the AI agent."""

    model_config = SettingsConfigDict(
        extra="ignore",
        # Relative to the current directory, so it works for installed packages too.
        env_file=".env",
    )
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


class TurnError(Exception):
    """A response that did not complete with text; the history is unchanged."""


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
    ) -> bool:
        """Send one user message, passing the reply text to on_text.

        The turn is added to the conversation only if the response completes
        with text.

        Args:
            prompt: The user's input prompt.
            on_text: Called with each piece of reply text as it arrives when
                streaming, or once with the whole reply otherwise.

        Returns:
            Whether old turns were dropped to fit the token budget.

        Raises:
            TurnError: If the response is incomplete, failed, refused or has no
                text, or if a stream ends without a final response.
        """
        draft, trimmed = self.context.draft(prompt)
        if self.config.streaming:
            response = None
            with self.client.responses.stream(
                store=False,
                input=draft,
                model=self.config.model,
                instructions=self._instructions(),
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
                instructions=self._instructions(),
            )

        self._check(response)
        self.context.commit(draft, response.output_text)
        if not self.config.streaming:
            on_text(response.output_text)
        return trimmed

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
        if response.status != "completed":
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
            if refusal:
                raise TurnError(f"Refused: {refusal}")
            raise TurnError("No text in response")

    def _instructions(self) -> str:
        """Build the system instructions, including details of the environment."""
        return (
            f"{SYSTEM_PROMPT}\n"
            "Environment:\n"
            f"- Operating system: {platform.platform()}\n"
            f"- Current date: {date.today().isoformat()}\n"
        )


class Repl:
    """Interactive terminal loop: reads prompts, runs commands and turns."""

    def __init__(self, agent: Agent) -> None:
        """Create a REPL for an agent.

        Args:
            agent: The agent that answers prompts.
        """
        self.agent = agent
        self.running = False
        # Each command maps to its handler and the description shown by /help.
        self.commands: dict[str, tuple[Callable[[], None], str]] = {
            "/clear": (self.clear, "clear the screen"),
            "/exit": (self.stop, "quit"),
            "/help": (self.help, "list commands"),
            "/quit": (self.stop, "quit"),
            "/reset": (self.reset, "clear the conversation history"),
        }

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

            command = self.commands.get(prompt.lower())
            if command:
                handler, _ = command
                handler()
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
        wrote_text = False

        def write(text: str) -> None:
            nonlocal wrote_text
            wrote_text = True
            print(text, end="", flush=True)

        notice = None
        try:
            if self.agent.run_turn(prompt, on_text=write):
                notice = "[Context trimmed: oldest turns dropped]"
        except TurnError as exc:
            notice = f"[{exc}]"
        except KeyboardInterrupt:
            notice = "[Cancelled]"
        except Exception as exc:  # noqa: BLE001
            notice = f"Agent error: {type(exc).__name__}: {exc}"

        if wrote_text:
            print()  # end the reply line
        if notice:
            print(notice)
        print()

    def help(self) -> None:
        """List the available commands."""
        width = max(len(name) for name in self.commands)
        for name, (_, description) in self.commands.items():
            print(f"  {name:<{width}}  {description}")
        print()

    def reset(self) -> None:
        """Clear the conversation history."""
        self.agent.context.reset()
        print("[History reset]\n")

    def clear(self) -> None:
        """Clear the terminal screen; the history is kept."""
        # Clear the screen and scrollback, then move the cursor home, like
        # `clear`; not when output is redirected, where the codes are garbage.
        if sys.stdout.isatty():
            print("\033[2J\033[3J\033[H", end="", flush=True)

    def stop(self) -> None:
        """Leave the loop after the current command."""
        self.running = False


def main() -> None:
    """Entry point for the AI coding agent CLI.

    Raises:
        SystemExit: If the configuration is invalid.
    """
    try:
        config = Config()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None

    Repl(Agent(config)).run()


if __name__ == "__main__":
    main()
