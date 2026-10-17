import json
import platform
import sys
from datetime import date
from pathlib import Path
from typing import Self

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
        env_file=Path(__file__).parent / ".env",
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
    streaming: bool = Field(
        default=True,
        validation_alias="STREAMING",
    )
    max_context_tokens: int = Field(
        default=64 * 1024,
        gt=0,
        validation_alias="MAX_CONTEXT_TOKENS",
    )

    @classmethod
    def load(cls) -> Self:
        """Load and validate the configuration from the environment and .env.

        Returns:
            The validated Config object.

        Raises:
            SystemExit: If the configuration is invalid.
        """
        try:
            return cls()  # pyright: ignore[reportCallIssue]
        except ValidationError as exc:
            print("Configuration error:", file=sys.stderr)
            for error in exc.errors():
                field = ".".join(str(part) for part in error["loc"])
                print(f"  - {field}: {error['msg']}", file=sys.stderr)
            raise SystemExit(1) from None


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

        Args:
            prompt: The user's new message.

        Returns:
            The history plus the new message, trimmed to the token budget, and
            whether any old turns were dropped.
        """
        items: ResponseInputParam = [*self.items, {"role": "user", "content": prompt}]
        start = self._trim_start(items)
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

    def _trim_start(self, items: ResponseInputParam) -> int:
        """Find where the kept items start so they fit within the token budget.

        Whole turns are dropped from the front, but never the last user turn.

        Args:
            items: The candidate input for the next request.

        Returns:
            The index of the first item to keep.
        """
        counts = [self._count_item_tokens(item) for item in items]
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
        return start

    @staticmethod
    def _count_item_tokens(item: object) -> int:
        """Estimate the number of tokens in one conversation item.

        Args:
            item: A single conversation context item.

        Returns:
            The estimated number of tokens in the JSON-serialized item.
        """
        # ~4 bytes per token on average; tokenizer-agnostic, so it suits any backend.
        # UTF-8 bytes rather than characters keep non-Latin text from being undercounted.
        return len(json.dumps(item, ensure_ascii=False).encode()) // 4 + 1


class Agent:
    """Sends conversation turns to the model and prints the replies."""

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

    def run_turn(self, prompt: str) -> None:
        """Send one user message and print the reply.

        The turn is added to the conversation only if the response completes
        with text.

        Args:
            prompt: The user's input prompt.
        """
        draft, trimmed = self.context.draft(prompt)
        if trimmed:
            print("[Context trimmed: oldest turns dropped]")
        print()

        request = {
            "store": False,
            "input": draft,
            "model": self.config.model,
            "instructions": self._instructions(),
        }
        if self.config.streaming:
            response = self._stream(request)
        else:
            response = self.client.responses.create(**request)

        if response.status != "completed":
            if response.status == "incomplete":
                reason = (
                    response.incomplete_details.reason
                    if response.incomplete_details
                    else "unknown"
                )
                print(f"[Response incomplete: {reason}]")
            elif response.status == "failed":
                error = response.error
                message = f"{error.code}: {error.message}" if error else "unknown error"
                print(f"[Response failed: {message}]")
            else:
                print(f"[Response not completed: {response.status}]")
            return

        # Committing a turn without a reply would leave two user messages in a row.
        if not response.output_text:
            print("[No text in response]")
            return

        self.context.commit(draft, response.output_text)
        if not self.config.streaming:
            print(response.output_text)

    def _instructions(self) -> str:
        """Build the system instructions, including details of the environment."""
        return (
            f"{SYSTEM_PROMPT}\n"
            "Environment:\n"
            f"- Operating system: {platform.platform()}\n"
            f"- Current date: {date.today().isoformat()}\n"
        )

    def _stream(self, request: dict) -> Response:
        """Stream a response, printing text deltas as they arrive.

        Args:
            request: Keyword arguments for the Responses API.

        Returns:
            The final response, whatever its status.

        Raises:
            RuntimeError: If the stream ends without a final response.
        """
        response = None
        with self.client.responses.stream(**request) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                # Take the final response from the terminal event: get_final_response()
                # raises unless the stream ends with response.completed.
                elif (
                    event.type == "response.completed"
                    or event.type == "response.incomplete"
                    or event.type == "response.failed"
                ):
                    response = event.response
        print()
        if response is None:
            raise RuntimeError("Stream ended without a final response")
        return response


class Repl:
    """Interactive terminal loop: reads prompts, runs commands and turns."""

    def __init__(self, agent: Agent) -> None:
        """Create a REPL for an agent.

        Args:
            agent: The agent that answers prompts.
        """
        self.agent = agent
        self.running = False
        self.commands = {
            "/reset": self.reset,
            "/clear": self.clear,
            "/exit": self.stop,
            "/quit": self.stop,
        }

    def run(self) -> None:
        """Read and handle prompts until the user quits."""
        print(f"AI coding agent | model: {self.agent.config.model}")
        print(
            "Commands: /reset (clear history), /clear (clear screen), /exit or /quit.\n"
        )

        self.running = True
        while self.running:
            try:
                prompt = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt:
                continue

            command = self.commands.get(prompt.lower())
            if command:
                command()
            else:
                self.ask(prompt)
        print("Bye.")

    def ask(self, prompt: str) -> None:
        """Run one turn, reporting errors without leaving the loop.

        Args:
            prompt: The user's input prompt.
        """
        try:
            self.agent.run_turn(prompt)
        except KeyboardInterrupt:
            print()  # end the partially streamed line
        except Exception as exc:  # noqa: BLE001
            print(f"Agent error: {type(exc).__name__}: {exc}")
        print()

    def reset(self) -> None:
        """Clear the conversation history."""
        self.agent.context.reset()
        print("[History reset]\n")

    def clear(self) -> None:
        """Clear the terminal screen; the history is kept."""
        # Clear the screen and scrollback, then move the cursor home, like `clear`.
        print("\033[2J\033[3J\033[H", end="", flush=True)

    def stop(self) -> None:
        """Leave the loop after the current command."""
        self.running = False


def main() -> None:
    """Entry point for the AI coding agent CLI."""
    Repl(Agent(Config.load())).run()


if __name__ == "__main__":
    main()
