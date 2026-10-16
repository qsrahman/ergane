import json
import platform
import sys
from datetime import date
from functools import cache
from pathlib import Path

from openai import OpenAI
from openai.types.responses import ResponseInputParam
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


@cache
def get_config() -> Config:
    """Load and validate the agent configuration on first use.

    Returns:
        The validated Config object.

    Raises:
        SystemExit: If the configuration is invalid.
    """
    try:
        return Config()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        print("Configuration error:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        raise SystemExit(1) from None


def count_item_tokens(item: object) -> int:
    """Estimate the number of tokens in one conversation item.

    Args:
        item: A single conversation context item.

    Returns:
        The estimated number of tokens in the JSON-serialized item.
    """
    # ~4 bytes per token on average; tokenizer-agnostic, so it suits any backend.
    # UTF-8 bytes rather than characters keep non-Latin text from being undercounted.
    return len(json.dumps(item, ensure_ascii=False).encode()) // 4 + 1


def trim_context(
    context: ResponseInputParam,
    max_tokens: int,
) -> bool:
    """Trim the conversation context to fit within the maximum token limit.

    Args:
        context: The current conversation context.
        max_tokens: The maximum allowed number of tokens.

    Returns:
        True if the context was trimmed, False otherwise.
    """
    counts = [count_item_tokens(item) for item in context]
    total = sum(counts)
    user_turn_starts = [
        i
        for i, item in enumerate(context)
        if isinstance(item, dict) and item.get("role") == "user"
    ]

    # Drop whole turns from the front, but never the last user turn.
    start = 0
    for turn_start in user_turn_starts[1:]:
        if total <= max_tokens:
            break
        total -= sum(counts[start:turn_start])
        start = turn_start

    del context[:start]
    return start > 0


def run_turn(
    client: OpenAI,
    context: ResponseInputParam,
    prompt: str,
) -> None:
    """Run a single turn of interaction with the AI assistant.

    Args:
        client: The OpenAI client instance.
        context: The current conversation context.
        prompt: The user's input prompt.
    """
    config = get_config()
    instructions = (
        f"{SYSTEM_PROMPT}\n"
        "Environment:\n"
        f"- Operating system: {platform.platform()}\n"
        f"- Current date: {date.today().isoformat()}\n"
    )

    # Create a copy of the context for this turn to avoid mutating the main
    # context if the API call fails.
    turn_context: ResponseInputParam = [*context, {"role": "user", "content": prompt}]
    if trim_context(turn_context, config.max_context_tokens):
        print("[Context trimmed: oldest turns dropped]")
    print()

    request = {
        "store": False,
        "input": turn_context,
        "model": config.model,
        "instructions": instructions,
    }
    if config.streaming:
        response = None
        with client.responses.stream(**request) as stream:
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
    else:
        response = client.responses.create(**request)

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

    # Commit only after a completed response so failed requests don't corrupt history.
    context[:] = turn_context
    context.append(
        {
            "role": "assistant",
            "content": response.output_text,
        }
    )
    if not config.streaming:
        print(response.output_text)


def main() -> None:
    """Entry point for the AI coding agent CLI."""
    config = get_config()
    context: ResponseInputParam = []

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )

    print(f"AI coding agent | model: {config.model}")
    print("Commands: /reset (clear history), /clear (clear screen), /exit or /quit.\n")

    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            break

        prompt = prompt.strip()
        if not prompt:
            continue
        command = prompt.lower()
        if command in {"/exit", "/quit"}:
            break
        if command == "/reset":
            context.clear()
            print("[History reset]\n")
            continue
        if command == "/clear":
            # Clear the screen and scrollback, then move the cursor home, like `clear`.
            print("\033[2J\033[3J\033[H", end="", flush=True)
            continue

        try:
            run_turn(client, context, prompt)
        except KeyboardInterrupt:
            print()  # end the partially streamed line
        except Exception as exc:  # noqa: BLE001
            print(f"Agent error: {type(exc).__name__}: {exc}")
        print()
    print("Bye.")


if __name__ == "__main__":
    main()
