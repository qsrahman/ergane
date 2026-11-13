"""Summarizing the oldest turns when the conversation outgrows its budget.

`Compactor` decides when (past `compact_at` of the budget), plans what (the oldest
turns, keeping the newest as they are), asks the model for the summary through the
function it is given, and changes the history only if the summary is good. A failure
never stops a turn: the turns that do not fit are dropped as before, which is the
worst that happens. The design follows research into opencode, Codex, OpenClaw, pi,
Qwen Code and DeepSeek Harness. Not done: a separate summary model, a summary request
that shares the chat's cache, splitting one oversized turn, and a stored transcript
with undo.
"""

from collections.abc import Callable
from typing import Protocol

from openai.types.responses import ResponseInputParam

from muagent.context import Context, extract_summary, transcript
from muagent.errors import TurnError, describe_error

# The summary of old turns may have this many tokens.
SUMMARY_MAX_TOKENS = 8000

# Summaries that failed one after another, after which none is tried by itself.
COMPACT_FAILURES = 3


class CompactionSettings(Protocol):
    """The settings summarizing reads, each where it is used, so a change takes effect
    at once. `Config` has them all."""

    auto_compact: bool
    compact_at: float
    max_context_tokens: int
    keep_recent_tokens: int


class Compactor:
    """Summarizes the oldest turns of a conversation."""

    def __init__(
        self,
        context: Context,
        settings: CompactionSettings,
        ask: Callable[[ResponseInputParam, Callable[[str], None]], str],
    ) -> None:
        """Create a compactor for a conversation.

        Args:
            context: The conversation, which `compact` changes.
            settings: What to read the limits from.
            ask: Sends a request to the model and returns its answer, given the
                request's input and where to report that a request is sent again.
                It raises `TurnError` if the answer is unusable.
        """
        self.context = context
        self.settings = settings
        self._ask = ask
        # Summaries that failed one after another (see COMPACT_FAILURES).
        self.failures = 0

    def tail_budget(self) -> int:
        """How many tokens of the newest turns a summary leaves as they are: the
        setting, but no more than half of where summarizing starts, or a small
        budget would keep everything."""
        settings = self.settings
        trigger = int(settings.compact_at * settings.max_context_tokens)
        return min(settings.keep_recent_tokens, trigger // 2)

    def compact_if_due(
        self, next_request: int, on_notice: Callable[[str], None]
    ) -> None:
        """Summarize the oldest turns when the next request would pass `compact_at`
        of the budget, unless the setting is off or summaries kept failing.

        A failure is told and counted, and does not stop the turn.

        Args:
            next_request: The tokens the next request would have.
            on_notice: Called with the lines `compact` says, and with a failure.
        """
        settings = self.settings
        if not settings.auto_compact or self.failures >= COMPACT_FAILURES:
            return
        if next_request <= settings.compact_at * settings.max_context_tokens:
            return
        try:
            self.compact(on_notice)
        except Exception as exc:  # noqa: BLE001
            self.failures += 1
            reason = str(exc) if isinstance(exc, TurnError) else describe_error(exc)
            paused = (
                "; summaries are paused until /compact works"
                if self.failures >= COMPACT_FAILURES
                else ""
            )
            on_notice(f"Could not summarize earlier messages: {reason}{paused}")

    def compact(
        self, on_notice: Callable[[str], None], focus: str | None = None
    ) -> bool:
        """Replace the oldest turns by a summary, keeping the newest as they are.

        The summary takes in the one there was, so a conversation keeps one. Nothing
        changes unless the summary is good: the history is the same after a failure.

        Args:
            on_notice: Called with a line before the model is asked and one after.
            focus: What the summary should take particular care of, if the user said.

        Returns:
            False if there is nothing older than what is kept; True if the oldest turns
            were replaced.

        Raises:
            TurnError: If the model's summary is unusable (see `extract_summary`).
            Exception: Whatever the API client raises.
            KeyboardInterrupt: On Ctrl-C, with the history as it was.
        """
        context = self.context
        count = context.plan_compaction(self.tail_budget())
        if count == 0:
            return False
        older = context.items[:count]
        before = context.estimate()
        on_notice("Summarizing earlier messages…")

        parts = []
        if context.summary:
            parts.append(f"Previous summary:\n{context.summary}")
        parts.append(f"Conversation:\n{transcript(older)}")
        if focus:
            parts.append(f"Focus: {focus[:800]}")
        draft: ResponseInputParam = [{"role": "user", "content": "\n\n".join(parts)}]
        summary = extract_summary(self._ask(draft, on_notice), "\n".join(parts))

        turns = sum(
            1 for item in older if isinstance(item, dict) and item.get("role") == "user"
        )
        context.compact(count, summary)
        self.failures = 0
        on_notice(
            f"Summarized {turns} earlier turn{'s' if turns != 1 else ''}: "
            f"about {before:,} to {context.estimate():,} tokens of history"
        )
        return True
