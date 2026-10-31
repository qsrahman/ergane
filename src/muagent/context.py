"""The conversation history, and how it is trimmed to fit a token budget."""

import json

from openai.types.responses import ResponseInputParam


class Context:
    """Conversation history, resent in full each turn because requests use store=False.

    A turn is staged with draft() and only becomes history through commit(), so
    failed requests never leave a user message without a reply.
    """

    def __init__(self) -> None:
        """Create an empty conversation."""
        self.items: ResponseInputParam = []
        self.last_prompt: str | None = None  # the prompt most recently drafted
        self.last_saved = False  # whether that prompt's turn is in items

    def draft(self, prompt: str, max_tokens: int) -> tuple[ResponseInputParam, bool]:
        """Start a turn: build the next request's input within a token budget.

        The history is not changed; only the prompt is remembered (as not saved
        yet) for rewind(). Old turns are dropped from the front, a whole turn at a
        time, until the input fits the budget. The new user message is always kept.

        Args:
            prompt: The user's new message.
            max_tokens: The most tokens the input should have (an estimate). Passed
                on every call, so a changed setting takes effect at once.

        Returns:
            The history plus the new message, trimmed to the budget, and whether
            any old turns were dropped.
        """
        self.last_prompt = prompt
        self.last_saved = False
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
            if total <= max_tokens:
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
        self.last_saved = True

    def reset(self) -> None:
        """Forget the whole history."""
        self.items = []
        self.last_saved = False

    def rewind(self) -> str | None:
        """Get ready to send the last prompt again: drop its turn if it was saved.

        Returns:
            The last prompt, or None if there has not been one.
        """
        if self.last_saved:
            self.undo()
        return self.last_prompt

    def undo(self) -> bool:
        """Drop the last turn: the last user message and everything after it.

        Returns:
            Whether there was a turn to drop.
        """
        for i in range(len(self.items) - 1, -1, -1):
            item = self.items[i]
            if isinstance(item, dict) and item.get("role") == "user":
                self.items = self.items[:i]
                self.last_saved = False
                return True
        return False
