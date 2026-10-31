import pytest
from openai.types.responses import ResponseInputParam

from helpers import make_context
from muagent.context import Context


class TestContext:
    def test_draft_appends_prompt_without_changing_history(self):
        context = Context(max_tokens=10_000)
        context.items = make_context(2)
        original = list(context.items)

        draft, trimmed = context.draft("new")

        assert not trimmed
        assert draft == [*original, {"role": "user", "content": "new"}]
        assert context.items == original

    @pytest.mark.parametrize("kept_turns", [1, 2])
    def test_draft_drops_only_as_many_oldest_turns_as_needed(self, kept_turns):
        # Each turn is about 2,000 tokens, so a budget of kept_turns and a half
        # keeps exactly kept_turns, whatever the estimate's per-item overhead.
        big = "x" * 4000
        history: ResponseInputParam = []
        for i in range(3):
            history.append({"role": "user", "content": f"{i}{big}"})
            history.append({"role": "assistant", "content": big})
        context = Context(max_tokens=kept_turns * 2000 + 1000)
        context.items = history

        draft, trimmed = context.draft("new")

        assert trimmed
        assert draft == [
            *history[-2 * kept_turns :],
            {"role": "user", "content": "new"},
        ]

    def test_draft_keeps_last_user_turn_even_if_over_budget(self):
        context = Context(max_tokens=1)
        prompt = "word " * 100

        draft, trimmed = context.draft(prompt)

        assert not trimmed
        assert draft == [{"role": "user", "content": prompt}]

    def test_commit_appends_reply_to_draft(self):
        context = Context(max_tokens=10_000)
        draft, _ = context.draft("hi")

        context.commit(draft, "hello")

        assert context.items == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_reset_forgets_history(self):
        context = Context(max_tokens=10_000)
        context.items = make_context(2)

        context.reset()

        assert context.items == []

    def test_draft_counts_utf8_bytes_so_non_latin_text_is_not_undercounted(self):
        # The same number of characters is about 3x the bytes in CJK as in ASCII,
        # so only the CJK history goes over the budget.
        def draft_trimmed(text: str) -> bool:
            context = Context(max_tokens=200)
            context.items = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "ok"},
            ]
            _, trimmed = context.draft("new")
            return trimmed

        assert not draft_trimmed("a" * 400)
        assert draft_trimmed("字" * 400)


class TestContextUndo:
    def test_drops_the_last_turn(self):
        context = Context(max_tokens=10_000)
        context.items = make_context(2)
        first_turn = context.items[:2]

        assert context.undo()
        assert context.items == first_turn

    def test_reports_when_there_is_nothing_to_undo(self):
        context = Context(max_tokens=10_000)

        assert not context.undo()
        assert context.items == []


class TestContextRewind:
    def test_draft_remembers_the_prompt_without_saving_it(self):
        context = Context(max_tokens=10_000)

        context.draft("hi")

        assert context.last_prompt == "hi"
        assert not context.last_saved
        assert context.items == []

    def test_rewind_drops_a_saved_turn_and_returns_its_prompt(self):
        context = Context(max_tokens=10_000)
        context.items = make_context(1)
        draft, _ = context.draft("hi")
        context.commit(draft, "hello")

        assert context.last_saved
        assert context.rewind() == "hi"
        assert context.items == make_context(1)

    def test_rewind_keeps_the_history_when_the_turn_was_not_saved(self):
        context = Context(max_tokens=10_000)
        context.items = make_context(1)
        context.draft("failed")  # no commit: the turn failed

        assert context.rewind() == "failed"
        assert context.items == make_context(1)

    def test_rewind_after_undo_does_not_drop_an_older_turn(self):
        context = Context(max_tokens=10_000)
        for prompt in ("first", "second"):
            draft, _ = context.draft(prompt)
            context.commit(draft, "reply")
        context.undo()

        assert context.rewind() == "second"
        assert context.items == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]

    def test_rewind_after_reset_returns_the_prompt_and_leaves_history_empty(self):
        context = Context(max_tokens=10_000)
        draft, _ = context.draft("hi")
        context.commit(draft, "hello")
        context.reset()

        assert context.rewind() == "hi"
        assert context.items == []

    def test_rewind_with_no_prompt_returns_none(self):
        assert Context(max_tokens=10_000).rewind() is None
