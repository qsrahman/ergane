import pytest
from openai.types.responses import ResponseInputParam

from helpers import make_context
from muagent.context import Context, estimate_tokens

ROOMY = 10_000  # a token budget that the tests which don't trim never reach


def commit_reply(context: Context, draft: ResponseInputParam, reply: str) -> None:
    context.commit([*draft, {"role": "assistant", "content": reply}])


class TestContext:
    def test_draft_appends_prompt_without_changing_history(self):
        context = Context()
        context.items = make_context(2)
        original = list(context.items)

        draft, trimmed = context.draft("new", ROOMY)

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
        context = Context()
        context.items = history

        draft, trimmed = context.draft("new", kept_turns * 2000 + 1000)

        assert trimmed
        assert draft == [
            *history[-2 * kept_turns :],
            {"role": "user", "content": "new"},
        ]

    def test_draft_keeps_last_user_turn_even_if_over_budget(self):
        context = Context()
        prompt = "word " * 100

        draft, trimmed = context.draft(prompt, 1)

        assert not trimmed
        assert draft == [{"role": "user", "content": prompt}]

    def test_commit_appends_reply_to_draft(self):
        context = Context()
        draft, _ = context.draft("hi", ROOMY)

        commit_reply(context, draft, "hello")

        assert context.items == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_reset_forgets_history(self):
        context = Context()
        context.items = make_context(2)

        context.reset()

        assert context.items == []

    def test_draft_counts_utf8_bytes_so_non_latin_text_is_not_undercounted(self):
        # The same number of characters is about 3x the bytes in CJK as in ASCII,
        # so only the CJK history goes over the budget.
        def draft_trimmed(text: str) -> bool:
            context = Context()
            context.items = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "ok"},
            ]
            _, trimmed = context.draft("new", 200)
            return trimmed

        assert not draft_trimmed("a" * 400)
        assert draft_trimmed("字" * 400)


class TestContextUndo:
    def test_drops_the_last_turn(self):
        context = Context()
        context.items = make_context(2)
        first_turn = context.items[:2]

        assert context.undo()
        assert context.items == first_turn

    def test_reports_when_there_is_nothing_to_undo(self):
        context = Context()

        assert not context.undo()
        assert context.items == []


class TestContextRewind:
    def test_draft_remembers_the_prompt_without_saving_it(self):
        context = Context()

        context.draft("hi", ROOMY)

        assert context.last_prompt == "hi"
        assert not context.last_saved
        assert context.items == []

    def test_rewind_drops_a_saved_turn_and_returns_its_prompt(self):
        context = Context()
        context.items = make_context(1)
        draft, _ = context.draft("hi", ROOMY)
        commit_reply(context, draft, "hello")

        assert context.last_saved
        assert context.rewind() == "hi"
        assert context.items == make_context(1)

    def test_rewind_keeps_the_history_when_the_turn_was_not_saved(self):
        context = Context()
        context.items = make_context(1)
        context.draft("failed", ROOMY)  # no commit: the turn failed

        assert context.rewind() == "failed"
        assert context.items == make_context(1)

    def test_rewind_after_undo_does_not_drop_an_older_turn(self):
        context = Context()
        for prompt in ("first", "second"):
            draft, _ = context.draft(prompt, ROOMY)
            commit_reply(context, draft, "reply")
        context.undo()

        assert context.rewind() == "second"
        assert context.items == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]

    def test_rewind_after_reset_returns_the_prompt_and_leaves_history_empty(self):
        context = Context()
        draft, _ = context.draft("hi", ROOMY)
        commit_reply(context, draft, "hello")
        context.reset()

        assert context.rewind() == "hi"
        assert context.items == []

    def test_rewind_with_no_prompt_returns_none(self):
        assert Context().rewind() is None


TOOL_TURN: ResponseInputParam = [
    {"role": "user", "content": "read notes.txt"},
    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"},
    {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "c1", "output": "file contents"},
    {"role": "assistant", "content": "It says hello."},
]


class TestContextToolTurns:
    """A turn that used tools is more than a prompt and a reply."""

    def test_commit_keeps_the_whole_turn(self):
        context = Context()
        draft, _ = context.draft("read notes.txt", ROOMY)

        context.commit([*draft, *TOOL_TURN[1:]])

        assert context.items == TOOL_TURN

    def test_only_the_latest_turn_keeps_its_tool_work(self):
        context = Context()
        draft, _ = context.draft("read notes.txt", ROOMY)
        context.commit([*draft, *TOOL_TURN[1:]])
        draft, _ = context.draft("and again", ROOMY)

        context.commit([*draft, *TOOL_TURN[1:]])

        expected: ResponseInputParam = [
            TOOL_TURN[0],
            TOOL_TURN[4],  # the first turn: its messages only
            {"role": "user", "content": "and again"},
            *TOOL_TURN[1:],
        ]
        assert context.items == expected

    def test_undo_after_pruning_removes_only_the_latest_turn(self):
        context = Context()
        for prompt in ("read notes.txt", "and again"):
            draft, _ = context.draft(prompt, ROOMY)
            context.commit([*draft, *TOOL_TURN[1:]])

        context.undo()

        assert context.items == [TOOL_TURN[0], TOOL_TURN[4]]

    def test_undo_removes_all_of_a_tool_turn(self):
        context = Context()
        context.items = make_context(1)
        draft, _ = context.draft("read notes.txt", ROOMY)
        context.commit([*draft, *TOOL_TURN[1:]])

        context.undo()

        assert context.items == make_context(1)

    def test_rewind_removes_all_of_a_tool_turn_and_returns_its_prompt(self):
        context = Context()
        draft, _ = context.draft("read notes.txt", ROOMY)
        context.commit([*draft, *TOOL_TURN[1:]])

        assert context.rewind() == "read notes.txt"
        assert context.items == []

    def test_the_opaque_encrypted_content_of_a_reasoning_item_is_not_counted(self):
        # The API counts that blob as the few tokens of reasoning it holds (18 tokens
        # for 1,636 characters, measured on gpt-5.5), not as its length.
        blob = "x" * 100_000  # about 25,000 estimated tokens if it were counted
        history: ResponseInputParam = [
            {"role": "user", "content": "first"},
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": blob,
            },
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        context = Context()
        context.items = history

        draft, trimmed = context.draft("next", 1_000)

        assert not trimmed
        assert draft[:5] == history  # nothing dropped, and the blob is still sent

    # Budgets that drop every old turn, only the tool turn, or nothing.
    @pytest.mark.parametrize("budget", [1, 90, ROOMY])
    def test_trimming_never_leaves_part_of_a_tool_turn(self, budget):
        context = Context()
        context.items = [
            *TOOL_TURN,
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "welcome"},
        ]

        draft, _ = context.draft("next", budget)

        kinds = [dict(item).get("type") for item in draft]
        assert dict(draft[0]).get("role") == "user"  # never starts mid-turn
        assert ("function_call_output" in kinds) == ("function_call" in kinds)
        assert ("reasoning" in kinds) == ("function_call" in kinds)


class TestEstimateTokens:
    def test_it_grows_with_the_text(self):
        short = estimate_tokens({"role": "user", "content": "a" * 40})
        long = estimate_tokens({"role": "user", "content": "a" * 4000})

        assert long - short == 990  # 3,960 more bytes, 4 bytes to a token

    def test_non_latin_text_counts_by_its_bytes(self):
        latin = estimate_tokens({"role": "user", "content": "a" * 100})
        cjk = estimate_tokens({"role": "user", "content": "字" * 100})

        assert cjk > 2 * latin

    def test_the_encrypted_content_of_a_reasoning_item_is_not_counted(self):
        def reasoning(blob: str) -> int:
            return estimate_tokens(
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": blob,
                }
            )

        assert reasoning("x" * 100_000) == reasoning("")
