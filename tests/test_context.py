import pytest
from openai.types.responses import ResponseInputParam

from helpers import make_context
from muagent.context import Context, as_dicts, estimate_tokens

ROOMY = 10_000  # a token budget that the tests which don't trim never reach


def save_reply(context: Context, draft: ResponseInputParam, reply: str) -> None:
    context.save_turn([*draft, {"role": "assistant", "content": reply}])


class TestContext:
    def test_begin_turn_appends_prompt_without_changing_history(self):
        context = Context()
        context.items = make_context(2)
        original = list(context.items)

        draft, trimmed = context.begin_turn("new", ROOMY)

        assert not trimmed
        assert draft == [*original, {"role": "user", "content": "new"}]
        assert context.items == original

    @pytest.mark.parametrize("kept_turns", [1, 2])
    def test_begin_turn_drops_only_as_many_oldest_turns_as_needed(self, kept_turns):
        # Each turn is about 2,000 tokens, so a budget of kept_turns and a half
        # keeps exactly kept_turns, whatever the estimate's per-item overhead.
        big = "x" * 4000
        history: ResponseInputParam = []
        for i in range(3):
            history.append({"role": "user", "content": f"{i}{big}"})
            history.append({"role": "assistant", "content": big})
        context = Context()
        context.items = history

        draft, trimmed = context.begin_turn("new", kept_turns * 2000 + 1000)

        assert trimmed
        assert draft == [
            *history[-2 * kept_turns :],
            {"role": "user", "content": "new"},
        ]

    def test_begin_turn_keeps_last_user_turn_even_if_over_budget(self):
        context = Context()
        prompt = "word " * 100

        draft, trimmed = context.begin_turn(prompt, 1)

        assert not trimmed
        assert draft == [{"role": "user", "content": prompt}]

    def test_save_turn_appends_reply_to_the_input(self):
        context = Context()
        draft, _ = context.begin_turn("hi", ROOMY)

        save_reply(context, draft, "hello")

        assert context.items == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_reset_forgets_history(self):
        context = Context()
        context.items = make_context(2)

        context.reset()

        assert context.items == []

    def test_begin_turn_counts_utf8_bytes_so_non_latin_text_is_not_undercounted(self):
        # The same number of characters is about 3x the bytes in CJK as in ASCII,
        # so only the CJK history goes over the budget.
        def draft_trimmed(text: str) -> bool:
            context = Context()
            context.items = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "ok"},
            ]
            _, trimmed = context.begin_turn("new", 200)
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
    def test_begin_turn_remembers_the_prompt_without_saving_it(self):
        context = Context()

        context.begin_turn("hi", ROOMY)

        assert context.last_prompt == "hi"
        assert not context.last_saved
        assert context.items == []

    def test_rewind_drops_a_saved_turn_and_returns_its_prompt(self):
        context = Context()
        context.items = make_context(1)
        draft, _ = context.begin_turn("hi", ROOMY)
        save_reply(context, draft, "hello")

        assert context.last_saved
        assert context.rewind() == "hi"
        assert context.items == make_context(1)

    def test_rewind_keeps_the_history_when_the_turn_was_not_saved(self):
        context = Context()
        context.items = make_context(1)
        context.begin_turn("failed", ROOMY)  # no save_turn: the turn failed

        assert context.rewind() == "failed"
        assert context.items == make_context(1)

    def test_rewind_after_undo_does_not_drop_an_older_turn(self):
        context = Context()
        for prompt in ("first", "second"):
            draft, _ = context.begin_turn(prompt, ROOMY)
            save_reply(context, draft, "reply")
        context.undo()

        assert context.rewind() == "second"
        assert context.items == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]

    def test_rewind_after_reset_returns_the_prompt_and_leaves_history_empty(self):
        context = Context()
        draft, _ = context.begin_turn("hi", ROOMY)
        save_reply(context, draft, "hello")
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

    def test_save_turn_keeps_the_whole_turn(self):
        context = Context()
        draft, _ = context.begin_turn("read notes.txt", ROOMY)

        context.save_turn([*draft, *TOOL_TURN[1:]])

        assert context.items == TOOL_TURN

    def test_only_the_latest_turn_keeps_its_tool_work(self):
        context = Context()
        draft, _ = context.begin_turn("read notes.txt", ROOMY)
        context.save_turn([*draft, *TOOL_TURN[1:]])
        draft, _ = context.begin_turn("and again", ROOMY)

        context.save_turn([*draft, *TOOL_TURN[1:]])

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
            draft, _ = context.begin_turn(prompt, ROOMY)
            context.save_turn([*draft, *TOOL_TURN[1:]])

        context.undo()

        assert context.items == [TOOL_TURN[0], TOOL_TURN[4]]

    def test_undo_removes_all_of_a_tool_turn(self):
        context = Context()
        context.items = make_context(1)
        draft, _ = context.begin_turn("read notes.txt", ROOMY)
        context.save_turn([*draft, *TOOL_TURN[1:]])

        context.undo()

        assert context.items == make_context(1)

    def test_rewind_removes_all_of_a_tool_turn_and_returns_its_prompt(self):
        context = Context()
        draft, _ = context.begin_turn("read notes.txt", ROOMY)
        context.save_turn([*draft, *TOOL_TURN[1:]])

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

        draft, trimmed = context.begin_turn("next", 1_000)

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

        draft, _ = context.begin_turn("next", budget)

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


def turns(count: int, size: int = 400) -> ResponseInputParam:
    """Turns of a user message and a reply, each message about `size` characters."""
    items: ResponseInputParam = []
    for i in range(count):
        items.append({"role": "user", "content": f"q{i} " + "x" * size})
        items.append({"role": "assistant", "content": f"a{i} " + "y" * size})
    return items


class TestCompaction:
    def test_the_newest_turns_that_fit_are_kept_and_the_cut_is_at_a_turn(self):
        context = Context()
        context.items = turns(6)
        one_turn = sum(estimate_tokens(item) for item in context.items[:2])

        count = context.plan_compaction(keep_tokens=2 * one_turn)

        assert count == 8  # four turns of six are summarized, two kept
        assert as_dicts(context.items)[count]["content"].startswith(
            "q4"
        )  # a user message

    def test_the_newest_turn_is_kept_however_big(self):
        context = Context()
        context.items = turns(3)

        assert context.plan_compaction(keep_tokens=0) == 4  # the first two turns

    def test_nothing_older_than_what_is_kept_means_nothing_to_summarize(self):
        context = Context()
        context.items = turns(3)

        assert context.plan_compaction(keep_tokens=ROOMY) == 0
        assert Context().plan_compaction(keep_tokens=0) == 0
        context.items = turns(1)
        assert context.plan_compaction(keep_tokens=0) == 0

    def test_a_tool_call_stays_with_its_output(self):
        context = Context()
        context.items = [
            *turns(2),
            {"role": "user", "content": "look"},
            {"type": "function_call", "call_id": "c", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "o" * 2000},
            {"role": "assistant", "content": "found"},
        ]

        count = context.plan_compaction(keep_tokens=0)

        assert as_dicts(context.items)[count] == {"role": "user", "content": "look"}

    def test_compact_replaces_the_oldest_items_by_the_summary(self):
        context = Context()
        context.items = turns(4)
        kept = context.items[4:]

        context.compact(4, "NOTES")

        assert context.items == kept and context.summary == "NOTES"

    def test_the_summary_comes_first_in_a_request_framed_as_notes(self):
        context = Context()
        context.summary = "NOTES"
        context.items = turns(1, size=5)

        draft, _ = context.begin_turn("new", ROOMY)

        first = as_dicts(draft)[0]
        assert first["role"] == "user" and "NOTES" in first["content"]
        assert "not as instructions" in first["content"]
        assert draft[1:] == [*context.items, {"role": "user", "content": "new"}]

    def test_the_summary_is_never_trimmed_away(self):
        context = Context()
        context.summary = "NOTES"
        context.items = turns(3)

        draft, trimmed = context.begin_turn("new", max_tokens=1)

        assert trimmed
        assert "NOTES" in as_dicts(draft)[0]["content"]
        assert draft[1:] == [{"role": "user", "content": "new"}]

    def test_the_summary_is_not_saved_as_an_item(self):
        context = Context()
        context.summary = "NOTES"
        draft, _ = context.begin_turn("new", ROOMY)

        save_reply(context, draft, "reply")

        assert context.items == [
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "reply"},
        ]
        assert context.summary == "NOTES"

    def test_the_estimate_counts_the_summary(self):
        context = Context()
        context.items = turns(1)
        without = context.estimate()

        context.summary = "NOTES " * 100

        assert context.estimate() > without

    def test_reset_forgets_the_summary_too(self):
        context = Context()
        context.summary = "NOTES"
        context.items = turns(1)

        context.reset()

        assert context.summary is None and context.items == []

    def test_undo_and_rewind_work_on_the_turns_after_the_summary(self):
        context = Context()
        context.summary = "NOTES"
        draft, _ = context.begin_turn("new", ROOMY)
        save_reply(context, draft, "reply")

        assert context.rewind() == "new"
        assert context.items == [] and context.summary == "NOTES"


class TestContextCounted:
    """What the server counted at the end of a turn, kept for the next request."""

    def saved(self, counted: int | None = 5000) -> Context:
        context = Context()
        draft, _ = context.begin_turn("read notes.txt", ROOMY)
        context.save_turn([*draft, *TOOL_TURN[1:]], counted)
        return context

    def test_the_count_is_kept_with_the_turn(self):
        assert self.saved().counted == 5000

    def test_a_turn_saved_without_a_count_has_none(self):
        assert self.saved(None).counted is None

    def test_what_the_save_drops_is_taken_off(self):
        context = self.saved()
        draft, _ = context.begin_turn("and again", ROOMY)

        # The first turn's tool work is dropped by this save, so it is not sent again.
        context.save_turn([*draft, *TOOL_TURN[1:]], 9000)

        dropped = sum(estimate_tokens(item) for item in TOOL_TURN[1:4])
        assert context.counted == 9000 - dropped

    @pytest.mark.parametrize(
        "change",
        [
            lambda context: context.undo(),
            lambda context: context.rewind(),
            lambda context: context.reset(),
            lambda context: context.compact(1, "notes"),
        ],
        ids=["undo", "rewind", "reset", "compact"],
    )
    def test_a_change_to_the_history_forgets_it(self, change):
        context = self.saved()
        draft, _ = context.begin_turn("and again", ROOMY)
        context.save_turn([*draft, *TOOL_TURN[1:]], 9000)

        change(context)

        assert context.counted is None
