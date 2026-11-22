import pytest

from ergane.loops import (
    MAX_REPEATS,
    WARN_REPEATS,
    WINDOW,
    is_looping,
    last_repeats,
    loop_note,
    step_signature,
)


class TestStepSignature:
    def test_the_same_calls_with_the_same_results_give_the_same_signature(self):
        calls = [("read", '{"a": 1}', "text")]

        assert step_signature(calls) == step_signature(list(calls))

    @pytest.mark.parametrize(
        "other",
        [
            [("write", '{"a": 1}', "text")],  # another tool
            [("read", '{"a": 2}', "text")],  # other arguments
            [("read", '{"a": 1}', "changed")],  # another result: progress
            [("read", '{"a": 1}', "text"), ("read", '{"a": 1}', "text")],  # more
        ],
    )
    def test_a_difference_in_what_was_asked_or_returned_gives_another(self, other):
        assert step_signature([("read", '{"a": 1}', "text")]) != step_signature(other)

    def test_moving_text_between_parts_does_not_make_the_same_step(self):
        assert step_signature([("ab", "c", "")]) != step_signature([("a", "bc", "")])


class TestIsLooping:
    def test_a_step_may_come_back_up_to_the_limit(self):
        assert not is_looping(["a"] * MAX_REPEATS)

    def test_one_more_is_a_loop(self):
        assert is_looping(["a"] * (MAX_REPEATS + 1))

    def test_repeats_that_are_spread_out_count(self):
        steps = ["a", "b"] * MAX_REPEATS + ["a"]

        assert is_looping(steps[-WINDOW:]) == is_looping(steps)
        assert is_looping(["a", "b", "c"] * 3 + ["a"]) is False

    def test_a_step_that_left_the_window_does_not_count(self):
        steps = ["a"] * (MAX_REPEATS + 1) + [str(i) for i in range(WINDOW)]

        assert not is_looping(steps)

    def test_no_steps_is_no_loop(self):
        assert not is_looping([])


class TestLastRepeats:
    @pytest.mark.parametrize(
        ("steps", "want"),
        [
            ([], 0),
            (["a", "a", "a", "b"], 1),  # a new step: the model changed course
            (["a", "a", "a"], 3),
            (["a", "b", "a", "c", "a"], 3),
            (["a"] * 5 + ["x"] * (WINDOW - 1) + ["a"], 1),  # the old ones left
        ],
    )
    def test_it_counts_the_newest_step_in_the_window(self, steps, want):
        assert last_repeats(steps) == want

    def test_the_model_is_warned_before_it_is_stopped(self):
        assert WARN_REPEATS < MAX_REPEATS


def test_the_notice_says_how_often():
    note = loop_note(4)

    assert "4 times" in note and "different approach" in note
