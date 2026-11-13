"""Summaries of old turns, shortened tool results, and the retry of a request that
was too long: the ways the context is kept inside the budget."""

from dataclasses import dataclass

import pytest

from fake_api import TEXT
from helpers import anchor, tool_with
from muagent.compaction import COMPACT_FAILURES, SUMMARY_MAX_TOKENS, Compactor
from muagent.context import PRUNE_CHARS, Context
from muagent.errors import TurnError, describe_error, is_context_overflow
from muagent.prompts import COMPACT_PROMPT


def old_turns(count: int = 10, marker: str = "") -> list:
    """`count` turns of about 200 tokens each; `marker` goes into the first prompt."""
    items = []
    for i in range(count):
        items.append(
            {"role": "user", "content": f"q{i} {marker if i == 0 else ''}" + "x" * 400}
        )
        items.append({"role": "assistant", "content": f"a{i} " + "y" * 400})
    return items


def summarizing(request: dict) -> bool:
    return request["instructions"].startswith(COMPACT_PROMPT[:20])


@pytest.mark.parametrize("streaming", [True, False])
class TestCompaction:
    """About 700 tokens of instructions, 4,000 of budget, summaries from 3,000."""

    def agent(self, make_agent, streaming, history=None, **config):
        config = {"MAX_CONTEXT_TOKENS": 3600, "KEEP_RECENT_TOKENS": 500, **config}
        agent = make_agent(STREAMING=streaming, **config)
        agent.context.items = old_turns() if history is None else history
        return agent

    def test_a_long_history_is_summarized_before_the_turn(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming)
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices[0] == "Summarizing earlier messages…"
        assert notices[1].startswith("Summarized 8 earlier turns")
        summary_request, turn_request = fake_server.requests
        assert summarizing(summary_request) and not summarizing(turn_request)
        assert "tools" not in summary_request
        assert summary_request["max_output_tokens"] == SUMMARY_MAX_TOKENS
        # The turn is sent with the summary first, then the newest turns kept as they were.
        first = turn_request["input"][0]
        assert first["role"] == "user" and "NOTES 8" in first["content"]
        assert turn_request["input"][1]["content"].startswith("q8")
        assert agent.context.summary == "## Goal\nNOTES 8"
        assert agent.context.items[0]["content"].startswith("q8")

    def test_the_servers_count_decides_when_to_summarize(
        self, make_agent, fake_server, streaming
    ):
        # The estimate is under the trigger; what the server counted is over it.
        agent = self.agent(make_agent, streaming, old_turns(8))
        anchor(agent, 3000)
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices[0] == "Summarizing earlier messages…"

    def test_a_small_count_from_the_server_keeps_the_history(
        self, make_agent, fake_server, streaming
    ):
        # The estimate is over the trigger; what the server counted is under it.
        agent = self.agent(make_agent, streaming)
        anchor(agent, 500)
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices == []
        assert len(fake_server.requests) == 1

    def test_the_summary_is_asked_for_as_labelled_text_with_tool_results_cut(
        self, make_agent, fake_server, streaming
    ):
        history = old_turns(10)
        history[1:1] = [  # the first turn used a tool
            {
                "type": "function_call",
                "call_id": "c",
                "name": "look",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c", "output": "o" * 5000},
        ]
        agent = self.agent(make_agent, streaming, history)

        agent.run_turn("go")

        text = fake_server.requests[0]["input"][0]["content"]
        assert text.startswith("Conversation:\n[User]: q0")
        assert "[Tool call]: look({})" in text
        assert "[Tool result]: " + "o" * 2000 + " [... 3,000 more characters]" in text
        assert "[Assistant]: a0" in text

    def test_a_history_under_the_trigger_is_left_alone(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, old_turns(3))
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices == [] and len(fake_server.requests) == 1
        assert agent.context.summary is None

    def test_with_summaries_off_the_history_is_sent_as_it_is_while_it_fits(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, AUTO_COMPACT=False)
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices == [] and len(fake_server.requests) == 1
        assert len(fake_server.requests[0]["input"]) == 21
        assert agent.context.summary is None

    def test_a_summary_that_fails_does_not_stop_the_turn(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, old_turns(10, "SUMMARY_FAIL"))
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices[0] == "Summarizing earlier messages…"
        assert notices[1].startswith(
            "Could not summarize earlier messages: API error 400"
        )
        assert agent.context.summary is None and agent.compactor.failures == 1
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}
        assert len(agent.context.items) == 22  # the history is whole

    def test_what_does_not_fit_is_still_dropped_when_the_summary_fails(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(
            make_agent,
            streaming,
            old_turns(10, "SUMMARY_FAIL"),
            MAX_CONTEXT_TOKENS=1800,
        )
        notices: list[str] = []

        agent.run_turn("go", on_notice=notices.append)

        assert notices[-1] == "Context trimmed: oldest turns dropped"
        assert len(fake_server.requests[-1]["input"]) < 21

    @pytest.mark.parametrize(
        ("marker", "reason"),
        [
            ("SUMMARY_EMPTY", "the summary is empty"),
            ("SUMMARY_NOHEADINGS", "it has no headings"),
            ("SUMMARY_LONG", "longer than the text it replaces"),
            ("SUMMARY_INCOMPLETE", "Response incomplete: max_output_tokens"),
        ],
    )
    def test_a_summary_that_is_no_good_is_not_kept(
        self, make_agent, streaming, marker, reason
    ):
        agent = self.agent(make_agent, streaming, old_turns(10, marker))
        before = list(agent.context.items)

        with pytest.raises(TurnError, match=reason):
            agent.compact(lambda text: None)

        assert agent.context.items == before and agent.context.summary is None

    @pytest.mark.parametrize(
        ("marker", "summary"),
        [
            ("SUMMARY_PLAIN", "## Goal\nNOTES plain"),
            ("SUMMARY_UNCLOSED", "## Goal\nNOTES unclosed"),
        ],
    )
    def test_a_good_summary_without_the_tags_is_kept(
        self, make_agent, streaming, marker, summary
    ):
        # gemma4 writes the headings but not the <summary> tags.
        agent = self.agent(make_agent, streaming, old_turns(10, marker))

        assert agent.compact(lambda text: None)

        assert agent.context.summary == summary

    def test_summaries_pause_after_repeated_failures_and_a_manual_one_resumes_them(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, old_turns(10, "SUMMARY_FAIL"))
        for _ in range(COMPACT_FAILURES + 2):
            agent.context.items = old_turns(10, "SUMMARY_FAIL")
            agent.run_turn("go")

        tried = [r for r in fake_server.requests if summarizing(r)]
        assert len(tried) == COMPACT_FAILURES  # then none by itself

        agent.context.items = old_turns(10)
        assert agent.compact(lambda text: None)
        assert agent.compactor.failures == 0

    def test_a_second_summary_updates_the_first(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming)
        agent.compact(lambda text: None)
        agent.context.items = [*agent.context.items, *old_turns(8)]

        agent.compact(lambda text: None)

        second = fake_server.requests[1]["input"][0]["content"]
        assert second.startswith("Previous summary:\n## Goal\nNOTES 8\n\nConversation:")
        # The new one replaces the old, which was an input to it: one summary, not two.
        assert (agent.context.summary or "").startswith("## Goal\nNOTES ")
        assert agent.context.summary.count("## Goal") == 1  # type: ignore[union-attr]

    def test_a_focus_is_passed_on(self, make_agent, fake_server, streaming):
        agent = self.agent(make_agent, streaming)

        agent.compact(lambda text: None, focus="the schema")

        assert fake_server.requests[0]["input"][0]["content"].endswith(
            "Focus: the schema"
        )

    def test_a_reasoning_model_is_asked_to_think_little_for_the_summary(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming, REASONING_EFFORT="high")

        agent.run_turn("go")

        summary_request, turn_request = fake_server.requests
        assert summary_request["reasoning"] == {"effort": "low"}
        assert turn_request["reasoning"] == {"effort": "high"}

    @pytest.mark.parametrize(
        ("effort", "asked"),
        [
            ("none", "none"),
            ("minimal", "minimal"),
            ("low", "low"),
            ("medium", "low"),
            ("high", "low"),
            ("xhigh", "low"),
        ],
    )
    def test_the_summary_never_asks_for_more_thinking_than_the_chat(
        self, make_agent, fake_server, streaming, effort, asked
    ):
        agent = self.agent(make_agent, streaming, REASONING_EFFORT=effort)

        agent.compact(lambda text: None)

        assert fake_server.requests[0]["reasoning"] == {"effort": asked}

    def test_the_summary_is_not_part_of_the_instructions_of_the_chat(
        self, make_agent, fake_server, streaming
    ):
        agent = self.agent(make_agent, streaming)

        agent.run_turn("go")

        assert "NOTES" not in fake_server.requests[1]["instructions"]

    def test_ctrl_c_after_the_summary_keeps_it_and_not_the_turn(
        self, make_agent, streaming
    ):
        agent = self.agent(make_agent, streaming)
        before = list(agent.context.items)

        def interrupt(text: str) -> None:
            if text.startswith("Summarized"):
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            agent.run_turn("go", on_notice=interrupt)

        # The interrupt came after the summary was made, so it was kept; the turn was not.
        assert agent.context.summary == "## Goal\nNOTES 8"
        assert all(item["content"] != "go" for item in agent.context.items)
        assert agent.context.items != before


@pytest.mark.parametrize("streaming", [True, False])
class TestTooLong:
    def test_a_request_the_server_refuses_as_too_long_is_summarized_and_sent_again(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(
            STREAMING=streaming, MAX_CONTEXT_TOKENS=4000, AUTO_COMPACT=True,
            COMPACT_AT=1, KEEP_RECENT_TOKENS=500,
        )  # fmt: skip
        agent.context.items = old_turns(10)[:12]  # under the trigger: nothing yet
        notices: list[str] = []

        agent.run_turn("overflow_once", on_notice=notices.append)

        kinds = ["summary" if summarizing(r) else "chat" for r in fake_server.requests]
        assert kinds == ["chat", "summary", "chat"]  # refused, summarized, sent again
        assert (
            "The request was too long for the model; trying again, shorter" in notices
        )
        assert agent.context.items[-1] == {"role": "assistant", "content": TEXT}

    def test_after_a_tool_the_results_of_earlier_rounds_are_shortened(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "head" + "x" * 20000 + "tail"

        agent = make_agent(tools=[tool_with(huge)], STREAMING=streaming)
        notices: list[str] = []

        # Round 1 calls the tool, round 2 is refused, round 3 is the same request,
        # but the result is shortened.
        agent.run_turn("tool_then_overflow_once", on_notice=notices.append)

        assert len(fake_server.requests) == 3
        sent = fake_server.requests[2]["input"][-1]["output"]
        assert sent.startswith("head") and sent.endswith("tail")
        assert len(sent) < PRUNE_CHARS

    def test_nothing_to_shorten_is_the_servers_error(
        self, make_agent, fake_server, streaming
    ):
        agent = make_agent(STREAMING=streaming)

        with pytest.raises(Exception) as caught:
            agent.run_turn("overflow_once")

        assert is_context_overflow(caught.value)
        assert "Lower max_context_tokens, or use /compact." in describe_error(
            caught.value
        )
        assert len(fake_server.requests) == 1

    def test_after_a_tool_it_is_tried_once_too(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "x" * 20000  # shortened the first time, dropped the second

        agent = make_agent(tools=[tool_with(huge)], STREAMING=streaming)

        with pytest.raises(Exception) as caught:
            agent.run_turn("tool_then_overflow")

        assert is_context_overflow(caught.value)
        assert len(fake_server.requests) == 3  # the call, refused, shortened, refused

    def test_it_is_tried_once_in_a_turn(self, make_agent, fake_server, streaming):
        agent = make_agent(
            STREAMING=streaming, MAX_CONTEXT_TOKENS=4000, COMPACT_AT=1,
            KEEP_RECENT_TOKENS=500,
        )  # fmt: skip
        agent.context.items = old_turns(10)[:12]

        with pytest.raises(Exception) as caught:
            agent.run_turn("overflow")

        assert is_context_overflow(caught.value)
        kinds = ["summary" if summarizing(r) else "chat" for r in fake_server.requests]
        assert kinds == ["chat", "summary", "chat"]  # and no more


class TestOverflowDetection:
    def error(self, status: int, message: str, code: str | None = None):
        import httpx2
        from openai import APIStatusError

        response = httpx2.Response(status, request=httpx2.Request("POST", "http://x"))
        return APIStatusError(
            message,
            response=response,
            body={"message": message, "code": code},
        )

    @pytest.mark.parametrize(
        ("status", "message", "code"),
        [
            (400, "anything", "context_length_exceeded"),
            (400, "This model's maximum context length is 8192 tokens", None),
            (400, "prompt is too long: 9000 tokens", None),
            (413, "request too large: token limit exceeded", None),
            (400, "input exceeds the context window", None),
        ],
    )
    def test_the_ways_servers_say_it(self, status, message, code):
        assert is_context_overflow(self.error(status, message, code))

    @pytest.mark.parametrize(
        ("status", "message"),
        [(400, "invalid tool schema"), (429, "context length"), (500, "too long")],
    )
    def test_other_errors_are_not_it(self, status, message):
        assert not is_context_overflow(self.error(status, message))


@pytest.mark.parametrize("streaming", [True, False])
class TestShorteningResults:
    def test_a_long_result_keeps_its_start_and_end_and_says_what_is_hidden(
        self, make_agent, fake_server, streaming
    ):
        def huge(path: str) -> str:
            """Read a text file."""
            return "head" + "x" * 20000 + "tail"

        agent = make_agent(
            tools=[tool_with(huge)], STREAMING=streaming,
            MAX_CONTEXT_TOKENS=9000, MAX_TURNS=3, AUTO_COMPACT=False,
        )  # fmt: skip

        with pytest.raises(TurnError, match="^Stopped after 3"):
            agent.run_turn("tool_loop")

        outputs = [
            item["output"]
            for item in fake_server.requests[2]["input"]
            if item.get("type") == "function_call_output"
        ]
        assert outputs[0].startswith("head") and outputs[0].endswith("tail")
        assert "characters hidden; call the tool again" in outputs[0]
        assert len(outputs[0]) < PRUNE_CHARS
        assert outputs[-1].startswith("head") and "hidden" not in outputs[-1]


@dataclass
class Settings:
    auto_compact: bool = True
    compact_at: float = 0.5
    max_context_tokens: int = 1000
    keep_recent_tokens: int = 100


class TestCompactor:
    """The summarizing on its own: no agent, no API, only a function that answers."""

    def compactor(self, answer=None, **settings):
        asked: list[str] = []

        def ask(draft, on_notice):
            asked.append(draft[0]["content"])
            if isinstance(answer, Exception):
                raise answer
            return answer or "## Goal\nNOTES"

        context = Context()
        context.items = old_turns(6)
        return Compactor(context, Settings(**settings), ask), context, asked

    def test_nothing_is_asked_while_the_request_fits(self):
        compactor, context, asked = self.compactor()

        compactor.compact_if_due(500, lambda text: None)  # 0.5 of 1,000: not over

        assert asked == [] and context.summary is None

    def test_a_request_over_the_trigger_is_summarized(self):
        compactor, context, asked = self.compactor()
        notices: list[str] = []

        compactor.compact_if_due(501, notices.append)

        assert len(asked) == 1 and context.summary == "## Goal\nNOTES"
        assert notices[0] == "Summarizing earlier messages…"

    def test_it_is_off_when_the_setting_says_so(self):
        compactor, _, asked = self.compactor(auto_compact=False)

        compactor.compact_if_due(10_000, lambda text: None)

        assert asked == []

    def test_the_newest_turns_are_kept_within_the_budget_and_half_the_trigger(self):
        compactor, _, _ = self.compactor(keep_recent_tokens=10_000)

        assert compactor.tail_budget() == 250  # half of the trigger, 500

    def test_a_failure_is_told_and_counted_and_the_history_is_the_same(self):
        compactor, context, _ = self.compactor(answer=TurnError("no good"))
        notices: list[str] = []
        before = list(context.items)

        compactor.compact_if_due(900, notices.append)

        assert notices[-1] == "Could not summarize earlier messages: no good"
        assert compactor.failures == 1 and context.items == before

    def test_after_enough_failures_none_is_tried_by_itself(self):
        compactor, _, asked = self.compactor(answer=TurnError("no good"))
        notices: list[str] = []

        for _ in range(COMPACT_FAILURES + 2):
            compactor.compact_if_due(900, notices.append)

        assert len(asked) == COMPACT_FAILURES
        assert "paused until /compact works" in notices[-1]

    def test_a_summary_that_works_clears_the_failures(self):
        compactor, _, _ = self.compactor()
        compactor.failures = 2

        compactor.compact(lambda text: None)

        assert compactor.failures == 0

    def test_a_focus_goes_into_the_request(self):
        compactor, _, asked = self.compactor()

        compactor.compact(lambda text: None, focus="the schema")

        assert "Focus: the schema" in asked[0]

    def test_it_needs_nothing_of_the_agent(self):
        # A fresh interpreter: this process has imported everything already.
        import subprocess
        import sys

        code = (
            "import sys, muagent.compaction; "
            "print([m for m in ('muagent.agent', 'muagent.config') if m in sys.modules])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == "[]"
