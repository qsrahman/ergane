"""The agent: runs a turn against the model, with tools, inside a token budget.

`Agent` does no terminal I/O; what the user should see goes through the callbacks of
`run_turn`. TURN.md walks a turn through this module. What must stay true when you
change it:

- Every function call has its output before anything is sent or saved (the API
  rejects a call without one, and an output without its call). Rounds that fail or
  are interrupted are answered, never left open.
- A turn is saved whole or not at all, with one exception: once a round has
  completed (tools may have had effects, and the model must know), a failure, Ctrl-C
  or the `max_turns` cap keeps the turn with a closing note. A failure in the first
  round saves nothing.
- A stream that breaks before any text is shown is sent again (`STREAM_RETRIES`):
  Ollama's cloud proxy lost a connection 100 s into a request and a turn of 25 rounds
  was thrown away. Tools run only after a complete response, so a retry repeats a
  request, never an action.
- Summaries (`compact`) change nothing unless the summary is good, and a failure
  never stops the turn. The design follows research into opencode, Codex, OpenClaw, pi,
  Qwen Code and DeepSeek Harness. Not done: a separate summary model, a summary
  request that shares the chat's cache, splitting one oversized turn, and a stored
  transcript with undo.
"""

import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

from openai import APIConnectionError, OpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from openai.types.responses.response_create_params import ResponseCreateParamsBase

from muagent.config import Config
from muagent.context import (
    Context,
    answer_open_calls,
    drop_outputs,
    estimate_tokens,
    extract_summary,
    transcript,
)
from muagent.errors import StreamCut, TurnError, describe_error, is_context_overflow
from muagent.prompts import COMPACT_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_NO_TOOLS
from muagent.tools import Tool

# The summary of old turns may have this many tokens.
SUMMARY_MAX_TOKENS = 8000

# Summaries that failed one after another, after which none is tried by itself.
COMPACT_FAILURES = 3

# The reason in the note that closes a turn the user stopped.
INTERRUPTED = "interrupted by the user"

# How many times a round is sent again when its stream breaks before any text of the
# reply has reached the user. The SDK retries a request that fails to start, but not
# one that breaks halfway, and a turn of many rounds should not be lost to one dropped
# connection: it happened, 25 rounds into reading a codebase.
STREAM_RETRIES = 2


@dataclass
class _Turn:
    """The turn that is running: what it has sent and done so far."""

    prompt: str
    items: ResponseInputParam  # the request's input, which grows with each round
    rounds: int = 0  # rounds that got a response and ran its tools
    refused: bool = False  # the server refused a request as too long, and we tried


class Agent:
    """Sends conversation turns to the model. Does no terminal I/O of its own."""

    def __init__(
        self,
        config: Config,
        client: OpenAI | None = None,
        tools: Sequence[Tool] = (),
    ) -> None:
        """Create an agent with an empty conversation.

        Args:
            config: The agent configuration.
            client: The API client; built from the config if omitted.
            tools: The tools the model may call. The agent only ever uses the tools
                it is given here.

        Raises:
            ValueError: If two tools have the same name.
        """
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
        )
        self.context = Context()
        # Tools the user allowed for the rest of the session, by name.
        self.always_allowed: set[str] = set()
        # What the server counted in the last request it answered; None until then,
        # and when a compatible server sends no usage.
        self.last_input_tokens: int | None = None
        # Summaries that failed one after another (see COMPACT_FAILURES).
        self.compact_failures = 0
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) < len(tools):
            raise ValueError("Tool names must be unique")

    def run_turn(
        self,
        prompt: str,
        on_text: Callable[[str], None] = lambda text: None,
        on_notice: Callable[[str], None] = lambda text: None,
        confirm: Callable[
            [Tool, str], bool | Literal["always"]
        ] = lambda tool, arguments: False,
        on_call: Callable[[str, str], None] = lambda name, arguments: None,
    ) -> None:
        """Send one user message, passing the reply text to on_text.

        If the model calls tools, each call is run and answered, and the model is
        called again, up to max_turns times, until it gives a reply. The whole turn
        (the calls and their outputs included) is added to the conversation if the
        final response completes with text, or ends incomplete after some text,
        which the user has seen, or if the model hits the max_turns cap (closed with
        a note saying so), or if a round after the first fails or the user presses
        Ctrl-C (closed with a note saying why). Ctrl-C during the first model call
        saves nothing.

        Args:
            prompt: The user's input prompt.
            on_text: Called with each piece of reply text as it arrives when
                streaming, or once with the whole reply otherwise.
            on_notice: Called with status messages, such as old turns being
                dropped to fit the token budget.
            confirm: Called with a tool that is not read-only and the model's
                arguments for it, before it runs; it returns whether to allow
                the call (or "always", which allows it and every later call to that
                tool in this session). It is also what shows the user the call, so
                there is no other line for it. The default denies every such call.
            on_call: Called with the name and the arguments (JSON text) of each
                tool call that runs without asking, before it runs, to show it.
                A call that is asked about is shown by `confirm`, and a result is
                not shown at all: it goes to the model, and the reply says what it
                found.

        Raises:
            TurnError: If the response is incomplete, failed, refused or has no
                text, if a stream ends without a final response, if the model
                is still calling tools after max_turns calls, or if the tool
                results grow the turn past the context budget.
            Exception: Whatever the API client raises, such as an
                `openai.APIError`. Once a round has completed, the turn is kept
                (see above) before any of these is raised, except the budget one.
            KeyboardInterrupt: On Ctrl-C, after the turn is kept (see above).
        """
        self._compact_if_due(prompt, on_notice)
        # The budget is for the whole request, so the history gets what is left of it
        # after the instructions and the tool definitions, which every request has.
        items, trimmed = self.context.begin_turn(prompt, self._room())
        if trimmed:
            on_notice("Context trimmed: oldest turns dropped")
        turn = _Turn(prompt, items)

        for _ in range(self.config.max_turns):
            response = self._ask(turn, on_text, on_notice)
            if response is None:  # the request was shortened: ask again
                continue
            problem = self._response_status(response)
            calls = [item for item in response.output if item.type == "function_call"]
            if problem or not calls:
                break
            self._run_round(turn, response, calls, on_notice, on_call, confirm)
        else:
            # Every call has its output, so the turn is coherent. Keep it, closed with
            # a note, so the model still knows the question and the work already done
            # (some of it may have had effects) when the user says to carry on.
            message = (
                f"Stopped after {self.config.max_turns} model calls without a "
                "final answer"
            )
            self._save_with_note(turn.items, message)
            raise TurnError(message)

        # Keep the turn if it worked, or was cut off after text the user has already
        # seen. A turn whose first response has a problem is dropped: saving a prompt
        # with no reply would leave two user messages in a row. After a completed
        # round it is kept with a note, like the stop at the cap.
        cut_off = response.status == "incomplete"
        if response.output_text and (problem is None or cut_off):
            self.context.save_turn(
                [*turn.items, {"role": "assistant", "content": response.output_text}]
            )
        elif problem and turn.rounds:
            self._keep_stopped_turn(turn.items, turn.rounds, problem)
        if problem:
            raise TurnError(problem)

    def _room(self) -> int:
        """What is left of the budget for the history: the whole request is counted,
        and the instructions and the tool definitions are in every one."""
        return self.config.max_context_tokens - self._fixed_tokens()

    def _ask(
        self,
        turn: _Turn,
        on_text: Callable[[str], None],
        on_notice: Callable[[str], None],
    ) -> Response | None:
        """Send the turn's items and get the model's response.

        A request the server refuses as too long is shortened and the caller asks
        again (None), once in a turn. Any other failure ends the turn, and once a
        round has completed the turn is kept first: every call has its output at this
        point, so what the earlier rounds did (it may have had effects) is known.

        Raises:
            Exception: Whatever `_send_response` raises.
            KeyboardInterrupt: On Ctrl-C.
        """
        try:
            response = self._send_response(turn.items, on_text, on_notice)
        except (Exception, KeyboardInterrupt) as exc:
            if is_context_overflow(exc) and not turn.refused:
                turn.refused = True
                if self._shrink(turn, on_notice):
                    return None
            if turn.rounds:
                if isinstance(exc, KeyboardInterrupt):
                    reason = INTERRUPTED
                elif isinstance(exc, TurnError):
                    reason = str(exc)
                else:
                    reason = type(exc).__name__
                self._keep_stopped_turn(turn.items, turn.rounds, reason)
            raise
        if response.usage is not None:
            self.last_input_tokens = response.usage.input_tokens
        return response

    def _run_round(
        self,
        turn: _Turn,
        response: Response,
        calls: Sequence[ResponseFunctionToolCall],
        on_notice: Callable[[str], None],
        on_call: Callable[[str, str], None],
        confirm: Callable[[Tool, str], bool | Literal["always"]],
    ) -> None:
        """Add a response that calls tools to the turn, run the calls, and make sure
        the next request will fit.

        Raises:
            TurnError: If the turn is over the budget with nothing older to shorten.
            KeyboardInterrupt: On Ctrl-C, after the turn is kept.
        """
        turn.rounds += 1
        sent = len(turn.items)  # what the response answered
        try:
            turn.items += cast(
                ResponseInputParam,
                [item.model_dump(exclude_none=True) for item in response.output],
            )
            self._run_tools(calls, turn.items, on_call, confirm)
        except KeyboardInterrupt:
            # A call that was running, or had not run yet, has no output: say so, or
            # the API rejects the history. The call may have done part of its work,
            # and the model should know the user stopped it.
            answer_open_calls(turn.items)
            self._keep_stopped_turn(turn.items, turn.rounds, INTERRUPTED)
            raise
        budget = self.config.max_context_tokens
        used = self._next_request_tokens(response, turn.items, sent)
        if used > budget:
            # Results of earlier rounds go first, so the work goes on; the latest
            # ones are what the model needs to read next.
            freed = drop_outputs(turn.items, sent, used - budget)
            if freed:
                on_notice("Context full: older tool results dropped")
            used -= freed
        if used > budget:
            # Only the latest round is left: shorten its results, but keep their
            # start and end, which is what the model needs to read.
            freed = drop_outputs(
                turn.items, len(turn.items), used - budget, placeholders=False
            )
            if freed:
                on_notice("Context full: this round's tool results shortened")
            used -= freed
        if used > budget:
            # Still over. The calls have run and may have had effects: keep the turn,
            # with its results dropped, or every later request would carry them.
            drop_outputs(turn.items, len(turn.items), used)
            self._keep_stopped_turn(
                turn.items, turn.rounds, "its results passed the context budget"
            )
            raise TurnError(
                f"This turn grew past the context budget (about {used} of "
                f"{budget} tokens). Ask for less, or raise max_context_tokens"
            )

    def _keep_stopped_turn(
        self, items: ResponseInputParam, rounds: int, reason: str
    ) -> None:
        """Save a turn that failed after `rounds` completed rounds, closed with a
        note, so the model still knows the work already done."""
        self._save_with_note(items, f"Stopped after {rounds} model calls: {reason}")

    def _save_with_note(self, items: ResponseInputParam, message: str) -> None:
        """Save a turn that has no final answer, closed with an assistant note.
        Every call in `items` must have its output."""
        self.context.save_turn(
            [*items, {"role": "assistant", "content": f"[{message}]"}]
        )

    def _tail_budget(self) -> int:
        """How many tokens of the newest turns a summary leaves as they are: the
        setting, but no more than half of where summarizing starts, or a small
        budget would keep everything."""
        trigger = int(self.config.compact_at * self.config.max_context_tokens)
        return min(self.config.keep_recent_tokens, trigger // 2)

    def _compact_if_due(self, prompt: str, on_notice: Callable[[str], None]) -> None:
        """Summarize the oldest turns when the next request would pass `compact_at`
        of the budget, unless the setting is off or summaries kept failing.

        A failure is told and counted, and does not stop the turn: the turns that do
        not fit are dropped as before, which is the worst that happens.
        """
        config = self.config
        if not config.auto_compact or self.compact_failures >= COMPACT_FAILURES:
            return
        next_request = (
            self._fixed_tokens()
            + self.context.estimate()
            + estimate_tokens({"role": "user", "content": prompt})
        )
        if next_request <= config.compact_at * config.max_context_tokens:
            return
        try:
            self.compact(on_notice)
        except Exception as exc:  # noqa: BLE001
            self.compact_failures += 1
            reason = str(exc) if isinstance(exc, TurnError) else describe_error(exc)
            paused = (
                "; summaries are paused until /compact works"
                if self.compact_failures >= COMPACT_FAILURES
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
            TurnError: If the model's summary is unusable (see `_extract_summary`).
            Exception: Whatever the API client raises.
            KeyboardInterrupt: On Ctrl-C, with the history as it was.
        """
        count = self.context.plan_compaction(self._tail_budget())
        if count == 0:
            return False
        older = self.context.items[:count]
        before = self.context.estimate()
        on_notice("Summarizing earlier messages…")

        text = transcript(older)
        parts = []
        if self.context.summary:
            parts.append(f"Previous summary:\n{self.context.summary}")
        parts.append(f"Conversation:\n{text}")
        if focus:
            parts.append(f"Focus: {focus[:800]}")
        draft: ResponseInputParam = [{"role": "user", "content": "\n\n".join(parts)}]
        response = self._send_response(
            draft, lambda text: None, on_notice, summary=True
        )
        if problem := self._response_status(response):
            raise TurnError(problem)
        summary = extract_summary(response.output_text, "\n".join(parts))

        turns = sum(
            1 for item in older if isinstance(item, dict) and item.get("role") == "user"
        )
        self.context.compact(count, summary)
        self.compact_failures = 0
        on_notice(
            f"Summarized {turns} earlier turn{'s' if turns != 1 else ''}: "
            f"about {before:,} to {self.context.estimate():,} tokens of history"
        )
        return True

    def _shrink(self, turn: _Turn, on_notice: Callable[[str], None]) -> bool:
        """Make the turn's request shorter after the server refused it as too long.

        Before the first round that means summarizing, whatever the trigger says; in
        a later round, shortening and dropping the results of the rounds before it.

        Returns:
            Whether the turn's items were made shorter, to be sent again.
        """
        if turn.rounds == 0:
            try:
                if not self.compact(on_notice):
                    return False
            except Exception:  # noqa: BLE001
                return False
            turn.items, _ = self.context.begin_turn(turn.prompt, self._room())
        else:
            results = sum(
                estimate_tokens(item)
                for item in cast(list[dict[str, Any]], turn.items)
                if item.get("type") == "function_call_output"
            )
            # Half of them: shortened first, dropped only if that is not enough.
            if not drop_outputs(turn.items, len(turn.items), results // 2 + 1):
                return False
        on_notice("The request was too long for the model; trying again, shorter")
        return True

    def _instructions(self) -> str:
        """What every request carries as `instructions`: the system prompt, then the
        environment, so that the part that never changes comes first and can be
        cached."""
        prompt = SYSTEM_PROMPT if self.tools else SYSTEM_PROMPT_NO_TOOLS
        return (
            f"{prompt}\n"
            "Environment:\n"
            f"- Operating system: {platform.platform()}\n"
            f"- Current date: {date.today().isoformat()}\n"
        )

    def _fixed_estimate(self) -> tuple[int, int]:
        """Estimate the tokens that every request carries whatever the conversation:
        the instructions, and the definitions of the tools."""
        tools = [tool.param() for tool in self.tools.values()]
        return (
            estimate_tokens({"instructions": self._instructions()}),
            estimate_tokens({"tools": tools}) if tools else 0,
        )

    def _fixed_tokens(self) -> int:
        """The instructions and the definitions of the tools, together."""
        return sum(self._fixed_estimate())

    def context_estimate(self) -> tuple[int, int, int]:
        """Estimate the tokens of the instructions, the tool definitions and the
        history, which together are what the next request would carry."""
        return (*self._fixed_estimate(), self.context.estimate())

    def _send_response(
        self,
        draft: ResponseInputParam,
        on_text: Callable[[str], None],
        on_notice: Callable[[str], None],
        *,
        summary: bool = False,
    ) -> Response:
        """Get the model's final response, passing its text to on_text as it arrives.

        A stream that breaks (see `StreamCut`) before any text has been shown is
        sent again, up to STREAM_RETRIES times, and each time `on_notice` says so.
        Nothing of the round has run yet, so the same request is safe to repeat; once
        text has been shown it is not, because the reply would appear twice.

        Args:
            draft: The input for this turn.
            on_text: Called with each piece of reply text when streaming, or once
                with the whole reply otherwise (before any status check, so
                partial text is always shown).
            on_notice: Called with a line when a round is sent again.
            summary: Whether this is the request that summarizes old turns: the
                summarizing prompt in place of the system prompt, no tools, a cap on
                the answer, and a reasoning model asked to think as little as it can.

        Returns:
            The final response, whatever its status.

        Raises:
            StreamCut: If the stream keeps breaking, or breaks after text was shown.
            TurnError: If the server ends a stream with an `error` event; the message
                is the reason it gave.
        """
        params: ResponseCreateParamsBase = {
            "store": False,
            "input": draft,
            "model": self.config.model,
            "instructions": COMPACT_PROMPT if summary else self._instructions(),
            # store=False means the server keeps no history, so reasoning items are
            # resent by muagent in the input; their encrypted content is requested
            # here, otherwise a reasoning model cannot use it (and may fail the
            # request) when turning back to a tool to answer.
            "include": ["reasoning.encrypted_content"],
        }
        # Non-reasoning models reject a reasoning setting, so leave it out if unset.
        effort = self.config.reasoning_effort
        if effort is not None:
            params["reasoning"] = {
                "effort": (
                    "low"
                    if summary and effort not in ("none", "minimal", "low")
                    else effort
                )
            }
        if summary:
            params["max_output_tokens"] = SUMMARY_MAX_TOKENS
        elif self.tools:
            params["tools"] = [tool.param() for tool in self.tools.values()]

        if not self.config.streaming:
            response = self.client.responses.create(**params)
            on_text(response.output_text)
            return response

        shown = False

        def show(text: str) -> None:
            nonlocal shown
            shown = shown or bool(text)
            on_text(text)

        retries = 0
        while True:
            try:
                return self._stream(params, show)
            except StreamCut as cut:
                retries += 1
                if shown or retries > STREAM_RETRIES:
                    raise
                on_notice(f"{cut}; trying again ({retries} of {STREAM_RETRIES})")

    def _stream(
        self, params: ResponseCreateParamsBase, on_text: Callable[[str], None]
    ) -> Response:
        """Stream one request and return its final response.

        This is `responses.stream`, not `create(stream=True)`: it rebuilds the final
        output when a compatible server leaves it out of `response.completed`.

        Raises:
            StreamCut: If the stream ends without a final response and the server
                said nothing about why, or the connection drops after the stream
                had begun. (One that fails before it begins was already retried by
                the SDK.)
            TurnError: If the server ends the stream with an `error` event.
        """
        final = None
        reason = None
        started = False
        try:
            with self.client.responses.stream(**params) as stream:
                for event in stream:
                    started = True
                    if event.type == "response.output_text.delta":
                        on_text(event.delta)
                    elif event.type == "error":
                        detail = (
                            f"{event.code}: {event.message}"
                            if event.code
                            else event.message
                        )
                        reason = f"Stream error: {detail}"
                    # Take the final response from the terminal event:
                    # get_final_response() raises unless the stream ends with
                    # response.completed. Keep the == chain: pyright narrows the
                    # event type through it but not through `in (...)`.
                    elif (
                        event.type == "response.completed"
                        or event.type == "response.incomplete"
                        or event.type == "response.failed"
                    ):
                        final = event.response
        except APIConnectionError as exc:
            if not started:
                raise
            raise StreamCut(f"Stream cut off: {str(exc).rstrip('.')}") from exc
        if final is None:
            if reason:
                # The server said why; the same request would get the same answer.
                raise TurnError(reason)
            raise StreamCut("Stream ended without a final response")
        return final

    def _run_tools(
        self,
        calls: Sequence[ResponseFunctionToolCall],
        items: ResponseInputParam,
        on_call: Callable[[str, str], None],
        confirm: Callable[[Tool, str], bool | Literal["always"]],
    ) -> None:
        """Run the model's tool calls, adding each output to `items` as it is done.

        Every call gets an output, even a failed, unknown or denied one: the output
        is text for the model, which can read the error and correct its call. A
        tool that is not read-only runs only if `confirm` allows the call.

        Args:
            calls: The function calls in the model's response.
            items: The turn's items; the outputs are added at the end, in order. If
                Ctrl-C interrupts, the outputs made so far are already there.
            on_call: Called with each call that runs without asking, before it is
                run. A call that is asked about is shown by `confirm`, and a result
                is not shown at all: it goes to the model, and the reply says what
                it found.
            confirm: Asked before each call to a tool that is not read-only.
        """
        for call in calls:
            tool = self.tools.get(call.name)
            problem = tool.check(call.arguments) if tool else None
            # `confirm` shows the call it asks about, so such a call gets no line here.
            asks = (
                tool is not None
                and problem is None
                and not tool.read_only
                and tool.name not in self.always_allowed
            )
            if not asks:
                on_call(call.name, call.arguments)
            if tool is None:
                known = ", ".join(self.tools) or "none"
                output = f"Unknown tool: {call.name}. Available tools: {known}."
            elif problem:
                # A call that would be rejected is not worth asking the user about.
                output = problem
            else:
                answer = confirm(tool, call.arguments) if asks else True
                if answer == "always":
                    self.always_allowed.add(tool.name)
                output = (
                    tool.call(call.arguments)
                    if answer
                    else "The user denied this call."
                )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    @staticmethod
    def _next_request_tokens(
        response: Response, items: ResponseInputParam, sent: int
    ) -> int:
        """Estimate how many tokens the next request will have.

        The server's own count of the request it just answered is exact, so it is
        used when the response has one; the items added since are estimated. Some
        compatible servers send no usage, and then everything is estimated.

        Args:
            response: The response to the last request.
            items: The turn's items, including those added since that request.
            sent: How many of the items were in that request.

        Returns:
            The estimated number of tokens.
        """
        if response.usage is None:
            return sum(estimate_tokens(item) for item in items)
        return response.usage.input_tokens + sum(
            estimate_tokens(item) for item in items[sent:]
        )

    @staticmethod
    def _response_status(response: Response) -> str | None:
        """Describe what is wrong with a response, if anything.

        A response that calls tools has no text, and that is fine: only the
        response that ends the turn needs some.

        Args:
            response: A response from the model.

        Returns:
            A message for the user, or None if the response is fine.
        """
        if response.status == "incomplete":
            details = response.incomplete_details
            reason = details.reason if details else "unknown"
            return f"Response incomplete: {reason}"
        if response.status == "failed":
            error = response.error
            message = f"{error.code}: {error.message}" if error else "unknown error"
            return f"Response failed: {message}"
        # status is optional in the SDK; some compatible endpoints omit it.
        if response.status not in (None, "completed"):
            return f"Response not completed: {response.status}"
        if response.output_text or any(
            i.type == "function_call" for i in response.output
        ):
            return None
        refusal = "".join(
            part.refusal
            for item in response.output
            if item.type == "message"
            for part in item.content
            if part.type == "refusal"
        )
        return f"Refused: {refusal}" if refusal else "No text in response"
