import platform
from collections.abc import Callable, Sequence
from datetime import date
from typing import cast

from openai import OpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from openai.types.responses.response_create_params import ResponseCreateParamsBase

from muagent.config import Config
from muagent.context import Context, estimate_tokens
from muagent.tools import Tool

SYSTEM_PROMPT = """
You are an expert AI coding agent running in the user's terminal.
You help with software engineering: writing, reading, debugging and explaining code.
Operating rules:
- Be concise and direct.
- You cannot read files or run commands yet; ask the user to paste what you need.
- If a tool call fails, read the error message and correct your call.
- Write math in Unicode (∑ ∫ √ π α β θ ± × ÷ ≤ ≥ ≈ ∈ ∞, x² x₁, a/b for fractions), never LaTeX — this terminal doesn't render `$...$`, `\\frac{}{}`, or other LaTeX/KaTeX syntax; it shows literal backslashes and dollar signs instead.
"""


# Longest line a tool call or its result may take in a notice.
NOTICE_CHARS = 80


def _one_line(text: str) -> str:
    """The text as one line of at most NOTICE_CHARS characters, for a notice."""
    line = " ".join(text.split())
    return line if len(line) <= NOTICE_CHARS else line[: NOTICE_CHARS - 1] + "…"


class TurnError(Exception):
    """A turn that ended without a normal reply.

    History is left untouched, with two exceptions: an incomplete response after
    some text keeps the partial reply the user has already seen, and hitting the
    max_turns cap keeps the whole turn plus a closing note.
    """


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
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) < len(tools):
            raise ValueError("Tool names must be unique")

    def run_turn(
        self,
        prompt: str,
        on_text: Callable[[str], None] = lambda text: None,
        on_notice: Callable[[str], None] = lambda text: None,
        confirm: Callable[[Tool, str], bool] = lambda tool, arguments: False,
    ) -> None:
        """Send one user message, passing the reply text to on_text.

        If the model calls tools, each call is run and answered, and the model is
        called again, up to max_turns times, until it gives a reply. The whole turn
        (the calls and their outputs included) is added to the conversation if the
        final response completes with text, or ends incomplete after some text,
        which the user has seen, or if the model hits the max_turns cap (closed with
        a note saying so).

        Args:
            prompt: The user's input prompt.
            on_text: Called with each piece of reply text as it arrives when
                streaming, or once with the whole reply otherwise.
            on_notice: Called with status messages, such as old turns being
                dropped to fit the token budget, and each tool call as it is run.
            confirm: Called with a tool that is not read-only and the model's
                arguments for it, before it runs; it returns whether to allow
                the call. The default denies every such call.

        Raises:
            TurnError: If the response is incomplete, failed, refused or has no
                text, if a stream ends without a final response, if the model
                is still calling tools after max_turns calls, or if the tool
                results grow the turn past the context budget.
        """
        items, trimmed = self.context.begin_turn(prompt, self.config.max_context_tokens)
        if trimmed:
            on_notice("Context trimmed: oldest turns dropped")

        for _ in range(self.config.max_turns):
            sent = len(items)
            response = self._send_response(items, on_text)
            problem = self._response_status(response)
            calls = [item for item in response.output if item.type == "function_call"]
            if problem or not calls:
                break
            items += cast(
                ResponseInputParam,
                [item.model_dump(exclude_none=True) for item in response.output],
            )
            items += self._run_tools(calls, on_notice, confirm)
            used = self._next_request_tokens(response, items, sent)
            if used > self.config.max_context_tokens:
                raise TurnError(
                    f"This turn grew past the context budget (about {used} of "
                    f"{self.config.max_context_tokens} tokens). Ask for less, or "
                    "raise MAX_CONTEXT_TOKENS"
                )
        else:
            # Every call has its output, so the turn is coherent. Keep it, closed with
            # a note, so the model still knows the question and the work already done
            # (some of it may have had effects) when the user says to carry on.
            message = (
                f"Stopped after {self.config.max_turns} model calls without a "
                "final answer"
            )
            self.context.save_turn(
                [*items, {"role": "assistant", "content": f"[{message}]"}]
            )
            raise TurnError(message)

        # Keep the turn if it worked, or was cut off after text the user has already
        # seen. Any other turn is dropped: saving a prompt with no reply would leave
        # two user messages in a row.
        cut_off = response.status == "incomplete"
        if response.output_text and (problem is None or cut_off):
            self.context.save_turn(
                [*items, {"role": "assistant", "content": response.output_text}]
            )
        if problem:
            raise TurnError(problem)

    def _send_response(
        self, draft: ResponseInputParam, on_text: Callable[[str], None]
    ) -> Response:
        """Get the model's final response, passing its text to on_text as it arrives.

        Args:
            draft: The input for this turn.
            on_text: Called with each piece of reply text when streaming, or once
                with the whole reply otherwise (before any status check, so
                partial text is always shown).

        Returns:
            The final response, whatever its status.

        Raises:
            TurnError: If a stream ends without a final response; the message is the
                reason the server gave in an `error` event, if it sent one.
        """
        params: ResponseCreateParamsBase = {
            "store": False,
            "input": draft,
            "model": self.config.model,
            "instructions": (
                f"{SYSTEM_PROMPT}\n"
                "Environment:\n"
                f"- Operating system: {platform.platform()}\n"
                f"- Current date: {date.today().isoformat()}\n"
            ),
        }
        # Non-reasoning models reject a reasoning setting, so leave it out if unset.
        if self.config.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.tools:
            params["tools"] = [tool.param() for tool in self.tools.values()]

        if not self.config.streaming:
            response = self.client.responses.create(**params)
            on_text(response.output_text)
            return response

        final = None
        reason = "Stream ended without a final response"
        with self.client.responses.stream(**params) as stream:
            for event in stream:
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
                # response.completed. Keep the == chain: pyright narrows the event
                # type through it but not through `in (...)`.
                elif (
                    event.type == "response.completed"
                    or event.type == "response.incomplete"
                    or event.type == "response.failed"
                ):
                    final = event.response
        if final is None:
            raise TurnError(reason)
        return final

    def _run_tools(
        self,
        calls: Sequence[ResponseFunctionToolCall],
        on_notice: Callable[[str], None],
        confirm: Callable[[Tool, str], bool],
    ) -> ResponseInputParam:
        """Run the model's tool calls and build the outputs to send back.

        Every call gets an output, even a failed, unknown or denied one: the output
        is text for the model, which can read the error and correct its call. A
        tool that is not read-only runs only if `confirm` allows the call.

        Args:
            calls: The function calls in the model's response.
            on_notice: Called with each call before it is run, shortened to one line.
                The result is not shown: it goes to the model, and the reply says
                what it found.
            confirm: Asked before each call to a tool that is not read-only.

        Returns:
            One function_call_output item per call, in order.
        """
        outputs: ResponseInputParam = []
        for call in calls:
            on_notice(f"{call.name}({_one_line(call.arguments)})")
            tool = self.tools.get(call.name)
            if tool is None:
                known = ", ".join(self.tools) or "none"
                output = f"Unknown tool: {call.name}. Available tools: {known}."
            # A call that would be rejected is not worth asking the user about.
            elif problem := tool.check(call.arguments):
                output = problem
            elif tool.read_only or confirm(tool, call.arguments):
                output = tool.call(call.arguments)
            else:
                output = "The user denied this call."
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )
        return outputs

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
