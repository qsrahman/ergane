"""Talking to the Responses API: one request in, the model's final response out.

`Responder` builds the request (no storage, reasoning items back in, the reasoning effort
the user set), sends it, streams the text to a callback as it arrives, and sends a
request again when its stream breaks before any text was shown. It knows nothing of
conversations, tools or budgets: the caller gives it the input, the instructions and the
tool definitions, and reads what is wrong with the response with `response_problem`.
"""

from collections.abc import Callable, Sequence

from openai import APIConnectionError, OpenAI
from openai.types.responses import FunctionToolParam, Response, ResponseInputParam
from openai.types.responses.response_create_params import ResponseCreateParamsBase

from muagent.config import Config
from muagent.errors import StreamCut, TurnError

# How many times a round is sent again when its stream breaks before any text of the
# reply has reached the user. The SDK retries a request that fails to start, but not
# one that breaks halfway, and a turn of many rounds should not be lost to one dropped
# connection: it happened, 25 rounds into reading a codebase.
STREAM_RETRIES = 2

# What a response with neither text nor a call is reported as. Some compatible models
# now and then answer with nothing; asking again once usually gets a reply.
NO_TEXT = "No text in response"


def with_output(response: Response) -> Response:
    """The response, with an empty `output` where a compatible server sent `null`, which
    `output_text` and everything that reads the output cannot take. A streamed response
    needs no such care: `responses.stream` rebuilds its output."""
    if response.output is None:
        response.output = []
    return response


def response_problem(response: Response) -> str | None:
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
    if response.output_text or any(i.type == "function_call" for i in response.output):
        return None
    refusal = "".join(
        part.refusal
        for item in response.output
        if item.type == "message"
        for part in item.content
        if part.type == "refusal"
    )
    return f"Refused: {refusal}" if refusal else NO_TEXT


class Responder:
    """Sends requests to the Responses API."""

    def __init__(self, client: OpenAI, settings: Config) -> None:
        self.client = client
        self.settings = settings

    def send(
        self,
        draft: ResponseInputParam,
        instructions: str,
        on_text: Callable[[str], None],
        on_notice: Callable[[str], None],
        *,
        tools: Sequence[FunctionToolParam] = (),
        max_output_tokens: int | None = None,
        low_effort: bool = False,
    ) -> Response:
        """Get the model's final response, passing its text to on_text as it arrives.

        A stream that breaks (see `StreamCut`) before any text has been shown is
        sent again, up to STREAM_RETRIES times, and each time `on_notice` says so.
        Nothing of the round has run yet, so the same request is safe to repeat; once
        text has been shown it is not, because the reply would appear twice.

        Args:
            draft: The input for this request.
            instructions: The system prompt, and whatever goes with it.
            on_text: Called with each piece of reply text when streaming, or once
                with the whole reply otherwise (before any status check, so
                partial text is always shown).
            on_notice: Called with a line when a request is sent again.
            tools: The tools the model may call; none are offered if empty.
            max_output_tokens: A cap on the answer, if there is to be one.
            low_effort: Ask a reasoning model to think as little as it can, whatever
                effort the user set (for a summary, which needs none).

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
            "model": self.settings.model,
            "instructions": instructions,
            # store=False means the server keeps no history, so reasoning items are
            # resent by muagent in the input; their encrypted content is requested
            # here, otherwise a reasoning model cannot use it (and may fail the
            # request) when turning back to a tool to answer.
            "include": ["reasoning.encrypted_content"],
        }
        # Non-reasoning models reject a reasoning setting, so leave it out if unset.
        effort = self.settings.reasoning_effort
        if effort is not None:
            params["reasoning"] = {
                "effort": (
                    "low"
                    if low_effort and effort not in ("none", "minimal", "low")
                    else effort
                )
            }
        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens
        if tools:
            params["tools"] = list(tools)

        if not self.settings.streaming:
            response = with_output(self.client.responses.create(**params))
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
