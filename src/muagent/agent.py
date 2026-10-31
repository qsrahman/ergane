"""The agent: sends one conversation turn to the model and keeps the history."""

import platform
from collections.abc import Callable
from datetime import date

from openai import Omit, OpenAI, omit
from openai.types.responses import Response, ResponseInputParam
from openai.types.shared_params import Reasoning

from muagent.config import Config
from muagent.context import Context

SYSTEM_PROMPT = """
You are an expert AI coding agent running in the user's terminal.
You help with software engineering: writing, reading, debugging and explaining code.
Operating rules:
- Be concise and direct.
- You cannot read files or run commands yet; ask the user to paste what you need.
- Write math in Unicode (∑ ∫ √ π α β θ ± × ÷ ≤ ≥ ≈ ∈ ∞, x² x₁, a/b for fractions), never LaTeX — this terminal doesn't render `$...$`, `\\frac{}{}`, or other LaTeX/KaTeX syntax; it shows literal backslashes and dollar signs instead.
"""


class TurnError(Exception):
    """A turn that ended without a normal reply.

    Nothing is saved to the history, except that a partial reply is kept when the
    response ended incomplete after some text, which the user has already seen.
    """


class Agent:
    """Sends conversation turns to the model. Does no terminal I/O of its own."""

    def __init__(self, config: Config, client: OpenAI | None = None) -> None:
        """Create an agent with an empty conversation.

        Args:
            config: The agent configuration.
            client: The API client; built from the config if omitted.
        """
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
        )
        self.context = Context()

    def run_turn(
        self,
        prompt: str,
        on_text: Callable[[str], None] = lambda text: None,
        on_notice: Callable[[str], None] = lambda text: None,
    ) -> None:
        """Send one user message, passing the reply text to on_text.

        The turn is added to the conversation if the response completes with
        text, or ends incomplete after some text, which the user has seen.

        Args:
            prompt: The user's input prompt.
            on_text: Called with each piece of reply text as it arrives when
                streaming, or once with the whole reply otherwise.
            on_notice: Called with status messages, such as old turns being
                dropped to fit the token budget.

        Raises:
            TurnError: If the response is incomplete, failed, refused or has no
                text, or if a stream ends without a final response.
        """
        draft, trimmed = self.context.draft(prompt, self.config.max_context_tokens)
        if trimmed:
            on_notice("Context trimmed: oldest turns dropped")

        response = self._respond(draft, on_text)
        problem = self._problem(response)

        # Keep the reply if the turn worked, or was cut off after text the user has
        # already seen. Any other turn is dropped: saving a prompt with no reply
        # would leave two user messages in a row.
        cut_off = response.status == "incomplete"
        if response.output_text and (problem is None or cut_off):
            self.context.commit(draft, response.output_text)
        if problem:
            raise TurnError(problem)

    def _respond(
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
        instructions = self._instructions()
        # omit leaves reasoning out of the request; non-reasoning models reject it.
        effort = self.config.reasoning_effort
        reasoning: Reasoning | Omit = omit if effort is None else {"effort": effort}

        if not self.config.streaming:
            response = self.client.responses.create(
                store=False,
                input=draft,
                model=self.config.model,
                instructions=instructions,
                reasoning=reasoning,
            )
            on_text(response.output_text)
            return response

        final = None
        reason = "Stream ended without a final response"
        with self.client.responses.stream(
            store=False,
            input=draft,
            model=self.config.model,
            instructions=instructions,
            reasoning=reasoning,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    on_text(event.delta)
                elif event.type == "error":
                    # Keep the server's reason in case no final response follows.
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

    @staticmethod
    def _problem(response: Response) -> str | None:
        """Describe what is wrong with a response, if anything.

        Args:
            response: The final response of a turn.

        Returns:
            A message for the user, or None if it is a normal completed reply.
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
        if not response.output_text:
            refusal = "".join(
                part.refusal
                for item in response.output
                if item.type == "message"
                for part in item.content
                if part.type == "refusal"
            )
            return f"Refused: {refusal}" if refusal else "No text in response"
        return None

    def _instructions(self) -> str:
        """Build the system instructions, including details of the environment."""
        return (
            f"{SYSTEM_PROMPT}\n"
            "Environment:\n"
            f"- Operating system: {platform.platform()}\n"
            f"- Current date: {date.today().isoformat()}\n"
        )
