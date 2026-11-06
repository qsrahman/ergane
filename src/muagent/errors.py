"""How a turn can go wrong, and how the user is told."""

import re

from openai import APIConnectionError, APIError, APIStatusError


class TurnError(Exception):
    """A turn that ended without a normal reply.

    History is left untouched, with two exceptions: an incomplete response after
    some text keeps the partial reply the user has already seen, and hitting the
    max_turns cap keeps the whole turn plus a closing note.
    """


class StreamCut(TurnError):
    """A stream that stopped before its final response, with no reason from the server.

    A second try can get past it, as long as nothing of the reply has been shown.
    """


# What to check, by the status of an API error that has a usual cause.
_STATUS_HINTS = {
    401: "Check OPENAI_API_KEY.",
    403: "Check OPENAI_API_KEY and that it may use this model.",
    404: "Check OPENAI_MODEL and OPENAI_BASE_URL.",
    429: "The account is out of quota, or rate limited.",
}


def is_context_overflow(exc: BaseException) -> bool:
    """Whether a server refused a request for being longer than its context window.

    Servers word it in their own way; OpenAI sends the code `context_length_exceeded`.
    """
    if not isinstance(exc, APIStatusError) or exc.status_code not in (400, 413):
        return False
    if exc.code == "context_length_exceeded":
        return True
    text = exc.message.lower()
    return "too long" in text or bool(
        re.search(r"(context|token).*(length|window|limit|exceed)", text)
    )


def describe_error(exc: Exception) -> str:
    """One line for an exception that ended a turn, for the user.

    The SDK's own text ends in a Python dict, and the usual causes of an API error
    (a wrong key or model, no quota, a wrong address) are worth naming.
    """
    if isinstance(exc, APIStatusError):
        body = exc.body if isinstance(exc.body, dict) else {}
        message = body.get("message") or exc.message
        code = f" ({exc.code})" if exc.code else ""
        hint = _STATUS_HINTS.get(exc.status_code)
        if is_context_overflow(exc):
            hint = "Lower max_context_tokens, or use /compact."
        return f"API error {exc.status_code}{code}: {message}" + (
            f" {hint}" if hint else ""
        )
    if isinstance(exc, APIConnectionError):
        return f"{exc} Check OPENAI_BASE_URL and the network."
    if isinstance(exc, APIError):
        return f"API error: {exc.message}"
    return f"Agent error: {type(exc).__name__}: {exc}"
