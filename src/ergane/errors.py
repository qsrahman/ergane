"""How a turn can go wrong, and how the user is told."""

import re

from openai import APIConnectionError, APIError, APIStatusError


class TurnError(Exception):
    """A turn that ended without a normal reply.

    History is left untouched, with two exceptions: an incomplete response after
    some text keeps the partial reply the user has already seen, and a stop by the
    loop guard keeps the whole turn plus a closing note.
    """


class StreamCut(TurnError):
    """A stream that stopped before its final response, with no reason from the server.

    A second try can get past it, as long as nothing of the reply has been shown.
    """


class ServerBusy(StreamCut):
    """A stream the server ended with a fault of its own that passes (overloaded, a
    rate limit, an internal error): worth a second try after a pause."""


class ContextOverflow(TurnError):
    """A stream the server ended because the request is longer than the model takes,
    before any of the reply was shown, so that a shorter request can be sent."""


# The codes of an error inside a stream that a second try may get past: OpenAI's, those
# of compatible servers, and the HTTP statuses some of them give as the code.
_TRANSIENT_CODES = frozenset(
    {
        "server_error",
        "rate_limit_exceeded",
        "vector_store_timeout",
        "api_error",
        "overloaded_error",
        "rate_limit_error",
        "internal_server_error",
        "service_unavailable",
        "timeout",
        "429",
        "500",
        "502",
        "503",
        "504",
        "529",
    }
)
# The same, told by the words of an error that has no code.
_TRANSIENT_TEXT = re.compile(
    r"overloaded|rate.?limit|(service|temporarily) unavailable|"
    r"(internal|server) error|timed? ?out"
)
# How servers say that a request is longer than the model's context.
_OVERFLOW_TEXT = re.compile(
    r"too long"
    r"|(context|token).*(length|window|limit|exceed)"
    r"|exceed\w*.{0,60}(context|token)"
    r"|input (length|size).{0,40}exceed"
    r"|too many (input |prompt )?tokens"
)
# A limit on how fast, not on how long, which may be worded in tokens too.
_RATE_TEXT = re.compile(r"rate.?limit|per (second|minute|hour|day)")


def says_overflow(code: object, message: str) -> bool:
    """Whether an error's code or words say that the request was too long."""
    if code == "context_length_exceeded":
        return True
    text = message.lower()
    return not _RATE_TEXT.search(text) and bool(_OVERFLOW_TEXT.search(text))


def is_transient(code: object, message: str) -> bool:
    """Whether the error a server ended a stream with may pass if asked again. A code
    decides when there is one; the words only when there is none."""
    if code:
        return str(code) in _TRANSIENT_CODES
    return bool(_TRANSIENT_TEXT.search(message.lower()))


# What to check, by the status of an API error that has a usual cause.
_STATUS_HINTS = {
    401: "Check the API key (OPENAI_API_KEY, or api_key in config.json).",
    403: "Check the API key and that it may use this model.",
    404: "Check the model and base_url in config.json.",
    429: "The account is out of quota, or rate limited.",
}


def is_context_overflow(exc: BaseException) -> bool:
    """Whether a server refused a request for being longer than its context window,
    at once or inside a stream.

    Servers word it in their own way; OpenAI sends the code `context_length_exceeded`.
    """
    if isinstance(exc, ContextOverflow):
        return True
    if not isinstance(exc, APIStatusError) or exc.status_code not in (400, 413):
        return False
    return says_overflow(exc.code, exc.message)


def is_images_rejected(exc: BaseException) -> bool:
    """Whether a server refused a request because the model takes no pictures.

    Servers word it in their own way: Ollama says the model "does not support
    multimodal requests", OpenAI that "image_url is only supported by certain models".
    """
    if not isinstance(exc, APIStatusError) or exc.status_code != 400:
        return False
    text = exc.message.lower()
    return bool(re.search(r"image|multimodal|vision|picture", text)) and bool(
        re.search(r"not support|unsupported|only supported|does not accept", text)
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
        return f"{exc} Check base_url in config.json and the network."
    if isinstance(exc, APIError):
        return f"API error: {exc.message}"
    return f"Agent error: {type(exc).__name__}: {exc}"
