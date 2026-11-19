"""The results of tools in the history: their form, and how they are shortened or
dropped to make room.

A result is text, or the list of parts the Responses API takes when it has pictures.
When a turn does not fit, a picture goes first, then a long result is cut to its start
and end, and only then is a result replaced by a placeholder: the call stays, so the
model sees what it did and every call keeps an output, which the API requires.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openai.types.responses import (
    ResponseFunctionCallOutputItemListParam,
    ResponseInputImageContentParam,
    ResponseInputParam,
    ResponseInputTextContentParam,
)

from muagent.context import as_dicts, estimate_tokens
from muagent.tools import Image

# What stands in for a tool result that was dropped to make room (see drop_outputs).
DROPPED_OUTPUT = "[output dropped to fit the context]"


# A tool result longer than this is shortened, before it is dropped, when the turn
# does not fit: the start and the end are kept, since the start says what it is and
# the end often says how it went, and the marker says that something is hidden.
PRUNE_CHARS = 8000


PRUNE_HEAD = 4000


PRUNE_TAIL = 1000


IMAGE_DROPPED = "[image dropped to fit the context]"


IMAGE_REFUSED = "[image removed: the model does not take pictures]"


def function_output(
    text: str, images: Sequence[Image] = ()
) -> str | ResponseFunctionCallOutputItemListParam:
    """What a function call's output is for a result: the text, or, with pictures, the
    list of parts that the Responses API takes."""
    if not images:
        return text
    text_part: ResponseInputTextContentParam = {"type": "input_text", "text": text}
    pictures: list[ResponseInputImageContentParam] = [
        {
            "type": "input_image",
            "image_url": f"data:{image.mime_type};base64,{image.data}",
            "detail": "auto",
        }
        for image in images
    ]
    return [text_part, *pictures]


def output_text(output: str | Sequence[Mapping[str, str]]) -> str:
    """The text of a function call's output, with `[image]` where a picture is."""
    if isinstance(output, str):
        return output
    return "\n".join(
        part.get("text", "") if part["type"] == "input_text" else "[image]"
        for part in output
    )


def _text_parts(output: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(part["text"] for part in output if part["type"] == "input_text")


def _has_images(output: object) -> bool:
    return isinstance(output, list) and any(
        part.get("type") == "input_image" for part in output
    )


def drop_images(items: ResponseInputParam) -> int:
    """Take the pictures out of the tool results in `items`, which stay as text: what
    to do when the model takes none. An item is replaced, not edited: the items of
    earlier turns are shared with the history.

    Returns:
        How many results had pictures.
    """
    plain = as_dicts(items)
    dropped = 0
    for i, item in enumerate(plain):
        if item.get("type") == "function_call_output" and _has_images(item["output"]):
            text = f"{_text_parts(item['output'])}\n{IMAGE_REFUSED}"
            plain[i] = {**item, "output": text}
            dropped += 1
    return dropped


def shorten(text: str) -> str:
    """A long tool result with its middle hidden, and a marker that says so."""
    hidden = len(text) - PRUNE_HEAD - PRUNE_TAIL
    return (
        f"{text[:PRUNE_HEAD]}\n[... {hidden:,} characters hidden; call the tool "
        f"again if you need them ...]\n{text[-PRUNE_TAIL:]}"
    )


def answer_open_calls(items: ResponseInputParam) -> None:
    """Give every function call in `items` that has no output one that says the
    user interrupted it."""
    plain = as_dicts(items)
    answered = {i["call_id"] for i in plain if i.get("type") == "function_call_output"}
    for item in list(plain):
        if item.get("type") == "function_call" and item["call_id"] not in answered:
            plain.append(
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": "The user interrupted this call: it did not "
                    "finish, or did not run.",
                }
            )


def drop_outputs(
    items: ResponseInputParam, upto: int, needed: int, *, placeholders: bool = True
) -> int:
    """Shorten, then drop, tool results among the first `upto` items, oldest
    first, until `needed` tokens (estimated) are freed or none is left.

    The calls stay, so the model still sees what it did and every call keeps its
    output, which the API requires. An item is replaced, not edited: the items
    of earlier turns are shared with the history.

    A picture goes first: it is the most for the least the model loses (the text
    that came with it stays). Then a long result is shortened to its start and end
    (PRUNE_CHARS), which often frees enough and leaves the model something to read;
    only what is still needed after that is replaced by the placeholder, unless
    `placeholders` is false.

    Returns:
        The estimated number of tokens freed.
    """
    plain = as_dicts(items)
    freed = 0

    def replace(wanted: Callable[[Any], bool], new: Callable[[Any], str]) -> None:
        nonlocal freed
        for i in range(upto):
            item = plain[i]
            if freed >= needed:
                return
            if item.get("type") == "function_call_output" and wanted(item["output"]):
                replaced = {**item, "output": new(item["output"])}
                freed += estimate_tokens(item) - estimate_tokens(replaced)
                plain[i] = replaced

    replace(_has_images, lambda output: f"{_text_parts(output)}\n{IMAGE_DROPPED}")
    replace(
        lambda output: isinstance(output, str) and len(output) > PRUNE_CHARS, shorten
    )
    if placeholders:
        replace(lambda output: output != DROPPED_OUTPUT, lambda output: DROPPED_OUTPUT)
    return freed
