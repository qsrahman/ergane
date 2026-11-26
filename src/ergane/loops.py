"""Loop detection: a model that makes the same tool step again and again is told, then
stopped.

A turn has no cap on its model calls, so this is what ends a model that is stuck. A step
is one round of tool calls, identified by what was asked and what came back: a model
that retries a failing call gives the same step each time, while one that makes progress
gives different outputs. Adapted from chiron, which measured it on a real model (a tool
that never changes its answer: 6 of 6 turns ended by themselves with the notice, 1 of 6
without it).

The notice goes into the instructions of the next requests, not into a tool result: the
prompt tells the model that tool results are data, not orders. The pure functions here
take the steps of one turn; `Agent` keeps them and decides what to do.
"""

import hashlib
from collections import Counter
from collections.abc import Sequence

# A turn is stopped when one step shows up more than MAX_REPEATS times among its last
# WINDOW steps, and the model is told when it has shown up more than WARN_REPEATS times.
WINDOW = 10
MAX_REPEATS = 5
WARN_REPEATS = 2


def step_signature(calls: Sequence[tuple[str, str, str]]) -> str:
    """Identify one round of tool calls.

    Args:
        calls: For each call, its tool name, its arguments (JSON text) and its output.

    Returns:
        A digest that is the same for the same calls with the same outputs.
    """
    digest = hashlib.sha256()
    for part in (part for call in calls for part in call):
        # A lone surrogate, which an output can hold, has no strict UTF-8 form.
        digest.update(part.encode(errors="surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def is_looping(steps: Sequence[str]) -> bool:
    """Whether one step fills too much of the recent ones."""
    recent = Counter(steps[-WINDOW:])
    return any(count > MAX_REPEATS for count in recent.values())


def last_repeats(steps: Sequence[str]) -> int:
    """How often the newest step shows up among the last WINDOW steps, itself
    included: 1 for a new step, so a model that changes course is not warned any
    more, although its old repeats are still in the window. 0 with no steps."""
    return Counter(steps[-WINDOW:])[steps[-1]] if steps else 0


def loop_note(repeats: int) -> str:
    """What the model reads in its instructions while it repeats itself."""
    return (
        "\n# Notice\n"
        f"In your last steps you made the same tool call with the same result {repeats} "
        "times. Repeating it will not change the result. Try a different approach, or "
        "stop and tell the user what is blocking you.\n"
    )
