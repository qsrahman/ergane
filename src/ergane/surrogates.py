"""Lone surrogates: code points that text can hold and UTF-8 cannot.

JSON can write one (`"\\ud800"`), so a reply, a web page or the user's input can put one
in the history. A request that holds one cannot be sent, and a session file with one
cannot be read back, so whatever sends or saves the history passes it through
`without_surrogates`. Imports nothing of ergane.
"""

import json


def without_surrogates[T](value: T) -> T:
    """A JSON value with each lone surrogate in its text replaced by `?`.

    Args:
        value: Anything JSON can hold. It comes back as JSON reads it, so a tuple
            comes back as a list.

    Returns:
        A copy of the value that can be encoded as UTF-8.
    """
    return json.loads(json.dumps(value, ensure_ascii=False).encode(errors="replace"))
