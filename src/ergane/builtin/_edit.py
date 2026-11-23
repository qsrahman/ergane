"""The matching and replacing behind `edit_file`: pure text in, text out.

A replacement is made where `old` is found exactly. When it is not found, the lines are
compared again after the differences that a model gets wrong when it copies text are
removed: trailing whitespace, typographic quotes, Unicode dashes and Unicode spaces
(opencode and pi both do this). Only whole lines are compared in that second try, and
the first and last lines of the edit that the model left unchanged are taken from the
file as they are, so a curly quote in an untouched line is not rewritten.

A text with CRLF line endings is edited as LF and written back as CRLF; a file that mixes
the two is written back with the one it has most of.

Nothing here touches a file. The messages are for the model.
"""

import difflib


class EditError(ValueError):
    """An edit that cannot be made. The message tells the model why, and what to do."""


_QUOTES = str.maketrans("‘’‚‛“”„‟", "''''\"\"\"\"")
_DASHES = str.maketrans(dict.fromkeys("‐‑‒–—―−", "-"))
_SPACES = str.maketrans(dict.fromkeys("      　", " "))

_MAX_LISTED = 10  # line numbers named in an "ambiguous" error
_HINT_LINES = 5  # lines of the file shown when nothing matches
_HINT_WIDTH = 200
_SIMILAR = 0.6  # how alike a line must be to the start of old_text to be shown


def _plain(line: str) -> str:
    """A line as it is compared in the second try."""
    return line.rstrip().translate(_QUOTES).translate(_DASHES).translate(_SPACES)


def _exact(text: str, old: str) -> list[int]:
    """Where `old` is in `text`, without overlaps."""
    found = []
    at = text.find(old)
    while at != -1:
        found.append(at)
        at = text.find(old, at + len(old))
    return found


def _line_matches(lines: list[str], wanted: list[str]) -> list[int]:
    """The first lines of the runs of `lines` that equal `wanted` once plain, without
    overlaps."""
    plain = [_plain(line) for line in wanted]
    found = []
    i = 0
    while i <= len(lines) - len(wanted):
        if all(_plain(lines[i + j]) == plain[j] for j in range(len(wanted))):
            found.append(i)
            i += len(wanted)
        else:
            i += 1
    return found


def _kept_lines(old: list[str], new: list[str], matched: list[str]) -> list[str]:
    """The replacement for a run of file lines found in the second try: the new lines,
    except that the unchanged first and last ones are the file's own."""
    shared = min(len(old), len(new))
    head = 0
    while head < shared and old[head] == new[head]:
        head += 1
    tail = 0
    while tail < shared - head and old[-1 - tail] == new[-1 - tail]:
        tail += 1
    return [
        *matched[:head],
        *new[head : len(new) - tail],
        *matched[len(matched) - tail :],
    ]


def _closest(lines: list[str], old: str) -> str:
    """A few lines of the file that look like the start of `old`, with their numbers.

    The first of the lines that are equally close wins, and the empty piece after a final
    newline is not a line."""
    if lines and lines[-1] == "":
        lines = lines[:-1]
    first = next((line.strip() for line in old.split("\n") if line.strip()), "")
    matcher = difflib.SequenceMatcher()
    matcher.set_seq2(first)
    best, best_score = -1, 0.0
    for index, line in enumerate(lines):
        matcher.set_seq1(line.strip())
        if (
            matcher.real_quick_ratio() >= _SIMILAR
            and matcher.quick_ratio() >= _SIMILAR
            and (score := matcher.ratio()) >= _SIMILAR
            and score > best_score
        ):
            best, best_score = index, score
    if best < 0:
        return ""
    shown = [
        f"{number}: {line[:_HINT_WIDTH]}"
        for number, line in enumerate(lines[best : best + _HINT_LINES], best + 1)
    ]
    return "\n\nThe closest lines in the file:\n" + "\n".join(shown)


def edit_text(
    raw: str, old: str, new: str, replace_all: bool, shown: str
) -> tuple[str, int]:
    """Replace `old` with `new` in a file's text.

    Args:
        raw: The text of the file.
        old: The text to find. It must be found once, unless `replace_all`.
        new: What to put in its place.
        replace_all: Replace every place `old` is found.
        shown: The path as the model wrote it, for the messages.

    Returns:
        The new text, and how many places were replaced.

    Raises:
        EditError: If `old` is empty or the same as `new`, is not found, is found in
            several places and `replace_all` is not set, or the edit changes nothing.
    """
    if old == "":
        raise EditError("old_text must not be empty; to create a file, use write_file")
    if old == new:
        raise EditError("No change: old_text and new_text are the same")
    crlf = raw.count("\r\n") * 2 > raw.count("\n")
    text = raw.replace("\r\n", "\n")
    old = old.replace("\r\n", "\n")
    new = new.replace("\r\n", "\n")

    found = _exact(text, old)
    if found:
        if len(found) > 1 and not replace_all:
            at = [text.count("\n", 0, start) + 1 for start in found]
            raise _ambiguous(at)
        result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        count = len(found) if replace_all else 1
    else:
        # The newlines at the ends are not part of the comparison of lines.
        old_lines = old.rstrip("\n").split("\n")
        new_lines = new.rstrip("\n").split("\n") if new.strip("\n") else []
        lines = text.split("\n")
        starts = _line_matches(lines, old_lines)
        if not starts:
            raise EditError(
                f"Could not find old_text in {shown}. It must match exactly, including "
                f"whitespace and indentation.{_closest(lines, old)}"
            )
        if len(starts) > 1 and not replace_all:
            raise _ambiguous([start + 1 for start in starts])
        chosen = starts if replace_all else starts[:1]
        for start in reversed(chosen):  # last first, so earlier numbers stay right
            end = start + len(old_lines)
            lines[start:end] = _kept_lines(old_lines, new_lines, lines[start:end])
        result = "\n".join(lines)
        count = len(chosen)

    if result == text:
        raise EditError(f"No change: the edit leaves {shown} as it was")
    return (result.replace("\n", "\r\n") if crlf else result), count


def _ambiguous(lines: list[int]) -> EditError:
    listed = ", ".join(str(n) for n in lines[:_MAX_LISTED])
    if len(lines) > _MAX_LISTED:
        listed += ", ..."
    return EditError(
        f"old_text matches {len(lines)} places (lines {listed}). Add surrounding lines "
        "to old_text so that it matches once, or set replace_all to true to replace "
        "every one. Nothing was changed."
    )
