"""Property-based tests (Hypothesis) of the functions that take text from outside: the
model, a file, a command or a web page. Each says what must hold for any input.

`uv run pytest tests/test_properties.py --hypothesis-profile=thorough` tries many more
inputs (the profiles are in `conftest.py`).
"""

import gzip
import io
import json
import re
import unicodedata
import zlib
from itertools import pairwise
from typing import Any

import httpx2
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from ergane.builtin._edit import EditError, edit_text
from ergane.builtin._text import MAX_LINE_CHARS, head_tail
from ergane.builtin.fetch import (
    FetchError,
    _decode_page,
    _read_capped,
    extract_content_from_html,
)
from ergane.builtin.files import _read_lines
from ergane.builtin.search import MAX_RESULTS, _format, _parse, _target
from ergane.builtin.shell import _clean, _result
from ergane.builtin.writes import edit_file, write_file
from ergane.console import sanitize
from ergane.context import Context, as_dicts, estimate_tokens
from ergane.loops import step_signature
from ergane.markdown import Markdown
from ergane.outputs import (
    DROPPED_OUTPUT,
    answer_open_calls,
    drop_images,
    drop_outputs,
    function_output,
)
from ergane.sessions import SessionStore
from ergane.surrogates import without_surrogates
from ergane.tools import MAX_OUTPUT_CHARS, Image

# Any character, lone surrogates included: JSON can carry one ("\ud800"), so the model, a
# server or a tool's arguments can hold it. `st.text()` alone leaves them out.
any_text = st.text(st.one_of(st.characters(), st.characters(categories=["Cs"])))
# Few characters, so that old_text is often found; with what the second try forgives.
edit_chars = st.sampled_from(list("ab \t\n'\"-‘’“”— "))
edit_pieces = st.text(edit_chars, max_size=30)
plain_lines = st.lists(st.text(st.sampled_from("ab c"), max_size=5), max_size=8)


# --- edit_text ------------------------------------------------------------------------


@given(any_text, any_text, any_text, st.booleans())
def test_edit_raises_only_edit_error(raw, old, new, replace_all):
    try:
        edit_text(raw, old, new, replace_all, "f")
    except EditError:
        pass


@given(
    edit_pieces, edit_pieces, edit_pieces, edit_pieces, st.sampled_from(["\n", "\r\n"])
)
def test_edit_leaves_the_rest_and_the_line_endings(before, old, after, new, ending):
    text = before + old + after
    assume(old and old != new and "\n" in text)
    # The one place `old` is: where it was put.
    assume(text.find(old) == len(before) and text.find(old, len(before) + 1) == -1)
    raw = text.replace("\n", ending)
    result, count = edit_text(
        raw, old.replace("\n", ending), new.replace("\n", ending), False, "f"
    )
    assert result == (before + new + after).replace("\n", ending)
    assert count == 1


@example(raw="def f():\n\n    pass", old="\t")
@given(edit_pieces, st.text(st.sampled_from(" \t\u00a0"), min_size=1, max_size=6))
def test_edit_does_not_match_blank_text_that_is_not_there(raw, old):
    assume(old not in raw)
    try:
        result, _ = edit_text(raw, old, "X", False, "f")
    except EditError:
        return
    raise AssertionError(f"edited a text without old_text: {result!r}")


@example(before=[], block=["a"], after=[], old_newlines=0, new_newlines=2)
@given(plain_lines, plain_lines, plain_lines, st.integers(0, 3), st.integers(0, 3))
def test_edit_second_try_gives_what_an_exact_edit_gives(
    before, block, after, old_newlines, new_newlines
):
    """old_text with a trailing space (so only the second try finds it) must make the
    same file as the exact old_text, newlines at the end of new_text included."""
    assume(any(line.strip() for line in block))
    # Without a newline for the one old_text ends in, an exact edit joins two lines;
    # the second try keeps whole lines, on purpose.
    assume(new_newlines or not old_newlines)
    raw = "\n".join([*before, *block, *after]) + "\n"
    old = "\n".join(block) + "\n" * min(old_newlines, 1)
    new = "X" + "\n" * new_newlines
    sloppy = "\n".join(line + " " for line in block) + "\n" * min(old_newlines, 1)
    # The exact edit must be of the block itself, not of text that starts inside a line.
    at = len("".join(line + "\n" for line in before))
    places = [i for i in range(len(raw)) if raw.startswith(old, i)]
    assume(places == [at] and sloppy not in raw)
    # No other run of lines may look like the block once trailing spaces are ignored.
    plain = [line.rstrip() for line in raw.split("\n")]
    wanted = [line.rstrip() for line in block]
    runs = [plain[i : i + len(wanted)] for i in range(len(plain))]
    assume(runs.count(wanted) == 1)
    try:
        fuzzy, _ = edit_text(raw, sloppy, new, False, "f")
    except EditError:
        assume(False)
        return
    exact, _ = edit_text(raw, old, new, False, "f")
    assert fuzzy == exact


# One letter and newlines, so that old_text is often in several places, overlapping ones
# included.
@given(
    st.text(st.sampled_from("a\n"), min_size=8, max_size=20),
    st.text(st.sampled_from("a\n"), min_size=1, max_size=3),
)
def test_edit_without_replace_all_needs_old_text_in_one_place_only(raw, old):
    places = [at for at in range(len(raw)) if raw.startswith(old, at)]
    assume(len(places) > 1)
    with pytest.raises(EditError, match=f"matches {len(places)} places"):
        edit_text(raw, old, "X", False, "f")


# --- The files that are written -------------------------------------------------------


@given(content=any_text)
def test_write_writes_all_or_nothing(tmp_path_factory, content):
    path = tmp_path_factory.getbasetemp() / "property-write.txt"
    path.write_bytes(b"precious")
    write_file.call(json.dumps({"path": str(path), "content": content}))
    assert path.read_bytes() in (b"precious", content.encode("utf-8", "surrogatepass"))


@given(new=any_text)
def test_edit_writes_all_or_nothing(tmp_path_factory, new):
    path = tmp_path_factory.getbasetemp() / "property-edit.txt"
    path.write_bytes(b"keep OLD keep")
    edit_file.call(json.dumps({"path": str(path), "old_text": "OLD", "new_text": new}))
    left = path.read_bytes()
    assert left.startswith(b"keep ") and left.endswith(b" keep"), left


# --- Markdown and sanitize ------------------------------------------------------------

markdown_line = st.lists(
    st.sampled_from(["a", "b", " ", "*", "**", "`", "```", "#"]), max_size=5
).map("".join)
markdown_text = st.lists(st.one_of(st.just("```"), markdown_line), max_size=12).map(
    "\n".join
)


def _chunks(text: str, cuts: list[int]) -> list[str]:
    points = sorted({0, len(text), *(c for c in cuts if c < len(text))})
    return [text[a:b] for a, b in pairwise(points)]


@given(st.one_of(any_text, markdown_text), st.lists(st.integers(0, 60)))
def test_markdown_keeps_every_character(text, cuts):
    markdown = Markdown()
    pieces = [p for chunk in _chunks(text, cuts) for p in markdown.feed(chunk)]
    assert "".join(part for _, part in pieces) == text
    styles = {"", "class:fence", "class:md-bold", "class:md-code"}
    assert {style for style, _ in pieces} <= styles


@given(markdown_text, st.lists(st.integers(0, 60)))
def test_markdown_code_blocks_do_not_depend_on_the_chunks(text, cuts):
    whole, parts = Markdown(), Markdown()
    whole.feed(text)
    for chunk in _chunks(text, cuts):
        parts.feed(chunk)
    fences = sum(line.lstrip().startswith("```") for line in text.split("\n")[:-1])
    assert whole.fenced == (fences % 2 == 1)
    assert (parts.fenced, parts.line) == (whole.fenced, whole.line)


# What `sanitize` is for, often enough to be seen among all the other characters.
hostile = st.sampled_from(
    ["\x1b", "\x07", "\r", "\x9b", "\u061c", "\u200e", "\u202e", "\u2067", "\U000e0041"]
)


@given(st.lists(st.one_of(hostile, any_text), max_size=8).map("".join))
def test_sanitize_leaves_nothing_that_moves_or_reorders_the_terminal(text):
    clean = sanitize(text)
    for char in clean:
        assert char in "\n\t" or unicodedata.category(char) != "Cc", repr(char)
        assert unicodedata.bidirectional(char) not in {
            "LRE",
            "RLE",
            "LRO",
            "RLO",
            "PDF",
            "LRI",
            "RLI",
            "FSI",
            "PDI",
        }, repr(char)
        assert not 0xE0000 <= ord(char) <= 0xE007F
    assert sanitize(clean) == clean
    rest = iter(text)
    assert all(char in rest for char in clean)  # nothing added or moved


# --- Reading and cutting text ---------------------------------------------------------

file_line = st.builds(
    lambda char, n, tail: char * n + tail,
    st.sampled_from(["a", "é", "€", "😀", " ", " ", "\x0c"]),
    st.one_of(st.integers(0, 20), st.integers(3_000, 12_000)),
    st.text(st.sampled_from("xyz "), max_size=3),
)
_FOOTER = re.compile(r"\n\[\.\.\. showing lines \d+-\d+; .*offset=(\d+)\]\Z")


@given(
    st.lists(file_line, max_size=12),
    st.sampled_from(["\n", "\r\n"]),
    st.booleans(),
    st.one_of(st.none(), st.integers(1, 5)),
)
def test_reading_in_parts_gives_the_whole_file(lines, ending, final, limit):
    content = ending.join(lines) + (ending if final and lines else "")
    raw = content.encode()
    expected = content.replace("\r\n", "\n").split("\n")
    if expected[-1] == "":
        expected.pop()
    expected = [
        line
        if len(line) <= MAX_LINE_CHARS
        else f"{line[:MAX_LINE_CHARS]}... [line cut: {len(line):,} characters]"
        for line in expected
    ]
    got: list[str] = []
    offset = 1
    while True:
        result = _read_lines(io.BytesIO(raw), "f", raw[:8192], offset, limit)
        assert len(result) <= MAX_OUTPUT_CHARS
        footer = _FOOTER.search(result)
        text = result[: footer.start()] if footer else result
        if not expected:
            assert result == "(empty file)"
            return
        got += text.split("\n")
        if footer is None:
            break
        assert int(footer[1]) > offset
        offset = int(footer[1])
    # With a `limit` of its own the model is not told there is more: it has at least
    # what it asked for, in order.
    assert got == expected[: len(got)]
    assert len(got) >= (len(expected) if limit is None else min(limit, len(expected)))


long_text = st.builds(
    lambda start, piece, n, end: start + piece * n + end,
    st.text(max_size=20),
    st.text(min_size=1, max_size=5),
    st.integers(0, 300),
    st.text(max_size=20),
)


@given(long_text, st.integers(100, 400))
def test_head_tail_keeps_the_limit_and_both_ends(text, limit):
    result = head_tail(text, limit)
    assert len(result) <= limit
    if len(text) <= limit:
        assert result == text
    else:
        assert text.startswith(result[:10]) and text.endswith(result[-10:])


# --- Command output -------------------------------------------------------------------

output_bytes = st.one_of(
    st.binary(max_size=200),
    st.builds(
        lambda piece, n: piece * n,
        st.binary(min_size=1, max_size=8),
        st.integers(1, 6_000),
    ),
)


@given(output_bytes)
def test_clean_takes_any_bytes(data):
    clean = _clean(data)
    assert "\r" not in clean
    assert b"\0" in data or "\x1b[0m" not in clean


@given(output_bytes, output_bytes, st.integers(-15, 255))
def test_a_command_result_fits_and_encodes(out, err, code):
    result = _result(code, out, err, ["a note"])
    assert len(result) <= MAX_OUTPUT_CHARS
    result.encode("utf-8")


# --- Web pages ------------------------------------------------------------------------

link = st.one_of(
    st.text(max_size=40),
    st.builds(
        "{}{}".format,
        st.sampled_from(
            ["http://", "https://", "//", "//duckduckgo.com/l/?uddg=", "/", "x:"]
        ),
        st.text(st.sampled_from(list("a.[]:/?#@%=&0 é\\")), max_size=20),
    ),
)


@example("http://[")
@given(link)
def test_any_result_link_is_judged_not_raised(href):
    address = _target(href)
    assert address is None or address.lower().startswith(("http://", "https://"))


tag = st.sampled_from(
    [
        "div",
        "a",
        "p",
        "main",
        "article",
        "nav",
        "title",
        "body",
        "table",
        "tr",
        "td",
        "ul",
        "li",
        "pre",
        "code",
        "h1",
        "img",
        "span",
        "svg",
        "br",
        "ol",
        "template",
    ]
)
attribute = st.one_of(
    st.just(""),
    st.sampled_from(
        [
            'class="result"',
            'class="result__a"',
            "hidden",
            'role="main"',
            "colspan=x",
            'colspan="0"',
            'start="x"',
            'start="99999999999999999999"',
            "alt",
            "href",
            'style="display:none"',
            "class",
        ]
    ),
    link.map(lambda value: f'href="{value}"'),
    link.map(lambda value: f'src="{value}"'),
)
markup = st.recursive(
    st.text(max_size=15),
    lambda inner: st.builds(
        lambda name, attrs, children: (
            f"<{name} {' '.join(attrs)}>{''.join(children)}</{name}>"
        ),
        tag,
        st.lists(attribute, max_size=3),
        st.lists(inner, max_size=4),
    ),
    max_leaves=15,
)
html = st.one_of(any_text, markup, st.lists(markup, max_size=3).map("".join))


@example('<a href="http://[">x</a>')
@given(html)
def test_simplifying_any_page_fails_only_in_words(page):
    try:
        # As `_decode_page` hands it over: without lone surrogates.
        extract_content_from_html(without_surrogates(page), "https://example.com/a/b")
    except FetchError:
        pass


result_block = st.builds(
    '<div class="result"><a class="result__a" href="{}">{}</a>'
    '<span class="result__snippet">{}</span></div>'.format,
    link,
    markup,
    markup,
)


@example('<div class="result"><a class="result__a" href="http://[">x</a>')
@example("\ud800")
@given(st.one_of(html, st.lists(result_block, max_size=25).map("".join)))
def test_parsing_any_search_page(page):
    results = _parse(page, MAX_RESULTS)
    assert len(results) <= MAX_RESULTS
    assert len(_format("q", results)) <= MAX_OUTPUT_CHARS


charsets = st.sampled_from(
    [
        "utf-8",
        "utf-16",
        "utf-32",
        "latin-1",
        "ascii",
        "shift_jis",
        "gb18030",
        "idna",
        "punycode",
        "unicode_escape",
        "raw_unicode_escape",
        "rot13",
        "base64",
        "zip",
        "hex",
        "mbcs",
        "undefined",
        "utf-7",
        "nonsense",
        "",
    ]
)
body = st.one_of(
    st.binary(max_size=100),
    st.sampled_from([rb"<p>\ud800</p>", b"\xff\xfe\x00\xd8", b"+2AA-"]),
    st.builds(
        lambda name, rest: b'<meta charset="' + name.encode() + b'">' + rest,
        charsets,
        st.binary(max_size=50),
    ),
)


@example(content=rb"\ud800", charset="unicode_escape")
@given(body, charsets)
def test_any_body_decodes_to_text_that_can_be_sent(content, charset):
    response = httpx2.Response(
        200, headers={"content-type": f"text/html; charset={charset}"}
    )
    text = _decode_page(response, content)
    text.encode("utf-8")


def _raw_deflate(data: bytes) -> bytes:
    z = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return z.compress(data) + z.flush()


_CODINGS = {
    "identity": lambda data: data,
    "gzip": gzip.compress,
    "deflate": zlib.compress,
    "raw deflate": _raw_deflate,
}


def _streamed(coding: str, chunks: list[bytes]) -> httpx2.Response:
    header = "deflate" if coding == "raw deflate" else coding
    return httpx2.Response(
        200, headers={"content-encoding": header}, content=iter(chunks)
    )


@example(coding="raw deflate", data=b"hello " * 10, cuts=[1], limit=400)
@given(
    st.sampled_from(sorted(_CODINGS)),
    st.binary(max_size=300),
    st.lists(st.integers(0, 320), max_size=6),
    st.integers(0, 400),
)
def test_a_body_is_decoded_however_it_arrives(coding, data, cuts, limit):
    sent = _CODINGS[coding](data)
    points = sorted({0, len(sent), *(c for c in cuts if c < len(sent))})
    chunks = [sent[a:b] for a, b in pairwise(points)]
    body, truncated = _read_capped(_streamed(coding, chunks), limit)
    assert (body, truncated) == (data[:limit], len(data) > limit)


@given(st.sampled_from(sorted(_CODINGS)), st.lists(st.binary(max_size=60), max_size=5))
def test_garbage_for_a_body_is_a_decoding_error(coding, chunks):
    try:
        body, _ = _read_capped(_streamed(coding, chunks), 100)
    except httpx2.DecodingError:
        return
    assert len(body) <= 100


# --- The history ----------------------------------------------------------------------

result_text = st.one_of(
    st.text(max_size=40), st.integers(7_900, 20_000).map(lambda n: "r" * n)
)
tool_result = st.tuples(result_text, st.booleans(), st.booleans())  # image, answered


@example("\ud800")
@given(any_text)
def test_any_text_can_be_measured_sent_and_saved(tmp_path_factory, text):
    items = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": text},
    ]
    assert estimate_tokens(items[0]) > 0
    step_signature([("tool", "{}", text)])
    json.dumps(without_surrogates(items), ensure_ascii=False).encode()
    root = tmp_path_factory.getbasetemp() / "property-sessions"
    store = SessionStore(root, root)
    saved = store.save(None, text, items, "m")
    loaded = store.load(saved.id)
    assert [loaded.summary, loaded.items] == without_surrogates([text, items])


class History(RuleBasedStateMachine):
    """A Context driven as the agent drives it, with the invariants of CLAUDE.md."""

    def __init__(self) -> None:
        super().__init__()
        self.context = Context()
        self.calls = 0

    @rule(
        prompt=st.text(max_size=30),
        rounds=st.lists(st.lists(tool_result, min_size=1, max_size=3), max_size=3),
        budget=st.integers(0, 20_000),
        make_room=st.integers(0, 20_000),
        placeholders=st.booleans(),
        refuse_pictures=st.booleans(),
    )
    def turn(self, prompt, rounds, budget, make_room, placeholders, refuse_pictures):
        before = json.dumps(self.context.items)
        items, dropped = self.context.begin_turn(prompt, budget)
        summary = self.context.summary_message()
        assert items[: len(summary)] == summary
        assert items[-1] == {"role": "user", "content": prompt}
        kept: Any = items[len(summary) :]
        assert kept[0]["role"] == "user"
        if sum(map(estimate_tokens, as_dicts(items))) > budget:
            assert len([i for i in kept if i.get("role") == "user"]) == 1
        assert not dropped or len(kept) < len(self.context.items) + 1
        plain = as_dicts(items)
        for round_ in rounds:
            outputs = []
            for text, picture, answered in round_:
                self.calls += 1
                call_id = f"c{self.calls}"
                plain.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "t",
                        "arguments": "{}",
                    }
                )
                if answered:
                    images = [Image("image/png", "AAAA")] if picture else []
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": function_output(text, images),
                        }
                    )
            plain.extend(outputs)
            answer_open_calls(items)
        calls = [i for i in plain if i.get("type") == "function_call"]
        if refuse_pictures:
            drop_images(items)
        size = sum(map(estimate_tokens, plain))
        freed = drop_outputs(items, len(items), make_room, placeholders=placeholders)
        assert freed == size - sum(map(estimate_tokens, plain))
        assert [i for i in plain if i.get("type") == "function_call"] == calls
        if placeholders and freed < make_room:
            assert all(
                i["output"] == DROPPED_OUTPUT
                for i in plain
                if i.get("type") == "function_call_output"
            )
        assert json.dumps(self.context.items) == before  # shared items not edited
        plain.append({"type": "message", "role": "assistant", "content": "reply"})
        self.context.save_turn(items)

    @rule(keep=st.integers(0, 20_000), summary=st.text(max_size=20))
    def compact(self, keep, summary):
        count = self.context.plan_compaction(keep)
        assert 0 <= count <= len(self.context.items)
        if count:
            assert as_dicts(self.context.items)[count]["role"] == "user"
            self.context.compact(count, summary)

    @rule()
    def undo(self):
        had = bool(self.context.items)
        assert self.context.undo() == had

    @precondition(lambda self: self.context.last_prompt is not None)
    @rule()
    def rewind(self):
        assert self.context.rewind() is not None

    @rule()
    def restore(self):
        saved = json.loads(json.dumps([self.context.items, self.context.summary]))
        self.context.restore(*saved)
        assert [self.context.items, self.context.summary] == saved

    @rule()
    def reset(self):
        self.context.reset()

    @invariant()
    def calls_and_outputs_pair_up(self):
        items = as_dicts(self.context.items)
        calls = [i["call_id"] for i in items if i.get("type") == "function_call"]
        outputs = [
            i["call_id"] for i in items if i.get("type") == "function_call_output"
        ]
        assert sorted(calls) == sorted(outputs)

    @invariant()
    def messages_alternate_and_only_the_newest_turn_has_tool_work(self):
        items = as_dicts(self.context.items)
        roles = [i["role"] for i in items if "role" in i]
        assert roles == ["user", "assistant"] * (len(roles) // 2)
        starts = [n for n, i in enumerate(items) if i.get("role") == "user"]
        newest = starts[-1] if starts else 0
        assert all("role" in i for i in items[:newest])


TestHistory = History.TestCase
