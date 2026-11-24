from ergane.markdown import Markdown

BOLD = "class:md-bold"
CODE = "class:md-code"
FENCE = "class:fence"


def feed(*pieces: str) -> list[tuple[str, str]]:
    markdown = Markdown()
    return [piece for text in pieces for piece in markdown.feed(text)]


class TestInline:
    def test_bold_and_code_are_styled_with_their_marks_inside(self):
        assert feed("a **b** and `c` d\n") == [
            ("", "a "),
            (BOLD, "**"),
            (BOLD, "b"),
            (BOLD, "**"),
            ("", " and "),
            (CODE, "`"),
            (CODE, "c"),
            (CODE, "`"),
            ("", " d\n"),
        ]

    def test_text_is_never_changed_only_split(self):
        text = "**a** `b` # c\n```\nd **e**\n```\n- f\n"
        assert "".join(t for _, t in feed(text)) == text

    def test_a_mark_stays_open_across_pieces_of_a_line(self):
        assert feed("**bo", "ld** x\n") == [
            (BOLD, "**"),
            (BOLD, "bo"),
            (BOLD, "ld"),
            (BOLD, "**"),
            ("", " x\n"),
        ]

    def test_a_mark_left_open_ends_with_its_line(self):
        assert feed("**open\nnext `x\nlast\n") == [
            (BOLD, "**"),
            (BOLD, "open\n"),
            ("", "next "),
            (CODE, "`"),
            (CODE, "x\n"),
            ("", "last\n"),
        ]

    def test_bold_marks_inside_code_are_code(self):
        assert feed("`a **b** c`\n")[1:4] == [
            (CODE, "a "),
            (CODE, "**"),
            (CODE, "b"),
        ]

    def test_a_single_star_is_not_a_mark(self):
        assert feed("2 * 3 * 4\n") == [("", "2 * 3 * 4\n")]


class TestBlocks:
    def test_a_heading_line_is_bold_whole(self):
        assert feed("## Title with `x`\nbody\n") == [
            (BOLD, "## Title with `x`\n"),
            ("", "body\n"),
        ]

    def test_a_hash_without_a_space_is_no_heading(self):
        assert feed("#hashtag\n") == [("", "#hashtag\n")]

    def test_a_heading_that_arrives_in_pieces_is_bold_from_where_it_is_known(self):
        assert feed("#", "# T", "itle\n") == [
            ("", "#"),
            (BOLD, "# T"),
            (BOLD, "itle\n"),
        ]

    def test_a_code_block_has_dim_fences_and_plain_code(self):
        assert feed("```py\nx = **a** `b` # c\n```\nafter **b**\n") == [
            (FENCE, "```py\n"),
            ("", "x = **a** `b` # c\n"),
            (FENCE, "```\n"),
            ("", "after "),
            (BOLD, "**"),
            (BOLD, "b"),
            (BOLD, "**"),
            ("", "\n"),
        ]

    def test_an_open_block_goes_on_over_pieces(self):
        markdown = Markdown()
        markdown.feed("```\n")

        assert markdown.feed("**x**\n") == [("", "**x**\n")]

    def test_reset_closes_the_block_and_the_marks(self):
        markdown = Markdown()
        markdown.feed("```\n")
        markdown.feed("**half")
        markdown.reset()

        assert markdown.feed("**x**\n")[0] == (BOLD, "**")
