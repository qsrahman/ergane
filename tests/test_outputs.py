from typing import Any

from openai.types.responses import ResponseInputParam

from ergane.context import IMAGE_TOKENS, as_dicts, estimate_tokens
from ergane.outputs import (
    DROPPED_OUTPUT,
    IMAGE_DROPPED,
    IMAGE_REFUSED,
    PRUNE_CHARS,
    drop_images,
    drop_outputs,
    function_output,
    output_text,
)
from ergane.tools import Image

PICTURE = Image("A" * 100_000, "image/png")


def shot_output(text: str = "the page") -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": "c1",
        "output": function_output(text, (PICTURE,)),
    }


class TestPictures:
    def test_a_result_without_pictures_is_text(self):
        assert function_output("hi") == "hi"

    def test_a_result_with_pictures_is_the_list_of_parts_the_api_takes(self):
        output = function_output("hi", (Image("aGk=", "image/jpeg"),))

        assert output == [
            {"type": "input_text", "text": "hi"},
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,aGk=",
                "detail": "auto",
            },
        ]

    def test_the_text_of_an_output_marks_where_a_picture_is(self):
        assert output_text("hi") == "hi"
        assert output_text(function_output("hi", (PICTURE,))) == "hi\n[image]"  # pyright: ignore[reportArgumentType]

    def test_a_picture_counts_as_its_tokens_not_as_its_bytes(self):
        with_picture = estimate_tokens(shot_output())
        without = estimate_tokens(
            {"type": "function_call_output", "call_id": "c1", "output": "the page"}
        )

        assert IMAGE_TOKENS < with_picture - without < IMAGE_TOKENS + 50


class TestRefusedPictures:
    def test_every_picture_goes_and_the_text_stays(self):
        plain = {"type": "function_call_output", "call_id": "c2", "output": "text"}
        items: ResponseInputParam = [shot_output("the page"), plain]  # pyright: ignore[reportAssignmentType]

        dropped = drop_images(items)

        assert dropped == 1
        assert as_dicts(items)[0]["output"] == f"the page\n{IMAGE_REFUSED}"
        assert as_dicts(items)[1] is plain  # not touched

    def test_the_items_are_replaced_not_edited(self):
        original = shot_output()
        items: ResponseInputParam = [original]  # pyright: ignore[reportAssignmentType]

        drop_images(items)

        assert (
            original["output"] != as_dicts(items)[0]["output"]
        )  # the history is shared

    def test_nothing_to_drop_is_zero(self):
        items: ResponseInputParam = [{"role": "user", "content": "hi"}]

        assert drop_images(items) == 0


class TestDroppingPictures:
    def items(self) -> ResponseInputParam:
        long = {
            "type": "function_call_output",
            "call_id": "c2",
            "output": "y" * (PRUNE_CHARS + 1000),
        }
        return [shot_output("the page"), long]  # pyright: ignore[reportReturnType]

    def test_a_picture_goes_before_text_is_shortened(self):
        items = self.items()

        freed = drop_outputs(items, 2, IMAGE_TOKENS - 100)

        assert as_dicts(items)[0]["output"] == f"the page\n{IMAGE_DROPPED}"
        assert len(as_dicts(items)[1]["output"]) == PRUNE_CHARS + 1000  # not touched
        assert freed >= IMAGE_TOKENS - 100

    def test_the_result_stays_as_text_for_the_model_to_read(self):
        items = self.items()

        drop_outputs(items, 2, 10)

        assert "the page" in as_dicts(items)[0]["output"]

    def test_pictures_go_in_the_last_resort_too(self):
        items = self.items()

        drop_outputs(items, 2, 10**9, placeholders=False)

        assert as_dicts(items)[0]["output"] == f"the page\n{IMAGE_DROPPED}"
        assert as_dicts(items)[1]["output"] != DROPPED_OUTPUT  # shortened, not replaced

    def test_what_is_still_needed_after_that_is_replaced(self):
        items = self.items()

        drop_outputs(items, 2, 10**9)

        assert [row["output"] for row in as_dicts(items)] == [DROPPED_OUTPUT] * 2
