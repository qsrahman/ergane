import struct

import pytest

from ergane.builtin._images import picture_type


def chunk(kind: bytes, body: bytes = b"") -> bytes:
    return struct.pack(">I", len(body)) + kind + body + b"\0\0\0\0"


PNG = b"\x89PNG\r\n\x1a\n"
IHDR = chunk(b"IHDR", b"\0" * 13)


class TestWhichFilesArePictures:
    @pytest.mark.parametrize(
        ("head", "mime"),
        [
            (PNG + IHDR + chunk(b"IDAT", b"x") + chunk(b"IEND"), "image/png"),
            (b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
            (b"GIF89a\x01\x00\x01\x00", "image/gif"),
            (b"GIF87a\x01\x00\x01\x00", "image/gif"),
            (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),
        ],
    )
    def test_the_first_bytes_say_what_it_is(self, head, mime):
        assert picture_type(head) == mime

    def test_a_png_with_no_image_data_in_what_was_read_is_still_a_png(self):
        assert picture_type(PNG + IHDR + chunk(b"tEXt", b"k\0v")) == "image/png"

    def test_an_animated_png_is_not_a_picture(self):
        animated = PNG + IHDR + chunk(b"acTL", b"\0" * 8) + chunk(b"IDAT", b"x")

        assert picture_type(animated) is None

    def test_an_animation_chunk_after_the_image_data_does_not_count(self):
        # The data comes first, so this is a still (and acTL there is not a thing).
        assert picture_type(PNG + IHDR + chunk(b"IDAT") + chunk(b"acTL")) == "image/png"

    @pytest.mark.parametrize(
        "head",
        [
            PNG,  # a signature and nothing else
            PNG + chunk(b"tEXt", b"x"),  # no IHDR first
            b"RIFF\x24\x00\x00\x00WAVEfmt ",  # a RIFF file that is not a picture
            b"BM" + b"\0" * 30,  # a bitmap: the API does not take it
            b"GIF90a",
            b"hello, world\n",
            b"",
        ],
    )
    def test_anything_else_is_not(self, head):
        assert picture_type(head) is None
