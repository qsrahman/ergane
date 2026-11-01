import re

from muagent.demo_tools import add, get_time
from muagent.tools import TOOLS


def test_add():
    assert add.call('{"a": 17, "b": 25}') == "42"


def test_get_time_defaults_to_utc():
    result = get_time.call("{}")

    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", result)


def test_get_time_in_a_zone():
    assert get_time.call('{"timezone": "Asia/Tokyo"}').endswith("+09:00")


def test_an_unknown_zone_comes_back_as_text_for_the_model():
    result = get_time.call('{"timezone": "Tokyo"}')

    assert result.startswith("Error: ZoneInfoNotFoundError:")
    assert "Tokyo" in result


def test_they_are_registered_as_read_only_tools():
    assert TOOLS["add"].read_only
    assert TOOLS["get_time"].read_only
