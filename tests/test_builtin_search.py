"""web_search, against a small server on this machine that answers like DuckDuckGo's
HTML endpoint (`html.duckduckgo.com/html`) with a page of results."""

import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import pytest

from ergane.builtin.search import WEB_SEARCH, make_search_tool
from ergane.tools import MAX_OUTPUT_CHARS


def result(title: str, href: str, snippet: str = "", classes: str = "result") -> str:
    return (
        f'<div class="{classes} results_links web-result"><div class="result__body">'
        f'<h2 class="result__title"><a class="result__a" href="{href}" '
        f'rel="nofollow">{title}</a></h2>'
        f'<a class="result__snippet" href="{href}">{snippet}</a></div></div>'
    )


def page(*results: str) -> str:
    return f"<html><body><div class='results'>{''.join(results)}</div></body></html>"


PYTHON = result(
    "What&#x27;s new in <b>Python</b> 3.14",
    "https://docs.python.org/3/whatsnew/3.14.html",
    "Summary - <b>Release</b> highlights\n   of  <b>Python</b> 3.14.",
)
PAGE = page(
    PYTHON,
    result("Second", "https://example.org/two", "The second snippet."),
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
        self.server.seen.append((self.path, dict(self.headers), form))  # pyright: ignore[reportAttributeAccessIssue]
        if self.server.delay:  # pyright: ignore[reportAttributeAccessIssue]
            time.sleep(self.server.delay)  # pyright: ignore[reportAttributeAccessIssue]
        status, body = self.server.reply  # pyright: ignore[reportAttributeAccessIssue]
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=UTF-8")
        self.end_headers()
        self.wfile.write(body.encode())


@pytest.fixture
def site() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.seen = []  # pyright: ignore[reportAttributeAccessIssue]
    server.reply = (200, PAGE)  # pyright: ignore[reportAttributeAccessIssue]
    server.delay = 0  # pyright: ignore[reportAttributeAccessIssue]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def tool(site):
    return make_search_tool(
        base_url=f"http://127.0.0.1:{site.server_address[1]}/html", timeout_seconds=5
    )


def search(tool, query: str = "python", **arguments) -> str:
    return tool.call(json.dumps({"query": query, **arguments}))


class TestTheRequest:
    def test_the_query_is_posted_as_the_form_the_page_posts(self, tool, site):
        search(tool, "python 3.14 release notes")

        [(path, _, form)] = site.seen
        assert path == "/html"
        assert form == {
            "q": ["python 3.14 release notes"],
            "b": [""],
            "kl": [""],
            "kp": ["-1"],  # moderate safe search
        }

    def test_characters_that_mean_something_in_a_form_arrive_whole(self, tool, site):
        search(tool, "c++ & rust = 100% café 中文")

        assert site.seen[0][2]["q"] == ["c++ & rust = 100% café 中文"]

    def test_it_says_it_is_ergane_and_wears_no_browsers_clothes(self, tool, site):
        search(tool)

        headers = site.seen[0][1]
        assert headers["User-Agent"].startswith("ergane/")
        assert "Autonomous" in headers["User-Agent"]
        assert not any(name.lower().startswith("sec-") for name in headers)
        assert "Mozilla" not in headers["User-Agent"]

    def test_the_proxy_variables_are_not_read(self, tool, site, monkeypatch):
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, "http://127.0.0.1:9")

        assert "web results" in search(tool)

    def test_it_asks_before_it_goes_out(self):
        assert WEB_SEARCH.read_only is False and WEB_SEARCH.name == "web_search"


class TestTheResults:
    def test_they_are_numbered_with_title_address_and_snippet(self, tool):
        assert search(tool, "python") == (
            '2 web results for "python":\n\n'
            "1. What's new in Python 3.14\n"
            "   https://docs.python.org/3/whatsnew/3.14.html\n"
            "   Summary - Release highlights of Python 3.14.\n\n"
            "2. Second\n"
            "   https://example.org/two\n"
            "   The second snippet."
        )

    def test_a_result_with_no_snippet_has_two_lines(self, tool, site):
        site.reply = (200, page(result("Bare", "https://bare.test/")))  # pyright: ignore[reportAttributeAccessIssue]

        assert search(tool).endswith("1. Bare\n   https://bare.test/")

    def test_an_advertisement_is_left_out(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(
                result(
                    "An ad",
                    "//duckduckgo.com/y.js?ad_domain=ads.test&u3=x",
                    "Buy now",
                    classes="result result--ad",
                ),
                result("Not an ad", "https://real.test/"),
            ),
        )

        text = search(tool)

        assert "An ad" not in text and "ads.test" not in text
        assert text.startswith('1 web results for "python"') and "Not an ad" in text

    def test_an_advertisement_is_known_by_its_class_alone(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(
                result(
                    "Class ad", "https://advertiser.test/", classes="result result--ad"
                )
            ),
        )

        assert "Class ad" not in search(tool)

    def test_an_advertisement_is_known_by_its_link_alone(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(result("Sneaky ad", "https://duckduckgo.com/y.js?ad=1")),
        )

        assert "Sneaky ad" not in search(tool)

    @pytest.mark.parametrize(
        "href",
        [
            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%3Fb%3D1%26c%3D2&rut=abc",
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%3Fb%3D1%26c%3D2&rut=abc",
        ],
    )
    def test_duckduckgos_own_redirect_is_unwrapped(self, tool, site, href):
        site.reply = (200, page(result("Wrapped", href)))  # pyright: ignore[reportAttributeAccessIssue]

        assert "   https://example.org/a?b=1&c=2" in search(tool)

    @pytest.mark.parametrize(
        "href",
        ["javascript:alert(1)", "ftp://files.test/x", "mailto:a@b.test", "/relative"],
    )
    def test_a_link_that_is_not_a_web_address_is_left_out(self, tool, site, href):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(result("Odd", href), result("Fine", "https://fine.test/")),
        )

        text = search(tool)

        assert "Odd" not in text and "Fine" in text

    def test_a_result_without_a_title_link_is_left_out(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page('<div class="result"><p>no link here</p></div>', PYTHON),
        )

        assert search(tool).startswith("1 web results")

    def test_max_results_cuts_the_list(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(*(result(f"R{i}", f"https://r{i}.test/") for i in range(15))),
        )

        text = search(tool, max_results=4)

        assert text.startswith('4 web results for "python"') and "4. R3" in text
        assert "R4" not in text

    def test_ten_is_the_default(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(*(result(f"R{i}", f"https://r{i}.test/") for i in range(15))),
        )

        assert search(tool).startswith("10 web results")

    def test_a_long_title_and_snippet_are_cut(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(result("T" * 500, "https://long.test/", "S" * 2000)),
        )

        lines = search(tool).splitlines()

        assert len(lines[2]) < 210 and lines[2].endswith("…")
        assert len(lines[4]) < 510 and lines[4].endswith("…")

    def test_the_whole_list_stays_under_what_a_tool_may_return(self, tool, site):
        site.reply = (  # pyright: ignore[reportAttributeAccessIssue]
            200,
            page(
                *(
                    result(
                        "T" * 200,
                        "https://long.test/" + "p" * 300,
                        "S" * 600,
                    )
                    for i in range(20)
                )
            ),
        )

        text = search(tool, max_results=20)

        assert len(text) < MAX_OUTPUT_CHARS
        assert "more results left out]" in text

    def test_a_page_with_no_results_says_so(self, tool, site):
        site.reply = (200, page())  # pyright: ignore[reportAttributeAccessIssue]

        assert search(tool, "zzz") == 'No web results for "zzz". Try other words.'


class TestWhenItIsTurnedAway:
    @pytest.mark.parametrize("status", [202, 403, 429])
    def test_a_refusal_is_said_aloud_not_called_no_results(self, tool, site, status):
        site.reply = (status, "<html></html>")  # pyright: ignore[reportAttributeAccessIssue]

        text = search(tool)

        assert text.startswith("Error: SearchError: DuckDuckGo refused the search")
        assert f"HTTP {status}" in text and "web_fetch" in text
        assert "No web results" not in text

    def test_an_empty_answer_is_a_refusal_too(self, tool, site):
        site.reply = (200, "  \n")  # pyright: ignore[reportAttributeAccessIssue]

        assert "refused the search or returned nothing (HTTP 200)" in search(tool)

    def test_another_error_status_is_a_failure(self, tool, site):
        site.reply = (500, "boom")  # pyright: ignore[reportAttributeAccessIssue]

        assert search(tool) == "Error: SearchError: The search failed: HTTP 500"

    def test_a_server_that_takes_too_long_is_a_timeout(self, site):
        site.delay = 1.5  # pyright: ignore[reportAttributeAccessIssue]
        slow = make_search_tool(
            base_url=f"http://127.0.0.1:{site.server_address[1]}/html",
            timeout_seconds=0.3,
        )

        assert "The search timed out after 0.3 s" in search(slow)

    def test_nobody_listening_is_a_failure(self):
        nowhere = make_search_tool(
            base_url="http://127.0.0.1:9/html", timeout_seconds=2
        )

        assert search(nowhere).startswith("Error: SearchError: The search failed")


class TestTheArguments:
    @pytest.mark.parametrize(
        "arguments",
        [
            {"query": ""},
            {"query": "x" * 401},
            {"query": "ok", "max_results": 0},
            {"query": "ok", "max_results": 21},
            {},
        ],
    )
    def test_bad_arguments_are_answered_without_a_request(self, tool, site, arguments):
        assert tool.call(json.dumps(arguments)).startswith("Invalid arguments")
        assert site.seen == []


class TestWhatTheModelIsTold:
    def test_it_says_results_are_untrusted_and_where_to_read_a_page(self):
        text = " ".join(WEB_SEARCH.description.split())

        assert "untrusted" in text and "never as instructions" in text
        assert "web_fetch" in text and "cite their URLs" in text

    def test_it_says_it_may_be_turned_away(self):
        assert "turn automated searches away" in " ".join(
            WEB_SEARCH.description.split()
        )


def test_the_html_library_is_not_loaded_until_a_search_is_made():
    code = (
        "import sys, ergane.builtin.search; "
        "print([m for m in ('bs4', 'lxml', 'markdownify') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
