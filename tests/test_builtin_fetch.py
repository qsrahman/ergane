"""web_fetch, against a small HTTP server on this machine (so the tool is made with
`allow_private_ips=True`, except where the refusal itself is tested)."""

import gzip
import json
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpcore2
import pytest

from ergane.builtin import fetch
from ergane.builtin.fetch import WEB_FETCH, is_public_address, make_fetch_tool
from ergane.tools import MAX_OUTPUT_CHARS

ARTICLE = """<!DOCTYPE html><html><head><title>The page</title></head><body>
<nav><a href="/">Home</a> <a href="/about">About</a></nav>
<article><h1>A heading</h1>
<p>The first paragraph says something worth reading about the subject at hand,
long enough that the extractor takes it for the content of the page.</p>
<p>The second paragraph adds more words to the article, so that it is clearly the
main part of this page and not furniture around it.</p></article>
<footer>Copyright nobody</footer></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # a body ends when the connection does

    def log_message(self, format, *args):
        pass

    def send(
        self, status, body=b"", content_type: str | None = "text/plain", **headers
    ):
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.seen.append((self.path, dict(self.headers)))  # pyright: ignore[reportAttributeAccessIssue]
        path = self.path.split("?")[0]
        if path == "/page":
            return self.send(200, ARTICLE.encode(), "text/html; charset=utf-8")
        if path == "/plain":
            return self.send(200, b"hello plain")
        if path == "/long":
            return self.send(200, b"0123456789" * 10)
        if path == "/json":
            return self.send(200, b'{"a": 1}', "application/json")
        if path == "/notype":
            return self.send(200, b'{"a": 1}', None)
        if path == "/md":
            if "text/markdown" in self.headers.get("Accept", ""):
                return self.send(200, b"# Already markdown\n\ntext", "text/markdown")
            return self.send(200, ARTICLE.encode(), "text/html")
        if path == "/links":
            body = (
                "<html><head><title>L</title></head><body><nav><a href='/menu'>Menu"
                "</a></nav><article><h1>Links</h1><p>Read <a href='/about'>about us"
                "</a>, <a href='sub/page.html'>the sub page</a> and look at "
                "<img src='pic.png' alt='a picture'> and then some more words so that "
                "the paragraph is clearly the content of this page.</p></article>"
                "<footer>Footer words</footer></body></html>"
            )
            return self.send(200, body.encode(), "text/html")
        if path == "/untitled":
            body = (
                "<html><head><title>Example Domain</title></head><body><p>This domain "
                "is for use in documentation examples.</p></body></html>"
            )
            return self.send(200, body.encode(), "text/html")
        if path == "/titled":
            body = (
                "<html><head><title>Same - Site</title></head><body><h1>Same</h1>"
                "<p>Words that follow the heading of the page.</p></body></html>"
            )
            return self.send(200, body.encode(), "text/html")
        if path == "/bare":  # no <main> or <article>: the furniture is in the body
            body = (
                "<html><head><title>Bare</title><style>p {color: red}</style></head>"
                "<body><nav>Site menu words</nav><div><h1>Bare</h1><p>The words that "
                "are the page.</p><p hidden>Hidden words</p><p style='display: none'>"
                "Unseen words</p><script>var tracking = 1;</script></div>"
                "<aside>Related words</aside><footer>Footer words</footer></body></html>"
            )
            return self.send(200, body.encode(), "text/html")
        if path == "/pdf":
            return self.send(200, b"%PDF-1.4 binary", "application/pdf")
        if path == "/nul":
            return self.send(200, b"text\0text", "text/plain")
        if path == "/deep":
            body = "<html><body>" + "<div>" * 3000 + "deep" + "</div>" * 3000
            return self.send(200, body.encode(), "text/html")
        if path == "/untidy":  # unclosed <p> and <li>: they are not nesting
            body = (
                "<html><body><article>" + "<p>word " * 600 + "<ul>" + "<li>item " * 600
            )
            return self.send(
                200, (body + "</ul></article></body></html>").encode(), "text/html"
            )
        if path == "/private":
            return self.send(200, b"secret", "text/plain")
        if path == "/redirect":
            return self.send(302, b"", Location="/page")
        if path == "/redirect-to-private":
            return self.send(302, b"", Location="/private")
        if path == "/loop":
            return self.send(302, b"", Location="/loop")
        if path == "/missing":
            return self.send(404, b"nope")
        if path == "/latin1":
            body = (
                b'<html><head><meta charset="iso-8859-1"></head><body><article><p>'
                + "café ".encode("latin-1") * 40
                + b"</p></article></body></html>"
            )
            return self.send(200, body, "text/html")
        if path == "/big":
            return self.send(200, b"x" * 200_000)
        if path == "/big-pdf":  # over the cap: refused for its type, not its size
            return self.send(200, b"x" * 200_000, "application/pdf")
        if path == "/utf7":
            body = (
                b'<html><head><meta charset="utf-7"></head><body><article><p>'
                b"+AGE-bc</p></article></body></html>"
            )
            return self.send(200, body, "text/html")
        if path == "/big-declared":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "999999999")
            self.end_headers()
            return
        if path == "/bomb":
            body = gzip.compress(b"\0" * 5_000_000)
            return self.send(200, body, "text/plain", Content_Encoding="gzip")
        if path == "/brotli":
            return self.send(200, b"x", "text/plain", Content_Encoding="br")
        if path == "/slow":
            time.sleep(1.5)
            return self.send(200, b"late")
        self.send(404, b"unknown")


@pytest.fixture
def site() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.seen = []  # pyright: ignore[reportAttributeAccessIssue]
    # A short poll interval: `shutdown` waits for one, which is 0.5 s by default.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def url_of(site: ThreadingHTTPServer, path: str) -> str:
    return f"http://127.0.0.1:{site.server_address[1]}{path}"


def call(tool, site, path, **arguments) -> str:
    return tool.call(json.dumps({"url": url_of(site, path), **arguments}))


@pytest.fixture
def tool():
    return make_fetch_tool(allow_private_ips=True, timeout_seconds=5)


class TestThePage:
    def test_a_page_comes_back_as_markdown(self, tool, site):
        result = call(tool, site, "/page")

        assert result.startswith(f"Contents of {url_of(site, '/page')}:\n")
        assert "# A heading" in result
        assert "The second paragraph adds more words" in result
        assert "<article>" not in result  # it is markdown, not the page

    def test_the_page_title_is_not_printed_as_a_literal_tag(self, tool, site):
        result = call(tool, site, "/page")

        assert "<title>" not in result and "<p>" not in result
        # The title is the page's own, as a heading, since the article does not name it.
        assert result.split(":\n", 1)[1].startswith("# The page\n\n# A heading")

    def test_raw_gives_the_html_as_it_is(self, tool, site):
        result = call(tool, site, "/page", raw=True)

        assert "<nav>" in result and "<article>" in result

    def test_a_redirect_is_followed(self, tool, site):
        assert "# A heading" in call(tool, site, "/redirect")

    def test_a_site_that_can_answer_in_markdown_is_asked_to(self, tool, site):
        result = call(tool, site, "/md")

        assert result.endswith("# Already markdown\n\ntext")
        assert "text/markdown" in site.seen[-1][1]["Accept"]  # pyright: ignore[reportAttributeAccessIssue]

    def test_raw_asks_for_whatever_the_site_serves_by_default(self, tool, site):
        call(tool, site, "/md", raw=True)

        assert site.seen[-1][1]["Accept"] == "*/*"  # pyright: ignore[reportAttributeAccessIssue]

    def test_it_says_who_is_asking(self, tool, site):
        call(tool, site, "/plain")

        agent = site.seen[-1][1]["User-Agent"]  # pyright: ignore[reportAttributeAccessIssue]
        assert agent.startswith("ergane/") and "Autonomous" in agent

    def test_text_that_is_not_html_comes_back_with_a_note(self, tool, site):
        result = call(tool, site, "/json")

        assert result == (
            "Content type application/json cannot be simplified to markdown, but "
            f"here is the raw content:\nContents of {url_of(site, '/json')}:\n"
            '{"a": 1}'
        )

    def test_a_body_with_no_type_that_is_not_html_is_not_simplified(self, tool, site):
        assert "(not declared)" in call(tool, site, "/notype")

    def test_a_charset_declared_in_the_page_is_honoured(self, tool, site):
        assert "café café" in call(tool, site, "/latin1")

    def test_a_declared_utf7_is_not_used(self, tool, site):
        assert "+AGE-bc" in call(tool, site, "/utf7")


class TestWhatItReadsAndWhatItDoesNot:
    def test_navigation_and_footers_are_left_out(self, tool, site):
        result = call(tool, site, "/page")

        assert "Copyright nobody" not in result  # the footer
        assert "About" not in result  # the navigation

    def test_a_page_with_no_heading_gets_its_title_as_one(self, tool, site):
        result = call(tool, site, "/untitled")

        assert result.split(":\n", 1)[1].startswith("# Example Domain\n\nThis domain")

    def test_a_heading_that_is_the_title_is_not_said_twice(self, tool, site):
        result = call(
            tool, site, "/titled"
        )  # <title>Same - Site</title>, <h1>Same</h1>

        assert result.split(":\n", 1)[1].startswith("# Same\n\nWords that follow")

    def test_without_an_article_the_furniture_of_the_body_is_still_left_out(
        self, tool, site
    ):
        result = call(tool, site, "/bare")

        assert "The words that are the page." in result
        for left_out in (
            "Site menu",
            "Related",
            "Footer",
            "Hidden",
            "Unseen",
            "tracking",
            "color: red",
        ):
            assert left_out not in result

    def test_links_and_images_are_made_absolute(self, tool, site):
        result = call(tool, site, "/links")
        base = f"http://127.0.0.1:{site.server_address[1]}"

        assert f"]({base}/about)" in result
        assert f"]({base}/sub/page.html)" in result
        assert f"({base}/pic.png)" in result
        assert "/menu" not in result  # that is navigation

    def test_it_says_where_a_redirect_ended_up(self, tool, site):
        result = call(tool, site, "/redirect")

        assert result.startswith(
            f"Contents of {url_of(site, '/page')} (redirected from "
            f"{url_of(site, '/redirect')}):\n"
        )

    @pytest.mark.parametrize("path", ["/pdf", "/nul"])
    def test_a_body_that_is_not_text_is_refused(self, tool, site, path):
        result = call(tool, site, path)

        assert "not text; web_fetch returns text only" in result
        assert "%PDF" not in result

    def test_a_type_that_is_not_text_is_refused_before_the_body_is_read(
        self, tool, site
    ):
        assert "is a application/pdf file, not text" in call(tool, site, "/big-pdf")

    def test_a_page_that_nests_too_deep_is_refused_at_once(self, tool, site):
        started = time.monotonic()

        result = call(tool, site, "/deep")

        assert "nests its elements too deeply" in result
        assert "raw=true" in result
        assert time.monotonic() - started < 3

    def test_raw_still_reads_a_page_that_nests_too_deep(self, tool, site):
        assert "deep" in call(tool, site, "/deep", raw=True, max_length=100)

    def test_unclosed_paragraphs_and_list_items_are_not_nesting(self, tool, site):
        result = call(tool, site, "/untidy")

        assert "nests" not in result and "item" in result

    def test_a_page_larger_than_it_simplifies_is_cut_and_says_so(
        self, tool, site, monkeypatch
    ):
        monkeypatch.setattr(fetch, "MAX_EXTRACT_HTML_CHARS", 800)

        result = call(tool, site, "/untidy")

        assert result.startswith("The page is large: only its first 800 characters")
        assert "item" not in result  # the list comes after the cut


class TestReadingInParts:
    def test_a_page_is_fetched_once_for_all_its_parts(self, tool, site):
        for start in (0, 10, 20):
            call(tool, site, "/long", max_length=10, start_index=start)

        paths = [path for path, _ in site.seen]  # pyright: ignore[reportAttributeAccessIssue]
        assert paths == ["/long"]

    def test_the_raw_page_is_another_page(self, tool, site):
        call(tool, site, "/page")
        call(tool, site, "/page", raw=True)

        assert [p for p, _ in site.seen].count("/page") == 2  # pyright: ignore[reportAttributeAccessIssue]

    def test_it_is_fetched_again_when_the_time_is_up(self, tool, site, monkeypatch):
        monkeypatch.setattr(fetch, "CACHE_SECONDS", 0)

        call(tool, site, "/long", max_length=10)
        call(tool, site, "/long", max_length=10, start_index=10)

        assert [p for p, _ in site.seen].count("/long") == 2  # pyright: ignore[reportAttributeAccessIssue]

    def test_the_oldest_pages_are_forgotten(self, tool, site):
        for n in range(fetch.CACHE_ENTRIES + 1):
            call(tool, site, f"/long?n={n}", max_length=10)
        site.seen.clear()  # pyright: ignore[reportAttributeAccessIssue]

        call(tool, site, "/long?n=0", max_length=10)  # the first one was dropped
        call(
            tool, site, f"/long?n={fetch.CACHE_ENTRIES}", max_length=10
        )  # the last was not

        assert [p for p, _ in site.seen if p.startswith("/long")] == ["/long?n=0"]  # pyright: ignore[reportAttributeAccessIssue]

    def test_a_failure_is_not_remembered(self, tool, site):
        call(tool, site, "/missing")
        call(tool, site, "/missing")

        assert [p for p, _ in site.seen].count("/missing") == 2  # pyright: ignore[reportAttributeAccessIssue]


class TestWhatTheModelIsTold:
    def test_the_description_says_what_it_cannot_do_and_not_to_obey_pages(self):
        description = WEB_FETCH.description

        assert "does not run JavaScript" in description
        assert (
            "untrusted web text" in description
            and "never as instructions" in description
        )
        assert "start_index" in description

    def test_max_length_stays_under_what_a_tool_may_return(self):
        # A longer result would be cut by `Tool`, taking the hint to read on with it.
        assert fetch.MAX_LENGTH + 4200 < MAX_OUTPUT_CHARS
        assert WEB_FETCH.check(
            json.dumps({"url": "https://x.test/", "max_length": 16_001})
        )

    def test_a_url_that_is_too_long_is_refused_before_it_is_fetched(self):
        assert WEB_FETCH.check(json.dumps({"url": "https://x.test/" + "a" * 2048}))


def test_it_does_not_load_its_libraries_until_a_page_is_read():
    # A fresh interpreter: this process has imported everything already. They cost
    # about 0.1 s, and ergane should not start slower for a tool nobody has called.
    import subprocess
    import sys

    code = (
        "import sys, ergane.builtin.fetch; "
        "print([m for m in ('markdownify', 'bs4', 'lxml') "
        "if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"


class TestPagination:
    def test_a_long_page_is_cut_and_says_where_to_go_on(self, tool, site):
        result = call(tool, site, "/long", max_length=10)

        assert (
            "\n0123456789\n\n[... showing characters 0-10 of 100; more follows."
            in result
        )
        assert result.endswith("Continue with web_fetch start_index=10]")

    def test_start_index_reads_on(self, tool, site):
        result = call(tool, site, "/long", max_length=10, start_index=10)

        assert result.endswith(
            "[... showing characters 10-20 of 100; more follows. Continue with "
            "web_fetch start_index=20]"
        )

    def test_the_last_part_has_no_hint(self, tool, site):
        result = call(tool, site, "/long", max_length=10, start_index=90)

        assert result.endswith("0123456789") and "more follows" not in result

    def test_past_the_end_there_is_no_more(self, tool, site):
        assert call(tool, site, "/long", start_index=500).endswith(
            "[start_index 500 is past the end of the content, which has 100 characters]"
        )

    def test_the_arguments_are_checked_before_anything_is_fetched(self, tool, site):
        result = call(tool, site, "/long", max_length=0)

        assert result.startswith("Invalid arguments")
        assert site.seen == []  # pyright: ignore[reportAttributeAccessIssue]


class TestWhatItRefuses:
    def test_a_url_that_is_not_http(self, tool):
        assert (
            tool.call(json.dumps({"url": "file:///etc/passwd"}))
            == "Error: FetchError: 'file:///etc/passwd' is not an http:// or https:// URL"
        )

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]", "2130706433"])
    def test_an_address_on_this_machine(self, site, host):
        port = site.server_address[1]
        result = WEB_FETCH.call(json.dumps({"url": f"http://{host}:{port}/page"}))

        assert "Refused to fetch" in result and "not a public address" in result
        assert site.seen == []  # pyright: ignore[reportAttributeAccessIssue]

    @pytest.mark.parametrize(
        ("address", "public"),
        [
            ("93.184.216.34", True),
            ("8.8.8.8", True),
            ("2606:2800:220:1:248:1893:25c8:1946", True),
            ("127.0.0.1", False),
            ("10.1.2.3", False),
            ("172.16.0.1", False),
            ("192.168.1.1", False),
            ("169.254.169.254", False),  # the cloud metadata address
            ("100.100.100.200", False),  # Alibaba Cloud's
            ("0.0.0.0", False),
            ("::1", False),
            ("fd00:ec2::254", False),  # AWS's
            ("fe80::1", False),
            ("::ffff:127.0.0.1", False),  # IPv4-mapped
            ("64:ff9b::7f00:1", False),  # NAT64 around 127.0.0.1
            (
                "64:ff9b::808:808",
                True,
            ),  # NAT64 around 8.8.8.8: a DNS64 network's answer
            ("4000::1", False),  # outside global unicast, which is_global allows
            ("224.0.0.1", False),
        ],
    )
    def test_which_addresses_are_public(self, address, public):
        assert is_public_address(address) is public

    def test_the_address_that_is_checked_is_the_address_connected_to(self, monkeypatch):
        # The name resolves to a public address; the connection must go to that
        # address as an address, not look the name up again.
        lookups: list[str] = []

        def getaddrinfo(host, port, **kwargs):
            lookups.append(host)
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
            ]

        connected: list[str] = []

        def connect_tcp(self, host, port, *args, **kwargs):
            connected.append(host)
            raise httpcore2.ConnectError("stop here")

        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
        monkeypatch.setattr(httpcore2.SyncBackend, "connect_tcp", connect_tcp)

        WEB_FETCH.call(json.dumps({"url": "https://rebind.test/page"}))

        assert connected and set(connected) == {"93.184.216.34"}
        assert set(lookups) == {"rebind.test"}  # looked up by the connection, once each

    def test_a_name_that_resolves_to_this_machine_is_refused_whatever_it_looked_like(
        self, site, monkeypatch
    ):
        def getaddrinfo(host, port, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
        port = site.server_address[1]

        result = WEB_FETCH.call(
            json.dumps({"url": f"http://innocent.test:{port}/page"})
        )

        assert "resolves to 127.0.0.1, which is not a public address" in result
        assert site.seen == []  # pyright: ignore[reportAttributeAccessIssue]

    def test_one_private_address_among_public_ones_refuses_the_host(self, monkeypatch):
        def getaddrinfo(host, port, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

        result = WEB_FETCH.call(json.dumps({"url": "https://mixed.test/"}))

        assert "resolves to 10.0.0.5" in result

    def test_a_name_that_does_not_resolve_is_a_failure_to_fetch(self):
        result = WEB_FETCH.call(json.dumps({"url": "https://no-such-host.invalid/"}))

        assert result.startswith("Error: FetchError:")

    def test_a_status_error(self, tool, site):
        assert call(tool, site, "/missing").endswith("status code 404")

    def test_too_many_redirects(self, tool, site):
        assert "Failed to fetch" in call(tool, site, "/loop")

    def test_a_page_that_takes_too_long(self, site):
        result = call(
            make_fetch_tool(allow_private_ips=True, timeout_seconds=0.4), site, "/slow"
        )

        assert "timed out" in result


class TestProxies:
    def test_the_proxy_variables_are_not_read(self, tool, site, monkeypatch):
        # A proxy resolves the name on its side, where it cannot be checked. These
        # point at a port nobody listens on, so a fetch that used one would fail.
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, "http://127.0.0.1:9")

        assert "hello plain" in call(tool, site, "/plain")


class TestRobotsTxt:
    """It is not read: the fetches are the user's, asked about one at a time."""

    def test_a_page_is_fetched_without_asking_for_robots_txt(self, tool, site):
        call(tool, site, "/plain")

        assert [path for path, _ in site.seen] == ["/plain"]  # pyright: ignore[reportAttributeAccessIssue]

    def test_nor_are_the_hops_of_a_redirect(self, tool, site):
        call(tool, site, "/redirect")

        assert [path for path, _ in site.seen] == ["/redirect", "/page"]  # pyright: ignore[reportAttributeAccessIssue]


class TestTheCap:
    @pytest.fixture(autouse=True)
    def small_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fetch, "MAX_RESPONSE_BYTES", 100_000)

    def test_a_body_that_declares_its_size_is_refused_unread(self, tool, site):
        result = call(tool, site, "/big-declared")

        assert (
            "larger than the 100000 byte limit (it declares 999999999 bytes)" in result
        )

    def test_a_body_that_does_not_is_cut_off_while_it_is_read(self, tool, site):
        assert "larger than the 100000 byte limit" in call(tool, site, "/big")

    def test_a_small_body_that_expands_enormously_is_stopped(self, tool, site):
        result = call(tool, site, "/bomb")  # 5 MB of zeros in a few KB

        assert "larger than the 100000 byte limit" in result

    def test_a_coding_it_cannot_bound_is_refused(self, tool, site):
        assert "Unsupported content encoding br" in call(tool, site, "/brotli")


class TestReadCapped:
    """The streamed read, on its own: how much of an endless body is read."""

    class Streamed:
        def __init__(self, chunks: int) -> None:
            self.headers: dict[str, str] = {}
            self.read = 0
            self.chunks = chunks

        def iter_raw(self) -> Iterator[bytes]:
            for _ in range(self.chunks):
                self.read += 1
                yield b"x" * 100

    def test_the_read_stops_at_the_cap_and_does_not_go_on_for_ever(self):
        response = self.Streamed(chunks=1_000_000)

        body, cut = fetch._read_capped(response, 250)  # pyright: ignore[reportArgumentType, reportPrivateUsage]

        assert (len(body), cut) == (250, True)
        assert response.read <= 4  # not a million chunks

    def test_a_body_under_the_cap_is_whole(self):
        body, cut = fetch._read_capped(self.Streamed(chunks=2), 250)  # pyright: ignore[reportArgumentType, reportPrivateUsage]

        assert (len(body), cut) == (200, False)


def test_it_asks_before_it_goes_out():
    # A page can tell the model to fetch evil.example/?q=<the user's data>.
    assert WEB_FETCH.read_only is False and WEB_FETCH.name == "web_fetch"
