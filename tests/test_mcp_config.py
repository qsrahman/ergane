"""McpServerConfig and the file it is read from: what is accepted, and what is refused."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ergane.config import (
    LocalMcpServer,
    McpServerConfig,
    RemoteMcpServer,
    load_config,
)


def load_servers(path: Path) -> dict[str, McpServerConfig]:
    """The servers of a config file, as main() gets them."""
    return load_config(path, api_key="test").mcp_servers


def local(servers: dict[str, McpServerConfig], name: str) -> LocalMcpServer:
    server = servers[name]
    assert isinstance(server, LocalMcpServer)
    return server


class TestLoadServers:
    def write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "config.json"
        path.write_text(text)
        return path

    def test_a_server_has_a_command_and_optional_arguments_environment_and_trust(
        self, tmp_path
    ):
        path = self.write(
            tmp_path,
            """{"mcpServers": {
                "plain": {"command": "tool"},
                "full": {"command": "uvx", "args": ["a", "b"], "env": {"K": "v"},
                         "read_only": true}
            }}""",
        )

        servers = load_servers(path)

        plain, full = local(servers, "plain"), local(servers, "full")
        assert plain.args == [] and plain.env == {}
        assert plain.read_only is False  # trust must be asked for
        assert plain.startup_timeout == 30
        assert plain.call_timeout == 300
        assert plain.instructions is False
        assert plain.images is False
        assert full.args == ["a", "b"]
        assert full.env == {"K": "v"}
        assert full.read_only is True

    def test_an_empty_object_has_no_servers(self, tmp_path):
        assert load_servers(self.write(tmp_path, "{}")) == {}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("not json", "Invalid JSON"),
            (
                '{"mcpServers": {"a": {}}}',
                "mcpServers.a.local.command\n  Field required",
            ),
            ('{"mcpServers": {"a": {"command": ""}}}', "mcpServers.a.local.command"),
            (
                '{"mcpServers": {"a": {"command": "x", "arg": []}}}',
                "mcpServers.a.local.arg",
            ),
            ('{"mcpServers": {"a": {"url": "http://x"}}}', "url must be https"),
            (
                '{"mcpServers": {"a.b": {"command": "x"}}}',
                "String should match pattern",
            ),
            ('{"servers": {}}', "servers"),
            (
                '{"mcpServers": {"a": {"command": "x", "call_timeout": 0}}}',
                "call_timeout",
            ),
            (
                '{"mcpServers": {"a": {"command": "x", "startup_timeout": -1}}}',
                "startup_timeout",
            ),
            (
                '{"mcpServers": {"a": {"command": "x", "call_timeout": "soon"}}}',
                "call_timeout",
            ),
        ],
        ids=[
            "not-json",
            "no-command",
            "empty-command",
            "typo",
            "http",
            "name",
            "key",
            "zero-timeout",
            "negative-timeout",
            "text-timeout",
        ],
    )
    def test_a_mistake_names_where_it_is(self, tmp_path, text, message):
        with pytest.raises(ValidationError, match=message.replace(".", r"\.")):
            load_servers(self.write(tmp_path, text))


class TestUrls:
    def load(self, tmp_path: Path, entry: dict):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"mcpServers": {"web": entry}}))
        return load_servers(path)["web"]

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/mcp",
            "http://localhost:8000/mcp",
            "http://127.0.0.1:9/mcp",
            "http://[::1]:9/mcp",
        ],
    )
    def test_a_url_may_be_https_or_http_to_this_machine(self, tmp_path, url):
        config = self.load(tmp_path, {"url": url, "headers": {"A": "b"}})

        assert isinstance(config, RemoteMcpServer)
        assert config.url == url
        assert config.headers == {"A": "b"}

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"url": "http://example.com/mcp"}, "url must be https"),
            ({"url": "http://localhost.example.com/mcp"}, "url must be https"),
            ({"url": "ftp://example.com/mcp"}, "url must be http"),
            ({"url": "example.com/mcp"}, "url must be http"),
            ({"url": "https://"}, "url must be http"),
            ({"url": "https://user:pass@example.com/mcp"}, "user name or password"),
            (
                {"url": "https://example.com", "command": "x"},
                r"remote\.command\n  Extra",
            ),
            ({"url": "https://example.com", "args": ["x"]}, r"remote\.args\n  Extra"),
            (
                {"url": "https://example.com", "env": {"A": "b"}},
                r"remote\.env\n  Extra",
            ),
            ({"command": "x", "headers": {"A": "b"}}, r"local\.headers\n  Extra"),
            ({}, r"local\.command\n  Field required"),
        ],
    )
    def test_a_server_is_one_thing_and_its_url_must_be_safe(
        self, tmp_path, entry, message
    ):
        with pytest.raises(ValidationError, match=message):
            self.load(tmp_path, entry)


class TestMcpJsonKeys:
    """`type` and `enabled`, as other tools' MCP files have them."""

    def load(self, tmp_path: Path, entry: dict):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"mcpServers": {"s": entry}}))
        return load_servers(path)["s"]

    def test_a_server_is_on_by_default_and_can_be_switched_off(self, tmp_path):
        assert self.load(tmp_path, {"command": "x"}).enabled is True
        assert self.load(tmp_path, {"command": "x", "enabled": False}).enabled is False

    @pytest.mark.parametrize(
        "entry",
        [
            {"command": "x", "cwd": "/srv"},
            {"url": "https://example.com", "cwd": "/srv"},
        ],
        ids=["program", "url"],
    )
    def test_there_is_no_working_directory_setting(self, tmp_path, entry):
        # A server starts where ergane does; a file that still has the key is
        # refused, and the message names it.
        with pytest.raises(
            ValidationError, match=r"mcpServers\.s\.(local|remote)\.cwd"
        ):
            self.load(tmp_path, entry)

    @pytest.mark.parametrize(
        "entry",
        [
            {"type": "local", "command": "x"},
            {"type": "remote", "url": "https://example.com/mcp"},
        ],
    )
    def test_a_type_that_agrees_is_accepted(self, tmp_path, entry):
        assert self.load(tmp_path, entry).type == entry["type"]

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"type": "remote", "command": "x"}, r"remote\.url\n  Field required"),
            (
                {"type": "local", "url": "https://example.com"},
                r"local\.command\n  Field required",
            ),
            ({"type": "stdio", "command": "x"}, "local"),
            ({"command": "x", "enabled": "maybe"}, "enabled"),
        ],
    )
    def test_a_mistake_is_reported(self, tmp_path, entry, message):
        with pytest.raises(ValidationError, match=message):
            self.load(tmp_path, entry)

    def test_a_file_in_this_dialect_loads(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "docs": {
                            "type": "remote",
                            "url": "https://docs.example.com/mcp",
                            "headers": {"API_KEY": "${DOCS_KEY}"},
                            "enabled": True,
                        },
                        "files": {
                            "type": "local",
                            "command": "npx",
                            "args": ["-y", "files-server@1.2.3"],
                            "env": {"TOKEN": "${FILES_TOKEN}"},
                            "enabled": False,
                        },
                    }
                }
            )
        )

        servers = load_servers(path)

        assert list(servers) == ["docs", "files"]
        files = local(servers, "files")
        assert files.env == {"TOKEN": "${FILES_TOKEN}"}  # filled in at start
        assert files.enabled is False


class TestToolsKey:
    def test_an_empty_list_is_refused(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"mcpServers": {"s": {"command": "x", "tools": []}}}')

        with pytest.raises(ValidationError, match=r"mcpServers\.s\.local\.tools"):
            load_servers(path)


class TestEntriesThatAreNotServers:
    @pytest.mark.parametrize(
        "entry",
        [
            '"x"',
            "null",
            "[]",
            "5",
            '{"type": 5, "command": "x"}',
            '{"type": ["local"]}',
        ],
    )
    def test_they_are_reported_not_a_traceback(self, tmp_path, entry):
        path = tmp_path / "config.json"
        path.write_text(f'{{"mcpServers": {{"a": {entry}}}}}')

        with pytest.raises(ValidationError, match="Unable to extract tag"):
            load_servers(path)
