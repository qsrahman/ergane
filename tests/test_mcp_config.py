"""McpServerConfig and the file it is read from: what is accepted, and what is refused."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from muagent.config import load_config
from muagent.mcp_config import McpServerConfig


def load_servers(path: Path) -> dict[str, McpServerConfig]:
    """The servers of a config file, as main() gets them."""
    return load_config(path, api_key="test").mcp_servers


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

        assert servers["plain"].args == [] and servers["plain"].env == {}
        assert servers["plain"].read_only is False  # trust must be asked for
        assert servers["plain"].startup_timeout == 30
        assert servers["plain"].call_timeout == 300
        assert servers["plain"].instructions is False
        assert servers["full"].args == ["a", "b"]
        assert servers["full"].env == {"K": "v"}
        assert servers["full"].read_only is True

    def test_an_empty_object_has_no_servers(self, tmp_path):
        assert load_servers(self.write(tmp_path, "{}")) == {}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("not json", "Invalid JSON"),
            ('{"mcpServers": {"a": {}}}', "give either command"),
            ('{"mcpServers": {"a": {"command": ""}}}', "mcpServers.a.command"),
            ('{"mcpServers": {"a": {"command": "x", "arg": []}}}', "mcpServers.a.arg"),
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

        assert (config.url, config.command) == (url, None)
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
            ({"url": "https://example.com", "command": "x"}, "either command"),
            ({"url": "https://example.com", "args": ["x"]}, "go with command"),
            ({"url": "https://example.com", "env": {"A": "b"}}, "go with command"),
            ({"command": "x", "headers": {"A": "b"}}, "headers go with url"),
            ({}, "either command"),
        ],
    )
    def test_a_server_is_one_thing_and_its_url_must_be_safe(
        self, tmp_path, entry, message
    ):
        with pytest.raises(ValidationError, match=message):
            self.load(tmp_path, entry)


class TestMcpJsonKeys:
    """`type` and `enable`, as other tools' MCP files have them."""

    def load(self, tmp_path: Path, entry: dict):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"mcpServers": {"s": entry}}))
        return load_servers(path)["s"]

    def test_a_server_is_on_by_default_and_can_be_switched_off(self, tmp_path):
        assert self.load(tmp_path, {"command": "x"}).enable is True
        assert self.load(tmp_path, {"command": "x", "enable": False}).enable is False

    @pytest.mark.parametrize(
        "entry",
        [
            {"command": "x", "cwd": "/srv"},
            {"url": "https://example.com", "cwd": "/srv"},
        ],
        ids=["program", "url"],
    )
    def test_there_is_no_working_directory_setting(self, tmp_path, entry):
        # A server starts where muagent does; a file that still has the key is
        # refused, and the message names it.
        with pytest.raises(ValidationError, match=r"mcpServers\.s\.cwd"):
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
            (
                {"type": "remote", "command": "x"},
                "type is 'remote', but a server with a command",
            ),
            (
                {"type": "local", "url": "https://example.com"},
                "type is 'local', but a server with a url",
            ),
            ({"type": "stdio", "command": "x"}, "local"),
            ({"command": "x", "enable": "maybe"}, "enable"),
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
                            "enable": True,
                        },
                        "files": {
                            "type": "local",
                            "command": "npx",
                            "args": ["-y", "files-server@1.2.3"],
                            "env": {"TOKEN": "${FILES_TOKEN}"},
                            "enable": False,
                        },
                    }
                }
            )
        )

        servers = load_servers(path)

        assert list(servers) == ["docs", "files"]
        assert servers["files"].env == {"TOKEN": "${FILES_TOKEN}"}  # filled in at start
        assert servers["files"].enable is False


class TestToolsKey:
    def test_an_empty_list_is_refused(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"mcpServers": {"s": {"command": "x", "tools": []}}}')

        with pytest.raises(ValidationError, match=r"mcpServers\.s\.tools"):
            load_servers(path)
