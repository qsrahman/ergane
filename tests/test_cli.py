import json
import sys
import threading
from pathlib import Path

import pytest

from muagent.cli import main, parse_args


def test_main_wires_config_agent_and_repl(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

    main(["--model", "wired-model"])

    out = capsys.readouterr().out
    assert "model: wired-model" in out
    assert out.endswith("Bye.\n")


def test_main_exits_on_invalid_config(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_a_missing_key_says_where_to_put_it(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main([])

    err = capsys.readouterr().err
    assert "  - api_key: Field required" in err
    assert f'No API key. Put "api_key" in {tmp_path / "muagent" / "config.json"}' in err
    assert "or set the OPENAI_API_KEY environment variable" in err


def test_the_help_for_a_missing_key_names_the_file_in_use(tmp_path, capsys):
    other = tmp_path / "other.json"
    other.write_text("{}")

    with pytest.raises(SystemExit):
        main(["--config", str(other)])

    assert f'Put "api_key" in {other}' in capsys.readouterr().err


def test_a_config_file_that_is_not_text_is_reported_not_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe{")

    with pytest.raises(SystemExit) as exc:
        main(["--config", str(bad)])

    assert exc.value.code == 1
    assert "bad.json is not UTF-8" in capsys.readouterr().err


def test_a_directory_as_the_config_file_is_reported(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(tmp_path)])

    assert exc.value.code == 1
    assert "is a directory, not a file" in capsys.readouterr().err


def test_another_mistake_gets_no_such_help(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(SystemExit):
        main(["--max-turns", "0"])

    assert "No API key" not in capsys.readouterr().err


def test_main_names_a_bad_setting_by_its_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(SystemExit):
        main(["--max-turns", "0"])

    assert "  - max_turns:" in capsys.readouterr().err


class RecordingRepl:
    """Stands in for Repl so a test can see which Agent main([]) built."""

    agent = None

    def __init__(
        self, agent, commands, session=None, console=None, servers=(), hub=None
    ):
        RecordingRepl.agent = agent

    def run(self):
        pass


def test_without_an_mcp_json_the_agent_has_no_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

    main([])

    assert RecordingRepl.agent is not None
    assert RecordingRepl.agent.tools == {}


FAKE_SERVER = {
    "command": sys.executable,
    "args": [str(Path(__file__).parent / "fake_mcp_server.py")],
}


def write_config_json(config_home: Path, servers: dict) -> None:
    path = config_home / "muagent" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}))


def threads() -> list[str]:
    return [thread.name for thread in threading.enumerate()]


class TestMcpServers:
    @pytest.fixture(autouse=True)
    def api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")

    def test_the_agent_gets_the_tools_of_the_servers_in_mcp_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

        main([])

        assert RecordingRepl.agent is not None
        names = list(RecordingRepl.agent.tools)
        assert len(names) == 18  # the fake server's
        assert "fake__add" in names
        assert (tmp_path / "muagent" / "fake.log").exists()

    def test_a_server_at_a_url_in_mcp_json_gives_the_agent_its_tools(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from helpers import serve_http

        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)
        with serve_http() as url:
            write_config_json(tmp_path, {"web": {"url": url}})

            main([])

        assert RecordingRepl.agent is not None
        assert "web__add" in RecordingRepl.agent.tools

    def test_a_file_with_type_enable_and_placeholders_works(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("FAKE_KEY", "the-key")

        class CallingRepl(RecordingRepl):
            key = None

            def run(self):  # the servers are only running while the REPL does
                agent = RecordingRepl.agent
                assert agent is not None
                CallingRepl.key = agent.tools["on__getenv"].call('{"name": "KEY"}')

        monkeypatch.setattr("muagent.cli.Repl", CallingRepl)
        write_config_json(
            tmp_path,
            {
                "on": {
                    **FAKE_SERVER,
                    "type": "local",
                    "enable": True,
                    "env": {"KEY": "${FAKE_KEY}"},
                },
                "off": {**FAKE_SERVER, "type": "local", "enable": False},
                "unset": {**FAKE_SERVER, "env": {"KEY": "${NOT_SET_ANYWHERE}"}},
            },
        )

        main([])

        assert RecordingRepl.agent is not None
        tools = RecordingRepl.agent.tools
        assert "on__add" in tools
        assert not any(name.startswith(("off__", "unset__")) for name in tools)
        assert CallingRepl.key == "the-key"

    def test_the_servers_are_stopped_when_muagent_ends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

        main([])

        assert "mcp-fake" not in threads()

    def test_the_servers_are_stopped_when_the_repl_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        class FailingRepl(RecordingRepl):
            def run(self):
                assert "mcp-fake" in threads()
                raise RuntimeError("repl failed")

        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("muagent.cli.Repl", FailingRepl)

        with pytest.raises(RuntimeError):
            main([])

        assert "mcp-fake" not in threads()

    def test_a_server_that_starts_is_not_announced_but_counted_in_the_banner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

        main(["--model", "banner-model"])

        first_line = capsys.readouterr().out.splitlines()[0]
        # The server's 18.
        assert first_line == (
            "muagent | model: banner-model | 18 tools | /help lists commands"
        )

    def test_a_server_that_fails_does_not_stop_muagent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_config_json(tmp_path, {"broken": {"command": "/no/such/server"}})
        monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

        main([])

        out = capsys.readouterr().out
        assert out.startswith("[MCP server 'broken' failed to start: FileNotFoundError")
        # A blank line sets the failure apart from the banner under it.
        _failure, blank, banner = out.splitlines()[:3]
        assert blank == "" and banner.startswith("muagent | model: ")
        assert out.endswith("Bye.\n")

    def test_servers_tells_again_what_became_of_a_server_that_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_config_json(
            tmp_path,
            {
                "broken": {"command": "/no/such/server"},
                "good": {**FAKE_SERVER, "tools": ["add"]},
                "off": {**FAKE_SERVER, "enable": False},
            },
        )
        inputs = iter(["/servers", "/exit"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        main([])

        out = capsys.readouterr().out
        assert out.count("failed to start: FileNotFoundError") == 2  # shown, and again
        assert '  good    running, 1 tool (17 left out by "tools"), ~' in out
        assert "  off     off: enable is false" in out
        # A program that failed may have said why on stderr, so its log is named.
        assert "good.log" in out and "broken.log" in out

    def test_servers_with_no_mcp_says_they_were_not_started(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        inputs = iter(["/servers", "/exit"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        main(["--no-mcp"])

        assert "  fake  off: --no-mcp" in capsys.readouterr().out

    def test_a_mistake_in_mcp_json_is_reported_and_muagent_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        write_config_json(tmp_path, {"fake": {"command": "x", "arg": []}})

        with pytest.raises(SystemExit) as exc:
            main([])

        err = capsys.readouterr().err
        assert exc.value.code == 1
        assert err.startswith("Configuration error:")
        assert "  - mcpServers.fake.arg:" in err

    def test_ctrl_c_while_the_servers_start_exits_quietly(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("muagent.cli.McpHub.start", interrupted)

        with pytest.raises(SystemExit) as exc:
            main([])

        assert exc.value.code == 130


def test_version_prints_the_version_of_the_package_and_exits(capsys):
    from importlib.metadata import version

    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out == f"muagent {version('muagent')}\n"


class TestFlags:
    @pytest.fixture(autouse=True)
    def api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

    def test_a_flag_that_is_not_given_is_not_in_the_result(self):
        assert vars(parse_args([])) == {"no_mcp": False}

    def test_every_flag(self):
        args = parse_args(
            [
                "--config", "c.json", "--model", "m", "--base-url", "u",
                "--no-streaming", "--max-context-tokens", "9", "--max-turns", "2",
                "--reasoning-effort", "low", "--no-mcp",
            ]
        )  # fmt: skip

        assert vars(args) == {
            "config": Path("c.json"),
            "model": "m",
            "base_url": "u",
            "streaming": False,
            "max_context_tokens": 9,
            "max_turns": 2,
            "reasoning_effort": "low",
            "no_mcp": True,
        }

    def test_there_is_no_flag_for_the_key(self, capsys):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--api-key", "sk-in-the-process-list"])

        assert exc.value.code == 2

    def test_a_flag_wins_over_the_file(self, tmp_path: Path):
        (tmp_path / "muagent").mkdir()
        (tmp_path / "muagent" / "config.json").write_text(
            json.dumps({"model": "file-model", "max_turns": 5})
        )

        main(["--model", "flag-model"])

        agent = RecordingRepl.agent
        assert agent is not None
        assert agent.config.model == "flag-model"  # flag > file
        assert agent.config.max_turns == 5  # the file, with no flag

    def test_default_as_the_effort_sends_none_over_the_file(self, tmp_path: Path):
        (tmp_path / "muagent").mkdir()
        (tmp_path / "muagent" / "config.json").write_text(
            json.dumps({"reasoning_effort": "high"})
        )

        main(["--reasoning-effort", "default"])

        assert RecordingRepl.agent is not None
        assert RecordingRepl.agent.config.reasoning_effort is None

    def test_a_bad_value_is_reported_by_its_key(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--max-turns", "0"])

        assert exc.value.code == 1
        assert "  - max_turns:" in capsys.readouterr().err

    def test_the_config_flag_names_the_file(self, tmp_path: Path):
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"model": "other-model"}))

        main(["--config", str(other)])

        assert RecordingRepl.agent is not None
        assert RecordingRepl.agent.config.model == "other-model"

    def test_a_config_file_that_does_not_exist_is_reported(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--config", str(tmp_path / "nowhere.json")])

        assert exc.value.code == 1
        assert "Config file not found" in capsys.readouterr().err

    def test_no_mcp_starts_none_of_the_servers(self, tmp_path: Path):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})

        main(["--no-mcp"])

        assert RecordingRepl.agent is not None
        assert RecordingRepl.agent.tools == {}
        assert "mcp-fake" not in threads()


class TestMcpServersInTheBackground:
    """At a terminal the prompt comes up before the servers are ready."""

    @pytest.fixture(autouse=True)
    def terminal(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setattr("muagent.cli.terminal_session", lambda *a, **k: object())
        write_config_json(tmp_path, {"fake": FAKE_SERVER})

    def test_the_agent_starts_without_tools_and_the_hub_has_them_coming(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        seen = {}

        class Recording(RecordingRepl):
            def __init__(self, agent, commands, session=None, console=None, *rest):
                super().__init__(agent, commands, session, console)
                self.hub = rest[1]
                seen["hub"] = self.hub
                seen["tools_at_start"] = dict(agent.tools)

        monkeypatch.setattr("muagent.cli.Repl", Recording)

        main([])

        assert seen["tools_at_start"] == {}
        assert seen["hub"] is not None

    def test_the_servers_are_stopped_when_muagent_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

        main([])

        assert not any(name.startswith("mcp-") for name in threads())

    def test_without_a_terminal_the_start_waits_for_the_servers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("muagent.cli.terminal_session", lambda *a, **k: None)
        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

        main([])

        assert RecordingRepl.agent is not None
        assert "fake__add" in RecordingRepl.agent.tools
