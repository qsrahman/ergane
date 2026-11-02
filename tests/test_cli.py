import json
import sys
import threading
from pathlib import Path

import pytest

from muagent.cli import main


def test_main_wires_config_agent_and_repl(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "wired-model")
    monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

    main()

    out = capsys.readouterr().out
    assert "model: wired-model" in out
    assert out.endswith("Bye.\n")


def test_main_exits_on_invalid_config(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_names_a_bad_setting_by_its_environment_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MAX_TURNS", "0")

    with pytest.raises(SystemExit):
        main()

    assert "  - MAX_TURNS:" in capsys.readouterr().err


def test_settings_are_read_from_the_environment_in_any_case(
    monkeypatch: pytest.MonkeyPatch,
):
    from muagent.config import load_config

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MAX_TURNS", "7")
    monkeypatch.setenv("streaming", "false")

    config = load_config()

    assert (config.max_turns, config.streaming) == (7, False)


class RecordingRepl:
    """Stands in for Repl so a test can see which Agent main() built."""

    agent = None

    def __init__(self, agent, commands):
        RecordingRepl.agent = agent

    def run(self):
        pass


def test_main_gives_the_agent_the_demo_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

    main()

    assert RecordingRepl.agent is not None
    assert list(RecordingRepl.agent.tools) == ["add", "get_time"]


FAKE_SERVER = {
    "command": sys.executable,
    "args": [str(Path(__file__).parent / "fake_mcp_server.py")],
}


def write_mcp_json(config_home: Path, servers: dict) -> None:
    path = config_home / "muagent" / "mcp.json"
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
        write_mcp_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

        main()

        assert RecordingRepl.agent is not None
        names = list(RecordingRepl.agent.tools)
        assert names[:2] == ["add", "get_time"]
        assert "fake__add" in names
        assert (tmp_path / "muagent" / "fake.log").exists()

    def test_a_server_at_a_url_in_mcp_json_gives_the_agent_its_tools(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from test_mcp_client import serve_http

        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)
        with serve_http() as url:
            write_mcp_json(tmp_path, {"web": {"url": url}})

            main()

        assert RecordingRepl.agent is not None
        assert "web__add" in RecordingRepl.agent.tools

    def test_the_servers_are_stopped_when_muagent_ends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        write_mcp_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

        main()

        assert "mcp-fake" not in threads()

    def test_the_servers_are_stopped_when_the_repl_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        class FailingRepl(RecordingRepl):
            def run(self):
                assert "mcp-fake" in threads()
                raise RuntimeError("repl failed")

        write_mcp_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("muagent.cli.Repl", FailingRepl)

        with pytest.raises(RuntimeError):
            main()

        assert "mcp-fake" not in threads()

    def test_each_server_is_announced_before_the_banner(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_mcp_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

        main()

        out = capsys.readouterr().out
        assert out.startswith("[MCP server 'fake': 13 tools]\n\nAI coding agent")

    def test_a_server_that_fails_does_not_stop_muagent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_mcp_json(tmp_path, {"broken": {"command": "/no/such/server"}})
        monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

        main()

        out = capsys.readouterr().out
        assert "[MCP server 'broken' failed to start: FileNotFoundError" in out
        assert out.endswith("Bye.\n")

    def test_a_mistake_in_mcp_json_is_reported_and_muagent_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        write_mcp_json(tmp_path, {"fake": {"command": "x", "arg": []}})

        with pytest.raises(SystemExit) as exc:
            main()

        err = capsys.readouterr().err
        assert exc.value.code == 1
        assert f"Configuration error in {tmp_path / 'muagent' / 'mcp.json'}:" in err
        assert "  - mcpServers.fake.arg:" in err  # not upper case: not a variable

    def test_ctrl_c_while_the_servers_start_exits_quietly(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def interrupted(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr("muagent.cli.start_servers", interrupted)

        with pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code == 130
