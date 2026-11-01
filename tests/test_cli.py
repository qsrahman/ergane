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


def test_main_gives_the_agent_the_registered_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

    main()

    assert RecordingRepl.agent is not None
    assert list(RecordingRepl.agent.tools) == ["add", "get_time"]
