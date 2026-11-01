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


class RecordingRepl:
    """Stands in for Repl so a test can see which Agent main() built."""

    agent = None

    def __init__(self, agent):
        RecordingRepl.agent = agent

    def run(self):
        pass


@pytest.mark.parametrize(
    ("setting", "expected"),
    [(None, ["add", "get_time"]), ("read", ["add", "get_time"]), ("off", [])],
)
def test_main_gives_the_agent_the_tools_the_setting_allows(
    monkeypatch: pytest.MonkeyPatch, setting, expected
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    if setting:
        monkeypatch.setenv("TOOLS", setting)
    monkeypatch.setattr("muagent.cli.Repl", RecordingRepl)

    main()

    assert RecordingRepl.agent is not None
    assert list(RecordingRepl.agent.tools) == expected
