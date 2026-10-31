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
