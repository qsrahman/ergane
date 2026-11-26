import importlib
import json
import sys
import threading
from pathlib import Path

import pytest

from ergane.cli import main, parse_args
from ergane.config import private_state_dir
from ergane.sessions import SessionInfo, SessionStore


def test_main_wires_config_agent_and_repl(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

    main(["--model", "wired-model"])

    out = capsys.readouterr().out
    assert "ergane · wired-model" in out
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
    assert f'No API key. Put "api_key" in {tmp_path / "ergane" / "config.json"}' in err
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
    assert "bad.json" in capsys.readouterr().err


def test_a_directory_as_the_config_file_is_reported(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(tmp_path)])

    assert exc.value.code == 1
    assert "Is a directory" in capsys.readouterr().err


def test_another_mistake_gets_no_such_help(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(SystemExit):
        main(["--max-context-tokens", "0"])

    assert "No API key" not in capsys.readouterr().err


def test_main_names_a_bad_setting_by_its_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(SystemExit):
        main(["--max-context-tokens", "0"])

    assert "  - max_context_tokens:" in capsys.readouterr().err


class RecordingRepl:
    """Stands in for Repl so a test can see which Agent main([]) built."""

    agent = None
    store = None
    resumed = None

    def __init__(
        self, agent, commands, session=None, console=None, hub=None, store=None
    ):
        RecordingRepl.agent = agent
        RecordingRepl.store = store
        RecordingRepl.resumed = None

    def run(self, resume=None):
        RecordingRepl.resumed = resume


def test_a_built_in_tool_is_offered_without_being_listed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    package = tmp_path / "fake_builtin"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "look.py").write_text(
        "from ergane.tools import tool\n\n@tool\ndef look() -> str:\n"
        '    """Look."""\n    return "seen"\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)
    monkeypatch.setattr("ergane.cli.builtin", importlib.import_module("fake_builtin"))

    main([])

    assert RecordingRepl.agent is not None
    assert list(RecordingRepl.agent.tools) == ["look"]


def test_a_built_in_module_that_fails_to_load_is_reported_and_the_rest_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    package = tmp_path / "broken_builtin"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "a_broken.py").write_text("import not_installed_anywhere\n")
    (package / "look.py").write_text(
        "from ergane.tools import tool\n\n@tool\ndef look() -> str:\n"
        '    """Look."""\n    return "seen"\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)
    monkeypatch.setattr("ergane.cli.builtin", importlib.import_module("broken_builtin"))

    main([])

    assert RecordingRepl.agent is not None
    assert list(RecordingRepl.agent.tools) == ["look"]
    assert (
        "Tools in a_broken not loaded: ModuleNotFoundError" in capsys.readouterr().out
    )


def test_two_built_in_tools_of_one_name_are_reported_and_the_first_is_offered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    package = tmp_path / "twice_builtin"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for module, text in (("a", "first"), ("b", "second")):
        (package / f"{module}.py").write_text(
            "from ergane.tools import tool\n\n@tool\ndef look() -> str:\n"
            f'    """Look."""\n    return "{text}"\n'
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)
    monkeypatch.setattr("ergane.cli.builtin", importlib.import_module("twice_builtin"))

    main([])

    assert RecordingRepl.agent is not None
    assert RecordingRepl.agent.tools["look"].call("{}") == "first"
    assert "Tool look is defined twice" in capsys.readouterr().out


def test_the_agent_offers_the_built_in_tools_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

    main([])

    assert RecordingRepl.agent is not None
    assert {
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "web_fetch",
        "web_search",
    } <= set(RecordingRepl.agent.tools)


def test_no_builtin_leaves_the_agent_with_no_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

    main(["--no-builtin"])

    assert RecordingRepl.agent is not None
    assert RecordingRepl.agent.tools == {}


class TestBuiltinTools:
    @pytest.fixture(autouse=True)
    def fake_package(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        package = tmp_path / "pick_builtin"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in ("alpha", "beta", "gamma"):
            (package / f"{name}.py").write_text(
                "from ergane.tools import tool\n\n@tool\n"
                f'def {name}() -> str:\n    """Say {name}."""\n    return "{name}"\n'
            )
        monkeypatch.syspath_prepend(str(tmp_path))
        for loaded in [m for m in sys.modules if m.split(".")[0] == "pick_builtin"]:
            monkeypatch.delitem(sys.modules, loaded)  # from another test's directory
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)
        monkeypatch.setattr(
            "ergane.cli.builtin", importlib.import_module("pick_builtin")
        )

    def test_all_are_offered_unless_the_setting_says_otherwise(self):
        main([])

        assert RecordingRepl.agent is not None
        assert list(RecordingRepl.agent.tools) == ["alpha", "beta", "gamma"]

    def test_false_in_the_file_offers_none(self, tmp_path: Path):
        write_config_json(tmp_path, {}, builtin_tools=False)

        main([])

        assert RecordingRepl.agent is not None
        assert RecordingRepl.agent.tools == {}

    @pytest.mark.parametrize("value", [["alpha"], "alpha", 5])
    def test_a_list_or_anything_but_true_or_false_is_refused_naming_the_key(
        self, tmp_path: Path, capsys, value: object
    ):
        write_config_json(tmp_path, {}, builtin_tools=value)

        with pytest.raises(SystemExit) as exc:
            main([])

        assert exc.value.code == 1
        assert "builtin_tools" in capsys.readouterr().err

    def test_the_flag_beats_the_file(self, tmp_path: Path):
        write_config_json(tmp_path, {}, builtin_tools=True)

        main(["--no-builtin"])

        assert RecordingRepl.agent is not None
        assert RecordingRepl.agent.tools == {}


FAKE_SERVER = {
    "command": sys.executable,
    "args": [str(Path(__file__).parent / "fake_mcp_server.py")],
}


def write_config_json(config_home: Path, servers: dict, **settings: object) -> None:
    path = config_home / "ergane" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers, **settings}))


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
        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

        main(["--no-builtin"])

        assert RecordingRepl.agent is not None
        names = list(RecordingRepl.agent.tools)
        assert len(names) == 18  # the fake server's
        assert "fake__add" in names
        assert (tmp_path / "ergane" / "fake.log").exists()

    def test_a_server_at_a_url_in_mcp_json_gives_the_agent_its_tools(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from helpers import serve_http

        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)
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

            def run(self, resume=None):  # the servers run only while the REPL does
                agent = RecordingRepl.agent
                assert agent is not None
                CallingRepl.key = agent.tools["on__getenv"].call('{"name": "KEY"}')

        monkeypatch.setattr("ergane.cli.Repl", CallingRepl)
        write_config_json(
            tmp_path,
            {
                "on": {
                    **FAKE_SERVER,
                    "type": "local",
                    "enabled": True,
                    "env": {"KEY": "${FAKE_KEY}"},
                },
                "off": {**FAKE_SERVER, "type": "local", "enabled": False},
                "unset": {**FAKE_SERVER, "env": {"KEY": "${NOT_SET_ANYWHERE}"}},
            },
        )

        main([])

        assert RecordingRepl.agent is not None
        tools = RecordingRepl.agent.tools
        assert "on__add" in tools
        assert not any(name.startswith(("off__", "unset__")) for name in tools)
        assert CallingRepl.key == "the-key"

    def test_the_servers_are_stopped_when_ergane_ends(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

        main([])

        assert "mcp-fake" not in threads()

    def test_the_servers_are_stopped_when_the_repl_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        class FailingRepl(RecordingRepl):
            def run(self, resume=None):
                assert "mcp-fake" in threads()
                raise RuntimeError("repl failed")

        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("ergane.cli.Repl", FailingRepl)

        with pytest.raises(RuntimeError):
            main([])

        assert "mcp-fake" not in threads()

    def test_a_server_that_starts_is_not_announced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")

        main(["--model", "banner-model"])

        # The banner is the first line: nothing was said of the server.
        first_line = capsys.readouterr().out.splitlines()[0]
        assert first_line.startswith("ergane · banner-model · ")
        assert first_line.endswith(" · /help for commands")

    def test_a_server_that_fails_does_not_stop_ergane(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_config_json(tmp_path, {"broken": {"command": "/no/such/server"}})
        monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")
        monkeypatch.setenv("COLUMNS", "500")  # the failure is one line, not wrapped

        main([])

        out = capsys.readouterr().out
        assert out.startswith("[MCP server 'broken' failed to start: FileNotFoundError")
        # A blank line sets the failure apart from the banner under it.
        _failure, blank, banner = out.splitlines()[:3]
        assert blank == "" and banner.startswith("ergane · ")
        assert out.endswith("Bye.\n")

    def test_servers_tells_again_what_became_of_a_server_that_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
    ):
        write_config_json(
            tmp_path,
            {
                "broken": {"command": "/no/such/server"},
                "good": {**FAKE_SERVER, "tools": ["add"]},
                "off": {**FAKE_SERVER, "enabled": False},
            },
        )
        inputs = iter(["/servers", "/exit"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        main([])

        out = capsys.readouterr().out
        assert out.count("failed to start: FileNotFoundError") == 2  # shown, and again
        assert '  good    running, 1 tool (17 left out by "tools"), ~' in out
        assert "  off     off: enabled is false" in out
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

    def test_a_mistake_in_mcp_json_is_reported_and_ergane_exits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        write_config_json(tmp_path, {"fake": {"command": "x", "arg": []}})

        with pytest.raises(SystemExit) as exc:
            main([])

        err = capsys.readouterr().err
        assert exc.value.code == 1
        assert err.startswith("Configuration error:")
        assert "  - mcpServers.fake.local.arg:" in err

    def test_ctrl_c_while_the_servers_start_exits_quietly(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("ergane.cli.McpHub.start", interrupted)

        with pytest.raises(SystemExit) as exc:
            main([])

        assert exc.value.code == 130


def test_version_prints_the_version_of_the_package_and_exits(capsys):
    from importlib.metadata import version

    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out == f"ergane {version('ergane')}\n"


class TestFlags:
    @pytest.fixture(autouse=True)
    def api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

    def test_a_flag_that_is_not_given_is_not_in_the_result(self):
        assert vars(parse_args([])) == {"no_mcp": False}

    def test_every_flag(self):
        args = parse_args(
            [
                "--config", "c.json", "--model", "m", "--base-url", "u",
                "--no-streaming", "--max-context-tokens", "9", "--no-builtin",
                "--reasoning-effort", "low", "--no-mcp",
            ]
        )  # fmt: skip

        assert vars(args) == {
            "config": Path("c.json"),
            "model": "m",
            "base_url": "u",
            "streaming": False,
            "max_context_tokens": 9,
            "builtin_tools": False,
            "reasoning_effort": "low",
            "no_mcp": True,
        }

    def test_there_is_no_flag_for_the_key(self, capsys):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--api-key", "sk-in-the-process-list"])

        assert exc.value.code == 2

    def test_a_flag_wins_over_the_file(self, tmp_path: Path):
        (tmp_path / "ergane").mkdir()
        (tmp_path / "ergane" / "config.json").write_text(
            json.dumps({"model": "file-model", "max_context_tokens": 5000})
        )

        main(["--model", "flag-model"])

        agent = RecordingRepl.agent
        assert agent is not None
        assert agent.config.model == "flag-model"  # flag > file
        assert agent.config.max_context_tokens == 5000  # the file, with no flag

    def test_default_as_the_effort_sends_none_over_the_file(self, tmp_path: Path):
        (tmp_path / "ergane").mkdir()
        (tmp_path / "ergane" / "config.json").write_text(
            json.dumps({"reasoning_effort": "high"})
        )

        main(["--reasoning-effort", "default"])

        assert RecordingRepl.agent is not None
        assert RecordingRepl.agent.config.reasoning_effort is None

    def test_a_bad_value_is_reported_by_its_key(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--max-context-tokens", "0"])

        assert exc.value.code == 1
        assert "  - max_context_tokens:" in capsys.readouterr().err

    def test_the_removed_max_turns_flag_is_refused(self, capsys):
        with pytest.raises(SystemExit) as exc:
            parse_args(["--max-turns", "5"])

        assert exc.value.code == 2
        assert "--max-turns" in capsys.readouterr().err

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
        assert "nowhere.json" in capsys.readouterr().err

    def test_no_mcp_starts_none_of_the_servers(self, tmp_path: Path):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})

        main(["--no-mcp", "--no-builtin"])

        assert RecordingRepl.agent is not None
        assert RecordingRepl.agent.tools == {}
        assert "mcp-fake" not in threads()


class TestMcpServersInTheBackground:
    """At a terminal the prompt comes up before the servers are ready."""

    @pytest.fixture(autouse=True)
    def terminal(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setattr("ergane.cli.terminal_session", lambda *a, **k: object())
        write_config_json(tmp_path, {"fake": FAKE_SERVER})

    def test_the_agent_starts_without_tools_and_the_hub_has_them_coming(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        seen = {}

        class Recording(RecordingRepl):
            def __init__(
                self, agent, commands, session=None, console=None, hub=None, store=None
            ):
                super().__init__(agent, commands, session, console, hub, store)
                seen["hub"] = hub
                seen["tools_at_start"] = dict(agent.tools)

        monkeypatch.setattr("ergane.cli.Repl", Recording)

        main(["--no-builtin"])

        assert seen["tools_at_start"] == {}
        assert seen["hub"] is not None

    def test_the_servers_are_stopped_when_ergane_ends(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

        main([])

        assert not any(name.startswith("mcp-") for name in threads())

    def test_without_a_terminal_the_start_waits_for_the_servers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("ergane.cli.terminal_session", lambda *a, **k: None)
        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

        main([])

        assert RecordingRepl.agent is not None
        assert "fake__add" in RecordingRepl.agent.tools


@pytest.mark.parametrize("terminal", [True, False])
def test_only_input_from_a_terminal_can_answer_a_question(
    monkeypatch: pytest.MonkeyPatch, terminal: bool
):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr("sys.stdin.isatty", lambda: terminal, raising=False)
    monkeypatch.setattr("ergane.cli.terminal_session", lambda *a, **k: None)
    seen = []

    class Recording(RecordingRepl):
        def __init__(
            self, agent, commands, session=None, console=None, hub=None, store=None
        ):
            super().__init__(agent, commands, session, console, hub, store)
            seen.append(console)

    monkeypatch.setattr("ergane.cli.Repl", Recording)

    main([])

    assert seen[0].can_ask is terminal


class TestResume:
    """`--resume [ID]` continues a saved session of the directory ergane runs in."""

    @pytest.fixture(autouse=True)
    def repl(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setattr("ergane.cli.Repl", RecordingRepl)

    @staticmethod
    def save(prompt: str) -> SessionInfo:
        """A session saved where `main` looks for it: the state and working directory."""
        store = SessionStore(private_state_dir() / "sessions", Path.cwd())
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "hi"},
        ]
        return store.save(None, None, conversation, "gpt-4o")

    def test_the_flag_takes_an_id_or_none(self):
        assert vars(parse_args(["--resume"])) == {"no_mcp": False, "resume": ""}
        assert vars(parse_args(["--resume", "2026"])) == {
            "no_mcp": False,
            "resume": "2026",
        }
        assert "resume" not in vars(parse_args([]))

    def test_without_an_id_the_last_session_is_continued(self, clock):
        self.save("older")
        newest = self.save("newest")

        main(["--resume"])

        assert RecordingRepl.resumed is not None
        assert RecordingRepl.resumed.info == newest

    def test_an_id_or_the_start_of_one_names_the_session(self, clock):
        older = self.save("older")
        self.save("newest")

        main(["--resume", older.id[:-1]])

        assert RecordingRepl.resumed is not None
        assert RecordingRepl.resumed.info == older

    def test_without_the_flag_nothing_is_continued(self, clock):
        self.save("older")

        main([])

        assert RecordingRepl.resumed is None

    def test_the_store_is_the_state_directory_of_the_working_directory(self):
        main([])

        assert RecordingRepl.store is not None
        assert RecordingRepl.store.cwd == Path.cwd().resolve()
        assert RecordingRepl.store.directory.parent == private_state_dir() / "sessions"

    def test_with_nothing_saved_it_says_so_and_starts_a_new_session(self, capsys):
        main(["--resume"])

        assert RecordingRepl.resumed is None
        assert "[No saved session for this directory]" in capsys.readouterr().out

    def test_an_id_that_names_nothing_is_an_error_before_anything_starts(
        self, tmp_path, capsys
    ):
        write_config_json(tmp_path, {"fake": FAKE_SERVER})
        RecordingRepl.agent = None
        self.save("one")

        with pytest.raises(SystemExit) as exit:
            main(["--resume", "nope"])

        assert exit.value.code == 1
        assert "No saved session 'nope' for this directory" in capsys.readouterr().err
        assert RecordingRepl.agent is None  # no agent, and no servers either
        assert "mcp-fake" not in threads()

    def test_a_session_from_another_directory_is_not_found(
        self, tmp_path, monkeypatch, capsys
    ):
        info = self.save("here")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        with pytest.raises(SystemExit):
            main(["--resume", info.id])

        assert "No saved session" in capsys.readouterr().err
