import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import make_config
from muagent.config import load_config, private_state_dir


def test_invalid_reasoning_effort_is_a_config_error():
    with pytest.raises(ValidationError, match="reasoning_effort"):
        make_config(REASONING_EFFORT="extreme")


def write_json(path: Path, values: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(values if isinstance(values, str) else json.dumps(values))
    return path


class TestLoadConfig:
    """isolate_config makes tmp_path both the working directory and XDG_CONFIG_HOME."""

    @pytest.fixture
    def user_file(self, tmp_path: Path) -> Path:
        return tmp_path / "muagent" / "config.json"

    def test_reads_the_user_config_file(self, user_file: Path):
        write_json(user_file, {"api_key": "user", "model": "user-model"})

        config = load_config()

        assert config.model == "user-model"
        assert config.api_key.get_secret_value() == "user"

    def test_a_missing_file_is_fine_while_the_environment_has_what_is_needed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "x")

        assert load_config().model == "gpt-5.4-mini"

    def test_every_setting_has_a_key_in_the_file(self, user_file: Path):
        write_json(
            user_file,
            {
                "api_key": "k",
                "model": "m",
                "base_url": "http://localhost:1/v1",
                "streaming": False,
                "max_context_tokens": 1000,
                "max_turns": 3,
                "reasoning_effort": "low",
            },
        )

        config = load_config()

        assert (
            config.model,
            config.base_url,
            config.streaming,
            config.max_context_tokens,
            config.max_turns,
            config.reasoning_effort,
        ) == ("m", "http://localhost:1/v1", False, 1000, 3, "low")

    @pytest.mark.parametrize(
        ("field", "file", "flag"),
        [
            ("model", "from-file", "from-flag"),
            ("base_url", "http://f/v1", "http://g/v1"),
            ("streaming", False, True),
            ("max_context_tokens", 100, 300),
            ("max_turns", 1, 3),
            ("auto_compact", False, True),
            ("compact_at", 0.5, 0.9),
            ("keep_recent_tokens", 10, 20),
            ("reasoning_effort", "low", "high"),
        ],
    )
    def test_the_flag_beats_the_file_beats_the_default(
        self, user_file, monkeypatch, field, file, flag
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        default = getattr(load_config(), field)

        write_json(user_file, {field: file})
        assert getattr(load_config(), field) == file != default

        assert getattr(load_config(**{field: flag}), field) == flag

    @pytest.mark.parametrize(
        "name",
        [
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
            "STREAMING",
            "MAX_CONTEXT_TOKENS",
            "REASONING_EFFORT",
            "AUTO_COMPACT",
            "COMPACT_AT",
            "KEEP_RECENT_TOKENS",
            "MAX_TURNS",
        ],
    )
    def test_no_variable_but_the_key_sets_anything(self, user_file, monkeypatch, name):
        # Not even a mistaken one: a variable for these is never looked at.
        write_json(user_file, {"api_key": "k"})
        before = load_config().model_dump()
        monkeypatch.setenv(
            name, "http://elsewhere.example/v1" if "URL" in name else "0"
        )

        assert load_config().model_dump() == before

    def test_the_api_key_comes_from_the_environment_over_the_file(
        self, user_file, monkeypatch
    ):
        write_json(user_file, {"api_key": "file-key"})
        assert load_config().api_key.get_secret_value() == "file-key"

        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        assert load_config().api_key.get_secret_value() == "env-key"

    def test_a_flag_can_ask_for_the_default_over_a_value_in_the_file(
        self, user_file, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        write_json(user_file, {"reasoning_effort": "high"})

        assert load_config(reasoning_effort=None).reasoning_effort is None

    def test_a_config_file_in_the_working_directory_is_not_read(
        self, user_file: Path, tmp_path: Path
    ):
        # A project must not be able to point the API key and the prompts elsewhere.
        write_json(user_file, {"api_key": "user", "model": "user-model"})
        write_json(
            tmp_path / "config.json",
            {"api_key": "stolen", "base_url": "https://attacker.example/v1"},
        )
        write_json(tmp_path / ".muagent" / "config.json", {"model": "project-model"})

        config = load_config()

        assert config.model == "user-model"
        assert config.api_key.get_secret_value() == "user"
        assert config.base_url is None

    def test_a_dotenv_file_is_not_read_any_more(self, user_file: Path, tmp_path: Path):
        (user_file.parent).mkdir(parents=True)
        (user_file.parent / ".env").write_text("OPENAI_API_KEY=a\n")
        (tmp_path / ".env").write_text("OPENAI_API_KEY=b\n")

        with pytest.raises(ValidationError, match="api_key"):
            load_config()

    def test_reads_the_default_location_when_xdg_config_home_is_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        home = tmp_path / "home"
        write_json(
            home / ".config" / "muagent" / "config.json",
            {"api_key": "user", "model": "default-location"},
        )
        monkeypatch.delenv("XDG_CONFIG_HOME")
        monkeypatch.setenv("HOME", str(home))

        assert load_config().model == "default-location"

    def test_a_relative_xdg_config_home_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The XDG spec says relative paths are invalid, so ~/.config is used instead.
        home = tmp_path / "home"
        write_json(
            home / ".config" / "muagent" / "config.json",
            {"api_key": "user", "model": "from-home"},
        )
        write_json(  # what honoring the relative path would have read
            tmp_path / "relative" / "muagent" / "config.json",
            {"api_key": "wrong", "model": "from-relative"},
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative")
        monkeypatch.setenv("HOME", str(home))

        assert load_config().model == "from-home"


class TestTheEnvironment:
    """One variable, OPENAI_API_KEY; the file's keys are the fields' names."""

    @pytest.mark.parametrize("name", ["MODEL", "API_KEY", "BASE_URL", "MCP_SERVERS"])
    def test_a_bare_variable_other_programs_may_export_is_ignored(
        self, monkeypatch, name
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "right-key")
        monkeypatch.setenv(name, "http://other.example/v1" if "URL" in name else "x")

        config = load_config()

        assert config.model == "gpt-5.4-mini" and config.base_url is None
        assert config.api_key.get_secret_value() == "right-key"
        assert config.mcp_servers == {}

    def test_a_bare_api_key_does_not_stand_in_for_the_key(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "some-other-programs-key")

        with pytest.raises(ValidationError, match="api_key"):
            load_config()

    @pytest.mark.parametrize("name", ["MCPSERVERS", "MCP_SERVERS", "mcpServers"])
    def test_a_stray_variable_for_the_servers_is_not_even_parsed(
        self, monkeypatch, name
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv(name, "not json")

        assert load_config().mcp_servers == {}

    def test_openai_api_key_is_the_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")

        assert load_config().api_key.get_secret_value() == "k"

    def test_the_file_uses_the_names_of_the_settings_and_no_other(self, tmp_path: Path):
        write_json(
            tmp_path / "muagent" / "config.json",
            {"OPENAI_API_KEY": "k", "OPENAI_MODEL": "by-env-name"},
        )

        with pytest.raises(ValidationError, match=r"OPENAI_MODEL\n  Extra inputs"):
            load_config()


class TestNamedFile:
    def test_a_file_the_user_names_is_read_instead(self, tmp_path: Path):
        write_json(tmp_path / "muagent" / "config.json", {"api_key": "k", "model": "a"})
        other = write_json(tmp_path / "other.json", {"api_key": "k", "model": "b"})

        assert load_config(other).model == "b"

    def test_a_muagent_config_variable_names_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # A project's .envrc could point it at a file of its own.
        write_json(
            tmp_path / "muagent" / "config.json", {"api_key": "k", "model": "mine"}
        )
        evil = write_json(
            tmp_path / "evil.json",
            {
                "api_key": "k",
                "model": "theirs",
                "mcpServers": {"x": {"command": "touch"}},
            },
        )
        monkeypatch.setenv("MUAGENT_CONFIG", str(evil))

        config = load_config()

        assert config.model == "mine" and config.mcp_servers == {}

    def test_a_file_that_does_not_exist_is_an_error_when_the_user_named_it(
        self, tmp_path: Path
    ):
        with pytest.raises(FileNotFoundError, match="nowhere.json"):
            load_config(tmp_path / "nowhere.json")


class TestUnreadableFile:
    @pytest.fixture
    def user_file(self, tmp_path: Path) -> Path:
        return tmp_path / "muagent" / "config.json"

    def test_a_file_that_is_not_utf8_is_a_config_error(self, user_file: Path):
        user_file.parent.mkdir(parents=True)
        user_file.write_bytes(b"\xff\xfe{")

        with pytest.raises(ValidationError, match=r"config\.json is not UTF-8"):
            load_config()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
    def test_a_file_that_cannot_be_read_is_a_config_error(self, user_file: Path):
        write_json(user_file, {"api_key": "k"})
        user_file.chmod(0o000)
        try:
            with pytest.raises(
                ValidationError, match=r"cannot read .*Permission denied"
            ):
                load_config()
        finally:
            user_file.chmod(0o644)

    def test_a_directory_the_user_named_is_refused(self, tmp_path: Path):
        with pytest.raises(IsADirectoryError, match="is a directory"):
            load_config(tmp_path)


class TestMistakes:
    @pytest.fixture
    def user_file(self, tmp_path: Path) -> Path:
        return tmp_path / "muagent" / "config.json"

    def test_a_key_that_is_not_a_setting_is_named(self, user_file: Path):
        write_json(user_file, {"api_key": "k", "modle": "x"})

        with pytest.raises(ValidationError, match=r"modle\n  Extra inputs"):
            load_config()

    def test_a_bad_value_names_its_key(self, user_file: Path):
        write_json(user_file, {"api_key": "k", "max_turns": 0})

        with pytest.raises(ValidationError, match="max_turns"):
            load_config()

    def test_a_file_that_is_not_json_says_which(self, user_file: Path):
        write_json(user_file, "{ not json")

        with pytest.raises(ValidationError, match=r"Invalid JSON: .*config\.json"):
            load_config()

    def test_a_file_that_is_not_an_object_is_refused(self, user_file: Path):
        write_json(user_file, '["model"]')

        with pytest.raises(ValidationError, match="valid dictionary"):
            load_config()

    def test_an_environment_variable_that_is_not_a_setting_is_ignored(
        self, user_file: Path, monkeypatch
    ):
        monkeypatch.setenv("SOME_OTHER_TOOL", "1")
        write_json(user_file, {"api_key": "k"})

        assert load_config().api_key.get_secret_value() == "k"


class TestApiKeyPlaceholder:
    @pytest.fixture
    def user_file(self, tmp_path: Path) -> Path:
        return tmp_path / "muagent" / "config.json"

    def test_the_key_can_be_a_variable(self, user_file: Path, monkeypatch):
        write_json(user_file, {"api_key": "${MY_KEY}"})
        monkeypatch.setenv("MY_KEY", "sk-from-env")

        assert load_config().api_key.get_secret_value() == "sk-from-env"

    def test_a_variable_that_is_not_set_names_itself_not_a_value(self, user_file: Path):
        write_json(user_file, {"api_key": "${MY_KEY}"})

        with pytest.raises(ValidationError, match=r"\$\{MY_KEY\} is not set"):
            load_config()

    def test_a_placeholder_the_environment_overrode_is_never_looked_at(
        self, user_file: Path, monkeypatch
    ):
        write_json(user_file, {"api_key": "${MY_KEY}"})
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")

        assert load_config().api_key.get_secret_value() == "from-env"

    def test_the_text_a_variable_holds_is_not_looked_at_again(
        self, user_file: Path, monkeypatch
    ):
        write_json(user_file, {"api_key": "${A}"})
        monkeypatch.setenv("A", "${B}")
        monkeypatch.setenv("B", "wrong")

        assert load_config().api_key.get_secret_value() == "${B}"


class TestMcpServers:
    @pytest.fixture
    def user_file(self, tmp_path: Path) -> Path:
        return tmp_path / "muagent" / "config.json"

    def test_the_servers_are_listed_in_the_file(self, user_file: Path):
        write_json(
            user_file,
            {
                "api_key": "k",
                "mcpServers": {
                    "files": {
                        "command": "my-server",
                        "args": ["--x"],
                        "read_only": True,
                    }
                },
            },
        )

        servers = load_config().mcp_servers

        assert list(servers) == ["files"]
        assert servers["files"].command == "my-server" and servers["files"].read_only

    def test_there_are_none_by_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")

        assert load_config().mcp_servers == {}

    @pytest.mark.parametrize("name", ["MCPSERVERS", "mcpServers", "mcpservers"])
    def test_the_environment_cannot_define_servers(
        self, user_file: Path, monkeypatch, name
    ):
        # A server is a program: only the user's own file may list them, or a
        # project's .envrc could start one.
        write_json(
            user_file,
            {"api_key": "k", "mcpServers": {"mine": {"command": "my-server"}}},
        )
        monkeypatch.setenv(name, '{"evil": {"command": "echo"}}')

        assert list(load_config().mcp_servers) == ["mine"]

    def test_a_mistake_in_a_server_names_where_it_is(self, user_file: Path):
        write_json(
            user_file,
            {"api_key": "k", "mcpServers": {"s": {"command": "x", "cwd": "/srv"}}},
        )

        with pytest.raises(ValidationError, match=r"mcpServers\.s\.cwd"):
            load_config()

    def test_a_bad_server_name_is_refused(self, user_file: Path):
        write_json(
            user_file, {"api_key": "k", "mcpServers": {"two words": {"command": "x"}}}
        )

        with pytest.raises(ValidationError, match="mcpServers"):
            load_config()


def test_the_api_key_is_not_revealed_when_the_config_is_printed():
    config = make_config(api_key="sk-secret-123")

    for text in (repr(config), str(config), str(config.model_dump())):
        assert "sk-secret-123" not in text
    assert config.api_key.get_secret_value() == "sk-secret-123"


def test_an_empty_api_key_is_a_config_error():
    with pytest.raises(ValidationError, match="api_key"):
        make_config(api_key="")


def test_the_state_directory_is_closed_to_other_users(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir(mode=0o755)
    state.chmod(0o755)
    monkeypatch.setenv("XDG_STATE_HOME", str(state))

    path = private_state_dir()

    assert path == state / "muagent"
    assert path.stat().st_mode & 0o777 == 0o700
    path.chmod(0o755)
    private_state_dir()  # an existing one is closed too
    assert path.stat().st_mode & 0o777 == 0o700
