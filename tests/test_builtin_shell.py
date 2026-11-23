import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from ergane.builtin.shell import run_command
from ergane.tools import MAX_OUTPUT_CHARS


def run(command: str, **arguments: object) -> str:
    return run_command.call(json.dumps({"command": command, **arguments}))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone(pid: int, wait: float = 3.0) -> bool:
    """Whether a process ends within a few seconds (a killed child may be a zombie
    for a moment, until init collects it)."""
    end = time.monotonic() + wait
    while time.monotonic() < end:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


@pytest.fixture(autouse=True)
def private_temp_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Output that is cut is saved in the temp directory: keep it out of the real one."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def saved_path(result: str) -> Path:
    return Path(result.rsplit("saved to ", 1)[1].split(":", 1)[0])


class TestTheFullOutput:
    def test_output_that_was_cut_is_saved_whole_and_the_result_says_where(self):
        result = run("seq 1 100000")

        saved = saved_path(result).read_text()
        assert saved.startswith("1\n2\n") and saved.endswith("99999\n100000")
        assert saved.count("\n") == 99_999
        assert result.endswith("read it with read_file]")

    def test_the_model_can_read_the_middle_with_read_file(self):
        from ergane.builtin.files import read_file

        path = saved_path(run("seq 1 100000"))

        text = read_file.call(json.dumps({"path": str(path), "offset": 50_000}))

        assert text.startswith("50000\n50001\n")

    def test_both_streams_are_saved_labelled(self):
        saved = saved_path(run("seq 1 50000; seq 1 50000 >&2; exit 1")).read_text()

        assert saved.startswith("exit code 1\n[stdout]\n1\n")
        assert "\n[stderr]\n1\n" in saved and "omitted" not in saved

    def test_a_long_stderr_alone_is_saved_too(self):
        saved = saved_path(run("seq 1 100000 >&2")).read_text()

        assert saved.startswith("[stderr]\n1\n") and saved.endswith("99999\n100000")

    def test_output_that_fits_leaves_no_file(self, tmp_path: Path):
        result = run("seq 1 100")

        assert "full output" not in result
        assert not list(tmp_path.glob("ergane-output-*"))

    def test_the_saved_file_is_private(self):
        mode = saved_path(run("seq 1 100000")).stat().st_mode

        assert stat.S_IMODE(mode) == 0o600

    def test_a_disk_that_cannot_be_written_still_gives_the_result(self, monkeypatch):
        def full(*args, **kwargs):
            raise OSError("no space left")

        monkeypatch.setattr(tempfile, "mkstemp", full)

        result = run("seq 1 100000")

        assert "characters omitted" in result and "full output" not in result


class TestTheResult:
    def test_the_output_of_a_command_that_succeeded_is_just_that(self):
        assert run("echo hello") == "hello"

    def test_a_command_that_prints_nothing_says_so(self):
        assert run("true") == "(no output)"

    def test_a_failed_command_starts_with_its_exit_code(self):
        assert run("echo partial; exit 3") == "exit code 3\n[stdout]\npartial"

    def test_stderr_is_labelled_when_there_is_any(self):
        assert run("echo out; echo err >&2") == "[stdout]\nout\n[stderr]\nerr"

    def test_stderr_alone_from_a_command_that_succeeded_is_labelled(self):
        assert run("echo warning >&2") == "[stderr]\nwarning"

    def test_a_command_that_is_not_found_is_an_exit_code_not_an_error(self):
        result = run("definitely-not-a-command-xyz")

        assert result.startswith("exit code 127") and "[stderr]" in result

    def test_a_command_killed_by_a_signal_says_which(self):
        assert run("kill -9 $$").startswith("killed by signal 9")

    def test_output_that_is_not_utf8_does_not_break_it(self):
        assert run(r"printf 'caf\351'") == "caf�"


class TestWhatIsLeftOutOfTheOutput:
    def test_terminal_escape_codes_are_removed(self):
        assert run(r"printf '\033[31mred\033[0m \033]0;title\007ok'") == "red ok"

    def test_a_progress_bar_that_rewrites_its_line_keeps_the_last_version(self):
        assert run(r"printf '10%%\r50%%\r100%%\ndone\n'") == "100%\ndone"

    def test_crlf_becomes_a_newline(self):
        assert run(r"printf 'a\r\nb\r\n'") == "a\nb"

    def test_binary_output_becomes_a_note(self):
        assert run(r"printf '\000\001\002 binary'") == (
            "[binary output omitted: 10 bytes]"
        )

    def test_long_output_keeps_its_start_and_its_end(self):
        result = run("seq 1 100000")

        assert result.startswith("1\n2\n") and "99999\n100000\n[full output" in result
        assert "characters omitted" in result
        assert len(result) < MAX_OUTPUT_CHARS

    def test_when_both_streams_are_long_each_gets_room(self):
        result = run("seq 1 50000; seq 1 50000 >&2; exit 1")

        assert "[stdout]\n1\n" in result and "[stderr]\n1\n" in result
        assert result.count("characters omitted") == 2
        assert len(result) < MAX_OUTPUT_CHARS


class TestTheProcess:
    def test_stdin_is_closed_so_a_command_that_reads_ends_at_once(self):
        started = time.monotonic()

        assert run("cat") == "(no output)"
        assert time.monotonic() - started < 5

    def test_stdin_is_closed_even_when_ergane_has_one_that_stays_open(
        self, tmp_path: Path
    ):
        # Under pytest fd 0 is /dev/null, which hides a command that inherits it. Here
        # ergane's own stdin is a pipe nobody writes to or closes.
        code = (
            "import json;from ergane.builtin.shell import run_command;"
            "print(run_command.call(json.dumps({'command': 'cat', 'timeout_s': 20})))"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
        )
        try:
            # Not `communicate`: it closes the child's stdin, which is the point here.
            child.wait(timeout=15)
            assert child.stdout is not None
            output = child.stdout.read()
        finally:
            child.kill()

        assert output.strip() == "(no output)"

    def test_it_runs_in_the_working_directory_by_default(self, tmp_path: Path):
        assert Path(run("pwd")).resolve() == tmp_path.resolve()

    def test_cwd_picks_another_directory(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()

        assert Path(run("pwd", cwd="sub")).resolve() == (tmp_path / "sub").resolve()

    def test_cwd_outside_the_allowed_directories_is_refused_and_nothing_runs(
        self, tmp_path: Path
    ):
        result = run("touch ran.txt", cwd="/etc")

        assert "Access denied" in result and not (tmp_path / "ran.txt").exists()

    def test_cwd_that_is_a_file_is_refused(self, tmp_path: Path):
        (tmp_path / "f.txt").write_text("")

        assert "is not a directory" in run("true", cwd="f.txt")

    def test_cd_does_not_carry_over_to_the_next_call(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        run("cd sub")

        assert Path(run("pwd")).resolve() == tmp_path.resolve()

    def test_the_api_key_is_not_in_the_commands_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("MY_TOOL_SETTING", "visible")

        assert run('echo "key=${OPENAI_API_KEY:-unset} tool=$MY_TOOL_SETTING"') == (
            "key=unset tool=visible"
        )

    def test_the_command_itself_is_not_confined_only_cwd_is(self):
        result = run("ls /etc")

        assert result.strip() and "Access denied" not in result


class TestStoppingIt:
    def test_a_command_over_its_timeout_is_stopped_and_says_so(self):
        started = time.monotonic()

        result = run("echo before; sleep 30", timeout_s=1)

        assert time.monotonic() - started < 6
        assert "command timed out after 1 s and was stopped" in result
        assert "before" in result

    def test_the_whole_process_group_is_stopped(self, tmp_path: Path):
        run("sleep 30 & echo $! > child.pid; wait", timeout_s=1)

        assert gone(int((tmp_path / "child.pid").read_text()))

    def test_a_command_that_ignores_sigterm_is_killed(self, tmp_path: Path):
        started = time.monotonic()

        result = run("trap '' TERM; echo $$ > pid; sleep 30", timeout_s=1)

        assert "timed out" in result and time.monotonic() - started < 8
        assert gone(int((tmp_path / "pid").read_text()))

    def test_a_background_process_does_not_hold_the_call_up(self, tmp_path: Path):
        started = time.monotonic()

        result = run("sleep 30 & echo $! > bg.pid; echo done")

        assert time.monotonic() - started < 6
        assert "done" in result and "background processes" in result
        assert gone(int((tmp_path / "bg.pid").read_text()))

    def test_a_process_that_left_the_group_does_not_hold_the_call_up(
        self, tmp_path: Path
    ):
        leave = (
            "import subprocess; print(subprocess.Popen(['sleep', '30'], "
            "start_new_session=True).pid)"
        )
        started = time.monotonic()
        try:
            result = run(f'python3 -c "{leave}" > left.pid; echo done', timeout_s=5)
        finally:
            for pid_file in tmp_path.glob("left.pid"):
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ValueError, ProcessLookupError:
                    pass

        assert time.monotonic() - started < 6
        assert "done" in result

    def test_a_runaway_output_stops_the_command(self, monkeypatch):
        monkeypatch.setattr("ergane.builtin.shell._MAX_CAPTURE_BYTES", 200_000)
        started = time.monotonic()

        result = run("yes")

        assert time.monotonic() - started < 10
        assert "over 10 MB" in result and "command was stopped" in result
        assert len(result) < MAX_OUTPUT_CHARS

    @pytest.mark.parametrize("stream", ["", ">&2"])
    def test_output_over_the_cap_from_a_command_that_ends_says_so(
        self, monkeypatch, stream
    ):
        monkeypatch.setattr("ergane.builtin.shell._MAX_CAPTURE_BYTES", 1000)

        result = run(f"head -c 5000 /dev/zero | tr '\\0' x {stream}")

        assert "was cut off" in result and "command was stopped" not in result

    @pytest.mark.skipif(sys.platform == "win32", reason="needs signals")
    def test_ctrl_c_stops_the_command_and_is_not_swallowed(self, tmp_path: Path):
        timer = threading.Timer(0.7, os.kill, (os.getpid(), signal.SIGINT))
        timer.start()
        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                run("sleep 30 & echo $! > child.pid; wait")
        finally:
            timer.cancel()

        # Back at once, not when the command would have ended on its own.
        assert time.monotonic() - started < 6
        assert gone(int((tmp_path / "child.pid").read_text()))


class TestAsAnTool:
    def test_it_asks_first(self):
        assert not run_command.read_only

    @pytest.mark.parametrize("seconds", [0, 3601])
    def test_a_timeout_out_of_range_is_answered_without_running(self, seconds):
        assert run("true", timeout_s=seconds).startswith("Invalid arguments")

    def test_an_empty_command_is_answered_without_running(self):
        assert run("").startswith("Invalid arguments")
