import tempfile
from pathlib import Path

import pytest

from ergane.builtin._paths import PathError, absolute_path, default_roots, resolve


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A directory to use as the only root, with an outside directory beside it."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.txt").write_text("a")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("s")
    return root.resolve()


class TestInsideTheRoots:
    def test_an_absolute_path_comes_back_real(self, project: Path):
        assert resolve(str(project / "a.txt"), [project]) == project / "a.txt"

    def test_a_relative_path_is_taken_from_the_working_directory(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(project)

        assert resolve("a.txt", [project]) == project / "a.txt"

    def test_a_file_that_does_not_exist_yet_is_accepted(self, project: Path):
        assert resolve(str(project / "new.txt"), [project]) == project / "new.txt"

    def test_missing_directories_are_added_as_written(self, project: Path):
        wanted = project / "x" / "y" / "z.txt"

        assert resolve(str(wanted), [project]) == wanted

    def test_a_root_is_inside_itself(self, project: Path):
        assert resolve(str(project), [project]) == project

    def test_a_dot_dot_that_stays_inside_is_fine(self, project: Path):
        (project / "sub").mkdir()

        assert resolve(str(project / "sub" / ".." / "a.txt"), [project]) == (
            project / "a.txt"
        )

    def test_tilde_is_the_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))

        assert resolve("~/notes.txt", [home.resolve()]) == home.resolve() / "notes.txt"

    def test_a_file_in_the_middle_is_left_to_the_tools_own_error(self, project: Path):
        # The tool's open() says "Not a directory"; this is only about where it is.
        assert resolve(str(project / "a.txt" / "x"), [project]) == (
            project / "a.txt" / "x"
        )


class TestOutsideTheRoots:
    def test_a_dot_dot_that_leaves_is_refused(self, project: Path):
        with pytest.raises(PathError, match="Access denied"):
            resolve(str(project / ".." / "outside" / "secret.txt"), [project])

    def test_a_sibling_that_shares_the_prefix_is_refused(self, project: Path):
        sibling = project.parent / "proj-evil"
        sibling.mkdir()

        with pytest.raises(PathError, match="Access denied"):
            resolve(str(sibling / "x.txt"), [project])

    def test_the_message_names_the_path_as_written_and_what_to_do(self, project: Path):
        with pytest.raises(PathError) as raised:
            resolve("../outside/secret.txt", [project])

        assert "../outside/secret.txt" in str(raised.value)
        assert "ask the user" in str(raised.value)

    def test_a_directory_above_the_root_is_refused(self, project: Path):
        with pytest.raises(PathError, match="Access denied"):
            resolve(str(project.parent), [project])

    def test_a_missing_file_outside_is_refused_too(self, project: Path):
        with pytest.raises(PathError, match="Access denied"):
            resolve(str(project.parent / "outside" / "new.txt"), [project])


class TestSymlinks:
    def test_a_link_to_a_file_outside_is_refused(self, project: Path):
        (project / "link").symlink_to(project.parent / "outside" / "secret.txt")

        with pytest.raises(PathError, match="Access denied"):
            resolve(str(project / "link"), [project])

    def test_a_link_to_a_directory_outside_is_refused_for_what_is_below_it(
        self, project: Path
    ):
        (project / "dir").symlink_to(project.parent / "outside")

        with pytest.raises(PathError, match="Access denied"):
            resolve(str(project / "dir" / "secret.txt"), [project])

    def test_a_new_file_below_a_link_to_outside_is_refused(self, project: Path):
        (project / "dir").symlink_to(project.parent / "outside")

        with pytest.raises(PathError, match="Access denied"):
            resolve(str(project / "dir" / "new.txt"), [project])

    def test_a_link_that_stays_inside_comes_back_as_its_target(self, project: Path):
        (project / "link").symlink_to(project / "a.txt")

        assert resolve(str(project / "link"), [project]) == project / "a.txt"

    def test_a_link_whose_target_is_missing_is_named(self, project: Path):
        (project / "dangling").symlink_to(project / "nowhere.txt")

        with pytest.raises(PathError, match="symlink whose target does not exist"):
            resolve(str(project / "dangling"), [project])

    def test_a_missing_path_below_a_dangling_link_is_named_too(self, project: Path):
        (project / "dangling").symlink_to(project / "nowhere")

        with pytest.raises(PathError, match="symlink whose target does not exist"):
            resolve(str(project / "dangling" / "x.txt"), [project])

    def test_a_loop_is_an_error_not_a_hang(self, project: Path):
        (project / "a").symlink_to(project / "b")
        (project / "b").symlink_to(project / "a")

        with pytest.raises(PathError):
            resolve(str(project / "a"), [project])

    def test_dot_dot_after_a_link_is_collapsed_before_anything_is_looked_up(
        self, project: Path
    ):
        # `link/..` would be the parent of the link's target in a shell. Here the path
        # is collapsed first, so what is checked is what is returned.
        (project / "link").symlink_to(project.parent / "outside")

        assert resolve(str(project / "link" / ".." / "a.txt"), [project]) == (
            project / "a.txt"
        )


class TestTheDefaults:
    def test_the_roots_are_the_working_directory_and_the_temp_directory(
        self, tmp_path: Path
    ):
        assert default_roots() == (
            tmp_path.resolve(),
            Path(tempfile.gettempdir()).resolve(),
        )

    def test_the_temp_directory_is_allowed(self):
        wanted = Path(tempfile.gettempdir()).resolve() / "scratch" / "script.py"

        assert resolve(str(wanted)) == wanted

    def test_the_working_directory_is_allowed(self, tmp_path: Path):
        assert resolve("src/new.py") == tmp_path.resolve() / "src" / "new.py"

    def test_somewhere_else_is_refused(self):
        with pytest.raises(PathError, match="Access denied"):
            resolve("/etc/hosts")

    def test_the_home_directory_is_refused_unless_it_is_a_root(self):
        with pytest.raises(PathError, match="Access denied"):
            resolve("~/.ssh/id_ed25519")


class TestBadPaths:
    @pytest.mark.parametrize("path", ["", "   "])
    def test_an_empty_path_is_refused(self, path: str):
        with pytest.raises(PathError, match="empty"):
            resolve(path)

    def test_a_nul_byte_is_refused(self):
        with pytest.raises(PathError, match="Not a valid path"):
            resolve("a\0b")


class TestAbsolutePath:
    def test_a_relative_path_starts_at_the_working_directory(self, tmp_path: Path):
        assert absolute_path("a/b.txt") == tmp_path / "a" / "b.txt"

    def test_dot_dot_is_collapsed_without_looking_at_the_disk(self, tmp_path: Path):
        assert absolute_path("a/../b/./c.txt") == tmp_path / "b" / "c.txt"

    def test_tilde_is_the_home_directory(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))

        assert absolute_path("~/x") == tmp_path / "home" / "x"

    def test_a_symbolic_link_is_not_followed(self, tmp_path: Path):
        (tmp_path / "real.txt").write_text("")
        (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")

        found = absolute_path("link.txt")

        assert found == tmp_path / "link.txt" and found.is_symlink()

    @pytest.mark.parametrize(("path", "why"), [("", "empty"), ("a\0b", "valid path")])
    def test_a_path_that_is_not_one_is_refused(self, path: str, why: str):
        with pytest.raises(PathError, match=why):
            absolute_path(path)
