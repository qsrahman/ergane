import json
import os
import stat
from pathlib import Path

import pytest

from ergane import sessions
from ergane.sessions import SavedSession, SessionError, SessionInfo, SessionStore, ago
from helpers import make_context


@pytest.fixture
def project(tmp_path: Path) -> Path:
    directory = tmp_path / "project"
    directory.mkdir()
    return directory


@pytest.fixture
def store(tmp_path: Path, project: Path) -> SessionStore:
    return SessionStore(tmp_path / "state" / "sessions", project)


def items(turns: int = 2) -> list[dict]:
    return make_context(turns)  # pyright: ignore[reportReturnType]


def save(store: SessionStore, prompt: str = "hello", previous=None) -> SessionInfo:
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "hi"},
    ]
    return store.save(previous, None, conversation, "gpt-4o")


class TestSave:
    def test_a_saved_session_is_listed_and_read_as_it_was(self, store):
        info = store.save(None, "NOTES", items(2), "gpt-4o")

        [listed] = store.sessions()
        saved = store.load(info.id)

        assert listed == info and info.turns == 2 and info.model == "gpt-4o"
        assert saved == SavedSession(info, "NOTES", items(2))

    def test_saving_again_keeps_the_id_the_title_and_the_time_of_creation(
        self, store, clock
    ):
        first = save(store, "the first prompt")

        again = store.save(first, None, items(3), "gpt-4o-mini")

        assert [s.id for s in store.sessions()] == [first.id]
        assert again.id == first.id and again.title == "the first prompt"
        assert again.created == first.created and again.updated > first.updated
        assert again.model == "gpt-4o-mini" and again.turns == 3

    def test_the_title_is_the_first_prompt_on_one_line_cut_short(self, store):
        long = "line one\n\n  line   two " + "x" * 100

        info = save(store, long)

        assert info.title.startswith("line one line two xxx")
        assert len(info.title) == 60 and info.title.endswith("…")

    def test_a_conversation_with_no_prompt_has_a_title_all_the_same(self, store):
        assert store.save(None, "NOTES", [], "gpt-4o").title == "(no prompt)"

    def test_the_file_and_the_directories_are_the_users_only(self, store):
        info = save(store)

        file = store.directory / f"{info.id}.jsonl"
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(store.directory.parent.stat().st_mode) == 0o700

    def test_no_temporary_file_is_left(self, store):
        save(store)

        assert [p.suffix for p in store.directory.iterdir()] == [".jsonl"]

    def test_each_save_has_a_temporary_file_of_its_own(self, store, monkeypatch):
        # Two erganes that resume one session save the same path; a fixed temporary
        # name would let one write the file the other is about to move into place.
        first = save(store)
        sources: list[Path] = []
        real_replace = sessions.os.replace

        def replace(source, target):
            sources.append(Path(source))
            real_replace(source, target)

        monkeypatch.setattr(sessions.os, "replace", replace)
        save(store, previous=first)
        save(store, previous=first)

        assert len(sources) == 2 and len(set(sources)) == 2
        assert all(source.parent == store.directory for source in sources)

    def test_a_save_that_fails_leaves_no_temporary_file(self, store, monkeypatch):
        first = save(store)
        path = store.directory / f"{first.id}.jsonl"
        before = path.read_bytes()

        def replace(source, target):
            raise OSError("disk full")

        monkeypatch.setattr(sessions.os, "replace", replace)
        with pytest.raises(OSError, match="disk full"):
            save(store, previous=first)

        assert list(store.directory.iterdir()) == [path]
        assert path.read_bytes() == before

    def test_old_temporary_files_are_removed_and_new_ones_are_left(self, store):
        first = save(store)
        # The names of a save that was killed, now and in earlier versions.
        old = [store.directory / name for name in (".a.x1.tmp", "b.tmp")]
        new = store.directory / ".c.x2.tmp"  # perhaps another ergane's, in progress
        for path in (*old, new):
            path.write_text("half")
        for path in old:
            os.utime(path, (0, 0))

        save(store, previous=first)

        assert not any(path.exists() for path in old)
        assert new.exists()

    def test_a_file_has_two_lines_the_first_of_which_lists_it(self, store):
        info = save(store)

        first, second, rest = (
            (store.directory / f"{info.id}.jsonl").read_text().split("\n")
        )

        assert json.loads(first)["id"] == info.id and rest == ""
        assert set(json.loads(second)) == {"summary", "items"}

    def test_a_save_that_fails_leaves_the_session_as_it_was(self, store, monkeypatch):
        first = save(store, "kept")
        before = (store.directory / f"{first.id}.jsonl").read_text()

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", fail)
        with pytest.raises(OSError):
            store.save(first, None, items(3), "gpt-4o")

        assert (store.directory / f"{first.id}.jsonl").read_text() == before

    def test_only_the_newest_sessions_are_kept(self, store, clock, monkeypatch):
        monkeypatch.setattr(sessions, "KEEP", 3)

        made = [save(store, f"prompt {n}") for n in range(5)]

        assert [s.title for s in store.sessions()] == [
            "prompt 4",
            "prompt 3",
            "prompt 2",
        ]
        assert made[0].id not in [s.id for s in store.sessions()]

    def test_a_session_that_is_saved_again_is_not_the_one_that_is_pruned(
        self, store, clock, monkeypatch
    ):
        monkeypatch.setattr(sessions, "KEEP", 2)
        oldest = save(store, "oldest")
        save(store, "middle")
        save(store, "newest")

        again = store.save(oldest, None, items(1), "gpt-4o")

        assert again.id in [s.id for s in store.sessions()]


class TestFind:
    def test_the_most_recently_saved_comes_first(self, store, clock):
        first = save(store, "one")
        second = save(store, "two")
        third = save(store, "three")
        store.save(first, None, items(2), "gpt-4o")  # saved again: now the newest

        assert [s.id for s in store.sessions()] == [first.id, third.id, second.id]

    def test_latest_is_the_first_of_them(self, store, clock):
        save(store, "one")
        newest = save(store, "two")

        latest = store.latest()

        assert latest is not None and latest.info.id == newest.id

    def test_latest_is_none_when_nothing_is_saved(self, store):
        assert store.latest() is None and store.sessions() == []

    def test_a_session_is_found_by_its_id_or_by_the_start_of_it(self, store, clock):
        info = save(store)

        assert store.load(info.id).info == info
        assert store.load(info.id[:12]).info == info

    def test_an_id_that_names_nothing_is_said_so(self, store):
        save(store)

        with pytest.raises(SessionError, match="No saved session 'nope'"):
            store.load("nope")

    def test_the_start_of_two_ids_is_ambiguous_and_names_them(self, store, clock):
        first = save(store, "one")
        second = save(store, "two")

        with pytest.raises(SessionError, match="starts 2 sessions") as raised:
            store.load("20")  # the year is the start of every id

        assert first.id in str(raised.value) and second.id in str(raised.value)

    def test_a_whole_id_wins_over_the_start_of_another(self, store, clock, monkeypatch):
        short = save(store, "one")
        monkeypatch.setattr(store, "_new_id", lambda: short.id + "-longer")
        save(store, "two")

        assert store.load(short.id).info.id == short.id


class TestDirectories:
    def test_each_directory_has_its_own_sessions(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        root = tmp_path / "sessions"

        save(SessionStore(root, a), "in a")

        assert SessionStore(root, b).sessions() == []
        assert [s.title for s in SessionStore(root, a).sessions()] == ["in a"]

    def test_the_name_of_the_directory_is_readable_and_has_a_hash(self, store, project):
        assert store.directory.name.startswith("-") is False
        assert store.directory.name.endswith(
            f"{project.name}-" + store.directory.name[-8:]
        )

    def test_two_paths_that_read_alike_are_still_two_directories(self, tmp_path):
        # "x/a-b" and "x/a/b" both read as x-a-b: only the hash tells them apart.
        one = tmp_path / "x" / "a-b"
        two = tmp_path / "x" / "a" / "b"
        one.mkdir(parents=True)
        two.mkdir(parents=True)
        root = tmp_path / "sessions"

        save(SessionStore(root, one), "in one")

        assert SessionStore(root, two).sessions() == []
        assert SessionStore(root, one).directory != SessionStore(root, two).directory

    def test_a_link_to_a_directory_shares_its_sessions(self, tmp_path, project):
        link = tmp_path / "link"
        link.symlink_to(project)
        root = tmp_path / "sessions"

        save(SessionStore(root, project), "through the directory")

        assert [s.title for s in SessionStore(root, link).sessions()] == [
            "through the directory"
        ]

    def test_the_root_has_a_name_too(self, tmp_path):
        assert len(SessionStore(tmp_path, Path("/")).directory.name) == 8

    def test_sessions_in_the_working_directory_are_never_read(self, store, project):
        # A project that ships a session would speak in the model's own voice.
        info = save(store, "mine")
        planted = project / ".ergane" / "sessions"
        planted.mkdir(parents=True)
        (planted / f"{info.id}.jsonl").write_text(
            (store.directory / f"{info.id}.jsonl").read_text().replace("mine", "theirs")
        )

        assert [s.title for s in store.sessions()] == ["mine"]


class TestFilesThatAreNotSessions:
    def put(self, store: SessionStore, name: str, text: str) -> Path:
        store.directory.mkdir(parents=True, exist_ok=True)
        path = store.directory / name
        path.write_text(text)
        return path

    def header(self, id: str, **changes) -> str:
        info = {
            "version": 1,
            "id": id,
            "cwd": "/x",
            "title": "t",
            "created": "2026-11-19T10:00:00+00:00",
            "updated": "2026-11-19T10:00:00+00:00",
            "model": "m",
            "turns": 1,
        }
        return json.dumps({**info, **changes})

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "not json\n{}\n",
            '{"version": 2}\n{"summary": null, "items": []}\n',
        ],
        ids=["empty", "not-json", "another-version"],
    )
    def test_they_are_left_out_of_the_list_and_left_alone(self, store, text):
        path = self.put(store, "20261119-1000-aaaa.jsonl", text)
        good = save(store)

        assert [s.id for s in store.sessions()] == [good.id]
        assert path.exists()

    def test_a_file_that_says_another_id_than_its_name_is_left_out(self, store):
        # The name is what a file is removed by: it must not be made to point elsewhere.
        self.put(
            store, "20261119-1000-aaaa.jsonl", self.header("../../elsewhere") + "\n"
        )

        assert store.sessions() == []

    def test_loading_one_that_cannot_be_read_says_which_and_why(self, store):
        id = "20261119-1000-aaaa"
        self.put(store, f"{id}.jsonl", self.header(id) + "\n")  # no second line

        with pytest.raises(SessionError, match=f"The session {id} cannot be read"):
            store.load(id)

    @pytest.mark.parametrize(
        "body",
        ['{"summary": 3, "items": []}', '{"summary": null, "items": [1]}', "[]"],
        ids=["summary", "item", "shape"],
    )
    def test_a_conversation_of_the_wrong_shape_is_refused(self, store, body):
        id = "20261119-1000-aaaa"
        self.put(store, f"{id}.jsonl", self.header(id) + "\n" + body + "\n")

        with pytest.raises(SessionError, match="cannot be read"):
            store.load(id)

    def test_pruning_leaves_files_it_cannot_read_alone(self, store, clock, monkeypatch):
        monkeypatch.setattr(sessions, "KEEP", 1)
        stray = self.put(store, "20200101-0000-aaaa.jsonl", "garbage")

        save(store, "one")
        save(store, "two")

        assert stray.exists() and len(store.sessions()) == 1


class TestAgo:
    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            (5, "just now"),
            (120, "2 min ago"),
            (3 * 3600, "3 h ago"),
            (86400, "1 day ago"),
            (5 * 86400, "5 days ago"),
        ],
    )
    def test_it_says_how_long_ago_in_words(self, store, monkeypatch, seconds, text):
        from datetime import UTC, datetime, timedelta

        info = save(store)
        later = datetime.fromisoformat(info.updated) + timedelta(seconds=seconds)

        class Later(datetime):
            @classmethod
            def now(cls, tz=None):
                return later.astimezone(tz or UTC)

        monkeypatch.setattr(sessions, "datetime", Later)

        assert ago(info) == text
