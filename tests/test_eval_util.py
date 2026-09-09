"""Tests for shared evals helpers."""

import json
import os
import stat

import pytest

from evals import atomic_write_jsonl, plural


class _Rec:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


class TestAtomicWriteJsonl:
    def test_writes_one_json_line_per_record(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1), _Rec(2)], path)
        lines = path.read_text().strip().splitlines()
        assert [json.loads(line) for line in lines] == [{"value": 1}, {"value": 2}]

    def test_accepts_str_path(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1)], str(path))  # str coerced to Path
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1)], path)
        atomic_write_jsonl([_Rec(2)], path)
        assert path.read_text().strip() == json.dumps({"value": 2})

    def test_leaves_no_temp_file_on_success(self, tmp_path):
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1)], path)
        assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]

    def test_write_failure_cleans_temp_and_leaves_original(self, tmp_path):
        # A mid-write failure must unlink the temp file and leave the original
        # file untouched (the rename never happens).
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec("original")], path)

        class _Bad:
            def to_dict(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            atomic_write_jsonl([_Rec("new"), _Bad()], path)

        assert json.loads(path.read_text().strip()) == {"value": "original"}
        assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]  # no temp left

    @pytest.mark.parametrize("mode", [0o640, 0o600, 0o750])
    def test_preserves_existing_file_mode(self, tmp_path, mode):
        # mkstemp creates the temp file 0600; the rename must not silently
        # narrow the permissions the owner had set on the real file (issue: the
        # golden set became root-only after every review session, and a second
        # user had to chmod o+r it each time).
        #
        # The seeded mode must be one the new-file fallback (0666 & ~umask)
        # cannot produce on this host, or the test passes by coincidence: the
        # original 0o664 equalled the fallback under umask 002 and stayed green
        # with the existing-mode branch bypassed. 0o640 is what umask 027 would
        # yield, so 0o750 is the one no umask at all can produce — the fallback
        # can never set an execute bit. 0o600 is mkstemp's own default, so it
        # pins the branch but would not notice a dropped chmod; the other two do.
        path = tmp_path / "out.jsonl"
        atomic_write_jsonl([_Rec(1)], path)
        path.chmod(mode)
        atomic_write_jsonl([_Rec(2)], path)
        assert stat.S_IMODE(path.stat().st_mode) == mode

    def test_new_file_gets_umask_default_mode_not_0600(self, tmp_path):
        # A file that did not exist before should come out the way a plain
        # open(path, "w") would leave it, i.e. 0666 minus the umask — not the
        # 0600 that mkstemp defaults to.
        path = tmp_path / "fresh.jsonl"
        old = os.umask(0o022)
        try:
            atomic_write_jsonl([_Rec(1)], path)
        finally:
            os.umask(old)
        assert stat.S_IMODE(path.stat().st_mode) == 0o644


class TestPlural:
    def test_singular(self):
        assert plural(1, "story", "stories") == "1 story"

    def test_plural(self):
        assert plural(2, "story", "stories") == "2 stories"

    def test_zero_is_plural(self):
        assert plural(0, "error", "errors") == "0 errors"

    def test_default_plural_appends_s(self):
        assert plural(3, "row") == "3 rows"
        assert plural(1, "row") == "1 row"
