"""R4 C12 controls for the S2 verifiers (agent-authored, D-024).

No test here runs the suite, Ruff or git: every subprocess goes through a fake runner, and the
snapshots are tiny synthetic trees. Known answers are hand-derived from ``git diff --check``'s
default rules (blank-at-eol, space-before-tab, blank-at-eof); the A1-R1 literals are git's
recorded output for those inputs.
"""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import verify_s2
from scripts import verify_s2_ingestion as ingestion

BASELINE = ingestion.BASELINE


def sha(data):
    return hashlib.sha256(data).hexdigest()


def baseline_snapshot():
    protected = {"gate/verdict.py": b"VERDICT\n", "gate/stats/bound.py": b"BOUND\n"}
    manifest = {"source_sha256": {name: sha(data) for name, data in protected.items()}}
    return {**protected, BASELINE: json.dumps(manifest).encode(), "gate/other.py": b"x = 1\n"}


@pytest.mark.parametrize("module", [verify_s2, ingestion])
def test_protected_check_known_answers(module):
    snapshot = baseline_snapshot()
    assert module.protected_sources_unchanged(snapshot) is True
    added = {**snapshot, "gate/stats/new_bound.py": b"NEW\n"}
    assert module.protected_sources_unchanged(added) is False  # R4 C12: added file detected
    changed = {**snapshot, "gate/stats/bound.py": b"BOUND2\n"}
    assert module.protected_sources_unchanged(changed) is False
    removed = {k: v for k, v in snapshot.items() if k != "gate/verdict.py"}
    assert module.protected_sources_unchanged(removed) is False
    unrelated = {**snapshot, "gate/other.py": b"x = 2\n"}
    assert module.protected_sources_unchanged(unrelated) is True


@pytest.mark.parametrize(
    "data,expected",
    [
        (b"ok\n", []),
        (b"ok", []),
        (b"", []),  # an empty file has no lines, so no blank line at EOF
        (b"\tindented\n", []),
        (b"x \n", [("trailing_whitespace", 1)]),
        (b"a\nb\t\n", [("trailing_whitespace", 2)]),
        (b"a\r\n", [("trailing_whitespace", 1)]),
        (b"a\nlast ", [("trailing_whitespace", 2)]),
        (b" \tx\n", [("space_before_tab", 1)]),
        (b"\t x\n", []),
        (b"a\n\n", [("blank_at_eof", 2)]),
        (b"a\n\n\n", [("blank_at_eof", 2)]),
        # A1-R1: whitespace-only final lines are blank lines for blank-at-eof, as in git
        # (repro outputs of ``git diff --no-index --check`` recorded by the review).
        (b"   \n", [("trailing_whitespace", 1), ("blank_at_eof", 1)]),
        (b"a\n\t\n", [("trailing_whitespace", 2), ("blank_at_eof", 2)]),
        (b"a\n \n\n", [("trailing_whitespace", 2), ("blank_at_eof", 2)]),
        # A1R: git's count_trailing_blank never examines a leading line whose last byte before
        # its LF is byte 0, so b"\n", b"\n\n" and b" \n" report no blank_at_eof; the rows after
        # them are git's other all-blank/edge cases. Each answer is the recorded output of
        # ``git diff --cached --check`` on the literal bytes as a new file.
        (b"\n", []),
        (b"\n\n", []),
        (b" \n", [("trailing_whitespace", 1)]),
        (b"\n \n", [("trailing_whitespace", 2), ("blank_at_eof", 2)]),
        (b"\n\n\n", [("blank_at_eof", 3)]),
        (b"\n \n\n", [("trailing_whitespace", 2), ("blank_at_eof", 2)]),
        (b"\n\n \n", [("trailing_whitespace", 3), ("blank_at_eof", 3)]),
        (b"\n\n\n\n", [("blank_at_eof", 3)]),
        (b"  ", [("trailing_whitespace", 1), ("blank_at_eof", 1)]),
        (b"  \n", [("trailing_whitespace", 1), ("blank_at_eof", 1)]),
        (b"x\n \n", [("trailing_whitespace", 2), ("blank_at_eof", 2)]),
        (b"\n\nx", []),
        (b"\nx\n\n", [("blank_at_eof", 3)]),
    ],
)
def test_whitespace_known_answers(data, expected):
    issues = ingestion.whitespace_issues({"f.py": data})
    assert [(i["code"], i["line"]) for i in issues] == expected
    assert all(i["path"] == "f.py" for i in issues)


def test_init_file_of_one_blank_line_is_clean():
    """A1R repro: a package __init__.py holding only a newline formerly reported blank_at_eof."""
    snapshot = {"teacher_rl/__init__.py": b"\n", "gate/__init__.py": b"\n\n", "a.py": b"ok\n"}
    assert ingestion.whitespace_issues(snapshot) == []


def test_binary_files_are_skipped():
    assert ingestion.whitespace_issues({"a.parquet": b"PAR1 \n\0 \n", "b.bin": b"\xff \n"}) == []


def provenance_snapshot(vendored_bytes, declared_sha):
    provenance = {
        "license_file": "LICENSE",
        "sources": [{"file": "proto/x.proto", "sha256": declared_sha}],
    }
    return {
        "tests/fixtures/up/provenance.json": json.dumps(provenance).encode(),
        "tests/fixtures/up/proto/x.proto": vendored_bytes,
        "tests/fixtures/up/LICENSE": b"License text \n",
        "tests/fixtures/other/x.proto": vendored_bytes,
    }


def test_hash_verified_vendored_files_are_exempt_but_copies_are_not():
    vendored = b"message X { \n}\n"
    snapshot = provenance_snapshot(vendored, sha(vendored))
    # A1-R4: a bare-string license_file carries no hash, so it exempts nothing.
    assert ingestion.vendored_paths(snapshot) == {"tests/fixtures/up/proto/x.proto"}
    issues = ingestion.whitespace_issues(snapshot)
    assert [(i["path"], i["line"]) for i in issues] == [
        ("tests/fixtures/other/x.proto", 1),
        ("tests/fixtures/up/LICENSE", 1),
    ]


def test_a1_r4_license_file_is_exempt_only_with_a_matching_sha256():
    license_bytes = b"License text \n"
    for declared, exempt in ((sha(license_bytes), True), ("0" * 64, False), (None, False)):
        record = {"file": "LICENSE", **({} if declared is None else {"sha256": declared})}
        snapshot = {
            "tests/fixtures/up/provenance.json": json.dumps({"license_file": record}).encode(),
            "tests/fixtures/up/LICENSE": license_bytes,
        }
        assert (ingestion.vendored_paths(snapshot) == {"tests/fixtures/up/LICENSE"}) is exempt


@pytest.mark.parametrize(
    "provenance_dir,file",
    [
        ("gate", "x.py"),  # provenance outside tests/fixtures/
        ("", "gate/x.py"),  # top-level provenance
        ("tests/fixtures/up", "../../../gate/x.py"),  # escapes tests/fixtures/
        ("tests/fixtures", "../gate/x.py"),
    ],
)
def test_a1_r4_provenance_cannot_exempt_files_outside_test_fixtures(provenance_dir, file):
    """Review A1-R4: a hash-matched provenance.json anywhere formerly hid source whitespace."""
    source = b"x = 1 \n"
    base = provenance_dir + "/" if provenance_dir else ""
    snapshot = {
        base + "provenance.json": json.dumps(
            {"sources": [{"file": file, "sha256": sha(source)}]}
        ).encode(),
        "gate/x.py": source,
        "tests/fixtures/gate/x.py": source,
    }
    assert ingestion.vendored_paths(snapshot) == set()
    paths = [i["path"] for i in ingestion.whitespace_issues(snapshot)]
    assert "gate/x.py" in paths


def test_vendored_file_with_wrong_hash_is_checked():
    vendored = b"message X { \n}\n"
    snapshot = provenance_snapshot(vendored, "0" * 64)
    paths = {i["path"] for i in ingestion.whitespace_issues(snapshot)}
    assert "tests/fixtures/up/proto/x.proto" in paths


def test_unreadable_provenance_exempts_nothing():
    snapshot = {"d/provenance.json": b"{not json", "d/x.proto": b"x \n"}
    assert ingestion.vendored_paths(snapshot) == set()
    assert len(ingestion.whitespace_issues(snapshot)) == 1


def make_tree(root):
    """Minimal S2 source tree; the trailing-space file stands for an untracked or staged one."""
    snapshot = baseline_snapshot()
    files = {
        **snapshot,
        "pyproject.toml": b"[project]\n",
        "uv.lock": b"version = 1\n",
        ".claude/starred-cores.txt": b"gate/verdict.py\n",
        "docs/02_PHYSLOOP_PLAN.md": b"plan\n",
        "docs/research/2026-11-07-pilot-launch-audit.md": b"audit\n",
        "docs/S2_INGESTION.md": b"doc\n",
        **{
            f"docs/{name}": b"doc\n"
            for name in (
                "DECISIONS.md",
                "00_START_HERE.md",
                "S2_ENGINEERING_SPEC.md",
                "S2_IMPLEMENTATION_PLAN.md",
                "RESEARCH_OPERATING_MODEL.md",
            )
        },
        "tests/test_new_untracked.py": b"x = 1 \n",
        "experiments/S2_PILOT_DRAFT.json": b'{"schema_version": 2}\n',
        "experiments/S2_FULL_DRAFT.json": b'{"schema_version": 2}\n',
        "scripts/tool.py": b"pass\n",
        "gate/__pycache__/ignored.pyc": b"\0",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


@pytest.mark.parametrize("which", ["verify_s2", "ingestion"])
def test_snapshot_covers_files_git_diff_check_would_miss(tmp_path, monkeypatch, which):
    make_tree(tmp_path)
    if which == "verify_s2":
        snapshot = verify_s2.source_snapshot(tmp_path)
    else:
        monkeypatch.setattr(ingestion, "ROOT", tmp_path)
        snapshot = ingestion.source_snapshot()
    assert "tests/test_new_untracked.py" in snapshot
    assert "gate/__pycache__/ignored.pyc" not in snapshot
    issues = ingestion.whitespace_issues(snapshot)
    assert [(i["path"], i["code"]) for i in issues] == [
        ("tests/test_new_untracked.py", "trailing_whitespace")
    ]


def test_snapshot_refuses_symlinks(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "gate/link.py").symlink_to(tmp_path / "gate/other.py")
    with pytest.raises(ValueError, match="symlink"):
        verify_s2.source_snapshot(tmp_path)


def test_drafts_are_parsed_strictly_from_the_snapshot(tmp_path):
    make_tree(tmp_path)
    snapshot = verify_s2.source_snapshot(tmp_path)
    reports = verify_s2.draft_reports(snapshot)
    assert set(reports) == {"S2_PILOT_DRAFT", "S2_FULL_DRAFT"}
    assert all(not r["ready_for_planning"] for r in reports.values())
    snapshot["experiments/S2_PILOT_DRAFT.json"] = b'{"schema_version": 2, "schema_version": 2}'
    with pytest.raises(ValueError, match="Duplicate"):
        verify_s2.draft_reports(snapshot)


class FakeRunner:
    def __init__(self, returncode=0):
        self.calls, self.returncode = [], returncode

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return SimpleNamespace(returncode=self.returncode, stdout="out\n", stderr="")


def test_run_checks_uses_no_cache_ruff_no_git_and_records_snapshot_whitespace(tmp_path):
    runner = FakeRunner()
    snapshot = {"a.py": b"ok\n", "b.py": b"bad \n"}
    checks = ingestion.run_checks(tmp_path, snapshot, runner=runner)
    commands = [command for command, _ in runner.calls]
    assert [c["name"] for c in checks] == ["tests", "lint", "snapshot_whitespace"]
    lint = next(c for c in commands if "ruff" in c)
    assert "--no-cache" in lint and lint[-4:] == ["gate", "teacher_rl", "tests", "scripts"]
    assert all("git" not in c for c in commands)  # git diff --check is no longer used
    assert all(kwargs["check"] is False for _, kwargs in runner.calls)
    whitespace = checks[-1]
    assert whitespace["returncode"] == 1 and whitespace["issue_count"] == 1
    assert (tmp_path / "snapshot_whitespace.log").read_text() == "b.py:1: trailing_whitespace\n"
    assert verify_s2.run_checks is ingestion.run_checks


def test_run_checks_clean_snapshot_and_failed_runner(tmp_path):
    def broken(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 900)

    checks = ingestion.run_checks(tmp_path, {"a.py": b"ok\n"}, runner=broken)
    assert [c["returncode"] for c in checks] == [-1, -1, 0]
    assert "no whitespace issues in 1 snapshot files" in (
        tmp_path / "snapshot_whitespace.log"
    ).read_text()


def test_a1_r3_run_checks_forwards_only_an_allowlisted_environment(tmp_path, monkeypatch):
    """Review A1-R3: the whole caller environment, secrets included, reached every subprocess."""
    monkeypatch.setenv("PHYSLOOP_CANARY_SECRET", "canary-value")
    monkeypatch.setenv("VASTAI_API_KEY", "canary-key")
    monkeypatch.setenv("PYTHONPATH", "/canary/path")
    monkeypatch.setenv("PATH", "/canary/bin")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    runner = FakeRunner()
    ingestion.run_checks(tmp_path, {"a.py": b"ok\n"}, runner=runner)
    assert len(runner.calls) == 2
    for _, kwargs in runner.calls:
        env = kwargs["env"]
        assert set(env) <= set(ingestion.CHECK_ENVIRONMENT_KEYS) | {"PYTHONDONTWRITEBYTECODE"}
        assert "canary-value" not in env.values() and "canary-key" not in env.values()
        assert "PYTHONPATH" not in env
        assert env["PATH"] == "/canary/bin"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
