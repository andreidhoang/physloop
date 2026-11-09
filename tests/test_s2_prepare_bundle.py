"""Agent-authored known answers for the sealed source bundle builder (F3; CPU-09).

Synthetic trees in ``tmp_path`` only; the guest's ``tar -x`` is mimicked with Python's
``tarfile`` (no process is run). The one real-repository test reads files and writes only into
``tmp_path``. No network, provider or credential is touched; the secret is a synthetic canary.
"""

import ast
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from gate.s2_bootstrap import evaluate_rule
from gate.s2_identity import identity_hash, read_json_bytes
from gate.s2_runtime import SOURCE_MANIFEST_NAME, validate_source_manifest
from scripts import s2_prepare_bundle as bundle

REPO = Path(__file__).resolve().parents[1]
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ABC_SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
TREE = {
    "pyproject.toml": b"[project]\nname = 'x'\n",
    "uv.lock": b"version = 1\n",
    ".python-version": b"3.12\n",
    "gate/__init__.py": b"",
    "gate/a.py": b"abc",
    "gate/sub/b.json": b"{}\n",
    "scripts/run.sh": b"#!/bin/sh\nexit 0\n",
    "experiments/S2_SETUP_DRAFT.json": b"{\"status\": \"draft\"}\n",
    # Never bundled: outside the allowlist or ignored by name.
    "experiments/S2_PILOT_DRAFT.json": b"{}\n",
    "tests/test_x.py": b"assert True\n",
    "teacher_rl/x.py": b"x = 1\n",
    "evidence/e.json": b"{}\n",
    "docs/d.md": b"# d\n",
    "gate/__pycache__/a.cpython-312.pyc": b"\x00pyc",
    "gate/.DS_Store": b"\x00",
    "scripts/x.swp": b"swap",
    "gate/model.safetensors": b"weights",
    "gate/stale.pyc": b"\x00",
}
BUNDLED = [".python-version", "experiments/S2_SETUP_DRAFT.json", "gate/__init__.py",
           "gate/a.py", "gate/sub/b.json", "pyproject.toml", "scripts/run.sh", "uv.lock"]
EXCLUDED = ["gate/.DS_Store", "gate/__pycache__/", "gate/model.safetensors", "gate/stale.pyc",
            "scripts/x.swp"]


def make_tree(root, order=None, mtime=None):
    root.mkdir(parents=True, exist_ok=True)
    for name in order or sorted(TREE):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(TREE[name])
        if name.endswith(".sh"):
            path.chmod(0o755)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
    return root


def build(tmp_path, root=None, out="out", scope="synthetic"):
    root = root or make_tree(tmp_path / "src")
    out_dir = tmp_path / out
    out_dir.mkdir(exist_ok=True)
    return bundle.build_bundle(str(root), str(out_dir), scope=scope), out_dir


def codes(report):
    return sorted({i["code"] for i in report["issues"]})


def archive_bytes(report, out_dir):
    return (out_dir / report["archive"]["name"]).read_bytes()


def test_manifest_known_answer_and_exclusions(tmp_path):
    report, out_dir = build(tmp_path)
    assert report["status"] == "passed", report["issues"]
    manifest = read_json_bytes((out_dir / report["archive"]["name"].replace(
        ".tar", ".manifest.json")).read_bytes())
    assert validate_source_manifest(manifest) == []
    assert [row["path"] for row in manifest["files"]] == BUNDLED
    rows = {row["path"]: row for row in manifest["files"]}
    assert rows["gate/__init__.py"] == {"path": "gate/__init__.py", "size_bytes": 0,
                                        "sha256": EMPTY_SHA}
    assert rows["gate/a.py"] == {"path": "gate/a.py", "size_bytes": 3, "sha256": ABC_SHA}
    assert report["excluded"] == EXCLUDED
    assert report["bundle_sha256"] == identity_hash(manifest)
    assert report["archive"]["name"] == f"physloop-source-{report['bundle_sha256']}.tar"
    data = archive_bytes(report, out_dir)
    assert report["archive"]["size_bytes"] == len(data)
    assert report["archive"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert report["files"] == 8 and report["network"] == "none"
    assert report["scope"] == "synthetic" and report["artifact_kind"] == "s2_source_bundle_report"
    for name in ("tests/", "teacher_rl/", "evidence/", "docs/", "experiments/S2_PILOT"):
        assert not any(row["path"].startswith(name) for row in manifest["files"])


def test_archive_members_are_deterministic_ustar(tmp_path):
    report, out_dir = build(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes(report, out_dir)), mode="r:") as archive:
        members = archive.getmembers()
        assert [m.name for m in members] == [SOURCE_MANIFEST_NAME, *BUNDLED]
        assert all(m.isreg() and m.mtime == 0 and m.uid == 0 and m.gid == 0
                   and m.uname == "" and m.gname == "" for m in members)
        modes = {m.name: m.mode for m in members}
        assert modes["scripts/run.sh"] == 0o755 and modes["gate/a.py"] == 0o644
        manifest = read_json_bytes(archive.extractfile(members[0]).read())
    assert identity_hash(manifest) == report["bundle_sha256"]


def test_two_builds_from_differently_ordered_copies_are_byte_identical(tmp_path):
    first, out1 = build(tmp_path, make_tree(tmp_path / "one", mtime=1_000_000), out="o1")
    second, out2 = build(tmp_path, make_tree(tmp_path / "two", order=sorted(TREE, reverse=True),
                                             mtime=2_000_000_000), out="o2")
    assert first["status"] == second["status"] == "passed"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert archive_bytes(first, out1) == archive_bytes(second, out2)
    third, _ = build(tmp_path, make_tree(tmp_path / "one"), out="o3")
    (tmp_path / "one" / "gate" / "a.py").write_bytes(b"abd")
    changed, _ = build(tmp_path, tmp_path / "one", out="o4")
    assert changed["bundle_sha256"] != third["bundle_sha256"]


def test_verify_archive_known_answer_and_negative_controls(tmp_path):
    report, out_dir = build(tmp_path)
    data = archive_bytes(report, out_dir)
    assert bundle.verify_archive(data, report["bundle_sha256"]) == []
    assert codes({"issues": bundle.verify_archive(data, "0" * 64)}) == ["bundle_sha256_mismatch"]
    assert codes({"issues": bundle.verify_archive(b"not a tar", report["bundle_sha256"])}) == [
        "unreadable_archive"]
    files = {name: (TREE[name], name.endswith(".sh")) for name in BUNDLED}
    manifest = bundle.build_manifest(files, scope="synthetic")
    sha = identity_hash(manifest)
    tampered = dict(files, **{"gate/a.py": (b"abd", False)})
    assert codes({"issues": bundle.verify_archive(bundle.build_archive(tampered, manifest),
                                                  sha)}) == ["member_mismatch"]
    extra = dict(files, **{"gate/z.py": (b"z", False)})
    assert codes({"issues": bundle.verify_archive(bundle.build_archive(extra, manifest),
                                                  sha)}) == ["extra_member"]
    fewer = {k: v for k, v in files.items() if k != "uv.lock"}
    assert codes({"issues": bundle.verify_archive(bundle.build_archive(fewer, manifest),
                                                  sha)}) == ["missing_member"]
    for name, kind in (("../evil", tarfile.REGTYPE), ("/abs", tarfile.REGTYPE),
                       ("gate/link", tarfile.SYMTYPE)):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.SYMTYPE:
                info.linkname = "/etc/passwd"
            archive.addfile(info, io.BytesIO(b"") if kind == tarfile.REGTYPE else None)
        assert codes({"issues": bundle.verify_archive(stream.getvalue(), sha)}) == [
            "unsafe_member"], name


def test_symlinked_file_or_directory_refuses_and_writes_nothing(tmp_path):
    root = make_tree(tmp_path / "src")
    (root / "gate" / "link.py").symlink_to(root / "gate" / "a.py")
    report, out_dir = build(tmp_path, root)
    assert report["status"] == "blocked" and codes(report) == ["symlink_rejected"]
    assert list(out_dir.iterdir()) == [] and report["archive"] is None
    root = make_tree(tmp_path / "src2")
    (root / "scripts" / "linked").symlink_to(root / "docs", target_is_directory=True)
    report, out_dir = build(tmp_path, root, out="out2")
    assert codes(report) == ["symlink_rejected"] and list(out_dir.iterdir()) == []
    root = make_tree(tmp_path / "src3")
    (root / "uv.lock").unlink()
    (root / "uv.lock").symlink_to(root / "pyproject.toml")
    assert codes(build(tmp_path, root, out="out3")[0]) == ["symlink_rejected"]


@pytest.mark.parametrize("entry, code", [
    ("/etc/passwd", "unsafe_path"), ("../outside.txt", "unsafe_path"),
    ("gate/../../x", "unsafe_path"), ("tests/conftest.py", "excluded_root"),
    ("docs/x.md", "excluded_root"), ("evidence/x.json", "excluded_root"),
])
def test_absolute_parent_and_never_included_entries_refuse(tmp_path, entry, code):
    root = make_tree(tmp_path / "src")
    _, _, issues = bundle.collect_sources(str(root), include_files=(entry,))
    assert [i["code"] for i in issues] == [code]
    _, _, issues = bundle.collect_sources(str(root), include_dirs=(entry,),
                                          include_files=(), experiment_files=())
    assert [i["code"] for i in issues] == [code]


def test_missing_include_and_unportable_names_refuse(tmp_path):
    root = make_tree(tmp_path / "src")
    (root / ".python-version").unlink()
    assert codes(build(tmp_path, root)[0]) == ["missing_input"]
    root = make_tree(tmp_path / "src2")
    (root / "gate" / "has space.py").write_bytes(b"x")
    assert codes(build(tmp_path, root, out="o2")[0]) == ["unsafe_path"]


def test_output_must_be_new_absolute_and_outside_the_root(tmp_path):
    root = make_tree(tmp_path / "src")
    first, out_dir = build(tmp_path, root)
    name = first["archive"]["name"]
    before = (out_dir / name).read_bytes()
    again, _ = build(tmp_path, root)
    assert codes(again) == ["output_exists"]
    assert (out_dir / name).read_bytes() == before
    inside = root / "dist"
    inside.mkdir()
    assert codes(bundle.build_bundle(str(root), str(inside), scope="synthetic")) == [
        "out_dir_inside_root"]
    assert codes(bundle.build_bundle(str(root), "relative/out", scope="synthetic")) == [
        "invalid_out_dir"]
    assert codes(bundle.build_bundle("relative", str(out_dir), scope="synthetic")) == [
        "invalid_root"]
    assert codes(bundle.build_bundle(str(root), str(out_dir), scope="production")) == [
        "invalid_scope"]


def test_secret_canary_refuses_naming_the_path_never_the_value(tmp_path):
    root = make_tree(tmp_path / "src")
    canary = "hf_" + "Syn7hetic" * 4
    (root / "gate" / "leak.py").write_text(f"TOKEN = '{canary}'\n")
    report, out_dir = build(tmp_path, root)
    assert codes(report) == ["secret_like_content"]
    assert report["issues"][0]["path"] == "gate/leak.py"
    assert canary not in json.dumps(report) and list(out_dir.iterdir()) == []


def test_guest_unpack_round_trip_satisfies_the_source_tree_rule(tmp_path):
    """Mimic A6g: unpack the archive with tarfile, then the plan's source_tree rule."""
    report, out_dir = build(tmp_path)
    guest = tmp_path / "guest"
    guest.mkdir()
    with tarfile.open(out_dir / report["archive"]["name"], mode="r:") as archive:
        archive.extractall(guest, filter="data")
    rule = {"kind": "source_tree", "root": "physloop_root", "path": str(guest),
            "bundle_sha256": report["bundle_sha256"]}
    roots = {"physloop_root": str(guest)}
    result = evaluate_rule(rule, roots=roots)
    assert result["ok"] is True, result
    assert result["recorded"] == {"bundle_sha256": report["bundle_sha256"], "files_checked": 8}
    (guest / ".venv" / "bin").mkdir(parents=True)
    (guest / ".venv" / "bin" / "python").write_bytes(b"x")
    (guest / "gate" / "__pycache__").mkdir()
    (guest / "gate" / "__pycache__" / "a.cpython-310.pyc").write_bytes(b"x")
    assert evaluate_rule(rule, roots=roots)["ok"] is True


def test_real_repository_bundle_holds_the_runner_and_excludes_tests(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    report = bundle.build_bundle(str(REPO), str(out_dir), scope="synthetic")
    assert report["status"] == "passed", report["issues"]
    manifest = read_json_bytes((out_dir / report["archive"]["name"].replace(
        ".tar", ".manifest.json")).read_bytes())
    paths = [row["path"] for row in manifest["files"]]
    for needed in ("pyproject.toml", "uv.lock", ".python-version", "gate/s2_bootstrap.py",
                   "gate/s2_host_fit.py", "gate/s2_executor.py", "gate/s2_capture.py",
                   "scripts/s2_guest_bootstrap.py", "scripts/vast_s2_host_probe.sh",
                   "experiments/S2_SETUP_DRAFT.json"):
        assert needed in paths, needed
    assert all(p.split("/")[0] in ("gate", "scripts", "experiments") or "/" not in p
               for p in paths)
    assert not any(p.startswith(("tests/", "teacher_rl/", "evidence/", "docs/")) or
                   "__pycache__" in p or p.endswith((".pyc", ".DS_Store")) for p in paths)
    assert bundle.verify_archive(archive_bytes(report, out_dir), report["bundle_sha256"]) == []


def test_exclusions_cover_the_repository_gitignore():
    lines = [line.strip() for line in (REPO / ".gitignore").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    allowlisted = set(bundle.INCLUDE_DIRS) | {"experiments"}
    for pattern in lines:
        body = pattern.rstrip("/")
        if "/" in body:  # root-scoped path: only matters if it is inside the allowlist
            assert body.split("/")[0] not in allowlisted, pattern
        elif pattern.endswith("/"):
            assert body in bundle.EXCLUDE_DIR_NAMES or any(
                p == body for p in bundle.EXCLUDE_DIR_PATTERNS), pattern
        else:
            assert pattern in bundle.EXCLUDE_FILE_PATTERNS, pattern


def test_module_has_no_network_or_process_imports():
    tree = ast.parse(Path(bundle.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"socket", "urllib", "http", "subprocess", "requests", "ssl", "ftplib"}


def test_cli_build_then_verify(tmp_path, capsys):
    root = make_tree(tmp_path / "src")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert bundle.main(["build", "--root", str(root), "--out-dir", str(out_dir),
                        "--scope", "synthetic"]) == 0
    report = json.loads(capsys.readouterr().out)
    archive = str(out_dir / report["archive"]["name"])
    assert bundle.main(["verify", archive, "--bundle-sha256", report["bundle_sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert bundle.main(["verify", archive, "--bundle-sha256", "0" * 64]) == 2
    capsys.readouterr()
    shutil.rmtree(out_dir)
    assert bundle.main(["build", "--root", str(root), "--out-dir", str(out_dir),
                        "--scope", "synthetic"]) == 2


def test_symlinked_parent_of_a_nested_include_file_refuses(tmp_path):
    """F3-3 review: experiments/ -> an outside directory must not seal the outside draft."""
    root = make_tree(tmp_path / "src")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "S2_SETUP_DRAFT.json").write_bytes(b"{\"status\": \"outside\"}\n")
    shutil.rmtree(root / "experiments")
    (root / "experiments").symlink_to(outside, target_is_directory=True)
    files, _, issues = bundle.collect_sources(str(root))
    assert [(i["code"], i["path"]) for i in issues] == [
        ("symlink_rejected", "experiments/S2_SETUP_DRAFT.json")]
    assert "experiments/S2_SETUP_DRAFT.json" not in files
    report, out_dir = build(tmp_path, root)
    assert report["status"] == "blocked" and codes(report) == ["symlink_rejected"]
    assert list(out_dir.iterdir()) == [] and report["archive"] is None


@pytest.mark.parametrize("relative", [
    "gate/" + "a" * 101 + ".py",  # one component longer than the ustar name field
    "gate/" + "/".join(["d" * 60] * 4) + "/x.py",  # more than ustar prefix + name hold
])
def test_path_longer_than_ustar_holds_blocks_without_raising(tmp_path, relative):
    """F3-5 review: build_bundle reports a blocking issue instead of tarfile's ValueError."""
    root = make_tree(tmp_path / "src")
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).write_bytes(b"x = 1\n")
    report, out_dir = build(tmp_path, root)
    assert report["status"] == "blocked" and codes(report) == ["path_too_long"]
    assert [i["path"] for i in report["issues"]] == [relative]
    assert list(out_dir.iterdir()) == []


def test_cli_verify_reads_archives_up_to_the_archive_bound(tmp_path, capsys, monkeypatch):
    """F3-6 review: verify used the per-file bound for the whole archive."""
    report, out_dir = build(tmp_path)
    archive = out_dir / report["archive"]["name"]
    size = archive.stat().st_size
    monkeypatch.setattr(bundle, "MAX_FILE_BYTES", 1024)  # every source file is smaller
    monkeypatch.setattr(bundle, "MAX_ARCHIVE_BYTES", size)
    assert size > bundle.MAX_FILE_BYTES
    assert bundle.main(["verify", str(archive), "--bundle-sha256", report["bundle_sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    monkeypatch.setattr(bundle, "MAX_ARCHIVE_BYTES", size - 1)
    assert bundle.main(["verify", str(archive), "--bundle-sha256", report["bundle_sha256"]]) == 2
    assert [i["code"] for i in json.loads(capsys.readouterr().out)["issues"]] == [
        "unreadable_archive"]


def test_bundle_bounds_are_labelled_agent_technical_defaults():
    source = (REPO / "scripts" / "s2_prepare_bundle.py").read_text(encoding="utf-8")
    for name in ("MAX_FILE_BYTES", "MAX_ARCHIVE_BYTES"):
        line = next(text for text in source.splitlines() if text.startswith(name + " ="))
        assert "agent technical default (D-018)" in line, line
