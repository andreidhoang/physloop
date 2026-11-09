"""Known answers and negative controls for the no-follow guest manifest (build-spec B1)."""

import hashlib
import json
import os

import pytest

from gate.s2_identity import identity_hash
from gate.s2_pilot import verify_export
from scripts.s2_guest_manifest import (
    build_manifest,
    duplicate_normalized_issues,
    main,
    write_manifest,
)

# Hand-derived: sha256(b"abc") and sha256(b"") are the standard FIPS 180-2 test vectors.
SHA_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def tree(root):
    (root / "rollouts" / "clip").mkdir(parents=True)
    (root / "rollouts" / "clip" / "metrics.parquet").write_bytes(b"abc")
    (root / "rollouts" / "clip" / "_complete").write_bytes(b"")
    (root / "run_metadata.yaml").write_bytes(b"abc")
    return root


def test_known_answer_files_sorted_with_exact_receipt_shape(tmp_path):
    root = tree(tmp_path / "run")
    report = build_manifest(root, scope="synthetic")
    assert report["status"] == "passed" and report["issues"] == []
    assert report["files"] == [
        {"path": "rollouts/clip/_complete", "sha256": SHA_EMPTY, "size_bytes": 0},
        {"path": "rollouts/clip/metrics.parquet", "sha256": SHA_ABC, "size_bytes": 3},
        {"path": "run_metadata.yaml", "sha256": SHA_ABC, "size_bytes": 3},
    ]
    assert report["totals"] == {"files": 3, "symlinks": 0, "size_bytes": 6}
    assert report["files_sha256"] == identity_hash(report["files"])
    assert report["artifact_kind"] == "s2_guest_manifest" and report["scope"] == "synthetic"
    assert report["producer"]["module"] == "scripts.s2_guest_manifest"


def test_manifest_round_trips_through_verify_export(tmp_path):
    root = tree(tmp_path / "run")
    report = build_manifest(root, scope="synthetic")
    assert verify_export(root, report["files"])["verified"]
    (root / "run_metadata.yaml").write_bytes(b"abd")
    assert not verify_export(root, report["files"])["verified"]


def test_large_file_streams_in_chunks_to_the_reference_hash(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    data = os.urandom(3 * 1024 * 1024 + 17)
    (root / "rollout.asl").write_bytes(data)
    report = build_manifest(root, scope="synthetic")
    assert report["files"] == [{"path": "rollout.asl", "size_bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()}]


def test_output_is_deterministic(tmp_path):
    root = tree(tmp_path / "run")
    first = build_manifest(root, scope="synthetic")
    second = build_manifest(root, scope="synthetic")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_symlink_is_rejected_by_default_and_never_followed(tmp_path):
    root = tree(tmp_path / "run")
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"outside")
    (root / "escape").symlink_to(outside)
    report = build_manifest(root, scope="synthetic")
    assert report["status"] == "blocked"
    assert [(i["code"], i["path"]) for i in report["issues"]] == [("symlink_rejected", "escape")]
    assert all(item["path"] != "escape" for item in report["files"])
    assert hashlib.sha256(b"outside").hexdigest() not in json.dumps(report)


def test_symlinked_directory_is_not_descended(tmp_path):
    root = tree(tmp_path / "run")
    other = tmp_path / "other"
    other.mkdir()
    (other / "inner.txt").write_bytes(b"x")
    (root / "linked").symlink_to(other, target_is_directory=True)
    report = build_manifest(root, scope="synthetic")
    assert {i["code"] for i in report["issues"]} == {"symlink_rejected"}
    assert not any(item["path"].startswith("linked/") for item in report["files"])


def test_record_policy_lists_relative_inside_links_and_blocks_escapes(tmp_path):
    root = tree(tmp_path / "run")
    (root / "aggregate" / "videos" / "all").mkdir(parents=True)
    (root / "aggregate" / "videos" / "all" / "a.mp4").symlink_to(
        "../../../rollouts/clip/metrics.parquet")
    report = build_manifest(root, scope="synthetic", symlink_policy="record")
    assert report["status"] == "passed"
    assert report["symlinks"] == [{"path": "aggregate/videos/all/a.mp4",
                                   "target": "../../../rollouts/clip/metrics.parquet"}]
    (root / "up").symlink_to("../../etc")
    (root / "abs").symlink_to("/etc/hosts")
    report = build_manifest(root, scope="synthetic", symlink_policy="record")
    assert report["status"] == "blocked"
    assert sorted(i["path"] for i in report["issues"]
                  if i["code"] == "symlink_outside_root") == ["abs", "up"]


def test_root_reached_through_a_symlink_is_refused(tmp_path):
    real = tree(tmp_path / "real")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    report = build_manifest(tmp_path / "alias", scope="synthetic")
    assert report["status"] == "blocked" and report["files"] == []
    assert report["issues"][0]["code"] == "root_unopenable"


def test_special_file_blocks(tmp_path):
    root = tree(tmp_path / "run")
    os.mkfifo(root / "pipe")
    report = build_manifest(root, scope="synthetic")
    assert [(i["code"], i["path"]) for i in report["issues"]] == [("special_file", "pipe")]


def test_duplicate_normalized_names_block(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "café").write_bytes(b"1")
    try:
        (root / "café").write_bytes(b"2")
    except OSError:
        pytest.skip("filesystem cannot hold both normalization forms")
    names = os.listdir(root)
    if len(names) < 2:
        pytest.skip("filesystem normalizes names itself")
    report = build_manifest(root, scope="synthetic")
    assert "duplicate_normalized_path" in {i["code"] for i in report["issues"]}


def test_duplicate_normalized_paths_known_answer():
    issues = duplicate_normalized_issues(["caf\u00e9/x", "cafe\u0301/x", "Readme", "README", "ok"])
    assert sorted((i["code"], i["path"]) for i in issues) == [
        ("duplicate_normalized_path", "Readme"), ("duplicate_normalized_path", "caf\u00e9/x")]
    assert duplicate_normalized_issues(["a", "b/a"]) == []


def test_control_character_name_blocks(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "bad\nname").write_bytes(b"1")
    report = build_manifest(root, scope="synthetic")
    assert [i["code"] for i in report["issues"]] == ["unportable_name"]
    assert report["files"] == []


@pytest.mark.parametrize("kwargs", [{"scope": None}, {"scope": "real"},
                                    {"scope": "synthetic", "symlink_policy": "follow"}])
def test_invalid_scope_or_policy_raises(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_manifest(tmp_path, **kwargs)


def test_write_refuses_inside_root_and_existing_output(tmp_path):
    root = tree(tmp_path / "run")
    report = build_manifest(root, scope="synthetic")
    with pytest.raises(ValueError):
        write_manifest(report, root / "manifest.json", root=root)
    output = write_manifest(report, tmp_path / "manifest.json", root=root)
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError):
        write_manifest(report, tmp_path / "manifest.json", root=root)


def test_cli_exit_codes(tmp_path):
    root = tree(tmp_path / "run")
    assert main([str(root), "--scope", "synthetic", "--output", str(tmp_path / "m1.json")]) == 0
    (root / "link").symlink_to("run_metadata.yaml")
    assert main([str(root), "--scope", "synthetic", "--output", str(tmp_path / "m2.json")]) == 1
    assert main([str(root), "--scope", "synthetic", "--record-symlinks",
                 "--output", str(tmp_path / "m3.json")]) == 0
