"""Known answers and negative controls for gate.s2_bundle (CPU-11; build-spec B2).

Every file here is invented; credential canaries are synthetic shapes, never real
secrets. Hash literals were computed independently (``printf 'hello\\n' | shasum -a
256``), not by the module under test.
"""

import hashlib
import json
import os

import pytest

from gate import s2_bundle as sb
from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.s2_pilot import next_unit_check

HELLO_SHA256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
# Synthetic shapes only: they match the scanner's patterns and authenticate nothing.
FAKE_HF = "hf_" + "Q" * 34
BUDGET = {
    "scopes": {name: {"cap_usd": "10", "used_usd": "6", "liability_usd": "1",
                      "reserve_usd": "1"} for name in ("pilot", "stage", "project")},
    "next_unit_usd": "1.99", "now_epoch": 100, "deadline_epoch": 200,
    "next_unit_seconds": 60, "cleanup_seconds": 20,
}


def entry(path, role="report", restriction="public_safe", licensed=False, replay=None):
    return {"path": path, "role": role, "restriction_class": restriction,
            "licensed_lineage": licensed, "replay": replay}


def index(entries, scope="synthetic", required=()):
    return {"artifact_kind": "s2_bundle_index", "schema_version": 1, "scope": scope,
            "bundle_id": "synthetic-bundle-1", "required_roles": list(required),
            "entries": entries}


def write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if type(data) is bytes else canonical_json_bytes(data) + b"\n")
    return path


def codes(report):
    return {item["code"] for item in report["issues"]}


def analysis_bundle(root, *, tamper=False):
    """A budget input plus its recorded analysis output declared for replay."""
    write(root, "inputs/budget.json", BUDGET)
    output = next_unit_check(BUDGET)
    if tamper:
        output = {**output, "can_start_next_unit": not output["can_start_next_unit"]}
    write(root, "outputs/next-unit.json", output)
    write(root, "notes/hello.txt", b"hello\n")
    return index([
        entry("inputs/budget.json", "analysis_input"),
        entry("outputs/next-unit.json", "analysis_output", replay={
            "function": "next_unit_check", "inputs": {"request": "inputs/budget.json"},
            "kwargs": {}}),
        entry("notes/hello.txt", "review"),
    ], required=("analysis_input", "analysis_output"))


def test_seal_manifest_has_literal_entries_and_excludes_itself(tmp_path):
    write(tmp_path, "notes/hello.txt", b"hello\n")
    report = sb.seal_bundle(tmp_path, index([entry("notes/hello.txt", "review")]))
    assert report["status"] == "passed" and report["sealed"]
    manifest = report["manifest"]
    assert manifest["entries"] == [{
        "path": "notes/hello.txt", "size_bytes": 6, "sha256": HELLO_SHA256, "role": "review",
        "restriction_class": "public_safe", "licensed_lineage": False, "replay": None}]
    assert {item["path"] for item in manifest["entries"]}.isdisjoint(sb.RESERVED_NAMES)
    manifest_bytes = (tmp_path / sb.MANIFEST_NAME).read_bytes()
    assert manifest_bytes == canonical_json_bytes(manifest) + b"\n"
    seal = json.loads((tmp_path / sb.SEAL_NAME).read_text())
    assert seal == {"artifact_kind": "s2_bundle_seal", "schema_version": 1,
                    "manifest_name": "bundle-manifest.json",
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "manifest_size_bytes": len(manifest_bytes)}
    assert manifest["publication_class"] == "public_safe_candidate"
    assert manifest["publication_authorized"] is False
    assert report["authenticity"] == "not_verified"
    verified = sb.verify_bundle(tmp_path, manifest)
    assert verified["verified"] and verified["seal_state"] == "sealed"
    assert verified["entries_checked"] == 1


def test_sealing_twice_is_refused(tmp_path):
    write(tmp_path, "a.txt", b"hello\n")
    first = sb.seal_bundle(tmp_path, index([entry("a.txt")]))
    assert first["sealed"]
    second = sb.seal_bundle(tmp_path, index([entry("a.txt")]))
    assert not second["sealed"] and "bundle_manifest_exists" in codes(second)


@pytest.mark.parametrize("change, code", [
    (lambda root: (root / "a.txt").write_bytes(b"hellO\n"), "changed_bytes"),
    (lambda root: (root / "a.txt").write_bytes(b"hello"), "size_mismatch"),
    (lambda root: (root / "a.txt").unlink(), "missing_artifact"),
    (lambda root: (root / "extra.json").write_bytes(b"{}"), "unindexed_file"),
    (lambda root: (root / "link").symlink_to(root / "a.txt"), "symlink_in_bundle"),
])
def test_verify_rejects_changed_bundle(tmp_path, change, code):
    write(tmp_path, "a.txt", b"hello\n")
    assert sb.seal_bundle(tmp_path, index([entry("a.txt")]))["sealed"]
    change(tmp_path)
    report = sb.verify_bundle(tmp_path)
    assert not report["verified"] and report["status"] == "blocked"
    assert code in codes(report)


def test_manifest_edit_after_seal_is_detected(tmp_path):
    write(tmp_path, "a.txt", b"hello\n")
    assert sb.seal_bundle(tmp_path, index([entry("a.txt")]))["sealed"]
    path = tmp_path / sb.MANIFEST_NAME
    path.write_bytes(path.read_bytes().replace(b'"review"', b'"report"').replace(
        b'"size_bytes":6', b'"size_bytes":7'))
    report = sb.verify_bundle(tmp_path)
    assert "manifest_changed_after_seal" in codes(report)
    assert report["seal_state"] == "invalid" and not report["verified"]


def test_supplied_manifest_must_equal_bundled_manifest(tmp_path):
    write(tmp_path, "a.txt", b"hello\n")
    sealed = sb.seal_bundle(tmp_path, index([entry("a.txt")]))
    other = json.loads(json.dumps(sealed["manifest"]))
    other["bundle_id"] = "another"
    assert "manifest_mismatch" in codes(sb.verify_bundle(tmp_path, other))


def test_interrupted_seal_stays_visibly_unsealed(tmp_path, monkeypatch):
    write(tmp_path, "a.txt", b"hello\n")
    original = sb._write_new

    def crash_before_marker(root, relative, data):
        if relative == sb.SEAL_NAME:
            raise OSError("synthetic crash before the seal marker")
        return original(root, relative, data)

    monkeypatch.setattr(sb, "_write_new", crash_before_marker)
    with pytest.raises(OSError):
        sb.seal_bundle(tmp_path, index([entry("a.txt")]))
    monkeypatch.setattr(sb, "_write_new", original)
    assert (tmp_path / sb.MANIFEST_NAME).exists() and not (tmp_path / sb.SEAL_NAME).exists()
    report = sb.verify_bundle(tmp_path)
    assert report["seal_state"] == "unsealed" and not report["verified"]
    assert "bundle_unsealed" in codes(report)
    again = sb.seal_bundle(tmp_path, index([entry("a.txt")]))
    assert not again["sealed"] and "bundle_manifest_exists" in codes(again)


def test_seal_refuses_symlink_and_unindexed_file_and_writes_nothing(tmp_path):
    write(tmp_path, "a.txt", b"hello\n")
    (tmp_path / "escape").symlink_to("/etc")
    write(tmp_path, "stray.txt", b"x")
    report = sb.seal_bundle(tmp_path, index([entry("a.txt")]))
    assert {"symlink_in_bundle", "unindexed_file"} <= codes(report)
    assert not (tmp_path / sb.MANIFEST_NAME).exists()
    assert not (tmp_path / sb.SEAL_NAME).exists()


@pytest.mark.parametrize("path", ["../outside.txt", "/abs.txt", "a/./b.txt", "a//b.txt",
                                  "a\\b.txt", sb.MANIFEST_NAME])
def test_index_rejects_traversal_absolute_and_reserved_paths(tmp_path, path):
    report = sb.seal_bundle(tmp_path, index([entry(path)]))
    assert not report["sealed"]
    assert codes(report) & {"unsafe_path", "reserved_name"}


def test_duplicate_normalized_names_are_rejected(tmp_path):
    report = sb.seal_bundle(tmp_path, index([entry("Report.json"), entry("report.json")]))
    assert "duplicate_normalized_name" in codes(report)


def test_missing_required_raw_stream_blocks(tmp_path):
    write(tmp_path, "a.txt", b"hello\n")
    report = sb.seal_bundle(tmp_path, index([entry("a.txt")], required=("raw_artifact",)))
    assert "missing_required_role" in codes(report)


def test_declared_but_absent_raw_stream_blocks(tmp_path):
    write(tmp_path, "a.txt", b"hello\n")
    report = sb.seal_bundle(tmp_path, index([entry("a.txt"), entry("raw/run.asl",
                                                                    "raw_artifact")]))
    assert "missing_artifact" in codes(report)


@pytest.mark.parametrize("data, pattern", [
    (f'{{"HF_TOKEN": "{FAKE_HF}"}}'.encode(), "credential_env_assignment"),
    (b"JUPYTER_TOKEN=fake-jupyter-token\n", "credential_env_assignment"),
    (b"export CONTAINER_API_KEY='fake-container-api-key'\n", "credential_env_assignment"),
    (b"PAM_ADDED_SECRET=fake-pam-environment-value\n", "credential_env_assignment"),
    (b"SSH_CONNECTION=203.0.113.7 50000 198.51.100.9 22\n", "credential_env_assignment"),
    (f"log: token {FAKE_HF} leaked".encode(), "token_shape_0"),
    (b"hf download --token fake-value repo\n", "credential_cli_flag"),
    (b"AKIA" + b"Z" * 16, "aws_access_key_id"),
    (b'{"HF_TOKEN": "hf_tokenvalue"}', "credential_env_assignment"),
])
def test_secret_canaries_block_seal_without_echoing_values(tmp_path, data, pattern):
    write(tmp_path, "log.txt", data)
    report = sb.seal_bundle(tmp_path, index([entry("log.txt", "lifecycle", "internal")]))
    assert not report["sealed"]
    found = [item for item in report["issues"] if item["code"] == "secret_canary_detected"]
    assert found and any(pattern in item["message"] for item in found)
    rendered = json.dumps(report)
    for secret in (FAKE_HF, "fake-jupyter-token", "fake-container-api-key", "fake-value",
                   "fake-pam-environment-value", "203.0.113.7"):
        assert secret not in rendered


@pytest.mark.parametrize("data", [
    b'{"env_allowlist": ["HF_HOME", "HF_HUB_OFFLINE", "PATH"]}',
    b'{"HF_TOKEN": null}',
    b'{"secret_env": {"HF_TOKEN": "hf_token", "NGC_API_KEY": "ngc_api_key"}}',
    b"Never pass HF_TOKEN to the driver; JUPYTER_TOKEN is dropped.\n",
    b'{"argv": ["--token", "--next"]}',
])
def test_benign_mentions_are_not_findings(data):
    assert sb.scan_bytes_for_secrets([data]) == []


def test_scan_finds_a_canary_split_across_chunks():
    text = b"x" * 10 + FAKE_HF.encode() + b"y"
    findings = sb.scan_bytes_for_secrets([text[:14], text[14:]])
    assert findings == [("token_shape_0", 10)]


def test_secret_added_after_seal_fails_verification(tmp_path):
    write(tmp_path, "log.txt", b"hello\n")
    assert sb.seal_bundle(tmp_path, index([entry("log.txt")]))["sealed"]
    (tmp_path / "log.txt").write_bytes(b"JUPYTER_TOKEN=fake-jupyter-token\n")
    assert {"secret_canary_detected", "size_mismatch"} <= codes(sb.verify_bundle(tmp_path))


@pytest.mark.parametrize("item, scope, code", [
    (entry("a.txt", restriction="confidential"), "synthetic", "unknown_restriction_class"),
    (entry("a.txt", licensed=True), "synthetic", "licensed_lineage_not_restricted"),
    (entry("a.txt", restriction="internal", licensed=True), "actual",
     "licensed_lineage_not_restricted"),
    (entry("a.txt"), "actual", "actual_marked_public_safe"),
    ({**entry("a.txt"), "licensed_lineage": None}, "synthetic", "unknown_lineage"),
])
def test_restricted_file_classes_fail_closed(tmp_path, item, scope, code):
    write(tmp_path, "a.txt", b"hello\n")
    report = sb.seal_bundle(tmp_path, index([item], scope=scope))
    assert not report["sealed"] and code in codes(report)


def test_licensed_actual_bundle_is_internal_and_never_public(tmp_path):
    write(tmp_path, "scores.json", b"{}")
    write(tmp_path, "receipt.json", b"{}")
    report = sb.seal_bundle(tmp_path, index([
        entry("scores.json", "extraction", "internal_kc_l", licensed=True),
        entry("receipt.json", "lifecycle", "internal"),
        *_actual_minimum_entries(tmp_path / "minimum", prefix="minimum/")], scope="actual"))
    assert report["sealed"]
    assert report["manifest"]["publication_class"] == "internal"
    assert report["manifest"]["publication_authorized"] is False


def test_fresh_directory_replay_reproduces_analysis_hashes(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    sealed = sb.seal_bundle(root, analysis_bundle(root))
    assert sealed["sealed"], sealed["issues"]
    report = sb.replay_bundle(root, tmp_path / "replay")
    assert report["status"] == "passed" and report["replay_identical"]
    assert report["copy"]["seal_state"] == "sealed"
    assert report["dependency_installation_tested"] is False
    expected = identity_hash(next_unit_check(BUDGET))
    assert report["analyses"] == [{
        "path": "outputs/next-unit.json", "function": "next_unit_check",
        "recorded_sha256": expected, "replayed_sha256": expected, "identical": True}]
    copied = sorted(p.relative_to(tmp_path / "replay").as_posix()
                    for p in (tmp_path / "replay").rglob("*") if p.is_file())
    assert copied == sorted(["inputs/budget.json", "outputs/next-unit.json", "notes/hello.txt",
                             sb.MANIFEST_NAME, sb.SEAL_NAME])


def test_replay_reports_a_bundled_output_that_does_not_reproduce(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    assert sb.seal_bundle(root, analysis_bundle(root, tamper=True))["sealed"]
    report = sb.replay_bundle(root, tmp_path / "replay")
    assert not report["replay_identical"] and "replay_mismatch" in codes(report)
    assert report["analyses"][0]["identical"] is False


def test_replay_refuses_existing_destination_and_unverified_source(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    assert sb.seal_bundle(root, analysis_bundle(root))["sealed"]
    (tmp_path / "taken").mkdir()
    assert "destination_exists" in codes(sb.replay_bundle(root, tmp_path / "taken"))
    assert list((tmp_path / "taken").iterdir()) == []
    (root / "notes/hello.txt").write_bytes(b"HELLO\n")
    report = sb.replay_bundle(root, tmp_path / "replay")
    assert "source_bundle_unverified" in codes(report)
    assert not (tmp_path / "replay").exists()


@pytest.mark.parametrize("replay, code", [
    ({"function": "os.system", "inputs": {}, "kwargs": {}}, "unsupported_replay_function"),
    ({"function": "next_unit_check", "inputs": {"request": "missing.json"}, "kwargs": {}},
     "replay_input_not_bundled"),
    ({"function": "next_unit_check", "inputs": {}, "kwargs": {"shell": True}}, "invalid_replay"),
])
def test_replay_specifications_are_closed(tmp_path, replay, code):
    write(tmp_path, "out.json", b"{}")
    report = sb.seal_bundle(tmp_path, index([entry("out.json", "analysis_output",
                                                   replay=replay)]))
    assert not report["sealed"] and code in codes(report)


def test_unknown_index_version_fails_closed(tmp_path):
    bad = index([entry("a.txt")])
    bad["schema_version"] = 2
    assert "unsupported_index" in codes(sb.seal_bundle(tmp_path, bad))
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------- review fixes (B2)


def _resign(root, mutate):
    """Rewrite the manifest via ``mutate`` with a consistent seal (same-origin forgery)."""
    manifest = json.loads((root / sb.MANIFEST_NAME).read_bytes())
    mutate(manifest)
    data = canonical_json_bytes(manifest) + b"\n"
    (root / sb.MANIFEST_NAME).write_bytes(data)
    seal = {"artifact_kind": sb.SEAL_KIND, "schema_version": 1,
            "manifest_name": sb.MANIFEST_NAME,
            "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "manifest_size_bytes": len(data)}
    (root / sb.SEAL_NAME).write_bytes(canonical_json_bytes(seal) + b"\n")


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_unreadable_subdirectory_blocks_seal_and_verify(tmp_path):
    """Review B2-02: a directory the walk cannot list is never treated as empty."""
    sealed_root = tmp_path / "sealed"
    write(sealed_root, "a.txt", b"hello\n")
    assert sb.seal_bundle(sealed_root, index([entry("a.txt")]))["sealed"]
    write(sealed_root, "hidden/extra.txt", b"JUPYTER_TOKEN=synthetic-canary-value\n")
    fresh = tmp_path / "fresh"
    write(fresh, "a.txt", b"hello\n")
    write(fresh, "hidden/extra.txt", b"JUPYTER_TOKEN=synthetic-canary-value\n")
    os.chmod(sealed_root / "hidden", 0o000)
    os.chmod(fresh / "hidden", 0o000)
    try:
        verified = sb.verify_bundle(sealed_root)
        sealed = sb.seal_bundle(fresh, index([entry("a.txt")]))
    finally:
        os.chmod(sealed_root / "hidden", 0o755)
        os.chmod(fresh / "hidden", 0o755)
    assert not verified["verified"] and "unreadable_directory" in codes(verified)
    assert not sealed["sealed"] and "unreadable_directory" in codes(sealed)
    assert not (fresh / sb.MANIFEST_NAME).exists()


def test_seal_marker_of_unknown_version_fails_closed(tmp_path):
    """Review B2-04."""
    write(tmp_path, "a.txt", b"hello\n")
    assert sb.seal_bundle(tmp_path, index([entry("a.txt")]))["sealed"]
    seal = json.loads((tmp_path / sb.SEAL_NAME).read_bytes())
    seal["schema_version"] = 999
    (tmp_path / sb.SEAL_NAME).write_bytes(canonical_json_bytes(seal) + b"\n")
    report = sb.verify_bundle(tmp_path)
    assert not report["verified"] and report["seal_state"] != "sealed"
    assert "invalid_seal" in codes(report)


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(scope="ACTUAL"),
    lambda m: m["entries"][0].update(role="bogus_role"),
    lambda m: m["entries"][0].update(role=["report"]),
    lambda m: m.update(authenticity="verified"),
    lambda m: m.update(bundle_id="../../x y"),
    lambda m: m.update(units={"size_bytes": "kilobytes"}),
    lambda m: m.update(index_sha256="not-a-hash"),
])
def test_resealed_manifest_enumerations_fail_closed(tmp_path, mutate):
    """Review B2-04: a consistent same-origin re-seal cannot widen the closed sets."""
    write(tmp_path, "a.txt", b"hello\n")
    assert sb.seal_bundle(tmp_path, index([entry("a.txt")]))["sealed"]
    _resign(tmp_path, mutate)
    report = sb.verify_bundle(tmp_path)
    assert not report["verified"] and report["status"] == "blocked"
    assert "invalid_manifest" in codes(report)


@pytest.mark.parametrize("item", [
    {**entry("out.json"), "role": ["report"]},
    {**entry("out.json"), "restriction_class": ["public_safe"]},
    {**entry("out.json"), "path": ["out.json"]},
    entry("out.json", "analysis_output", replay={
        "function": "next_unit_check", "inputs": {"request": ["a.txt"]}, "kwargs": {}}),
    entry("out.json", "analysis_output", replay={
        "function": ["next_unit_check"], "inputs": {}, "kwargs": {}}),
])
def test_malformed_index_shapes_are_issues_not_exceptions(tmp_path, item):
    """Review B2-07."""
    write(tmp_path, "out.json", b"{}")
    write(tmp_path, "a.txt", b"hello\n")
    report = sb.seal_bundle(tmp_path, index([item, entry("a.txt")]))
    assert not report["sealed"] and report["issues"]
    assert sb.validate_index(index([item, entry("a.txt")]))


def test_replay_kwarg_secret_never_reaches_the_manifest(tmp_path):
    """Review B2-08: the manifest bytes are scanned before they are written."""
    write(tmp_path, "in.json", {"x": 1})
    write(tmp_path, "out.json", b"{}\n")
    report = sb.seal_bundle(tmp_path, index([
        entry("in.json", "analysis_input"),
        entry("out.json", "analysis_output", replay={
            "function": "describe_repeats", "inputs": {"observations": "in.json"},
            "kwargs": {"scope": FAKE_HF}})]))
    assert not report["sealed"] and "secret_canary_detected" in codes(report)
    assert FAKE_HF not in json.dumps(report)
    assert not (tmp_path / sb.MANIFEST_NAME).exists()


def test_forged_manifest_carrying_a_secret_fails_verification(tmp_path):
    write(tmp_path, "a.txt", b"hello\n")
    assert sb.seal_bundle(tmp_path, index([entry("a.txt")]))["sealed"]
    _resign(tmp_path, lambda m: m.update(bundle_id=FAKE_HF))  # matches the id alphabet
    report = sb.verify_bundle(tmp_path)
    assert not report["verified"] and "secret_canary_detected" in codes(report)
    assert FAKE_HF not in json.dumps(report)


@pytest.mark.parametrize("data, pattern", [
    (b'{"api_key": "' + b"0123456789abcdef" * 4 + b'"}', "credential_env_assignment"),
    (b'{"jupyter_token": "abcdef0123456789abcdef0123456789"}', "credential_env_assignment"),
    (b"hf_token=synthetic-hf-canary-0123456789abcdef\n", "credential_env_assignment"),
    (b"Authorization: Bearer synthetic0123456789abcdefABCDEF\n", "bearer_token"),
    (b"vastai set api-key 0123456789abcdef0123456789abcdef\n", "api_key_argument"),
])
def test_scanner_covers_lowercase_names_bearer_and_api_key_argv(data, pattern):
    """Review B2-08: synthetic shapes the scanner previously missed."""
    assert pattern in [pattern_id for pattern_id, _offset in sb.scan_bytes_for_secrets([data])]


@pytest.mark.parametrize("data", [
    b'{"api_key": null}',
    b'{"api_key": {"type": "string"}}',
    b'{"hf_token": "hf_token", "ngc_api_key": "ngc_api_key"}',
    b'{"secret_files": ["hf_token", "ngc_api_key"]}',
    b"Bearer tokens are never logged; the api-key argument is dropped.\n",
    b'{"public_key_fingerprint": "SHA256:abc"}',
])
def test_widened_scanner_keeps_benign_mentions_clean(data):
    assert sb.scan_bytes_for_secrets([data]) == []


def _actual_minimum_entries(root, prefix=""):
    entries = []
    for role in sb.ACTUAL_MINIMUM_ROLES:
        write(root, f"{role}.json", b"{}\n")
        entries.append(entry(f"{prefix}{role}.json", role, "internal"))
    return entries


def test_actual_bundle_needs_the_cpu11_minimum(tmp_path):
    """Review B2-10: an actual bundle cannot be one 'other' receipt."""
    write(tmp_path, "receipt.json", b"{}\n")
    report = sb.seal_bundle(tmp_path, index([entry("receipt.json", "other", "internal")],
                                            scope="actual"))
    assert not report["sealed"] and "missing_required_role" in codes(report)
    missing = {item["message"] for item in report["issues"]
               if item["code"] == "missing_required_role"}
    assert len(missing) == len(sb.ACTUAL_MINIMUM_ROLES)
    complete = tmp_path / "complete"
    report = sb.seal_bundle(complete, index(_actual_minimum_entries(complete), scope="actual"))
    assert report["sealed"], report["issues"]
    assert sb.verify_bundle(complete)["verified"]


def test_verify_rejects_an_actual_manifest_without_the_minimum(tmp_path):
    root = tmp_path / "b"
    assert sb.seal_bundle(root, index(_actual_minimum_entries(root), scope="actual"))["sealed"]
    _resign(root, lambda m: m.update(entries=[e for e in m["entries"] if e["role"] != "cost"]))
    (root / "cost.json").unlink()
    report = sb.verify_bundle(root)
    assert not report["verified"] and "missing_required_role" in codes(report)


def _source_bundle(root, source_root, recorded_sha):
    write(source_root, "gate/thing.py", b"hello\n")
    write(root, "source/source-manifest.json", {
        "artifact_kind": "s2_source_manifest", "schema_version": 1, "scope": "synthetic",
        "files": [{"path": "gate/thing.py", "sha256": recorded_sha, "size_bytes": 6}],
        "note": "synthetic"})
    return index([entry("source/source-manifest.json", "source_snapshot")])


def test_replay_compares_the_recorded_source_snapshot(tmp_path):
    """Review B2-10: replay states whether it ran against the recorded source bytes."""
    root, source = tmp_path / "bundle", tmp_path / "src"
    assert sb.seal_bundle(root, _source_bundle(root, source, HELLO_SHA256))["sealed"]
    report = sb.replay_bundle(root, tmp_path / "replay", source_root=source)
    assert report["replay_identical"], report["issues"]
    assert report["source_comparison"] == {"status": "identical", "files_checked": 1,
                                           "changed": [], "missing": []}


def test_replay_reports_source_drift(tmp_path):
    root, source = tmp_path / "bundle", tmp_path / "src"
    assert sb.seal_bundle(root, _source_bundle(root, source, "0" * 64))["sealed"]
    report = sb.replay_bundle(root, tmp_path / "replay", source_root=source)
    assert not report["replay_identical"] and "source_drift" in codes(report)
    assert report["source_comparison"]["changed"] == ["gate/thing.py"]
    (source / "gate/thing.py").unlink()
    report = sb.replay_bundle(root, tmp_path / "replay2", source_root=source)
    assert report["source_comparison"]["missing"] == ["gate/thing.py"]


def test_replay_without_source_snapshot_says_so(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    assert sb.seal_bundle(root, analysis_bundle(root))["sealed"]
    report = sb.replay_bundle(root, tmp_path / "replay")
    assert report["source_comparison"]["status"] == "not_recorded"
