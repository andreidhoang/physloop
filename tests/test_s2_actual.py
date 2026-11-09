"""Literal known answers for descriptive actual-capture inspection (build-spec B1, P5)."""

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gate.s2_actual import inspect_actual_unit, inspect_metrics_parquet
from gate.s2_extract import PARQUET_COLUMNS, RAW_AGGREGATIONS
from scripts.s2_guest_manifest import build_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "s2_actual"
SCENE = "synthetic-scene-0"
RUN_UUID = "00000000-1111-4111-8111-111111111111"
A = "00000000-0000-4000-8000-000000000001"
B = "00000000-0000-4000-8000-000000000002"
BINDING = {"scene_id": SCENE, "requested_steps": 3}


def metrics_table(*, rollout=A, run=RUN_UUID, drop=("progress_rel",), short=("offroad",),
                  invalid=("collision_front", 200), string_type=None):
    """Long-format rows in the pinned writer layout; polars writes large strings."""
    string_type = string_type or pa.large_string()
    rows = []
    for name, aggregation in RAW_AGGREGATIONS.items():
        if name in drop:
            continue
        stamps = [100, 200] if name in short else [100, 200, 300]
        for stamp in stamps:
            rows.append({"name": name, "timestamps_us": stamp, "values": 0.0,
                         "valid": (name, stamp) != invalid, "time_aggregation": aggregation,
                         "clipgt_id": SCENE, "rollout_id": rollout, "run_uuid": run,
                         "run_name": "e2e_challenge_dev"})
    schema = pa.schema([("name", string_type), ("timestamps_us", pa.int64()),
                        ("values", pa.float64()), ("valid", pa.bool_()),
                        ("time_aggregation", string_type), ("clipgt_id", string_type),
                        ("rollout_id", string_type), ("run_uuid", string_type),
                        ("run_name", string_type)])
    assert set(schema.names) == PARQUET_COLUMNS
    return pa.Table.from_pylist(rows, schema=schema)


def parquet_bytes(table):
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def make_run(tmp_path, metrics=None):
    run = tmp_path / "run"
    shutil.copytree(FIXTURES / "run", run)
    folder_a = run / "rollouts" / SCENE / A
    folder_b = run / "rollouts" / SCENE / B
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)
    (folder_a / "_complete").write_bytes(b"")
    (folder_a / "rollout.asl").write_bytes(b"asl-a")
    (folder_a / "metrics.parquet").write_bytes(
        metrics if metrics is not None else parquet_bytes(metrics_table()))
    (folder_b / "rollout.asl").write_bytes(b"asl-b")
    manifest = build_manifest(run, scope="synthetic")
    assert manifest["status"] == "passed"
    return run, manifest["files"]


def codes(report):
    return {item["code"]: item["severity"] for item in report["issues"]}


def test_fixture_files_match_their_provenance():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert provenance["scope"] == "synthetic"
    for name, digest in provenance["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


def test_unit_inspection_known_answers(tmp_path):
    run, receipts = make_run(tmp_path)
    report = inspect_actual_unit(run, receipts, BINDING, scope="synthetic")
    assert report["status"] == "passed"
    assert report["diagnostic_only"] is True and report["score_admitted"] is False
    assert "score" not in json.dumps(report["summary"])
    assert report["run_uuid_binding"]["run_uuid"] == RUN_UUID
    assert report["steps"] == {
        "requested": 3, "generated_config_n_sim_steps": 3,
        "logged_effective_by_rollout": {A: [3], B: [2]},
        "log_sources": ["compose_up_stdout"], "differs_from_requested": [B],
        "acceptance": "owner_decision_not_made_here"}
    assert report["folders_without_complete"] == [B]
    assert [(r["rollout_uuid"], r["complete_marker"], r["metrics"], r["asl"])
            for r in report["rollouts"]] == [(A, True, True, True), (B, False, False, True)]
    assert report["rollouts"][1]["metrics_inspection"] is None
    assert report["summary"] == {
        "rows": 3, "status_counts": {"fail": 2, "pass": 1},
        "failed_rows_null_run_uuid": [B, "failed-1"],
        "rows_without_folder": ["failed-1"], "folders_without_row": []}
    assert codes(report) == {
        "effective_steps_differ_from_requested": "warning",
        "invalid_samples_preserved": "info",
        "required_streams_missing": "warning",
        "required_stream_coverage_partial": "warning",
        "folders_without_complete": "info",
        "failed_rows_null_run_uuid": "info"}


def test_metrics_streams_preserve_invalid_samples_and_report_coverage(tmp_path):
    run, receipts = make_run(tmp_path)
    metrics = inspect_actual_unit(run, receipts, BINDING, scope="synthetic")["rollouts"][0][
        "metrics_inspection"]
    assert metrics["schema_matches"] and metrics["rows"] == 3 * 8 - 1
    assert metrics["identity"] == {"clipgt_id": [SCENE], "rollout_id": [A],
                                   "run_uuid": [RUN_UUID], "run_name": ["e2e_challenge_dev"]}
    assert metrics["identity_matches"] == {"clipgt_id": True, "rollout_id": True,
                                           "run_uuid": True}
    assert metrics["invalid_samples_preserved"] == 1
    assert metrics["streams"]["collision_front"] == {
        "rows": 3, "valid_true": 2, "valid_false": 1, "valid_null": 0, "values_null": 0,
        "values_nonfinite": 0, "timestamps_null": 0, "first_timestamp_us": 100,
        "last_timestamp_us": 300, "distinct_timestamps": 3, "duplicate_timestamps": 0,
        "strictly_increasing": True, "time_aggregations": ["max"]}
    required = metrics["required_streams"]
    assert required["missing"] == ["progress_rel"]
    assert required["coverage_of_required_union"]["offroad"] == {"covered": 2, "union": 3}
    assert required["coverage_of_required_union"]["dist_to_gt_trajectory"] == {
        "covered": 3, "union": 3}
    assert required["aggregation_mismatch"] == []


def test_duplicates_nonfinite_identity_and_type_differences_are_described():
    table = metrics_table(drop=(), short=(), invalid=None, run="other-run",
                          string_type=pa.string())
    rows = table.to_pylist()
    rows.append({**rows[0]})
    rows.append({**rows[1], "values": float("nan"), "timestamps_us": 50})
    rows.append({**rows[2], "time_aggregation": "mean"})
    data = parquet_bytes(pa.Table.from_pylist(rows, schema=table.schema))
    result = inspect_metrics_parquet(data, clipgt_id=SCENE, rollout_uuid=A, run_uuid=RUN_UUID)
    stream = result["streams"][rows[0]["name"]]
    # Six rows at 100, 200, 300, 100, 50, 300: four distinct timestamps, two duplicates.
    assert stream["rows"] == 6 and stream["distinct_timestamps"] == 4
    assert stream["duplicate_timestamps"] == 2 and not stream["strictly_increasing"]
    assert stream["values_nonfinite"] == 1
    assert stream["time_aggregations"] == ["last", "mean"]
    assert result["identity_matches"]["run_uuid"] is False
    # The extra timestamp 50 widens the required union, so every other stream covers 3 of 4.
    assert result["required_streams"]["coverage_of_required_union"]["offroad"] == {
        "covered": 3, "union": 4}
    assert {item["code"] for item in result["issues"]} == {
        "metrics_identity_mismatch", "required_stream_aggregation_differs",
        "required_stream_coverage_partial"}


def test_schema_difference_is_a_warning_and_unreadable_bytes_block(tmp_path):
    table = metrics_table().drop_columns(["run_name"])
    result = inspect_metrics_parquet(parquet_bytes(table), clipgt_id=SCENE, rollout_uuid=A,
                                     run_uuid=RUN_UUID)
    assert result["schema_matches"] is False and result["streams"] == {}
    assert result["issues"][0]["code"] == "metrics_schema_differs"
    run, receipts = make_run(tmp_path, metrics=b"not parquet")
    report = inspect_actual_unit(run, receipts, BINDING, scope="synthetic")
    assert report["status"] == "blocked" and codes(report)["metrics_unreadable"] == "blocking"


def test_tampered_bytes_after_manifest_block(tmp_path):
    run, receipts = make_run(tmp_path)
    (run / "aggregate" / "results-summary.json").write_text('{"schema_version": 1}')
    report = inspect_actual_unit(run, receipts, BINDING, scope="synthetic")
    assert report["status"] == "blocked"
    assert codes(report)["receipt_mismatch"] == "blocking" and report["summary"] is None


def test_byte_bound_blocks_before_reading(tmp_path):
    run, receipts = make_run(tmp_path)
    report = inspect_actual_unit(run, receipts, BINDING, scope="synthetic",
                                 max_parquet_bytes=10)
    assert codes(report)["file_exceeds_bound"] == "blocking"
    assert report["rollouts"][0]["metrics_inspection"] is None


def test_missing_inputs_are_absent_not_zero(tmp_path):
    # B1-F3: nothing inspected is never a pass (contract 2.2); the earlier version of this
    # test asserted "passed" for zero receipts and encoded the defect.
    report = inspect_actual_unit(tmp_path, [], BINDING, scope="synthetic")
    assert report["status"] == "not_run"
    assert report["summary"] is None and report["run_uuid_binding"] is None
    assert report["steps"]["logged_effective_by_rollout"] == {}
    assert report["steps"]["generated_config_n_sim_steps"] is None
    assert codes(report) == {"run_metadata_absent": "warning", "summary_absent": "warning",
                             "logs_absent": "warning", "generated_config_absent": "warning",
                             "no_rollout_folders": "warning"}


@pytest.mark.parametrize("removed, code", [
    ("physloop-capture/compose-up.stdout.txt", "logs_absent"),
    ("generated-user-config-0.yaml", "generated_config_absent"),
    ("run_metadata.yaml", "run_metadata_absent"),
    ("aggregate/results-summary.json", "summary_absent"),
])
def test_any_required_input_absent_is_unverified_not_passed(tmp_path, removed, code):
    """B1-F3: run_metadata, summary, one Compose log and the generated config must be read."""
    run, receipts = make_run(tmp_path)
    kept = [item for item in receipts if item["path"] != removed]
    assert len(kept) == len(receipts) - 1
    report = inspect_actual_unit(run, kept, BINDING, scope="synthetic")
    assert report["status"] == "unverified"
    assert codes(report)[code] == "warning"


def test_no_rollout_folders_is_reported(tmp_path):
    """B1-F3: a unit without rollout folders says so instead of reading as agreement."""
    run, receipts = make_run(tmp_path)
    kept = [item for item in receipts if not item["path"].startswith("rollouts/")]
    report = inspect_actual_unit(run, kept, BINDING, scope="synthetic")
    assert report["rollouts"] == [] and codes(report)["no_rollout_folders"] == "warning"
    assert report["status"] == "passed"  # every required input was read and described


@pytest.mark.parametrize("rollout_id", [["x"], {"a": 1}, 7, None])
def test_non_string_summary_rollout_id_is_malformed_and_blocks(tmp_path, rollout_id):
    """B1-F11: a list/object rollout_id must not crash the inspection."""
    run = tmp_path / "run"
    shutil.copytree(FIXTURES / "run", run)
    path = run / "aggregate" / "results-summary.json"
    data = json.loads(path.read_text())
    data["rollouts"][0]["rollout_id"] = rollout_id
    path.write_text(json.dumps(data))
    receipts = build_manifest(run, scope="synthetic")["files"]
    report = inspect_actual_unit(run, receipts, BINDING, scope="synthetic")
    assert report["status"] == "blocked"
    assert codes(report)["summary_rows_malformed"] == "blocking"
    assert report["summary"]["status_counts"]["malformed"] == 1
    assert report["summary"]["rows"] == 3


def test_symlinked_receipt_is_not_followed(tmp_path):
    run, receipts = make_run(tmp_path)
    target = run / "run_metadata.yaml"
    moved = tmp_path / "elsewhere.yaml"
    target.rename(moved)
    target.symlink_to(moved)
    report = inspect_actual_unit(run, receipts, BINDING, scope="synthetic")
    assert codes(report)["receipt_mismatch"] == "blocking"


@pytest.mark.parametrize("data", [
    # The SafeLoader timestamp constructor raises ValueError on an impossible date.
    b"simulation_config:\n  n_sim_steps: 3\nwhen: 2026-13-45\n",
    # Nesting deeper than the interpreter recursion limit raises RecursionError.
    b"{a: " * 3000 + b"1" + b"}" * 3000 + b"\n",
])
def test_generated_config_constructor_errors_block_instead_of_raising(tmp_path, data):
    run, _receipts = make_run(tmp_path)
    (run / "generated-user-config-0.yaml").write_bytes(data)
    manifest = build_manifest(run, scope="synthetic")
    assert manifest["status"] == "passed"
    report = inspect_actual_unit(run, manifest["files"], BINDING, scope="synthetic")
    assert report["status"] == "blocked"
    assert codes(report)["generated_config_unreadable"] == "blocking"
    assert report["steps"]["generated_config_n_sim_steps"] is None


@pytest.mark.parametrize("arguments", [
    {"binding": {"scene_id": SCENE}},
    {"binding": {**BINDING, "requested_steps": 0}},
    {"binding": {**BINDING, "requested_steps": True}},
    {"scope": "real"},
    {"receipts": [{"path": "a"}]},
    {"receipts": [{"path": "a", "sha256": "0" * 64, "size_bytes": 1}] * 2},
])
def test_invalid_inputs_raise(tmp_path, arguments):
    values = {"receipts": [], "binding": BINDING, "scope": "synthetic", **arguments}
    with pytest.raises(ValueError):
        inspect_actual_unit(tmp_path, values["receipts"], values["binding"],
                            scope=values["scope"])
