"""Independent negative controls across multi-batch ingestion boundaries."""

import json
from copy import deepcopy

import pytest
from s2_ingestion_fixture import (
    fixture_bundle,
    json_receipt,
    metric_rows,
    parquet_receipt,
    receipt,
)

from gate.s2_extract import extract_bundle


def codes(report):
    return {entry["code"] for entry in report["issues"]}


def assert_blocked(report):
    assert not report["admitted"]
    assert report["issues"]
    assert all(not batch["normalization"]["admitted"] for batch in report["batches"])
    assert all(batch["normalization"]["records"] == [] for batch in report["batches"])
    json.dumps(report, allow_nan=False)


def paired_bundle(root):
    """Two different upstream identities, one shared scientific protocol."""
    root_a, root_b = root / "a", root / "b"
    root_a.mkdir()
    root_b.mkdir()
    first, events_a, _, _ = fixture_bundle(root_a)
    second, events_b, trace_b, summary_b = fixture_bundle(root_b)
    upstream_b = {"run_uuid": "run-b", "clipgt_id": "clip", "rollout_id": "rollout-b"}
    second["schedule"][0].update(job_id="job-b", arm="B", policy_id="reexport")
    second["artifacts"][0]["attempt_id"] = "attempt-b"
    second["batches"][0]["arm"] = "B"
    for event in events_b:
        event["event_id"] += "-b"
        event["job_id"] = "job-b"
        event["attempt_id"] = "attempt-b"
        event["timestamp_us"] += 2
        event["upstream_identity"] = upstream_b
    trace_b["upstream_identity"] = upstream_b
    summary_b["rollouts"][0].update(upstream_b)
    summary_b["rollouts"][0]["metrics"].update(upstream_b)
    second["artifacts"][0]["trace"] = json_receipt(root_b, "trace.json", trace_b)
    rows_b = metric_rows()
    for row in rows_b:
        row.update(upstream_b)
    second["artifacts"][0]["metrics"] = parquet_receipt(root_b, rows_b)
    second["batches"][0]["summary"] = json_receipt(root_b, "summary.json", summary_b)
    for prefix, request in (("a/", first), ("b/", second)):
        for entry in request["artifacts"]:
            for kind in ("metrics", "trace", "completion"):
                entry[kind]["path"] = prefix + entry[kind]["path"]
        request["batches"][0]["summary"]["path"] = prefix + "summary.json"
    combined = {
        **first,
        "schedule": first["schedule"] + second["schedule"],
        "events": json_receipt(root, "all-events.json", events_a + events_b),
        "batches": first["batches"] + second["batches"],
        "artifacts": first["artifacts"] + second["artifacts"],
    }
    return combined, events_a + events_b


def test_independent_two_arm_known_answer(tmp_path):
    request, _ = paired_bundle(tmp_path)
    report = extract_bundle(tmp_path, request)
    assert report["admitted"], report["issues"]
    assert len(report["batches"]) == 2
    assert [b["arm"] for b in report["batches"]] == ["A", "B"]
    assert [b["normalization"]["records"][0]["value"] for b in report["batches"]] == [0.8, 0.8]
    assert report["ledger"]["counts"]["scheduled_jobs"] == 2
    assert report["ledger"]["counts"]["completed_attempts"] == 2


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("last_summary_corrupt", "artifact_integrity"),
        ("last_summary_wrong_arm", "missing_expected_rollout"),
        ("last_batch_missing", "missing_batch"),
        ("last_batch_unknown", "unexpected_batch"),
        ("last_batch_duplicate", "unexpected_batch"),
        ("last_batch_malformed", "invalid_batches"),
        ("last_metrics_missing", "artifact_integrity"),
        ("last_attempt_missing", "missing_attempt_artifacts"),
        ("last_event_failed", "job_not_completed"),
        ("last_events_absent", "missing_job"),
    ],
)
def test_late_failure_never_leaks_first_success(tmp_path, mutation, expected):
    request, events = paired_bundle(tmp_path)
    if mutation == "last_summary_corrupt":
        (tmp_path / "b/summary.json").write_bytes(b"changed")
    elif mutation == "last_summary_wrong_arm":
        request["batches"][1]["summary"] = deepcopy(request["batches"][0]["summary"])
    elif mutation == "last_batch_missing":
        request["batches"].pop()
    elif mutation == "last_batch_unknown":
        request["batches"][1]["protocol_hash"] = "b" * 64
    elif mutation == "last_batch_duplicate":
        request["batches"][1] = deepcopy(request["batches"][0])
    elif mutation == "last_batch_malformed":
        request["batches"][1]["arm"] = []
    elif mutation == "last_metrics_missing":
        (tmp_path / "b/metrics.parquet").unlink()
    elif mutation == "last_attempt_missing":
        request["artifacts"].pop()
    elif mutation == "last_event_failed":
        events[-1].update(type="failed", reason="synthetic host failure")
        request["events"] = json_receipt(tmp_path, "all-events.json", events)
    elif mutation == "last_events_absent":
        request["events"] = json_receipt(tmp_path, "all-events.json", events[:2])
    report = extract_bundle(tmp_path, request)
    assert_blocked(report)
    assert expected in codes(report)
    assert report["ledger"]["counts"]["scheduled_jobs"] == 2


def test_missing_second_attempt_is_not_hidden_by_successful_raw_files(tmp_path):
    request, events = paired_bundle(tmp_path)
    request["events"] = json_receipt(tmp_path, "all-events.json", events[:3])
    report = extract_bundle(tmp_path, request)
    assert_blocked(report)
    assert report["ledger"]["counts"]["attempts"] == 2
    assert report["ledger"]["counts"]["incomplete_attempts"] == 1
    assert report["ledger"]["counts"]["completed_attempts"] == 1


def test_valid_raw_files_cannot_hide_a_prior_failed_attempt(tmp_path):
    request, events = paired_bundle(tmp_path)
    first = deepcopy(events[0])
    first.update(
        event_id="earlier-start", attempt_id="earlier", timestamp_us=0, upstream_identity=None
    )
    failure = deepcopy(events[1])
    failure.update(
        event_id="earlier-failure",
        attempt_id="earlier",
        timestamp_us=0,
        type="failed",
        reason="bootstrap failure",
        upstream_identity=None,
    )
    events[0].update(parent_attempt_id="earlier", retry_reason="restart")
    request["events"] = json_receipt(tmp_path, "all-events.json", [first, failure, *events])
    report = extract_bundle(tmp_path, request)
    assert_blocked(report)
    assert "retry_selection_unregistered" in codes(report)
    assert report["ledger"]["counts"]["attempts"] == 3
    assert report["ledger"]["counts"]["failed_attempts"] == 1
    assert report["ledger"]["counts"]["completed_attempts"] == 2


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("extra_required_sample", "metric_coverage"),
        ("negative_deviation", "invalid_metric_sample"),
        ("mixed_run_name", "raw_identity_mismatch"),
        ("missing_valid", "invalid_metric_sample"),
        ("extra_trace_sample", "horizon_unverified"),
        ("completion_text", "invalid_completion_marker"),
        ("parquet_garbage", "invalid_parquet"),
    ],
)
def test_source_shapes_and_extra_samples_are_not_coerced(tmp_path, mutation, expected):
    request, _, trace, _ = fixture_bundle(tmp_path)
    rows = metric_rows()
    if mutation == "extra_required_sample":
        extra = deepcopy(rows[-1])
        extra["timestamps_us"] = 400_000
        rows.append(extra)
    elif mutation == "negative_deviation":
        next(row for row in rows if row["name"] == "dist_to_gt_trajectory")["values"] = -1.0
    elif mutation == "mixed_run_name":
        rows[0]["run_name"] = "other"
    elif mutation == "missing_valid":
        rows[0]["valid"] = None
    elif mutation == "extra_trace_sample":
        trace["samples"].append(deepcopy(trace["samples"][-1]))
        request["artifacts"][0]["trace"] = json_receipt(tmp_path, "trace.json", trace)
    elif mutation == "completion_text":
        request["artifacts"][0]["completion"] = receipt(tmp_path, "_complete", b"done")
    if mutation == "parquet_garbage":
        request["artifacts"][0]["metrics"] = receipt(tmp_path, "metrics.parquet", b"not parquet")
    else:
        request["artifacts"][0]["metrics"] = parquet_receipt(tmp_path, rows)
    report = extract_bundle(tmp_path, request)
    assert_blocked(report)
    assert expected in codes(report)


@pytest.mark.parametrize(
    "key,value",
    [
        ("events", []),
        ("events", {"path": None, "sha256": [], "size_bytes": True}),
        ("artifacts", {}),
        ("artifacts", [None]),
        ("batches", {}),
        ("batches", [None]),
        ("schedule", [None]),
        ("schedule_request", []),
    ],
)
def test_malformed_request_members_return_json_report(tmp_path, key, value):
    request, _, _, _ = fixture_bundle(tmp_path)
    request[key] = value
    assert_blocked(extract_bundle(tmp_path, request))


def test_same_size_summary_tamper_fails_checksum_before_parsing(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    path = tmp_path / "summary.json"
    original = path.read_bytes()
    changed = original.replace(b'"score": 0.8', b'"score": 0.9')
    assert len(changed) == len(original) and changed != original
    path.write_bytes(changed)
    report = extract_bundle(tmp_path, request)
    assert_blocked(report)
    assert "artifact_integrity" in codes(report)
    assert any("checksum" in item["message"] for item in report["issues"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", {}),
        ("samples", [None] * 4),
        ("step_interval_us", True),
        ("start_timestamp_us", -1),
        ("coordinate_frame", []),
        ("upstream_identity", None),
    ],
)
def test_malformed_trace_members_return_json_report(tmp_path, field, value):
    request, _, trace, _ = fixture_bundle(tmp_path)
    trace[field] = value
    request["artifacts"][0]["trace"] = json_receipt(tmp_path, "trace.json", trace)
    assert_blocked(extract_bundle(tmp_path, request))


def test_duplicate_artifact_attempt_cannot_overwrite_original(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    duplicate = deepcopy(request["artifacts"][0])
    duplicate["trace"] = {"path": "absent", "sha256": "0" * 64, "size_bytes": 0}
    request["artifacts"].append(duplicate)
    report = extract_bundle(tmp_path, request)
    assert_blocked(report)
    assert "duplicate_artifact_attempt" in codes(report)
