"""Known answers at the synthetic raw-file → unchanged adapter boundary."""

from copy import deepcopy

import pytest
from s2_ingestion_fixture import fixture_bundle, json_receipt, metric_rows, parquet_receipt

from gate.s2_extract import extract_bundle


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_complete_source_shaped_fixture_reaches_adapter(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    result = extract_bundle(tmp_path, request)
    assert result["admitted"], result["issues"]
    assert result["runtime_qualified"] is False
    assert result["cloud_launch_authorized"] is False
    batch = result["batches"][0]["normalization"]
    assert batch["records"][0]["value"] == 0.8
    assert batch["records"][0]["deviation_m"] == 4.2
    assert batch["counts"]["out_of_envelope"] == 1
    assert result["observations"][0]["actual_steps"] == 3
    assert result["ledger"]["counts"]["scheduled_jobs"] == 1


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("offroad_absent", "missing_raw_metric"),
        ("truncated", "metric_coverage"),
        ("invalid_mask", "invalid_metric_sample"),
        ("duplicate", "metric_timestamp_order"),
        ("reordered", "metric_timestamp_order"),
        ("identity", "raw_identity_mismatch"),
        ("nan", "invalid_metric_sample"),
        ("aggregation", "metric_aggregation"),
    ],
)
def test_bad_metric_stream_never_admits_summary_zero(tmp_path, mutation, expected):
    request, _, _, _ = fixture_bundle(tmp_path)
    rows = metric_rows()
    if mutation == "offroad_absent":
        rows = [row for row in rows if row["name"] != "offroad"]
    elif mutation == "truncated":
        rows = [row for row in rows if row["timestamps_us"] < 300_000]
    elif mutation == "invalid_mask":
        rows[0]["valid"] = False
    elif mutation == "duplicate":
        rows.insert(0, deepcopy(rows[0]))
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "identity":
        rows[0]["run_uuid"] = "another-run"
    elif mutation == "nan":
        rows[0]["values"] = float("nan")
    elif mutation == "aggregation":
        rows[0]["time_aggregation"] = "mean"
    request["artifacts"][0]["metrics"] = parquet_receipt(tmp_path, rows)
    result = extract_bundle(tmp_path, request)
    assert not result["admitted"]
    assert expected in codes(result)
    assert all(not b["normalization"]["records"] for b in result["batches"])


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("short", "horizon_unverified"),
        ("missing_render", "render_incomplete"),
        ("black", "black_frame_observed"),
        ("pose", "invalid_pose"),
        ("map", "map_unverified"),
        ("time", "trace_timestamp_order"),
        ("frame", "unsupported_coordinate_frame"),
        ("identity", "trace_identity_mismatch"),
    ],
)
def test_execution_trace_does_not_guess_validity(tmp_path, mutation, expected):
    request, _, trace, _ = fixture_bundle(tmp_path)
    if mutation == "short":
        trace["samples"].pop()
    elif mutation == "missing_render":
        trace["samples"][1]["render_status"] = "missing"
    elif mutation == "black":
        trace["samples"][1]["black_frame"] = True
    elif mutation == "pose":
        trace["samples"][1]["ego_xyz_m"][0] = None
    elif mutation == "map":
        trace["samples"][1]["map_available"] = False
    elif mutation == "time":
        trace["samples"][1]["timestamp_us"] = 0
    elif mutation == "frame":
        trace["coordinate_frame"] = "unknown"
    elif mutation == "identity":
        trace["upstream_identity"] = {**trace["upstream_identity"], "rollout_id": "other"}
    request["artifacts"][0]["trace"] = json_receipt(tmp_path, "trace.json", trace)
    result = extract_bundle(tmp_path, request)
    assert not result["admitted"] and expected in codes(result)


def test_crash_before_uuid_remains_in_denominator(tmp_path):
    request, events, _, _ = fixture_bundle(tmp_path)
    events[0]["upstream_identity"] = None
    events[1].update(type="failed", reason="bootstrap_failure", upstream_identity=None)
    request["events"] = json_receipt(tmp_path, "events.json", events)
    request["artifacts"] = []
    result = extract_bundle(tmp_path, request)
    assert not result["admitted"]
    assert result["ledger"]["counts"]["scheduled_jobs"] == 1
    assert result["ledger"]["counts"]["failed_attempts"] == 1


def test_file_tampering_blocks_before_parsing(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    (tmp_path / "metrics.parquet").write_bytes(b"corrupt")
    result = extract_bundle(tmp_path, request)
    assert not result["admitted"] and "artifact_integrity" in codes(result)


def test_torn_event_json_blocks(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    from s2_ingestion_fixture import receipt

    request["events"] = receipt(tmp_path, "events.json", b'[{"event_id":')
    result = extract_bundle(tmp_path, request)
    assert not result["admitted"] and "invalid_events" in codes(result)


def test_exact_four_meters_remains_out_of_envelope(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    rows = metric_rows()
    for row in rows:
        if row["name"] == "dist_to_gt_trajectory" and row["timestamps_us"] == 300_000:
            row["values"] = 4.0
    request["artifacts"][0]["metrics"] = parquet_receipt(tmp_path, rows)
    result = extract_bundle(tmp_path, request)
    assert result["admitted"]
    assert result["batches"][0]["normalization"]["counts"]["out_of_envelope"] == 1


def test_collision_summary_cutoff_does_not_hide_later_deviation(tmp_path):
    request, _, _, summary = fixture_bundle(tmp_path)
    rows = metric_rows()
    for row in rows:
        if row["name"] == "collision_front" and row["timestamps_us"] >= 200_000:
            row["values"] = 1.0
    scored = summary["rollouts"][0]
    scored.update(status="fail", passed=False, score=0.0, failure_reason="collision_at_fault")
    scored["metrics"]["collision_at_fault"] = 1.0
    scored["score_metrics"]["collision_at_fault"] = 1.0
    request["batches"][0]["summary"] = json_receipt(tmp_path, "summary.json", summary)
    request["artifacts"][0]["metrics"] = parquet_receipt(tmp_path, rows)
    result = extract_bundle(tmp_path, request)
    assert result["admitted"], result["issues"]
    record = result["batches"][0]["normalization"]["records"][0]
    assert record["value"] == 0.0
    assert record["deviation_m"] == 4.2


@pytest.mark.parametrize(
    "metric,value",
    [
        ("collision_front", -1.0),
        ("collision_lateral", 0.5),
        ("offroad", 2.0),
        ("left_corridor_laterally", -1.0),
        ("gt_dist_traveled_m", -2.0),
        ("lateral_dist_to_gt_trajectory", -1.0),
        ("dist_to_gt_trajectory", -1.0),
    ],
)
def test_source_metric_domains_reject_corrupt_values(tmp_path, metric, value):
    request, _, _, _ = fixture_bundle(tmp_path)
    rows = metric_rows()
    for row in rows:
        if row["name"] == metric:
            row["values"] = value
    request["artifacts"][0]["metrics"] = parquet_receipt(tmp_path, rows)
    result = extract_bundle(tmp_path, request)
    assert not result["admitted"] and "invalid_metric_sample" in codes(result)


@pytest.mark.parametrize("payload", [None, [], {}, {"schema_version": True}])
def test_malformed_request_is_structured_blocker(tmp_path, payload):
    result = extract_bundle(tmp_path, payload)
    assert not result["admitted"] and result["issues"]


def test_real_capture_is_not_claimed_supported(tmp_path):
    request, _, _, _ = fixture_bundle(tmp_path)
    request["evidence_kind"] = "alpasim_runtime"
    result = extract_bundle(tmp_path, request)
    assert not result["admitted"] and "unsupported_evidence_kind" in codes(result)


# --- R4 C2 binding rule, unit level (agent-authored, D-024) --------------------------------


def _attempt(job_id, attempt_id, clip, rollout, run="run"):
    identity = {"run_uuid": run, "clipgt_id": clip, "rollout_id": rollout}
    return {"job_id": job_id, "attempt_id": attempt_id, "upstream_identity": identity}


def test_binding_issues_literal_known_answers():
    from gate.s2_extract import _binding_issues

    jobs = {"j0": {"scene_id": "s0"}, "j1": {"scene_id": "s1"}}
    attempts = [
        _attempt("j0", "a0", "s0", "r0"),
        _attempt("j1", "a1", "s0", "r1"),  # ran scene s0 although scheduled for s1
        _attempt("j0", "a2", "s0", "r0", run="other-run"),  # retry reusing a0's session
        {"job_id": "j1", "attempt_id": "a3", "upstream_identity": None},  # pre-UUID failure
        _attempt("j1", "a4", "s1", "r4"),
    ]
    found = _binding_issues(attempts, jobs)
    assert [(code, path) for code, path, _ in found] == [
        ("scene_clip_mismatch", "a1"),
        ("session_reused", "a2"),
    ]
    assert "'j0'" in found[1][2] and "'a0'" in found[1][2]
    assert _binding_issues(attempts[:1] + attempts[3:], jobs) == []
