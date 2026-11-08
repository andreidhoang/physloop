"""Receipt/attempt binding controls; parser behavior has separate wire fixtures."""

import copy

import pytest
from s2_asl_fixture import asl_bytes
from s2_capture_fixture import wire_capture_request
from s2_ingestion_fixture import IDENTITY, SHA, fixture_bundle, json_receipt, receipt

from gate.s2_capture import inspect_capture


def capture_fixture(root):
    source, events, _, _ = fixture_bundle(root)
    source["schedule"][0]["scene_id"] = "clip"
    return {
        "schema_version": 1,
        "evidence_kind": "alpasim_asl_inspection_v1",
        "capture_origin": "synthetic_fixture",
        "evaluator_sha": SHA,
        "schedule": source["schedule"],
        "schedule_request": None,
        "events": source["events"],
        "artifacts": [{"attempt_id": "attempt", "asl": receipt(root, "capture.asl", b"wire")}],
    }, events


@pytest.fixture
def decoder(monkeypatch):
    calls = []

    def decode(data, **kwargs):
        calls.append((data, kwargs))
        return {"parsed": True, "inspection_valid": True, "issues": [], "runtime_qualified": False,
                "actual_steps": None, "completion_status": "not_established"}

    monkeypatch.setattr("gate.s2_capture.inspect_asl", decode)
    return calls


def test_verified_receipts_bind_rollout_and_scene_without_admitting_scores(tmp_path, decoder):
    request, _ = capture_fixture(tmp_path)
    result = inspect_capture(tmp_path, request)
    assert result["capture_inspection_passed"]
    assert result["ledger"]["admitted"]
    assert decoder[0][0] == b"wire"
    assert decoder[0][1]["expected_session_uuid"] == IDENTITY["rollout_id"]
    assert decoder[0][1]["expected_scene_id"] == IDENTITY["clipgt_id"]
    assert decoder[0][1]["requested_steps"] == 3
    assert result["captures"][0]["inspection"]["actual_steps"] is None
    assert not result["score_admitted"]
    assert not result["runtime_qualified"]
    assert not result["cloud_launch_authorized"]
    assert result["provenance_authenticated"] is False
    assert result["request"] == request
    assert result["request"] is not request
    assert len(result["request_sha256"]) == 64
    assert "records" not in result


@pytest.mark.parametrize("mutation,code", [
    ("missing", "missing_asl"),
    ("duplicate", "duplicate_attempt_artifact"),
    ("unknown", "unknown_attempt_artifact"),
    ("corrupt", "artifact_unverified"),
    ("scene", "scene_identity_mismatch"),
])
def test_bad_bindings_block_capture(tmp_path, decoder, mutation, code):
    request, _ = capture_fixture(tmp_path)
    if mutation == "missing":
        request["artifacts"] = []
    elif mutation == "duplicate":
        request["artifacts"].append(copy.deepcopy(request["artifacts"][0]))
    elif mutation == "unknown":
        request["artifacts"][0]["attempt_id"] = "other"
    elif mutation == "corrupt":
        (tmp_path / "capture.asl").write_bytes(b"changed")
    else:
        request["schedule"][0]["scene_id"] = "wrong"
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert code in {issue["code"] for issue in result["issues"]}
    assert not result["score_admitted"]


def test_pre_uuid_failure_remains_visible_without_inventing_a_log(tmp_path, decoder):
    request, events = capture_fixture(tmp_path)
    events[0]["upstream_identity"] = None
    events[1].update(type="failed", reason="driver_start_failed", upstream_identity=None)
    request["events"] = json_receipt(tmp_path, "failure-events.json", events)
    request["artifacts"] = []
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert result["ledger"]["counts"]["failed_attempts"] == 1
    assert result["captures"][0]["status"] == "no_upstream_identity"
    assert not decoder


def test_retry_keeps_both_logs_and_ledger_blocks_selection(tmp_path, decoder):
    request, events = capture_fixture(tmp_path)
    events[1].update(type="failed", reason="service_error")
    new_identity = {**IDENTITY, "rollout_id": "retry-session"}
    events.extend([
        {**events[0], "event_id": "retry-start", "attempt_id": "retry", "timestamp_us": 3,
         "parent_attempt_id": "attempt", "retry_reason": "service_error",
         "upstream_identity": new_identity},
        {**events[1], "event_id": "retry-end", "attempt_id": "retry", "timestamp_us": 4,
         "type": "completed", "reason": "exact_horizon", "upstream_identity": new_identity},
    ])
    request["events"] = json_receipt(tmp_path, "retry-events.json", events)
    request["artifacts"].append({"attempt_id": "retry",
                                 "asl": receipt(tmp_path, "retry.asl", b"retry wire")})
    result = inspect_capture(tmp_path, request)
    assert len(result["captures"]) == len(decoder) == 2
    assert not result["capture_inspection_passed"]
    assert "retry_selection_unregistered" in {x["code"] for x in result["ledger"]["issues"]}
    assert result["ledger"]["jobs"][0]["selected_attempt_id"] is None


def test_parser_failure_does_not_erase_other_results(tmp_path, monkeypatch):
    request, _ = capture_fixture(tmp_path)
    monkeypatch.setattr("gate.s2_capture.inspect_asl", lambda *a, **k: {
        "parsed": False, "inspection_valid": False,
        "issues": [{"code": "truncated_frame", "message": "torn"}],
    })
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert len(result["ledger"]["events"]) == 2
    assert result["captures"][0]["inspection"]["issues"][0]["code"] == "truncated_frame"


def test_parsed_but_wrong_session_is_not_a_passing_inspection(tmp_path, monkeypatch):
    request, _ = capture_fixture(tmp_path)
    monkeypatch.setattr("gate.s2_capture.inspect_asl", lambda *a, **k: {
        "parsed": True, "inspection_valid": False,
        "issues": [{"code": "session_identity_mismatch", "message": "other session"}],
    })
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert result["captures"][0]["inspection"]["parsed"]


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("evidence_kind", "synthetic_source_contract_v1"),
    ("evaluator_sha", "a" * 40), ("capture_origin", "authenticated_runtime"),
    ("artifacts", {}), ("schedule_request", {}),
])
def test_malformed_request_is_rejected(tmp_path, decoder, key, value):
    request, _ = capture_fixture(tmp_path)
    request[key] = value
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert result["issues"]


def test_runtime_origin_claim_does_not_authenticate_producer(tmp_path, decoder):
    request, _ = capture_fixture(tmp_path)
    request["capture_origin"] = "runtime_claim_unverified"
    result = inspect_capture(tmp_path, request)
    assert result["capture_inspection_passed"]
    assert result["provenance_authenticated"] is False
    assert result["runtime_qualified"] is False


def test_original_request_is_not_mutated(tmp_path, decoder):
    request, _ = capture_fixture(tmp_path)
    before = copy.deepcopy(request)
    inspect_capture(tmp_path, request)
    assert request == before


def test_unknown_and_duplicate_receipts_are_retained_for_audit(tmp_path, decoder):
    request, _ = capture_fixture(tmp_path)
    request["artifacts"].extend([
        copy.deepcopy(request["artifacts"][0]),
        {"attempt_id": "unknown", "asl": {"path": "unknown.asl"}},
    ])
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert result["request"]["artifacts"] == request["artifacts"]
    assert result["request"]["events"] == request["events"]


def test_oversized_receipt_is_rejected_before_reading_asl(tmp_path, decoder, monkeypatch):
    request, _ = capture_fixture(tmp_path)
    request["artifacts"][0]["asl"]["size_bytes"] = 256 * 1024 * 1024 + 1
    import gate.s2_capture as capture

    original = capture.read_artifact
    reads = []

    def tracked_read(root, artifact):
        reads.append(artifact["path"])
        return original(root, artifact)

    monkeypatch.setattr(capture, "read_artifact", tracked_read)
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert reads == ["events.json"]
    assert "byte limit" in result["issues"][0]["message"]
    assert not decoder


def test_actual_wire_decoder_through_receipts_retains_intermediate_pose(tmp_path):
    request = wire_capture_request(tmp_path)
    report = inspect_capture(tmp_path, request)
    assert report["capture_inspection_passed"], report
    inspection = report["captures"][0]["inspection"]
    assert inspection["observations"]["pose_event_count"] == 4
    assert inspection["observations"]["first_timestamp_us"] == 1000
    assert inspection["observations"]["last_timestamp_us"] == 1200
    assert inspection["actual_steps"] is None
    assert not report["score_admitted"]


@pytest.mark.parametrize("failure", ["session", "scene", "torn"])
def test_real_decoder_rejects_wrong_identity_and_torn_log(tmp_path, failure):
    request = wire_capture_request(tmp_path)
    data = asl_bytes(session_uuid="other" if failure == "session" else "rollout",
                     scene_id="other" if failure == "scene" else "clip", n_steps=3)
    if failure == "torn":
        data = data[:-1]
    request["artifacts"][0]["asl"] = receipt(tmp_path, "bad.asl", data)
    report = inspect_capture(tmp_path, request)
    assert not report["capture_inspection_passed"]
    assert report["captures"][0]["inspection"]["issues"]
    assert report["ledger"]["counts"]["attempts"] == 1


def test_external_run_id_cannot_disambiguate_reused_asl_session(tmp_path):
    request = wire_capture_request(tmp_path)
    from gate.s2_identity import read_artifact, read_json_bytes

    events = read_json_bytes(read_artifact(tmp_path, request["events"]))
    request["schedule"].append({
        **request["schedule"][0], "job_id": "job-b", "arm": "B", "policy_id": "reexport",
    })
    second_events = copy.deepcopy(events)
    for event in second_events:
        event.update(event_id=event["event_id"] + "-b", job_id="job-b", attempt_id="attempt-b",
                     timestamp_us=event["timestamp_us"] + 2)
        event["upstream_identity"]["run_uuid"] = "other-run"
    request["events"] = json_receipt(tmp_path, "all-events.json", events + second_events)
    data = read_artifact(tmp_path, request["artifacts"][0]["asl"])
    request["artifacts"].append({
        "attempt_id": "attempt-b", "asl": receipt(tmp_path, "copied.asl", data),
    })

    report = inspect_capture(tmp_path, request)

    # The general evaluator ledger's full identities differ legitimately. ASL
    # contains only scene/session, so the separate capture binding is ambiguous.
    assert report["ledger"]["admitted"]
    assert not report["capture_inspection_passed"]
    assert "reused_asl_identity" in {issue["code"] for issue in report["issues"]}
    assert [capture["attempt_id"] for capture in report["captures"]] == ["attempt", "attempt-b"]
    assert all(capture["inspection"]["inspection_valid"] for capture in report["captures"])
    assert report["request"]["artifacts"] == request["artifacts"]


def test_explicit_capture_byte_limit_reaches_parser(tmp_path):
    request = wire_capture_request(tmp_path)
    exact_size = request["artifacts"][0]["asl"]["size_bytes"]
    report = inspect_capture(tmp_path, request, max_asl_bytes=exact_size)
    assert report["capture_inspection_passed"]
    assert report["max_asl_bytes"] == exact_size


def test_capture_override_rejects_larger_receipt_before_reading(tmp_path, monkeypatch):
    request = wire_capture_request(tmp_path)
    import gate.s2_capture as capture

    original = capture.read_artifact
    reads = []

    def tracked_read(root, artifact):
        reads.append(artifact["path"])
        return original(root, artifact)

    monkeypatch.setattr(capture, "read_artifact", tracked_read)
    maximum = request["artifacts"][0]["asl"]["size_bytes"] - 1
    report = inspect_capture(tmp_path, request, max_asl_bytes=maximum)
    assert not report["capture_inspection_passed"]
    assert reads == ["events.json"]
    assert "byte limit" in report["issues"][0]["message"]


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1.0, "1024"])
def test_invalid_capture_byte_limits_fail_before_artifact_reads(tmp_path, monkeypatch, limit):
    request = wire_capture_request(tmp_path)

    def unexpected_read(*args):
        pytest.fail("Invalid size limit must be rejected before artifact I/O")

    monkeypatch.setattr("gate.s2_capture.read_artifact", unexpected_read)
    report = inspect_capture(tmp_path, request, max_asl_bytes=limit)
    assert not report["capture_inspection_passed"]
    assert "max_asl_bytes" in report["issues"][0]["message"]
