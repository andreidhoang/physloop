"""Receipt/attempt binding controls; parser behavior has separate wire fixtures."""

import ast
import copy
import logging
import re
from pathlib import Path

import pytest
from s2_asl_fixture import asl_bytes
from s2_capture_fixture import wire_capture_request
from s2_ingestion_fixture import IDENTITY, SHA, fixture_bundle, json_receipt, receipt

from gate.s2_capture import (
    bind_run_uuid,
    inspect_capture,
    parse_runtime_log,
    reconcile_unit_attempts,
    scan_rollout_files,
)


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

    def decode(stream, **kwargs):
        data = b""
        while chunk := stream.read(1 << 20):
            data += chunk
        calls.append((data, kwargs))
        return {"parsed": True, "inspection_valid": True, "issues": [], "runtime_qualified": False,
                "actual_steps": None, "completion_status": "not_established"}

    monkeypatch.setattr("gate.s2_capture.inspect_asl_stream", decode)
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
    assert decoder[0][1]["max_file_bytes"] == 2 * 1024 * 1024 * 1024
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
    monkeypatch.setattr("gate.s2_capture.inspect_asl_stream", lambda *a, **k: {
        "parsed": False, "inspection_valid": False,
        "issues": [{"code": "truncated_frame", "message": "torn"}],
    })
    result = inspect_capture(tmp_path, request)
    assert not result["capture_inspection_passed"]
    assert len(result["ledger"]["events"]) == 2
    assert result["captures"][0]["inspection"]["issues"][0]["code"] == "truncated_frame"


def test_parsed_but_wrong_session_is_not_a_passing_inspection(tmp_path, monkeypatch):
    request, _ = capture_fixture(tmp_path)
    monkeypatch.setattr("gate.s2_capture.inspect_asl_stream", lambda *a, **k: {
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
    request["artifacts"][0]["asl"]["size_bytes"] = 2 * 1024 * 1024 * 1024 + 1
    import gate.s2_capture as capture

    original = capture.read_artifact
    reads = []

    def tracked_read(root, artifact):
        reads.append(artifact["path"])
        return original(root, artifact)

    monkeypatch.setattr(capture, "read_artifact", tracked_read)
    monkeypatch.setattr(capture, "_open_receipted", lambda *a: pytest.fail("ASL was opened"))
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
    monkeypatch.setattr(capture, "_open_receipted", lambda *a: pytest.fail("ASL was opened"))
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


# --- A4: streaming hash verification (C1/A1) --------------------------------------------


def test_tail_tamper_after_receipt_blocks_even_when_frames_parse(tmp_path):
    request = wire_capture_request(tmp_path)
    path = tmp_path / "capture.asl"
    data = bytearray(path.read_bytes())
    # Flip one byte inside the last float of the last pose: framing still parses.
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    report = inspect_capture(tmp_path, request)
    assert not report["capture_inspection_passed"]
    assert report["captures"][0]["status"] == "artifact_unverified"
    assert report["captures"][0]["inspection"] is None
    assert "checksum" in report["issues"][-1]["message"]


def test_capture_streams_with_bounded_reads(tmp_path, monkeypatch):
    import gate.s2_capture as capture

    big = asl_bytes(session_uuid="rollout", scene_id="clip", n_steps=3) + capture_frame_padding()
    request = wire_capture_request(tmp_path)
    request["artifacts"][0]["asl"] = receipt(tmp_path, "big.asl", big)
    sizes = []
    original = capture.os.fdopen

    class Recording:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size):
            sizes.append(size)
            return self.handle.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

    monkeypatch.setattr(capture.os, "fdopen", lambda *a, **k: Recording(original(*a, **k)))
    report = inspect_capture(tmp_path, request)
    assert report["capture_inspection_passed"], report["issues"]
    assert len(big) > 3 * 1024 * 1024
    assert sizes and max(sizes) <= 1024 * 1024


def capture_frame_padding():
    from s2_asl_fixture import field, framed, text_field

    return framed(field(25, text_field(1, "x" * (3 * 1024 * 1024))))


def test_symlinked_asl_is_rejected_without_following(tmp_path):
    request = wire_capture_request(tmp_path)
    (tmp_path / "link.asl").symlink_to(tmp_path / "capture.asl")
    request["artifacts"][0]["asl"] = {**request["artifacts"][0]["asl"], "path": "link.asl"}
    report = inspect_capture(tmp_path, request)
    assert not report["capture_inspection_passed"]
    assert report["captures"][0]["status"] == "artifact_unverified"


def test_default_capture_bound_is_the_documented_provisional_agent_default():
    import gate.s2_capture as capture

    assert capture.MAX_ASL_BYTES == 2 * 1024 * 1024 * 1024
    assert "agent technical default (D-018)" in capture.__doc__
    assert "provisional until the first measured ASL" in capture.__doc__


def test_effective_steps_warning_is_surfaced_without_blocking(tmp_path):
    request = wire_capture_request(tmp_path)
    data = asl_bytes(session_uuid="rollout", scene_id="clip", n_steps=2)
    request["artifacts"][0]["asl"] = receipt(tmp_path, "short.asl", data)
    report = inspect_capture(tmp_path, request)
    assert report["capture_inspection_passed"], report["issues"]
    warnings = [item for item in report["issues"] if item["severity"] == "warning"]
    assert [item["code"] for item in warnings] == ["effective_steps_below_requested"]
    assert all(set(item) == {"code", "path", "severity", "evidence_refs", "message"}
               for item in report["issues"])


# --- A4: C2 runtime log parsing against pinned formats ----------------------------------

UPSTREAM = Path(__file__).parent / "fixtures" / "alpasim_launch_upstream" / "alpasim"
ASL_SOURCES = Path(__file__).parent / "fixtures" / "alpasim_asl"
UUID_A = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
UUID_B = "6fa459ea-ee8a-3ca4-894e-db77e160355e"
JOB_A = "0" * 31 + "a"
JOB_B = "0" * 31 + "b"


def pinned_messages():
    """Message format strings exactly as the pinned sources pass them to logging."""
    found = {}
    for path in (ASL_SOURCES / "event_loop_e121e591.py.txt",
                 UPSTREAM / "src/runtime/alpasim_runtime/worker/main.py.txt",
                 UPSTREAM / "src/runtime/alpasim_runtime/daemon/scheduler.py.txt"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for marker in ("Session STARTING", "Session COMPLETED", "Rollout FAILED",
                               "Retrying rollout"):
                    if node.value.startswith(marker):
                        found[marker] = node.value
    return found


def pinned_formatter(worker):
    """Build the pinned worker or scheduler logging.Formatter from source text."""
    if worker:
        source = (UPSTREAM / "src/runtime/alpasim_runtime/worker/main.py.txt").read_text()
        fmt = re.search(r'f"(%\(asctime\)s[^"]+)"', source)[1].replace("{args.worker_id}", "0")
    else:
        source = (UPSTREAM / "src/runtime/alpasim_runtime/simulate/__main__.py.txt").read_text()
        fmt = re.search(r'format="(%\(asctime\)s[^"]+)"', source)[1]
    assert 'datefmt="%H:%M:%S"' in source
    formatter = logging.Formatter(fmt.replace("\\t", "\t"), datefmt="%H:%M:%S")
    formatter.converter = __import__("time").gmtime
    return formatter


def render(message, args, *, worker=True, level=logging.INFO):
    record = logging.LogRecord("alpasim_runtime", level, __file__, 1, message, args, None)
    record.created, record.msecs = 36000.0, 123.0
    return pinned_formatter(worker).format(record)


def test_pinned_message_formats_and_line_numbers():
    messages = pinned_messages()
    assert messages == {
        "Session STARTING": "Session STARTING: uuid=%s scene=%s steps=%d",
        "Session COMPLETED": (
            "Session COMPLETED: uuid=%s scene=%s simulated %.2f sim seconds in %.2f wall "
            "clock seconds for %.2fx real time (total rollout %.2fs incl. setup/warmup)"),
        "Rollout FAILED": "Rollout FAILED: job=%s scene=%s uuid=%s retryable=%s error=%s\n%s",
        "Retrying rollout": (
            "Retrying rollout after retryable infrastructure failure: scene=%s failed_job=%s "
            "retry_job=%s attempt=%d/%d error=%s"),
    }
    event_loop = (ASL_SOURCES / "event_loop_e121e591.py.txt").read_text().splitlines()
    worker = (UPSTREAM / "src/runtime/alpasim_runtime/worker/main.py.txt").read_text()
    scheduler = (UPSTREAM / "src/runtime/alpasim_runtime/daemon/scheduler.py.txt").read_text()
    assert "Session STARTING" in event_loop[623]
    assert "Session COMPLETED" in event_loop[688]
    assert "Rollout FAILED" in worker.splitlines()[168]
    assert "Retrying rollout" in scheduler.splitlines()[203]
    assert "job_id=uuid4().hex" in scheduler
    unbound = (ASL_SOURCES / "unbound_rollout_e121e591.py.txt").read_text()
    assert "rollout_uuid=session_uuid or str(uuid.uuid1())" in unbound


def rendered_log():
    messages = pinned_messages()
    started_a = render(messages["Session STARTING"], (UUID_A, "clip-1", 200))
    failed_a = render(messages["Rollout FAILED"], (
        JOB_A, "clip-1", UUID_A, True, "renderer died",
        "Traceback (most recent call last):\n  raise RetryableRendererInfrastructureError"),
        level=logging.WARNING)
    retry = render(messages["Retrying rollout"], (
        "clip-1", JOB_A, JOB_B, 1, 1, "renderer died"), worker=False, level=logging.WARNING)
    started_b = render(messages["Session STARTING"], (UUID_B, "clip-1", 200))
    completed_b = render(messages["Session COMPLETED"], (UUID_B, "clip-1", 20.0, 31.5,
                                                          0.63, 60.25))
    return [started_a, failed_a, retry, started_b, completed_b]


def compose(lines, service="runtime-0-1"):
    out = []
    for line in lines:
        for part in line.split("\n"):
            out.append(f"{service}  | {part}")
    return "\n".join(out)


def test_pinned_formatter_lines_parse_to_literal_events():
    text = compose(rendered_log())
    assert text.splitlines()[0] == (
        f"runtime-0-1  | 10:00:00.123 [W0] INFO:\tSession STARTING: uuid={UUID_A} "
        "scene=clip-1 steps=200")
    parsed = parse_runtime_log(text, source="compose_up_stdout")
    assert parsed["issues"] == []
    assert [(event["kind"], event["line"]) for event in parsed["events"]] == [
        ("session_started", 1), ("rollout_failed", 2), ("retry_scheduled", 5),
        ("session_started", 6), ("session_completed", 7)]
    failed = parsed["events"][1]
    assert failed["uuid"] == UUID_A and failed["job"] == JOB_A and failed["retryable"] is True
    assert failed["error_first_line"] == "renderer died"
    assert failed["worker"] == 0
    retry = parsed["events"][2]
    assert retry["worker"] is None
    assert (retry["failed_job"], retry["retry_job"], retry["attempt"], retry["max_attempts"]) == (
        JOB_A, JOB_B, 1, 1)
    assert parsed["events"][0]["steps"] == 200


def test_compose_logs_timestamps_and_worker_log_formats_parse():
    line = rendered_log()[0]
    with_ts = f"runtime-0  | 2026-11-08T10:00:00.123456789Z {line}"
    assert [e["kind"] for e in parse_runtime_log(with_ts, source="compose_logs")["events"]] == [
        "session_started"]
    assert [e["kind"] for e in parse_runtime_log(line, source="runtime_worker_log")["events"]] == [
        "session_started"]
    # A worker log has no Compose prefix; a prefixed line there is not a worker line.
    assert parse_runtime_log(with_ts, source="runtime_worker_log")["events"] == []


def test_other_services_and_unprefixed_lines_never_count():
    line = rendered_log()[0]
    text = "\n".join([f"renderer-0-1  | {line}", line, "Container runtime-0-1  Started"])
    parsed = parse_runtime_log(text, source="compose_up_stdout")
    assert parsed["events"] == []
    assert parsed["other_service_lines"] == 1
    assert parsed["runtime_service_lines"] == 0
    # Review A4-F1: marker lines outside the runtime prefix block instead of vanishing.
    assert [(i["code"], i["path"]) for i in parsed["issues"]] == [
        ("unparsed_marker_line", "line:1"), ("unparsed_marker_line", "line:2")]


def test_pre_uuid_failure_line_parses_with_na_identity():
    line = render(pinned_messages()["Rollout FAILED"], (
        JOB_A, "N/A", "N/A", False, "no artifact", "tb"), level=logging.WARNING)
    events = parse_runtime_log(compose([line]), source="compose_up_stdout")["events"]
    assert events[0]["uuid"] == "N/A" and events[0]["scene"] == "N/A"
    assert events[0]["retryable"] is False


@pytest.mark.parametrize("message", [
    "Session STARTING: uuid=not-a-uuid scene=clip-1 steps=200",
    f"Session STARTING: uuid={UUID_A} scene=clip-1 steps=-1",
    f"Rollout FAILED: job=short scene=clip-1 uuid={UUID_A} retryable=True error=x",
    f"Rollout FAILED: job={JOB_A} scene=clip-1 uuid={UUID_A} retryable=maybe error=x",
])
def test_marker_lines_that_do_not_match_pinned_formats_block(message):
    line = f"runtime-0-1  | 10:00:00.123 [W0] INFO:\t{message}"
    parsed = parse_runtime_log(line, source="compose_up_stdout")
    assert parsed["events"] == []
    assert [item["code"] for item in parsed["issues"]] == ["unparsed_marker_line"]


@pytest.mark.parametrize("source,text", [("stdout", ""), ("compose_logs", b"bytes")])
def test_invalid_log_inputs_fail_closed(source, text):
    parsed = parse_runtime_log(text, source=source)
    assert parsed["events"] == [] and parsed["issues"]


# --- A4: P2 run_uuid binding ------------------------------------------------------------

RUN_UUID = "9b2f3c1e-7d4a-4f6b-8c2d-1e3f5a7b9c0d"
RUN_METADATA = (
    "description: null\nrun_args: unknownArgs\nrun_dir: /synthetic\n"
    "run_name: e2e_challenge_dev\nrun_time: '2026-11-08 10:00:00'\n"
    f"run_user: synthetic\nrun_uuid: {RUN_UUID}\nslurm_job_id: 0\nsubmitter: null\n"
    "test_suite_id: null\n")


def test_run_uuid_binding_known_answer_and_pinned_writer():
    assert bind_run_uuid(RUN_METADATA.encode()) == {
        "status": "passed", "run_uuid": RUN_UUID, "issues": [],
        "source": "<RUN_DIR>/run_metadata.yaml#run_uuid"}
    config = (UPSTREAM / "src/wizard/alpasim_wizard/configuration.py.txt").read_text()
    lines = config.splitlines()
    assert "run_uuid = uuid.uuid4()" in lines[275]
    assert '"run_uuid": str(run_uuid),' in lines[285]
    assert 'self._write_config("run_metadata.yaml", run_metadata)' in config


@pytest.mark.parametrize("text", [
    "", RUN_METADATA + f"run_uuid: {RUN_UUID}\n", RUN_METADATA.replace("run_uuid:", "  run_uuid:"),
    RUN_METADATA.upper(), RUN_METADATA.replace(RUN_UUID, "null"),
])
def test_run_uuid_binding_negative_controls(text):
    result = bind_run_uuid(text.encode())
    assert result["status"] == "blocked" and result["run_uuid"] is None


# --- A4: rollout folder scan ------------------------------------------------------------


def test_rollout_scan_groups_pinned_paths():
    paths = [
        f"rollouts/clip-1/{UUID_A}/rollout.asl",
        f"rollouts/clip-1/{UUID_B}/_complete", f"rollouts/clip-1/{UUID_B}/metrics.parquet",
        f"rollouts/clip-1/{UUID_B}/rollout.asl", f"rollouts/clip-1/{UUID_B}/clip-1_x_cam_0.mp4",
        "aggregate/results-summary.json", "run_metadata.yaml",
    ]
    report = scan_rollout_files(paths)
    assert report["issues"] == []
    assert report["folders"] == [
        {"clipgt_id": "clip-1", "rollout_uuid": UUID_A, "complete_marker": False,
         "metrics": False, "asl": True, "other_files": []},
        {"clipgt_id": "clip-1", "rollout_uuid": UUID_B, "complete_marker": True,
         "metrics": True, "asl": True, "other_files": ["clip-1_x_cam_0.mp4"]},
    ]


@pytest.mark.parametrize("path", ["rollouts/clip-1/not-a-uuid/_complete",
                                  f"rollouts/../{UUID_A}/_complete", "rollouts/clip-1"])
def test_rollout_scan_rejects_unrecognized_paths(path):
    assert scan_rollout_files([path])["issues"]


# --- A4: C3 unit reconciliation ---------------------------------------------------------


def scored_row(uuid, status="pass", run_uuid=RUN_UUID):
    return {"run_uuid": run_uuid, "run_name": "e2e_challenge_dev", "clipgt_id": "clip-1",
            "rollout_id": uuid, "status": status, "passed": status == "pass",
            "score": 0.8 if status == "pass" else 0.0,
            "failure_reason": None if status == "pass" else "collision_at_fault",
            "metrics": {"run_uuid": run_uuid, "clipgt_id": "clip-1", "rollout_id": uuid,
                        "collision_at_fault": 0.0 if status == "pass" else 1.0,
                        "progress_rel": 0.5},
            "score_metrics": {}}


def failure_row(rollout_id):
    # Literal shape of failed_rollouts.py:48-64.
    metrics = {"run_name": "e2e_challenge_dev", "run_uuid": None, "clipgt_id": "clip-1",
               "rollout_id": rollout_id, "error": "boom"}
    return {"run_uuid": None, "run_name": "e2e_challenge_dev", "clipgt_id": "clip-1",
            "rollout_id": rollout_id, "status": "fail", "passed": False,
            "failure_reason": "boom", "metrics": metrics, "score": 0.0,
            "score_metrics": {}}


def unit(lines, files, rows, *, exit_code=0, metadata=RUN_METADATA, steps=200):
    return {
        "scope": "synthetic", "scene_id": "clip-1", "requested_steps": steps,
        "logs": [{"source": "compose_up_stdout", "text": compose(lines)}],
        "rollout_files": files, "run_metadata": metadata,
        "summary": None if rows is None else {"schema_version": 1, "rollouts": rows},
        "compose_exit_code": exit_code,
    }


def complete_files(uuid):
    return [f"rollouts/clip-1/{uuid}/{name}" for name in ("_complete", "metrics.parquet",
                                                          "rollout.asl")]


def test_single_complete_attempt_ignores_nonzero_exit_and_keeps_scored_failure():
    lines = rendered_log()[3:]
    # A scored "fail" row (collision) is a completed rollout, not a runtime failure.
    report = reconcile_unit_attempts(unit(lines, complete_files(UUID_B),
                                          [scored_row(UUID_B, "fail")], exit_code=1))
    assert report["status"] == "passed", report["issues"]
    assert report["unit_outcome"] == "complete" and report["scope"] == "synthetic"
    assert report["selected_attempt"] == UUID_B
    assert report["attempts"][0]["state"] == "complete"
    assert report["attempts"][0]["summary_rows"][0]["failure_reason"] == "collision_at_fault"
    assert report["compose_exit_code"] == {"value": 1, "used_for_success": False}
    assert report["run_uuid_binding"]["run_uuid"] == RUN_UUID
    assert report["issues"] == []


def test_failed_rollout_with_exit_zero_is_failed_not_complete():
    lines = rendered_log()[:2]
    failed = render(pinned_messages()["Rollout FAILED"], (
        JOB_A, "clip-1", UUID_A, False, "Camera camera_rear_left_70fov not in desired cameras",
        "tb"), level=logging.WARNING)
    lines = [lines[0], failed]
    report = reconcile_unit_attempts(unit(lines, [f"rollouts/clip-1/{UUID_A}/rollout.asl"],
                                          [failure_row(UUID_A)], exit_code=0))
    assert report["status"] == "blocked" and report["unit_outcome"] == "failed"
    assert report["selected_attempt"] is None
    assert report["attempts"][0]["state"] == "failed"
    assert report["attempts"][0]["retryable"] is False
    codes_ = {item["code"]: item["severity"] for item in report["issues"]}
    assert codes_ == {"incomplete_rollout_folder": "info", "unit_not_complete": "blocking",
                      "exit_zero_without_completion": "info"}


def test_internal_retry_keeps_both_attempts_and_blocks_selection():
    files = [f"rollouts/clip-1/{UUID_A}/rollout.asl", *complete_files(UUID_B)]
    report = reconcile_unit_attempts(unit(rendered_log(), files, [scored_row(UUID_B)]))
    assert report["status"] == "blocked"
    assert [(a["rollout_uuid"], a["state"]) for a in report["attempts"]] == [
        (UUID_A, "failed"), (UUID_B, "complete")]
    assert report["internal_retries"][0]["retry_job"] == JOB_B
    assert "retry_selection_unregistered" in {item["code"] for item in report["issues"]}
    assert report["selected_attempt"] is None


def test_pre_uuid_crash_and_failed_row_are_visible():
    line = render(pinned_messages()["Rollout FAILED"], (
        JOB_A, "N/A", "N/A", False, "no artifact", "tb"), level=logging.WARNING)
    report = reconcile_unit_attempts(unit([line], [], [failure_row("failed-0")]))
    assert report["status"] == "blocked" and report["unit_outcome"] == "failed"
    assert [i["code"] for i in report["issues"]] == [
        "unit_not_complete", "exit_zero_without_completion"]
    assert [item["kind"] for item in report["pre_uuid_failures"]] == [
        "rollout_failed", "summary_failure_row"]
    assert report["attempts"] == []


@pytest.mark.parametrize("mutation,code", [
    ("scene", "scene_clip_mismatch"),
    ("run_uuid", "run_uuid_mismatch"),
    ("metadata", "run_metadata_missing"),
    ("duplicate_row", "duplicate_summary_rows"),
    ("contradiction", "contradictory_attempt_evidence"),
    ("no_logs", "runtime_logs_missing"),
    ("nothing", "no_upstream_attempt_evidence"),
    ("steps_above", "effective_steps_exceed_requested"),
])
def test_unit_negative_controls_block(mutation, code):
    lines, files, rows = rendered_log()[3:], complete_files(UUID_B), [scored_row(UUID_B)]
    evidence = unit(lines, files, rows)
    if mutation == "scene":
        evidence["scene_id"] = "clip-2"
    elif mutation == "run_uuid":
        evidence["summary"]["rollouts"] = [scored_row(UUID_B, run_uuid=UUID_A)]
    elif mutation == "metadata":
        evidence["run_metadata"] = None
    elif mutation == "duplicate_row":
        evidence["summary"]["rollouts"] = [scored_row(UUID_B), scored_row(UUID_B)]
    elif mutation == "contradiction":
        evidence["summary"]["rollouts"] = [failure_row(UUID_B)]
    elif mutation == "no_logs":
        evidence["logs"] = []
    elif mutation == "nothing":
        evidence.update(logs=[{"source": "compose_up_stdout", "text": ""}], rollout_files=[],
                        summary=None)
    else:
        evidence["requested_steps"] = 100
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked"
    assert code in {item["code"] for item in report["issues"]}
    assert report["selected_attempt"] is None


def test_effective_steps_below_requested_is_a_warning_on_a_complete_unit():
    report = reconcile_unit_attempts(unit(rendered_log()[3:], complete_files(UUID_B),
                                          [scored_row(UUID_B)], steps=250))
    assert report["status"] == "passed" and report["unit_outcome"] == "complete"
    assert [(i["code"], i["severity"]) for i in report["issues"]] == [
        ("effective_steps_below_requested", "warning")]


@pytest.mark.parametrize("missing", ["files", "row", "marker"])
def test_completion_needs_marker_metrics_and_summary_row(missing):
    files = complete_files(UUID_B)
    rows = [scored_row(UUID_B)]
    if missing == "files":
        files = [path for path in files if not path.endswith("metrics.parquet")]
    elif missing == "row":
        rows = []
    else:
        files = [path for path in files if not path.endswith("_complete")]
    report = reconcile_unit_attempts(unit(rendered_log()[3:], files, rows))
    assert report["status"] == "blocked" and report["unit_outcome"] == "incomplete"
    assert "unit_not_complete" in {item["code"] for item in report["issues"]}
    assert report["selected_attempt"] is None


@pytest.mark.parametrize("evidence", [None, [], {}, {"scene_id": "clip-1"}])
def test_malformed_unit_evidence_names_missing_fields(evidence):
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked"
    assert report["issues"][0]["code"] == "invalid_unit_evidence"


def test_failed_summary_row_shape_matches_pinned_source():
    source = (UPSTREAM / "src/eval/src/eval/aggregation/failed_rollouts.py.txt").read_text()
    lines = source.splitlines()
    assert lines[59].strip() == '"status": "fail",'
    for key in ("run_name", "run_uuid", "clipgt_id", "rollout_id", "error"):
        assert f'"{key}": ' in "\n".join(lines[47:54])
    processing = (UPSTREAM / "src/eval/src/eval/aggregation/processing.py.txt").read_text()
    assert '"status": "pass" if passed else "fail",' in processing.splitlines()[307]
    simulate = (UPSTREAM / "src/runtime/alpasim_runtime/simulate/__main__.py.txt").read_text()
    assert 'rollout_id=rollout_return.rollout_uuid or f"failed-{idx}",' in simulate
    assert "sys.exit(0 if success else 1)" in simulate.splitlines()[309]


def with_compose_logs(evidence, lines):
    """Add the same lines as `docker compose logs --timestamps` output (second capture)."""
    stamped = "\n".join(f"runtime-0  | 2026-11-08T10:00:00.000000000Z {part}"
                        for line in lines for part in line.split("\n"))
    evidence["logs"].append({"source": "compose_logs", "text": stamped})
    return evidence


def test_two_captures_of_one_complete_attempt_stay_complete():
    lines = rendered_log()[3:]
    evidence = with_compose_logs(unit(lines, complete_files(UUID_B), [scored_row(UUID_B)]), lines)
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "passed", report["issues"]
    assert len(report["attempts"][0]["evidence_refs"]) == 4


def test_two_captures_of_one_pre_uuid_failure_count_once():
    line = render(pinned_messages()["Rollout FAILED"], (
        JOB_A, "N/A", "N/A", False, "no artifact", "tb"), level=logging.WARNING)
    evidence = with_compose_logs(unit([line], [], [failure_row("failed-0")]), [line])
    report = reconcile_unit_attempts(evidence)
    assert report["unit_outcome"] == "failed", report["issues"]
    assert report["status"] == "blocked"
    log_failures = [f for f in report["pre_uuid_failures"] if f["kind"] == "rollout_failed"]
    assert len(log_failures) == 1
    assert log_failures[0]["evidence_refs"] == ["logs[0]:line:1", "logs[1]:line:1"]


def test_two_captures_of_one_internal_retry_record_it_once():
    lines = rendered_log()
    files = [f"rollouts/clip-1/{UUID_A}/rollout.asl", *complete_files(UUID_B)]
    evidence = with_compose_logs(unit(lines, files, [scored_row(UUID_B)]), lines)
    report = reconcile_unit_attempts(evidence)
    assert len(report["internal_retries"]) == 1
    assert report["status"] == "blocked"


def test_summary_row_scene_feeds_the_scene_binding():
    row = scored_row(UUID_B)
    row["clipgt_id"] = "clip-9"
    report = reconcile_unit_attempts(unit(rendered_log()[3:], complete_files(UUID_B), [row]))
    assert report["status"] == "blocked"
    assert "scene_clip_mismatch" in {item["code"] for item in report["issues"]}


# --- Review A4-F1..F4 regression controls (fail-closed attempt evidence) ----------------

ANSI_PREFIX = "\x1b[36mruntime-0-1  |\x1b[0m "
PROJECT_PREFIX = "0123abcd-runtime-0-1  | "


def prefixed(lines, prefix):
    return "\n".join(prefix + part for line in lines for part in line.split("\n"))


@pytest.mark.parametrize("prefix", [ANSI_PREFIX, PROJECT_PREFIX, "renderer-0-1  | ", ""])
def test_compose_marker_lines_outside_the_runtime_prefix_block(prefix):
    # A-F1: five pinned marker lines whose Compose prefix is unrecognised.
    parsed = parse_runtime_log(prefixed(rendered_log(), prefix), source="compose_up_stdout")
    assert parsed["events"] == []
    assert [(i["code"], i["severity"]) for i in parsed["issues"]] == [
        ("unparsed_marker_line", "blocking")] * 5
    assert parsed["runtime_service_lines"] == 0


@pytest.mark.parametrize("prefix", [ANSI_PREFIX, PROJECT_PREFIX])
def test_unrecognised_compose_prefix_never_hides_a_retry(prefix):
    # A-F1 repro R1: the failed first attempt is visible only in the log.
    evidence = unit([], complete_files(UUID_B), [scored_row(UUID_B)])
    evidence["logs"][0]["text"] = prefixed(rendered_log(), prefix)
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked"
    assert report["selected_attempt"] is None
    assert "unparsed_marker_line" in {item["code"] for item in report["issues"]}


def test_ansi_prefix_never_hides_a_late_failure_of_the_completed_attempt():
    # A-F1 repro R4: a late Rollout FAILED for the completed UUID.
    late = render(pinned_messages()["Rollout FAILED"], (
        JOB_B, "clip-1", UUID_B, False, "late", "tb"), level=logging.WARNING)
    evidence = unit(rendered_log()[3:], complete_files(UUID_B), [scored_row(UUID_B)])
    evidence["logs"][0]["text"] += "\n" + prefixed([late], ANSI_PREFIX)
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked"
    assert report["selected_attempt"] is None


@pytest.mark.parametrize("text", [
    "", "renderer-0-1  | 10:00:00.123 INFO:\tready",
    "Container runtime-0-1  Started\nruntime-0-1 exited with code 0",
])
def test_compose_capture_without_runtime_lines_blocks(text):
    # A-F1 repro R2: folders and a summary row alone never complete a unit.
    evidence = unit([], complete_files(UUID_B), [scored_row(UUID_B)])
    evidence["logs"][0]["text"] = text
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked"
    assert report["selected_attempt"] is None
    assert "runtime_log_without_runtime_lines" in {item["code"] for item in report["issues"]}


def test_complete_attempt_requires_its_start_and_completion_log_events():
    other = render("Loaded configuration for %s", ("clip-1",))
    for lines in ([other], [rendered_log()[3]], [other, rendered_log()[4]]):
        report = reconcile_unit_attempts(unit(lines, complete_files(UUID_B),
                                              [scored_row(UUID_B)]))
        assert report["status"] == "blocked", lines
        assert report["selected_attempt"] is None
        assert "attempt_log_evidence_missing" in {item["code"] for item in report["issues"]}


def test_worker_log_alone_cannot_show_scheduler_retries():
    evidence = unit(rendered_log()[3:], complete_files(UUID_B), [scored_row(UUID_B)])
    evidence["logs"] = [{"source": "runtime_worker_log", "text": "\n".join(rendered_log()[3:])}]
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked"
    assert "compose_log_missing" in {item["code"] for item in report["issues"]}


@pytest.mark.parametrize("mutation", ["none", "absent", "integer"])
def test_scored_summary_row_must_name_the_unit_scene(mutation):
    # A-F2: clipgt_id None/absent previously completed the unit.
    row = scored_row(UUID_B)
    if mutation == "none":
        row["clipgt_id"] = None
    elif mutation == "absent":
        del row["clipgt_id"]
    else:
        row["clipgt_id"] = 1
    report = reconcile_unit_attempts(unit(rendered_log()[3:], complete_files(UUID_B), [row]))
    assert report["status"] == "blocked"
    assert report["selected_attempt"] is None
    assert "summary_row_scene_missing" in {item["code"] for item in report["issues"]}


def test_summary_only_pre_uuid_failure_beside_a_complete_attempt_blocks():
    # A-F3 repro R3: an extra failed-0 row coexisting with a complete attempt.
    report = reconcile_unit_attempts(unit(rendered_log()[3:], complete_files(UUID_B),
                                          [scored_row(UUID_B), failure_row("failed-0")]))
    assert report["status"] == "blocked"
    assert report["selected_attempt"] is None
    assert "unattributed_failure_with_complete_attempt" in {
        item["code"] for item in report["issues"]}


def test_lone_surrogate_run_metadata_is_reported_not_raised():
    # A-F4: str.encode raised UnicodeEncodeError before the report was returned.
    evidence = unit(rendered_log()[3:], complete_files(UUID_B), [scored_row(UUID_B)],
                    metadata=f"run_uuid: {RUN_UUID}\n\udc80\n")
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked"
    assert report["run_uuid_binding"]["status"] == "blocked"
    assert "invalid_run_metadata" in {item["code"] for item in report["issues"]}


# --- Review A4-F6: shared artifact contract ----------------------------------------------

CONTRACT_STATUSES = {"passed", "blocked", "not_run", "not_estimable", "unverified",
                     "not_applicable", "qualify_on_host"}


def test_reconcile_report_uses_contract_status_scope_and_producer():
    source = Path(__file__).parents[1] / "gate" / "s2_capture.py"
    cases = [
        (unit(rendered_log()[3:], complete_files(UUID_B), [scored_row(UUID_B)]),
         "passed", "complete"),
        (unit(rendered_log()[3:], [f"rollouts/clip-1/{UUID_B}/rollout.asl"], []),
         "blocked", "incomplete"),
        (unit(rendered_log(), [f"rollouts/clip-1/{UUID_A}/rollout.asl", *complete_files(UUID_B)],
              [scored_row(UUID_B)]), "blocked", "unresolved"),
    ]
    for evidence, status, outcome in cases:
        report = reconcile_unit_attempts(evidence)
        assert (report["status"], report["unit_outcome"]) == (status, outcome), report["issues"]
        assert report["status"] in CONTRACT_STATUSES
        assert report["scope"] == "synthetic"
        assert report["producer"] == {
            "module": "gate.s2_capture", "function": "reconcile_unit_attempts",
            "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest()}


@pytest.mark.parametrize("scope", [None, "assumed", "ACTUAL"])
def test_reconcile_requires_a_declared_scope(scope):
    evidence = unit(rendered_log()[3:], complete_files(UUID_B), [scored_row(UUID_B)])
    evidence["scope"] = scope
    report = reconcile_unit_attempts(evidence)
    assert report["status"] == "blocked" and report["scope"] is None
    assert [i["code"] for i in report["issues"]] == ["invalid_unit_evidence"]
    del evidence["scope"]
    assert reconcile_unit_attempts(evidence)["issues"][0]["code"] == "invalid_unit_evidence"
