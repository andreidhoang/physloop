"""Inspect pinned ASL receipts against scheduled attempts; never admit scores.

The upstream evaluator binds metadata.scene_id to clipgt_id and metadata.session_uuid
to rollout_id. run_uuid comes from external run metadata, not the ASL session UUID.
Receipt checks establish byte integrity, not producer authenticity or completion.
"""

from gate.alpasim_adapter import SUPPORTED_EVALUATOR_SHA
from gate.s2_asl import inspect_asl
from gate.s2_identity import canonical_json_bytes, identity_hash, read_artifact, read_json_bytes
from gate.s2_ledger import reconcile_attempts
from gate.s2_schedule import build_schedule

REQUEST_FIELDS = {
    "schema_version", "evidence_kind", "capture_origin", "evaluator_sha",
    "schedule", "schedule_request", "events", "artifacts",
}
MAX_ASL_BYTES = 256 * 1024 * 1024
MAX_EVENTS_BYTES = 16 * 1024 * 1024


def _bounded_read(root, receipt, maximum):
    if (
        type(receipt) is not dict or type(receipt.get("size_bytes")) is not int
        or not 0 <= receipt["size_bytes"] <= maximum
    ):
        raise ValueError(f"Artifact exceeds or lacks the declared byte limit ({maximum})")
    return read_artifact(root, receipt)


def inspect_capture(root, request):
    """Return every ledger attempt and available ASL inspection, including failures.

    ``capture_inspection_passed`` requires both ledger admission and successful
    parsing/binding of all expected logs. It is strictly a structural check:
    horizon completion, render/map evidence and actual scorer execution remain
    unqualified even when this field is true.
    """
    report = {
        "schema_version": 1,
        "evidence_kind": "alpasim_asl_inspection_v1",
        "capture_origin": None,
        "capture_inspection_passed": False,
        "ledger": None,
        "captures": [],
        "issues": [],
        "request": None,
        "request_sha256": None,
        "score_admitted": False,
        "runtime_qualified": False,
        "cloud_launch_authorized": False,
        "provenance_authenticated": False,
        "max_asl_bytes": MAX_ASL_BYTES,
        "max_events_bytes": MAX_EVENTS_BYTES,
        "remaining": [
            "actual completion and requested/effective horizon reconciliation",
            "complete renderer, black-frame and map evidence",
            "raw metrics and actual scorer-path qualification",
        ],
    }

    def issue(code, path, message):
        report["issues"].append({"code": code, "path": path, "message": message})

    try:
        request = read_json_bytes(canonical_json_bytes(request))
        report["request"] = request
        report["request_sha256"] = identity_hash(request)
        if type(request) is not dict or set(request) != REQUEST_FIELDS:
            raise ValueError("Expected the exact ASL inspection request fields")
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Unsupported inspection schema version")
        if request["evidence_kind"] != "alpasim_asl_inspection_v1":
            raise ValueError("ASL inspection is distinct from synthetic score ingestion")
        if request["capture_origin"] not in ("synthetic_fixture", "runtime_claim_unverified"):
            raise ValueError("Origin must be synthetic_fixture or runtime_claim_unverified")
        report["capture_origin"] = request["capture_origin"]
        if request["evaluator_sha"] != SUPPORTED_EVALUATOR_SHA:
            raise ValueError("Unsupported evaluator/source pin")
        schedule = request["schedule"]
        if request["schedule_request"] is not None:
            if type(request["schedule_request"]) is not dict:
                raise ValueError("schedule_request must be an object")
            if build_schedule(**request["schedule_request"]) != schedule:
                raise ValueError("Schedule differs from its complete source request")
        elif not (
            type(schedule) is list and schedule
            and all(type(job) is dict and job.get("phase") == "synthetic" for job in schedule)
        ):
            raise ValueError("Pilot/full schedules require their source request")
        if type(request["artifacts"]) is not list:
            raise ValueError("artifacts must be a list")
        events = read_json_bytes(_bounded_read(root, request["events"], MAX_EVENTS_BYTES))
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        issue("invalid_capture_request", "request", str(exc))
        return report

    ledger = report["ledger"] = reconcile_attempts(schedule, events)
    # Do not short-circuit on ledger failure: partial and retried captures still
    # provide useful diagnostics, and all original receipts remain in the ledger.
    jobs = {job["job_id"]: job for job in ledger["jobs"]}
    attempts = {attempt["attempt_id"] for attempt in ledger["attempts"]}
    artifacts, paths = {}, set()
    for index, artifact in enumerate(request["artifacts"]):
        path = f"artifacts[{index}]"
        if (
            type(artifact) is not dict or set(artifact) != {"attempt_id", "asl"}
            or type(artifact["attempt_id"]) is not str or not artifact["attempt_id"].strip()
        ):
            issue("invalid_attempt_artifact", path, "Expected attempt_id and ASL receipt")
            continue
        attempt_id = artifact["attempt_id"]
        if attempt_id not in attempts:
            issue("unknown_attempt_artifact", path, "Artifact has no recorded attempt")
            continue
        if attempt_id in artifacts:
            issue("duplicate_attempt_artifact", path, "Only one ASL per attempt is supported")
            continue
        artifacts[attempt_id] = artifact["asl"]
        receipt = artifact["asl"]
        filename = receipt.get("path") if type(receipt) is dict else None
        if type(filename) is str:
            if filename in paths:
                issue("reused_asl_path", path, "One ASL path cannot represent two attempts")
            paths.add(filename)

    for attempt in ledger["attempts"]:
        attempt_id = attempt["attempt_id"]
        path = f"attempt:{attempt_id}"
        capture = {
            "attempt_id": attempt_id, "job_id": attempt["job_id"],
            "state": attempt["state"], "status": "unverified", "inspection": None,
        }
        report["captures"].append(capture)
        identity = attempt["upstream_identity"]
        if identity is None:
            capture["status"] = "no_upstream_identity"
            if attempt_id in artifacts:
                issue("unbound_asl", path, "ASL cannot bind to an attempt without upstream IDs")
            continue
        job = jobs.get(attempt["job_id"])
        if job is None:
            issue("unknown_capture_job", path, "Attempt is absent from the schedule")
            continue
        if job["scene_id"] != identity["clipgt_id"]:
            issue("scene_identity_mismatch", path, "Schedule scene differs from clipgt_id")
            continue
        if attempt_id not in artifacts:
            capture["status"] = "missing_asl"
            issue("missing_asl", path, "An attempt with upstream IDs requires its ASL receipt")
            continue
        try:
            data = _bounded_read(root, artifacts[attempt_id], MAX_ASL_BYTES)
            inspection = inspect_asl(
                data, expected_session_uuid=identity["rollout_id"],
                expected_scene_id=identity["clipgt_id"], requested_steps=job["requested_steps"],
            )
            capture["inspection"] = inspection
            capture["status"] = "verified" if inspection["inspection_valid"] else "invalid_asl"
            if not inspection["inspection_valid"]:
                issue("asl_inspection_failed", path, "See retained parser diagnostics")
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            capture["status"] = "artifact_unverified"
            issue("artifact_unverified", path, str(exc))
    report["capture_inspection_passed"] = ledger["admitted"] and not report["issues"]
    return report
