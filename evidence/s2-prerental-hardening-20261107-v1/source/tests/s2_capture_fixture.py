"""Actual ASL wire format with invented poses, explicitly synthetic provenance."""

from s2_asl_fixture import asl_bytes
from s2_ingestion_fixture import SHA, fixture_bundle, receipt


def wire_capture_request(root):
    source, _, _, _ = fixture_bundle(root)
    source["schedule"][0]["scene_id"] = "clip"
    data = asl_bytes(session_uuid="rollout", scene_id="clip", n_steps=3)
    return {
        "schema_version": 1,
        "evidence_kind": "alpasim_asl_inspection_v1",
        "capture_origin": "synthetic_fixture",
        "evaluator_sha": SHA,
        "schedule": source["schedule"],
        "schedule_request": None,
        "events": source["events"],
        "artifacts": [{"attempt_id": "attempt", "asl": receipt(root, "capture.asl", data)}],
    }
