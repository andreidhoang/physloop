"""Known answers use literal protobuf wire bytes independent of decoder code."""

import ast
import asyncio
import hashlib
import json
import struct
from pathlib import Path

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from gate.s2_asl import inspect_asl
from tests.s2_asl_fixture import (
    actor_event,
    asl_bytes,
    field,
    fixed64,
    framed,
    metadata,
    pose,
    text_field,
)

ROOT = Path(__file__).parent / "fixtures" / "alpasim_asl"
SCHEMA = Path(__file__).parents[1] / "gate" / "asl_schema"


def inspect(data, **kwargs):
    return inspect_asl(data, expected_session_uuid="rollout-1", expected_scene_id="scene-1",
                       requested_steps=4, **kwargs)


def codes(result):
    return {item["code"] for item in result["issues"]}


def test_preserves_context_intermediate_duplicate_samples_without_counting_steps():
    data = asl_bytes(timestamps=(1000, 1000, 1100, 1150, 1200))
    result = inspect(data)
    assert result["parsed"] and result["inspection_valid"]
    assert [event["timestamp_us"] for event in result["pose_events"]] == [
        1000, 1000, 1100, 1150, 1200]
    assert result["pose_events"][0]["actors"] == [{
        "actor_id": "EGO", "pose": {"translation": [1.0, 2.0, 3.0],
                                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}]
    assert result["coordinate_frame"] == "local_to_aabb"
    assert result["requested_steps"] == 4
    assert result["effective_n_sim_steps"] == 2
    assert result["actual_steps"] is None
    assert result["observations"] == {
        "pose_event_count": 5, "first_timestamp_us": 1000, "last_timestamp_us": 1200,
        "observed_span_us": 200, "duplicate_timestamps": 1}
    assert result["metadata"]["start_timestamp_semantics"] == "egomotion_context_start_us"
    for key in ("runtime_qualified", "score_admitted", "cloud_launch_authorized"):
        assert result[key] is False
    for key in ("completion", "renderer_coverage", "black_frame_coverage", "map_coverage"):
        assert result[key] is None
    assert result["receipt"] == {"sha256": hashlib.sha256(data).hexdigest(),
                                  "size_bytes": len(data)}
    assert result["frame_receipts"][0]["byte_offset"] == 0
    assert result["frame_receipts"][0]["entry_type"] == "rollout_metadata"
    json.dumps(result, allow_nan=False)


def test_quaternion_is_preserved_without_normalizing():
    data = framed(metadata()) + framed(actor_event(1000, quaternion=(2.0, 0.5, -0.25, 0.0)))
    result = inspect(data)
    assert not result["inspection_valid"]
    assert "upstream_quaternion_incompatible" in codes(result)
    assert result["pose_events"][0]["actors"][0]["pose"]["quaternion_wxyz"] == [
        2.0, 0.5, -0.25, 0.0]


@pytest.mark.parametrize(("quaternion", "expected", "code"), [
    ((0.0, 0.0, 0.0, 0.0), False, "upstream_quaternion_incompatible"),
    ((1e-7, 0.0, 0.0, 0.0), False, "upstream_quaternion_incompatible"),
    ((1.0000499486923218, 0.0, 0.0, 0.0), True, None),
    ((1.0000500679016113, 0.0, 0.0, 0.0), False, "upstream_quaternion_incompatible"),
    ((0.9999500513076782, 0.0, 0.0, 0.0), True, None),
    ((0.9999499320983887, 0.0, 0.0, 0.0), False, "upstream_quaternion_incompatible"),
    ((0.009820855222642422, -0.6184602975845337, -0.5168955326080322,
      -0.5918863415718079), False, "quaternion_rounding_ambiguous"),
])
def test_upstream_float32_quaternion_compatibility_boundaries(quaternion, expected, code):
    # Adjacent f32 values around the pinned Rust length-squared +/-1e-4 boundary.
    # Final case differs across pinned glam SIMD/scalar reduction layouts.
    data = framed(metadata()) + framed(actor_event(1000, quaternion=quaternion))
    result = inspect(data)
    assert result["inspection_valid"] is expected
    if code:
        assert code in codes(result)
    assert result["pose_events"][0]["actors"][0]["pose"]["quaternion_wxyz"] == [
        struct.unpack("<f", struct.pack("<f", component))[0] for component in quaternion]


SESSION_ENTRIES = {
    "driver_request": 4, "traffic_request": 6, "driver_session_request": 10,
    "driver_camera_image": 11, "driver_ego_trajectory": 12, "traffic_session_request": 13,
    "controller_request": 14, "route_request": 17, "ground_truth_request": 18,
}


@pytest.mark.parametrize(("kind", "number"), SESSION_ENTRIES.items())
@pytest.mark.parametrize("session", ["other-session", "", "rollout-1"])
def test_session_bearing_services_match_bound_rollout_without_claiming_full_semantics(
        kind, number, session):
    service = field(number, text_field(1, session) if session else b"")
    result = inspect(asl_bytes() + framed(service))
    assert result["inspection_valid"] is (session == "rollout-1")
    assert result["frame_receipts"][-1]["entry_type"] == kind
    assert result["frame_receipts"][-1]["semantic_inspection"] is False
    assert result["frame_receipts"][-1]["session_identity_checked"] is True
    if session != "rollout-1":
        assert "service_session_identity_mismatch" in codes(result)


def test_video_model_session_is_not_the_rollout_uuid():
    # Separate service-session namespace: must not be compared to the rollout UUID.
    result = inspect(asl_bytes() + framed(field(25, text_field(1, "video-model-session"))))
    assert result["inspection_valid"]
    assert result["frame_receipts"][-1]["session_identity_checked"] is False


@pytest.mark.parametrize(("data", "code"), [
    (b"", "missing_metadata"),
    (b"\0\0", "truncated_header"),
    (struct.pack(">L", 20) + b"\x0a", "truncated_frame"),
    (framed(b"\x0a\xff"), "invalid_protobuf"),
    (framed(b""), "empty_log_entry"),
    (framed(field(99, b"")), "unsupported_field"),
    (framed(metadata() + actor_event(1000)), "ambiguous_field"),
    (framed(metadata() + metadata()), "ambiguous_field"),
    (framed(metadata()) * 2, "duplicate_metadata"),
    (framed(actor_event(1000)), "missing_metadata"),
    (framed(metadata()), "missing_pose_events"),
    (asl_bytes(session_uuid="wrong"), "session_identity_mismatch"),
    (asl_bytes(scene_id="wrong"), "scene_identity_mismatch"),
    (asl_bytes(n_steps=0), "invalid_effective_steps"),
    (asl_bytes(n_steps=5), "effective_steps_exceed_requested"),
    (asl_bytes(dt=0), "invalid_control_timestep"),
    (asl_bytes(timestamps=(1100, 1000)), "timestamp_regression"),
])
def test_malformed_or_inconsistent_data_fails_explicitly(data, code):
    result = inspect(data)
    assert not result["inspection_valid"]
    assert code in codes(result)
    assert result["actual_steps"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"max_file_bytes": 1}, "file_limit_exceeded"),
    ({"max_frame_bytes": 1}, "frame_limit_exceeded"),
    ({"max_frames": 1}, "frame_count_exceeded"),
])
def test_resource_limits(kwargs, code):
    assert code in codes(inspect(asl_bytes(), **kwargs))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_data_kept_as_explicit_json_tokens_and_blocks_inspection(value):
    result = inspect(framed(metadata()) + framed(actor_event(1000, position=(value, 0.0, 0.0))))
    assert result["parsed"] and not result["inspection_valid"]
    assert "nonfinite_pose" in codes(result)
    raw = result["pose_events"][0]["actors"][0]["pose"]["translation"][0]
    assert raw in {"NaN", "Infinity", "-Infinity"}
    json.dumps(result, allow_nan=False)


def test_valid_service_message_is_receipted_without_claiming_semantic_validation():
    result = inspect(asl_bytes() + framed(field(5, b"")))
    assert result["inspection_valid"]
    assert result["frame_receipts"][-1]["entry_type"] == "driver_return"
    assert result["frame_receipts"][-1]["semantic_inspection"] is False
    assert result["completion"] is None


@pytest.mark.parametrize(("entry", "code"), [
    (field(1, b""), "missing_session_metadata"),
    (field(2, fixed64(1, 1000) + field(2, text_field(1, "EGO"))), "missing_pose"),
    (field(2, fixed64(1, 1000) + field(2, text_field(1, "EGO") + field(2, b""))),
     "missing_pose_component"),
])
def test_absent_submessages_never_become_observed_zero_poses(entry, code):
    data = framed(entry) if code == "missing_session_metadata" else framed(metadata()) + framed(entry)
    result = inspect(data)
    assert result["parsed"] and not result["inspection_valid"]
    assert code in codes(result)
    if result["pose_events"]:
        observed = result["pose_events"][0]["actors"][0]["pose"]
        assert observed is None or observed == {"translation": None, "quaternion_wxyz": None}


def test_all_actor_samples_including_source_duplicate_ids_are_preserved():
    # The upstream dict-of-lists writer can repeat IDs when given duplicate timestamps.
    actor = text_field(1, "EGO") + field(2, pose())
    traffic = text_field(1, "vehicle-1") + field(2, pose((4.0, 5.0, 6.0)))
    entry = field(2, fixed64(1, 1000) + field(2, actor) + field(2, traffic) + field(2, actor))
    result = inspect(framed(metadata()) + framed(entry))
    assert result["inspection_valid"]
    actors = result["pose_events"][0]["actors"]
    assert [actor["actor_id"] for actor in actors] == ["EGO", "vehicle-1", "EGO"]
    assert actors[1]["pose"]["translation"] == [4.0, 5.0, 6.0]


def test_zero_timestamp_and_explicit_zero_vector_are_not_missing_messages():
    result = inspect(framed(metadata(start=0)) + framed(actor_event(0, position=(0.0, 0.0, 0.0))))
    assert result["inspection_valid"]
    assert result["pose_events"][0]["timestamp_us"] == 0
    assert result["pose_events"][0]["actors"][0]["pose"]["translation"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("requested", [None, True, 0, -1, float("nan"), object()])
def test_invalid_arguments_report_json_safe_failures(requested):
    result = inspect_asl(asl_bytes(), expected_session_uuid="rollout-1",
                         expected_scene_id="scene-1", requested_steps=requested)
    assert codes(result) == {"invalid_arguments"}
    assert not result["parsed"] and not result["inspection_valid"]
    json.dumps(result, allow_nan=False)


def test_literal_fixture_decodes_with_full_upstream_descriptor_and_all_imports():
    files = descriptor_pb2.FileDescriptorSet.FromString((SCHEMA / "e121e591.pb").read_bytes())
    assert len(files.file) == 8
    assert {f.name for f in files.file} == {
        str(path.relative_to(ROOT / "proto")) for path in (ROOT / "proto").rglob("*.proto")}
    pool = descriptor_pool.DescriptorPool()
    for file in files.file:
        pool.Add(file)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("logging.LogEntry"))
    decoded = cls.FromString(metadata())
    assert decoded.WhichOneof("log_entry") == "rollout_metadata"
    assert decoded.rollout_metadata.session_metadata.session_uuid == "rollout-1"
    assert decoded.rollout_metadata.session_metadata.n_sim_steps == 2
    assert cls.FromString(actor_event(1000)).actor_poses.actor_poses[0].actor_pose.vec.x == 1.0


def test_complete_public_source_and_descriptor_receipts():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert len(provenance["sources"]) == 14
    for entry in [*provenance["sources"], provenance["license_file"]]:
        raw = (ROOT / entry["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == (
            entry["git_blob_sha"])
    packaged = json.loads((SCHEMA / "provenance.json").read_text())
    assert packaged["commit"] == provenance["commit"]
    assert packaged["sources"] == [e for e in provenance["sources"] if e["file"].endswith(".proto")]
    assert hashlib.sha256((SCHEMA / "e121e591.pb").read_bytes()).hexdigest() == (
        packaged["descriptor"]["sha256"])


def test_literal_frame_matches_execution_of_pinned_upstream_writer():
    source = ast.parse((ROOT / "logs_e121e591.py.txt").read_text())
    writer = next(node for node in source.body if getattr(node, "name", None) == "write_pb_stream")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        writer,
    ], type_ignores=[])
    namespace = {"struct": struct}
    # Only execute the hash-verified writer function, not runtime imports/startup.
    exec(compile(ast.fix_missing_locations(module), "pinned_asl_writer", "exec"), namespace)  # noqa: S102

    class Buffer:
        data = b""

        async def write(self, data):
            self.data += data

    class Payload:
        def SerializeToString(self):
            return metadata()

    buffer = Buffer()
    asyncio.run(namespace["write_pb_stream"](buffer, Payload()))
    assert buffer.data == framed(metadata())
