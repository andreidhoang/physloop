"""Known answers use literal protobuf wire bytes independent of decoder code."""

import ast
import asyncio
import hashlib
import io
import json
import struct
from pathlib import Path

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from gate.s2_asl import inspect_asl, inspect_asl_stream
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


def test_all_actor_samples_including_source_duplicate_ids_are_preserved_but_block():
    # The upstream dict-of-lists writer (step.py:141-178) repeats IDs when given
    # duplicate timestamps. R4 C9 (confirmed): a repeated ID in one frame is
    # ambiguous for any per-actor consumer, so it now blocks inspection. The
    # previous version of this test accepted it; every sample is still preserved.
    actor = text_field(1, "EGO") + field(2, pose())
    traffic = text_field(1, "vehicle-1") + field(2, pose((4.0, 5.0, 6.0)))
    entry = field(2, fixed64(1, 1000) + field(2, actor) + field(2, traffic) + field(2, actor))
    result = inspect(framed(metadata()) + framed(entry))
    assert result["parsed"] and not result["inspection_valid"]
    assert "duplicate_actor_id" in codes(result)
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


# --- A4: R4 C8/C9, A3 hood-mask evidence, issue shape and streaming (A1) ---------------


def test_every_issue_has_shared_contract_shape():
    result = inspect(asl_bytes(session_uuid="wrong"))
    assert result["issues"]
    for item in result["issues"]:
        assert set(item) >= {"code", "path", "severity", "evidence_refs", "message"}
        assert item["severity"] in {"blocking", "warning"}
        assert type(item["evidence_refs"]) is list


def test_effective_steps_below_requested_is_a_nonblocking_warning_with_both_numbers():
    # R4 C8 repro: metadata n_sim_steps=2 against requested 4.
    result = inspect(asl_bytes(n_steps=2))
    assert result["inspection_valid"]
    warning = [item for item in result["issues"]
               if item["code"] == "effective_steps_below_requested"]
    assert len(warning) == 1
    assert warning[0]["severity"] == "warning"
    assert "2" in warning[0]["message"] and "4" in warning[0]["message"]
    assert result["requested_steps"] == 4
    assert result["effective_n_sim_steps"] == 2
    assert result["horizon_status"] == "effective_below_requested"
    assert result["actual_steps"] is None


def test_equal_effective_steps_add_no_issue():
    result = inspect(asl_bytes(n_steps=4))
    assert result["inspection_valid"]
    assert result["issues"] == []
    assert result["horizon_status"] == "effective_equals_requested"


def test_effective_steps_above_requested_remain_blocking():
    result = inspect(asl_bytes(n_steps=5))
    assert not result["inspection_valid"]
    blocking = {item["code"] for item in result["issues"] if item["severity"] == "blocking"}
    assert "effective_steps_exceed_requested" in blocking
    assert result["horizon_status"] == "effective_exceeds_requested"


def _actor_frame(timestamp, actors):
    body = fixed64(1, timestamp)
    for actor_id, position in actors:
        body += field(2, text_field(1, actor_id) + field(2, pose(position)))
    return field(2, body)


def test_r4_c9_conflicting_ego_poses_in_one_frame_block():
    # R4 repro_10: one actor_poses frame carrying two different EGO poses.
    entries = [metadata(session_uuid="rollout-1", scene_id="scene-1", n_steps=4)]
    entries += [_actor_frame(1000 + 100 * i, [("EGO", (float(i), 0.0, 0.0))]) for i in range(3)]
    entries.append(_actor_frame(1300, [("EGO", (3.0, 0.0, 0.0)), ("EGO", (250.0, -40.0, 0.0))]))
    result = inspect(b"".join(map(framed, entries)))
    assert not result["inspection_valid"]
    duplicate = [item for item in result["issues"] if item["code"] == "duplicate_actor_id"]
    assert len(duplicate) == 1
    assert duplicate[0]["severity"] == "blocking"
    assert duplicate[0]["frame_index"] == 4
    assert [actor["pose"]["translation"] for actor in result["pose_events"][-1]["actors"]] == [
        [3.0, 0.0, 0.0], [250.0, -40.0, 0.0]]


def test_distinct_actor_ids_in_one_frame_do_not_trigger_duplicate_control():
    entry = _actor_frame(1000, [("EGO", (1.0, 2.0, 3.0)), ("vehicle-1", (4.0, 5.0, 6.0))])
    result = inspect(framed(metadata(n_steps=4)) + framed(entry))
    assert result["inspection_valid"], result["issues"]


def _rgb_request(camera, *, mask=None):
    body = text_field(1, "scene-1") + field(4, text_field(5, camera))
    if mask is not None:
        rig, mask_camera = mask
        body += field(11, varint(1), 0) + field(12, text_field(1, mask_camera)
                                               + text_field(2, rig))
    return body


def varint(value):
    from tests.s2_asl_fixture import varint as encode
    return encode(value)


def test_hood_mask_evidence_counts_request_flags_and_available_masks_literally():
    available = field(23, field(1, field(1, text_field(1, "camera_front_wide_120fov")
                                         + text_field(2, "hyperion_8_1"))))
    wide = ("hyperion_8_1", "camera_front_wide_120fov")
    single = field(3, _rgb_request("camera_front_wide_120fov", mask=wide))
    aggregated = field(21, field(1, _rgb_request("camera_front_wide_120fov", mask=wide))
                       + field(1, _rgb_request("camera_front_tele_30fov")))
    batch = field(30, field(1, text_field(1, "tele") + field(2, _rgb_request(
        "camera_front_tele_30fov"))))
    data = asl_bytes(n_steps=4) + b"".join(map(framed, (available, single, aggregated, batch)))
    result = inspect(data)
    assert result["inspection_valid"], result["issues"]
    assert result["hood_mask_evidence"] == {
        "status": "present",
        "available_ego_masks_returns": 1,
        "available_ego_masks": ["hyperion_8_1/camera_front_wide_120fov"],
        "rgb_render_requests": 4,
        "with_insert_ego_mask": 2,
        "without_insert_ego_mask": 2,
        "by_camera": {
            "camera_front_tele_30fov": {
                "rgb_render_requests": 2, "with_insert_ego_mask": 0, "ego_mask_ids": []},
            "camera_front_wide_120fov": {
                "rgb_render_requests": 2, "with_insert_ego_mask": 2,
                "ego_mask_ids": ["hyperion_8_1/camera_front_wide_120fov"]},
        },
        "hood_applied": None,
        "semantics": "request_flags_only_not_rendered_pixels",
    }
    assert result["renderer_coverage"] is None
    hood_receipts = [r for r in result["frame_receipts"] if r.get("hood_fields_inspected")]
    assert [r["entry_type"] for r in hood_receipts] == [
        "available_ego_masks_return", "render_request", "aggregated_render_request",
        "batch_render_request"]


def test_absent_hood_evidence_is_absent_not_zero_coverage():
    result = inspect(asl_bytes(n_steps=4))
    assert result["hood_mask_evidence"]["status"] == "absent"
    assert result["hood_mask_evidence"]["hood_applied"] is None


def test_hood_source_lines_are_pinned():
    root = Path(__file__).parent / "fixtures" / "alpasim_launch_upstream" / "alpasim"
    service = (root / "src/runtime/alpasim_runtime/services/sensorsim_service.py.txt").read_text()
    lines = service.splitlines()
    assert "LogEntry(available_ego_masks_return=self._available_ego_masks)" in lines[193]
    assert "insert_ego_mask=ego_mask_id is not None," in lines[308]
    assert "LogEntry(batch_render_request=request)" in lines[452]
    assert "LogEntry(aggregated_render_request=request)" in lines[513]
    assert "LogEntry(render_request=request)" in lines[593]


class RecordingStream:
    def __init__(self, data):
        self.buffer = io.BytesIO(data)
        self.requests = []

    def read(self, size):
        self.requests.append(size)
        return self.buffer.read(size)


def _large_asl():
    # video_model_session_id (logging.proto field 25): SessionId.session_id string.
    big_payload = text_field(1, "x" * (3 * 1024 * 1024))
    return asl_bytes(n_steps=4) + framed(field(25, big_payload))


def test_stream_reads_bounded_chunks_and_matches_in_memory_result():
    data = _large_asl()
    stream = RecordingStream(data)
    streamed = inspect_asl_stream(stream, expected_session_uuid="rollout-1",
                                  expected_scene_id="scene-1", requested_steps=4)
    assert max(stream.requests) <= 1024 * 1024
    assert len(data) > 3 * 1024 * 1024
    assert streamed["inspection_valid"], streamed["issues"]
    assert streamed == inspect(data)
    assert streamed["receipt"] == {"sha256": hashlib.sha256(data).hexdigest(),
                                   "size_bytes": len(data)}


def test_stream_receipt_covers_drained_suffix_after_framing_error():
    data = asl_bytes(n_steps=4) + b"\0\0"
    result = inspect_asl_stream(io.BytesIO(data), expected_session_uuid="rollout-1",
                                expected_scene_id="scene-1", requested_steps=4)
    assert "truncated_header" in codes(result)
    assert result["receipt"]["sha256"] == hashlib.sha256(data).hexdigest()
    corrupt = framed(b"\x0a\xff") + b"tail bytes after the bad frame"
    result = inspect_asl_stream(io.BytesIO(corrupt), expected_session_uuid="rollout-1",
                                expected_scene_id="scene-1", requested_steps=4)
    assert not result["parsed"]
    assert result["receipt"] == {"sha256": hashlib.sha256(corrupt).hexdigest(),
                                 "size_bytes": len(corrupt)}


def test_stream_file_bound_stops_reading_without_a_partial_receipt():
    data = _large_asl()
    stream = RecordingStream(data)
    result = inspect_asl_stream(stream, expected_session_uuid="rollout-1",
                                expected_scene_id="scene-1", requested_steps=4,
                                max_file_bytes=2 * 1024 * 1024)
    assert codes(result) == {"file_limit_exceeded"}
    assert result["receipt"] is None
    assert result["bytes_read"] <= 2 * 1024 * 1024 + 1024 * 1024
    assert stream.buffer.tell() < len(data)
    assert not result["inspection_valid"]


@pytest.mark.parametrize("stream", [None, b"bytes", object()])
def test_stream_api_rejects_non_streams(stream):
    result = inspect_asl_stream(stream, expected_session_uuid="r", expected_scene_id="s",
                                requested_steps=1)
    assert codes(result) == {"invalid_arguments"}


def test_overlong_stream_reads_are_rejected():
    class Liar:
        def read(self, size):
            return b"\0" * (size + 1)

    with pytest.raises(TypeError):
        inspect_asl_stream(Liar(), expected_session_uuid="r", expected_scene_id="s",
                           requested_steps=1)


def test_provisional_file_bound_is_documented_agent_default():
    import gate.s2_asl as module

    assert module.MAX_FILE_BYTES == 2 * 1024 * 1024 * 1024
    assert "agent technical default (D-018)" in module.__doc__
    assert "provisional until the first measured ASL" in module.__doc__
