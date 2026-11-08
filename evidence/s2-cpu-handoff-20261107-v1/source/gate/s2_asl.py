"""Strict, inspection-only reader for the pinned upstream AlpaSim ASL format.

Framing and the complete protobuf descriptor come from public e121e591 source.
Original local-to-AABB poses are not transformed, normalized, interpolated or
deduplicated. Parsing establishes neither provenance nor runtime completion.
"""

import hashlib
import math
import struct
from collections import Counter
from functools import lru_cache
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError

SOURCE_COMMIT = "e121e59141dc391f6c295513d6dadc840a419149"
DESCRIPTOR_SHA256 = "94d684ffa268538bd385cd05041d2f360f88d7073f054d65a8097020d4e89314"
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 1_000_000


class _WireError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@lru_cache(maxsize=1)
def _entry_class():
    data = (Path(__file__).parent / "asl_schema" / "e121e591.pb").read_bytes()
    if hashlib.sha256(data).hexdigest() != DESCRIPTOR_SHA256:
        raise ValueError("Packaged ASL descriptor does not match its pinned hash")
    descriptors = descriptor_pb2.FileDescriptorSet.FromString(data)
    pool = descriptor_pool.DescriptorPool()
    for file in descriptors.file:
        pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("logging.LogEntry"))


def _varint(data, offset, end):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= end:
            raise _WireError("invalid_protobuf", "Truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if value > 2**64 - 1:
                break
            return value, offset
    raise _WireError("invalid_protobuf", "Protobuf varint exceeds 64 bits")


def _wire_type(field):
    if field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FIXED64,
                      FieldDescriptor.TYPE_SFIXED64):
        return 1
    if field.type in (FieldDescriptor.TYPE_STRING, FieldDescriptor.TYPE_MESSAGE,
                      FieldDescriptor.TYPE_BYTES):
        return 2
    if field.type in (FieldDescriptor.TYPE_FLOAT, FieldDescriptor.TYPE_FIXED32,
                      FieldDescriptor.TYPE_SFIXED32):
        return 5
    return 0


def _scalar(data, offset, end, wire):
    if wire == 0:
        return _varint(data, offset, end)
    width = {1: 8, 5: 4}[wire]
    if offset + width > end:
        raise _WireError("invalid_protobuf", "Truncated protobuf scalar")
    return None, offset + width


def _check_wire(data, descriptor, start=0, end=None, depth=0):
    """Reject ambiguity/unknown fields that the permissive protobuf reader merges.

    This is structural checking against the full descriptor, not service-level
    semantic validation. Unknown fields are unsupported under this pin, distinct
    from corrupt framing. Repeated scalar packed/unpacked forms are both valid.
    """
    if depth > 64:
        raise _WireError("protobuf_depth_exceeded", "Protobuf nesting exceeds 64 levels")
    if end is None:
        end = len(data)
    offset, seen, oneofs = start, set(), set()
    while offset < end:
        tag, offset = _varint(data, offset, end)
        number, wire = tag >> 3, tag & 7
        if number == 0 or number > (2**29 - 1) or wire not in (0, 1, 2, 5):
            raise _WireError("invalid_protobuf", "Invalid protobuf tag or wire type")
        field = descriptor.fields_by_number.get(number)
        if field is None:
            raise _WireError("unsupported_field", f"Unknown field {descriptor.full_name}.{number}")
        if not field.is_repeated and number in seen:
            raise _WireError("ambiguous_field", f"Repeated singular field {field.full_name}")
        seen.add(number)
        if field.containing_oneof:
            oneof = field.containing_oneof.name
            if oneof in oneofs:
                raise _WireError("ambiguous_field", f"Multiple members of oneof {oneof}")
            oneofs.add(oneof)
        expected = _wire_type(field)
        packed = field.is_repeated and wire == 2 and expected in (0, 1, 5)
        if wire != expected and not packed:
            raise _WireError("invalid_protobuf", f"Wrong wire type for {field.full_name}")
        if wire == 2:
            size, offset = _varint(data, offset, end)
            stop = offset + size
            if stop > end:
                raise _WireError("invalid_protobuf", "Truncated length-delimited protobuf field")
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                _check_wire(data, field.message_type, offset, stop, depth + 1)
            elif packed:
                cursor = offset
                while cursor < stop:
                    _, cursor = _scalar(data, cursor, stop, expected)
            offset = stop
        else:
            value, offset = _scalar(data, offset, end, wire)
            if wire == 0 and field.type == FieldDescriptor.TYPE_UINT32 and value > 2**32 - 1:
                raise _WireError("invalid_protobuf", f"Out-of-range uint32 {field.full_name}")


def _json_float(value):
    if math.isfinite(value):
        return value
    return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")


def _pose(message, issue, frame_index):
    components = {}
    for name, key, axes in (("vec", "translation", "xyz"),
                            ("quat", "quaternion_wxyz", "wxyz")):
        if not message.HasField(name):
            components[key] = None
            issue("missing_pose_component", f"Pose lacks {name} submessage", frame_index)
            continue
        values = [getattr(getattr(message, name), axis) for axis in axes]
        if not all(math.isfinite(value) for value in values):
            issue("nonfinite_pose", f"Pose {name} has nonfinite values", frame_index)
        components[key] = [_json_float(value) for value in values]
    return components


def _metadata(message, issue, frame_index):
    # MessageToDict retains other pinned metadata fields; fixed64 fields there are
    # protobuf JSON decimal strings. The inspected session scalars below are ints.
    raw = MessageToDict(message, preserving_proto_field_name=True,
                        always_print_fields_with_no_presence=True)
    session = message.session_metadata
    if not message.HasField("session_metadata"):
        issue("missing_session_metadata", "Metadata lacks session_metadata", frame_index)
    result = {
        "session_uuid": session.session_uuid,
        "scene_id": session.scene_id,
        "batch_size": session.batch_size,
        "n_sim_steps": session.n_sim_steps,
        "start_timestamp_us": session.start_timestamp_us,
        "start_timestamp_semantics": "egomotion_context_start_us",
        "control_timestep_us": session.control_timestep_us,
        "source_fields": raw,
    }
    if message.HasField("transform_ego_coords_rig_to_aabb"):
        result["transform_ego_coords_rig_to_aabb"] = _pose(
            message.transform_ego_coords_rig_to_aabb, issue, frame_index)
    for sample in message.ego_rig_recorded_ground_truth_trajectory.poses:
        if not sample.HasField("pose"):
            issue("missing_pose", "Recorded ground-truth sample lacks pose", frame_index)
        else:
            _pose(sample.pose, issue, frame_index)
    return result


def inspect_asl(data, *, expected_session_uuid, expected_scene_id, requested_steps,
                max_file_bytes=MAX_FILE_BYTES, max_frame_bytes=MAX_FRAME_BYTES,
                max_frames=MAX_FRAMES):
    """Inspect pinned framed protobuf bytes without admitting score/runtime evidence.

    Session UUID binds to upstream ``rollout_id``; scene ID to ``clipgt_id``.
    ``run_uuid`` is external evaluator run metadata, absent from this binding.
    Limits are configurable engineering limits, not qualified runtime capacities.
    Invalid input reports preserve decoded-prefix diagnostics and frame receipts.
    """
    report = {
        "schema_version": 1, "kind": "alpasim_asl_inspection_v1",
        "source_commit": SOURCE_COMMIT, "parsed": False, "inspection_valid": False,
        "issues": [], "receipt": None, "metadata": None, "frame_receipts": [],
        "pose_events": [], "coordinate_frame": "local_to_aabb",
        "quaternion_component_order": "wxyz",
        "requested_steps": requested_steps if type(requested_steps) is int else None,
        "effective_n_sim_steps": None, "actual_steps": None,
        "completion": None, "renderer_coverage": None, "black_frame_coverage": None,
        "map_coverage": None, "runtime_qualified": False, "score_admitted": False,
        "cloud_launch_authorized": False,
        "observations": {"pose_event_count": 0, "first_timestamp_us": None,
                         "last_timestamp_us": None, "observed_span_us": None,
                         "duplicate_timestamps": 0},
    }

    def issue(code, message, frame_index=None):
        item = {"code": code, "message": message}
        if frame_index is not None:
            item["frame_index"] = frame_index
        report["issues"].append(item)

    if (type(data) is not bytes or not isinstance(expected_session_uuid, str)
            or not expected_session_uuid or not isinstance(expected_scene_id, str)
            or not expected_scene_id or any(type(value) is not int or value <= 0 for value in (
                requested_steps, max_file_bytes, max_frame_bytes, max_frames))):
        issue("invalid_arguments", "Expected bytes, explicit nonempty IDs and positive integer limits")
        return report
    report["receipt"] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    if len(data) > max_file_bytes:
        issue("file_limit_exceeded", "ASL exceeds configured file size limit")
        return report
    try:
        entry_class = _entry_class()
    except (OSError, ValueError, DecodeError) as error:
        issue("schema_unavailable", str(error))
        return report
    offset, syntax_ok = 0, True
    while offset < len(data):
        frame_index = len(report["frame_receipts"])
        if frame_index >= max_frames:
            issue("frame_count_exceeded", "ASL exceeds configured frame count", frame_index)
            syntax_ok = False
            break
        if len(data) - offset < 4:
            issue("truncated_header", "ASL length prefix has fewer than four bytes", frame_index)
            syntax_ok = False
            break
        size = struct.unpack_from(">L", data, offset)[0]
        if size > max_frame_bytes:
            issue("frame_limit_exceeded", "ASL frame exceeds configured byte limit", frame_index)
            syntax_ok = False
            break
        start, stop = offset + 4, offset + 4 + size
        if stop > len(data):
            issue("truncated_frame", "ASL frame body is shorter than its length prefix", frame_index)
            syntax_ok = False
            break
        payload = data[start:stop]
        receipt = {"frame_index": frame_index, "byte_offset": offset, "payload_size_bytes": size,
                   "payload_sha256": hashlib.sha256(payload).hexdigest(), "entry_type": None,
                   "semantic_inspection": False}
        report["frame_receipts"].append(receipt)
        offset = stop
        try:
            _check_wire(payload, entry_class.DESCRIPTOR)
            entry = entry_class.FromString(payload)
            kind = entry.WhichOneof("log_entry")
            if kind is None:
                raise _WireError("empty_log_entry", "ASL frame has no log_entry member")
        except (_WireError, DecodeError) as error:
            issue(getattr(error, "code", "invalid_protobuf"), str(error), frame_index)
            syntax_ok = False
            break
        receipt["entry_type"] = kind
        if kind == "rollout_metadata":
            receipt["semantic_inspection"] = True
            current = _metadata(entry.rollout_metadata, issue, frame_index)
            if report["metadata"] is not None:
                issue("duplicate_metadata", "ASL contains multiple metadata entries", frame_index)
                continue
            if frame_index != 0:
                issue("metadata_not_first", "Rollout metadata is not the first frame", frame_index)
            report["metadata"] = current
            report["effective_n_sim_steps"] = current["n_sim_steps"]
            if current["session_uuid"] != expected_session_uuid:
                issue("session_identity_mismatch", "ASL session_uuid differs from expected rollout_id",
                      frame_index)
            if current["scene_id"] != expected_scene_id:
                issue("scene_identity_mismatch", "ASL scene_id differs from expected clipgt_id", frame_index)
            if current["n_sim_steps"] <= 0:
                issue("invalid_effective_steps", "Metadata n_sim_steps must be positive", frame_index)
            elif current["n_sim_steps"] > requested_steps:
                issue("effective_steps_exceed_requested", "Logged effective steps exceed requested steps",
                      frame_index)
            if current["control_timestep_us"] <= 0:
                issue("invalid_control_timestep", "Metadata control timestep must be positive", frame_index)
        elif kind == "actor_poses":
            receipt["semantic_inspection"] = True
            event = {"frame_index": frame_index, "timestamp_us": entry.actor_poses.timestamp_us,
                     "actors": []}
            for actor in entry.actor_poses.actor_poses:
                if not actor.HasField("actor_pose"):
                    pose = None
                    issue("missing_pose", "Actor lacks actor_pose submessage", frame_index)
                else:
                    pose = _pose(actor.actor_pose, issue, frame_index)
                if not actor.actor_id:
                    issue("missing_actor_id", "Actor ID is empty", frame_index)
                event["actors"].append({"actor_id": actor.actor_id, "pose": pose})
            actor_ids = [actor["actor_id"] for actor in event["actors"]]
            if "EGO" not in actor_ids:
                issue("missing_ego_pose", "Actor pose event lacks EGO", frame_index)
            if report["pose_events"] and event["timestamp_us"] < report["pose_events"][-1]["timestamp_us"]:
                issue("timestamp_regression", "Actor pose event timestamps regress", frame_index)
            report["pose_events"].append(event)
    if report["metadata"] is None:
        issue("missing_metadata", "No rollout metadata decoded")
    if not report["pose_events"]:
        issue("missing_pose_events", "No actor pose events decoded")
    else:
        timestamps = [event["timestamp_us"] for event in report["pose_events"]]
        report["observations"] = {
            "pose_event_count": len(timestamps), "first_timestamp_us": timestamps[0],
            "last_timestamp_us": timestamps[-1], "observed_span_us": max(timestamps) - min(timestamps),
            "duplicate_timestamps": sum(count - 1 for count in Counter(timestamps).values()),
        }
    report["parsed"] = syntax_ok
    report["inspection_valid"] = syntax_ok and not report["issues"]
    return report
