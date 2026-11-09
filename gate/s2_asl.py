"""Strict, inspection-only reader for the pinned upstream AlpaSim ASL format.

Tickets: S2 CPU handoff E1; build-spec A4 A1 (streaming, explicit size bound),
A3 (effective-step and hood-mask evidence), R4 C8 (effective steps below
requested) and R4 C9 (repeated actor ID in one frame). Serves S2.1-S2.7 [C].

Framing and the complete protobuf descriptor come from public e121e591 source.
Original local-to-AABB poses are not transformed, normalized, interpolated or
deduplicated. Parsing establishes neither provenance nor runtime completion.

Verification level per function (build-spec section 1):

- ``inspect_asl`` / ``inspect_asl_stream``: L2. Literal wire known answers and
  negative controls on CPU; checked against the pinned .proto set, the pinned
  ``write_pb_stream`` writer and the pinned glam quaternion consumer. Real
  simulator ASL files are ``qualify_on_host``.
- ``_check_wire``, ``_check_quaternion``, ``_varint``: L1.

Size bounds: ``MAX_FILE_BYTES`` (2 GiB) is an agent technical default (D-018),
provisional until the first measured ASL. R2 infers 0.3-0.8 GB per 200-step
six-camera rollout (every driver JPEG is logged); the bound is about 2.5x the
upper inference. ``inspect_asl_stream`` reads one bounded frame at a
time (``MAX_FRAME_BYTES``), hashing incrementally, so the file bound limits
work, not RAM; the decoded report still holds every pose sample and frame
receipt. Callers choose the bound; the artifact never authorizes it.

This module never: transforms or deduplicates poses; infers executed steps,
completion, renderer/black-frame/map coverage or "hood applied"; admits scores;
authenticates the producer; reads files or calls subprocesses itself.
"""

import hashlib
import io
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
# Agent technical default (D-018): provisional until the first measured ASL.
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_FRAMES = 1_000_000
_READ_CHUNK_BYTES = 1024 * 1024
SESSION_IDENTITY_ENTRIES = frozenset({
    "driver_request", "traffic_request", "driver_session_request", "driver_camera_image",
    "driver_ego_trajectory", "traffic_session_request", "controller_request",
    "route_request", "ground_truth_request",
})
# Pinned runtime logs RGB render requests through three LogEntry members:
# sensorsim_service.py:453 (batch_render_request), :514 (aggregated_render_request)
# and :594 (render_request); each request sets insert_ego_mask (:309).
_RENDER_ENTRIES = frozenset({"render_request", "aggregated_render_request",
                             "batch_render_request"})


def _f32(value):
    """Round one source float32 operation, retaining overflow as infinity."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


# Exact source consumer constants: pinned utils_rs/array_utils.rs ensure_unit_quat.
# These are serialization/consumer compatibility checks, not scientific tolerances.
_QUAT_NEAR_ZERO_LEN_SQ = _f32(1e-12)
_QUAT_UNIT_LEN_SQ_TOL = _f32(1e-4)


def _check_quaternion(values, issue, frame_index):
    """Apply the pinned glam 0.29.3 float32 backend checks without normalization.

    SSE2, scalar and NEON use different addition orders. If their classifications
    disagree at the exact source boundary, retain that uncertainty and fail the
    portable inspection rather than choose a backend or enlarge the tolerance.
    Source/license/hash receipts live in the quaternion_compat source fixtures.
    This does not qualify a compiler, floating-point mode or actual runtime.
    """
    w, x, y, z = values
    xx, yy, zz, ww = (_f32(value * value) for value in (x, y, z, w))
    lengths = (
        _f32(_f32(xx + zz) + _f32(yy + ww)),  # SSE2
        _f32(_f32(_f32(xx + yy) + zz) + ww),  # scalar
        _f32(_f32(xx + yy) + _f32(zz + ww)),  # NEON pairwise reduction
    )
    accepted = [
        length >= _QUAT_NEAR_ZERO_LEN_SQ
        and abs(_f32(length - 1.0)) <= _QUAT_UNIT_LEN_SQ_TOL
        for length in lengths
    ]
    if all(accepted):
        return
    if any(accepted):
        issue("quaternion_rounding_ambiguous",
              "Pinned upstream float32 quaternion reduction layouts disagree at the boundary",
              frame_index)
    else:
        issue("upstream_quaternion_incompatible",
              "Quaternion fails the pinned upstream near-zero/unit-length consumer checks",
              frame_index)


class _WireError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class _FileLimitExceeded(Exception):
    pass


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
        elif name == "quat":
            _check_quaternion(values, issue, frame_index)
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


def _render_requests(entry, kind):
    if kind == "render_request":
        return [entry.render_request]
    if kind == "aggregated_render_request":
        return list(entry.aggregated_render_request.rgb_requests)
    return [item.request for item in entry.batch_render_request.items]


def _hood_record(hood, entry, kind):
    """Record ego-mask facts only; presence of a mask request is not 'hood applied'.

    Proto3 omits ``insert_ego_mask=false``; requests without the flag count as
    ``without_insert_ego_mask``. ``available_ego_masks_return`` (logging.proto
    field 23, logged at sensorsim_service.py:194) lists the renderer's masks.
    """
    if kind == "available_ego_masks_return":
        hood["available_ego_masks_returns"] += 1
        for metadata in entry.available_ego_masks_return.ego_mask_metadata:
            mask = metadata.ego_mask_id
            label = f"{mask.rig_config_id}/{mask.camera_logical_id}"
            if label not in hood["available_ego_masks"]:
                hood["available_ego_masks"].append(label)
        return
    for request in _render_requests(entry, kind):
        camera = request.camera_intrinsics.logical_id
        counts = hood["by_camera"].setdefault(
            camera, {"rgb_render_requests": 0, "with_insert_ego_mask": 0, "ego_mask_ids": []})
        hood["rgb_render_requests"] += 1
        counts["rgb_render_requests"] += 1
        if request.insert_ego_mask:
            hood["with_insert_ego_mask"] += 1
            counts["with_insert_ego_mask"] += 1
            mask = request.ego_mask_id
            label = f"{mask.rig_config_id}/{mask.camera_logical_id}"
            if label not in counts["ego_mask_ids"]:
                counts["ego_mask_ids"].append(label)


def _new_report(requested_steps):
    return {
        "schema_version": 1, "kind": "alpasim_asl_inspection_v1",
        "source_commit": SOURCE_COMMIT, "parsed": False, "inspection_valid": False,
        "issues": [], "receipt": None, "metadata": None, "frame_receipts": [],
        "pose_events": [], "coordinate_frame": "local_to_aabb",
        "quaternion_component_order": "wxyz",
        "requested_steps": requested_steps if type(requested_steps) is int else None,
        "effective_n_sim_steps": None, "horizon_status": None, "actual_steps": None,
        "completion": None, "renderer_coverage": None, "black_frame_coverage": None,
        "map_coverage": None, "hood_mask_evidence": None, "runtime_qualified": False,
        "score_admitted": False, "cloud_launch_authorized": False,
        "observations": {"pose_event_count": 0, "first_timestamp_us": None,
                         "last_timestamp_us": None, "observed_span_us": None,
                         "duplicate_timestamps": 0},
    }


def _issue_adder(report):
    def issue(code, message, frame_index=None, severity="blocking"):
        item = {
            "code": code,
            "path": "asl" if frame_index is None else f"asl.frames[{frame_index}]",
            "severity": severity,
            "evidence_refs": [] if frame_index is None else [f"frame_receipts[{frame_index}]"],
            "message": message,
        }
        if frame_index is not None:
            item["frame_index"] = frame_index
        report["issues"].append(item)
    return issue


def _arguments_valid(expected_session_uuid, expected_scene_id, requested_steps, limits):
    return (
        isinstance(expected_session_uuid, str) and bool(expected_session_uuid)
        and isinstance(expected_scene_id, str) and bool(expected_scene_id)
        and all(type(value) is int and value > 0 for value in (requested_steps, *limits))
    )


def inspect_asl(data, *, expected_session_uuid, expected_scene_id, requested_steps,
                max_file_bytes=MAX_FILE_BYTES, max_frame_bytes=MAX_FRAME_BYTES,
                max_frames=MAX_FRAMES):
    """Inspect in-memory ASL bytes; a thin wrapper over ``inspect_asl_stream``.

    Session UUID binds to upstream ``rollout_id``; scene ID to ``clipgt_id``.
    ``run_uuid`` is external evaluator run metadata, absent from this binding.
    Bytes larger than ``max_file_bytes`` are rejected before parsing; the whole
    file receipt is still reported. Prefer the stream API for real captures.
    """
    report = _new_report(requested_steps)
    issue = _issue_adder(report)
    if type(data) is not bytes or not _arguments_valid(
            expected_session_uuid, expected_scene_id, requested_steps,
            (max_file_bytes, max_frame_bytes, max_frames)):
        issue("invalid_arguments",
              "Expected bytes, explicit nonempty IDs and positive integer limits")
        return report
    if len(data) > max_file_bytes:
        report["receipt"] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        issue("file_limit_exceeded", "ASL exceeds configured file size limit")
        return report
    return inspect_asl_stream(
        io.BytesIO(data), expected_session_uuid=expected_session_uuid,
        expected_scene_id=expected_scene_id, requested_steps=requested_steps,
        max_file_bytes=max_file_bytes, max_frame_bytes=max_frame_bytes, max_frames=max_frames)


def inspect_asl_stream(stream, *, expected_session_uuid, expected_scene_id, requested_steps,
                       max_file_bytes=MAX_FILE_BYTES, max_frame_bytes=MAX_FRAME_BYTES,
                       max_frames=MAX_FRAMES):
    """Inspect pinned framed protobuf from ``stream.read(n)`` one frame at a time.

    Never requests more than ``_READ_CHUNK_BYTES`` per read and never holds more
    than one frame payload. The SHA-256/size receipt covers every byte read; after
    a framing error the remainder is drained (hashed, not parsed) so the receipt
    still describes the whole input. Exceeding ``max_file_bytes`` stops reading:
    ``receipt`` is then ``None`` and ``bytes_read`` records the stopping point.
    Invalid input preserves decoded-prefix diagnostics and frame receipts.
    Warnings (``severity != "blocking"``) never make ``inspection_valid`` false.
    """
    report = _new_report(requested_steps)
    issue = _issue_adder(report)
    if not hasattr(stream, "read") or not _arguments_valid(
            expected_session_uuid, expected_scene_id, requested_steps,
            (max_file_bytes, max_frame_bytes, max_frames)):
        issue("invalid_arguments",
              "Expected a readable stream, explicit nonempty IDs and positive integer limits")
        return report
    try:
        entry_class = _entry_class()
    except (OSError, ValueError, DecodeError) as error:
        issue("schema_unavailable", str(error))
        return report

    hasher, consumed = hashlib.sha256(), [0]

    def take(count):
        parts, remaining = [], count
        while remaining > 0:
            request = min(remaining, _READ_CHUNK_BYTES)
            chunk = stream.read(request)
            if type(chunk) is not bytes or len(chunk) > request:
                raise TypeError("ASL stream must return at most the requested bytes")
            if not chunk:
                break
            if consumed[0] + len(chunk) > max_file_bytes:
                consumed[0] += len(chunk)
                raise _FileLimitExceeded
            hasher.update(chunk)
            consumed[0] += len(chunk)
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

    hood = {"available_ego_masks_returns": 0, "available_ego_masks": [],
            "rgb_render_requests": 0, "with_insert_ego_mask": 0, "by_camera": {}}
    syntax_ok, offset, complete_read = True, 0, False
    try:
        while True:
            frame_index = len(report["frame_receipts"])
            header = take(4)
            if not header:
                complete_read = True
                break
            if frame_index >= max_frames:
                issue("frame_count_exceeded", "ASL exceeds configured frame count", frame_index)
                syntax_ok = False
                break
            if len(header) < 4:
                issue("truncated_header", "ASL length prefix has fewer than four bytes",
                      frame_index)
                syntax_ok = False
                break
            size = struct.unpack(">L", header)[0]
            if size > max_frame_bytes:
                issue("frame_limit_exceeded", "ASL frame exceeds configured byte limit",
                      frame_index)
                syntax_ok = False
                break
            payload = take(size)
            if len(payload) < size:
                issue("truncated_frame", "ASL frame body is shorter than its length prefix",
                      frame_index)
                syntax_ok = False
                break
            receipt = {"frame_index": frame_index, "byte_offset": offset,
                       "payload_size_bytes": size,
                       "payload_sha256": hashlib.sha256(payload).hexdigest(),
                       "entry_type": None, "semantic_inspection": False,
                       "session_identity_checked": False}
            report["frame_receipts"].append(receipt)
            offset += 4 + size
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
            _inspect_entry(report, issue, hood, entry, kind, receipt, frame_index,
                           expected_session_uuid, expected_scene_id, requested_steps)
        if not complete_read:
            while take(_READ_CHUNK_BYTES):
                pass
            complete_read = True
    except _FileLimitExceeded:
        issue("file_limit_exceeded", "ASL exceeds configured file size limit")
        report["bytes_read"] = consumed[0]
        report["hood_mask_evidence"] = _hood_summary(hood)
        return report
    report["receipt"] = {"sha256": hasher.hexdigest(), "size_bytes": consumed[0]}
    report["hood_mask_evidence"] = _hood_summary(hood)
    _finish(report, issue, syntax_ok)
    return report


def _hood_summary(hood):
    present = hood["available_ego_masks_returns"] > 0 or hood["rgb_render_requests"] > 0
    return {
        "status": "present" if present else "absent",
        "available_ego_masks_returns": hood["available_ego_masks_returns"],
        "available_ego_masks": hood["available_ego_masks"],
        "rgb_render_requests": hood["rgb_render_requests"],
        "with_insert_ego_mask": hood["with_insert_ego_mask"],
        "without_insert_ego_mask": hood["rgb_render_requests"] - hood["with_insert_ego_mask"],
        "by_camera": {camera: hood["by_camera"][camera] for camera in sorted(hood["by_camera"])},
        "hood_applied": None,
        "semantics": "request_flags_only_not_rendered_pixels",
    }


def _inspect_entry(report, issue, hood, entry, kind, receipt, frame_index,
                   expected_session_uuid, expected_scene_id, requested_steps):
    if kind in SESSION_IDENTITY_ENTRIES:
        service_session = getattr(entry, kind).session_uuid
        receipt["session_identity_checked"] = True
        receipt["session_uuid"] = service_session
        if service_session != expected_session_uuid:
            issue("service_session_identity_mismatch",
                  f"{kind} session_uuid differs from the bound rollout_id", frame_index)
    if kind in _RENDER_ENTRIES or kind == "available_ego_masks_return":
        receipt["hood_fields_inspected"] = True
        _hood_record(hood, entry, kind)
    if kind == "rollout_metadata":
        receipt["semantic_inspection"] = True
        current = _metadata(entry.rollout_metadata, issue, frame_index)
        if report["metadata"] is not None:
            issue("duplicate_metadata", "ASL contains multiple metadata entries", frame_index)
            return
        if frame_index != 0:
            issue("metadata_not_first", "Rollout metadata is not the first frame", frame_index)
        report["metadata"] = current
        effective = current["n_sim_steps"]
        report["effective_n_sim_steps"] = effective
        if current["session_uuid"] != expected_session_uuid:
            issue("session_identity_mismatch",
                  "ASL session_uuid differs from expected rollout_id", frame_index)
        if current["scene_id"] != expected_scene_id:
            issue("scene_identity_mismatch", "ASL scene_id differs from expected clipgt_id",
                  frame_index)
        if effective <= 0:
            issue("invalid_effective_steps", "Metadata n_sim_steps must be positive", frame_index)
        elif effective > requested_steps:
            report["horizon_status"] = "effective_exceeds_requested"
            issue("effective_steps_exceed_requested",
                  f"Logged effective steps {effective} exceed requested steps "
                  f"{requested_steps}", frame_index)
        elif effective < requested_steps:
            # Upstream clips to ground-truth availability (unbound_rollout.py:163-171).
            report["horizon_status"] = "effective_below_requested"
            issue("effective_steps_below_requested",
                  f"Logged effective steps {effective} are below requested steps "
                  f"{requested_steps}; horizon is reported, not hidden", frame_index,
                  severity="warning")
        else:
            report["horizon_status"] = "effective_equals_requested"
        if current["control_timestep_us"] <= 0:
            issue("invalid_control_timestep", "Metadata control timestep must be positive",
                  frame_index)
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
        repeated = sorted(actor_id for actor_id, count in Counter(actor_ids).items()
                          if count > 1 and actor_id)
        if repeated:
            # R4 C9. Upstream log_actor_poses (step.py:141-178) appends one pose per
            # occurrence of a timestamp, so repeated timestamps repeat IDs; any repeat
            # is ambiguous for a per-actor consumer. All samples stay preserved.
            issue("duplicate_actor_id",
                  f"Actor IDs repeat within one actor_poses frame: {repeated}", frame_index)
        if "EGO" not in actor_ids:
            issue("missing_ego_pose", "Actor pose event lacks EGO", frame_index)
        if (report["pose_events"]
                and event["timestamp_us"] < report["pose_events"][-1]["timestamp_us"]):
            issue("timestamp_regression", "Actor pose event timestamps regress", frame_index)
        report["pose_events"].append(event)


def _finish(report, issue, syntax_ok):
    if report["metadata"] is None:
        issue("missing_metadata", "No rollout metadata decoded")
    if not report["pose_events"]:
        issue("missing_pose_events", "No actor pose events decoded")
    else:
        timestamps = [event["timestamp_us"] for event in report["pose_events"]]
        report["observations"] = {
            "pose_event_count": len(timestamps), "first_timestamp_us": timestamps[0],
            "last_timestamp_us": timestamps[-1],
            "observed_span_us": max(timestamps) - min(timestamps),
            "duplicate_timestamps": sum(count - 1 for count in Counter(timestamps).values()),
        }
    report["parsed"] = syntax_ok
    report["inspection_valid"] = syntax_ok and not any(
        item["severity"] == "blocking" for item in report["issues"])
