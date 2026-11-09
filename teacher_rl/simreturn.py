"""Strict decoder for captured AlpaSim ``SimulationReturn`` bytes at the training pin (G3).

Tickets: completeness-audit high gap "S4.2-S4.4 capture to checks" (``simulate_call`` reply bytes
were never decoded), E10-E14 inputs (dense timestep metrics and online aggregates per rollout),
E01 (simulator outputs compared across runs). Serves S4.1-S4.4 [C] (G9, G7) under D-024.
Agent-authored evaluator engineering (D-018).

The capture (``teacher_rl.capture.entrypoint._SimulateProxy``) stores
``reply.SerializeToString()`` of every ``RuntimeService.simulate`` reply as a content-addressed
blob. This module decodes those bytes with a hash-checked descriptor set compiled from the
pinned training AlpaSim protos (``alpasim@46b401b src/grpc/alpasim_grpc/v0/runtime.proto`` and
its one import ``common.proto``; vendored byte-identical under ``teacher_rl/proto/`` with the
Apache-2.0 LICENSE and ``provenance.json``). ``runtime.proto`` declares no package, so the type
is ``SimulationReturn``. Only the protobuf runtime is used (no grpc dependency).

Output (``s4_simulation_return`` v1): the six service ``VersionId`` fields and every
``RolloutReturn`` as ``{index, scenario_id, nr_rollouts, session_uuids, success, rollout_uuid,
error, timestep_metrics: [{name, timestamps_us, values, valid, time_aggregation}],
aggregated_metrics: {name: value}}``. Values keep the module value convention of
``teacher_rl.checks.semantics``: finite floats, or the tokens ``"nan"``, ``"inf"``, ``"-inf"``.
``time_aggregation`` is the lower-case name the pinned producer maps from
``eval.data.AggregationType`` (``engine.py:28-36``); ``TIME_AGGREGATION_UNSPECIFIED`` or an
unknown enum number is an issue and the value is ``None`` (never defaulted).

Fail closed: before parsing, a structural walk over the full descriptor rejects truncation,
invalid tags or wire types, unknown fields (unsupported under this pin), a repeated singular
field, a duplicate ``aggregated_metrics`` key (the permissive parser keeps the last one
silently) and a varint wider than its declared type (``varint_out_of_range``: ``bool`` other
than 0/1, ``uint32`` at or above 2**32, ``enum`` outside int32; the runtime would truncate
them silently), including packed elements. Any such defect, invalid UTF-8, or a reply above
the pinned AlpaGym gRPC receive limit (``MAX_REPLY_BYTES``) gives ``status: blocked`` and
``rollout_returns: None``: malformed bytes are issues, never zero or empty. Per-metric defects
(series of unequal length, an unspecified aggregation, an empty or repeated dense name, an
``aggregated_metrics`` entry whose key is absent or empty) are blocking issues located by
path; the rollout is kept as decoded but ``to_semantics_input`` refuses it. The walk checks
wire structure and declared widths only; semantic values (an unknown enum number, a negative
int32 enum) are reported by the per-metric checks, never repaired. Zero rollout returns is
``no_rollout_returns`` (AlpaGym reads ``rollout_returns[0]`` and would raise ``IndexError``,
which upstream turns into a silent retry).

Verification level per function (build-spec section 1):

- ``decode_simulation_return``, ``to_semantics_input``, ``message_class``: **L2** - literal
  hand-encoded wire known answers and negative controls in ``tests/test_s4_simreturn.py``,
  checked against the vendored pinned ``.proto`` text and the pinned producer/reader; real
  simulator replies are ``qualify_on_host``.
- ``_walk``, ``_varint``, ``_check_range``, ``_metric``: **L1**.

This module never reads files other than its vendored descriptor and its own source, never
calls a subprocess or the network, never imports grpc or torch, never computes a reward, never
interpolates, fills or reorders a series, and never decides whether a rollout is valid.
"""

import hashlib
import math
from functools import lru_cache
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError

ARTIFACT_KIND = "s4_simulation_return"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
SOURCE_COMMIT = "46b401beaec27977cbb86826df8e39cec2d49e49"
DESCRIPTOR_FILE = "46b401be.pb"
DESCRIPTOR_SHA256 = "a77ac18f62f85ac75ae5d4add305bc850d6f4881ddd8d8d69936e42aaa5be634"
MESSAGE = "SimulationReturn"
# Pinned AlpaGym gRPC receive limit (alpagym@972d160 cosmos/rollout_backend.py:38, "matches
# AlpaSim runtime defaults"): a larger reply cannot have reached the upstream reader.
MAX_REPLY_BYTES = 256 * 1024 * 1024
# runtime.proto TimeAggregation (:76-83) -> eval.data.AggregationType names (engine.py:28-36).
AGGREGATIONS = {1: "mean", 2: "median", 3: "max", 4: "min", 5: "last"}
VERSION_FIELDS = ("runtime_version", "nre_version", "physics_version", "driver_version",
                  "traffic_version", "video_model_version")
_MAX_DEPTH = 32


class _WireError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@lru_cache(maxsize=1)
def message_class():
    """The ``SimulationReturn`` class from the hash-checked vendored descriptor set (L2)."""
    data = (Path(__file__).parent / "proto" / DESCRIPTOR_FILE).read_bytes()
    if hashlib.sha256(data).hexdigest() != DESCRIPTOR_SHA256:
        raise ValueError("vendored SimulationReturn descriptor does not match its pinned hash")
    files = descriptor_pb2.FileDescriptorSet.FromString(data)
    pool = descriptor_pool.DescriptorPool()
    for file in files.file:
        pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName(MESSAGE))


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _varint(data, offset, end):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= end:
            raise _WireError("invalid_protobuf", "truncated varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value > 2**64 - 1:
                break
            return value, offset
    raise _WireError("invalid_protobuf", "varint exceeds 64 bits")


def _wire_type(field):
    if field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FIXED64,
                      FieldDescriptor.TYPE_SFIXED64):
        return 1
    if field.type in (FieldDescriptor.TYPE_STRING, FieldDescriptor.TYPE_BYTES,
                      FieldDescriptor.TYPE_MESSAGE):
        return 2
    if field.type in (FieldDescriptor.TYPE_FLOAT, FieldDescriptor.TYPE_FIXED32,
                      FieldDescriptor.TYPE_SFIXED32):
        return 5
    return 0


_INT32_TYPES = (FieldDescriptor.TYPE_INT32, FieldDescriptor.TYPE_ENUM)
_UINT32_TYPES = (FieldDescriptor.TYPE_UINT32, FieldDescriptor.TYPE_SINT32)


def _check_range(field, value):
    """Reject a varint wider than its declared type (the runtime would truncate it, L1).

    ``bool`` must be 0 or 1; ``uint32``/``sint32`` below 2**32; ``int32``/``enum`` within int32
    (a negative is sign-extended to ten bytes on the wire). 64-bit types take any 64-bit value.
    """
    if field.type == FieldDescriptor.TYPE_BOOL:
        ok = value in (0, 1)
    elif field.type in _UINT32_TYPES:
        ok = value < 2**32
    elif field.type in _INT32_TYPES:
        ok = value < 2**31 or value >= 2**64 - 2**31
    else:
        ok = True
    if not ok:
        raise _WireError("varint_out_of_range",
                         f"{field.full_name} varint {value} exceeds its declared type")


def _skip_scalar(data, offset, end, wire, field=None):
    if wire == 0:
        value, offset = _varint(data, offset, end)
        if field is not None:
            _check_range(field, value)
        return offset
    width = 8 if wire == 1 else 4
    if offset + width > end:
        raise _WireError("invalid_protobuf", "truncated fixed-width scalar")
    return offset + width


def _map_key(data, start, end):
    """The string key (field 1) of one map entry; proto3 default ``""`` when absent."""
    offset, key = start, b""
    while offset < end:
        tag, offset = _varint(data, offset, end)
        number, wire = tag >> 3, tag & 7
        if wire == 2:
            size, offset = _varint(data, offset, end)
            if number == 1:
                key = bytes(data[offset:offset + size])
            offset += size
        else:
            offset = _skip_scalar(data, offset, end, wire)
    return key


def _walk(data, descriptor, start, end, depth=0):
    """Reject what the permissive parser would accept silently (structure only, L1)."""
    if depth > _MAX_DEPTH:
        raise _WireError("protobuf_depth_exceeded", f"nesting exceeds {_MAX_DEPTH} levels")
    offset, seen, keys = start, set(), {}
    while offset < end:
        tag, offset = _varint(data, offset, end)
        number, wire = tag >> 3, tag & 7
        if number == 0 or number > 2**29 - 1 or wire not in (0, 1, 2, 5):
            raise _WireError("invalid_protobuf", "invalid tag or wire type")
        field = descriptor.fields_by_number.get(number)
        if field is None:
            raise _WireError("unsupported_field",
                             f"unknown field {descriptor.full_name}.{number} at this pin")
        if not field.is_repeated:
            if number in seen:
                raise _WireError("ambiguous_field", f"singular {field.full_name} repeated")
            seen.add(number)
        expected = _wire_type(field)
        packed = field.is_repeated and wire == 2 and expected != 2
        if wire != expected and not packed:
            raise _WireError("invalid_protobuf", f"wrong wire type for {field.full_name}")
        if wire != 2:
            offset = _skip_scalar(data, offset, end, wire, field)
            continue
        size, offset = _varint(data, offset, end)
        stop = offset + size
        if stop > end:
            raise _WireError("invalid_protobuf", f"truncated {field.full_name}")
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            _walk(data, field.message_type, offset, stop, depth + 1)
            if field.message_type.GetOptions().map_entry:
                key = _map_key(data, offset, stop)
                if key in keys.setdefault(number, set()):
                    raise _WireError("ambiguous_field", f"duplicate key {key!r} in map "
                                                        f"{field.full_name}")
                keys[number].add(key)
        elif packed:
            cursor = offset
            while cursor < stop:
                cursor = _skip_scalar(data, cursor, stop, expected, field)
        offset = stop


def _token(value):
    if math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")


def _version(message, name):
    if not message.HasField(name):
        return None
    vid = getattr(message, name)
    api = vid.grpc_api_version if vid.HasField("grpc_api_version") else None
    return {"version_id": vid.version_id, "git_hash": vid.git_hash,
            "grpc_api_version": None if api is None else [api.major, api.minor, api.patch]}


def _metric(metric, path, issues):
    timestamps = [int(t) for t in metric.timestamps_us]
    values = [_token(float(v)) for v in metric.values]
    valid = [bool(v) for v in metric.valid]
    number = int(metric.time_aggregation)
    aggregation = AGGREGATIONS.get(number)
    if not metric.name:
        issues.append(_issue("metric_name_empty", f"{path}.name", "a metric has no name"))
    if not len(timestamps) == len(values) == len(valid):
        issues.append(_issue("metric_series_misaligned", path,
                             f"timestamps_us/values/valid lengths {len(timestamps)}/"
                             f"{len(values)}/{len(valid)} differ; nothing is padded"))
    if aggregation is None:
        code = "aggregation_unspecified" if number == 0 else "aggregation_unknown"
        issues.append(_issue(code, f"{path}.time_aggregation",
                             f"TimeAggregation {number} has no pinned producer mapping"))
    return {"name": metric.name, "timestamps_us": timestamps, "values": values, "valid": valid,
            "time_aggregation": aggregation}


def _rollout(index, ret, issues):
    path = f"rollout_returns[{index}]"
    metrics = []
    for position, metric in enumerate(ret.timestep_metrics):
        metrics.append(_metric(metric, f"{path}.timestep_metrics[{position}]", issues))
    names = [m["name"] for m in metrics]
    for name in sorted({n for n in names if names.count(n) > 1}):
        issues.append(_issue("duplicate_metric_name", f"{path}.timestep_metrics",
                             f"metric {name!r} appears {names.count(name)} times"))
    if "" in ret.aggregated_metrics:
        issues.append(_issue("aggregate_name_empty", f"{path}.aggregated_metrics",
                             "an aggregated_metrics entry has no key (absent or empty)"))
    spec = ret.rollout_spec
    return {"index": index, "scenario_id": spec.scenario_id, "nr_rollouts": spec.nr_rollouts,
            "session_uuids": list(spec.session_uuids), "success": bool(ret.success),
            "rollout_uuid": ret.rollout_uuid, "error": ret.error,
            "timestep_metrics": metrics,
            "aggregated_metrics": {k: _token(float(v))
                                   for k, v in sorted(ret.aggregated_metrics.items())}}


def _producer():
    return {"module": __name__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "descriptor_sha256": DESCRIPTOR_SHA256, "source_commit": SOURCE_COMMIT}


def decode_simulation_return(data, *, scope):
    """Decode one captured ``SimulationReturn`` (module docstring; L2)."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES} (required, no default)")
    issues, versions, returns = [], None, None
    digest = hashlib.sha256(data).hexdigest() if type(data) is bytes else None
    if type(data) is not bytes:
        issues.append(_issue("input_not_bytes", "$", f"expected bytes, got {type(data).__name__}"))
    elif len(data) > MAX_REPLY_BYTES:
        issues.append(_issue("reply_exceeds_grpc_limit", "$",
                             f"{len(data)} bytes exceed the pinned receive limit "
                             f"{MAX_REPLY_BYTES}"))
    else:
        cls = message_class()
        try:
            _walk(data, cls.DESCRIPTOR, 0, len(data))
            message = cls.FromString(data)
        except _WireError as exc:
            issues.append(_issue(exc.code, "$", str(exc)))
        except DecodeError as exc:
            issues.append(_issue("invalid_protobuf", "$", f"protobuf runtime refused: {exc}"))
        else:
            versions = {name: _version(message, name) for name in VERSION_FIELDS}
            returns = [_rollout(i, r, issues) for i, r in enumerate(message.rollout_returns)]
            if not returns:
                issues.append(_issue("no_rollout_returns", "rollout_returns",
                                     "the reply holds no RolloutReturn; the pinned reader "
                                     "indexes [0] (IndexError -> silent retry)"))
    blocking = any(i["severity"] == "blocking" for i in issues)
    return {"artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": scope,
            "producer": _producer(),
            "input_hashes": {"simulation_return": digest},
            "units": {"timestamps_us": "microseconds", "bytes": "bytes",
                      "values": "as emitted by the pinned AlpaSim scorers"},
            "status": "blocked" if blocking else "passed", "issues": issues,
            "bytes": len(data) if type(data) is bytes else None,
            "versions": versions, "rollout_returns": returns}


def _defects(decoded, index):
    prefix = f"rollout_returns[{index}]"
    return sorted({i["code"] for i in decoded["issues"] if i["severity"] == "blocking"
                   and (i["path"] == prefix or i["path"].startswith(prefix + "."))})


def to_semantics_input(decoded, index):
    """The ``teacher_rl.checks.semantics`` rollout of one decoded return (L2).

    ``{session_uuid, timestep_metrics, aggregated_metrics}`` where ``aggregated_metrics`` is
    ``None`` when the reply map is empty, as the pinned reader builds
    ``EpisodeMetrics(...) if aggregated else None`` (``streaming_worker.py:240``). Raises
    ``ValueError`` for an undecoded artifact, a failed rollout (upstream reads no metrics from
    it), a rollout spec without exactly one session UUID (AlpaGym sends one per call) or any
    blocking defect located in that rollout. Nothing is filled or repaired.
    """
    if (type(decoded) is not dict or decoded.get("artifact_kind") != ARTIFACT_KIND
            or type(decoded.get("rollout_returns")) is not list):
        raise ValueError("a decoded s4_simulation_return with rollout_returns is required")
    returns = decoded["rollout_returns"]
    if type(index) is not int or not 0 <= index < len(returns):
        raise ValueError(f"rollout index {index!r} is outside the decoded returns")
    ret = returns[index]
    defects = _defects(decoded, index)
    if defects:
        raise ValueError(f"rollout_returns[{index}] has blocking defects {defects}")
    if not ret["success"]:
        raise ValueError(f"rollout_returns[{index}] failed upstream: {ret['error']!r}")
    if len(ret["session_uuids"]) != 1:
        raise ValueError(f"rollout_returns[{index}] names {len(ret['session_uuids'])} session "
                         "UUIDs; exactly one is required to key the rollout")
    return {"session_uuid": ret["session_uuids"][0],
            "timestep_metrics": [dict(m) for m in ret["timestep_metrics"]],
            "aggregated_metrics": dict(ret["aggregated_metrics"]) or None}
