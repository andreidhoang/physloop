"""Known answers and negative controls for teacher_rl.simreturn (G3; E10-E14 inputs, E01).

The wire bytes are hand-encoded here with ``struct`` from the field numbers and types written
in the vendored pinned ``runtime.proto`` (read as text, not through the module), so the oracle
is independent of the decoder and of the protobuf runtime.
"""

import hashlib
import json
import math
import struct
from pathlib import Path

import pytest

from teacher_rl import simreturn
from teacher_rl.checks import semantics

PROTO = Path(simreturn.__file__).parent / "proto"


# ---- an independent proto3 wire encoder ---------------------------------------------------


def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def tag(number, wire):
    return varint(number << 3 | wire)


def ld(number, payload):
    return tag(number, 2) + varint(len(payload)) + payload


def text(number, value):
    return ld(number, value.encode())


def boolean(number, value):
    return tag(number, 0) + varint(int(value))


def fixed64s(number, values):
    return ld(number, b"".join(struct.pack("<Q", v) for v in values))


def doubles(number, values):
    return ld(number, b"".join(struct.pack("<d", v) for v in values))


def bools(number, values):
    return ld(number, b"".join(varint(int(v)) for v in values))


def metric(name, timestamps, values, valid, aggregation):
    # SimulationReturn.TimestepMetric: name=1 string, timestamps_us=2 repeated fixed64,
    # values=3 repeated double, valid=4 repeated bool, time_aggregation=5 enum
    return (text(1, name) + fixed64s(2, timestamps) + doubles(3, values) + bools(4, valid)
            + tag(5, 0) + varint(aggregation))


def map_entry(key, value):
    # map<string, double>: key=1, value=2 (double, wire type 1)
    return text(1, key) + tag(2, 1) + struct.pack("<d", value)


def spec(scenario, sessions):
    # RolloutSpec: scenario_id=1, nr_rollouts=2 (uint32), session_uuids=3
    return text(1, scenario) + tag(2, 0) + varint(1) + b"".join(text(3, s) for s in sessions)


def rollout(*, sessions=("u1",), success=True, metrics=(), aggregated=(), error=""):
    # RolloutReturn: rollout_spec=1, success=2, rollout_uuid=3, error=4,
    # timestep_metrics=5, aggregated_metrics=6 (map)
    body = ld(1, spec("clipgt-a", sessions)) + boolean(2, success) + text(3, "r1")
    if error:
        body += text(4, error)
    body += b"".join(ld(5, m) for m in metrics)
    body += b"".join(ld(6, map_entry(k, v)) for k, v in aggregated)
    return body


def version(version_id, git_hash, api=(0, 1, 2)):
    # common.VersionId: version_id=1, git_hash=2, grpc_api_version=3 {major=1, minor=2, patch=3}
    triple = b"".join(tag(n, 0) + varint(v) for n, v in zip((1, 2, 3), api))
    return text(1, version_id) + text(2, git_hash) + ld(3, triple)


def reply(*rollouts, runtime=True):
    # SimulationReturn: runtime_version=1 ... rollout_returns=6
    head = ld(1, version("rt-1", "abc")) if runtime else b""
    return head + b"".join(ld(6, r) for r in rollouts)


COLLISION = metric("collision_any", [0, 100_000], [1.5, math.nan], [True, False], 3)
GOOD = reply(rollout(metrics=[COLLISION], aggregated=[("collision_any", 1.5)]))


def decode(data):
    return simreturn.decode_simulation_return(data, scope="synthetic")


def codes(report):
    return sorted({i["code"] for i in report["issues"]})


# ---- vendored provenance -------------------------------------------------------------------


def test_vendored_files_match_their_provenance():
    prov = json.loads((PROTO / "provenance.json").read_text())
    assert prov["commit"] == simreturn.SOURCE_COMMIT == "46b401beaec27977cbb86826df8e39cec2d49e49"
    for entry in prov["sources"] + [prov["license_file"], prov["descriptor"]]:
        assert hashlib.sha256((PROTO / entry["file"]).read_bytes()).hexdigest() == (
            entry["sha256"]), entry["file"]
    assert prov["descriptor"]["sha256"] == simreturn.DESCRIPTOR_SHA256
    # the same Apache-2.0 file gate/asl_schema vendors from the S2 pin
    assert prov["license_file"]["sha256"] == (
        "50201750cc0542d4656d8d95f002f78357021b7c3c975b53924000b7ddc5a24e")
    assert "Apache License" in (PROTO / "LICENSE").read_text()


def test_descriptor_field_numbers_are_the_pinned_proto_text():
    source = (PROTO / "source/alpasim_grpc/v0/runtime.proto").read_text()
    for line in ("repeated fixed64 timestamps_us = 2;", "repeated double values = 3;",
                 "repeated bool valid = 4;", "TimeAggregation time_aggregation = 5;",
                 "map<string, double> aggregated_metrics = 6;",
                 "repeated RolloutReturn rollout_returns = 6;",
                 "TIME_AGGREGATION_UNSPECIFIED = 0;", "TIME_AGGREGATION_LAST = 5;"):
        assert line in source
    cls = simreturn.message_class()
    tm = cls.DESCRIPTOR.nested_types_by_name["TimestepMetric"]
    assert {f.name: f.number for f in tm.fields} == {
        "name": 1, "timestamps_us": 2, "values": 3, "valid": 4, "time_aggregation": 5}
    rr = cls.DESCRIPTOR.nested_types_by_name["RolloutReturn"]
    assert {f.name: f.number for f in rr.fields} == {
        "rollout_spec": 1, "success": 2, "rollout_uuid": 3, "error": 4, "timestep_metrics": 5,
        "aggregated_metrics": 6}
    assert simreturn.AGGREGATIONS == {1: "mean", 2: "median", 3: "max", 4: "min", 5: "last"}


def test_tampered_descriptor_is_refused(monkeypatch):
    simreturn.message_class.cache_clear()
    monkeypatch.setattr(simreturn, "DESCRIPTOR_SHA256", "0" * 64)
    try:
        with pytest.raises(ValueError, match="pinned hash"):
            simreturn.message_class()
    finally:
        simreturn.message_class.cache_clear()


# ---- known answers -------------------------------------------------------------------------


def test_hand_encoded_reply_known_answer():
    report = decode(GOOD)
    assert report["status"] == "passed", report["issues"]
    assert report["input_hashes"] == {"simulation_return": hashlib.sha256(GOOD).hexdigest()}
    assert report["bytes"] == len(GOOD)
    assert report["versions"]["runtime_version"] == {"version_id": "rt-1", "git_hash": "abc",
                                                     "grpc_api_version": [0, 1, 2]}
    assert report["versions"]["nre_version"] is None  # absent is None, never an empty version
    assert report["rollout_returns"] == [{
        "index": 0, "scenario_id": "clipgt-a", "nr_rollouts": 1, "session_uuids": ["u1"],
        "success": True, "rollout_uuid": "r1", "error": "",
        "timestep_metrics": [{"name": "collision_any", "timestamps_us": [0, 100_000],
                              "values": [1.5, "nan"], "valid": [True, False],
                              "time_aggregation": "max"}],
        "aggregated_metrics": {"collision_any": 1.5}}]


def test_nonfinite_values_become_tokens():
    m = metric("x", [0, 1, 2], [math.inf, -math.inf, -0.0], [True] * 3, 1)
    report = decode(reply(rollout(metrics=[m], aggregated=[("x", math.nan)])))
    ret = report["rollout_returns"][0]
    assert ret["timestep_metrics"][0]["values"][:2] == ["inf", "-inf"]
    assert ret["aggregated_metrics"] == {"x": "nan"}


def test_semantics_input_feeds_the_e10_check():
    rollout_input = simreturn.to_semantics_input(decode(GOOD), 0)
    assert rollout_input == {
        "session_uuid": "u1",
        "timestep_metrics": [{"name": "collision_any", "timestamps_us": [0, 100_000],
                              "values": [1.5, "nan"], "valid": [True, False],
                              "time_aggregation": "max"}],
        "aggregated_metrics": {"collision_any": 1.5}}
    e10 = semantics.reconcile_metric_domains(rollout_input, scope="synthetic")
    # online max over valid entries = 1.5 equals the captured aggregate
    assert e10["metrics"][0]["online_recomputed"] == 1.5
    assert e10["metrics"][0]["online_matches_capture"] is True


def test_empty_aggregate_map_is_none_like_the_pinned_reader():
    report = decode(reply(rollout(metrics=[COLLISION])))
    assert simreturn.to_semantics_input(report, 0)["aggregated_metrics"] is None


# ---- negative controls ---------------------------------------------------------------------


@pytest.mark.parametrize(("data", "code"), [
    (GOOD[:-3], "invalid_protobuf"),                                    # truncated
    (GOOD + tag(9, 0) + varint(1), "unsupported_field"),               # unknown top field
    (reply(rollout() + boolean(2, True)), "ambiguous_field"),          # singular repeated
    (reply(rollout(aggregated=[("a", 1.0), ("a", 2.0)])), "ambiguous_field"),  # map key twice
    (reply(tag(2, 2) + varint(1) + b"\x01"), "invalid_protobuf"),     # success as bytes
    (reply(rollout(metrics=[ld(1, b"\xff\xfe")])), "invalid_protobuf"),  # invalid UTF-8
    (b"\x00", "invalid_protobuf"),                                     # field number 0
])
def test_malformed_bytes_are_blocking_issues(data, code):
    report = decode(data)
    assert report["status"] == "blocked"
    assert code in codes(report)
    assert report["rollout_returns"] is None and report["versions"] is None


def test_misaligned_series_and_unspecified_aggregation_are_issues_never_padded():
    bad = metric("m", [0, 1], [0.0], [True, True], 0)
    report = decode(reply(rollout(metrics=[bad])))
    assert report["status"] == "blocked"
    assert codes(report) == ["aggregation_unspecified", "metric_series_misaligned"]
    decoded = report["rollout_returns"][0]["timestep_metrics"][0]
    assert decoded["values"] == [0.0] and decoded["time_aggregation"] is None
    with pytest.raises(ValueError, match="blocking defects"):
        simreturn.to_semantics_input(report, 0)


def test_unknown_enum_number_and_repeated_or_empty_names():
    report = decode(reply(rollout(metrics=[metric("m", [0], [1.0], [True], 9),
                                           metric("m", [0], [1.0], [True], 1),
                                           metric("", [0], [1.0], [True], 1)])))
    assert codes(report) == ["aggregation_unknown", "duplicate_metric_name", "metric_name_empty"]


def test_zero_rollout_returns_is_not_an_empty_success():
    for data in (b"", reply()):
        report = decode(data)
        assert report["status"] == "blocked" and codes(report) == ["no_rollout_returns"]
        assert report["rollout_returns"] == []


def test_input_type_and_size_bounds(monkeypatch):
    assert codes(decode("not bytes")) == ["input_not_bytes"]
    monkeypatch.setattr(simreturn, "MAX_REPLY_BYTES", 10)
    assert codes(decode(GOOD)) == ["reply_exceeds_grpc_limit"]
    with pytest.raises(ValueError, match="scope"):
        simreturn.decode_simulation_return(GOOD, scope="real")


@pytest.mark.parametrize(("sessions", "success", "match"), [
    ((), True, "exactly one"), (("u1", "u2"), True, "exactly one"),
    (("u1",), False, "failed upstream"),
])
def test_semantics_input_refusals(sessions, success, match):
    report = decode(reply(rollout(sessions=sessions, success=success, error="boom",
                                  metrics=[COLLISION])))
    assert report["status"] == "passed"
    with pytest.raises(ValueError, match=match):
        simreturn.to_semantics_input(report, 0)
    with pytest.raises(ValueError, match="outside"):
        simreturn.to_semantics_input(report, 1)
    with pytest.raises(ValueError, match="decoded"):
        simreturn.to_semantics_input(decode(b"\x00"), 0)


def test_library_round_trip_agrees_with_the_hand_encoding():
    cls = simreturn.message_class()
    again = decode(cls.FromString(GOOD).SerializeToString(deterministic=True))
    first = decode(GOOD)
    assert again["status"] == "passed"
    assert (again["versions"], again["rollout_returns"]) == (first["versions"],
                                                             first["rollout_returns"])


# ---- G3-05: varints wider than their field type and keyless map entries -------------------


def varint64(value):
    """Two's-complement 64-bit varint (proto ``int32``/``enum`` negatives use 10 bytes)."""
    return varint(value & (2**64 - 1))


def _spec_nr(nr_varint):
    return text(1, "clipgt-a") + tag(2, 0) + nr_varint + text(3, "u1")


ONE = varint(1)


def _rollout_raw(*, nr=ONE, success=ONE, metrics=(), extra=b""):
    return (ld(1, _spec_nr(nr)) + tag(2, 0) + success + text(3, "r1")
            + b"".join(ld(5, m) for m in metrics) + extra)


def _metric_raw(aggregation_varint, valid_payload=b"\x01"):
    return (text(1, "m") + fixed64s(2, [0]) + doubles(3, [1.0]) + ld(4, valid_payload)
            + tag(5, 0) + aggregation_varint)


@pytest.mark.parametrize("data", [
    reply(_rollout_raw(metrics=[_metric_raw(varint(2**32 + 3))])),   # enum: int32 is 32 bits
    reply(_rollout_raw(metrics=[_metric_raw(varint(2**31))])),       # enum just above int32
    reply(_rollout_raw(nr=varint(2**32 + 1), metrics=[_metric_raw(varint(3))])),  # uint32
    reply(_rollout_raw(success=varint(2), metrics=[_metric_raw(varint(3))])),     # bool 2
    reply(_rollout_raw(metrics=[_metric_raw(varint(3), valid_payload=b"\x02")])),  # packed bool
], ids=["enum_2**32+3", "enum_2**31", "uint32_2**32+1", "bool_2", "packed_bool_2"])
def test_varints_wider_than_their_field_type_block(data):
    report = decode(data)
    assert report["status"] == "blocked"
    assert codes(report) == ["varint_out_of_range"]
    assert report["rollout_returns"] is None


def test_in_range_varint_edges_still_decode():
    # uint32 max, and a negative enum (int32 -1, ten bytes) is a legal encoding whose number
    # has no pinned mapping -> aggregation_unknown, not a wire defect
    report = decode(reply(_rollout_raw(nr=varint(2**32 - 1),
                                       metrics=[_metric_raw(varint64(-1))])))
    assert codes(report) == ["aggregation_unknown"]
    assert report["rollout_returns"][0]["nr_rollouts"] == 2**32 - 1


@pytest.mark.parametrize("entry", [
    tag(2, 1) + struct.pack("<d", 7.0),                 # map entry with no key field
    text(1, "") + tag(2, 1) + struct.pack("<d", 7.0),   # explicit empty key
], ids=["key_absent", "key_empty"])
def test_aggregate_entry_without_a_name_blocks_like_an_empty_dense_name(entry):
    report = decode(reply(rollout(metrics=[COLLISION]) + ld(6, entry)))
    assert report["status"] == "blocked" and codes(report) == ["aggregate_name_empty"]
    with pytest.raises(ValueError, match="blocking defects"):
        simreturn.to_semantics_input(report, 0)
