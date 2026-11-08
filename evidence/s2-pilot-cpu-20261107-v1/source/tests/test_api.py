"""Raw-arm admission through public JSON output; no Monte Carlo qualification."""

import json
from dataclasses import asdict, replace

import pytest

from gate.api import evaluate, parse_request
from gate.contracts import ComparisonSpec, MetricSpec, Protocol, RolloutRecord


def spec(*, kind="binary", higher_is_worse=True):
    return ComparisonSpec(
        "teacher",
        "candidate",
        Protocol("test-v1", "unseeded", "config", "eval", "test", "max-lateral-m"),
        ("urban",),
        0.05,
        0.05,
        MetricSpec("test", kind, higher_is_worse, "fraction"),
    )


def records(n=220, *, a=0, b=0, single_road=False):
    return [
        RolloutRecord(arm, f"scene-{i}", "road" if single_road else f"road-{i}", "urban", 0, value)
        for i in range(n)
        for arm, value in (("A", a), ("B", b))
    ]


def request_payload():
    return json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "spec": asdict(spec()),
                "records": [asdict(record) for record in records()],
            }
        )
    )


def test_raw_arms_identical_on_220_roads_are_go():
    result = evaluate(records(), spec())
    assert result["schema_version"] == 1
    assert result["verdict"]["overall"] == "GO"
    assert result["pairing"]["n_records"] == 440
    assert result["pairing"]["n_pairs"] == 220
    assert result["pairing"]["n_roads"] == 220
    assert "rows" not in result["pairing"]
    assert result["difference_convention"] == "positive means candidate worse"
    assert "one-sided" in result["interval_convention"]


def test_220_scenes_on_one_road_are_insufficient():
    result = evaluate(records(single_road=True), spec())
    assert result["verdict"]["overall"] == "INSUFFICIENT"
    assert result["pairing"]["n_scenes"] == 220
    assert result["pairing"]["n_roads"] == 1
    assert result["verdict"]["details"]["urban"]["n_roads"] == 1


def test_missing_required_stratum_is_not_lost_in_the_adapter():
    result = evaluate(records(), replace(spec(), required_strata=("urban", "rare")))
    assert result["pairing"]["missing_strata"] == ["rare"]
    assert result["verdict"]["overall"] == "INSUFFICIENT"
    assert result["verdict"]["strata"]["rare"] == "INSUFFICIENT"


def test_incomplete_arm_rejects_before_a_verdict():
    raw = records()
    raw[0] = replace(raw[0], completed=False)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(raw, spec())


@pytest.mark.parametrize("higher_is_worse,a,b", [(True, 0, 1), (False, 1, 0)])
def test_binary_direction_is_normalized_exactly_once(higher_is_worse, a, b):
    result = evaluate(records(a=a, b=b), spec(higher_is_worse=higher_is_worse))
    assert result["verdict"]["overall"] == "NO-GO"
    assert result["verdict"]["details"]["urban"]["estimate"] == 1


@pytest.mark.parametrize("higher_is_worse,a,b", [(True, 0.1, 0.9), (False, 0.9, 0.1)])
def test_continuous_direction_reaches_the_core(higher_is_worse, a, b):
    result = evaluate(records(a=a, b=b), spec(kind="continuous", higher_is_worse=higher_is_worse))
    assert result["verdict"]["overall"] == "NO-GO"
    assert result["verdict"]["details"]["urban"]["estimate"] == pytest.approx(0.8)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_continuous_scores_outside_unit_interval_are_rejected(value):
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        evaluate(records(a=value), spec(kind="continuous"))


def test_deviation_exits_survive_public_admission():
    raw = records()
    raw[0] = replace(raw[0], deviation_m=4)
    raw[1] = replace(raw[1], deviation_m=4.1)
    raw[2] = replace(raw[2], deviation_m=3.999)
    result = evaluate(raw, spec())
    details = result["verdict"]["details"]["urban"]
    assert details["out_of_envelope_a"] == 1
    assert details["out_of_envelope_b"] == 1
    assert result["pairing"]["n_pairs"] == 220


def test_json_request_and_result_roundtrip_without_mutating_payload():
    payload = request_payload()
    original = json.dumps(payload, sort_keys=True)
    raw, comparison, candidate_protocol = parse_request(payload)
    assert isinstance(raw[0], RolloutRecord)
    assert isinstance(comparison, ComparisonSpec)
    assert comparison.required_strata == ("urban",)
    assert candidate_protocol is None
    result = evaluate(raw, comparison, candidate_protocol)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert json.dumps(payload, sort_keys=True) == original


def test_candidate_protocol_is_parsed_and_mismatch_is_rejected():
    payload = request_payload()
    payload["candidate_protocol"] = asdict(replace(spec().protocol, evaluator="other"))
    raw, comparison, candidate_protocol = parse_request(payload)
    assert isinstance(candidate_protocol, Protocol)
    with pytest.raises(ValueError, match="protocol"):
        evaluate(raw, comparison, candidate_protocol)


@pytest.mark.parametrize("version", [None, True, "1", 1.0, 0, 2])
def test_schema_version_is_exactly_integer_one(version):
    payload = request_payload()
    payload["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        parse_request(payload)


@pytest.mark.parametrize("target", ["top", "spec", "protocol", "metric", "record", "candidate"])
def test_unknown_fields_are_not_silently_ignored(target):
    payload = request_payload()
    if target == "top":
        payload["extra"] = 1
    elif target == "spec":
        payload["spec"]["extra"] = 1
    elif target in ("protocol", "metric"):
        payload["spec"][target]["extra"] = 1
    elif target == "record":
        payload["records"][0]["extra"] = 1
    else:
        payload["candidate_protocol"] = {**asdict(spec().protocol), "extra": 1}
    with pytest.raises(ValueError):
        parse_request(payload)


@pytest.mark.parametrize("field", ["spec", "records", "schema_version"])
def test_missing_required_top_level_field(field):
    payload = request_payload()
    del payload[field]
    with pytest.raises(ValueError):
        parse_request(payload)


def test_parse_rejects_non_object_payload():
    with pytest.raises(ValueError):
        parse_request([])


def test_required_strata_must_be_an_array_not_a_string():
    payload = request_payload()
    payload["spec"]["required_strata"] = "urban"
    with pytest.raises(ValueError, match="required_strata"):
        parse_request(payload)


def test_records_must_be_an_array():
    payload = request_payload()
    payload["records"] = {}
    with pytest.raises(ValueError, match="records"):
        parse_request(payload)
