"""Known answers for the S2 registered-gate glue (build-spec A7, lane-r1 F8).

Oracles are independent of ``gate.s2_gate``: literal zero-difference road thresholds
(36 at margin 0.10, alpha 0.05), the protected ``gate.api`` called directly on the same
request, hand-counted out-of-envelope rollouts and the pinned adapter's own output.
Every record here is synthetic; nothing is a driving outcome.
"""

import copy
import json

import pytest

from gate import api
from gate.alpasim_adapter import SUPPORTED_EVALUATOR_SHA, normalize_summary
from gate.s2_gate import (
    ENVELOPE_M,
    REGISTRATION_FIELDS,
    GateInputError,
    evaluate_registered,
    gate_request_from_records,
)
from gate.verdict import VerdictResult

# The adapter's own metric declaration (gate.alpasim_adapter.normalize_summary(...)["metric"]),
# written out as a literal; the end-to-end test below checks the adapter still emits it.
SCORE_METRIC = {
    "name": "challenge_scene_score",
    "kind": "continuous",
    "higher_is_worse": False,
    "unit": "score_in_0_1",
}


def evaluate(records, reg, *, scope="synthetic", adapter_metric=SCORE_METRIC):
    return evaluate_registered(records, reg, scope=scope, adapter_metric=adapter_metric)


def build_request(records, reg, *, adapter_metric=SCORE_METRIC):
    return gate_request_from_records(records, reg, adapter_metric=adapter_metric)


def registration(**overrides):
    value = {
        "schema_version": 1,
        "registration_id": "synthetic-registration-not-huy",
        "reference": "teacher-bf16",
        "candidate": "teacher-reexport",
        "protocol": {
            "identity": "synthetic-protocol",
            "config_hash": "0" * 64,
            "evaluator": SUPPORTED_EVALUATOR_SHA,
        },
        "seed_mode": "unseeded",
        "score_definition": "synthetic score definition (test fixture)",
        "deviation_convention": "synthetic deviation convention (test fixture)",
        "estimand": "equal road / equal scene within road / equal replicate within scene",
        "required_strata": ["urban"],
        "margin": 0.10,
        "alpha": 0.05,
        "metric": {
            "name": "challenge_scene_score",
            "kind": "continuous",
            "higher_is_worse": False,
            "unit": "score_in_0_1",
        },
        "analysis_mode": "fixed_sample",
    }
    value.update(overrides)
    return value


def record(arm, road, value=0.5, *, stratum="urban", replicate=0, deviation=0.5, scene=None):
    return {
        "arm": arm,
        "scene_id": scene or f"{stratum}/road-{road:03d}/scene-0",
        "road_id": f"{stratum}/road-{road:03d}",
        "stratum": stratum,
        "replicate_id": replicate,
        "value": value,
        "completed": True,
        "deviation_m": deviation,
        "seed": None,
    }


def paired(n_roads, *, a=0.5, b=0.5, stratum="urban"):
    rows = []
    for road in range(n_roads):
        rows.append(record("A", road, a, stratum=stratum))
        rows.append(record("B", road, b, stratum=stratum))
    return rows


def codes(report):
    return {item["code"] for item in report["issues"]}


def paths(report):
    return {item["path"] for item in report["issues"]}


# --- request construction ---------------------------------------------------------


def test_request_is_a_gate_api_schema_v1_request_that_parses_unchanged():
    request = build_request(paired(3), registration())
    assert json.loads(json.dumps(request)) == request
    records, spec, candidate = api.parse_request(request)
    assert candidate is None
    assert len(records) == 6
    assert spec.margin == 0.10 and spec.alpha == 0.05
    assert spec.required_strata == ("urban",)
    assert spec.protocol.seed_mode == "unseeded"
    assert spec.protocol.score_definition == "synthetic score definition (test fixture)"
    assert spec.protocol.deviation_convention == "synthetic deviation convention (test fixture)"
    assert spec.metric.higher_is_worse is False
    assert set(request) == {"schema_version", "spec", "records"}


def test_request_records_are_sorted_so_input_order_does_not_matter():
    rows = paired(4)
    assert build_request(rows, registration()) == build_request(
        list(reversed(rows)), registration()
    )


def test_request_refuses_an_incomplete_registration_with_named_issues():
    broken = registration()
    del broken["margin"]
    with pytest.raises(GateInputError) as caught:
        build_request(paired(2), broken)
    assert {item["path"] for item in caught.value.issues} == {"registration.margin"}


# --- evaluation through the protected gate ----------------------------------------


def test_thirty_six_identical_roads_go_and_match_a_direct_gate_api_call():
    rows = paired(36)
    report = evaluate(rows, registration(), scope="synthetic")
    assert report["status"] == "passed"
    assert report["verdict"] == "GO"
    direct = api.evaluate(*api.parse_request(build_request(rows, registration())))
    assert report["gate_response"] == direct
    assert report["gate_response"]["verdict"]["overall"] == "GO"


def test_thirty_five_identical_roads_preserve_insufficient():
    report = evaluate(paired(35), registration(), scope="synthetic")
    assert report["status"] == "passed"
    assert report["verdict"] == "INSUFFICIENT"
    assert report["gate_response"]["verdict"]["strata"] == {"urban": "INSUFFICIENT"}


def test_large_regression_preserves_no_go():
    # Higher-is-better score; the candidate loses 0.5 on every one of 40 roads.
    report = evaluate(paired(40, a=0.9, b=0.4), registration(), scope="synthetic")
    assert report["verdict"] == "NO-GO"


def test_registered_required_stratum_without_records_is_insufficient_not_dropped():
    report = evaluate(
        paired(40), registration(required_strata=["urban", "highway"]), scope="synthetic"
    )
    assert report["status"] == "passed"
    assert report["verdict"] == "INSUFFICIENT"
    assert report["gate_response"]["pairing"]["missing_strata"] == ["highway"]
    assert report["gate_response"]["verdict"]["details"]["highway"]["reason"] == (
        "missing_stratum"
    )


def test_estimand_is_the_protected_gate_estimand():
    assert REGISTRATION_FIELDS["estimand"] == VerdictResult.estimand
    report = evaluate(
        paired(36), registration(estimand="scene-weighted mean"), scope="synthetic"
    )
    assert report["status"] == "blocked"
    assert report["verdict"] is None
    assert "estimand_unsupported" in codes(report)


def test_only_a_registered_fixed_sample_analysis_is_supported():
    report = evaluate(
        paired(36), registration(analysis_mode="group_sequential"), scope="synthetic"
    )
    assert report["status"] == "blocked"
    assert "analysis_mode_unsupported" in codes(report)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "registration_id",
        "reference",
        "candidate",
        "protocol",
        "seed_mode",
        "score_definition",
        "deviation_convention",
        "estimand",
        "required_strata",
        "margin",
        "alpha",
        "metric",
        "analysis_mode",
    ],
)
def test_every_registration_field_is_required_and_never_defaulted(field):
    broken = registration()
    del broken[field]
    report = evaluate(paired(36), broken, scope="synthetic")
    assert report["status"] == "blocked"
    assert report["verdict"] is None and report["gate_response"] is None
    assert f"registration.{field}" in paths(report)
    assert "registration_field_missing" in codes(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("margin", True),
        ("margin", 0.0),
        ("margin", 1.0),
        ("margin", "0.1"),
        ("alpha", float("nan")),
        ("seed_mode", "deterministic"),
        ("required_strata", []),
        ("required_strata", ["urban", "urban"]),
        ("score_definition", "  "),
        ("schema_version", 2),
        ("metric", {"name": "x", "kind": "ordinal", "higher_is_worse": False, "unit": "u"}),
        ("protocol", {"identity": "p", "config_hash": "c"}),
    ],
)
def test_invalid_registration_values_block(field, value):
    report = evaluate(paired(36), registration(**{field: value}), scope="synthetic")
    assert report["status"] == "blocked"
    assert any(path.startswith(f"registration.{field}") for path in paths(report))


def test_unknown_registration_field_blocks():
    report = evaluate(
        paired(36), registration(default_margin=0.1), scope="synthetic"
    )
    assert report["status"] == "blocked"
    assert "registration_field_unknown" in codes(report)


def test_registration_must_be_an_object():
    report = evaluate(paired(2), None, scope="synthetic")
    assert report["status"] == "blocked"
    assert "registration_invalid" in codes(report)


# --- out-of-envelope rollouts are scored by the gate AND reported separately ---------


def test_out_of_envelope_rollouts_are_reported_separately_and_never_dropped():
    rows = paired(36)
    rows[1]["deviation_m"] = 4.0  # B, road 0: exactly at the 4 m boundary counts
    rows[2]["deviation_m"] = 7.5  # A, road 1
    rows[3]["deviation_m"] = 3.999  # B, road 1: inside the envelope
    report = evaluate(rows, registration(), scope="synthetic")
    section = report["out_of_envelope"]
    assert ENVELOPE_M == 4.0 and section["threshold_m"] == 4.0
    assert section["count"] == 2
    assert section["by_arm"] == {"A": 1, "B": 1}
    assert section["by_stratum"] == {"urban": {"A": 1, "B": 1}}
    assert [(row["arm"], row["deviation_m"]) for row in section["rollouts"]] == [
        ("B", 4.0),
        ("A", 7.5),
    ]
    assert section["gate_counts_agree"] is True
    assert section["handling"] == "scored_by_protected_gate_and_reported_separately"
    # Nothing was dropped: every pair reached the gate.
    assert report["gate_response"]["pairing"]["n_pairs"] == 36
    details = report["gate_response"]["verdict"]["details"]["urban"]
    assert (details["out_of_envelope_a"], details["out_of_envelope_b"]) == (1, 1)
    assert report["verdict"] == "GO"


def test_missing_full_deviation_blocks_instead_of_being_treated_as_inside():
    rows = paired(36)
    rows[0]["deviation_m"] = None
    report = evaluate(rows, registration(), scope="synthetic")
    assert report["status"] == "blocked"
    assert "deviation_missing" in codes(report)
    assert report["verdict"] is None


# --- record admission ---------------------------------------------------------------


def test_record_with_an_unknown_key_blocks():
    rows = paired(36)
    rows[0]["score_metrics"] = {}
    report = evaluate(rows, registration(), scope="synthetic")
    assert report["status"] == "blocked"
    assert "record_invalid" in codes(report)


def test_record_missing_a_key_blocks():
    rows = paired(36)
    del rows[0]["completed"]
    report = evaluate(rows, registration(), scope="synthetic")
    assert "record_invalid" in codes(report)


def test_records_must_be_a_list():
    report = evaluate({"records": []}, registration(), scope="synthetic")
    assert "records_invalid" in codes(report)


def test_unmatched_arm_is_rejected_by_the_gate_pairing():
    rows = paired(36)[:-1]
    report = evaluate(rows, registration(), scope="synthetic")
    assert report["status"] == "blocked"
    assert "gate_rejected" in codes(report)
    assert report["verdict"] is None


# A7-03 regressions: a synthetic seed mode is a contradiction outside synthetic scope.
def _seeded(n_roads):
    rows = paired(n_roads)
    for index, row in enumerate(rows):
        row["seed"] = index // 2
    return rows


@pytest.mark.parametrize("scope", ["actual", "public_source_fixture"])
def test_synthetic_seed_mode_blocks_outside_synthetic_scope(scope):
    report = evaluate(_seeded(36), registration(seed_mode="synthetic_seeded"), scope=scope)
    assert report["status"] == "blocked"
    assert report["verdict"] is None and report["gate_response"] is None
    assert ("seed_mode_scope_mismatch", "registration.seed_mode") in {
        (item["code"], item["path"]) for item in report["issues"]
    }


def test_synthetic_seed_mode_is_accepted_in_synthetic_scope():
    report = evaluate(_seeded(36), registration(seed_mode="synthetic_seeded"))
    assert report["status"] == "passed", report["issues"]
    assert report["verdict"] == "GO"


def test_seeded_registration_rejects_unseeded_records():
    report = evaluate(paired(36), registration(seed_mode="seeded"), scope="synthetic")
    assert report["status"] == "blocked"
    assert "gate_rejected" in codes(report)


def test_unregistered_stratum_in_records_blocks():
    rows = paired(36) + paired(1, stratum="rural")
    report = evaluate(rows, registration(), scope="synthetic")
    assert report["status"] == "blocked"
    assert "gate_rejected" in codes(report)


def test_continuous_value_outside_unit_interval_blocks():
    # Caught by the glue's own value check (A7-02) before the gate is called; it used to
    # surface only as the gate's ``gate_rejected``.
    rows = paired(36)
    rows[0]["value"] = 1.5
    report = evaluate(rows, registration(), scope="synthetic")
    assert report["status"] == "blocked"
    assert report["verdict"] is None
    assert ("record_invalid", "records[0].value") in {
        (item["code"], item["path"]) for item in report["issues"]
    }


# A7-02 regressions: a value the glue promises to block must never escape as an exception
# or be silently coerced (bool -> 1.0) by the contract dataclass.
@pytest.mark.parametrize(
    ("kind", "base", "bad"),
    [
        ("continuous", 0.5, 10**400),
        ("binary", 1, 10**400),
        ("continuous", 0.5, True),
        ("binary", 1, True),
        ("continuous", 0.5, float("nan")),
        ("continuous", 0.5, -0.25),
        ("binary", 1, 0.5),
        ("binary", 1, "1"),
        ("continuous", 0.5, None),
    ],
)
def test_invalid_record_value_blocks_without_raising(kind, base, bad):
    metric = {**SCORE_METRIC, "kind": kind}
    rows = paired(36, a=base, b=base)
    rows[0]["value"] = bad
    report = evaluate(rows, registration(metric=metric), adapter_metric=metric)
    assert report["status"] == "blocked"
    assert report["verdict"] is None and report["gate_response"] is None
    assert ("record_invalid", "records[0].value") in {
        (item["code"], item["path"]) for item in report["issues"]
    }


def test_all_bool_values_block_instead_of_scoring_as_one():
    rows = paired(36)
    for row in rows:
        row["value"] = True
    report = evaluate(rows, registration())
    assert report["status"] == "blocked"
    assert "record_invalid" in codes(report)


def test_binary_values_zero_and_one_are_accepted():
    metric = {**SCORE_METRIC, "kind": "binary", "higher_is_worse": True}
    rows = paired(36, a=0, b=0)
    rows[1]["value"] = 1.0
    report = evaluate(rows, registration(metric=metric), adapter_metric=metric)
    assert report["status"] == "passed", report["issues"]


def test_duplicate_rollout_identity_blocks_rather_than_selecting_one():
    rows = paired(36)
    rows.append(copy.deepcopy(rows[0]))
    report = evaluate(rows, registration(), scope="synthetic")
    assert report["status"] == "blocked"
    assert "gate_rejected" in codes(report)


# --- artifact contract --------------------------------------------------------------


def test_artifact_contract_fields_and_determinism():
    rows = paired(36)
    first = evaluate(rows, registration(), scope="synthetic")
    second = evaluate(list(reversed(rows)), registration(), scope="synthetic")
    assert first == second
    assert first["artifact_kind"] == "s2_registered_gate_evaluation"
    assert first["schema_version"] == 1
    assert first["scope"] == "synthetic"
    assert first["producer"]["module"] == "gate.s2_gate"
    assert len(first["producer"]["source_sha256"]) == 64
    assert set(first["input_hashes"]) == {"records", "registration", "adapter_metric"}
    assert first["registration_authenticity"] == "supplied_not_authenticated"
    assert first["superiority"] == "not_evaluated"
    assert first["units"]["deviation"] == "m"
    assert first["issues"] == []
    assert json.loads(json.dumps(first, allow_nan=False)) == first


def test_registration_change_changes_its_hash():
    a = evaluate(paired(36), registration(), scope="synthetic")
    b = evaluate(paired(36), registration(margin=0.11), scope="synthetic")
    assert a["input_hashes"]["registration"] != b["input_hashes"]["registration"]
    assert a["input_hashes"]["records"] == b["input_hashes"]["records"]


@pytest.mark.parametrize("scope", [None, "real", ""])
def test_scope_must_be_declared_from_the_shared_vocabulary(scope):
    with pytest.raises(ValueError, match="scope"):
        evaluate(paired(2), registration(), scope=scope)


def test_blocked_issues_follow_the_shared_issue_shape():
    broken = registration()
    del broken["alpha"]
    report = evaluate(paired(2), broken, scope="synthetic")
    for item in report["issues"]:
        assert set(item) == {"code", "path", "severity", "evidence_refs", "message"}
        assert item["severity"] == "error"


# --- end to end from the pinned adapter --------------------------------------------


def _summary_row(index, *, score, dist):
    ids = {"run_uuid": "run", "clipgt_id": f"clip-{index}", "rollout_id": "r0"}
    metrics = {
        **ids,
        "run_name": "synthetic",
        "progress_clipped_rel": score,
        "progress_rel": score,
        "gt_dist_traveled_m": 20.0,
        "collision_at_fault": 0.0,
        "offroad": 0.0,
        "left_corridor_laterally": 0.0,
        "dist_to_gt_trajectory": dist,
        "lateral_dist_to_gt_trajectory": dist,
    }
    score_metrics = {k: metrics[k] for k in metrics if k not in ("run_uuid", "clipgt_id",
                                                                   "rollout_id", "run_name")}
    score_metrics["progress_score"] = score
    return {
        **ids,
        "run_name": "synthetic",
        "status": "pass",
        "passed": True,
        "score": score,
        "failure_reason": None,
        "metrics": metrics,
        "score_metrics": score_metrics,
    }


def _normalized(arm, policy, n, *, full_deviation, score=0.5):
    summary = {
        "schema_version": 1,
        "scene_score_enabled": True,
        "rollouts": [_summary_row(i, score=score, dist=0.5) for i in range(n)],
    }
    identities = [
        {
            "run_uuid": "run",
            "clipgt_id": f"clip-{i}",
            "rollout_id": "r0",
            "policy_id": policy,
            "scene_id": f"scene-{i}",
            "road_id": f"road-{i}",
            "stratum": "urban",
            "replicate_id": 0,
        }
        for i in range(n)
    ]
    observations = [
        {
            "run_uuid": "run",
            "clipgt_id": f"clip-{i}",
            "rollout_id": "r0",
            "requested_steps": 200,
            "actual_steps": 200,
            "completion_validated": True,
            "full_max_deviation_m": full_deviation(i),
            "required_metrics_verified": True,
            "provenance_ref": "synthetic-fixture-not-real-telemetry",
        }
        for i in range(n)
    ]
    result = normalize_summary(
        summary,
        identities,
        observations,
        arm=arm,
        policy_id=policy,
        expected_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
        observed_evaluator_sha=SUPPORTED_EVALUATOR_SHA,
    )
    assert result["admitted"], result["issues"]
    return result


def test_adapter_records_flow_through_and_full_deviation_not_summary_sets_the_envelope():
    # The summary aggregate says 0.5 m everywhere (it may be truncated upstream); the
    # preserved full stream says rollout 3 of the candidate reached 6 m.
    a = _normalized("A", "teacher-bf16", 36, full_deviation=lambda i: 0.5)
    b = _normalized("B", "teacher-reexport", 36, full_deviation=lambda i: 6.0 if i == 3 else 0.5)
    assert a["metric"] == b["metric"] == SCORE_METRIC
    report = evaluate(
        a["records"] + b["records"], registration(), scope="synthetic", adapter_metric=a["metric"]
    )
    assert report["status"] == "passed", report["issues"]
    assert report["verdict"] == "GO"
    assert report["out_of_envelope"]["count"] == 1
    assert report["out_of_envelope"]["rollouts"][0]["scene_id"] == "scene-3"
    assert report["out_of_envelope"]["rollouts"][0]["deviation_m"] == 6.0


# --- A7-01: the adapter's metric declaration is cross-checked, never dropped ---------------


def test_inverted_direction_against_the_adapter_metric_blocks_instead_of_flipping_to_go():
    # Built through the real adapter: the candidate scores 0.4 versus 0.9 on all 40 roads, so
    # under the adapter's higher-is-better score the truth is NO-GO. A registration that
    # inverts higher_is_worse used to turn this into a passed GO with no issue.
    a = _normalized("A", "teacher-bf16", 40, full_deviation=lambda i: 0.5, score=0.9)
    b = _normalized("B", "teacher-reexport", 40, full_deviation=lambda i: 0.5, score=0.4)
    rows = a["records"] + b["records"]
    correct = evaluate(rows, registration(), scope="actual", adapter_metric=a["metric"])
    assert correct["status"] == "passed" and correct["verdict"] == "NO-GO"
    inverted = registration(metric={**SCORE_METRIC, "higher_is_worse": True})
    report = evaluate(rows, inverted, scope="actual", adapter_metric=a["metric"])
    assert report["status"] == "blocked"
    assert report["verdict"] is None and report["gate_response"] is None
    assert ("metric_mismatch", "registration.metric.higher_is_worse") in {
        (item["code"], item["path"]) for item in report["issues"]
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("name", "progress_score"),
        ("kind", "binary"),
        ("higher_is_worse", True),
        ("unit", "metres"),
    ],
)
def test_each_metric_field_must_match_the_adapter_declaration(key, value):
    report = evaluate(paired(36), registration(metric={**SCORE_METRIC, key: value}))
    assert report["status"] == "blocked"
    assert ("metric_mismatch", f"registration.metric.{key}") in {
        (item["code"], item["path"]) for item in report["issues"]
    }
    with pytest.raises(GateInputError) as caught:
        build_request(paired(2), registration(metric={**SCORE_METRIC, key: value}))
    assert "metric_mismatch" in {item["code"] for item in caught.value.issues}


@pytest.mark.parametrize(
    "adapter_metric",
    [
        None,
        [],
        {"name": "challenge_scene_score", "kind": "continuous", "higher_is_worse": False},
        {**SCORE_METRIC, "extra": 1},
        {**SCORE_METRIC, "higher_is_worse": "false"},
        {**SCORE_METRIC, "kind": "ordinal"},
        {**SCORE_METRIC, "name": " "},
    ],
)
def test_malformed_adapter_metric_blocks(adapter_metric):
    report = evaluate(paired(36), registration(), adapter_metric=adapter_metric)
    assert report["status"] == "blocked"
    assert report["verdict"] is None
    assert any(item["path"].startswith("adapter_metric") for item in report["issues"])


def test_adapter_metric_is_required_with_no_default():
    with pytest.raises(TypeError):
        evaluate_registered(paired(2), registration(), scope="synthetic")
    with pytest.raises(TypeError):
        gate_request_from_records(paired(2), registration())


def test_adapter_metric_change_changes_its_hash():
    other = {**SCORE_METRIC, "unit": "other"}
    a = evaluate(paired(36), registration())
    b = evaluate(paired(36), registration(metric=other), adapter_metric=other)
    assert b["status"] == "passed"
    assert a["input_hashes"]["adapter_metric"] != b["input_hashes"]["adapter_metric"]
