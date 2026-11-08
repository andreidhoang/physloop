"""Synthetic schema fixtures derived from the pinned public writer, not licensed rollouts."""

from copy import deepcopy

import pytest

from gate.alpasim_adapter import normalize_summary

SHA = "e121e59141dc391f6c295513d6dadc840a419149"


def fixture():
    row = {
        "run_uuid": "run",
        "clipgt_id": "clip",
        "rollout_id": "rollout",
        "run_name": "bf16",
        "status": "pass",
        "passed": True,
        "score": 0.8,
        "failure_reason": None,
        "metrics": {
            "run_uuid": "run",
            "clipgt_id": "clip",
            "rollout_id": "rollout",
            "run_name": "bf16",
            "progress_clipped_rel": 0.64,
            "progress_rel": 0.64,
            "gt_dist_traveled_m": 20.0,
            "collision_at_fault": 0.0,
            "offroad": 0.0,
            "left_corridor_laterally": 0.0,
            "dist_to_gt_trajectory": 0.2,
            "lateral_dist_to_gt_trajectory": 0.2,
        },
        "score_metrics": {
            "progress_clipped_rel": 0.64,
            "progress_rel": 0.64,
            "progress_score": 0.8,
            "collision_at_fault": 0.0,
            "offroad": 0.0,
            "dist_to_gt_trajectory": 0.2,
            "lateral_dist_to_gt_trajectory": 0.2,
            "left_corridor_laterally": 0.0,
            "gt_dist_traveled_m": 20.0,
        },
    }
    identity = {
        "run_uuid": "run",
        "clipgt_id": "clip",
        "rollout_id": "rollout",
        "policy_id": "teacher-bf16",
        "scene_id": "scene",
        "road_id": "road",
        "stratum": "urban",
        "replicate_id": 0,
    }
    observation = {
        "run_uuid": "run",
        "clipgt_id": "clip",
        "rollout_id": "rollout",
        "requested_steps": 200,
        "actual_steps": 200,
        "completion_validated": True,
        "full_max_deviation_m": 4.0,
        "required_metrics_verified": True,
        "provenance_ref": "synthetic-fixture-not-real-telemetry",
    }
    return (
        {"schema_version": 1, "scene_score_enabled": True, "rollouts": [row]},
        [identity],
        [observation],
    )


def call(summary, identities, observations, **kwargs):
    return normalize_summary(
        summary,
        identities,
        observations,
        arm=kwargs.get("arm", "A"),
        policy_id=kwargs.get("policy_id", "teacher-bf16"),
        expected_evaluator_sha=SHA,
        observed_evaluator_sha=kwargs.get("sha", SHA),
    )


def test_valid_schema_preserves_score_and_full_deviation():
    result = call(*fixture())
    assert result["admitted"]
    assert result["records"][0]["value"] == 0.8
    assert result["records"][0]["deviation_m"] == 4.0
    assert result["counts"]["out_of_envelope"] == 1


def test_zero_score_is_not_automatically_an_infrastructure_or_binary_failure():
    s, i, o = fixture()
    s["rollouts"][0]["score"] = 0
    s["rollouts"][0]["score_metrics"]["progress_score"] = 0
    s["rollouts"][0]["score_metrics"]["progress_clipped_rel"] = 0
    s["rollouts"][0]["metrics"]["progress_clipped_rel"] = 0
    result = call(s, i, o)
    assert result["admitted"] and result["records"][0]["value"] == 0
    assert result["diagnostics"][0]["passed"] is True


@pytest.mark.parametrize("error", ["renderer OOM", None, ""])
def test_error_key_quarantines_even_when_falsey(error):
    s, i, o = fixture()
    s["rollouts"][0].update(status="fail", passed=False, score=0, metrics={"error": error})
    result = call(s, i, o)
    assert not result["admitted"] and result["records"] == []
    assert "infrastructure_error" in {r["code"] for r in result["issues"]}


@pytest.mark.parametrize(
    "mutation",
    [
        "mapping",
        "ledger",
        "short",
        "unknown_deviation",
        "missing_metrics",
        "duplicate",
        "unscored",
        "nan",
        "disabled",
    ],
)
def test_incomplete_evidence_never_silently_drops_rows(mutation):
    s, i, o = fixture()
    if mutation == "mapping":
        i = []
    elif mutation == "ledger":
        o = []
    elif mutation == "short":
        o[0]["actual_steps"] = 199
    elif mutation == "unknown_deviation":
        o[0]["full_max_deviation_m"] = None
    elif mutation == "missing_metrics":
        o[0]["required_metrics_verified"] = False
    elif mutation == "duplicate":
        s["rollouts"].append(deepcopy(s["rollouts"][0]))
    elif mutation == "unscored":
        s["rollouts"][0].update(status="unscored", score=None)
    elif mutation == "nan":
        s["rollouts"][0]["score"] = float("nan")
    else:
        s["scene_score_enabled"] = False
    result = call(s, i, o)
    assert not result["admitted"] and result["records"] == []


def test_version_one_does_not_establish_compatible_evaluator():
    assert not call(*fixture(), sha="a" * 40)["admitted"]


def test_missing_expected_rollout_and_extra_sidecar_are_reported():
    s, i, o = fixture()
    i.append({**i[0], "rollout_id": "missing", "scene_id": "missing"})
    result = call(s, i, o)
    assert not result["admitted"]
    assert "missing_expected_rollout" in {r["code"] for r in result["issues"]}


@pytest.mark.parametrize("reason", ["collision_at_fault", "offroad", "left_corridor_laterally"])
def test_driving_failure_is_retained_as_a_scored_observation(reason):
    s, i, o = fixture()
    s["rollouts"][0].update(status="fail", passed=False, score=0, failure_reason=reason)
    s["rollouts"][0]["metrics"][reason] = 1
    s["rollouts"][0]["score_metrics"][reason] = 1
    result = call(s, i, o)
    assert result["admitted"] and result["records"][0]["completed"]


def test_road_and_stratum_are_never_inferred_from_clip_id():
    s, i, o = fixture()
    del i[0]["road_id"]
    assert not call(s, i, o)["admitted"]


@pytest.mark.parametrize("value", [[], {}, None, "", "  "])
def test_bad_scene_identity_is_quarantined_before_hashing(value):
    s, i, o = fixture()
    i[0]["scene_id"] = value
    result = call(s, i, o)
    assert not result["admitted"] and result["records"] == []


@pytest.mark.parametrize("field", ["score", "full_max_deviation_m"])
def test_oversized_integer_is_reported_without_overflow(field):
    s, i, o = fixture()
    target = s["rollouts"][0] if field == "score" else o[0]
    target[field] = 10**400
    result = call(s, i, o)
    assert not result["admitted"] and result["records"] == []


def test_whitespace_policy_identity_is_rejected_even_when_mapping_matches():
    s, i, o = fixture()
    i[0]["policy_id"] = "   "
    assert not call(s, i, o, policy_id="   ")["admitted"]


@pytest.mark.parametrize("field", ["run_uuid", "clipgt_id", "rollout_id"])
def test_metric_identity_must_match_rollout_identity(field):
    s, i, o = fixture()
    s["rollouts"][0]["metrics"][field] = "different"
    assert not call(s, i, o)["admitted"]


@pytest.mark.parametrize(
    "field",
    [
        "progress_clipped_rel",
        "gt_dist_traveled_m",
        "collision_at_fault",
        "offroad",
        "left_corridor_laterally",
    ],
)
@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf"), "0", True, 10**400],
    ids=["null", "nan", "inf", "string", "bool", "huge_integer"],
)
def test_required_source_metrics_are_finite_numbers(field, value):
    s, i, o = fixture()
    s["rollouts"][0]["metrics"][field] = value
    result = call(s, i, o)
    assert not result["admitted"] and result["records"] == []


@pytest.mark.parametrize(
    "field",
    [
        "progress_clipped_rel",
        "progress_rel",
        "progress_score",
        "collision_at_fault",
        "offroad",
        "dist_to_gt_trajectory",
        "lateral_dist_to_gt_trajectory",
        "left_corridor_laterally",
        "gt_dist_traveled_m",
    ],
)
def test_all_score_metric_keys_are_required_even_optional_values(field):
    s, i, o = fixture()
    del s["rollouts"][0]["score_metrics"][field]
    assert not call(s, i, o)["admitted"]


def test_optional_diagnostic_values_may_be_null():
    s, i, o = fixture()
    for key in ("progress_rel", "dist_to_gt_trajectory", "lateral_dist_to_gt_trajectory"):
        s["rollouts"][0]["score_metrics"][key] = None
        s["rollouts"][0]["metrics"][key] = None
    assert call(s, i, o)["admitted"]


def test_duplicate_metric_values_must_match_source():
    s, i, o = fixture()
    s["rollouts"][0]["score_metrics"]["gt_dist_traveled_m"] = 21.0
    assert not call(s, i, o)["admitted"]


@pytest.mark.parametrize("reason", [None, "collision", "unknown"])
def test_failure_reason_must_be_a_source_hard_failure(reason):
    s, i, o = fixture()
    s["rollouts"][0].update(status="fail", passed=False, score=0, failure_reason=reason)
    assert not call(s, i, o)["admitted"]


def test_pass_cannot_carry_failure_reason():
    s, i, o = fixture()
    s["rollouts"][0]["failure_reason"] = "offroad"
    assert not call(s, i, o)["admitted"]


def test_hard_failure_cannot_carry_positive_score():
    s, i, o = fixture()
    s["rollouts"][0].update(status="fail", passed=False, score=0.8, failure_reason="offroad")
    s["rollouts"][0]["metrics"]["offroad"] = 1
    s["rollouts"][0]["score_metrics"]["offroad"] = 1
    assert not call(s, i, o)["admitted"]


def test_pass_score_must_match_reported_progress_score():
    s, i, o = fixture()
    s["rollouts"][0]["score_metrics"]["progress_score"] = 0.7
    assert not call(s, i, o)["admitted"]


def test_diagnostics_preserve_raw_metric_provenance():
    s, i, o = fixture()
    original = deepcopy(s)
    result = call(s, i, o)
    assert result["diagnostics"][0]["metrics"] == s["rollouts"][0]["metrics"]
    assert result["diagnostics"][0]["score_metrics"] == s["rollouts"][0]["score_metrics"]
    assert s == original


def test_one_road_cannot_span_registered_strata():
    summary, identities, observations = fixture()
    row = deepcopy(summary["rollouts"][0])
    identity = deepcopy(identities[0])
    observation = deepcopy(observations[0])
    for item in (row, row["metrics"], identity, observation):
        item.update(clipgt_id="clip-2", rollout_id="rollout-2")
    identity.update(scene_id="scene-2", stratum="highway")
    summary["rollouts"].append(row)
    identities.append(identity)
    observations.append(observation)
    result = call(summary, identities, observations)
    assert not result["admitted"]
    assert result["records"] == []
