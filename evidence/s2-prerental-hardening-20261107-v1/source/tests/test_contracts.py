"""Adversarial admission tests; these do not test statistical acceptance."""

from dataclasses import FrozenInstanceError, replace

import pytest

from gate.contracts import (
    ComparisonSpec,
    MetricSpec,
    PairingError,
    Protocol,
    RolloutRecord,
    is_pair,
    pair_records,
)


def protocol(seed_mode="unseeded"):
    return Protocol("synthetic-v1", seed_mode, "cfg", "eval", "failure", "none")


def spec(**changes):
    base = ComparisonSpec(
        "teacher",
        "candidate",
        protocol(),
        ("urban", "rare"),
        0.05,
        0.05,
        MetricSpec("failure", "binary", True, "fraction"),
    )
    return replace(base, **changes)


def record(arm_label, **changes):
    base = RolloutRecord(arm_label, "scene-1", "road-1", "urban", 0, 0.0)
    return replace(base, **changes)


def test_pairing_preserves_metadata_and_missing_required_strata():
    report = pair_records([record("B", value=1, deviation_m=4), record("A")], spec())
    assert (report.n_records, report.n_pairs, report.n_scenes, report.n_roads) == (2, 1, 1, 1)
    assert report.missing_strata == ("rare",)
    assert report.rows == (
        {
            "scene_id": "scene-1",
            "road_id": "road-1",
            "stratum": "urban",
            "replicate_id": 0,
            "a_value": 0.0,
            "b_value": 1,
            "a_fail": False,
            "b_fail": True,
            "a_deviation_m": None,
            "b_deviation_m": 4,
            "a_seed": None,
            "b_seed": None,
        },
    )


def test_empty_records_retain_every_required_stratum():
    report = pair_records([], spec())
    assert report.rows == ()
    assert report.missing_strata == ("urban", "rare")


def test_replicates_are_not_counted_as_independent_scenes_or_roads():
    records = [record(arm, replicate_id=i) for i in range(3) for arm in ("A", "B")]
    report = pair_records(records, spec())
    assert (report.n_pairs, report.n_scenes, report.n_roads) == (3, 1, 1)


def test_multiple_scenes_on_one_road_remain_one_road():
    records = [record(arm, scene_id=f"scene-{i}") for i in range(2) for arm in ("A", "B")]
    report = pair_records(records, spec())
    assert (report.n_pairs, report.n_scenes, report.n_roads) == (2, 2, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity", "other"),
        ("seed_mode", "seeded"),
        ("config_hash", "other"),
        ("evaluator", "other"),
        ("score_definition", "other"),
        ("deviation_convention", "other"),
    ],
)
def test_every_protocol_field_must_match(field, value):
    different = replace(protocol(), **{field: value})
    assert not is_pair(record("A"), record("B"), protocol(), different)
    with pytest.raises(PairingError, match="protocol"):
        pair_records([record("A"), record("B")], spec(), candidate_protocol=different)


@pytest.mark.parametrize(
    "records,reason",
    [
        ([record("A"), record("A"), record("B")], "duplicate"),
        ([record("A")], "unmatched"),
        ([record("A"), record("B", road_id="road-2")], "metadata"),
        ([record("A"), record("B", stratum="rare")], "metadata"),
        ([record("A"), record("B", completed=False)], "incomplete"),
        ([record("A", stratum="unknown"), record("B", stratum="unknown")], "stratum"),
        ([record("A", value=0.5), record("B")], "binary"),
    ],
)
def test_admission_never_silently_drops_invalid_rows(records, reason):
    with pytest.raises(PairingError, match=reason):
        pair_records(records, spec())


def test_scene_metadata_must_be_stable_across_replicates():
    records = [record(arm) for arm in ("A", "B")]
    records += [record(arm, replicate_id=1, road_id="road-2") for arm in ("A", "B")]
    with pytest.raises(PairingError, match="metadata"):
        pair_records(records, spec())


def test_seeded_pairs_need_explicit_matching_seed():
    seeded = protocol("synthetic_seeded")
    assert not is_pair(record("A"), record("B"), seeded, seeded)
    assert not is_pair(record("A", seed=1), record("B", seed=2), seeded, seeded)
    assert is_pair(record("A", seed=1), record("B", seed=1), seeded, seeded)
    with pytest.raises(PairingError, match="seed"):
        pair_records([record("A"), record("B")], spec(protocol=seeded))


def test_unseeded_pairing_does_not_claim_matching_random_streams():
    assert is_pair(record("A", seed=1), record("B", seed=2), protocol(), protocol())
    report = pair_records([record("A", seed=1), record("B", seed=2)], spec())
    assert report.rows[0]["a_seed"] == 1
    assert report.rows[0]["b_seed"] == 2


def test_is_pair_rejects_same_arm_and_different_replicate():
    assert not is_pair(record("A"), record("A"), protocol(), protocol())
    assert not is_pair(record("A"), record("B", replicate_id=1), protocol(), protocol())


def test_binary_success_metric_normalizes_failures_but_keeps_raw_values():
    metric = MetricSpec("success", "binary", False, "fraction")
    row = pair_records([record("A", value=1), record("B", value=0)], spec(metric=metric)).rows[0]
    assert (row["a_value"], row["b_value"]) == (1, 0)
    assert (row["a_fail"], row["b_fail"]) == (False, True)


def test_continuous_values_are_not_thresholded_or_direction_flipped():
    metric = MetricSpec("scene_score", "continuous", False, "score")
    row = pair_records(
        [record("A", value=0.75), record("B", value=0.70)], spec(metric=metric)
    ).rows[0]
    assert (row["a_value"], row["b_value"]) == (0.75, 0.70)
    assert "a_fail" not in row


def test_same_checkpoint_control_is_permitted():
    report = pair_records([record("A"), record("B")], spec(candidate="teacher"))
    assert report.n_pairs == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "0"])
def test_record_rejects_nonfinite_or_nonnumeric_values(value):
    with pytest.raises(ValueError):
        record("A", value=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"arm": "C"},
        {"scene_id": ""},
        {"replicate_id": -1},
        {"replicate_id": True},
        {"deviation_m": -1},
        {"seed": -1},
        {"completed": 1},
    ],
)
def test_record_structural_validation(changes):
    with pytest.raises((ValueError, TypeError)):
        record("A", **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"margin": -1},
        {"margin": 0},
        {"margin": 1},
        {"margin": 1.1},
        {"margin": float("nan")},
        {"alpha": 0},
        {"alpha": 1},
        {"required_strata": ()},
        {"required_strata": ("urban", "urban")},
    ],
)
def test_spec_structural_validation(changes):
    with pytest.raises(ValueError):
        spec(**changes)


def test_contracts_are_frozen():
    with pytest.raises(FrozenInstanceError):
        record("A").scene_id = "changed"
