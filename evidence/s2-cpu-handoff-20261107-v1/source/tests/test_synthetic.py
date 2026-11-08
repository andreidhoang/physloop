"""Small exact generator and scheduling checks; no Monte Carlo qualification."""

from collections import Counter
from itertools import groupby

import pytest

from gate.synthetic import _pair_probabilities, generate, split_road_batches


@pytest.mark.parametrize(
    "p_a,q_plus,q_minus,expected",
    [
        (0, 0, 0, (False, False)),
        (1, 0, 1, (True, False)),
        (0, 1, 0, (False, True)),
        (1, 0, 0, (True, True)),
    ],
)
def test_four_categorical_vertices_have_exact_outcomes(p_a, q_plus, q_minus, expected):
    rows = generate(
        1, {"urban": 2}, scenes_per_road=2, replicates=3, p_a=p_a, q_plus=q_plus, q_minus=q_minus
    )
    assert len(rows) == 12
    assert {(row["a_fail"], row["b_fail"]) for row in rows} == {expected}


def test_all_latent_copying_keeps_each_road_together():
    rows = generate(
        3,
        {"urban": 20},
        scenes_per_road=4,
        replicates=3,
        p_a=0.5,
        q_plus=0.25,
        q_minus=0.25,
        road_correlation=1,
        replicate_correlation=1,
    )
    for _, road_rows in groupby(rows, key=lambda row: row["road_id"]):
        assert len({(row["a_fail"], row["b_fail"]) for row in road_rows}) == 1


def test_single_replicate_preserves_scene_latent_road_dependence():
    rows = generate(
        3,
        {"urban": 20},
        scenes_per_road=5,
        replicates=1,
        p_a=0.5,
        q_plus=0.25,
        q_minus=0.25,
        road_correlation=1,
        replicate_correlation=0,
    )
    for _, road_rows in groupby(rows, key=lambda row: row["road_id"]):
        assert len({(row["a_fail"], row["b_fail"]) for row in road_rows}) == 1


def test_seed_is_reproducible_independent_of_strata_mapping_order():
    first = generate(19, {"urban": 5, "rare": 3}, p_a=0.5, q_plus=0.25, q_minus=0.25)
    second = generate(19, {"rare": 3, "urban": 5}, p_a=0.5, q_plus=0.25, q_minus=0.25)
    assert first == second
    assert first != generate(20, {"urban": 5, "rare": 3}, p_a=0.5, q_plus=0.25, q_minus=0.25)


def test_pinned_seed_snapshot():
    """A compatibility snapshot, not evidence of statistical calibration."""
    rows = generate(19, {"urban": 5, "rare": 3}, p_a=0.5, q_plus=0.25, q_minus=0.25)
    assert [(row["a_fail"], row["b_fail"]) for row in rows] == [
        (False, False),
        (True, False),
        (False, True),
        (False, False),
        (True, False),
        (True, True),
        (True, False),
        (False, False),
    ]


def test_joint_probabilities_have_declared_marginals_exactly():
    probabilities = _pair_probabilities(0.5, 0.25, 0.125)
    assert probabilities == (0.25, 0.125, 0.25, 0.375)
    _neither, a_only, b_only, both = probabilities
    assert sum(probabilities) == 1
    assert a_only + both == 0.5
    assert b_only + both == 0.625
    assert b_only - a_only == 0.125
    # Copying a latent with this same distribution preserves each marginal exactly.
    assert tuple(0.75 * p + 0.25 * p for p in probabilities) == probabilities


def test_stratum_effect_override_does_not_change_other_strata():
    rows = generate(
        2, {"common": 2, "rare": 2}, p_a=0, q_plus=0, q_minus=0, effects={"rare": (1, 0)}
    )
    assert all(not row["a_fail"] for row in rows)
    assert all(row["b_fail"] == (row["stratum"] == "rare") for row in rows)


def test_record_identity_is_unique_and_clusters_are_retained():
    rows = generate(10, {"urban": 3, "rare": 2}, scenes_per_road=2, replicates=3)
    assert len(rows) == 30
    assert len({(row["scene_id"], row["replicate_id"]) for row in rows}) == 30
    assert len({row["road_id"] for row in rows}) == 5
    assert len({row["scene_id"] for row in rows}) == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"strata": {}},
        {"strata": {"": 3}},
        {"strata": {"urban": 0}},
        {"strata": {"urban": 1.5}},
        {"scenes_per_road": 0},
        {"replicates": 0},
        {"p_a": -0.1},
        {"q_plus": -0.1},
        {"q_minus": -0.1},
        {"p_a": 0.9, "q_plus": 0.2},
        {"p_a": 0.01, "q_minus": 0.02},
        {"p_a": float("nan")},
        {"road_correlation": 1.1},
        {"replicate_correlation": -0.1},
        {"effects": {"missing": (0.1, 0.02)}},
        {"effects": {"urban": (0.95, 0.02)}},
    ],
)
def test_invalid_generator_configuration_is_rejected_before_sampling(kwargs):
    parameters = {"seed": 1, "strata": {"urban": 3}}
    parameters.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        generate(**parameters)


def test_road_batches_are_balanced_stratified_and_lossless():
    rows = generate(3, {"urban": 7, "rare": 8}, scenes_per_road=2, replicates=3)
    batches = split_road_batches(rows, K=3, seed=42)
    assert len(batches) == 3
    assert Counter((r["scene_id"], r["replicate_id"]) for b in batches for r in b) == Counter(
        (r["scene_id"], r["replicate_id"]) for r in rows
    )
    assigned = {}
    for batch_index, batch in enumerate(batches):
        assert {r["stratum"] for r in batch} == {"urban", "rare"}
        for row in batch:
            assert assigned.setdefault(row["road_id"], batch_index) == batch_index
    for stratum in ("urban", "rare"):
        counts = [len({r["road_id"] for r in b if r["stratum"] == stratum}) for b in batches]
        assert max(counts) - min(counts) <= 1


def test_batch_order_reproduces_and_ignores_input_order():
    rows = generate(3, {"urban": 9, "rare": 9})
    batches = split_road_batches(rows, seed=42)
    assert batches == split_road_batches(list(reversed(rows)), seed=42)
    assert batches != split_road_batches(rows, seed=43)


def test_road_cannot_cross_strata():
    rows = generate(3, {"urban": 3, "rare": 3})
    rows[0]["road_id"] = rows[-1]["road_id"]
    with pytest.raises(ValueError, match="strata"):
        split_road_batches(rows)


def test_every_look_requires_each_stratum():
    with pytest.raises(ValueError, match="fewer"):
        split_road_batches(generate(3, {"urban": 5, "rare": 2}), K=3)


@pytest.mark.parametrize("K", [0, -1, 1.5, True])
def test_invalid_look_count(K):
    with pytest.raises((ValueError, TypeError)):
        split_road_batches(generate(3, {"urban": 3}), K=K)


def test_empty_batch_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        split_road_batches([])
