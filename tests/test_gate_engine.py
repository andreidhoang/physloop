"""Adversarial contracts authored before the operational gate implementation."""

import pytest

from gate.sequential import SequentialGate
from gate.verdict import verdict


def data(n=220, stratum="urban", value=0, road_size=1, replicates=1):
    return [
        {
            "scene_id": f"{stratum}-{i}",
            "road_id": f"{stratum}-{i // road_size}",
            "stratum": stratum,
            "replicate_id": j,
            "a_fail": 0,
            "b_fail": value,
        }
        for i in range(n)
        for j in range(replicates)
    ]


def test_same_road_duplicates_do_not_manufacture_information():
    result = verdict(data(220, road_size=220), margin=0.05)
    assert result.overall == "INSUFFICIENT"
    assert result.details["urban"].n_roads == 1


def test_replicates_do_not_increase_independent_count():
    a = verdict(data(9), margin=0.05)
    b = verdict(data(9, replicates=30), margin=0.05)
    assert a.details["urban"].upper == b.details["urban"].upper
    assert b.overall == "INSUFFICIENT"


def test_absent_required_stratum_blocks_go():
    result = verdict(data(), margin=0.05, required_strata=("urban", "rain"))
    assert result.strata["rain"] == "INSUFFICIENT"
    assert result.overall == "INSUFFICIENT"


@pytest.mark.parametrize("mutation", ["duplicate", "nonfinite", "crossroad", "mixed"])
def test_invalid_paired_input_rejected(mutation):
    rows = data(10)
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "nonfinite":
        rows[0]["a_fail"] = float("nan")
    elif mutation == "crossroad":
        rows[1].update(stratum="other", road_id=rows[0]["road_id"])
    else:
        rows[0].update(a_value=0.1, b_value=0.2)
    with pytest.raises(ValueError):
        verdict(rows, margin=0.05)


def test_continuous_direction_and_deviation_accounting():
    rows = [
        {
            "scene_id": str(i),
            "road_id": str(i),
            "stratum": "s",
            "a_value": 0.8,
            "b_value": 0.2,
            "a_deviation_m": 4.0,
            "b_deviation_m": 3.9,
        }
        for i in range(220)
    ]
    result = verdict(rows, margin=0.05, higher_is_worse=False)
    assert result.overall == "NO-GO"
    assert result.details["s"].out_of_envelope_a == 220


def test_order_invariance_and_no_input_mutation():
    rows = data()
    before = [row.copy() for row in rows]
    assert verdict(rows, margin=0.05) == verdict(rows[::-1], margin=0.05)
    assert rows == before


def test_sequential_locks_terminal_and_used_roads():
    gate = SequentialGate(("urban",), margin=0.05, weights=(0.2, 0.3, 0.5))
    gate.update(data(20))
    with pytest.raises(ValueError, match="road"):
        gate.update(data(20))
    assert gate.look == 1


def test_sequential_efficacy_freeze_and_replay():
    gate = SequentialGate(("urban", "rare"), margin=0.05, weights=(0.2, 0.3, 0.5))
    result = gate.update(data(220) + data(3, "rare"))
    assert result.overall == "INSUFFICIENT"
    assert gate.active_strata == ("rare",)
    saved = gate.to_dict()
    resumed = SequentialGate.from_dict(saved)
    batch = data(3, "rare")
    for row in batch:
        row["scene_id"] += "-new"
        row["road_id"] += "-new"
    assert resumed.update(batch) == gate.update(batch)
    with pytest.raises(ValueError):
        gate.update(data(2))


def test_futility_is_insufficient_and_terminal():
    gate = SequentialGate(("rare",), margin=0.05, weights=(0.5, 0.5), futility_at={1: ("rare",)})
    assert gate.update(data(3, "rare")).overall == "INSUFFICIENT"
    assert gate.terminal
    with pytest.raises(ValueError, match="terminal"):
        gate.update(data(3, "rare"))
