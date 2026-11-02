"""What docs/02_PHYSLOOP_PLAN.md S1 fixes about verdict(), whatever its internals.

Candidate B is tested against reference A. Every scene gets its own road here, so these cases say
nothing about clustering. The margin is an illustrative test input, not a pre-registered Δ.
"""

import pytest

pytestmark = pytest.mark.star

MARGIN = 0.05


def rows(stratum, n, b_only=0, both=0):
    """n scenes: `both` fail under A and B, `b_only` fail only under B, the rest pass under both."""
    fails = [(True, True)] * both + [(False, True)] * b_only
    fails += [(False, False)] * (n - len(fails))
    return [
        {
            "scene_id": f"{stratum}-{i}",
            "road_id": f"{stratum}-{i}",
            "stratum": stratum,
            "a_fail": a_fail,
            "b_fail": b_fail,
        }
        for i, (a_fail, b_fail) in enumerate(fails)
    ]


def test_clean_candidate_on_large_strata_is_go(run_verdict):
    overall, strata = run_verdict(
        rows("urban", 220, both=20) + rows("highway", 221, both=20), MARGIN
    )
    assert overall == "GO"
    assert strata == {"urban": "GO", "highway": "GO"}


def test_nine_scene_stratum_is_insufficient_and_blocks_go(run_verdict):
    overall, strata = run_verdict(rows("urban", 432, both=20) + rows("rare", 9), MARGIN)
    assert strata["rare"] == "INSUFFICIENT"
    assert overall == "INSUFFICIENT"


def test_gross_regression_is_no_go(run_verdict):
    overall, strata = run_verdict(
        rows("urban", 220, b_only=60) + rows("highway", 221, both=20), MARGIN
    )
    assert strata["urban"] == "NO-GO"
    assert overall == "NO-GO"
