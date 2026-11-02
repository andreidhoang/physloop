"""Known answers from docs/02_PHYSLOOP_PLAN.md S1 Acceptance and docs/03_MASTERY_PLAN.md §4 S1.

Each expected value is the plan's printed number, compared at the plan's printed precision. The
MDEs are exact McNemar power averaged over D ~ Bin(n, psi), not Connor's formula, which gives
1.93 / 2.66 / 3.76 pp. Sources: research/2026-10-03-verification/notes/evaluation_release_gating.md
rows 6.1-6.11; research/2026-10-27-role-alignment/notes/arith_check.out.txt.
"""

from fractions import Fraction
from math import ceil

import pytest

pytestmark = pytest.mark.star


@pytest.mark.parametrize(
    "sided, exact, printed",
    [
        (1, Fraction(31_180, 2**18), 0.11894),
        (2, Fraction(31_180, 2**17), 0.23788),
    ],
)
def test_twelve_vs_six_exact_mcnemar(core, sided, exact, printed):
    p = core("exact_mcnemar_p")(12, 6, sided=sided)
    assert Fraction(p).limit_denominator(2**18) == exact
    assert round(float(p), 5) == printed


@pytest.mark.parametrize(
    "n, tail, percent, digits",
    [
        (441, 0.05, 0.677, 3),
        (441, 0.025, 0.833, 3),
        (1_000, 0.05, 0.2991, 4),
        (1_000, 0.025, 0.3682, 4),
    ],
)
def test_zero_failure_bound(core, n, tail, percent, digits):
    """A two-sided 95% Clopper-Pearson upper limit on 0/n is the one-sided bound at tail 0.025."""
    assert round(100 * core("zero_failure_upper")(n, tail), digits) == percent


@pytest.mark.parametrize("psi, percentage_points", [(0.021, 1.94), (0.04, 2.72), (0.08, 3.90)])
def test_mde_on_441_scenes(core, psi, percentage_points):
    mde = core("exact_mcnemar_mde")(psi, n=441, alpha=0.05, power=0.80)
    assert round(100 * mde, 2) == percentage_points


def test_twelve_vs_six_has_power_022(core):
    power = core("exact_mcnemar_power")(delta=6 / 441, psi=18 / 441, n=441, alpha=0.05)
    assert round(power, 2) == 0.22


@pytest.mark.parametrize("psi, percentage_points", [(0.04, 2.67), (0.08, 3.77)])
def test_smallest_non_inferiority_margin(core, psi, percentage_points):
    margin = core("ni_margin")(psi, n=441, alpha=0.025, power=0.80)
    assert round(100 * margin, 2) == percentage_points


def test_connor_n_for_the_rl_gate(core):
    """A 20% relative cut of an 8% rate is 1.6 pp; at psi = 4% it needs 964 paired scenes."""
    n = core("connor_n")(delta=0.016, psi=0.04, alpha=0.05, power=0.80, sided=1)
    assert ceil(n) == 964


@pytest.mark.parametrize("sigma_d, percentage_points", [(0.3, 3.55), (0.5, 5.92)])
def test_continuous_score_mde(core, sigma_d, percentage_points):
    mde = core("continuous_mde")(sigma_d, n=441, alpha=0.05, power=0.80)
    assert round(100 * mde, 2) == percentage_points


def test_four_strata_at_080_give_04096_overall(core):
    assert round(core("overall_go_power")([0.80] * 4), 4) == 0.4096


def test_080_overall_needs_09457_per_stratum(core):
    assert round(core("per_stratum_power")(0.80, k=4), 4) == 0.9457


def test_replicate_branch_at_three_replicates_and_rho_06(core):
    scenes = core("equal_precision_scenes")(m=3, rho=0.6)
    assert round(scenes, 4) == 1.3636
    assert round(100 * (1 - scenes / 3), 1) == 54.5


@pytest.mark.parametrize("alpha, first, second", [(0.025, 2.7965, 1.9774), (0.05, 2.3730, 1.6780)])
def test_obrien_fleming_two_looks(core, alpha, first, second):
    boundaries = core("obrien_fleming")(K=2, alpha=alpha)
    assert list(boundaries) == pytest.approx([first, second], abs=1e-4)


@pytest.mark.parametrize("alpha, boundary", [(0.025, 2.1783), (0.05, 1.8754)])
def test_pocock_two_looks(core, alpha, boundary):
    boundaries = core("pocock")(K=2, alpha=alpha)
    assert list(boundaries) == pytest.approx([boundary, boundary], abs=1e-4)
