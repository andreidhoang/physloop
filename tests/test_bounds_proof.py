"""Independent finite-support coverage checks; these enumerate, not simulate.

The oracle is the binomial/multinomial probability of every sufficient statistic,
not another implementation of the confidence-bound formulas. Float summation
slack below concerns probability accumulation only, not a scientific tolerance.
"""

from math import comb, factorial, fsum

import pytest

from gate.sequential import SequentialGate
from gate.verdict import bounded_mean_bounds, paired_road_bounds

pytestmark = pytest.mark.star


@pytest.mark.parametrize("n", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("p", [0.0, 0.01, 0.1, 0.3, 0.5, 0.8, 0.99, 1.0])
@pytest.mark.parametrize("tail", [0.01, 0.025, 0.05, 0.1, 0.25])
def test_bernoulli_bounds_have_exact_enumerated_one_sided_coverage(n, p, tail):
    upper_errors = []
    lower_errors = []
    for successes in range(n + 1):
        mass = comb(n, successes) * p**successes * (1 - p) ** (n - successes)
        lower, upper = bounded_mean_bounds(successes / n, n, tail)
        assert 0 <= lower <= successes / n <= upper <= 1
        upper_errors.append(mass if upper < p else 0.0)
        lower_errors.append(mass if lower > p else 0.0)
    assert fsum(upper_errors) <= tail + 2e-14
    assert fsum(lower_errors) <= tail + 2e-14


@pytest.mark.parametrize("n", [1, 3, 6, 9])
@pytest.mark.parametrize(
    "p_plus,p_minus",
    [
        (0.0, 0.0),
        (0.0, 0.1),
        (0.05, 0.05),
        (0.2, 0.1),
        (0.1, 0.6),
        (0.8, 0.1),
        (0.4, 0.0),
        (0.5, 0.5),
    ],
)
@pytest.mark.parametrize("alpha", [0.025, 0.05, 0.25])
def test_paired_bounds_cover_trinomial_road_population(n, p_plus, p_minus, alpha):
    p_zero = 1 - p_plus - p_minus
    delta = p_plus - p_minus
    upper_errors = []
    lower_errors = []
    masses = []
    for plus in range(n + 1):
        for minus in range(n - plus + 1):
            zero = n - plus - minus
            multiplicity = factorial(n) // (factorial(plus) * factorial(minus) * factorial(zero))
            mass = multiplicity * p_plus**plus * p_minus**minus * p_zero**zero
            differences = [1.0] * plus + [-1.0] * minus + [0.0] * zero
            lower, upper = paired_road_bounds(differences, alpha)
            assert -1 <= lower <= upper <= 1
            masses.append(mass)
            upper_errors.append(mass if upper < delta else 0.0)
            lower_errors.append(mass if lower > delta else 0.0)
    assert fsum(masses) == pytest.approx(1.0, abs=2e-14)
    assert fsum(upper_errors) <= alpha + 2e-14
    assert fsum(lower_errors) <= alpha + 2e-14


@pytest.mark.parametrize("mean", [1e-320, 1e-250, 1e-10, 1e-5, 1 - 1e-10])
def test_bounds_remain_finite_when_kl_root_is_beyond_float_range(mean):
    lower, upper = bounded_mean_bounds(mean, 100, 0.05)
    assert 0 <= lower <= mean <= upper <= 1


def test_closed_stratum_scene_cannot_reenter_under_a_new_road_and_stratum():
    def row(scene, road, stratum):
        return {
            "scene_id": scene,
            "road_id": road,
            "stratum": stratum,
            "a_fail": False,
            "b_fail": False,
        }

    gate = SequentialGate(["common", "rare"], margin=0.1)
    first = [row(f"common-{i}", f"common-road-{i}", "common") for i in range(80)]
    first += [row(f"rare-{i}", f"rare-road-{i}", "rare") for i in range(9)]
    result = gate.update(first)
    assert result.strata == {"common": "GO", "rare": "INSUFFICIENT"}
    before = gate.to_dict()
    with pytest.raises(ValueError, match="scene|metadata|duplicate"):
        gate.update([row("common-0", "fresh-road", "rare")])
    assert gate.to_dict() == before


def test_mutating_saved_checkpoint_does_not_change_live_futility_rule():
    gate = SequentialGate(["s"], margin=0.1)
    saved = gate.to_dict()
    saved["futility_at"][1] = ("s",)
    assert gate.futility_at == {}
