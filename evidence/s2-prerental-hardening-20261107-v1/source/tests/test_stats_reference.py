"""Agent-authored independent reference checks; these are not experiment results."""

from fractions import Fraction
from math import comb, sqrt

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import binom, norm

from gate import stats


def test_one_sided_mcnemar_preserves_direction():
    assert stats.exact_mcnemar_p(12, 6, sided=1) < 0.12
    assert stats.exact_mcnemar_p(6, 12, sided=1) > 0.95
    assert stats.exact_mcnemar_p(0, 0) == 1


def test_exact_power_matches_enumerated_multinomial_small_case():
    n, psi, delta, alpha = 8, 0.6, 0.4, 0.1
    expected = 0.0
    for d in range(n + 1):
        for b in range(d + 1):
            p = min(1, 2 * float(sum(Fraction(comb(d, k), 2**d) for k in range(min(b, d - b) + 1))))
            if p <= alpha:
                expected += binom.pmf(d, n, psi) * binom.pmf(b, d, (psi + delta) / (2 * psi))
    assert stats.exact_mcnemar_power(delta, psi, n, alpha) == pytest.approx(expected)


def test_mde_unattainable_is_infinite():
    assert np.isinf(stats.exact_mcnemar_mde(0.001, 9, 0.05, 0.8))


@pytest.mark.parametrize(
    "b,c,n,delta", [(12, 6, 441, 0.02), (0, 8, 30, -0.2), (9, 0, 30, 0.1), (0, 0, 100, 0.1)]
)
def test_tango_constrained_mle_against_direct_likelihood(b, c, n, delta):
    def negative_log_likelihood(q):
        probabilities = (q + delta, q, 1 - 2 * q - delta)
        counts = (b, c, n - b - c)
        return -sum(
            count * np.log(p) for p, count in zip(probabilities, counts, strict=True) if count
        )

    lo, hi = max(0, -delta), (1 - delta) / 2
    fit = minimize_scalar(
        negative_log_likelihood,
        bounds=(lo + 1e-12, hi - 1e-12),
        method="bounded",
        options={"xatol": 1e-13},
    )
    candidates = [(fit.fun, fit.x)]
    for q in (lo, hi):
        probabilities = (q + delta, q, 1 - 2 * q - delta)
        if all(
            p > 0 or count == 0 for p, count in zip(probabilities, (b, c, n - b - c), strict=True)
        ):
            candidates.append((negative_log_likelihood(q), q))
    q = min(candidates)[1]
    expected = (b - c - n * delta) / sqrt(n * (2 * q + delta - delta**2))
    assert stats.tango_score(b, c, n, delta) == pytest.approx(expected, abs=1e-7)


def test_tango_zero_discordance_nonzero_width_and_swap_symmetry():
    lo, hi = stats.tango_interval(0, 0, 100, alpha=0.05, sided=1)
    z = norm.isf(0.05)
    assert hi == pytest.approx(z * z / (100 + z * z))
    assert lo == pytest.approx(-hi)
    lo, hi = stats.tango_interval(12, 6, 441)
    reverse_lo, reverse_hi = stats.tango_interval(6, 12, 441)
    assert lo == pytest.approx(-reverse_hi)
    assert hi == pytest.approx(-reverse_lo)


@pytest.mark.parametrize("method", ["obrien_fleming", "pocock", "lan_demets_obf"])
@pytest.mark.parametrize("K", [2, 3])
def test_boundary_calibration_and_independent_conditional_integral(method, K):
    boundaries = stats.group_sequential_boundaries(K, 0.025, method)
    times = np.arange(1, K + 1) / K
    assert stats.sequential_crossing_probability(boundaries, times) == pytest.approx(
        0.025, abs=1e-9
    )
    if K == 2:
        r = sqrt(0.5)
        survival = quad(
            lambda x: norm.pdf(x) * norm.cdf((boundaries[1] - r * x) / sqrt(1 - r * r)),
            -np.inf,
            boundaries[0],
            epsabs=1e-11,
        )[0]
        assert 1 - survival == pytest.approx(0.025, abs=1e-9)
    else:
        # For the three-look Markov process, Z1 and Z3 are independent given Z2.
        # This gives an independent one-dimensional integral, unlike the
        # implementation's repeated Gauss--Legendre density propagation.
        r12, r23 = sqrt(0.5), sqrt(2 / 3)
        survival = quad(
            lambda z2: (
                norm.pdf(z2)
                * norm.cdf((boundaries[0] - r12 * z2) / sqrt(1 - r12**2))
                * norm.cdf((boundaries[2] - r23 * z2) / sqrt(1 - r23**2))
            ),
            -np.inf,
            boundaries[1],
            epsabs=1e-11,
        )[0]
        assert 1 - survival == pytest.approx(0.025, abs=1e-9)


def test_three_look_expected_sample_fraction_by_disjoint_stops():
    boundaries = stats.pocock(3, 0.05)
    drift = 2.5
    first = norm.sf(boundaries[0] - drift / sqrt(3))
    r = sqrt(0.5)
    b1, b2 = boundaries[0] - drift / sqrt(3), boundaries[1] - drift * sqrt(2 / 3)
    second = quad(
        lambda z1: norm.pdf(z1) * norm.sf((b2 - r * z1) / sqrt(1 - r * r)),
        -np.inf,
        b1,
        epsabs=1e-11,
    )[0]
    expected = first / 3 + second * 2 / 3 + (1 - first - second)
    assert stats.expected_sample_fraction(boundaries, drift=drift) == pytest.approx(expected)


def test_ld_cumulative_spending_not_constant_obf_shape():
    boundaries = stats.group_sequential_boundaries(3, 0.05, "lan_demets_obf")
    for k in range(1, 4):
        times = np.arange(1, k + 1) / 3
        spending = 2 * norm.sf(norm.isf(0.05 / 2) / sqrt(k / 3))
        assert stats.sequential_crossing_probability(boundaries[:k], times) == pytest.approx(
            spending, abs=1e-9
        )


@pytest.mark.parametrize("times", [[0.999, 1.0], [0.4, 1.0], [0.1, 0.9, 1.0]])
def test_irregular_or_near_coincident_information_times_are_rejected(times):
    with pytest.raises(ValueError, match="equally spaced"):
        stats.sequential_crossing_probability([2.0] * len(times), times)


def test_single_look_design_reduces_to_fixed_n():
    design = stats.sequential_design(1, 0.05, "obrien_fleming")
    assert design["n_max_inflation"] == pytest.approx(1)
    assert design["expected_fraction_null"] == 1
    assert design["expected_fraction_alternative"] == 1


def test_replicate_limits_and_iut_validation():
    assert stats.equal_precision_scenes(3, 0) == 3
    assert stats.equal_precision_scenes(3, 1) == 1
    assert stats.replicate_variance(2, 10, 3, 1) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        stats.overall_go_power([])


def test_rate_ratio_planning_null_and_exposure_order():
    assert stats.rate_ratio_power(0.1, 1, 100) == pytest.approx(0.05)
    assert stats.rate_ratio_power(0.1, 0.8, 1000) > stats.rate_ratio_power(0.1, 0.8, 100)


def test_bootstrap_keeps_roads_together_and_is_reproducible():
    values = [1, 1, -1, -1]
    roads = ["a", "a", "b", "b"]
    first = stats.road_bootstrap_interval(values, roads, seed=17, draws=200)
    assert first == stats.road_bootstrap_interval(values, roads, seed=17, draws=200)
    assert first == (-1.0, 1.0)
    with pytest.raises(ValueError):
        stats.road_bootstrap_interval([0, 0], ["a", "a"])


@pytest.mark.parametrize(
    "call",
    [
        lambda: stats.exact_mcnemar_p(-1, 2),
        lambda: stats.exact_mcnemar_power(0.2, 0.1, 100, 0.05),
        lambda: stats.zero_failure_upper(0, 0.05),
        lambda: stats.tango_interval(5, 6, 10),
        lambda: stats.equal_precision_scenes(3, -0.1),
        lambda: stats.sequential_crossing_probability([2, 2], [0.5, 0.5]),
    ],
)
def test_invalid_parameters_rejected(call):
    with pytest.raises(ValueError):
        call()
