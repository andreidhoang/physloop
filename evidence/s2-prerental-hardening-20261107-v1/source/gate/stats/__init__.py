"""Agent-authored S1 statistical references, authorized for this task by Huy.

These functions expose assumptions; numerical agreement is not calibration of a
clustered release gate. Independent-pair functions must not consume flattened
replicates. Sequential planning assumes a canonical Gaussian process with known
information fractions and independent increments, efficacy stopping only.

Primary references: Tango (1998), equations 24--26 and the inverted score test,
https://www.eiti.uottawa.ca/~nat/Courses/csi5388/Tango.paired.pdf ; gsDesign's
documented Lan--DeMets spending function,
https://keaven.github.io/gsDesign/reference/sfLDOF.html . The plan's numerical
known answers are independent references in tests/test_known_answers.py.
"""

from functools import lru_cache
from math import expm1, isfinite, log, prod, sqrt
from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import binom, norm


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value, name, closed=False):
    if not isfinite(value) or not (0 <= value <= 1 if closed else 0 < value < 1):
        raise ValueError(f"{name} must be a probability")
    return float(value)


def _sided(sided):
    if sided not in (1, 2):
        raise ValueError("sided must be 1 or 2")


def exact_mcnemar_p(b, c, sided=2):
    """Exact conditional p-value; the one-sided alternative is b > c.

    b counts candidate-only failures; c reference-only failures. Two-sided
    testing doubles the smaller tail (symmetry at p=1/2). No discordance gives 1.
    """
    b, c = _integer(b, "b", 0), _integer(c, "c", 0)
    _sided(sided)
    if b + c == 0:
        return 1.0
    if sided == 1:
        return float(binom.sf(b - 1, b + c, 0.5))
    return min(1.0, float(2 * binom.cdf(min(b, c), b + c, 0.5)))


@lru_cache(maxsize=64)
def _mcnemar_rejection(n, alpha, sided):
    d = np.arange(n + 1)
    tail = alpha / sided
    lower = binom.ppf(tail, d, 0.5).astype(int)
    lower -= binom.cdf(lower, d, 0.5) > tail
    return d, lower


def exact_mcnemar_power(delta, psi, n, alpha=0.05, sided=2):
    """Unconditional IID-pair power averaged over D~Bin(n, psi).

    delta=P(candidate-only failure)-P(reference-only failure); psi is total
    discordance. Positive delta is the directional alternative when sided=1.
    """
    n = _integer(n, "n")
    _probability(psi, "psi", closed=True)
    _probability(alpha, "alpha")
    _sided(sided)
    if not isfinite(delta) or abs(delta) > psi:
        raise ValueError("abs(delta) must not exceed discordance psi")
    if psi == 0:
        return 0.0
    d, lower = _mcnemar_rejection(n, float(alpha), sided)
    q = (1 + delta / psi) / 2
    rejection = binom.sf(d - lower - 1, d, q)
    if sided == 2:
        rejection += binom.cdf(lower, d, q)
    return float(np.dot(binom.pmf(d, n, psi), rejection))


def exact_mcnemar_mde(psi, n, alpha=0.05, power=0.8, sided=2):
    """Smallest nonnegative IID-pair difference; infinity if unattainable."""
    _probability(power, "power")
    attained = exact_mcnemar_power(psi, psi, n, alpha, sided)
    if attained < power:
        return float("inf")
    if exact_mcnemar_power(0, psi, n, alpha, sided) >= power:
        return 0.0
    return float(
        brentq(
            lambda delta: exact_mcnemar_power(delta, psi, n, alpha, sided) - power,
            0,
            psi,
            xtol=1e-13,
        )
    )


def ni_margin(psi, n, alpha=0.025, power=0.8):
    """Normal planning margin at true difference zero; not a Tango CI."""
    _probability(psi, "psi", closed=True)
    return continuous_mde(sqrt(psi), n, alpha, power)


def connor_n(delta, psi, alpha=0.05, power=0.8, sided=1):
    """Connor normal sample-size approximation, no continuity correction.

    Returns a real-valued IID-pair count, to be rounded up by the caller.
    """
    _probability(psi, "psi")
    _probability(alpha, "alpha")
    _probability(power, "power")
    _sided(sided)
    if not isfinite(delta) or not 0 < abs(delta) <= psi:
        raise ValueError("nonzero abs(delta) must not exceed psi")
    return float(
        ((norm.isf(alpha / sided) * sqrt(psi) + norm.ppf(power) * sqrt(psi - delta**2)) / delta)
        ** 2
    )


def continuous_mde(sigma_d, n, alpha=0.05, power=0.8):
    """One-sided known-variance normal planning MDE for paired differences."""
    _integer(n, "n")
    _probability(alpha, "alpha")
    _probability(power, "power")
    if not isfinite(sigma_d) or sigma_d < 0:
        raise ValueError("sigma_d must be finite and nonnegative")
    return float((norm.isf(alpha) + norm.ppf(power)) * sigma_d / sqrt(n))


def zero_failure_upper(n, tail=0.05):
    """Exact one-sided Clopper--Pearson upper bound for independent 0/n."""
    _integer(n, "n")
    _probability(tail, "tail")
    return -expm1(log(tail) / n)


def overall_go_power(stratum_powers):
    """Product under INDEPENDENT stratum decisions, not merely disjoint IDs."""
    powers = tuple(stratum_powers)
    if not powers:
        raise ValueError("at least one stratum is required")
    return prod(_probability(p, "stratum power", closed=True) for p in powers)


def per_stratum_power(overall, k):
    _probability(overall, "overall", closed=True)
    _integer(k, "k")
    return overall ** (1 / k)


def equal_precision_scenes(m, rho):
    """Single-replicate scene multiple m/[1+(m-1)rho]; nonnegative ICC model."""
    _integer(m, "m")
    _probability(rho, "rho", closed=True)
    return m / (1 + (m - 1) * rho)


def replicate_variance(sigma, n, m, rho):
    _integer(n, "n")
    if not isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    return sigma**2 / (n * equal_precision_scenes(m, rho))


def tango_score(b, c, n, delta=0.0):
    """Tango's efficient score for IID paired risk difference (b-c)/n.

    Constrained MLE q=P(reference-only failure) satisfies
    2n q² + [-(b+c)+(2n-b+c)delta]q - c delta(1-delta)=0.
    Var(X_B-X_A) = 2q+delta-delta². This sign convention changes
    Tango's non-inferiority margin Delta to -delta (eqs. 24--26).
    """
    b, c, n = _integer(b, "b", 0), _integer(c, "c", 0), _integer(n, "n")
    if b + c > n or not isfinite(delta) or not -1 <= delta <= 1:
        raise ValueError("invalid paired counts or difference")
    coefficient = -(b + c) + (2 * n - b + c) * delta
    constant = -c * delta * (1 - delta)
    discriminant = max(0.0, coefficient**2 - 8 * n * constant)
    q = (-coefficient + sqrt(discriminant)) / (4 * n)
    q = min((1 - delta) / 2, max(max(0.0, -delta), q))
    variance = max(0.0, 2 * q + delta - delta**2)
    numerator = (b - c) / n - delta
    if variance == 0:
        return 0.0 if numerator == 0 else float(np.copysign(np.inf, numerator))
    return numerator * sqrt(n / variance)


def tango_interval(b, c, n, alpha=0.05, sided=2):
    """Invert the IID-pair efficient score; no continuity correction.

    sided=2 returns joint 1-alpha nominal limits. sided=1 returns two
    *separate* marginal 1-alpha bounds, not a joint 1-alpha interval.
    This asymptotic reference is not a road-cluster-adjusted verdict.
    """
    _probability(alpha, "alpha")
    _sided(sided)
    tango_score(b, c, n)
    estimate = (b - c) / n
    z = norm.isf(alpha / sided)
    lower = (
        -1.0
        if estimate == -1
        else brentq(lambda d: tango_score(b, c, n, d) - z, -1, estimate, xtol=1e-13)
    )
    upper = (
        1.0
        if estimate == 1
        else brentq(lambda d: tango_score(b, c, n, d) + z, estimate, 1, xtol=1e-13)
    )
    return float(lower), float(upper)


@lru_cache(maxsize=2)
def _quadrature(order):
    return leggauss(order)


def sequential_crossing_probability(boundaries, times=None, drift=0.0, quadrature_order=128):
    """P(any upper crossing) for canonical normal Z(t), mean=drift*sqrt(t).

    Deterministic Gauss--Legendre quadrature integrates Markov transitions.
    Centered Z is integrated over [-12, 12], an omitted marginal mass <4e-33.
    The supported reference scope is one to three equally spaced looks from
    zero (including prefixes of such a schedule). Near-coincident or otherwise
    irregular information times are rejected: their narrow transition densities
    have not been qualified for this quadrature. No futility boundary.
    """
    boundaries = np.asarray(boundaries, dtype=float)
    if boundaries.ndim != 1 or not 1 <= len(boundaries) <= 3:
        raise ValueError("one to three boundaries required")
    if times is None:
        times = np.arange(1, len(boundaries) + 1) / len(boundaries)
    times = np.asarray(times, dtype=float)
    if (
        times.shape != boundaries.shape
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
        or times[0] <= 0
        or times[-1] > 1
        or not np.all(np.isfinite(boundaries))
        or not isfinite(drift)
    ):
        raise ValueError("finite boundaries and strictly increasing times in (0,1] required")
    if not np.allclose(times, times[0] * np.arange(1, len(times) + 1), rtol=1e-12, atol=0):
        raise ValueError("only equally spaced information times from zero are supported")
    _integer(quadrature_order, "quadrature_order", 32)
    shifted = boundaries - drift * np.sqrt(times)
    if len(shifted) == 1:
        return float(norm.sf(shifted[0]))
    roots, weights = _quadrature(quadrature_order)
    density = points = None
    for k, boundary in enumerate(shifted):
        if boundary <= -12:
            return 1.0
        high = min(float(boundary), 12.0)
        nodes = -12 + (roots + 1) * (high + 12) / 2
        node_weights = weights * (high + 12) / 2
        if k == 0:
            new_density = np.exp(-(nodes**2) / 2) / sqrt(2 * np.pi)
        else:
            correlation = sqrt(times[k - 1] / times[k])
            sd = sqrt(1 - correlation**2)
            transition = np.exp(-0.5 * ((nodes[:, None] - correlation * points) / sd) ** 2)
            transition /= sqrt(2 * np.pi) * sd
            new_density = transition @ density
        points = nodes
        density = new_density * node_weights
    return float(np.clip(1 - density.sum(), 0, 1))


@lru_cache(maxsize=64)
def group_sequential_boundaries(K=2, alpha=0.05, method="obrien_fleming"):
    """One-sided canonical-normal boundaries at equally spaced information."""
    _integer(K, "K")
    _probability(alpha, "alpha")
    if K > 3:
        raise ValueError("only K<=3 is qualified by this reference")
    if method not in ("obrien_fleming", "pocock", "lan_demets_obf"):
        raise ValueError("unknown boundary method")
    times = np.arange(1, K + 1) / K
    if method in ("obrien_fleming", "pocock"):
        shape = 1 / np.sqrt(times) if method == "obrien_fleming" else np.ones(K)
        coefficient = brentq(
            lambda c: sequential_crossing_probability(c * shape, times) - alpha, -5, 10, xtol=1e-12
        )
        return tuple(float(x) for x in coefficient * shape)
    boundaries = []
    for k, t in enumerate(times):
        # gsDesign sfLDOF convention; alpha is one-sided total spending.
        spent = 2 * norm.sf(norm.isf(alpha / 2) / sqrt(t))
        boundary = brentq(
            lambda b, look=k, spending=spent: (
                sequential_crossing_probability([*boundaries, b], times[: look + 1]) - spending
            ),
            -5,
            12,
            xtol=1e-12,
        )
        boundaries.append(float(boundary))
    return tuple(boundaries)


def obrien_fleming(K=2, alpha=0.05):
    return group_sequential_boundaries(K, alpha, "obrien_fleming")


def pocock(K=2, alpha=0.05):
    return group_sequential_boundaries(K, alpha, "pocock")


def lan_demets_obf(K=2, alpha=0.05):
    return group_sequential_boundaries(K, alpha, "lan_demets_obf")


def expected_sample_fraction(boundaries, times=None, drift=0.0):
    """Expected n/n_max with upper-efficacy stopping only."""
    if times is None:
        times = np.arange(1, len(boundaries) + 1) / len(boundaries)
    sequential_crossing_probability(boundaries, times, drift)
    return float(
        times[0]
        + sum(
            (times[k] - times[k - 1])
            * (1 - sequential_crossing_probability(boundaries[:k], times[:k], drift))
            for k in range(1, len(boundaries))
        )
    )


def sequential_design(K=2, alpha=0.05, method="obrien_fleming", power=0.8):
    """Normal-planning inflation and expected n fractions, no futility.

    The null boundary delta=-Delta maps to drift 0; midpoint maps to theta/2;
    true difference zero maps to theta. Effect scale cancels in inflation.
    These tables are planning diagnostics, not sequential confidence bounds.
    """
    _probability(power, "power")
    if power <= alpha:
        raise ValueError("target power must exceed alpha")
    boundaries = group_sequential_boundaries(K, alpha, method)
    times = np.arange(1, K + 1) / K
    theta = brentq(
        lambda x: sequential_crossing_probability(boundaries, times, x) - power, 0, 20, xtol=1e-12
    )
    fixed_theta = norm.isf(alpha) + norm.ppf(power)
    inflation = (theta / fixed_theta) ** 2
    fractions = {
        name: expected_sample_fraction(boundaries, times, drift)
        for name, drift in (("null", 0.0), ("midpoint", theta / 2), ("alternative", theta))
    }
    return {
        "method": method,
        "K": K,
        "alpha": alpha,
        "power": power,
        "boundaries": boundaries,
        "information_times": tuple(times),
        "alternative_drift": theta,
        "n_max_inflation": float(inflation),
        **{f"expected_fraction_{name}": value for name, value in fractions.items()},
        **{f"expected_n_over_fixed_{name}": value * inflation for name, value in fractions.items()},
        "assumptions": "canonical Gaussian independent increments; efficacy only; known variance",
    }


def rate_ratio_power(base_rate, ratio, exposure_per_arm, alpha=0.05, null_ratio=1.0):
    """Independent-Poisson log-rate-ratio normal planning power (lower is better).

    Exposure is an abstract simulation denominator, never real-world miles.
    No road pairing, overdispersion or covariance is represented here. Expected
    event counts must be sufficiently large for this asymptotic approximation.
    """
    _probability(alpha, "alpha")
    if any(not isfinite(x) or x <= 0 for x in (base_rate, ratio, exposure_per_arm, null_ratio)):
        raise ValueError("rates, ratio, exposure and null_ratio must be positive")
    se = sqrt(1 / (base_rate * exposure_per_arm) + 1 / (base_rate * ratio * exposure_per_arm))
    return float(ndtr(log(null_ratio / ratio) / se - norm.isf(alpha)))


def rate_ratio_exposure(base_rate, ratio, alpha=0.05, power=0.8, null_ratio=1.0):
    """Per-arm exposure for independent-Poisson normal planning only."""
    rate_ratio_power(base_rate, ratio, 1.0, alpha, null_ratio)
    _probability(power, "power")
    if ratio >= null_ratio or power <= alpha:
        raise ValueError("planning alternative must be below null_ratio and power above alpha")
    return float(
        (norm.isf(alpha) + norm.ppf(power)) ** 2
        * (1 + 1 / ratio)
        / (base_rate * log(null_ratio / ratio) ** 2)
    )


def road_bootstrap_interval(values, road_ids, alpha=0.05, seed=0, draws=2000):
    """Diagnostic percentile bootstrap, resampling whole paired road clusters.

    Input is one pre-aggregated paired difference per scene. Resampled cluster
    sums/counts retain the scene-weighted estimand for unequal cluster sizes.
    A degenerate bootstrap is possible and must NOT automatically authorize GO.
    No finite-sample coverage guarantee or early-stopping correction is claimed.
    """
    _probability(alpha, "alpha")
    _integer(draws, "draws")
    _integer(seed, "seed", 0)
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(road_ids) or not np.all(np.isfinite(values)):
        raise ValueError("aligned finite paired differences and road IDs required")
    roads = {}
    for value, road in zip(values, road_ids, strict=True):
        if road is None or road == "":
            raise ValueError("road IDs must be present")
        total, count = roads.get(road, (0.0, 0))
        roads[road] = total + value, count + 1
    if len(roads) < 2:
        raise ValueError("at least two independent roads required")
    totals, counts = np.array(list(roads.values())).T
    rng = np.random.default_rng(seed)
    result = np.empty(draws)
    for i in range(draws):
        sample = rng.integers(0, len(roads), len(roads))
        result[i] = totals[sample].sum() / counts[sample].sum()
    return tuple(float(x) for x in np.quantile(result, [alpha / 2, 1 - alpha / 2]))
