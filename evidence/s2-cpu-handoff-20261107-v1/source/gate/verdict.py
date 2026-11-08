"""Conservative road-cluster gate. Agent-authored under the S1 autonomy instruction.

See docs/S1_CORRECTNESS.md for assumptions and finite-sample error guarantees.
The row API is a trusted, already-paired boundary; use contracts for raw arms.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy


def _probability(value, name):
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be between zero and one")


def _kl(p, q):
    return float(xlogy(p, p) - xlogy(p, q) + xlogy(1 - p, 1 - p) - xlogy(1 - p, 1 - q))


def bounded_mean_bounds(mean, n, tail):
    """Two separately one-sided Chernoff bounds for independent [0,1] samples."""
    _probability(tail, "tail")
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive independent-unit count")
    if not math.isfinite(mean) or not 0 <= mean <= 1:
        raise ValueError("mean must be in [0,1]")
    c = -math.log(tail) / n
    if mean == 0:
        lower, upper = 0.0, -math.expm1(-c)
    elif mean == 1:
        lower, upper = math.exp(-c), 1.0
    else:
        eps = np.finfo(float).eps
        lo, hi = np.nextafter(0.0, 1.0), 1 - eps
        lower = (
            0.0 if _kl(mean, lo) <= c else brentq(lambda q: _kl(mean, q) - c, lo, mean, xtol=5e-15)
        )
        upper = (
            1.0 if _kl(mean, hi) <= c else brentq(lambda q: _kl(mean, q) - c, mean, hi, xtol=5e-15)
        )
    return max(0.0, lower - 1e-12), min(1.0, upper + 1e-12)


def paired_road_bounds(differences, alpha=0.05):
    """Each endpoint is individually one-sided 1-alpha, not joint 1-alpha."""
    _probability(alpha, "alpha")
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or np.any(abs(d) > 1):
        raise ValueError("road differences must be a nonempty vector in [-1,1]")
    lx, ux = bounded_mean_bounds(float(np.maximum(d, 0).mean()), len(d), alpha / 2)
    ly, uy = bounded_mean_bounds(float(np.maximum(-d, 0).mean()), len(d), alpha / 2)
    return max(-1.0, lx - uy), min(1.0, ux - ly)


def holm_flags(pvalues, alpha=0.05):
    _probability(alpha, "alpha")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("p values must be finite probabilities")
    rejected = set()
    for i, (name, p) in enumerate(sorted(pvalues.items(), key=lambda item: (item[1], item[0]))):
        if p > alpha / (len(pvalues) - i):
            break
        rejected.add(name)
    return rejected


@dataclass(frozen=True)
class StratumResult:
    estimate: float | None
    lower: float | None
    upper: float | None
    n_roads: int
    n_scenes: int
    n_replicates: int
    regression_p: float
    reason: str
    out_of_envelope_a: int = 0
    out_of_envelope_b: int = 0


@dataclass(frozen=True)
class VerdictResult:
    overall: str
    strata: dict
    details: dict
    alpha: float
    margin: float
    method: str = "road_positive_negative_KL; Hoeffding_Holm_regression"
    estimand: str = "equal road / equal scene within road / equal replicate within scene"


def verdict(
    rows, margin, alpha=0.05, required_strata=None, higher_is_worse=True, regression_family=None
):
    """GO only when every registered stratum's upper bound is below margin."""
    _probability(alpha, "alpha")
    _probability(margin, "margin")
    if not isinstance(higher_is_worse, bool):
        raise TypeError("metric direction must be boolean")
    rows = list(rows)
    seen, scene_meta, road_meta = set(), {}, {}
    by_stratum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    exits = defaultdict(lambda: [0, 0])
    representation = None
    for row in rows:
        for name in ("scene_id", "road_id", "stratum"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"missing or invalid {name}")
        scene, road, stratum = (row[k] for k in ("scene_id", "road_id", "stratum"))
        replicate = row.get("replicate_id", 0)
        if (
            isinstance(replicate, bool)
            or not isinstance(replicate, (str, int))
            or replicate == ""
            or (isinstance(replicate, int) and replicate < 0)
        ):
            raise ValueError("invalid replicate identity")
        key = (scene, replicate)
        if key in seen:
            raise ValueError("duplicate scene/replicate")
        seen.add(key)
        if scene in scene_meta and scene_meta[scene] != (road, stratum):
            raise ValueError("scene metadata changed")
        if road in road_meta and road_meta[road] != stratum:
            raise ValueError("road crosses strata")
        scene_meta[scene], road_meta[road] = (road, stratum), stratum
        if row.get("completed", True) is not True:
            raise ValueError("incomplete paired observation")
        binary = "a_fail" in row or "b_fail" in row
        continuous = "a_value" in row or "b_value" in row
        if binary and continuous:
            # Raw contract rows also contain values; do not accept ambiguous dual interfaces.
            raise ValueError("mixed outcome representations")
        current = "binary" if binary else "continuous"
        if representation is not None and representation != current:
            raise ValueError("mixed metric kinds")
        representation = current
        keys = ("a_fail", "b_fail") if binary else ("a_value", "b_value")
        try:
            a, b = (float(row[k]) for k in keys)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("missing or invalid outcome") from exc
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (a, b)):
            raise ValueError("outcomes must be finite in [0,1]")
        if binary and (a not in (0, 1) or b not in (0, 1)):
            raise ValueError("binary outcomes must be zero or one")
        diff = b - a if binary or higher_is_worse else a - b
        by_stratum[stratum][road][scene].append(diff)
        for index, arm in enumerate(("a", "b")):
            dev = row.get(f"{arm}_deviation_m")
            if dev is not None:
                if not math.isfinite(dev) or dev < 0:
                    raise ValueError("invalid deviation")
                exits[stratum][index] += int(dev >= 4)
    required = tuple(sorted(by_stratum)) if required_strata is None else tuple(required_strata)
    if (
        not required
        or len(set(required)) != len(required)
        or any(not isinstance(s, str) or not s for s in required)
    ):
        raise ValueError("register unique nonempty required strata")
    if set(by_stratum) - set(required):
        raise ValueError("unexpected stratum")
    details, pvalues = {}, {}
    for stratum in required:
        roads = by_stratum.get(stratum, {})
        if not roads:
            details[stratum] = StratumResult(None, None, None, 0, 0, 0, 1.0, "missing_stratum")
            pvalues[stratum] = 1.0
            continue
        d = np.array(
            [
                math.fsum(
                    math.fsum(sorted(values)) / len(values) for _, values in sorted(scenes.items())
                )
                / len(scenes)
                for _, scenes in sorted(roads.items())
            ]
        )
        lower, upper = paired_road_bounds(d, alpha)
        mean = float(d.mean())
        p = math.exp(-len(d) * max(mean, 0.0) ** 2 / 2)
        details[stratum] = StratumResult(
            mean,
            lower,
            upper,
            len(roads),
            sum(len(s) for s in roads.values()),
            sum(len(v) for s in roads.values() for v in s.values()),
            p,
            "noninferiority_bound" if upper < margin else "bound_does_not_certify",
            *exits[stratum],
        )
        pvalues[stratum] = p
    family = set(required) if regression_family is None else set(regression_family)
    if not set(required) <= family:
        raise ValueError("regression family cannot omit tested strata")
    flags = holm_flags({s: pvalues.get(s, 1.0) for s in family}, alpha)
    states = {
        s: (
            "NO-GO"
            if s in flags
            else "GO"
            if details[s].upper is not None and details[s].upper < margin
            else "INSUFFICIENT"
        )
        for s in required
    }
    overall = (
        "NO-GO"
        if "NO-GO" in states.values()
        else "GO"
        if all(s == "GO" for s in states.values())
        else "INSUFFICIENT"
    )
    return VerdictResult(overall, states, details, alpha, margin)
