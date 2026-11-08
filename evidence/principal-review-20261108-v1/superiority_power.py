"""Superiority sensitivity of the locked gate bound (agent-authored planning check, 2026-11-08).

Serves S2.6 [C] (operational power) for the S4/G9 improvement question. Read-only use of the
protected `gate.verdict.paired_road_bounds`; nothing protected is modified. All inputs are
illustrative assumptions, not measurements and not Huy's margins or predictions.

Question: with R independent roads in ONE stratum (the rule needs every stratum to pass), how large must a paired improvement be before the locked
one-sided upper endpoint of (candidate - teacher) failure difference falls below zero, i.e. before
"improvement" (superiority) is supported at alpha = 0.05? The qualified `verdict()` only certifies
non-inferiority: it rejects margin <= 0, so no qualified superiority rule exists yet.

Run: .venv/bin/python evidence/principal-review-20261108-v1/superiority_power.py
"""

import importlib

import numpy as np
from scipy.stats import binom

pb = importlib.import_module("gate.verdict").paired_road_bounds
ALPHA = 0.05


def binary_gate(R, b):
    """Smallest OBSERVED net improvement (about 50% power if the truth equals it) (a - b) with upper < 0 when a fraction a of roads improve by -1,
    a fraction b worsen by +1 and the rest are tied (one binary outcome per road)."""
    nb = round(b * R)
    for na in range(R - nb + 1):
        d = np.array([-1.0] * na + [1.0] * nb + [0.0] * (R - na - nb))
        if pb(d, ALPHA)[1] < 0:
            return (na - nb) / R
    return None


def binary_sign_test(R, b):
    """Same quantity for an exact one-sided sign test on discordant roads (reference only)."""
    nb = round(b * R)
    for na in range(R - nb + 1):
        n = na + nb
        if n and binom.sf(na - 1, n, 0.5) < ALPHA:
            return (na - nb) / R
    return None


def exact_binary_power(R, p_better, p_worse):
    """Exact power of the rule upper < 0 when each road independently improves with probability
    p_better and worsens with probability p_worse (multinomial). The bound depends only on the
    counts, so power = sum_nb P(NB = nb) * P(NA >= threshold(nb) | NB = nb)."""
    total = 0.0
    for nb in range(R + 1):
        w = binom.pmf(nb, R, p_worse)
        if w < 1e-12:
            continue
        thr = None
        for na in range(R - nb + 1):
            d = np.array([-1.0] * na + [1.0] * nb + [0.0] * (R - na - nb))
            if pb(d, ALPHA)[1] < 0:
                thr = na
                break
        if thr is not None:
            total += w * binom.sf(thr - 1, R - nb, p_better / (1 - p_worse))
    return total


def continuous_gate(R, sigma, rng, trials=200, power=0.8):
    """Smallest mean improvement delta for which >= power of trials give upper < 0, with road
    differences ~ N(-delta, sigma) clipped to [-1, 1]."""
    for delta in np.arange(0.0, 0.40, 0.005):
        hits = sum(
            pb(np.clip(rng.normal(-delta, sigma, R), -1, 1), ALPHA)[1] < 0 for _ in range(trials)
        )
        if hits >= power * trials:
            return float(delta)
    return None


def main():
    print("A. Binary road outcome: minimum net improvement, percentage points of roads")
    print("   b = fraction of roads where the candidate is worse (true regressions plus noise)")
    for R in (100, 200, 441, 1000):
        cells = []
        for b in (0.0, 0.02, 0.05, 0.10):
            g, s = binary_gate(R, b), binary_sign_test(R, b)
            cells.append(f"b={b:.2f}: gate {100 * g:.1f} / sign {100 * s:.1f}")
        print(f"   R={R:<5}", " | ".join(cells))

    print("\nC. Exact binary power at R=441 (per stratum): roads worse with prob b, net improvement n")
    for b in (0.0, 0.02, 0.05):
        cells = []
        for net in (0.016, 0.023, 0.030, 0.050, 0.073, 0.095):
            cells.append(f"net {100 * net:.1f}pp: {exact_binary_power(441, b + net, b):.2f}")
        print(f"   b={b:.2f}", " | ".join(cells))

    print("\nB. Continuous [0,1] road score: mean improvement for 80% power (200 trials, seed 20261108; Monte Carlo precision about +/-0.005)")
    rng = np.random.default_rng(20261108)
    for R in (200, 441):
        for sigma in (0.05, 0.10, 0.20):
            g = continuous_gate(R, sigma, rng)
            z = (1.645 + 0.842) * sigma / np.sqrt(R)
            print(f"   R={R:<4} sigma={sigma:.2f}: gate {g:.3f} | paired-z reference {z:.3f} | ratio {g / z:.1f}x")


if __name__ == "__main__":
    main()
