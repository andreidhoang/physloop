"""Exact guard pass probability versus stratum size (agent-authored, S2.6 [C]).

One binary outcome per road, no true harm: each road is better or worse with probability b (noise
discordance), so P(non-inferiority GO) = P(upper < margin) by enumeration of (better, worse)
counts through the locked bound; the Holm regression flag is ignored (it can only lower GO, and
needs a road-mean difference far above these noise levels). Illustrative inputs, not measurements.
Run from the repo root: .venv/bin/python evidence/strata-pricing-20261108-v1/guard_size_exact.py
"""

from scipy.stats import binom

from gate.verdict import paired_road_bounds


def p_upper_below(R, p_better, p_worse, t, alpha=0.05):
    total = 0.0
    for nb in range(R + 1):
        w = binom.pmf(nb, R, p_worse)
        if w < 1e-13:
            continue
        for na in range(R - nb + 1):
            if paired_road_bounds([-1.0] * na + [1.0] * nb + [0.0] * (R - na - nb), alpha)[1] < t:
                total += w * binom.sf(na - 1, R - nb, p_better / (1 - p_worse))
                break
    return total


print("P(guard GO | no harm) by stratum size R, margin m, discordance b (exact)")
print(f"{'R':>5s} | m=.05 b=0  b=.02  b=.05 | m=.10 b=0  b=.02  b=.05")
for R in (36, 41, 50, 72, 100, 110, 150, 220, 300):
    cells = []
    for m in (0.05, 0.10):
        for b in (0.0, 0.02, 0.05):
            cells.append(f"{p_upper_below(R, b, b, m):5.3f}")
    print(f"{R:5d} | {'  '.join(cells[:3])} | {'  '.join(cells[3:])}")
