"""Compare the locked operational bound with a direct variance-adaptive bound
(Maurer & Pontil 2009 empirical Bernstein, one-sided) on the same road differences.
Analysis only for a PROPOSAL; it does not authorize or replace the locked gate.
Also: minimum all-zero roads per margin, and per-stratum road support."""
import math, numpy as np, json, sys
from gate.verdict import paired_road_bounds

def eb_upper(d, alpha=0.05):
    x = (np.asarray(d) + 1) / 2  # [0,1]
    n = len(x); v = x.var(ddof=1) if n > 1 else 0.25
    l = math.log(2 / alpha)
    ub = x.mean() + math.sqrt(2 * v * l / n) + 7 * l / (3 * (n - 1))
    return 2 * min(ub, 1) - 1

def draw(rng, R, reps, p, f, scenes=441):
    psi = f * 2 * p * (1 - p); k = scenes // R
    u = rng.random(R * k * reps)
    d = np.where(u < psi / 2, 1.0, np.where(u < psi, -1.0, 0.0))
    return d.reshape(R, k, reps).mean(axis=2).mean(axis=1)

rng = np.random.default_rng(7)
rows = []
for R in (100, 441):
    for f in (1.0, 0.2):
        a, b = [], []
        for _ in range(300):
            d = draw(rng, R, 3, 0.04, f)
            a.append(paired_road_bounds(d, 0.05)[1]); b.append(eb_upper(d))
        rows.append(dict(R=R, f=f, locked_median=round(float(np.median(a)), 4), eb_median=round(float(np.median(b)), 4)))
minroads = {m: math.ceil(math.log(40) / -math.log1p(-m)) for m in (0.02, 0.05, 0.07, 0.10)}
json.dump({"bound_compare_R_f_p0.04_reps3": rows, "min_all_zero_roads_per_stratum": minroads}, sys.stdout, indent=1)
