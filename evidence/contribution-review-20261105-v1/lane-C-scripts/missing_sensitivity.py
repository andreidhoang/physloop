"""Researcher C: worst-case (Manski-style) imputation of missing/failed rollouts through
the unmodified gate. Illustrative equivalent-pair model as in power_tables.py.
Run from repo root: uv run --frozen python <this file> <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

from gate.verdict import paired_road_bounds, verdict

ALPHA = 0.05
rng = np.random.default_rng(20261106)
N, REPS, TRIALS = 441, 3, 300
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def scores(p, size):
    fail = rng.random(size) < p[:, None]
    prog = np.where(rng.random(size) < 0.85, 1.0, rng.uniform(0.3, 1.0, size))
    return np.where(fail, 0.0, prog)


rows_out = []
for n_roads in (147, 441):
    assign = np.arange(N) % n_roads
    for miss_rate in (0.0, 0.005, 0.01, 0.02, 0.05):
        go = {"complete_case_drop_scene": 0, "adverse": 0, "favourable_nogo": 0}
        ups_adv = []
        for _ in range(TRIALS):
            p = rng.beta(0.1 * 0.5, 0.9 * 0.5, N)  # mu=0.1, kappa=0.5 (illustrative)
            a, b = scores(p, (N, REPS)), scores(p, (N, REPS))
            miss_a = rng.random((N, REPS)) < miss_rate  # missing completely at random
            miss_b = rng.random((N, REPS)) < miss_rate
            # adverse to GO: missing B -> 0, missing A -> 1 (higher score is better)
            a_adv, b_adv = np.where(miss_a, 1.0, a), np.where(miss_b, 0.0, b)
            d_adv = (a_adv - b_adv).mean(1)
            d_road = np.array([d_adv[assign == r].mean() for r in range(n_roads)])
            up = paired_road_bounds(d_road, ALPHA)[1]
            ups_adv.append(up)
            go["adverse"] += int(up < 0.10)
            # complete-case: drop any scene with a missing rollout in either arm
            keep = ~(miss_a.any(1) | miss_b.any(1))
            d_cc = (a - b).mean(1)
            roads_cc = [d_cc[(assign == r) & keep] for r in range(n_roads)]
            d_road_cc = np.array([x.mean() for x in roads_cc if len(x)])
            go["complete_case_drop_scene"] += int(paired_road_bounds(d_road_cc, ALPHA)[1] < 0.10)
        rows_out.append({"roads": n_roads, "missing_rate_per_arm": miss_rate, "margin": 0.10,
                         "P_GO_adverse_imputation": go["adverse"] / TRIALS,
                         "P_GO_complete_case": go["complete_case_drop_scene"] / TRIALS,
                         "median_UB_adverse": float(np.median(ups_adv)),
                         "max_mean_shift_bound": 2 * miss_rate})

# Known-answer: imputation enters verdict() as complete rows, identical bound
p = rng.beta(0.05, 0.45, N)
a, b = scores(p, (N, REPS)), scores(p, (N, REPS))
b[0, 0] = 0.0
rows = [{"scene_id": f"s{i}", "road_id": f"r{i}", "stratum": "all", "replicate_id": j,
         "a_value": float(a[i, j]), "b_value": float(b[i, j])}
        for i in range(N) for j in range(REPS)]
vr = verdict(rows, margin=0.10, alpha=ALPHA, higher_is_worse=False)
assert abs(vr.details["all"].upper - paired_road_bounds((a - b).mean(1), ALPHA)[1]) < 1e-9
(out / "missing_sensitivity.json").write_text(json.dumps(rows_out, indent=1))
print("| roads | missing/arm | P(GO@0.10) adverse | P(GO@0.10) complete-case | median UB adverse |")
for r in rows_out:
    print(f"| {r['roads']} | {r['missing_rate_per_arm']} | {r['P_GO_adverse_imputation']:.3f} | "
          f"{r['P_GO_complete_case']:.3f} | {r['median_UB_adverse']:.4f} |")
