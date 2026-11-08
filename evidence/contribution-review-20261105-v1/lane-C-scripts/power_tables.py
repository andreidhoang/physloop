"""Researcher C power tables: operational gate upper bounds for EQUIVALENT pairs.

Agent-authored, read-only use of gate.verdict (locked; imported, never modified).
Run from the repo root:  uv run --frozen python <this file> <out_dir>

All margins, failure rates and heterogeneity values below are ILLUSTRATIVE planning
inputs, not measured AlpaSim quantities and not Huy's registered margins.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

from gate.verdict import bounded_mean_bounds, paired_road_bounds, verdict

ALPHA = 0.05
MARGINS = (0.01, 0.02, 0.05, 0.07, 0.10)
ROADS = (30, 60, 110, 147, 220, 441)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
results = {"alpha": ALPHA, "margins_illustrative": MARGINS}


def ub_symmetric(mplus, n, alpha=ALPHA):
    """Operational UB when mean(d+) = mean(d-) = mplus (equivalent pair, zero mean)."""
    upper_x = bounded_mean_bounds(mplus, n, alpha / 2)[1]
    lower_y = bounded_mean_bounds(mplus, n, alpha / 2)[0]
    return min(1.0, upper_x - lower_y)


def constructed(n, k, mag):
    """n roads, k discordant (k even): k/2 at +mag, k/2 at -mag, rest 0."""
    return [mag] * (k // 2) + [-mag] * (k // 2) + [0.0] * (n - k)


# ---- Table A: operational UB on constructed equivalent road vectors -----------------
table_a = []
for n in ROADS:
    for f in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        for mag in (0.05, 0.1, 0.25, 0.5, 1.0):
            if f == 0.0 and mag != 0.05:
                continue
            k = 2 * int(f * n // 2)
            d = constructed(n, k, mag if k else 0.0)
            lo, up = paired_road_bounds(d, ALPHA)
            mplus = float(np.maximum(d, 0).mean())
            # cross-check: UB depends only on the two component means
            assert abs(up - ub_symmetric(mplus, n)) < 1e-12 or mplus == 0
            # cross-check against the full verdict() path (single stratum)
            rows = [
                {"scene_id": f"s{i}", "road_id": f"r{i}", "stratum": "all",
                 "a_value": max(v, 0.0), "b_value": max(-v, 0.0)}
                for i, v in enumerate(d)
            ]
            vr = verdict(rows, margin=0.05, alpha=ALPHA, higher_is_worse=False)
            assert abs(vr.details["all"].upper - up) < 1e-9
            table_a.append({
                "roads": n, "discordant_fraction": k / n, "abs_road_diff": mag if k else 0.0,
                "mean_d_plus": mplus, "upper": up,
                **{f"GO_at_{m}": up < m for m in MARGINS},
            })
results["table_a_constructed_equivalent"] = table_a

# ---- Table B: largest mean(d+) (=mean(d-)) still permitting GO; all-zero floor -------
table_b = []
for n in ROADS:
    row = {"roads": n, "all_zero_upper": paired_road_bounds([0.0] * n, ALPHA)[1],
           "closed_form_all_zero": 1 - (ALPHA / 2) ** (1 / n),
           "holm_single_stratum_flag_threshold_mean": math.sqrt(2 * math.log(1 / ALPHA) / n)}
    for m in MARGINS:
        if ub_symmetric(1e-12, n) >= m:
            row[f"max_mean_d_plus_for_GO_at_{m}"] = None  # even all-zero cannot certify
            continue
        lo, hi = 0.0, 0.5
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ub_symmetric(mid, n) < m else (lo, mid)
        row[f"max_mean_d_plus_for_GO_at_{m}"] = lo
    table_b.append(row)
results["table_b_go_feasibility"] = table_b

# ---- Table C: illustrative rollout model -> road differences -> operational verdict ---
# Equivalent pair (e.g. bit-identical re-export) under UNSEEDED replicates: A and B are
# independent draws from the same per-scene law. Per rollout: fail w.p. p_s -> score 0;
# else score 1 w.p. 0.85, else Uniform(0.3,1) (illustrative progress shortfall).
# p_s ~ Beta(mu*kappa, (1-mu)*kappa): small kappa = scenes mostly always-pass/always-fail.
rng = np.random.default_rng(20261105)
N_SCENES, REPS, TRIALS = 441, 3, 300


def rollout_scores(p, size):
    fail = rng.random(size) < p[:, None]
    prog = np.where(rng.random(size) < 0.85, 1.0, rng.uniform(0.3, 1.0, size))
    return np.where(fail, 0.0, prog)


table_c = []
for mu in (0.1, 0.3, 0.5):
    for kappa in (0.5, 2.0, 10.0):
        for n_roads in (60, 147, 441):
            assign = np.arange(N_SCENES) % n_roads  # equal-ish scenes per road
            ups, sds, mps, ebs, normals, decisions = [], [], [], [], [], {m: 0 for m in MARGINS}
            for t in range(TRIALS):
                p = rng.beta(mu * kappa, (1 - mu) * kappa, N_SCENES)
                a = rollout_scores(p, (N_SCENES, REPS)).mean(1)
                b = rollout_scores(p, (N_SCENES, REPS)).mean(1)
                d_scene = a - b  # positive = B worse (higher is better metric)
                d_road = np.array([d_scene[assign == r].mean() for r in range(n_roads)])
                lo, up = paired_road_bounds(d_road, ALPHA)
                pval = math.exp(-n_roads * max(d_road.mean(), 0.0) ** 2 / 2)
                for m in MARGINS:
                    decisions[m] += int(up < m and pval > ALPHA)
                ups.append(up)
                sd = d_road.std(ddof=1)
                sds.append(sd)
                mps.append(np.maximum(d_road, 0).mean())
                # Diagnostic comparators (NOT authorization methods):
                c = math.log(2 / ALPHA)
                ebs.append(d_road.mean() + math.sqrt(2 * sd**2 * c / n_roads)
                           + 14 * c / (3 * (n_roads - 1)))  # Maurer-Pontil EB, range 2
                normals.append(d_road.mean() + norm.isf(ALPHA) * sd / math.sqrt(n_roads))
            if t == 0:
                pass
            table_c.append({
                "fail_rate_mu": mu, "kappa": kappa, "roads": n_roads,
                "scenes_per_road": N_SCENES / n_roads, "replicates": REPS, "trials": TRIALS,
                "mean_within_scene_flip_var": mu * (1 - mu) * kappa / (kappa + 1),
                "median_mean_d_plus": float(np.median(mps)),
                "median_sd_road": float(np.median(sds)),
                "median_upper_operational": float(np.median(ups)),
                "median_upper_empirical_bernstein_diag": float(np.median(ebs)),
                "median_upper_normal_oracle_diag": float(np.median(normals)),
                **{f"P_GO_at_{m}": decisions[m] / TRIALS for m in MARGINS},
            })
results["table_c_illustrative_rollout_model"] = table_c

# verdict() consistency on one simulated trial
p = rng.beta(0.3 * 2, 0.7 * 2, N_SCENES)
a = rollout_scores(p, (N_SCENES, REPS))
b = rollout_scores(p, (N_SCENES, REPS))
assign = np.arange(N_SCENES) % 147
rows = [{"scene_id": f"s{i}", "road_id": f"r{assign[i]}", "stratum": "all", "replicate_id": j,
         "a_value": float(a[i, j]), "b_value": float(b[i, j])}
        for i in range(N_SCENES) for j in range(REPS)]
vr = verdict(rows, margin=0.05, alpha=ALPHA, higher_is_worse=False)
d_road = np.array([(a - b).mean(1)[assign == r].mean() for r in range(147)])
assert abs(vr.details["all"].upper - paired_road_bounds(d_road, ALPHA)[1]) < 1e-9
results["verdict_consistency_check"] = {"upper": vr.details["all"].upper,
                                        "overall": vr.overall}

(out / "power_tables.json").write_text(json.dumps(results, indent=1))


def fmt(x):
    return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


lines = ["# Table B", "| roads | all-zero UB | Holm flag mean | " +
         " | ".join(f"max m+ GO@{m}" for m in MARGINS) + " |"]
for r in table_b:
    lines.append(f"| {r['roads']} | {fmt(r['all_zero_upper'])} | "
                 f"{fmt(r['holm_single_stratum_flag_threshold_mean'])} | " +
                 " | ".join(fmt(r[f'max_mean_d_plus_for_GO_at_{m}']) for m in MARGINS) + " |")
lines += ["", "# Table A (subset: f in {0.1,0.5}, |d| in {0.1,0.25,1.0})",
          "| roads | f | abs d | mean d+ | UB | " + " | ".join(f"GO@{m}" for m in MARGINS) + " |"]
for r in table_a:
    if r["discordant_fraction"] == 0 or (round(r["discordant_fraction"], 1) in (0.1, 0.5)
                                         and r["abs_road_diff"] in (0.1, 0.25, 1.0)):
        lines.append(f"| {r['roads']} | {r['discordant_fraction']:.3f} | {r['abs_road_diff']} | "
                     f"{r['mean_d_plus']:.4f} | {r['upper']:.4f} | " +
                     " | ".join("Y" if r[f"GO_at_{m}"] else "n" for m in MARGINS) + " |")
lines += ["", "# Table C", "| mu | kappa | roads | S/road | med m+ | med SD_road | med UB op | "
          "med UB EB(diag) | med UB normal(diag) | " +
          " | ".join(f"P(GO@{m})" for m in MARGINS) + " |"]
for r in table_c:
    lines.append(f"| {r['fail_rate_mu']} | {r['kappa']} | {r['roads']} | "
                 f"{r['scenes_per_road']:.2f} | {r['median_mean_d_plus']:.4f} | "
                 f"{r['median_sd_road']:.4f} | {r['median_upper_operational']:.4f} | "
                 f"{r['median_upper_empirical_bernstein_diag']:.4f} | "
                 f"{r['median_upper_normal_oracle_diag']:.4f} | " +
                 " | ".join(f"{r[f'P_GO_at_{m}']:.3f}" for m in MARGINS) + " |")
(out / "power_tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
