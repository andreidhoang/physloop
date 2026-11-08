"""Follow-ups to run_pricing.py (agent-authored, S2.6 [C]); same assumptions and seed.

H: harm in one stratum when the others improve strongly (delta 0.05, b 0), the regime where a
   pooled claim alone is likely to pass. Harmed stratum: q_plus = h, q_minus = 0.
R: why three replicates lowered power in block E. The locked bound depends on road-level means of
   d+ and d-, not on their variance, so replicates shrink the spread of the upper bound without
   moving its median: power rises when the median is below zero and falls when it is above.

Run from the repo root: .venv/bin/python evidence/strata-pricing-20261108-v1/run_harm_and_replicates.py
"""

import json

import numpy as np

from gate.superiority_power import design_power
from gate.synthetic import generate
from gate.verdict import verdict

OUT = "evidence/strata-pricing-20261108-v1/"
SEED, TRIALS, P_A = 20261108, 1000, 0.08
CONFIGS = {
    "4 strata (110 each)": ({"s1": 110, "s2": 110, "s3": 110, "s4": 111}, "s4"),
    "3 unequal (300/100/41)": ({"common": 300, "mid": 100, "rare": 41}, "rare"),
}
out = {"harm_b0": [], "replicates": []}
print("H. Harm in one stratum, others delta 0.05 and b 0: P(improvement claim), lower is safer")
for name, (config, harmed) in CONFIGS.items():
    for h in (0.0, 0.05, 0.10, 0.25):
        r = design_power(
            strata=config, p_a=P_A, q_plus=0.0, q_minus=0.05,
            effects=None if h == 0 else {harmed: (h, 0.0)},
            margin=0.05, guard_margins=(0.05, 0.10), alpha=0.05, delta_min=0.016,
            trials=TRIALS, seed=SEED,
        )
        x = {k: v["improvement_rate"] for k, v in r["results"].items()}
        out["harm_b0"].append({"config": name, "harmed": harmed, "harm": h, "report": r})
        print(f"{name:24s} harm {h:.2f} in {harmed:5s} | all_strata {x['all_strata']:.3f}"
              f" pooled {x['pooled']:.3f} +guard@.05 {x['pooled_guard@0.05']:.3f}"
              f" +guard@.10 {x['pooled_guard@0.1']:.3f}")

print("\nR. One versus three replicates, one stratum of 441 roads, 300 trials")
for qp, qm in ((0.02, 0.05), (0.0, 0.03), (0.0, 0.02)):
    for reps in (1, 3):
        up = np.array([
            verdict(generate(5000 + t, {"all": 441}, replicates=reps, p_a=P_A, q_plus=qp,
                             q_minus=qm), 0.05, 0.05).details["all"].upper
            for t in range(300)
        ])
        row = {"b": qp, "delta": qm - qp, "replicates": reps, "p_upper_below_0": float(np.mean(up < 0)),
               "median_upper": float(np.median(up)), "sd_upper": float(up.std())}
        out["replicates"].append(row)
        print(f"b {qp:.2f} delta {qm - qp:.2f} reps {reps} | P(upper<0) {row['p_upper_below_0']:.3f}"
              f" median {row['median_upper']:+.4f} sd {row['sd_upper']:.4f}")
json.dump(out, open(OUT + "harm_and_replicates.json", "w"), indent=1)
