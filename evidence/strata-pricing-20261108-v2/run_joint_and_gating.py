"""Re-pricing after independent review (agent-authored, S2.6 [C]); assumptions, not measurements.

J: exact JOINT pass rate of the per-stratum non-inferiority guard with no effect anywhere. Strata are
   independent in the model, so the joint rate is the product of exact per-stratum rates
   P(upper < m), enumerated through the locked bound (the Holm flag, which only lowers GO and needs
   far larger differences, is ignored).
D: P(improvement claim) by design when the effect is concentrated: every stratum improves except one
   flat stratum (no harm), versus uniform improvement, plus harm in the rare stratum. Designs:
   pooled, pooled + guard on every stratum, pooled + guard gating only the large strata (the small
   one reported; a regression flag anywhere still blocks). 1,000 trials, seed 20261108.

Run from the repo root: .venv/bin/python evidence/strata-pricing-20261108-v2/run_joint_and_gating.py
"""

import json
from functools import lru_cache

from scipy.stats import binom

from gate.superiority_power import design_power
from gate.verdict import paired_road_bounds

OUT = "evidence/strata-pricing-20261108-v2/"
B_VALUES, MARGINS = (0.0, 0.02, 0.05), (0.05, 0.10)


@lru_cache(maxsize=None)
def thresholds(R, t, nb_max):
    """Smallest improved count na with upper(na, nb) < t, for nb = 0..nb_max (None if none)."""
    out, start = [], 0
    for nb in range(nb_max + 1):
        found = None
        for na in range(start, R - nb + 1):
            if paired_road_bounds([-1.0] * na + [1.0] * nb + [0.0] * (R - na - nb), 0.05)[1] < t:
                found = na
                break
        out.append(found)
        if found is None:
            out.extend([None] * (nb_max - nb))
            break
        start = found  # the threshold cannot fall as worsened roads rise
    return tuple(out)


def p_pass(R, b, t):
    """Exact P(upper < t) with each road better or worse with probability b (no effect)."""
    nb_max = int(binom.ppf(1 - 1e-12, R, max(B_VALUES))) + 1
    th = thresholds(R, t, min(nb_max, R))
    total = 0.0
    for nb, na in enumerate(th):
        if na is None:
            continue
        total += binom.pmf(nb, R, b) * binom.sf(na - 1, R - nb, b / (1 - b) if b < 1 else 0)
    return total


CONFIGS = {
    "1 x 441": [441],
    "2 x 220/221": [220, 221],
    "4 x 110": [110, 110, 110, 111],
    "300/100/41, all gating": [300, 100, 41],
    "300/100 gating, 41 reported": [300, 100],
}
out = {"joint_exact": [], "designs": []}
print("J. Exact joint guard pass rate, no effect anywhere (product over gating strata)")
print(f"{'gating strata':30s} | " + "  ".join(f"m={m:.2f} b={b:.2f}" for m in MARGINS for b in B_VALUES))
for name, sizes in CONFIGS.items():
    cells = []
    for m in MARGINS:
        for b in B_VALUES:
            joint = 1.0
            for R in sizes:
                joint *= p_pass(R, b, m)
            cells.append(joint)
            out["joint_exact"].append({"config": name, "sizes": sizes, "margin": m, "b": b, "joint": joint})
    print(f"{name:30s} | " + "  ".join(f"{c:15.3f}" for c in cells))

SCEN = [
    ("4 x 110, uniform +5pp, b .02", {"s1": 110, "s2": 110, "s3": 110, "s4": 111}, 0.02, 0.07, None, None),
    ("4 x 110, s4 flat, b .02", {"s1": 110, "s2": 110, "s3": 110, "s4": 111}, 0.02, 0.07, {"s4": (0.02, 0.02)}, None),
    ("300/100/41, uniform +5pp, b .02", {"common": 300, "mid": 100, "rare": 41}, 0.02, 0.07, None, ("common", "mid")),
    ("300/100/41, rare flat, b .02", {"common": 300, "mid": 100, "rare": 41}, 0.02, 0.07, {"rare": (0.02, 0.02)}, ("common", "mid")),
    ("300/100/41, uniform +5pp, b 0", {"common": 300, "mid": 100, "rare": 41}, 0.0, 0.05, None, ("common", "mid")),
    ("300/100/41, rare harmed +10pp, b 0", {"common": 300, "mid": 100, "rare": 41}, 0.0, 0.05, {"rare": (0.10, 0.0)}, ("common", "mid")),
]
print("\nD. P(improvement claim) by design; guard margin 0.10 (0.05 in brackets)")
for name, strata, qp, qm, effects, gating in SCEN:
    r = design_power(strata=strata, p_a=0.08, q_plus=qp, q_minus=qm, effects=effects, margin=0.05,
                     guard_margins=(0.05, 0.10), alpha=0.05, delta_min=0.016, trials=1000,
                     seed=20261108, gating_strata=gating)
    x = {k: v["improvement_rate"] for k, v in r["results"].items()}
    line = (f"{name:36s} | pooled {x['pooled']:.3f} | guard all {x['pooled_guard@0.1']:.3f}"
            f" [{x['pooled_guard@0.05']:.3f}] | joint guard GO {r['guard_joint_go_rate']['0.1']:.3f}")
    if gating:
        line += f" | gating {'/'.join(gating)} {x['pooled_gating@0.1']:.3f} [{x['pooled_gating@0.05']:.3f}]"
    print(line)
    out["designs"].append({"scenario": name, "report": r})
json.dump(out, open(OUT + "joint_and_gating.json", "w"), indent=1)
