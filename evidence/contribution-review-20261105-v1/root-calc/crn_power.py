"""Operational-gate power under independent vs common-random-number (CRN) pairing.

Illustrative only: base rates, discordance fractions, road counts and margins are
NOT measured AlpaSim values and NOT Huy's registered margins. Uses the locked
gate's own bound (read-only import) so the numbers are the operational method's.
Model: R independent roads, 441 scenes split evenly, r replicates per scene,
binary endpoint per rollout, equal marginal rate p in both arms (equivalence) or
candidate rate (1-cut)*p. Joint law per (scene, replicate): P(a!=b)=psi.
Independent arms: psi = p_a(1-p_b)+p_b(1-p_a). CRN: psi scaled by f (residual).
"""
import json, sys
import numpy as np
from gate.verdict import paired_road_bounds, bounded_mean_bounds

def draw(rng, R, scenes, reps, pa, pb, f):
    # residual discordance fraction f of the independent-arm discordance
    psi_ind = pa * (1 - pb) + pb * (1 - pa)
    psi = f * psi_ind
    # need P(a=1,b=0)=x, P(a=0,b=1)=y with x+y=psi, x-y=pa-pb
    x = (psi + (pa - pb)) / 2
    y = (psi - (pa - pb)) / 2
    if x < 0 or y < 0:
        return None
    k = scenes // R
    n = R * k * reps
    u = rng.random(n)
    # diff b-a: +1 with prob y (a=0,b=1), -1 with prob x
    d = np.where(u < y, 1.0, np.where(u < y + x, -1.0, 0.0))
    d = d.reshape(R, k, reps).mean(axis=2).mean(axis=1)
    return d

def run(trials=400, seed=20261105):
    rng = np.random.default_rng(seed)
    out = []
    for R in (50, 100, 200, 441):
        for reps in (1, 3):
            for p in (0.04, 0.10):
                for cut in (0.0, 0.2):
                    for f in (1.0, 0.5, 0.2, 0.05):
                        ups = []
                        for _ in range(trials):
                            d = draw(rng, R, 441, reps, p, p * (1 - cut), f)
                            if d is None:
                                break
                            ups.append(paired_road_bounds(d, 0.05)[1])
                        if not ups:
                            continue
                        ups = np.array(ups)
                        out.append(dict(R=R, reps=reps, p=p, cut=cut, f=f,
                            median_upper=round(float(np.median(ups)), 4),
                            go_at_0_02=round(float((ups < 0.02).mean()), 3),
                            go_at_0_05=round(float((ups < 0.05).mean()), 3),
                            go_at_0_10=round(float((ups < 0.10).mean()), 3)))
    return out

if __name__ == "__main__":
    rows = run()
    # all-zero floor per R
    floor = {R: bounded_mean_bounds(0.0, R, 0.025)[1] for R in (50, 100, 183, 200, 368, 441)}
    json.dump({"rows": rows, "all_zero_upper": floor}, sys.stdout, indent=0)
