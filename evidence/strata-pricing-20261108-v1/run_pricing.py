"""Price strata and guard-margin options for the S4 improvement claim (agent-authored, S2.6 [C]).

Every input is an assumption, not a measurement and not one of Huy's registered values:
441 independent roads at most, hypothetical teacher failure rate 8% per road, one scene and one
binary outcome per road unless stated, alpha 0.05, delta_min 0.016 (affects only "ruled out").
b = probability a road is worse with the candidate (true regressions plus noise discordance);
delta = net improvement = P(better) - P(worse). All cells reuse one seed, so they share random
numbers and differences between cells are not noise artefacts.

Run from the repo root: .venv/bin/python evidence/strata-pricing-20261108-v1/run_pricing.py
"""

import json
import sys
import time

from gate.superiority_power import design_power

OUT = "evidence/strata-pricing-20261108-v1/"
SEED, TRIALS, P_A, ALPHA, DMIN = 20261108, 1000, 0.08, 0.05, 0.016
CONFIGS = {
    "1 stratum (441)": {"all": 441},
    "2 strata (220/221)": {"s1": 220, "s2": 221},
    "4 strata (110 each)": {"s1": 110, "s2": 110, "s3": 110, "s4": 111},
    "3 unequal (300/100/41)": {"common": 300, "mid": 100, "rare": 41},
}
EFFECTS = [(d, b) for d in (0.0, 0.016, 0.03, 0.05) for b in (0.0, 0.02, 0.05)]
EFFECTS = [(d, b) for d, b in EFFECTS if (d, b) != (0.0, 0.0) and d + b <= P_A + 1e-12]


def run(config, *, q_plus, q_minus, effects=None, replicates=1, replicate_correlation=0.0):
    return design_power(
        strata=config,
        p_a=P_A,
        q_plus=q_plus,
        q_minus=q_minus,
        effects=effects,
        margin=0.05,
        guard_margins=(0.05, 0.10),
        alpha=ALPHA,
        delta_min=DMIN,
        trials=TRIALS,
        seed=SEED,
        replicates=replicates,
        replicate_correlation=replicate_correlation,
    )


def rates(report):
    return {k: v["improvement_rate"] for k, v in report["results"].items()}


def main():
    start = time.time()
    out = {"block_A_uniform": [], "block_C_harm": [], "block_E_replicates": []}
    print("A. Uniform effect: P(improvement claim) by design")
    print(f"{'config':24s} {'delta':>5s} {'b':>5s} | all_strata  pooled  +guard@.05  +guard@.10")
    for name, config in CONFIGS.items():
        for delta, b in EFFECTS:
            r = run(config, q_plus=b, q_minus=b + delta)
            x = rates(r)
            out["block_A_uniform"].append({"config": name, "delta": delta, "b": b, "report": r})
            print(
                f"{name:24s} {delta:5.3f} {b:5.2f} | {x['all_strata']:10.3f} {x['pooled']:7.3f}"
                f" {x['pooled_guard@0.05']:10.3f} {x['pooled_guard@0.1']:10.3f}"
            )
        sys.stdout.flush()

    print("\nC. Harm in one stratum (others delta 0.03, b 0.02): P(improvement claim), lower is safer")
    harms = [("4 strata (110 each)", "s4"), ("3 unequal (300/100/41)", "rare")]
    for name, harmed in harms:
        for h in (0.05, 0.10):
            r = run(CONFIGS[name], q_plus=0.02, q_minus=0.05, effects={harmed: (0.02 + h, 0.02)})
            x = rates(r)
            out["block_C_harm"].append({"config": name, "harmed": harmed, "harm": h, "report": r})
            print(
                f"{name:24s} harm {h:.2f} in {harmed:6s} | all_strata {x['all_strata']:.3f}"
                f" pooled {x['pooled']:.3f} +guard@.05 {x['pooled_guard@0.05']:.3f}"
                f" +guard@.10 {x['pooled_guard@0.1']:.3f}"
            )

    print("\nE. Three replicates per road (delta 0.03, b 0.02 per replicate)")
    for name in ("1 stratum (441)", "4 strata (110 each)"):
        for rho in (0.0, 0.5):
            r = run(CONFIGS[name], q_plus=0.02, q_minus=0.05, replicates=3, replicate_correlation=rho)
            x = rates(r)
            out["block_E_replicates"].append({"config": name, "replicate_correlation": rho, "report": r})
            print(
                f"{name:24s} copy prob {rho:.1f} | all_strata {x['all_strata']:.3f}"
                f" pooled {x['pooled']:.3f} +guard@.05 {x['pooled_guard@0.05']:.3f}"
                f" +guard@.10 {x['pooled_guard@0.1']:.3f}"
            )

    print("\nG. Guard cost: per-stratum non-inferiority GO rate with no harm (delta 0, b as shown)")
    for row in out["block_A_uniform"]:
        if row["delta"] == 0.0:
            g = row["report"]["guard_go_rate"]
            per = {m: min(v.values()) for m, v in g.items()}
            print(f"{row['config']:24s} b {row['b']:.2f} | lowest stratum GO@.05 {per['0.05']:.3f} GO@.10 {per['0.1']:.3f}")

    out["meta"] = {"seed": SEED, "trials": TRIALS, "p_a": P_A, "alpha": ALPHA, "delta_min": DMIN,
                   "seconds": round(time.time() - start, 1)}
    json.dump(out, open(OUT + "pricing.json", "w"), indent=1)
    print(f"\n{out['meta']['seconds']} s")


if __name__ == "__main__":
    main()
