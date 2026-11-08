"""Illustrative superiority/falsifier power through the real gate (agent-authored, D-023, S2.6).
Assumptions only, not measurements or Huy's margins. Run from repo root:
  .venv/bin/python evidence/superiority-rule-20261108-v1/run_power.py
"""
import json
from gate.superiority_power import superiority_power

SCEN = [
    # name, strata, p_a, q_plus (cand worse), q_minus (cand better)
    ("R441_b0_net3.0pp", {"a": 441}, 0.08, 0.0, 0.030),
    ("R441_b0_net1.6pp_target", {"a": 441}, 0.08, 0.0, 0.016),
    ("R441_b5_net9.5pp", {"a": 441}, 0.15, 0.05, 0.145),
    ("R441_null_b5", {"a": 441}, 0.08, 0.05, 0.05),
    ("R441_null_b0", {"a": 441}, 0.08, 0.0, 0.0),
    ("two_strata_220_each_b0_net3pp", {"a": 220, "b": 221}, 0.08, 0.0, 0.030),
]
out = []
for name, strata, p_a, qp, qm in SCEN:
    r = superiority_power(strata=strata, p_a=p_a, q_plus=qp, q_minus=qm, margin=0.05,
                          alpha=0.05, delta_min=0.016, trials=1000, seed=20261108)
    r["name"] = name
    out.append(r)
    o = r["outcomes"]
    print(f"{name:32s} supported {o['IMPROVEMENT_SUPPORTED']['rate']:.3f} "
          f"ruled_out {o['MEANINGFUL_IMPROVEMENT_RULED_OUT']['rate']:.3f} "
          f"insufficient {o['INSUFFICIENT']['rate']:.3f} regression {o['REGRESSION_FLAGGED']['rate']:.3f}")
json.dump(out, open("evidence/superiority-rule-20261108-v1/power.json", "w"), indent=1)
