> **ARCHIVED — historical, superseded on 2026-10-03 (DECISIONS D-010, D-011). Never act on this file; the live version is listed in `DOC_AUTHORITY.md`.**

# 05 — How the deliverables map to NVIDIA roles (Sep 2026)

Reference only. Postings change; verify live before applying.

## Role A (live, ~Sep 2026): Senior Machine Learning and Simulation Engineer – Autonomous Vehicles
AV Simulation team, Santa Clara. Level 5 $224k–356k, Level 6 $272k–431k + equity.
Mission: closed-loop RL frameworks for end-to-end driving models.
Source (job-board mirror): https://www.dreamworkhq.com/job/90345157-b49f-4ea3-a830-0d2e1e6dcf69

```
   ROLE RESPONSIBILITY                          PHYSLOOP DELIVERABLE
 ① Lead large-scale RL training frameworks ──►  G7 AlpaGym throughput profile + PRs; G5 RL fine-tune
 ② Build/optimize simulation & data pipes  ──►  G2 validity envelope; G8 flywheel interface
 ③ Improve sim quality & reward functions  ──►  G2 (where the sim stops being true); G1 taxonomy → rewards; G4
 ④ Monitoring & debugging for GPU training ──►  G7 timing/idle/sync/staleness; G1 evidence store
 ⑤ Integrate SOTA models into pipelines    ──►  G5 teacher → ≤3B student; G6 FP8/NVFP4 incl. expert
```

| # | Requirement | Evidence | Fit |
|---|---|---|---|
| R1 | RL algorithms (PPO, GRPO) | G5 GRPO on the student; DSRL ablation | Strong |
| R2 | Reward design | G1 taxonomy → reward terms; G4 shows what the guardrail hides | Strong |
| R3 | Large-scale ML training | G5 on 8×H100; G7 profile | Medium |
| R4 | Simulation for AV | G1–G2 on AlpaSim + NuRec | Strong |
| R5 | AI infrastructure | G7 perf + upstream PRs | Strong |
| R6 | Python | everything | Strong |
| R7 | C++ | G4 guardrail and the deploy runtime in C++ | Gap → fixed inside the project |
| R8 | GPU clusters, K8s/SLURM | run G3 through AlpaSim's Slurm/enroot path on a rented cluster | Gap → fixed |
| R9 | 12+ years | not addressable; answer with public evidence | N/A |
| P1 | RL/LLM training infra (preferred) | background + G7 | Strong |
| P2 | Closed-loop eval of E2E driving (preferred) | G1 + G2 + G3 — the core | Strongest |
| P3 | Large-scale data pipelines (preferred) | G8 (spec) | Medium |
| P4 | Algorithm optimization (preferred) | G6 + roofline | Strong |

## Role B (target of the mastery plan; posting closed Oct 2025): Principal Deep Learning Engineer, End-to-End Autonomous Driving
$272k–425.5k + equity. Duties 1–5 map to nanoAlpamayo (1–2) and PhysLoop (3–5); see
`03_MASTERY_PLAN.md` §0–§3 and the "standout" lines (deployable VLA, open-source/publications,
motion planning, large datasets, real-time optimization).
Source: https://builtin.com/job/deep-learning-engineer-end-end-autonomous-driving/6878054

## Role C (research track; closed Apr 2026): Senior Research Scientist – Autonomous Vehicles
AV Research Group (Alpamayo, OPTED). Needs publications; G3 preprint and the nanoAlpamayo
RL-through-expert study are the matching artifacts.
Source: https://builtin.com/job/senior-research-scientist-autonomous-vehicles/7323531

## What a hiring manager will check
```
 "Can they run our stack?"          →  G1 scored AlpaSim runs with manifests
 "Do they understand sim limits?"   →  G2 fidelity-vs-deviation curve
 "Can they measure, not guess?"     →  G3 gate validity (ρ, inversions)
 "Can they speed up our RL loop?"   →  G7 profile + merged PRs
 "Can they ship to Thor?"           →  G5 + G6 student under 16 GiB / 0.1 s
 "Do they think like production?"   →  G4 policy + guardrail, intervention rate
```
Apply after week 7 (G1–G4 exist, GTC poster submitted, ≥ 1 PR open).
