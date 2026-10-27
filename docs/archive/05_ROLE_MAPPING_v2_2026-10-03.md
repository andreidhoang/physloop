# 05 — How the deliverables map to live NVIDIA roles (verified 3 October 2026)

This is a reference document. It was read from NVIDIA's Workday career site on 2026-10-03 (`research/2026-10-03-verification/notes/role_skill_demand.md`). Postings change, so re-verify a posting before applying to it.

## 1. What NVIDIA is hiring for

Workday lists about 90 live AV-related postings. Most are platform, system-software, perception, planning or L4-integration roles. The core that matches PhysLoop is listed below; US base-pay ranges are by level.

| Posting | Title | Location | Base pay (USD) | What it asks for that PhysLoop evidences |
|---|---|---|---|---|
| **JR2010863** (Role A) | Senior Machine Learning and Simulation Engineer – Autonomous Vehicles | Santa Clara | L5 224,000–356,500; **L6 272,000–431,250** (the Principal band); 12+ years | Closed-loop RL frameworks; *measuring and improving* simulation quality; GPU-training monitoring; closed-loop evaluation of E2E models (preferred) |
| JR2005596 | Senior Research Scientist – Autonomous Vehicles | Santa Clara | L4 192,000–304,750; L5 224,000–356,500 | Publications *or* research artifacts such as software projects |
| JR2005597 | Senior Research Engineer – Autonomous Vehicles | Santa Clara | L4 184,000–287,500; L5 224,000–356,500 | RL fine-tuning of multimodal LLMs; simulation infrastructure at scale |
| JR2022179 | Senior Deep Reinforcement Learning Engineer – Autonomous Driving | Santa Clara | 224,000–356,500 | Flow-matching and diffusion policies; training on the policy's own rollouts; data flywheels |
| JR2025038 | Senior Integration Engineer, End-to-End Model – AV | Santa Clara | L3 152,000–241,500; L4 184,000–287,500 | Regression-performance tooling; quantization, compilation, profiling; C++ and QNX |
| JR2014651 | Senior Software Engineer, Metrics and Evaluation – AV | Santa Clara / remote | L4 184,000–287,500; L5 224,000–356,500 | Evaluation products; precision and recall of metrics |
| JR2024273 | Senior Software Engineer, AV Planner | Santa Clara | 224,000–356,500 | Closed-loop benchmarks; regression prevention; fallback and degraded modes; C/C++ |
| JR2016777 | Senior Software Performance Engineer – AV | Santa Clara | L5 224,000–356,500; L6 272,000–431,250 | Latency and throughput |
| JR2016768 | Manager, Deep Learning – AV and Robotics | Santa Clara | L3 224,000–356,500; L4 272,000–431,250 | TensorRT inference toolchains |
| JR2019461 | Senior Research Manager, World Model Evaluation | Santa Clara | 272,000–431,250 | Validity, reproducibility, leakage prevention, model comparison; an evaluation-standards roadmap |
| JR2020021 | Senior SWE, World Model Systems Engineering – AV | Santa Clara | L4 184,000–287,500; L5 224,000–356,500 | V&V of world-model stacks |
| JR2014137 | Lead Safety Architect – AV | Santa Clara / remote | 224,000–356,500 | Safety architecture |
| JR2023529 | Director, Engineering – AV | Santa Clara | 320,000–488,750 | Classical and learned systems in one production architecture |
| JR2019758 | Deep Learning Senior Engineer, End-to-End AD | Shanghai / Beijing | not listed | Building LLMs, VLMs or VLAs from scratch; C++ deployment (PhD + 2 years, or MS + 4 years) |

**Facts that correct v1:**
- **No US "Principal … End-to-End" posting is live.** NVIDIA's E2E, VLA and closed-loop-RL hiring is at Senior level (Levels 3–6). Role A's L6 band equals the Principal band.
- **v1's "Role B"** (Principal DL Engineer, E2E AD; closed Oct 2025) lives on only as the China Senior role JR2019758.
- **v1's "Role C"** is live again as JR2005596, without the Alpamayo or OPTED tags.
- **Role A's simulation duty** includes *measuring* simulation quality. The job-board mirror that v1 cited had dropped that word.

## 2. Skill demand (share of qualification sections, ±5–10 pts)

Base: 43 NVIDIA AV postings and 212 postings from 14 competitors.

| Cluster | NVIDIA | Competitors | Usually listed as |
|---|---|---|---|
| C++ | 67% | 58% | Required |
| Python | 65% | 64% | Required |
| VLA / VLM / LLM | 49% | 27% | "Ways to stand out" |
| CUDA / kernels | 40% | 13% | Stand out |
| Simulation | 35% | 24% | Mixed |
| Real-time / embedded | 35% | 18% | Mixed |
| Publications | 33% | 24% | Stand out |
| Motion planning | 30% | 29% | Mixed |
| Evaluation / metrics | 28% (51% of responsibilities) | 39% (57%) | Responsibility |
| RL / post-training | 23% | 16% | Stand out |
| Imitation / E2E driving | 23% | 7% | Stand out |
| Data pipelines | 23% | 17% | Mixed |
| World models / neural recon | 21% | 13% | Stand out |
| Safety / V&V | 19% | 19% | Mixed |
| Open source | 14% | 4% | Stand out |
| Distributed training | 14% | 18% | Mixed |
| Quantization / TensorRT | 7% (12% of responsibilities) | 16% | Mixed |
| Kubernetes / SLURM / Ray | 9% | 12% | Mixed |

**Experience.** 30% of NVIDIA postings (13 of 43) require 12+ years; none of the 212 competitor postings does. Postings treat public evidence as a complement to experience, never as a substitute.

## 3. Stage → posting map

| Stage (`02` §3) | Evidence it produces | Postings |
|---|---|---|
| S1 Gate core | Paired decision statistics, power, open-source artifact | JR2014651, JR2024273, JR2025038 |
| S2 Gate v1 on AlpaSim | Reliability of a closed-loop simulator; regression reports | JR2010863, JR2019461, JR2014651 |
| S3 Validity envelope | Neural-reconstruction limits, agreement statistics | JR2010863, JR2020021, JR2019461 |
| S4 Degradation ladders + precision | Closed-loop quantization verdicts; S4b kernel diagnosis | JR2025038, JR2016768, JR2016777 |
| S5 Construct validity | A preprint on simulator validity | JR2005596, JR2019461, JR2010863 |
| S6 Guardrail (C++) | Classical-plus-learned dual architecture; C++ | JR2023529, JR2024273, JR2025038 |
| S7 nanoAlpamayo | VLA built from scratch; flow-matching RL; distillation | JR2019758, JR2022179, JR2005597 |
| S8 Throughput | Closed-loop evaluation throughput; GPU monitoring | JR2010863, JR2005597, JR2016777 |

## 4. What a hiring manager will check

```
 "Can they run our stack?"            →  S2 scored AlpaSim runs with manifests
 "Do they understand sim limits?"     →  S3 deviation-binned envelope + agreement
 "Can they measure, not guess?"       →  S1 power table, S5 pre-registered verdict
 "Can they ship to Thor?"             →  S4 precision verdicts, S7 student under 16 GiB
 "Do they think like production?"     →  S6 policy + guardrail, intervention rate
 "Do they work upstream?"             →  AlpaSim issues/PRs (CAT-K seeding, paired layer, monitor)
```

**Gaps no stage closes:** shipped production, years of experience, multi-node training, and on-device Thor numbers (`03` §6).
