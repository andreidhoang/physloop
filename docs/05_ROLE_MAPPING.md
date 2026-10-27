# 05 — How the deliverables map to live NVIDIA roles (verified 3 October 2026)

This is a reference document, v2.1. §1–§2 were read from NVIDIA's Workday career site on 2026-10-03 (`research/2026-10-03-verification/notes/role_skill_demand.md`). On 2026-10-25, two postings were re-confirmed live:
- jobs.nvidia.com 893391627291 = JR2005596;
- jobs.nvidia.com 893392738069 = JR2010863.

JR2010863 lists closed-loop E2E evaluation as nice-to-have, not required (`research/2026-10-25-teacher-to-thor/verification.md` D9). §3–§4 follow plan v4's stages. Postings change, so re-verify a posting before applying to it.

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

## 3. Stage → posting map (plan v4 stages)

| Stage (`02` §3) | Evidence it produces | Postings |
|---|---|---|
| S1 Gate core | Paired decision statistics, power, open-source artifact | JR2014651, JR2024273, JR2025038 |
| S2 Reproduce + reliability | Reliability of a closed-loop simulator; pinned protocol; measured throughput | JR2010863, JR2019461, JR2014651 |
| S3 Student verdict | Closed-loop paired verdict of a distilled 2B student vs its teacher; regression reporting | JR2025038, JR2014651, JR2010863 |
| S4 Teacher RL | Closed-loop RL on a 10B VLA; reward design; reward-hacking audit | JR2010863, JR2005597, JR2022179 |
| S5 Correction + CoC budget | On-policy correction of a flow-matching VLA; the reasoning-vs-latency curve | JR2022179, JR2005596, JR2019758 |
| S6 Jetson Thor serving | TensorRT engines, CUDA kernels, quantization verdicts, latency on Thor | JR2016777, JR2016768, JR2025038 |
| S7 HIL + latency injection | Real-time closed-loop integration; latency's cost in score | JR2025038, JR2024273, JR2016777 |
| S8 Shadow mode (optional) | Real-sensor rig, trigger-based data mining, OOD set | JR2022179, JR2020021, JR2023529 |

## 4. What a hiring manager will check

```
 "Can they run our stack?"            →  S2 scored AlpaSim runs with pinned manifests
 "Can they measure, not guess?"       →  S1 power table; S3 paired verdict on NVIDIA's own student
 "Can they post-train in closed loop?"→  S4 teacher RL + hacking audit; S5 on-policy correction
 "Can they ship to Thor?"             →  S6 p99 on Jetson AGX Thor (device named); S7 HIL with latency injection
 "Do they think like production?"     →  allowed-claims discipline; S8 no-actuation rig and privacy plan
 "Do they work upstream?"             →  AlpaSim issues/PRs (paired layer, manifest fields), Thor serving PRs
```

**Gaps no stage closes:** shipped production, years of experience, multi-node training. On-device Thor numbers come only if P-002 is approved (`03` §6).
