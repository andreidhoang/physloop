# 05 — How the deliverables map to live NVIDIA roles (v3, 27 October 2026)

This is a reference document, v3. It replaces v2.1 (archived as `archive/05_ROLE_MAPPING_v2.1_2026-10-25.md`).
- **§1–§2** were first read from NVIDIA's Workday career site on 2026-10-03 (`research/2026-10-03-verification/notes/role_skill_demand.md`). §1 is refreshed from a sweep of NVIDIA's AV postings on 2026-10-27.
- **§3 is new.** It is the line-by-line evidence contract for the target role, JR2010863, under plan v4.1 (`02`, D-015). Its evidence is in `research/2026-10-27-role-alignment/`.
- **§4–§5** follow plan v4.1's stages.

The posting lines are paraphrased, never quoted. Postings change, so re-verify a posting before applying to it. Years, degrees and locations are out of scope for the plan by design (`03` §1). They appear in §1 only as the postings print them.

## 1. What NVIDIA is hiring for (refreshed 2026-10-27)

The 2026-10-27 sweep found 77 AV-titled engineering postings. Interns, Manager and Director roles, and JR2022903 were excluded. Most are platform, system-software, perception, planning or L4-integration roles. The core that matches PhysLoop is listed below, with US base-pay ranges by level. NVIDIA prints the same bands across these postings: L3 152,000–241,500; L4 184,000–287,500; L5 224,000–356,500; L6 272,000–431,250.

| Posting | Title | Location | Base pay (USD) | What it asks for that PhysLoop evidences |
|---|---|---|---|---|
| **JR2010863** (target) | Senior Machine Learning and Simulation Engineer – Autonomous Vehicles (AV Simulation team) | Santa Clara | L5 224,000–356,500; **L6 272,000–431,250**; 12+ years | A closed-loop, simulation-based RL framework for end-to-end AV models; measuring and improving simulation quality; RL reward design; reliable GPU-cluster training; closed-loop evaluation of end-to-end models (preferred). §3 maps every line |
| JR2022179 | Senior Deep Reinforcement Learning Engineer – Autonomous Driving | Santa Clara | 224,000–356,500; 12+ years | Flow-matching and diffusion policies; training on the policy's own rollouts; data flywheels (S4, S5) |
| JR2019292 | Senior Software Engineer – Autonomous Driving Simulation (AV Simulation team) | Santa Clara | L3–L4; 5 years in the AV industry | Simulation infrastructure on the same team as JR2010863 (S2) |
| JR2005596 | Senior Research Scientist – Autonomous Vehicles | Santa Clara | L4 192,000–304,750; L5 224,000–356,500 | Publications *or* research artifacts such as software projects |
| JR2005597 | Senior Research Engineer – Autonomous Vehicles | Santa Clara | L4 184,000–287,500; L5 224,000–356,500 | RL fine-tuning of multimodal LLMs; simulation infrastructure at scale |
| JR2025038 | Senior Integration Engineer, End-to-End Model – AV | Santa Clara | L3 152,000–241,500; L4 184,000–287,500 | Regression-performance tooling; quantization, compilation, profiling; C++ and QNX |
| JR2014651 | Senior Software Engineer, Metrics and Evaluation – AV | Santa Clara / remote (GA, DC, NC, WA) | L4 184,000–287,500; L5 224,000–356,500 | Evaluation products; precision and recall of metrics |
| JR2020021 | World-model evaluation systems – AV (reposted 2026-10-26) | Santa Clara | L4–L5 | Validation of world-model stacks (S1–S3) |
| JR2027161 | Deep Learning Senior Engineer, End-to-End Autonomous Driving | Santa Clara / CA remote | L4–L5 | End-to-end models (S3–S6) |
| JR2028518 | Senior Algorithm Engineer – AV (posted 2026-10-26) | — | not recorded | 10+ years, including 2+ in AV or robotics |
| JR2024273 | Senior Software Engineer, AV Planner | Santa Clara | 224,000–356,500 | Closed-loop benchmarks; regression prevention; fallback and degraded modes; C/C++ (not re-checked 2026-10-27) |
| JR2016777 | Senior Software Performance Engineer – AV | Santa Clara | L5 224,000–356,500; L6 272,000–431,250 | Latency and throughput (S6–S7) |
| JR2016768 | Manager, Deep Learning – AV and Robotics | Santa Clara | L3 224,000–356,500; L4 272,000–431,250 | TensorRT inference toolchains (not re-checked 2026-10-27) |
| JR2014137 | Lead Safety Architect – AV | Santa Clara / remote | 224,000–356,500 | Safety architecture (not re-checked 2026-10-27) |
| JR2023529 | Director, Engineering – AV | Santa Clara | 320,000–488,750 | Classical and learned systems in one production architecture |
| JR2019758 | Deep Learning Senior Engineer, End-to-End AD | Shanghai / Beijing | not listed | Building LLMs, VLMs or VLAs from scratch; C++ deployment (PhD + 2 years, or MS + 4 years) |

**Facts that correct v2.1:**
- **JR2019461** (Senior Research Manager, World Model Evaluation) was **not live** on 2026-10-27.
- **New since v2.1:** JR2019292, JR2027161 and JR2028518. JR2020021 was retitled and reposted on 2026-10-26.
- **Where years and degrees sit.** 71 of the 77 postings say "or equivalent experience", but always on the degree line, never in place of years. That is why the plan targets the work, not the screen (`03` §1).

**Facts kept from v2.1:**
- **No US "Principal … End-to-End" posting is live.** NVIDIA's E2E, VLA and closed-loop-RL hiring is at Senior level (Levels 3–6). JR2010863's L6 band equals the Principal band.
- **v1's "Role B"** (Principal DL Engineer, E2E AD; closed Oct 2025) lives on only as the China Senior role JR2019758.
- **JR2010863's simulation duty** includes *measuring* simulation quality. The job-board mirror that v1 cited had dropped that word.

## 2. Skill demand (share of qualification sections, ±5–10 pts; read 2026-10-03)

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

**Experience.** 30% of the 2026-10-03 NVIDIA postings (13 of 43) require 12+ years; none of the 212 competitor postings does. Postings treat public evidence as a complement to experience, never as a substitute.

## 3. JR2010863 evidence contract (plan v4.1)

**How to read it.** Each line is a paraphrase of the posting with a stable ID. "v4" is the earlier analysis of plan v4 (Full / Partial / None). "v4.1" is the mark once the finished plan delivers its [C] items; [O] and [G] items raise it only if they run. Every artifact is a measurement on NVIDIA's own open stack, made on one rented node unless P-012 runs. None is claimed as industry or production experience (`02` §6).

### 3.1 The map

| ID | The line (paraphrased) | v4 | Evidence after v4.1 (stage, tier, change tag) | v4.1 | What still cannot be shown |
|---|---|---|---|---|---|
| **I1** | The team's centre: a closed-loop, simulation-based RL framework that trains end-to-end AV models such as Alpamayo R1 | — | S4: closed-loop GRPO on Alpamayo 1.5 in AlpaGym, NVIDIA's own framework built on cosmos-rl. S5: a closed-loop, state-visited correction loop for NVIDIA's 2B student [C] | **Full** (as a user of the framework) | Owning that framework in production |
| **I2** | Improve the accuracy and performance of the RL framework and of the simulation, using NuRec, traffic models and the Cosmos world model | — | **Accuracy:** S2's reliability, instrument-noise decomposition, dose–response validity envelope and the pinned Harmonizer, NVIDIA's Cosmos-based enhancer [C] (L3-1…L3-4, L4-1). **Performance:** S2's rollout profile and the expert-CUDA-graph saving behind a bitwise test [C] (L7-1, L7-2); S4's AlpaGym balance and scaling model [C] (L2-1). **Traffic models:** only through P-013, replay vs CAT-K [G]. **Cosmos world model:** the Harmonizer only; a Cosmos3-Nano-Sim-Auto adapter is rejected because it would race NVIDIA, and does not run on the plan's setup | **Partial+**; Full on accuracy and performance if [C] lands | Traffic-model improvement without P-013; any Cosmos world-model simulator work |
| **D1** | Lead the design and development of large-scale RL training frameworks for multimodal AV foundation models | Partial | S4: a design document plus measurement. The AlpaGym bottleneck analysis and scaling model is fitted on 1–8 GPUs with NVIDIA's own presets, and predicts the 3- and 5-node presets. It comes with the estimator audit and the `perf` telemetry [C] (L2-1, L2-2, L6-5). A measured multi-node point needs P-012 [G] | **Partial+** (design-level, one node) | Leading a team; building or owning a framework at scale |
| **D2** | Design, build and optimize simulation and data-processing pipelines for scalable driving-policy training | Partial | S2: the measured rollout profile and goodput [C]; the capacity and service-balance model [O]. S5: the manifest-driven DAgger pipeline, with observations from AlpaSim's logs, offline teacher labels, deduplication, road-disjoint splits, and throughput per stage on NVIDIA's own metrics [C] (L5-5). A1's chunk-aware manifest [C] (L5-6) | **Full** (one node) | Fleet-scale pipelines |
| **D3** | Measure and improve simulation quality; refine the RL reward | Full | **Measure:** S2's reliability, noise decomposition, envelope, scorer known-answer test, and seed-coupling audit [C]. **Improve:** the Harmonizer pin, informed by a cost pilot [C]; renderer and traffic agreement only through P-013/P-014 [G]. **Reward:** S4's config-only, one-variable arms, zero-weight hacking monitors and hacking audit [C] (L2-3, L2-4) | **Full** (measure, reward); Partial (improve) | Improving the renderer or traffic model itself |
| **D4** | Keep training on large GPU clusters reliable and fast with monitoring and debugging tools | Partial | S2: DCGM fields, the goodput waterfall and alert rules [C] (L6-3); failure injection with idempotent resume [O] (L6-4). S4: the Flight Recorder and NCCL RAS on stalls [C]. S5: hang and crash drills under NVIDIA's Resiliency Extension [C] (L6-6) | **Partial+** (one node) | Fleet-scale incident handling |
| **D5** | Work with researchers to bring state-of-the-art architectures into scalable training pipelines | Partial | S3: a batched AlpaSim driver for the 2B student architecture, timing-only until weights exist, which S5 reuses as its collection driver [C] (L5-7). S5's learner is NVIDIA's own stage-2 distillation trainer [C] (L5-1). S2: NVIDIA's expert CUDA graph brought into the AlpaSim driver [C] (L7-2) | **Full** | Collaboration inside NVIDIA's research teams |
| **Q3** | Deep RL (PPO, GRPO), including hyperparameter tuning and reward design | Full (GRPO) | S4: Huy's derivation of the estimator as AlpaGym implements it, plus parity [C]; reward arms [C]; a successive-halving tuning study [O] (L2-5); anytime-valid monitoring of the arms [O] (L4-6) | **Full** on GRPO and reward; tuning only with [O] | PPO with a learned critic is not exercised |
| **Q4** | Exceptional C++ and Python for efficient systems and data pipelines | Partial | Python throughout. C++: TensorRT-11-ready engines and `IPluginV3` operations (S6) [C]; a production-grade C++ driver server with sanitizers, a golden bitwise replay and per-stage histograms (S7) [C] (L7-7, L7-9). On-device numbers need P-002 | **Full** if S6–S7 run | C++ outside the inference path |
| **Q5** | Large-scale GPU clusters, HPC, Kubernetes/SLURM | None | Slurm with Pyxis and Enroot through AlpaGym's shipped single-node presets (S4) [C] and AlpaSim's own Slurm run methods (S2) [O]; array jobs, requeue and autoresume. Multi-node only through P-012 [G]. **Kubernetes: none, by design.** AlpaSim ships no Kubernetes run method, and NVIDIA had no short-term fix for running its renderer under Kubernetes without Docker-in-Docker (#72) | **Partial** (one node) | Large clusters; Kubernetes |
| **X1** | (stand-out) RL or LLM training/fine-tuning infrastructure built in industry | None | Stand-in only: the S4 design document, the measured profiles, and upstream issues and PRs to AlpaSim | **None** (stand-in) | Industry experience, by definition |
| **X2** | (stand-out) Simulation and closed-loop evaluation of end-to-end driving models | Full | S1–S3: the gate, with interim looks and intersection–union accounting [C]; S7: HIL with latency injection [C] | **Full** | — |
| **X3** | (stand-out) Large-scale data pipelines and algorithm optimization | Partial | S2: the expert-graph saving and the Amdahl-bounded profile [C]. S5: the pipeline and the flow-step ladder anchored on NVIDIA's deployed settings [C]. S6: the kernel diagnosis [C], plus P-007 | **Full** (one node) | Fleet-scale data |

**Tally.**

| Mark | v4 (11 lines) | v4.1 (13 lines, adding I1 and I2) |
|---|---|---|
| Full | 3 | 7: I1, D2, D3, D5, Q3, X2, X3 |
| Full if S6–S7 run | — | Q4 |
| Partial+ | — | 3: I2, D1, D4 |
| Partial | 6 | Q5 |
| None | 2 | X1, by definition |

The two required engineering lines that were weak, Q4 and Q5, move to Full-if-S6–S7-run and to Partial.

### 3.2 What a reviewer can check, per line
- **"Can they measure a simulator?"** (D3, I2) S2's report: replicate reliability with CIs, the noise decomposition, the dose–response envelope, the seed-coupling audit, and the full simulator manifest.
- **"Can they make our loop faster, and prove it did not change the result?"** (I2, X3, D5) The expert-graph bitwise test and its measured GPU-hour saving. The rollout profile, with an Amdahl bound computed before each lever ran.
- **"Do they understand where an RL loop spends its time?"** (D1, X1 stand-in) The S4 design document: the balance equation, the fitted model, held-out prediction error, and the predicted 3-node speed-up, labelled as not measured.
- **"Can they run on our cluster stack?"** (Q5, D4) Slurm job records (`sacct`), requeue and resume behaviour, DCGM dashboards, goodput waterfalls, Flight Recorder and resiliency drills, all from one node.
- **"Can they post-train in closed loop without fooling themselves?"** (Q3, D3) The estimator audit and parity check, the one-variable reward arms, the zero-weight monitors and the hacking audit. Verdicts include INSUFFICIENT where power is lacking.
- **"Can they integrate a new model and ship it to Thor?"** (D5, Q4) The batched student driver; TensorRT-11-ready C++; the C++ driver server's sanitizer and golden-replay results; p99 on Jetson AGX Thor with the device named.

### 3.3 Rules that keep the evidence honest
- **Scale is always named:** "one node, 8 GPUs" or "2–4 GPUs". A multi-node number is either measured (P-012) or labelled predicted.
- **NVIDIA's own definitions are used** wherever they exist: AlpaSim's stabilized `seconds_per_rollout`, and GPU-hours from the mean wall clock (recipes PR #44). The numbers are then comparable with NVIDIA's.
- **Nothing races NVIDIA.** No Cosmos3-Nano-Sim-Auto adapter, no generic training-efficiency harness, no competing `n_sim_steps` PR, and no full-scale PR #42 student.
- **Licence gates apply to every public number** (KC-L) and every released checkpoint (KC-L2), including throughput measured on NuRec scenes, on a conservative reading of §3.

## 4. Stage → posting map (plan v4.1 stages)

| Stage (`02` §3) | Evidence it produces | JR2010863 lines | Other postings |
|---|---|---|---|
| S1 Gate core | Paired decision statistics, group-sequential tables, intersection–union power, open-source artifact | X2, D3 | JR2014651, JR2024273, JR2025038 |
| S2 Reproduce + reliability | Reliability of a closed-loop simulator with a noise decomposition; pinned protocol; rollout profile; the expert-graph saving; DCGM and goodput; Slurm [O] | I2, D2, D3, D4, X2, X3 (Q5 via [O]) | JR2019292, JR2014651, JR2005597 |
| S3 Student verdict | Paired closed-loop verdict of a distilled 2B student against its teacher; the batched student driver; regression reporting | X2, D3, D5, I1 | JR2025038, JR2014651, JR2027161 |
| S4 Teacher RL | Closed-loop RL on a 10B VLA; estimator audit; reward arms and hacking audit; AlpaGym bottleneck and scaling model; single-node Slurm presets | I1, I2, D1, D3, D4, Q3, Q5 | JR2022179, JR2005597 |
| S5 Correction + CoC budget | On-policy correction of a flow-matching VLA as a measured pipeline; flow-step ladder; reasoning-vs-latency curve; scaling and resiliency ladder | I1, D2, D4, D5, X3 | JR2022179, JR2005596, JR2027161, JR2019758 |
| S6 Jetson Thor serving | TensorRT engines (TensorRT-11-ready C++), CUDA kernels, quantization verdicts, latency on Thor | Q4, X3, I2 | JR2016777, JR2016768, JR2025038 |
| S7 HIL + latency injection | Real-time closed-loop integration through AlpaSim's external-driver mode; production C++ driver server; latency's cost in score | Q4, X2, D3 | JR2025038, JR2024273, JR2016777 |
| S8 Shadow mode (optional) | Real-sensor rig, trigger-based data mining, OOD set | D2 (weak) | JR2022179, JR2020021, JR2023529 |

## 5. What a hiring manager will check

```
 "Can they run our stack?"                 →  S2 scored AlpaSim runs with full simulator manifests (replicates, seed mode, Harmonizer)
 "Can they measure, not guess?"            →  S1 power and sequential tables; S2 noise decomposition; S3 paired verdict on NVIDIA's own student
 "Can they make the loop faster, safely?"  →  S2 rollout profile + expert CUDA graph behind a bitwise test; S4 AlpaGym balance and scaling model
 "Can they post-train in closed loop?"     →  S4 estimator audit, reward arms, monitors, hacking audit; S5 state-visited correction as a measured pipeline
 "Can they run on a cluster?"              →  Slurm/Enroot/Pyxis on one node, DCGM and goodput, Flight Recorder and resiliency drills; multi-node only via P-012
 "Can they ship to Thor?"                  →  S6 p99 on Jetson AGX Thor (device named); S7 HIL with latency injection; C++ driver server with sanitizers
 "Do they think like production?"          →  allowed-claims discipline; scale and origin always named; S8 no-actuation rig and privacy plan
 "Do they work upstream?"                  →  AlpaSim issues and PRs (paired layer, manifest fields, DCGM fields, scorers); Thor serving PRs
```

**Gaps no stage closes:**
- shipped production;
- years of experience;
- industry infrastructure (X1);
- Kubernetes;
- PPO with a learned critic;
- multi-node training at scale. P-012 adds one measured multi-node point at most.

On-device Thor numbers come only if P-002 is approved (`03` §6).
