# 03 — Mastery plan v3.1 (27 October 2026)

**Status:** authoritative for the skill pillars, the per-stage curriculum, the mastery gates and the skill-coverage map. v3.1 amends v3 (2026-10-25) under D-015:
- pillar P8 is added;
- the S4 derivation is corrected;
- the S1–S5 and S7 exercises gain known answers;
- §6 is updated.

v2 is archived. The stage numbers follow plan v4.1 (`02_PHYSLOOP_PLAN.md`). It contains **no schedule**: stages S1–S8, their order, costs and kill checks live only in `02`. The evidence base is `research/2026-10-03-verification/`, `research/2026-10-25-teacher-to-thor/` and `research/2026-10-27-role-alignment/`.

**Target.** The capabilities behind NVIDIA's live end-to-end autonomous-driving postings (`05_ROLE_MAPPING.md`). Since 2026-10-27 the primary target is JR2010863 (Senior ML and Simulation Engineer – AV, AV Simulation team), and its line-by-line evidence contract is `05` §3. JR2005596 and JR2010863 were re-confirmed live on 2026-10-25 and 2026-10-27. Credential lines (years, PhD) are ignored by design. Every requirement becomes a skill with a measurable acceptance test and a public artifact.

**"Principal", stripped of tenure:**
- own one link of the loop end to end;
- decide with measurements;
- write the design doc others build on;
- ship code that survives review by the people who wrote the stack.

The evidence must be public and checkable: an artifact published from this repo, issues and PRs in AlpaSim, a technical report, reproduced numbers.

---

## 1. Read the role as a system

```
 [3] Data engine ──► [1][2] Train VLA (IL → SFT → closed-loop correction → RL) ──► [4] Evaluate & release ──► [5] Runtime on Thor
        ▲                                                                              │
        └──────────────────────── failures found in [4] and on the road ───────────────┘
```

This is NVIDIA's "AI 2.0" framing: a reasoning VLA replaces hand-built modules and is distilled to run on DRIVE AGX Thor next to a classical safety stack. PhysLoop's stages map onto the links as follows:

| Link | Stages (plan v4.1) |
|---|---|
| [4] Evaluate and release | S1–S3, S7 |
| [1] Train: RL | S4 (teacher RL) |
| [1] Train: correction | S5 (on-policy correction of the student) |
| [5] Runtime on Thor | S6 |
| [3] Data engine | S8 (shadow mode, optional) |

## 2. Seven skill pillars, with verified demand

The demand figure is the share of qualification sections that mention the cluster: 43 comparable NVIDIA AV postings and 212 postings from 14 competitors, read 2026-10-03, ±5–10 points.

| Pillar | Demand (NVIDIA / competitors) | You must be able to… | Built in | The trap |
|---|---|---|---|---|
| **P1** Driving as a decision problem | Motion planning 30% / 29% | Derive the unicycle model and AR1's exact heading update; explain why (accel, curvature) × 64 at 10 Hz; explain open- vs closed-loop metrics | S5, S7, S8 | Optimizing open-loop L2, which shows no significant correlation with closed-loop score (Bosch, n = 7); reading shadow-mode agreement as closed-loop evidence |
| **P2** VLA modeling | VLA/VLM/LLM 49% / 27% | Explain a multi-camera VLM with 180 tokens per image (patch 16 × merge 2), a discrete trajectory tokenizer, and a flow expert that reads the KV cache; know which path is deployed | S3, S5 | Treating the flow expert as a black box; not knowing which path is deployed |
| **P3** Learning algorithms | RL 23% / 16%; imitation / E2E 23% / 7% | DAgger, flow matching (CFM, Euler sampler, ODE→SDE for log-probs), GRPO with reward design and KL control, distillation (GKD), RoaD / OPTED closed-loop correction | S4, S5 | The proxy reward improves while behaviour degrades (AR1's reasoning-only row); a reasoning reward on a frozen VLM; rollout and trainer log-prob mismatch |
| **P4** Data engine | Data 23% / 17%; world models / neural recon 21% / 13% | Location-grouped splits; teacher contamination (`alpamayo_aligned`); auto-labeling CoC with a teacher; 3DGS rendering limits; trigger classifiers | S2, S5, S8 | Frame-level random splits: with k = 20 frames per location and f = 0.2, 98.85% of locations land in both splits |
| **P5** Simulation and evaluation | Evaluation 28% / 39% (51% / 57% of responsibilities); simulation 35% / 24% | Treat the gate as a decision function; paired, cluster-adjusted non-inferiority; power; reliability (VIM, ICC); equivalence (TOST) | S1–S3, S7 | Seeds counted as independent scenes; aggregates hiding strata; a relative effect quoted without its base rate |
| **P6** Deployment and real time | C++ 67% / 58%; CUDA 40% / 13%; real-time 35% / 18%; quantization / TensorRT 7% / 16% | Roofline on Thor (273 GB/s); the CoC token budget; FP8/NVFP4 formats; real TensorRT engines; closed-loop precision verdicts; deterministic C++ | S6, S7 | Assuming "4-bit is 4× faster"; batch-1 eager latency called deployable; reading emulated FP8 as real; Jetson numbers quoted as DRIVE numbers |
| **P7** Safety and production | Safety / V&V / standards 19% / 19% | Allowed-claims writing; release notes; what ISO/PAS 8800, SOTIF, UL 4600 and UN R[185] ask of ML validation; a no-actuation rig; privacy of road recordings | S3 (release note), S8 | Claiming "safe" from simulation; calling shadow mode "validation" |
| **P8** Training systems and cluster operations (new in v3.1) | Distributed training 14% / 18%; Kubernetes / SLURM / Ray 9% / 12% | Model a closed-loop RL loop as two balances (driver vs renderer, rollout vs trainer) and bound each lever with Amdahl. Measure with NVIDIA's own definitions: stabilized seconds per rollout, GPU-hours from the mean wall clock, MFU/HFU/OFU, stall. Run Slurm with Enroot/Pyxis array jobs, requeue and resume. Read DCGM. Debug an NCCL hang with the Flight Recorder. Report strong and weak scaling honestly | S2, S4, S5 | GPU-hours from the median, or from the congested first minutes; scaling efficiency without warm-up or device sync; one node presented as a cluster; a speed-up that changes numerics reported without an equivalence verdict |

**Ladder rule.** Every pillar descends L5 user-visible behaviour → L4 system → L3 algorithm/math → L2 kernel, memory, precision → L1 silicon. A pillar is mastered only when you can go down to L2 and back up with numbers.

## 3. Ownership and the learning method

**★ Huy writes, from blank:** the measurement instruments, the statistics, the RL and distillation loss math, the verifier logic, every prediction, the tolerances and every causal diagnosis. **Agents write** plumbing, harnesses, scaffolds, failing tests, sweeps, plots and reviews. `02` §3 marks the ★ cores per stage.

**Order inside every rep:** concept and derivation first; then Huy's written prediction; then measurement; then reconciliation. A prediction comes after the mechanism has been derived. A prediction written before that is a guess and is logged as one.

**Why this split** (verified learning evidence):
- In a randomized Anthropic study, AI assistance cut quiz mastery by 17% (d = 0.738, p = 0.010) with no significant speed gain. The cognitively engaged interaction patterns preserved learning (Shen & Tamkin, arXiv 2601.20245).
- In a high-school field study, unguarded GPT access made students 17% worse once access was removed (Bastani et al. 2025, PNAS).
- Retrieval practice beats restudy on delayed tests (Roediger & Karpicke 2006).

## 4. Curriculum by stage

Every exercise has a known answer that can be checked on a CPU. Answers carried over from v2 were verified in the 2026-10-03 research notes. New answers were computed on 2026-10-25 (`research/2026-10-25-teacher-to-thor/notes/arith_check.py` and the review) and on 2026-10-27 (`research/2026-10-27-role-alignment/notes/arith_check.py`; marked **v3.1** in the table). Do the exercise *before* the stage's build work.

| Stage | Primary sources | Derivation to own | Exercise (known answer) | Trap |
|---|---|---|---|---|
| **S1** Gate core | McNemar 1947; Clopper & Pearson 1934; Hanley & Lippman-Hand 1983; Efron 1979; Cameron, Gelbach & Miller 2008; Blackwelder 1982; Tango 1998; Berger 1982; Durkalski 2003; Connor 1987; O'Brien & Fleming 1979; Pocock 1977; Lan & DeMets 1983; Jennison & Turnbull 2000 | Exact McNemar null; zero-failure bound (1 − p)ⁿ = α ⇒ p ≈ 3/n; Connor's paired sample size; alpha spending across interim looks; intersection–union power = the product of per-stratum powers | b = 12, c = 6 ⇒ Σ_{k≤6} C(18,k) = 31,180; one-sided p = 0.11894, two-sided 0.23788. 0/441 ⇒ 0.677% one-sided, 0.833% two-sided. A 20% relative cut of an 8% at-fault rate (1.6 pp) at ψ = 4% needs **964** paired scenes (one-sided, 80% power). **v3.1:** K = 2 equally spaced looks, one-sided α = 0.025 ⇒ O'Brien–Fleming **2.7965 / 1.9774**, Pocock **2.1783** at both (α = 0.05: 2.3730 / 1.6780; Pocock 1.8754). Four strata at 0.80 each ⇒ **0.4096** overall; 0.80 overall needs **0.9457** per stratum | Uncorrected χ² gives 0.157; an unpaired z-test gives 0.399; calling 3/n a two-sided 95% CI; quoting a relative effect without its base rate; peeking at a sweep without a spending function; reading per-stratum power as overall power |
| **S2** Reproduce + reliability | JCGM 200:2012 (VIM); Shrout & Fleiss 1979; Koo & Li 2016; NHTSA pre-crash typology DOT HS 810 767; AlpaSim TELEMETRY.md, OPERATIONS.md and its `optimize-slurm-topology` skill; NuRec scaling page; Law & Kelton (common random numbers); Amdahl 1967; recipes PR #44 (measurement) | Var(mean) = σ²/(nm)·[1 + (m − 1)ρ]; sweep cost = scenes × replicates × steps × seconds per step; σ_d² = σ_A² + σ_B² − 2ρσ_Aσ_B, which is what pairing by seed can buy, if the driver consumes the seed; Amdahl S = 1/((1 − f) + f/k) | m = 3 replicates, ρ = 0.6 ⇒ design effect 2.2 (simulation gives 2.236). One sweep = 441 × 3 × 200 = 264,600 steps × 0.463–1.15 s ⇒ **34.0–84.5 GPU-h**. **v3.1:** σ_A = σ_B = 0.4 at ρ = 0 / 0.5 / 0.8 ⇒ σ_d = **0.5657 / 0.4000 / 0.2530** ⇒ one-sided 80% MDE at n = 441 of **6.70 / 4.74 / 3.00 pp**. A lever that speeds up f = 0.75 of the step by k = 2 gives at most **1.600×** | Counting replicates as scenes; calling unseeded replicates "seeds" (the `e2e_challenge` Alpamayo driver ignores the session seed); ranking card scores from different protocols (0.81 vs 1.37 with unchanged weights); mixing the card's km-between-events score with the challenge's [0, 1] scene score; reading NuRec's "100 FPS" as per-camera; throughput read in the congested first 10–15 minutes; GPU-hours from the median |
| **S3** Student verdict | GKD (arXiv 2306.13649); alpamayo-recipes PR #42 and #45; Bosch (arXiv 2605.00066); nuPlan (arXiv 2306.07962); TacitVLA (arXiv 2610.17540); Cochran, *Sampling Techniques* (replicates vs clusters) | Why open-loop minADE cannot certify a closed-loop non-inferiority margin; paired MDE for a continuous score, (z_α + z_β)·σ_d/√n, vs the binary McNemar MDE; the margin on a [0, 1] score vs on a rate; equal precision from fewer replicates on more scenes | A 10% margin on a teacher mean scene score of 0.7 is **7.0 pp**. At n = 441 (one-sided, 80% power) the continuous MDE is **3.55 pp at σ_d = 0.3** and **5.92 pp at σ_d = 0.5**, so the margin is decidable only if σ_d ≲ **0.59**. The card's 0.81 → 1.37 drift with unchanged weights is **6.9×** the margin. **v3.1:** m = 3 replicates at ρ = 0.6 ⇒ the same precision from **1.3636 n** single-replicate scenes, which is **54.5%** fewer rollouts | Applying the binary McNemar MDE (2.7 pp) to a continuous score; comparing a student's score with a card number from another run; "+6.95% minADE" read as a closed-loop verdict; reading TacitVLA's ± (an SD over three rollouts) as a confidence interval; pooling replicates as if they were scenes |
| **S4** Closed-loop RL on the teacher | PPO (arXiv 1707.06347); DeepSeekMath GRPO (arXiv 2402.03300); Schulman's KL note; Flow-GRPO (arXiv 2505.05470); π_RL (arXiv 2510.25889); AR1 Eq. 10; AlpaGym `default.yaml` and `replay_objective.py` @ `972d160`; cosmos-rl `grpo.py` | Group-normalized advantage and its invariance to any positive affine change of the reward. **The estimator as AlpaGym implements it (corrected in v3.1):** a per-step-row geometric-mean ratio, exp of the mean over T × 64 × 2 per-element log-ratios (each clamped to ±5), clipped at [0.8, 1.2]. Its closed-form per-transition Gaussian KL, off by default. Compare these with k3, r − 1 − ln r with r = π_ref/π_θ (the open-loop recipe), and with AR1's unclipped softmax-weighted objective | Group rewards [1,0,0,1,1,0,0,0] ⇒ mean 0.375, std 0.48412, advantages **+1.29099 / −0.77460**. k3 at r = 0.8 ⇒ **0.02314**; inverting the ratio (r = 1.25) gives 0.02686. **v3.1:** a collision-only group [−10 × 3, 0 × 5] ⇒ advantages **−1.29099 / +0.77460**, identical with −1 penalties. The mean-ratio clip [0.8, 1.2] over 640 elements spans a joint log-ratio of **−142.8 to +116.7**. A Gaussian KL with Δμ = 0.1, σ = 0.5 ⇒ **0.0200** | The proxy reward improves while behaviour degrades (AR1: grade 3.1 → 4.5, consistency 0.62 → 0.53); a reasoning reward on a frozen VLM; a sign-flipped reward (recipes #40); **`n_generation = 1`**, where every advantage is 0 and the trainer skips the optimizer step; believing a bigger penalty weight changes a group in which only that event varies; deriving k3 and a trajectory ratio for AlpaGym |
| **S5** On-policy correction + CoC budget | AR1 §3.2.2 and Eq. 5; Ross & Bagnell 2010; DAgger (arXiv 1011.0686); Flow Matching (arXiv 2210.02747) and its Guide (arXiv 2412.06264); RoaD (arXiv 2512.01993); OPTED (arXiv 2609.20756); alpamayo-recipes PR #42 stage 2 and PR #45; FlashDrive (arXiv 2608.12932) | Exact heading update Δθ = κ(vΔT + aΔT²/2); BC's T²ε vs DAgger's uTε; the CFM marginal field; ODE→SDE drift | Constant arc v = 10 m/s, κ = 0.02 m⁻¹, T = 6.4 s ⇒ forward-Euler endpoint error **0.5972 m**; AR1 Eq. 5 gives **0.00199 m**. Alternate: 1-D Gaussian flow, 10 Euler steps ⇒ std 0.43078, relative MSE 1.92%. **v3.1, the ladder:** the same flow with uniform Euler steps gives K = 5 ⇒ std **0.36923** (6.84%), K = 4 ⇒ **0.34054** (10.17%), K = 2 ⇒ **0.20000** (36.00%) | Offline KD only (the student never sees its own states); a forward-Euler decoder (0.60 m systematic error); computing KV with 32 heads instead of 8 KV heads; comparing K arms without a shared noise file; a pilot PR #42 student read as NVIDIA's student; OPTED's rise in per-scene collisions read as "more collisions per km" (its km per infraction improved) |
| **S6** Jetson AGX Thor serving | Roofline (Williams et al. 2009); PMPP 5th ed. (2026); FP8 formats (arXiv 2209.05433); MX formats (arXiv 2310.10537); NVIDIA NVFP4 blog; SmoothQuant (arXiv 2211.10438); FlashDrive (arXiv 2608.12932); FlashRT PR #229; Jetson Thor datasheet; TensorRT 10.16 support matrix | B* = b·F / (2·BW); decode time = bytes read / BW; NVFP4's effective bits are 4 + 8/16 = 4.5 | Alpamayo 1.5 BF16 ⇒ 15.17 GB / 273 GB/s = **55.6 ms/token** on Jetson Thor. A 2B NVFP4 student ⇒ **4.12 ms/token**, so (100 − 49.5 − 7.3) / 4.12 ≈ **10.5 CoC tokens** fit in 100 ms at the bound. NVFP4 saves 16/4.5 = 3.556× vs FP16; E4M3 max = (1 + 6/8)·2⁸ = 448 | "4-bit is 4× faster"; reading fake-quantized results as speed-ups; quoting a roofline floor as a latency estimate (FlashDrive's expert ran at ~20% of its bound); "E4M3 tops out at 240" |
| **S7** HIL + latency injection | AlpaSim driver gRPC proto and runtime docs (external-driver mode; `delay_buffer.py`); Ross & Bagnell 2010; TOST equivalence (Schuirmann 1987) | Action delay in ticks = ⌈latency / Δt⌉; distance travelled during the latency = v·t | At 10 Hz, 139 ms of on-device latency ⇒ the action lands **2 ticks** late; at 15 m/s the car travels **2.085 m** meanwhile. **v3.1:** 139, 171 and 200 ms all land **2 ticks** late (⌈d/Δt⌉ = 2), while the car moves **2.085 / 2.565 / 3.000 m**. This assumes the delay buffer is read at control ticks; check it | Lockstep HIL scores latency as zero; testing equivalence with a significance test instead of TOST; assuming more latency always costs more ticks |
| **S8** Shadow mode (optional) | Tesla trigger-classifier patents (via the course below, optional); nuPlan (arXiv 2306.07962); Bosch (arXiv 2605.00066); pinhole camera model; Vietnam Law 91/2025/QH15 Art. 32 (recording in public places) | f = (W/2) / tan(FOV/2); Bayes for trigger precision | At W = 576 px: 120° ⇒ f = **166.28 px**; 30° ⇒ **1,074.83 px**. A trigger with TPR 0.9 and FPR 0.05 on a 1% base rate has precision **15.4%** | Treating shadow-mode disagreement as closed-loop validation; confounding scene shift with rig shift; publishing un-anonymized faces and plates |

**Deferred curricula.** These belong to deferred work and stay ready if their stage returns through a `DECISIONS.md` entry. The full rows are in `archive/03_MASTERY_PLAN_v2_2026-10-03.md` §4.

| Deferred work | Known answer kept |
|---|---|
| Validity envelope and agreement (v3 S3) | NeRF/3DGS compositing: σ = [1, 2], δ = [0.5, 0.5] ⇒ α = [0.3935, 0.6321], opacity 0.7769 |
| Cross-paradigm construct validity (v3 S5) | Gate [1..5] vs reference [2,1,3,5,4] ⇒ ρ = 0.8; P(ρ ≥ 0.8 \| H0, n = 5) = 8/120; at n = 10, CI ≈ [0.24, 0.96] |
| Guardrail (v3 S6) | RSS: v_r = v_f = 20 m/s, ρ = 0.5 s, a_accel = 3, a_min,brake = 4, a_max,brake = 8 m/s² ⇒ **d_min = 43.16 m** |

**Free lecture series** (latest editions verified in the curriculum notes): Tübingen Self-Driving Cars (Geiger) for P1/P4; Berkeley CS285 for P3; Stanford CS231n and CS336 for P2. Use them to support a stage, never as a separate track.

**Optional course supplement: Think Autonomous, *End-to-End Revolution*.** It teaches the 2023–2025 end-to-end canon. It does not cover VLAs with reasoning, RL post-training, 3DGS closed loop or edge serving, so it is a supplement, never the spine (decision input, 2026-10-25). Module names were verified 2026-10-25.

| Course item | Use | Stage |
|---|---|---|
| Tesla Patent Study Vault (trigger classifiers) | Template for the shadow-mode disagreement miner | S8 |
| Simulator lectures | Closed- vs open-loop mental model | S2 |
| TransFuser deep dive | AlpaSim ships Latent TransFuser v6 as its (provisional) baseline driver, the S2 sanity check | S2 |
| "Inside the big lie of End-To-End Learning" | Interpretability and auxiliary tasks, which map to reasoning–action consistency | S5 |
| CARLA projects, quizzes | Skip | — |

## 5. Mastery gates: a stage closes only when all three pass, in order

1. **Teach-back.** Explain the stage's mechanism from memory with one hand-worked numeric example, for example the McNemar 12-vs-6 case or the CoC token budget on Thor.
2. **Transfer question.** Apply the mechanism to an unfamiliar shape or failure mode. Examples:
   - a stratum with 5 scenes;
   - an FP8 student whose score rises while 18% of its rollouts leave the envelope;
   - a 2B model announced at 10 Hz on hardware where its sibling measures 171 ms.
3. **Evidence gate.** The artifact exists and reproduces from a fresh clone, and its prediction-vs-measurement reconciliation is written, with misses explained causally.

**Owner proof for ★ cores.** Either a cold reconstruction of a representative core from blank, or a diagnosis of a deliberately planted defect. Either one establishes ownership without redoing every delegated line. A failed owner proof means redoing the core. No stage is closed by a feeling of understanding.

## 6. Skill coverage and honest gaps

● = primary evidence, ◐ = partial. Stages follow plan v4.1.

| Cluster | NVIDIA / comp. | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|---|---|---|---|---|---|---|---|---|---|
| C++ | 67% / 58% | | | | | | ● | ◐ | ◐ |
| Python | 65% / 64% | ● | ● | ● | ● | ● | ◐ | ◐ | ◐ |
| VLA / VLM / LLM | 49% / 27% | | | ◐ | ◐ | ● | ◐ | | |
| CUDA / kernels | 40% / 13% | | | | | | ● | | |
| Simulation | 35% / 24% | | ● | ● | ● | ● | | ● | |
| Real-time / embedded | 35% / 18% | | | | | | ● | ● | ● |
| Publications / reports | 33% / 24% | | | ● | ● | ◐ | ◐ | | |
| Motion planning | 30% / 29% | | | | ◐ | ◐ | | ◐ | ◐ |
| Evaluation / metrics | 28% / 39% | ● | ● | ● | ● | ● | ● | ● | ◐ |
| RL / post-training | 23% / 16% | | | | ● | ● | | | |
| Imitation / E2E | 23% / 7% | | | ◐ | | ● | | | |
| Data pipelines | 23% / 17% | | ◐ | | | ◐ | | | ● |
| World models / neural recon | 21% / 13% | | ◐ | | ◐ | ◐ | | | |
| Safety / V&V | 19% / 19% | ◐ | ◐ | ◐ | ◐ | | | | ◐ |
| Open source | 14% / 4% | ● | ● | ◐ | ◐ | ◐ | ◐ | | |
| Distributed training | 14% / 18% | | | | ● (1→8 GPUs, one node) | ● (1→8 GPUs, one node) | | | |
| Kubernetes / SLURM / Ray | 9% / 12% | | ◐ (Slurm, [O]) | | ◐ (Slurm presets, one node) | ◐ (Slurm) | | | |
| GPU monitoring and reliability (JR2010863 D4) | not counted in the 10-03 sweep | | ● (DCGM, goodput) | | ◐ (Flight Recorder, NCCL RAS) | ● (resiliency drills) | | | |
| Quantization / TensorRT | 7% / 16% | | | ◐ | | | ● | | |

**Gaps, named honestly.** They stay open until a later stage is added to `02` through a `DECISIONS.md` entry.
- **C++ beyond the Thor runtime.** S6 covers C++/CUDA on the device. S7's production-grade C++ driver server (v3.1) is the second C++ artifact, but both are on the inference path. The deferred G4 monitor was the other C++ artifact.
- **Multi-node distributed training (14% / 18%).** One node is covered from v3.1: S4's AlpaGym balance and scaling model across 1→8 GPUs, and S5's DDP/FSDP2 ladder. A measured multi-node point exists only through P-012. PR #42-scale distillation is collaborator-tier.
- **Real-time work on actual Thor hardware.** Covered by S6–S7 only if P-002 is approved. Otherwise only roofline-scaled.
- **Kubernetes / SLURM.** Slurm on one node, through AlpaGym's shipped Slurm presets (S4, [C]) and AlpaSim's own Slurm run method (S2, [O]). `SLURM_ENROOT` and CUDA MPS exist on AlpaSim's `main` only. **Kubernetes: none, by design.** AlpaSim ships no Kubernetes run method, and NVIDIA had no short-term fix for its renderer under Kubernetes without Docker-in-Docker (#72).
- **Shipped production, industry infrastructure (JR2010863 X1) and years of experience.** Not addressable by a solo project. 13 of the 43 NVIDIA postings (30%) require 12+ years; none of the 212 competitor postings does. Postings treat public evidence as a complement, never as a substitute.
- **PPO with a learned critic.** Not exercised: AlpaGym runs GRPO.

**Beyond plan v4.1, not scheduled:**
- the deferred G3 legs (agreement, cross-paradigm). Minimal R2–R4 legs are now proposals P-013 and P-014;
- the G4 guardrail;
- the data-engine flywheel at scale;
- multi-node distillation at PR #42 scale (a pilot is P-017);
- OmniDreams as a second renderer tier.

Each enters `02` only through a new `DECISIONS.md` entry.

---

## Tóm tắt tiếng Việt

- **Vai trò của bản v3.** Đây chỉ là bản đồ kỹ năng và giáo trình; lịch, chi phí và kill check nằm duy nhất trong `02`. Số thứ tự stage S1–S8 theo plan v4.
- **Chuỗi stage mới:**
  - S1–S2: dựng thước đo (gate) và đo độ tin cậy;
  - S3: phán quyết closed-loop cho student 2B của chính NVIDIA;
  - S4: RL closed-loop cho teacher;
  - S5: sửa student on-policy, cộng với đường cong ngân sách token CoC;
  - S6: chạy trên Jetson AGX Thor;
  - S7: HIL có tiêm độ trễ;
  - S8: shadow mode ở TP.HCM (tùy chọn).
- **Bảy trụ cột P1–P7.** Mỗi trụ cột gắn với tần suất thực tế trong 43 tin tuyển dụng AV của NVIDIA và 212 tin của đối thủ. C++ (67%), VLA/VLM (49%), CUDA (40%) và evaluation (51% trong phần trách nhiệm) là các kỹ năng được đòi nhiều nhất.
- **Giáo trình theo stage.** Mỗi stage có nguồn gốc, phép dẫn xuất phải tự làm, một bài tập có đáp số kiểm tra được trên CPU, và cái bẫy hay gặp. Làm bài tập trước khi build.
- **Khóa Think Autonomous** chỉ là tài liệu bổ trợ, không phải xương sống.
- **Đóng một stage.** Cần qua ba cổng: teach-back có ví dụ số, câu hỏi chuyển giao, và bằng chứng tái lập từ clone mới. Phần ★ phải có owner proof (dựng lại từ trắng, hoặc chẩn đoán lỗi cài sẵn).
- **Khoảng trống còn mở.** Huấn luyện multi-node (chỉ có qua P-012), Kubernetes (AlpaSim không có đường chạy K8s), PPO có critic, kinh nghiệm hạ tầng trong công ty (X1), và phần cứng Thor thật nếu P-002 chưa được duyệt.
- **Bản v3.1 (2026-10-27, D-015):**
  - **P8.** Thêm trụ cột P8, "Training systems and cluster operations": coi vòng RL closed-loop như một hệ throughput, với hai phương trình cân bằng (driver/renderer, rollout/trainer) và cận Amdahl; đo bằng chính định nghĩa của NVIDIA (seconds per rollout đã ổn định, GPU-hours từ mean wall clock, MFU/HFU/OFU); Slurm/Enroot/Pyxis, DCGM, Flight Recorder.
  - **Sửa đạo hàm S4.** AlpaGym dùng tỉ số geometric-mean theo từng step row, cùng KL Gaussian dạng đóng, mặc định tắt; không phải k3 và tỉ số theo cả quỹ đạo.
  - **Đáp số kiểm tra mới, chạy được trên CPU:** biên group-sequential, công suất intersection–union, phân bổ replicate, σ_d dưới CRN, bất biến affine của advantage GRPO, thang K của flow, và lượng tử hóa độ trễ.
