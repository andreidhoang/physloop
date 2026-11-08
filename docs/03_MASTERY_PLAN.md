# 03 — Mastery plan v3.3 (7 November 2026)

**Status:** authoritative for curriculum and mastery. v3.3 (2026-11-07, on Huy's instruction) adds the lesson protocol (§8) and the project learning map (§9), and updates the order inside every rep (§3). It changes how topics are taught; it does not change ownership, stage admission, scientific prediction slots or the file guard.

v3.2 follows plan v5/D-021: prioritize S2 measurement understanding and S4 learning-algorithm research; S3/S5–S8 curricula are deferred reference. The new focused reading/derivation contract is §7. This changes emphasis, not Huy's scientific ownership or the file guard.

Historical v3.1 additions under D-015:
- pillar P8 is added;
- the S4 derivation is corrected;
- the S1–S5 and S7 exercises gain known answers;
- §6 is updated.

v2 is archived. Stage identifiers are retained under plan v5 (`02_PHYSLOOP_PLAN.md`); only S1/S2/S4 are active. It contains **no schedule**: stages S1–S8, their order, costs and kill checks live only in `02`. The evidence base is `research/2026-10-03-verification/`, `research/2026-10-25-teacher-to-thor/` and `research/2026-10-27-role-alignment/`.

**Target.** Deep capability in AV policy learning and rigorous empirical research. The posting map in `05_ROLE_MAPPING.md` is dated reference material, not a fresh job-market verification or a requirement to restore deferred work. The prior simulation-engineering target was JR2010863 (Senior ML and Simulation Engineer – AV, AV Simulation team), and its line-by-line evidence contract is `05` §3. JR2005596 and JR2010863 were re-confirmed live on 2026-10-25, 2026-10-27 and 2026-11-07. Credential lines (years, PhD) are ignored by design. Only requirements relevant to the admitted study become active learning commitments; any public artifact remains subject to KC-L/KC-L2.

**"Principal", stripped of tenure:**
- own one link of the loop end to end;
- decide with measurements;
- write the design doc others build on;
- ship code that survives review by the people who wrote the stack.

Evidence must be checkable. Publish code, synthetic tests and permitted reports where allowed; licensed results and weights remain gated by KC-L/KC-L2. Private licensed evidence does not automatically become a public portfolio artifact.

---

## 1. Read the role as a system

```
 [3] Data engine ──► [1][2] Train VLA (IL → SFT → closed-loop correction → RL) ──► [4] Evaluate & release ──► [5] Runtime on Thor
        ▲                                                                              │
        └──────────────────────── failures found in [4] and on the road ───────────────┘
```

This is NVIDIA's "AI 2.0" framing: a reasoning VLA replaces hand-built modules and is distilled to run on DRIVE AGX Thor next to a classical safety stack. The earlier broad PhysLoop map below is historical reference; the active learning route is §7:

| Link | Stages (historical v4.1) |
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
| **P3** Learning algorithms | RL 23% / 16%; imitation / E2E 23% / 7% | DAgger, flow matching (CFM, Euler sampler, ODE→SDE for log-probs), GRPO with reward design and KL control, distillation (GKD), RoaD / OPTED closed-loop correction | S4, S5 | The proxy reward improves while behaviour degrades (AR1's reasoning-only row); a group-constant reasoning score mistaken for an expert learning signal; rollout and trainer log-prob mismatch |
| **P4** Data engine | Data 23% / 17%; world models / neural recon 21% / 13% | Location-grouped splits; teacher contamination (`alpamayo_aligned`); auto-labeling CoC with a teacher; 3DGS rendering limits; trigger classifiers | S2, S5, S8 | Frame-level random splits: with k = 20 frames per location and f = 0.2, 98.85% of locations land in both splits |
| **P5** Simulation and evaluation | Evaluation 28% / 39% (51% / 57% of responsibilities); simulation 35% / 24% | Treat the gate as a decision function; paired, cluster-adjusted non-inferiority; power; reliability (VIM, ICC); equivalence (TOST) | S1–S3, S7 | Seeds counted as independent scenes; aggregates hiding strata; a relative effect quoted without its base rate |
| **P6** Deployment and real time | C++ 67% / 58%; CUDA 40% / 13%; real-time 35% / 18%; quantization / TensorRT 7% / 16% | Roofline on Thor (273 GB/s); the CoC token budget; FP8/NVFP4 formats; real TensorRT engines; closed-loop precision verdicts; deterministic C++ | S6, S7 | Assuming "4-bit is 4× faster"; batch-1 eager latency called deployable; reading emulated FP8 as real; Jetson numbers quoted as DRIVE numbers |
| **P7** Safety and production | Safety / V&V / standards 19% / 19% | Allowed-claims writing; release notes; what ISO/PAS 8800, SOTIF, UL 4600 and UN R[185] ask of ML validation; a no-actuation rig; privacy of road recordings | S3 (release note), S8 | Claiming "safe" from simulation; calling shadow mode "validation" |
| **P8** Training systems and cluster operations (new in v3.1) | Distributed training 14% / 18%; Kubernetes / SLURM / Ray 9% / 12% | Model a closed-loop RL loop as two balances (driver vs renderer, rollout vs trainer) and bound each lever with Amdahl. Measure with NVIDIA's own definitions: stabilized seconds per rollout, GPU-hours from the mean wall clock, MFU/HFU/OFU, stall. Run Slurm with Enroot/Pyxis array jobs, requeue and resume. Read DCGM. Debug an NCCL hang with the Flight Recorder. Report strong and weak scaling honestly | S2, S4, S5 | GPU-hours from the median, or from the congested first minutes; scaling efficiency without warm-up or device sync; one node presented as a cluster; a speed-up that changes numerics reported without an equivalence verdict |

**Ladder rule.** Every pillar descends L5 user-visible behaviour → L4 system → L3 algorithm/math → L2 kernel, memory, precision → L1 silicon. A pillar is mastered only when you can go down to L2 and back up with numbers.

## 3. Ownership and the learning method

**Delivery ownership updated by D-018/P-030:** agents implement evaluator statistics, reliability, validity, verifier logic and supporting engineering with independent review and known-answer tests. Huy retains scientific definitions, predictions, tolerances and interpretation, plus model-research loss/reward/estimator, latency-model and kernel design. Listed file guards remain binding. Writing or deriving from blank remains a personal mastery exercise; agent implementation does not satisfy it, and it is not a blanket prohibition on delegated evaluator delivery. `02` §3 and D-018 define the current allocation.

**Order inside every rep (v3.3):** plan first, then go deep. Place the topic on the project map (§9), then teach it in the lesson order of §8: goal → derivation from first principles → code, shown or written into its correct project file → an end-to-end ASCII trace of every tensor element, rendered by a script from the real code → tests → production view → Huy's part. For a scientific experiment the lesson comes first, then Huy's written prediction, then measurement, then reconciliation. A prediction comes after the mechanism has been derived. A prediction written before that is a guess and is logged as one. A lesson trace on tiny synthetic inputs is a reading aid: it is not a measurement and never fills an experiment's prediction slot.

**Why this split** (verified learning evidence):
- In a randomized Anthropic study, AI assistance cut quiz mastery by 17% (d = 0.738, p = 0.010) with no significant speed gain. The cognitively engaged interaction patterns preserved learning (Shen & Tamkin, arXiv 2601.20245).
- In a high-school field study, unguarded GPT access made students 17% worse once access was removed (Bastani et al. 2025, PNAS).
- Retrieval practice beats restudy on delayed tests (Roediger & Karpicke 2006).

## 4. Curriculum by stage

Every exercise has a known answer that can be checked on a CPU. Answers carried over from v2 were verified in the 2026-10-03 research notes. New answers were computed on 2026-10-25 (`research/2026-10-25-teacher-to-thor/notes/arith_check.py` and the review) and on 2026-10-27 (`research/2026-10-27-role-alignment/notes/arith_check.py`; marked **v3.1** in the table). Do the exercise *before* the stage's build work.

| Stage | Primary sources | Derivation to own | Exercise (known answer) | Trap |
|---|---|---|---|---|
| **S1** Gate core | McNemar 1947; Clopper & Pearson 1934; Hanley & Lippman-Hand 1983; Efron 1979; Cameron, Gelbach & Miller 2008; Blackwelder 1982; Tango 1998; Berger 1982; Durkalski 2003; Connor 1987; O'Brien & Fleming 1979; Pocock 1977; Lan & DeMets 1983; Jennison & Turnbull 2000 | Exact McNemar null; zero-failure bound (1 − p)ⁿ = α ⇒ p ≈ 3/n; Connor's paired sample size; alpha spending across interim looks; intersection–union power = the product of per-stratum powers only under independence | b = 12, c = 6 ⇒ Σ_{k≤6} C(18,k) = 31,180; one-sided p = 0.11894, two-sided 0.23788. 0/441 ⇒ 0.677% one-sided, 0.833% two-sided. A 20% relative cut of an 8% at-fault rate (1.6 pp) at ψ = 4% needs **964** paired scenes (one-sided, 80% power). **v3.1:** K = 2 equally spaced looks, one-sided α = 0.025 ⇒ O'Brien–Fleming **2.7965 / 1.9774**, Pocock **2.1783** at both (α = 0.05: 2.3730 / 1.6780; Pocock 1.8754). Four independent strata at 0.80 each ⇒ **0.4096** overall; 0.80 overall needs **0.9457** per stratum | Uncorrected χ² gives 0.157; an unpaired z-test gives 0.399; calling 3/n a two-sided 95% CI; quoting a relative effect without its base rate; peeking at a sweep without a spending function; reading per-stratum power as overall power |
| **S2** Reproduce + reliability | JCGM 200:2012 (VIM); Shrout & Fleiss 1979; Koo & Li 2016; NHTSA pre-crash typology DOT HS 810 767; AlpaSim TELEMETRY.md, OPERATIONS.md and its `optimize-slurm-topology` skill; NuRec scaling page; Law & Kelton (common random numbers); Amdahl 1967; recipes PR #44 (measurement) | Var(mean) = σ²/(nm)·[1 + (m − 1)ρ]; sweep cost = scenes × replicates × steps × seconds per step; σ_d² = σ_A² + σ_B² − 2ρσ_Aσ_B, which is what pairing by seed can buy, if the driver consumes the seed; Amdahl S = 1/((1 − f) + f/k) | m = 3 replicates, ρ = 0.6 ⇒ design effect 2.2 (simulation gives 2.236). One sweep = 441 × 3 × 200 = 264,600 steps × 0.463–1.15 s ⇒ **34.0–84.5 GPU-h**. **v3.1:** σ_A = σ_B = 0.4 at ρ = 0 / 0.5 / 0.8 ⇒ σ_d = **0.5657 / 0.4000 / 0.2530** ⇒ one-sided 80% MDE at n = 441 of **6.70 / 4.74 / 3.00 pp**. A lever that speeds up f = 0.75 of the step by k = 2 gives at most **1.600×** | Counting replicates as scenes; calling unseeded replicates "seeds" (the `e2e_challenge` Alpamayo driver ignores the session seed); ranking card scores from different protocols (0.81 vs 1.37 with unchanged weights); mixing the card's km-between-events score with the challenge's [0, 1] scene score; reading NuRec's "100 FPS" as per-camera; throughput read in the congested first 10–15 minutes; GPU-hours from the median |
| **S3** Student verdict | GKD (arXiv 2306.13649); alpamayo-recipes PR #42 and #45; Bosch (arXiv 2605.00066); nuPlan (arXiv 2306.07962); TacitVLA (arXiv 2610.17540); Cochran, *Sampling Techniques* (replicates vs clusters) | Why open-loop minADE cannot certify a closed-loop non-inferiority margin; paired MDE for a continuous score, (z_α + z_β)·σ_d/√n, vs the binary McNemar MDE; the margin on a [0, 1] score vs on a rate; equal precision from fewer replicates on more scenes | A 10% margin on a teacher mean scene score of 0.7 is **7.0 pp**. At n = 441 (one-sided, 80% power) the continuous MDE is **3.55 pp at σ_d = 0.3** and **5.92 pp at σ_d = 0.5**, so the margin is decidable only if σ_d ≲ **0.59**. The card's 0.81 → 1.37 drift with unchanged weights is **6.9×** the margin. **v3.1:** m = 3 replicates at ρ = 0.6 ⇒ the same precision from **1.3636 n** single-replicate scenes, which is **54.5%** fewer rollouts | Applying the binary McNemar MDE (2.7 pp) to a continuous score; comparing a student's score with a card number from another run; "+6.95% minADE" read as a closed-loop verdict; reading TacitVLA's ± (an SD over three rollouts) as a confidence interval; pooling replicates as if they were scenes |
| **S4** Closed-loop RL on the teacher | PPO (arXiv 1707.06347); DeepSeekMath GRPO (arXiv 2402.03300); Schulman's KL note; Flow-GRPO (arXiv 2505.05470); π_RL (arXiv 2510.25889); AR1 Eq. 10; AlpaGym `default.yaml` and `replay_objective.py` @ `972d160`; cosmos-rl `grpo.py` | Group-normalized advantage: exact translation invariance and approximate positive-scale invariance when epsilon is negligible relative to both original and rescaled group standard deviation. **The estimator as AlpaGym implements it (corrected in v3.1):** a per-step-row geometric-mean ratio, exp of the mean log-density difference over T × 64 × 2 coordinates, with that mean clamped to ±5, clipped at [0.8, 1.2]. Its closed-form per-transition Gaussian KL, off by default. Compare these with k3, r − 1 − ln r with r = π_ref/π_θ (the open-loop recipe), and with AR1's unclipped softmax-weighted objective | Group rewards [1,0,0,1,1,0,0,0] ⇒ mean 0.375, std 0.48412, advantages **+1.29099 / −0.77460**. k3 at r = 0.8 ⇒ **0.02314**; inverting the ratio (r = 1.25) gives 0.02686. **v3.1:** a collision-only group [−10 × 3, 0 × 5] ⇒ advantages **−1.29099 / +0.77460**, equal in the epsilon-free exercise and approximately equal in the pinned implementation with −1 penalties. A mean ratio lying in [0.8, 1.2] over 640 factors is consistent with joint log-ratios of **−142.8 to +116.7**. A Gaussian KL with Δμ = 0.1, σ = 0.5 ⇒ **0.0200** | The proxy reward improves while behaviour degrades (AR1: grade 3.1 → 4.5, consistency 0.62 → 0.53); a group-constant reasoning score mistaken for an expert learning signal; a sign-flipped reward (recipes #40); **`n_generation = 1`**, where centered reward advantages are zero and the pinned stock KL-off/no-auxiliary-gradient path skips optimizer.step (the scheduler can still advance); expecting a larger lone penalty to materially change standardized advantages when epsilon is negligible; deriving k3 and a trajectory ratio for AlpaGym |
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

● = proposed primary evidence, ◐ = partial in the historical v4.1 map. This is not a completion record or an active commitment to deferred stages/scaling.

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
- **Multi-node distributed training (14% / 18%).** One node is covered from v3.1: S4's AlpaGym balance and scaling model across 1→8 GPUs, and S5's DDP/FSDP2 ladder. These broad ladders are deferred by D-021. A measured multi-node point would require a separately admitted P-012 study. PR #42-scale distillation is collaborator-tier.
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
- **Bản v3.3 (2026-11-07, theo yêu cầu của Huy):**
  - **Lập bản đồ trước, đào sâu sau.** §9 là bản đồ học toàn dự án theo đúng dòng dữ liệu: Bài 0 đi một cảnh qua mọi ô, rồi Phần I (S1, quyết định), Phần II (S2, đo), Phần III (S4, học, theo thứ tự học của `02` §4). Mỗi bài mở đầu bằng việc đánh dấu ô của nó trên sơ đồ PIPELINE.
  - **Bảy phần cho mỗi bài (§8):** mục tiêu → toán dẫn từ gốc → code (đọc code có sẵn, hoặc viết vào đúng file của dự án) → trace ASCII từ đầu vào tới đầu ra, in đủ mọi phần tử tensor, do script chạy code thật → test với oracle độc lập → góc nhìn production → phần của em.
  - **Ai viết code** tùy chế độ của từng dòng: R (code repo), R★ (file khóa, chỉ đọc/test), U (code upstream ở SHA ghim, chỉ đọc), A (agent viết, test trước), H (Huy viết từ trang trắng), S (script đồ chơi trong `study/`). S4 chưa được admit nên các bài phần III chỉ đọc code upstream.
  - **Script bài học** nằm trong `study/` (git-ignore), chỉ dùng dữ liệu tổng hợp nhỏ, không bao giờ là evidence. Một dự đoán trong bài học là bài tập, không điền vào ô dự đoán khoa học.
- **Bản v3.1 (2026-10-27, D-015):**
  - **P8.** Thêm trụ cột P8, "Training systems and cluster operations": coi vòng RL closed-loop như một hệ throughput, với hai phương trình cân bằng (driver/renderer, rollout/trainer) và cận Amdahl; đo bằng chính định nghĩa của NVIDIA (seconds per rollout đã ổn định, GPU-hours từ mean wall clock, MFU/HFU/OFU); Slurm/Enroot/Pyxis, DCGM, Flight Recorder.
  - **Sửa đạo hàm S4.** AlpaGym dùng tỉ số geometric-mean theo từng step row, cùng KL Gaussian dạng đóng, mặc định tắt; không phải k3 và tỉ số theo cả quỹ đạo.
  - **Đáp số kiểm tra mới, chạy được trên CPU:** biên group-sequential, công suất intersection–union, phân bổ replicate, σ_d dưới CRN, bất biến affine của advantage GRPO, thang K của flow, và lượng tử hóa độ trễ.

## 7. Focused RL curriculum under plan v5

The active emphasis is P3 learning algorithms, supported by P1 decision processes, P2 model representation, P4 data separation and P5 evaluation. P8 supplies sufficient reliable operations for the study; a large cluster program is not required. P6/device work is deferred.

Read [the foundations ledger](research/2026-11-01-rl-direction-review/rl-foundations.md) and [the AV ledger](research/2026-11-01-rl-direction-review/av-literature.md) for exact paper versions, sections and transfer limits. Prioritize these deliverables:

| Learning unit | Principal readings | Huy's evidence of understanding |
|---|---|---|
| Policy-visited states | DAgger, LBC, Roach, World on Rails, OPTED | Explain direct policy-gradient RL versus supervised correction, privileged inputs and replay assumptions |
| Actual update | Policy-gradient theorem, GAE, TRPO/PPO, DeepSeekMath | Derive advantage/sign/mask behavior and explain why clipping is not a hard guarantee |
| Generative action distribution | DPPO, Flow-GRPO, ReinFlow, pinned Gym implementation | Separate driving time from denoising time and joint path from terminal marginal/mean-ratio surrogate |
| Fine-tuning a pretrained policy | RAD, Wołczyk et al., Dr. GRPO, pinned Gym advantage ([gap review](research/2026-11-01-rl-direction-review/gap-review.md)) | Derive the group-of-two advantage, and state what bounds the policy's distance from the teacher on the stock path |
| Objective and failure | Reward shaping, reward overoptimization, CPO, AV ablations | Construct a counterexample for each proposed reward and a single-variable falsifier |
| Evidence | Deep RL That Matters, Rliable, S1 contract | Distinguish checkpoint and algorithm targets, independent training runs and roads, selection and confirmation |
| Experiment ownership | Actual source audit and development evidence | Write mechanism, numeric prediction, stopping/selection rule and interpretation criteria before dependent execution; interpret measured results afterward |

Known-answer examples and future acceptance tests are in [the protocol design](research/2026-11-01-rl-direction-review/protocol-design.md). The numeric examples are exercises, not measured driving results or agent-filled scientific predictions. Three training runs are a minimum diagnostic floor; they do not automatically give a calibrated algorithm-level interval. Fresh roads alone do not make exploratory training runs independent confirmation.

The three existing mastery checks remain: teach the mechanism back without reading, solve a transfer case, and identify checkable evidence. Agent delivery does not pass them on Huy's behalf. The 70/20/10 research/interpretation/measurement-review time allocation is a recommendation after S2, not an adopted weekly-hours estimate.

## 8. Lesson protocol (v3.3)

One topic per lesson, and the plan comes first. In Claude Code, the `first-principles-explain` skill runs a lesson on one topic and `code-walk` walks one existing file component by component. Both defer to this section where they differ. Codex and other agents follow this section directly.

**Plan first.** §9 lists every topic on the active path in data-flow order. A lesson opens by marking its row on the §9 pipeline. A topic that is not in §9 is added there first, as a one-row edit, before it is taught.

**Seven parts, always in this order.** Vietnamese first, with English technical terms unchanged; code, docstrings and commit messages in English.

| Part | Content | PhysLoop rule |
|---|---|---|
| 1 · Mục tiêu | What the topic is for and where it sits; a verified incident with its link, or "no verified incident"; what "done" means as a check that runs | The drawing is the §9 pipeline with this lesson's box marked |
| 2 · Toán từ gốc | Numbered base truths T1…Tk, then the derivation one step per line, each step tagged with the truth it uses. Every symbol gets its shape, dtype and units. State the assumptions, give a "5 tuổi" box, list the failure modes | Known answers come from §4 or from the lesson's own closed form, computed by a script, never typed |
| 3 · Code | Existing code: read in full in the session, quoted with `file:line`. New code: written into the file that the row's mode allows | The mode table below decides who writes |
| 4 · ASCII trace | A script runs the real code on tiny inputs and prints the whole path, from first input to final output and back along any gradient or return path. It opens with a PIPELINE block. Each step block names `file:line` and the formula, draws every tensor as a labelled grid with every element shown, marks selected entries with `[ ]`, writes out the arithmetic for one row, and prints the maximum absolute difference between code and hand formula. It closes with the same path at production shapes | Inputs are synthetic and tiny. Never licensed frames, scenes, scores or anything derived from them (KC-L) |
| 5 · Tests | Derived from part 2's invariants, checked against an independent oracle; report the command and raw counts | In mode A, the failing test is written before the code (contract step 5) |
| 6 · Production view | Real shapes, bytes and throughput for this project's configuration; what breaks first at 10× and 100× | Shapes come from pinned configs; estimates are labelled |
| 7 · Phần của em | Two review questions, a teach-back in at most three lines, one modify-and-predict with the exact command to run; then stop until Huy answers | A lesson prediction is an exercise, not a registered scientific prediction |

For non-tensor topics (receipts, schedules, byte formats), the "tensor" is the state: draw timelines, tables and byte grids with every field shown.

**Who writes the code: one mode per §9 row.**

| Mode | Meaning | What the agent does |
|---|---|---|
| R | Agent-authored repo code in an admitted stage | Read in full, trace, test. Changes follow D-018 and the contract |
| R★ | A listed starred core: `gate/verdict.py`, `gate/stats/` | Read, trace and test only. Changes are described to Huy, never written |
| U | Pinned upstream code: AlpaSim, AlpaGym, alpamayo1.5 | Read at the pinned SHA. The trace copies the exact lines with `SHA:path:line` and checks them against an independent oracle. No weights, datasets or rentals |
| A | New evaluator engineering inside an admitted stage | Failing test first, then the code, in the correct module, marked agent-authored |
| H | Huy's research cores (reward, loss, estimator design) or his owner-proof reconstruction (§5) | Derive on a worked neighbour, then give the interface and the known-answer test. Huy writes from blank; the agent traces his code afterwards |
| S | A concept with no project code, such as DAgger's error bound | A toy script in `study/` only |

Agent-written code and agent-run traces are delivery, not Huy's mastery evidence (§3, `02` §4). A lesson never admits a stage. S4 rows stay in modes U, S or H until S4 is admitted, and no `teacher_rl/` code is created before then.

**Where lesson code lives.** Trace scripts and their printed output go in `study/<phase>/<lesson>.py` and `.txt`. The folder is git-ignored, so lessons never enter tracked source or `evidence/`. Each script takes its inputs as CLI flags so Huy can rerun it with changed values, and vendors any helper it needs inside `study/` instead of importing from `~/.claude`. Lesson status is kept in `study/LEDGER.md`, not in this document.

**Format exemplar.** This was rendered by a script, with the expression reproduced from AlpaGym @ `972d160`; it is not a run of AlpaGym. The 1e-6 term moves the sixth decimal, which is why §4 calls the pinned value approximately equal to the epsilon-free one.

```
PIPELINE
  r[G] ─► mean, np.std (ddof=0) ─► A[G] = (r − μ)/(σ + 1e-6) ─► broadcast to every replay row of candidate g

STEP 1 · μ = Σr/G, σ = √(Σ(r−μ)²/G)
  μ = 3/8 = +0.375000
  σ² = (3·0.625² + 5·0.375²)/8 = (1.171875 + 0.703125)/8 = 0.234375   σ = 0.484123

STEP 2 · A = (r − μ)/(σ + 1e-6)        entries with r = 1 marked [ ]
  reward and advantage  shape [8]  float64
         g=0      g=1      g=2      g=3      g=4      g=5      g=6      g=7
  r   │ +1.0000  +0.0000  +0.0000  +1.0000  +1.0000  +0.0000  +0.0000  +0.0000 │
  A   │[+1.2910] -0.7746  -0.7746 [+1.2910][+1.2910] -0.7746  -0.7746  -0.7746 │
  row g=0: (1 − 0.375)/(0.484123 + 0.000001) = +1.290992
  max |code − hand(ε=0)| = 2.7e-06   (the 1e-6 shifts the 6th decimal)
```

**Advancing.** A lesson ends at part 7. Huy writes "tiếp" to move to the next row. §5's teach-back, transfer and evidence gates still close a stage; lessons feed them but do not pass them.

## 9. Project learning map (v3.3): plan first

The active path in data-flow order. Lesson 0 walks one scene through every box; Phases I–III then take each box apart. Phase III follows the learning order in `02` §4. An anchor marked "Step 0" is located and opened at the start of that lesson; this map names no upstream path that has not been opened. Pinned upstream revisions: AlpaSim `e2e_challenge` @ `e121e59` and `main` @ `affc2ea`, AlpaGym @ `972d160`, alpamayo1.5 @ `36aeb4c`.

```
PIPELINE · the active path, one scene end to end                  ⟨n⟩ = row of the table below

S2 · đo     registration ─► schedule ─► launch ─► AlpaSim run ─► ASL log, metrics ─► receipts ─► records
            ⟨II-2⟩                                ⟨II-1⟩         ⟨II-3, II-4⟩        ⟨II-5, II-6⟩
                                                     │ one tick, lockstep
                                                     ▼
            state ─► Alpamayo 1.5 ─► 64 waypoints ─► controller ─► next state ─► next tick
                     ⟨III-4, III-5⟩   ⟨III-6⟩         ⟨III-6⟩

S1 · quyết  records ─► road pairs ─► bounds per stratum ─► Holm flags ─► looks ─► GO / NO-GO / INSUFFICIENT
            ⟨I-1, I-2⟩               ⟨I-3, I-4⟩            ⟨I-5⟩         ⟨I-6⟩    ⟨I-7, I-8⟩
            roads and replicates needed: ⟨II-7⟩ reliability, ⟨II-8⟩ cost

S4 · học    G candidates ─► reward ─► group advantage ─► ratio, clip ─► loss ─► optimizer steps ─► weights
            ⟨III-1⟩         ⟨III-8⟩   ⟨III-3⟩            ⟨III-7⟩        ⟨III-2⟩
            each candidate is an AlpaSim rollout on the training path, with the same tick loop
            weights ─► export ─► S2 · đo ─► S1 · quyết            ⟨III-9⟩ two paths, ⟨III-10⟩ the verdict
```

| # | Topic and goal | Math to derive | Code anchor | Data traced | Mode | Known answer |
|---|---|---|---|---|---|---|
| 0 | **Toàn cảnh**: one scene through every box, before any detail | Units and shapes at each box; sweep cost = scenes × replicates × steps × seconds per step | This map; `01_SYSTEM_DESIGN.md` | One scene's objects at each box, shapes only | R + U | 441 × 3 × 200 = 264,600 steps ⇒ 34.0–84.5 GPU-h at 0.463–1.15 s/step |
| **I-1** | **Records and pairing**: what a paired comparison is made of | Paired difference per scene and replicate; why pairing helps: Var(B − A) = Var A + Var B − 2 Cov | `gate/contracts.py` (`RolloutRecord`, `pair_records`) | Record list → pairs keyed by road, scene, replicate | R | McNemar b = 12, c = 6 ⇒ one-sided p = 0.11894 |
| I-2 | **Synthetic truth**: a world where the right answer is known | Scene latent copied across replicates with a chosen probability; road dependence | `gate/synthetic.py` | Road × scene × replicate × arm outcome grid | R | `tests/test_synthetic.py` |
| I-3 | **Bound for a bounded mean**: how far the true mean can be from what we saw | Chernoff bound → KL inversion for a mean in [0, 1]; all-zero case UB = 1 − (α/2)^(1/R) | `gate/verdict.py` (`_kl`, `bounded_mean_bounds`) | Per-road values → upper and lower bounds | R★ | With α/2 = 0.025 and every road difference zero, a stratum still needs 36 / 51 / 72 / 183 roads for margins 0.10 / 0.07 / 0.05 / 0.02 |
| I-4 | **Paired road bounds and the verdict** | Split d into d⁺ and d⁻; equal-road estimand; GO requires UB(mean d⁺) − LB(mean d⁻) < margin in every stratum (intersection–union) | `gate/verdict.py` (`paired_road_bounds`, `verdict`) | Per-road d → per-stratum bounds → verdict | R★ | Four independent strata at 0.80 ⇒ 0.4096; 0.80 overall needs 0.9457 per stratum |
| I-5 | **Holm regression flags** | p = exp(−R·max(mean, 0)²/2); Holm step-down | `gate/verdict.py` (`holm_flags`; p-value at L187) | Per-stratum mean d → p-values → Holm-ordered flags across the stratum family | R★ | Built in the lesson from the closed form; must match the function |
| I-6 | **Finite registered looks** | Union-bound alpha spending across looks | `gate/sequential.py` | Look schedule → spent α → decision per look | R | Reference only, not the gate's method: K = 2 O'Brien–Fleming 2.7965 / 1.9774, Pocock 2.1783 |
| I-7 | **Power and INSUFFICIENT**: why noisy equivalent controls rarely reach GO | P(GO) under the operational method vs the asymptotic planning formula | `gate/power.py` | Effect × roads grid → P(GO) table | R | 1.6 pp at ψ = 4% needs 964 paired scenes (planning formula) |
| I-8 | **S1 end to end** | None new: the composition of I-1 to I-7 | `gate/api.py`, `gate/__main__.py` | Request JSON → admission → pairs → bounds → verdict | R | `tests/test_api.py`, `tests/test_known_answers.py` |
| **II-1** | **AlpaSim tick loop and seeds**: why compute time never enters the outcome, and which repeats are really seeded | Lockstep tick; seed derivation session_seed = random_seed + rollout_idx, inference_seed = session.seed + inference_count | AlpaSim `main` @ `affc2ea`: `engine.py` L210–212, driver `main.py` L713 | Seed table across replicates and inference calls | U | Replicates 0 and 1 share 21 of 22 inference seeds |
| II-2 | **Registration → schedule → launch** | Matrix size and cost; what a schedule can and cannot verify | `gate/s2_preflight.py`, `gate/s2_schedule.py`, `gate/s2_launch.py` | Registration JSON → job matrix → launch commands | R | `tests/test_s2_schedule.py`, `tests/test_s2_launch.py` |
| II-3 | **ASL bytes**: reading AlpaSim's log without trusting it | Protobuf wire format: base-128 varints, wire types, length-delimited fields | `gate/s2_asl.py` | Byte grid (hex) → fields → entries → poses | R | Varint 300 = `0xAC 0x02`; `tests/test_s2_asl.py` |
| II-4 | **Metrics and summaries**: missing is not zero | Per-rollout metric arrays → summary; denominators of failures | `gate/s2_extract.py`, `gate/alpasim_adapter.py` | Metric arrays per rollout → normalized summary rows | R | `tests/test_s2_extract.py`, `tests/test_alpasim_adapter.py` |
| II-5 | **Identity and integrity**: what a hash proves | Canonical JSON bytes → SHA-256; binds bytes, not origin | `gate/s2_identity.py` | Object → canonical bytes → digest | R | `tests/test_s2_identity.py` |
| II-6 | **Receipts: ledger and capture** | Monotonic event order; one completed first attempt per job | `gate/s2_ledger.py`, `gate/s2_capture.py` | Event timeline per job → reconciliation table | R | `tests/test_s2_ledger.py`, `tests/test_s2_capture.py` |
| II-7 | **Reliability and paired variance**: how many replicates and roads | Var(mean) = σ²/(nm)·[1 + (m − 1)ρ]; σ_d² = σ_A² + σ_B² − 2ρσ_Aσ_B; CRN helps only if the driver consumes the seed | No module yet; built after S2 admission with Huy's frozen analysis | Replicate × scene score grid → ICC, design effect, σ_d | A | Design effect 2.2 at m = 3, ρ = 0.6; σ_d = 0.5657 / 0.4000 / 0.2530 at ρ = 0 / 0.5 / 0.8 |
| II-8 | **Cost and pilot affordability** | Cost = scenes × replicates × steps × s/step × price; Amdahl S = 1/((1 − f) + f/k) | `gate/s2_pilot.py`, `gate/s2.py` | Usage ledger → affordability check | R | Amdahl f = 0.75, k = 2 ⇒ 1.600× at most |
| **III-1** | **State-distribution shift**: why learn on the policy's own rollouts | Behaviour cloning T²ε vs DAgger uTε; privileged information. Reading: DAgger, LBC, Roach, World on Rails, OPTED | Toy only | Tiny 1-D driving loop: visited states under the expert vs the learner | S | Computed in the lesson from the closed forms |
| III-2 | **Policy gradient → PPO → GAE** | ∇J = E[A·∇log π]; clipped surrogate; why clipping is not a hard guarantee; k3 estimator | Toy only | Two-step bandit: log-probs, ratios, clipped terms | S | k3 at r = 0.8 ⇒ 0.02314 |
| III-3 | **Group-relative advantage** | A = (r − μ)/(σ + 1e-6); translation and approximate scale invariance; G = 2 is sign-only | AlpaGym @ `972d160` (file located at Step 0) | Group reward vector → advantage → per-row broadcast | U | G = 8 [1,0,0,1,1,0,0,0] ⇒ +1.290992 / −0.774595 (ε = 0: +1.290994 / −0.774597); G = 2 with a 0.001 gap ⇒ ±0.998004; G = 6, one −10 ⇒ −2.236067 ≈ −√5 |
| III-4 | **Model input**: cameras → patches → tokens → prefill → KV cache | Tokens per image = (H/16)·(W/16)/4; attention and KV bytes | alpamayo1.5 @ `36aeb4c` (files located at Step 0) | 16 images → token grid → KV shapes | U | 320×576 ⇒ 180 tokens per image, 2,880 for 16; 320×512 ⇒ 2,560 |
| III-5 | **Flow expert and its two time axes** | Conditional flow matching field; Euler ODE vs SDE sampler; driving time vs denoising time | alpamayo1.5 @ `36aeb4c`, AlpaGym @ `972d160` (Step 0) | Noise → K denoising steps → 64 × 2 action grid | U | 1-D Gaussian flow: K = 10 ⇒ std 0.43078; K = 5 ⇒ 0.36923; K = 2 ⇒ 0.20000 |
| III-6 | **Waypoints → trajectory → controller** | Exact heading update Δθ = κ(vΔT + aΔT²/2) (AR1 Eq. 5); what the controller does with the plan | alpamayo1.5 and AlpaSim (Step 0) | (accel, curvature) × 64 → poses → controller command | U | Constant arc v = 10 m/s, κ = 0.02 m⁻¹, T = 6.4 s: forward Euler endpoint error 0.5972 m, Eq. 5 0.00199 m |
| III-7 | **The actual Gym estimator** | Per-row geometric-mean ratio, mean clamped to ±5, clip [0.8, 1.2]; optimizer steps per trainer step ≤ iterations × ⌈rows / mini_batch⌉; `kl_beta: 0` ⇒ logged KL is zero by construction | AlpaGym @ `972d160`: `replay_objective.py`, `default.yaml` | Log-density grid T × 64 × 2 → mean → ratio → clipped loss | U | [0.8, 1.2] over 640 factors ⇔ joint log-ratio −142.8 to +116.7; Gaussian KL at Δμ = 0.1, σ = 0.5 ⇒ 0.0200; up to 22 optimizer steps in the smoke |
| III-8 | **Reward and its exploitation** | `progress_safety` = P − 10·C_any − 5·O − 0.01·D (terms opened at Step 0); `collision_any` includes rear contact while offline at-fault is front ∨ lateral; one counterexample per reward | AlpaGym @ `972d160` (Step 0); any new reward arm in `teacher_rl/` only after S4 admission | Per-tick event arrays → reward → group order | U, then H | Collision-only group [−10 × 3, 0 × 5] ⇒ −1.29099 / +0.77460 |
| III-9 | **Two paths for one set of weights** (proposal P-032) | Six factors: image size, reasoning text, expert RoPE offsets, integrator, policy rate, controller | AlpaGym @ `972d160`, AlpaSim driver, alpamayo1.5 (Step 0) | Same weights, two input and sampler paths, side by side | U | 2,560 vs 2,880 tokens; horizontal scale 512/576 = 0.889 |
| III-10 | **Checkpoint → export → the verdict**: the loop closes on S1 | None new: I-8 with arm B a trained checkpoint; training-run variation reported separately | `gate/api.py`; upstream export (Step 0) | Toy A/B records → I-8 trace | R + U | I-8's known answers |

**Where to start.** Lesson 0, then Phase I and Phase II, which serve the current stage. Phase III lessons are read-only (modes U and S) and may run alongside, since S4 is not admitted.
