# 01 — System design (systems-thinking view)

This document is authoritative for the system map and the twelve core concepts. It is v3 (2026-10-25): v2 (2026-10-03) plus the teacher → student → Thor loop of plan v4 (D-012), corrected against `research/2026-10-25-teacher-to-thor/verification.md`. v1 and v2 are in `archive/`. Read this before any code.

## 0. The production problem

Late 2026: an OEM running NVIDIA DRIVE AV is preparing an OTA update. The new model is smaller and faster than the one it replaces. It was distilled from a large teacher and quantized for DRIVE AGX Thor. Before release, someone has to answer one question:

> Does the new model drive worse than the old one? If so, in which situations?

This is no longer hypothetical. On 2026-10-21 NVIDIA announced Alpamayo 2 Edge, a 2B model for DRIVE AGX Thor at 10 Hz. A day earlier, a public pull request put up a recipe that distills Alpamayo 1 / 1.5 into a ~2B student (alpamayo-recipes PR #42, unmerged). Both report open-loop error and latency only. Neither says how the student drives in closed loop, or where it is worse than its teacher.

The open stack (Alpamayo, AlpaSim, AlpaGym) cannot answer it credibly, for two reasons verified on 2026-10-03:

1. **There is no release gate.** AlpaSim aggregates descriptive per-rollout statistics. The 2026 challenge adds Drive-IRT, which ranks a panel of six or more policies. Nothing compares checkpoint B to checkpoint A with paired, stratified confidence.
2. **The instrument is unstable and unvalidated.**
   - NVIDIA's own Alpamayo 1.5 card moved its AlpaSim score from 0.81 to 1.37 on 2026-08-07, with unchanged weights.
   - The challenge cannot separate teams #2–#5.
   - The only published sim-to-road coefficient (Sim2Val, ρ > 0.9) is per-episode and single-policy, over kinematic metrics.
   - No multi-policy study shows that AlpaSim rankings are reliable or that they converge with any independent reference.

## 1. Goal and vision

**Goal:** an open release gate for the Alpamayo teacher → student path. It must:
- (a) measure regression with paired, cluster-adjusted statistics;
- (b) know its own validity envelope;
- (c) be validated as an instrument. Plan v4 keeps two kinds of evidence: reliability across seeds, and the known-order ladders that the student pipeline produces (flow steps, bit-widths, reasoning budget). Renderer and traffic agreement and cross-paradigm convergence are deferred. Predictive validity against on-road outcomes is out of reach without fleet data, so we never claim it (`02` §6).

**The headline application (plan v4).** Use the gate on the full teacher → Thor-student path:
- NVIDIA's own 2B student vs its 10B teacher in closed loop;
- closed-loop RL of the teacher;
- on-policy correction of the student, and its reasoning-budget curve;
- serving on Jetson AGX Thor with hardware-in-the-loop;
- optionally, shadow mode in Ho Chi Minh City.

The gate is the instrument. The loop is what it measures.

**Vision (3–7 years).** In physical AI, the rate of progress is set by how fast and how credibly the "fix → measure → release" loop runs, not by model size. Models change weekly, but the measuring instrument must stay trusted for years. We build the instrument.

## 2. The system map

```
        ┌──────────────────────────────────────────────────────────┐
        │                REINFORCING LOOP (data flywheel)           │
        ▼                                                          │
 [fleet-style logs] → [NuRec reconstruction] → [AlpaSim closed-loop]│
                                                   │               │
                       [Teacher: Alpamayo 1.5 10B / 2 Super ~34B]  │
                                   │ distill / RL / quantize       │
                                   ▼                               │
                          [Student ≤3B, FP8, Thor]                 │
                                   ▼                               │
                    ╔════════════════════════════╗                 │
                    ║  RELEASE GATE (this repo)   ║ ◄── BALANCING   │
                    ║  paired + stratified +      ║     LOOP        │
                    ║  validity + self-validated  ║                 │
                    ╚════════════════════════════╝                 │
                                   │ go                            │
                                   ▼                               │
                         [release → fleet] ────────────────────────┘
```

Plan v4 instantiates the right-hand path for one engineer. Shadow-mode disagreements flow back as new evaluation data, a small version of the reinforcing loop:

```
 [Alpamayo 1.5 10B] ─ closed-loop RL (S4) ─► [teacher] ─ labels ─► [NVIDIA 2B student (Edge)]
                                                                     │ S3 verdict, S5 correction,
                                                                     │ CoC-budget curve (gate)
                                                                     ▼
                       [Jetson AGX Thor serving, S6] ─► [HIL + latency injection, S7]
                                                                     ▼
                       [shadow mode in HCMC, S8: open-loop only] ─► disagreement clips ─► eval set
```

Three system laws explain why the gate is the leverage point:

- **Systems optimize what they measure (Goodhart).** RL is the strongest optimizer we own, so it is also the strongest exploiter of a flawed metric. AR1 shows this directly: a reasoning-only reward raised the critic grade from 3.1 to 4.5 while consistency fell (0.62 → 0.53) and ADE worsened.
- **Information flows have higher leverage than parameters (Meadows).** Knowing what is broken, and when, beats tuning the model.
- **Every reinforcing loop needs a balancing loop.** Without the gate, a bad release reaches the fleet, generates bad data, and that data returns to training.

A common misconception is that the value lies in the best model. In production, the best model is replaced within months, while the release process lasts years. Waymo publishes its approval framework, not its model (arXiv 2505.09880).

## 3. Ten core concepts (verified 2026-10-03)

**C1. The driving VLA has two brains.**
- **Architecture.** Alpamayo 1.5 (released checkpoint) pairs an ~8.2B VLM with a flow-matching action expert. The VLM is Cosmos-Reason2-8B on a Qwen3-VL architecture. The card lists the expert as 2.3B; the paper says 2B. The VLM writes a Chain-of-Causation trace, and the expert reads the VLM's KV cache to emit 64 waypoints over 6.4 s at 10 Hz.
- **Action space.** Unicycle controls `a ∈ [B, 64, 2]` (acceleration, curvature), with each step `a ← a + δt·v_θ(a, t, KV)`.
- **Step count.** Alpamayo 1.5 defaults to 10 Euler steps in code (`num_inference_steps = 10`), matching the paper's default δt = 0.1. The paper's own latency table uses 5 steps, and FlashDrive runs 8.
- **Inputs.** The released checkpoints take 4 cameras × 4 frames at 320×576. The token count, 180 per image and 2,880 in total, is derived (Qwen3-VL patch 16 × merge 2); NVIDIA states neither number. Egomotion history is 16 × (3 + 9) floats, tokenized as 48 tokens.
- **Reasoning length.** About 16 tokens per tick, as measured by FlashDrive.
- **Sources:** arXiv 2511.00088 v2; HF `nvidia/Alpamayo-1.5-10B` card and config.

**C2. Teacher and student.**
- **Positioning.** NVIDIA positions Alpamayo as a teacher distilled into compact runtime models on DRIVE AGX Thor.
- **State on 2026-10-25:**
  - **Distillation recipe:** alpamayo-recipes PR #42 (open, 10-20; no weights; ≥ 128 H100s recommended) distills Alpamayo 1 / 1.5 into a Cosmos-Reason2-2B student plus a compact expert. Its result is open-loop: minADE6 0.8875 m vs the teacher's 0.8298 m.
  - **Thor latency:** PR #45 runs such students at 171 ms in FP8 on DRIVE AGX Thor.
  - **Alpamayo 2 Edge** (2B, DRIVE AGX Thor, 10 Hz) was announced on 10-21. Its weights are promised "later this month".
- **Memory.** Alpamayo 1.5 needs 22.16 GB in BF16 (11.08B parameters) and about 11 GB in FP8. A 3B student needs about 6 GB in BF16.
- **Challenge limits.** 16 GiB per driver instance is a **hard limit**, so BF16 Alpamayo 1.5 does not fit and FP8 does. 0.1 s per call is a **target**, enforced through a throughput budget.
- **Thor.** "Thor" names two devices.
  - Jetson AGX Thor: 128 GB at 273 GB/s, JetPack 7.2.1 with TensorRT 10.16.2.
  - DRIVE AGX Thor: 64 GB at 273 GB/s, an ASIL-D design running DriveOS 7, with GPU memory set by carveout. TensorRT 11.x supports DRIVE AGX Thor only.

  The silicon matches, but the software stack does not, so numbers do not transfer 1:1. Every Thor claim names the device. The challenge's 16 GiB rule has nothing to do with Thor.
- **The production question:** what does the student lose, and where?

**C3. Open-loop vs closed-loop.** Open-loop evaluation compares predicted trajectories with logged ones. Closed-loop evaluation lets the model drive, so its errors compound.
- **nuPlan.** It called the two tasks "fundamentally misaligned" (arXiv 2306.07962). That is a qualitative trade-off; no coefficient is given.
- **Bosch 2026 (arXiv 2605.00066).** Open-loop L2 vs Bench2Drive driving score shows **no significant correlation** (n = 7). The paper prints ρ = −0.36, but recomputing from its own Table 1 gives −0.14. Its NAVSIM v1 ↔ Bench2Drive agreement does reproduce: ρ = 0.905 (n = 8).

**C4. Neural reconstruction.** NuRec rebuilds real scenes with 3D Gaussian Splatting and renders novel views when the ego deviates. AlpaSim wires renderer, traffic (log replay by default; CAT-K optional and unseeded), controller (MPC) and policy over gRPC. Its strength is that the long tail comes from real roads. Its weakness is C5.

**C5. Validity envelope.**
- **Fidelity loss.** Extrapolated views degrade. On EUVS (ICCV 2025), averaged over 9 methods, translation cuts PSNR by 24.6% and translation plus rotation by 30.6%. NuRec was not among the methods tested. OmniDreams shows NuRec degrading with deviation (Fig. 14).
- **Conflicting conventions.** The challenge scores a lateral exit of ≥ 4 m as **failure**. NVIDIA's papers instead **exclude** events beyond 4 m.
- **What we do.** Every rollout carries a validity record: maximum deviation from the log, in bins < 1, 1–2, 2–4 and ≥ 4 m; renderer agreement; and replay-vs-CAT-K agreement. Out-of-envelope results are reported separately: never silently dropped, and never silently scored.

**C6. Paired regression statistics.** Both models run the same scenes with the same seeds, and only discordant pairs carry information.
- **Worked example.** A fails 30 and B fails 24, with A-fail/B-pass = 12 and B-fail/A-pass = 6. Exact McNemar: P(X ≤ 6 | n = 18, ½) = 31,180 / 262,144 = 0.119, so two-sided p = 0.238. That is not evidence that B is better, and the test has power of only 0.22.
- **Repairs.**
  - Scenes cluster by road, so use a cluster-adjusted McNemar or a road-level bootstrap.
  - Decide with **paired non-inferiority** against a pre-registered margin.
  - At 441 scenes and 4% discordance, the minimum detectable difference is about 2.7 pp.
  - The rule of three is a **one-sided** bound: 0/441 gives ≤ 0.68% one-sided and 0.83% two-sided.

**C7. Stratification and failure taxonomy.** Report per scenario type. A flat aggregate can hide a regression confined to a rare stratum: a 2.1 pp regression in a 2% stratum shows up as only 0.042 pp pooled.
- **Taxonomy.** Ground it in AlpaSim's native scorers (`collision_at_fault` = front or lateral, `collision_rear`, `offroad`, `wrong_lane`, `min_distance_to_obstacle_m`, `plan_deviation`, `safety_monitor_triggered`), mapped to the NHTSA pre-crash typology.
- **Coverage gaps.** Close-encounter, stall, oscillation and red-light have no challenge-branch scorer, so they are new work. The taxonomy is PhysLoop's own and not an external standard.
- **Minimum stratum size.** A 9-scene stratum has at most 0.2% power. Underpowered strata return INSUFFICIENT, which blocks GO.

**C8. Measuring the instrument (G3).**
- **Why not "correlate with an independent reference".** Drive-IRT re-fits the same rollouts the gate scores (reference R0, aggregation robustness), and on-road outcomes are unavailable.
- **Construct validity instead.** The full protocol runs: reliability → renderer and traffic agreement → known-group dose–response ladders → cross-paradigm convergence on 28–34 checkpoints → a discriminant check → a three-way verdict (archived `02` v3 §2).
- **What plan v4 keeps** (`02` v4 §2): reliability (S2), plus the known-order ladders the student pipeline produces anyway (S5, S6). The other legs are deferred.
- **Worked example** (5 policies). Gate [1,2,3,4,5] vs reference [2,1,3,5,4]: Σd² = 4, so ρ = 1 − 24/120 = 0.8. Under H0 at n = 5, P(ρ ≥ 0.8) = 8/120. Even at n = 10, an observed 0.8 has a 95% CI of about [0.24, 0.96]. That is why we need n ≈ 28–34.
- **Prior art.** Pseudo-simulation reports r = 0.89 (Pearson, 83 privileged planners, sim-to-sim against nuPlan). Nothing comparable exists for AlpaSim.

**C9. Guardrail in the loop.** DRIVE AV runs the end-to-end stack in parallel with a classical safety stack built on Halos, in both the L2 Mercedes CLA and the L4-ready S-Class. Scoring the bare model therefore scores something that never ships.
- **The gate scores model + guardrail.** The guardrail is a TTC / drivable-area monitor behind AlpaSim's `safety_monitor_safe` hook. It is **PhysLoop's own design**: no public source specifies Halos's runtime-monitor logic.
- **Intervention rate.** The gate reports it as an early regression signal.
- **Status in plan v4:** G4 is deferred, not dropped. NVIDIA's Alpamayo Stack pairs the reasoning model with policy evaluators, safety monitors and Halos (verified 2026-10-25), so the argument above still holds.

**C10. Precision and throughput.**
- **The expert is memory-bound.** For π0, VLA-Perf models the action expert at an operational intensity of 54 FLOP/byte, against ridges of about 279 (RTX PRO 6000 Workstation, BF16) and about 947 (Jetson AGX Thor). That is a roofline model, not a measurement, and it describes π0, not Alpamayo.
- **What FlashDrive measured** on Alpamayo 1.5: 717 → 151 ms, mostly from streaming KV reuse, speculative reasoning, adaptive steps and CUDA graphs. W4A8 removed only 24.6 ms. The expert still runs at about 20% of its bandwidth bound (113.9 ms measured vs ≈ 22.3 ms, 8 steps).
- **Quantization recipe.** NVIDIA's ModelOpt recipe already quantizes the expert, but publishes open-loop minADE only. The open gap is closed-loop per-format verdicts.
- **AlpaGym.** It trains only the expert, with the VLM frozen (`cotrain_vlm = False` at the pinned recipes commit). Its rollouts generate no reasoning text, so a reasoning reward cannot move anything there. It accepts no outside PRs and publishes no stage-level throughput. We profile AlpaSim telemetry instead.

**C11. The Thor latency budget is set by memory bandwidth.** Decoding one token reads every backbone weight once: t_tok ≈ bytes / 273 GB/s.
- **Teacher.** A 10B FP8 backbone costs about 30 ms per token. Prefill of 2,560 vision tokens alone takes ≈ 99 ms even at 100% of peak. So no 10B configuration reaches 10 Hz.
  - Measured on Jetson AGX Thor: official BF16 build 3,322 ms; FlashRT 811–897 ms (PR #229).
- **Student.** A 2B NVFP4 student costs ≈ 4.1 ms per token. At the bound, about 10 CoC tokens fit in 100 ms. At measured efficiencies (decode ≈ 78%, expert ≈ 20% of bound) only about 3 fit. Streaming reuse of history-frame prefill raises that to ≈ 10–20.
  - Measured: NVIDIA's 2B students run at 171 ms (FP8) on DRIVE AGX Thor.
- **So:** runtime reasoning is a budget, and the research question is a curve: score vs CoC tokens at fixed latency (`02` §1, S5).
- **Sources:** `research/2026-10-25-teacher-to-thor/review.md` §3.1; FlashRT PR #229; recipes PR #45.

**C12. Shadow mode is open-loop.** In shadow mode the student predicts, but the human drives. The student never experiences the consequences of its own errors, so covariate shift, the reason closed-loop evaluation exists (C3), is invisible. Disagreement with the driver belongs to the same metric class as open-loop L2.
- **What shadow mode is good for:** mining out-of-distribution data (PAI-AV has no Asian country), measuring real-sensor latency on the device, and building a real-world open-loop benchmark.
- **What it cannot do:** validate driving quality.
- **Confound:** the rig's own sensor shift mixes with scene shift, so a within-rig contrast is needed (`02` S8).

## 4. Teach-back (answer before re-reading)

1. Why is "A fails 30, B fails 24 on 441" not enough to release B? Recompute the p-value and the power.
2. An FP8 student scores higher than the teacher, but 18% of its rollouts exceed the deviation cap. What do you conclude, and how do you report it under both the 2 m and the 4 m conventions?
3. Prediction, written before the deferred cross-paradigm study (v3's S5): what ρ do you expect between the gate and Drive-IRT on 10 policies? (Preserved from v1; now reference R0, an aggregation-robustness check, not a validity check.)
4. How can a guardrail hide a model regression, and which metric exposes it?
5. Why can a ρ < 0.6 kill rule not be decided at n = 10, and what n can decide it?
6. "RL cut at-fault collisions by 20%" on 441 held-out scenes, with a base rate of 8% and 4% discordance. Is that claim decidable? How many paired scenes would it need?
7. A 2B student runs at 171 ms with reasoning on DRIVE AGX Thor, yet a 2B model is announced at 10 Hz on the same device. Name three levers that close the gap. Which of them change what the model does in closed loop?
