> **ARCHIVED — historical, superseded on 2026-10-03 (DECISIONS D-010, D-011). Never act on this file; the live version is listed in `DOC_AUTHORITY.md`.**

# 01 — System design (systems-thinking view)

Authoritative for the system map and the core concepts. Read before any code.

## 0. The production problem

Late 2026. An OEM running NVIDIA DRIVE AV prepares an OTA update. The new model is smaller
and faster: distilled from a large teacher and quantized to FP8 for DRIVE AGX Thor. Before
release, one question must be answered:

> Does the new model drive worse than the old one? If so, in which situations?

In the open stack (Alpamayo, AlpaSim, AlpaGym) nothing answers this credibly:
1. **No release gate.** AlpaSim scores rollouts and the 2026 challenge adds Drive-IRT, but
   there is no go/no-go rule that compares checkpoint B to checkpoint A with confidence.
2. **No one has shown that AlpaSim scores predict anything.** There is no published
   correlation between a neural-reconstruction simulator's closed-loop score and any
   independent outcome. An uncalibrated scale makes every measurement meaningless.

## 1. Ultimate goal and vision

**Goal:** an open release gate for the Alpamayo teacher → student path that
(a) measures regression with paired statistics, (b) knows its own validity limits, and
(c) is validated as an instrument against independent references.

**Vision (3–7 years):** in physical AI the rate of progress is set by how fast and how
credibly the loop "fix → measure → release" runs, not by model size. Whoever owns the
measuring instrument the OEMs trust sits at the center of the flywheel.
Models change weekly; the instrument must be trusted for years. We build the instrument.

## 2. The system map

```
        ┌──────────────────────────────────────────────────────────┐
        │                REINFORCING LOOP (data flywheel)           │
        ▼                                                          │
 [fleet-style logs] → [NuRec reconstruction] → [AlpaSim closed-loop]│
                                                   │               │
                          [Teacher: Alpamayo 2 Super 34B]          │
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

Three system laws explain why the gate is the leverage point:
- **Systems optimize what they measure (Goodhart).** RL is the strongest optimizer we
  own, so it is also the strongest exploiter of a flawed metric.
- **Information flows are higher leverage than parameters (Meadows).** Fixing what is
  known, and when, beats tuning the model.
- **Every reinforcing loop needs a balancing loop.** Without the gate, a bad release
  reaches the fleet, generates bad data, and the bad data returns to training.

Common misconception: "the value is in the best model." In a production system the best
model is replaced in three months; the release process lasts years. Waymo publishes its
approval framework, not its model (Favaro et al., arXiv 2505.09880).

## 3. Ten core concepts

**C1. The driving VLA has two brains.** Alpamayo 1.5: an ~8.2B VLM (Cosmos-Reason2)
reads camera frames and writes a Chain-of-Causation trace; a ~2.3B flow-matching action
expert reads the VLM KV cache and emits 64 waypoints for 6.4 s after ~10 Euler steps.
Tensor: `a_t ∈ [B, 64, 2]` (accel, curvature); each step `a ← a + 0.1·v_θ(a, t, KV)`.
Source: arXiv 2511.00088; HF model card nvidia/Alpamayo-1.5-10B.

**C2. Teacher and student.** NVIDIA positions Alpamayo as a teacher distilled into
compact runtime models on Thor. Memory arithmetic: Alpamayo 1.5 BF16 ≈ 22 GB; FP8 ≈ 11 GB;
a 3B student BF16 ≈ 6 GB. The challenge limit is 16 GiB and 0.1 s per call; Thor bandwidth
is 273 GB/s. Production question: what does the student lose, and where? No public
distillation procedure exists ("planned" per NVIDIA's closed-loop blog).

**C3. Open-loop vs closed-loop.** Open-loop compares predicted to logged trajectories;
closed-loop lets the model drive and errors compound. Rankings can invert: nuPlan showed
the two are "fundamentally misaligned" (arXiv 2306.07962); Bosch 2026 measured
ρ = −0.36 between open-loop L2 and Bench2Drive driving score (arXiv 2605.00066).

**C4. Neural reconstruction.** NuRec rebuilds real scenes with 3D Gaussian Splatting and
renders novel views when the ego deviates; AlpaSim wires renderer, traffic, controller and
policy over gRPC. Strength: the long tail comes from real roads. Weakness: C5.

**C5. Validity envelope.** A lane-change-scale translation cuts PSNR by 24.6%; translation
plus rotation by 30.6%, with all methods below PSNR 15 (EUVS, ICCV 2025). Every rollout
carries a validity record: max ego deviation from the log; renderer agreement (NuRec vs
OmniDreams); on-rails vs reactive (CAT-K) traffic agreement. Out-of-envelope results are
reported separately, never silently dropped and never silently scored.

**C6. Paired regression statistics.** Both models run the same 441 scenarios with the same
seeds. Only discordant pairs carry information. Example: A fails 30, B fails 24, but
A-fail/B-pass = 12 and B-fail/A-pass = 6. Exact McNemar on 18 discordant pairs:
P(X ≤ 6 | n=18, ½) = 31,180/262,144 ≈ 0.119 → two-sided p ≈ 0.24. Not enough evidence
that B is better. Traps: rule of three (0 failures in 441 → upper bound 3/441 ≈ 0.68%);
scenarios cluster by road, so bootstrap over logs, not rollouts.

**C7. Stratification and failure taxonomy.** Report per scenario type (unprotected
intersection, pedestrians, merges, night…). Label failures: at-fault collision, rear
collision, off-road, close-encounter, stall, oscillation, red-light. A flat aggregate can
hide a 3× regression in a rare stratum, which is exactly the kind that kills people.

**C8. Measuring the instrument (G3).** Do gate rankings agree with independent references?
Spearman ρ with bootstrap CI over ≥ 10 policies. Worked example, 5 policies: gate
[1,2,3,4,5] vs reference [2,1,3,5,4] → Σd² = 4 → ρ = 1 − 24/120 = 0.8 (CI far too wide at
n=5; hence ≥ 10). Best published: pseudo-simulation r = 0.89 over 83 planners
(arXiv 2506.04218), NAVSIM↔Bench2Drive ρ = 0.90 over 8 (arXiv 2605.00066). Nothing exists
for AlpaSim, and nothing links neural-reconstruction scores to on-road outcomes.
If ρ < 0.6 we publish the negative result.

**C9. Guardrail in the loop.** DRIVE AV on the Mercedes CLA runs the end-to-end stack in
parallel with a classical safety stack under Halos. Scoring the bare model scores something
that never ships. The gate scores model + guardrail (TTC / drivable-area override) and
reports the intervention rate; a 3× rise in interventions is an early regression signal.

**C10. Precision and throughput.** The action expert is memory-bound (≈ 54 FLOP/byte vs a
Blackwell BF16 ridge ≈ 281; VLA-Perf arXiv 2602.18397), so weight bytes set latency.
FlashDrive quantizes the VLM (W4A8) but keeps the expert BF16 (arXiv 2608.12932).
AlpaGym already runs closed-loop RL on the expert with SDE log-probs but publishes no
throughput; we profile it with its own `perf/` module and send PRs.

## 4. Teach-back (answer before re-reading)
1. Why is "A fails 30, B fails 24 on 441" not enough to release B? Recompute the p-value.
2. An FP8 student scores higher than the teacher but 18% of its rollouts exceed the
   deviation cap. Conclusion, and how to report it?
3. Prediction, written before week 5: what ρ between the gate and Drive-IRT on 10 policies?
4. How can a guardrail hide a model regression, and which metric exposes it?
