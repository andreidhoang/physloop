# nano_alpamayo — the student (stages S3 and S5; G5)

**The student is NVIDIA's own 2B model, not one we distill from scratch** (DECISIONS D-012):
- Alpamayo 2 Edge, once its weights are public (KC-E, re-check 2026-10-31);
- otherwise, a student trained at pilot scale with NVIDIA's public distillation recipe, alpamayo-recipes PR #42 (unmerged; it recommends at least 128 H100s at full scale).

nanoAlpamayo is our closed-loop work on top of that student:
- on-policy correction;
- the runtime-reasoning (CoC) budget curve;
- a small AR1-0.5B-style ablation if needed.

Every closed-loop number comes from the PhysLoop gate. Work here starts after S2 (the pinned protocol) and S3 (the student's closed-loop verdict) exist (`../docs/02_PHYSLOOP_PLAN.md` §3).

## Acceptance (re-based 2026-10-25; Huy sets every number)

The reference numbers are context only. PR #42 reports open-loop minADE6 on the full PAI-AV validation set: 0.8875 m for the 1.5-2B student vs 0.8298 m for the teacher. AR1's 0.5B targets use unpublished lists and unreleased weights.

| # | Criterion | Target |
|---|---|---|
| A1 | minADE6 on a **published** PAI-AV subset: a clip-UUID manifest with location-grouped splits that **excludes clips in `splits/alpamayo_aligned.parquet`** (teacher contamination). Anchors: the base student and the teacher, re-evaluated on the same subset | ____ |
| A2 | Correction gain: a paired closed-loop verdict, corrected vs base student, on the gate, over ≥ 3 seeds | ____ |
| A3 | Reasoning moves in the same direction, with a CI, on public stand-ins only (AlpaJudge on the 349-label public validation set). Auto-labels stay unpublished (license §4.6) | direction + CI |
| A4 | Closed loop: a paired non-inferiority verdict against the teacher on a named metric, under one pinned protocol (scene list, AlpaSim SHA, lockfile, seeds, 4 m convention). The owner's target margin is 10% of the teacher's score | Δ = ____ |
| A5 | The decoder reproduces AR1 Eq. 5: the known-answer arc gives 0.00199 m, vs 0.5972 m for forward Euler | exact |
| A6 | The CoC-budget curve: closed-loop score vs runtime token budget n, with latency measured on Jetson AGX Thor (S6) | shape = ____ |

## Order

1. Bring in the base student.
   - Check that its license permits evaluation and fine-tuning.
   - Overfit one batch, and check the loss at init, on the training path.
2. Pass the decoder known-answer test (A5).
3. On-policy correction in AlpaSim, in the DAgger/RoaD style. The teacher labels the states the student visits; the teacher is the S4 RL'd 1.5 if KC-4 passed, otherwise the released 1.5.
4. Flow-step ladder: K = 10 → 4 → 2.
5. CoC-budget arms. Streaming reuse of history-frame prefill is a prerequisite; FlashDrive reports that it costs +2.7% minADE6.
6. Optional: a short GRPO pass on the student. Name the estimator.
7. Last, as an ablation: Flow-SDE through the expert, with the expert-drift monitor.
8. Only if needed: the small AR1-0.5B layout (DINOv2 + Qwen2.5-0.5B).

## Kill checks
- **KC-E:** no Edge weights at S3's start → S4 runs first, and the base student comes from a PR #42 pilot under Huy's cost cap.
- **KC-5:** no paired correction gain after three ablation rounds → publish the negative result with its mechanism.
- **KC-L2:** no weights or labels trained on PAI-AV/NuRec-derived data are published without a written answer on license §4.6.

## Risks
- **Edge's license is unstated** as of 2026-10-25. Fine-tuning may not be permitted.
- **Compute.** No verified GPU-hour figure exists for closed-loop correction of a 2B VLA. Pilot first.
- **Data license.** PAI-AV expires 12 months after first download (S2).
