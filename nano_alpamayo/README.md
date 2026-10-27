# nano_alpamayo — the student (stages S3 and S5; G5)

**The student is NVIDIA's own 2B model, not one we distill from scratch** (DECISIONS D-012):
- Alpamayo 2 Edge, once its weights are public (KC-E, re-check 2026-10-31);
- otherwise, only if Huy approves P-017, a student trained at pilot scale with NVIDIA's public distillation recipe, alpamayo-recipes PR #42 (unmerged; its stage 1 recommends at least 128 H100s). A 10%-schedule pilot never reaches the recipe's on-policy phase, so it is not NVIDIA's student (`02` K12). PR #42's student VLM, Cosmos-Reason2-2B, is under the NVIDIA Open Model License.

nanoAlpamayo is our closed-loop work on top of that student:
- on-policy correction;
- the runtime-reasoning (CoC) budget curve;
- a small AR1-0.5B-style ablation if needed.

Every closed-loop number comes from the PhysLoop gate. Work here starts after S2 (the pinned protocol) and S3 (the student's closed-loop verdict) exist (`../docs/02_PHYSLOOP_PLAN.md` §3).

## Acceptance (re-based 2026-10-25; A1 corrected and A7 added 2026-10-27 under D-015; Huy sets every number)

The reference numbers are context only. PR #42 reports open-loop minADE6 on the full PAI-AV validation set: 0.8875 m for the 1.5-2B student vs 0.8298 m for the teacher. AR1's 0.5B targets use unpublished lists and unreleased weights.

| # | Criterion | Target |
|---|---|---|
| A1 | minADE6 on a **published** PAI-AV subset: a clip-UUID manifest with location-grouped splits.<br>• **Clips** are drawn only from those labelled **`test`** in `splits/alpamayo_aligned.parquet` (PAI-AV v26.10, revision `39414de6`, read with `physical_ai_av` ≥ 0.2.3). That file labels every clip (train 276,299 / test 28,317 / alpasim 1,536), so "excluding" it would exclude everything (corrected 2026-10-27).<br>• **`alpasim` clips stay out of A1:** S5's rollouts and the 441 evaluation scenes come from them.<br>• **Selection.** Clips are selected chunk-aware, because PAI-AV stores each sensor in chunks of ≤ 100 clips, and the selection is checked for bias. The overlap of PR #42's 644-clip list with each label is reported.<br>• **Anchors:** the base student and the teacher, re-evaluated on the same subset | ____ |
| A2 | Correction gain: a paired closed-loop verdict, corrected vs base student, on the gate, over ≥ 3 replicates (seed mode as recorded in the manifest) | ____ |
| A3 | Reasoning moves in the same direction, with a CI, on public stand-ins only (AlpaJudge on the 349-label public validation set). Auto-labels stay unpublished (license §4.6) | direction + CI |
| A4 | Closed loop: a paired non-inferiority verdict against the teacher on a named metric, under one pinned protocol (scene list, AlpaSim SHA, lockfile, seeds, 4 m convention). The owner's target margin is 10% of the teacher's score | Δ = ____ |
| A5 | The decoder reproduces AR1 Eq. 5: the known-answer arc gives 0.00199 m, vs 0.5972 m for forward Euler | exact |
| A6 | The CoC-budget curve: closed-loop score vs runtime token budget n, with latency measured on Jetson AGX Thor (S6). The student's natural CoC length is measured first; caps are set by budget forcing on one checkpoint; the realized length is reported per cap; the n = 0 point is labelled by architecture (an expert reading an empty CoC is not a tacit-trained model) | shape = ____ |
| A7 | Every correction round reproduces from its manifest: student and teacher hashes, AlpaSim SHA and lockfile, scene IDs, replicate or seed mode, label stride, `n_sim_steps`, trainer commit. Per-stage GPU-hours are reported from the mean wall clock (recipes PR #44) | exact, or the nondeterminism documented |

## Order

1. Bring in the base student.
   - Check that its license permits evaluation and fine-tuning.
   - Serve it through the S3 batched AlpaSim driver (`02` S3), which is also the collection driver.
   - Overfit one batch, and check the loss at init, on the training path.
2. Pass the decoder known-answer test (A5).
3. On-policy correction in AlpaSim, in the DAgger/RoaD style. The teacher labels the states the student visits; the teacher is the S4 RL'd 1.5 if KC-4 passed, otherwise the released 1.5.
   - **The learner** is PR #42's stage-2 expert-SFT trainer, with teacher actions entering through `target_action`. The first check is that its batch builder passes the key through; then the sampler parity check.
   - **Observations** come from AlpaSim's per-rollout `.asl` logs; **teacher labels** are produced offline by replay, keyed by (session, timestamp).
   - **Every round** has a manifest (A7), and per-stage throughput is measured.
4. Flow-step ladder: K ∈ {10, 5, 4, 2} in closed loop, plus FlashDrive's velocity-reuse schedule at equal network evaluations. K = 8 and K = 1 run open-loop only. One fixed noise file per scene and replicate pairs every arm.
5. CoC-budget arms. Streaming reuse of history-frame prefill is a prerequisite; FlashDrive reports that it costs +2.7% minADE6. Measure the natural CoC length first, then force budgets on the same checkpoint (A6).
6. Optional: a short GRPO pass on the student. Name the estimator.
7. Last, as an ablation: Flow-SDE through the expert, with the expert-drift monitor.
8. Only if needed: the small AR1-0.5B layout (DINOv2 + Qwen2.5-0.5B).

## Kill checks
- **KC-E:** no Edge weights at S3's start → S4 runs first, and S3 measures NVIDIA's deployment levers on the released teacher (`02` S3). If Edge is still not public at S5's start, Huy chooses between two paths: a PR #42 pilot student (P-017, limits stated), or running only the ladder and the CoC curve on the released teacher, labelled so.
- **KC-5:** no paired correction gain after three ablation rounds → publish the negative result with its mechanism.
- **KC-L2:** no weights or labels trained on PAI-AV/NuRec-derived data are published without a written answer on license §4.6.

## Risks
- **Edge's license is unstated** as of 2026-10-25. Fine-tuning may not be permitted.
- **Compute.** No verified GPU-hour figure exists for closed-loop correction of a 2B VLA. Pilot first.
- **Data license.** PAI-AV expires 12 months after first download (S2).
- **Race.** NVIDIA's alpamayo1.5 PR #36 thread shows a 2B Alpamayo policy already running inside AlpaGym. Re-check novelty before claiming the first closed-loop correction of an NVIDIA student.
