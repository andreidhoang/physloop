# nano_alpamayo — the driving VLA we build (stage S7; G5 merged)

nanoAlpamayo is the **gate-measured student**. Every closed-loop number it produces comes from the PhysLoop gate. It starts only after S2–S4 exist: the gate, the validity envelope, and the choice of deployment format (`../docs/02_PHYSLOOP_PLAN.md` §3 S7).

## Acceptance (re-based 2026-10-03; Huy sets every number)

AR1's published 0.5B targets are reference only:
- The 644-sample and 920-scene lists are unpublished, and the 0.5B weights were never released.
- The RL row (2.12 → 1.92 m) is a most-likely-mode ADE on NVIDIA's internal CoC data.
- The reasoning scores come from an internal critic.

| # | Criterion | Target |
|---|---|---|
| A1 | minADE6 on a **published** PAI-AV subset: a clip-UUID manifest with location-grouped splits. Anchor: a released checkpoint re-evaluated on the same subset. Reference: AR1-0.5B reports 0.913 m on its own 644-sample list | ____ |
| A2 | RL gain measured as public-validation minADE6 or most-likely-mode ADE over ≥ 3 paired seeds, plus a CoC-consistency score from an open judge on the 349-label public validation set | ____ |
| A3 | Reasoning moves in the same direction, with a CI, on public stand-ins only. Auto-labels stay unpublished (license §4.6) | direction + CI |
| A4 | Closed loop: a paired non-inferiority verdict on the gate against a released checkpoint, under one pinned protocol (scene list, AlpaSim SHA, seeds, 4 m convention). The margin can be no tighter than about 2.7 pp, what 441 scenes can certify | Δ = ____ |
| A5 | The decoder reproduces AR1 Eq. 5: the known-answer arc gives 0.00199 m, vs 0.5972 m for forward Euler | exact |

## Order

1. Data pipeline on a PAI-AV subset with location-grouped splits. Publish code, clip-UUID manifests and aggregates only.
2. Overfit one batch; check the loss at init.
3. SFT: a small VLM backbone (AR1's 0.5B is DINOv2 + Qwen2.5-0.5B), 128 discrete trajectory tokens, and a flow expert reading the VLM KV cache.
4. CoC SFT on auto-labels from Alpamayo 1.5, the cheap teacher NVIDIA recommends, or from Alpamayo 2 Super (one H100 per instance). Public CoC labels cover only 1,450 training and 290 validation clips.
5. Closed-loop correction in the RoaD/OPTED style. Both report orders-of-magnitude better sample efficiency than direct RL in AlpaSim.
6. GRPO on tokens. Name the estimator: AR1's variant is unclipped and softmax-weighted; AlpaGym's is clipped.
7. Flow-SDE through the expert, last, as an ablation, with the expert-drift monitor.

## Kill checks (KC-7)
- (a) NVIDIA releases its 10B → ~2B distillation script → pivot to evaluating NVIDIA's student on the gate.
- (b) A4 is missed after three ablation rounds → shrink the subset and fix the pipeline.
- (c) A1 is missed after three rounds → same response.

## Risks
- **Data scale.** 100–300 h is 0.1–0.4% of AR1's 80,000 h. AR1's own data-scaling rows show overfitting at small scale; this is a hypothesis, since the test sets differ.
- **License.** The PAI-AV license expires 12 months after first download.
- **Compute.** No verified GPU-hour figure exists for 0.5–3B VLA SFT or GRPO. Pilot first.
