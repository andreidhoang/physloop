# L3 — Simulation quality on AlpaSim (NuRec, traffic models, Cosmos), mapped to JR2010863 I2 / D2 / D3 / X2

**As of 2026-10-27.** Research only: no sign-in, no licence accepted, no PAI-AV/NuRec data or gated weights downloaded, no Git LFS objects pulled, nothing posted. AlpaSim was read from two shallow clones made on 2026-10-27, with LFS skipped:
- `main` @ `affc2eab209fa43bdfa2f26c0f8d437922d78a68` (committed 2026-08-18; unchanged since);
- `e2e_challenge` @ `e121e59141dc391f6c295513d6dadc840a419149` (2026-10-20).

The GitHub API rate limit (60/h, unauthenticated) ran out at the end of the session. Items left unchecked because of it are marked as gaps.

**Conventions.**
- **P / S:** primary / secondary source.
- **Confidence:** H / M / L.
- **"via fetch":** the number came from a WebFetch model summary of the page, not from my own full-text read. Treat it as M unless it was cross-checked.
- **Inference:** labelled as such.
- **Prices:** plan basis, RTX PRO 6000 Workstation on vast.ai, 2026-10-25: $1.00 cheapest / $1.60 median. The $ ranges below are low GPU-h × $1.00 to high GPU-h × $1.60. H100 SXM is $1.93 / $2.40.
- **Step time:** 0.463–1.150 s per rollout step (`04` §4, FlashDrive). One sweep is 441 × 3 × 200 = 264,600 steps, or 34.0–84.5 GPU-h. A **one-seed arm** is 441 × 200 = 88,200 steps, or 11.3–28.2 GPU-h ($11–45).
- **Arithmetic:** reproducible from `scratchpad/calc/l3_calc.py` and `scratchpad/calc/l3_costs.py`. Full paths:
  - `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/calc/l3_calc.py`
  - `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/calc/l3_costs.py`

**Summary.**
- The cheapest credible "measure **and** improve" package for a NuRec-based simulator has three tiers:
  - **Tier 0, $0, +7–9 h.** Four zero-cost changes to how S2 is run and reported: L3-1, L3-3, L3-4, L3-5.
  - **Tier 1, ≤ $20, +9–11 h.** A 5-scene closed-loop cost pilot (L3-2) and a render-level fidelity probe (L3-6). L3-6 uses NVIDIA's own Cosmos-based Harmonizer as the "improve" lever.
  - **Tier 3 (gated), $81–545.** Separately gated closed-loop arms for the deferred R2 and R3 legs, plus the performance levers: L3-8, L3-9, L3-10.
- Three repository facts reshape the deferred legs:
  1. **CAT-K is absent on `e2e_challenge`.** It exists only on `main`, while the 441-scene curated split and the post-#215 corridor scorer exist only on `e2e_challenge`. A clean replay-vs-CAT-K comparison therefore has to run both arms on one SHA.
  2. **CAT-K is still unseeded.** This is confirmed at code level: the seed reaches the service and is logged, but sampling never uses it.
  3. **The Harmonizer setting moves the cost.** The Harmonizer is ON in the `e2e_challenge` developer default, OFF in the competition preset, and ON for the challenge's final runs. Pinned ON, it could add up to $249–1,396 to S2, by inference. That holds only if the paper's 212 ms (H100, 1024×576) is per frame and the cost scales with pixels; the paper does not confirm the unit. A $0–2 pilot (L3-2) measures it before S2's protocol is pinned.
- **Policy-level rank agreement over S2's 3 policies is INSUFFICIENT by construction.** P(τ = 1 | H0) = 1/6 at n = 3. Every agreement claim in L3 is therefore scene-level (κ over 441 scenes, by deviation bin) or a decision-level check: whether S2's BF16-vs-FP8 verdict survives the change.
- **NuRec-vs-Cosmos ranking agreement (vi) is rejected for v4.** NVIDIA already published n = 4 for OmniDreams. AlpaSim's open video-model path is single-view only. Cosmos3-Nano-Sim-Auto (released 2026-10-20) has no AlpaSim adapter, no published closed-loop agreement and no sm_120 attention kernels. Building one would race NVIDIA.

---

## 1. Findings table

Columns:
- **ID**.
- **JR lines** (JR2010863).
- **Stage · deliverable**.
- **Kind:** (a) uses S2's pinned rollouts or a manifest/config field, with no extra sweep; (b) a design doc plus a real stand-in measurement; (c) a gated addition: a renderer/traffic config change, an added arm or seed, or an un-deferral.
- **Stage-card change:** B = Build, ★ = Huy's cores, A = Acceptance, P = Prediction slot (left blank as `____`), K = Kill.
- **GPU-h · $** (basis above).
- **Hours** delta.
- **Gate**.
- **Owner**.
- **Race check (2026-10-27)**.
- **Sources:** F-IDs in §2.

| ID | JR lines | Stage · deliverable | Kind | Stage-card change (B / ★ / A / P / K) | GPU-h · $ | Hours | Gate | Owner | Race check (2026-10-27) | Sources |
|---|---|---|---|---|---|---|---|---|---|---|
| **L3-1** Pin the full simulator configuration | D2, D3, D4 | S2 · G1, G2, G7 (manifest) | (a) | **B:** the manifest gains these fields: renderer image (`nre-ga:26.04`); Harmonizer flag; `--no-enable-nrend`; `render_bundling`; camera request (4 × 1080×1900, plus decoded width); `nre_max_workers`; `n_concurrent_rollouts`; `replicas_per_container`; driver `max_batch_size`; `trafficsim` mode; evaluator SHA (post-#215 corridor fix); renderer-OOM requeue count per policy. **★:** Huy picks the Harmonizer state of the pinned protocol, using L3-2 and #222, and records it in `experiments/` before S2's first run. **A:** a rollout record without these fields is "provisional". **P:** none (configuration). **K:** none | 0 · $0 | +1 | none (inside S2's manifest line) | agent; the choice is ★ Huy's | n/a | F1–F4, F18, F31, F51–F55 |
| **L3-2** Harmonizer cost pilot | I2, D3, D2 | S2 · G7 | (c) | **B:** 5 scenes × 1 seed, A1.5 BF16, Harmonizer OFF vs ON, renderer `--enable-timing` on. **★:** the decision. **A:** per-step renderer time ON vs OFF, p50 and p99. **P:** `P-L3-2: seconds added per step by the Harmonizer at 4 × 1080×1900 on RTX PRO 6000: ____`. **K:** if the added cost exceeds Huy's ceiling (`____` s/step), pin OFF and run ON only as L3-8's arm | 0.5–1.5 · $0–2 | +1 | P-005 (needed for any S2 spend) | agent runs; ★ Huy decides | Not published for AlpaSim. The paper gives 212 ms on H100 at 1024×576, unit not confirmed as per frame | F3–F6 |
| **L3-3** Instrument-noise decomposition | D3, D4, X2 | S2 · G3·R1 | (a) | **B:** split seed test–retest into renderer, policy and traffic parts: (1) render-determinism probe: 441 scenes × 10 fixed poses × 4 cameras, rendered twice and compared bitwise / by max-abs difference, per Harmonizer state; (2) policy probe: identical inputs passed twice; (3) traffic: replay recorded as deterministic; (4) the tick at which each seed pair's ego poses first diverge, read from logs. Together with S2's seed-vs-seed κ per deviation bin, this onset tick is the closed-loop floor against which L3-8 and L3-9 are read. **★:** the variance-component model and its reading. **A:** each component's share of seed discordance, with a CI; seed-vs-seed κ per bin is published as the floor. **P:** `P-L3-3: renderer share of seed-to-seed discordance: ____`. **K:** if renders are not repeatable at matched poses, report the renderer as a noise source before any verdict | ≤ 0.1 if run inside a sweep job (1.2–3.8 standalone) · $0–6 | +2–3 | none | agent harness; ★ Huy model | Not published. A maintainer said only that NuRec has "small" run-to-run variation (#128, 2026-07-28) | F21, F59, ledger #128 |
| **L3-4** Validity envelope as a dose–response curve, plus NVIDIA's clip-quality covariate | D3, I2, X2 | S2 · G2 (inputs to G3·R4) | (a) | **B:** per-policy failure rate and `scene_score` by deviation bin (< 1, 1–2, 2–4, ≥ 4 m) as a curve. Join `clip_ratings_*.csv` (`quality_score` "perception based (beta)", `time_of_day`, `ego_speed`). Add an endpoint-overshoot class (post-#215 semantics). Use `img_is_black` as a render-failure count. **★:** strata, validity semantics, covariate analysis, diagnosis. **A:** curves with road-clustered CIs; the rating-join coverage is stated. **P:** `P-L3-4: rank correlation between quality_score and per-scene validity-exit rate: ____`. **K:** none (descriptive) | 0 · $0 | +2 | KC-L to publish | agent join and plots; ★ Huy | Partial precedent only. NVIDIA **excludes** > 4 m (AR1, RoaD) and plots NuRec FVD against mean deviation without bin values (OmniDreams Fig. 14). No per-policy curve and no rating covariate are published | F9, F12, F14, F31 |
| **L3-5** Critical-path profile from S2 telemetry, with an Amdahl bound per lever | D4, D2, X3, I2 | S2 · G7 | (a) | **B:** export telemetry per run. `e2e`: 14 unprefixed series, including `rpc_duration_seconds{service,method,tag}`. `main`: 18 `alpasim_*` series. Turn on renderer `--enable-timing`. Compute each service's share of step time, queue depth and idle fraction. Bound each lever's speed-up with S = 1/((1−f) + f/k) **before** running it (E4 pixels, E2 batching, render bundling, replicas). **★:** the cost model and bottleneck diagnosis. **A:** every throughput number names its preset, SHA, GPU edition and Harmonizer state. **P:** `P-L3-5: renderer share of step wall time at 4 × 1080p, Harmonizer OFF: ____` (E1 and E4 are preserved separately). **K:** none | 0 · $0 | +2–3 | none | agent export; ★ Huy model | Not published: no AlpaSim technical report or stage profile was found | F50–F58 |
| **L3-6** Render-fidelity curve without novel-view ground truth | D3, I2, X2 | S2 · G2 (partial G3·R4, render-level R2) | (c) | **B:** render the 4 policy cameras at lateral offsets d ∈ {0, 1, 2, 4 m}, at 20 timestamps per scene along the log. (1) Perception consistency: project the ClipGT `obstacle` cuboids with the clip calibration, run an off-the-shelf detector on the renders, and report recall/IoU against d, normalised to d = 0. (2) KID of front-wide renders against the clip's recorded `camera_front_wide_120fov.mp4`. (3) Harmonizer OFF on all 441 scenes, ON on a 100-scene subset. **★:** metric semantics, interpretation, any cap. **A:** fidelity-vs-deviation curves with road-clustered CIs; detector and KID versions in the manifest. **P:** `P-L3-6a: detector recall at d = 2 m and 4 m relative to d = 0: ____`; `P-L3-6b: change in that recall from the Harmonizer at 4 m: ____`. **K:** if recall on renders at d = 0 is below `____` (detector unusable), keep KID only | 3.7–11.2 · $4–18 (ON on all 441: 10.5–34.3 · $10–55) | +8–10 | DECISIONS entry (partial un-deferral of R4 and render-level R2) + P-005; KC-L | agent harness; ★ Huy | Open. No public NuRec quality-vs-offset curve with values. DiffusionHarmonizer reports one 2 m point on 13 internal scenes. The Fixer blog gives one PSNR pair and a qualitative 3 m example | F5–F9, F12–F14, F60 |
| **L3-7** Policy-sensitivity probe (open loop) | D3, I2, X2 | S2 · G3·R2 | (c) | **B:** at 5 states × 4 offsets per scene, compare A1.5 BF16 plans under Harmonizer OFF vs ON (appearance-only change at the identical pose) with a seed-to-seed repeat as the noise floor. The same harness carries E4's resolution variant (L3-10) and can log the `open_loop_collision` flag. **★:** the sensitivity statistic and its floor. **A:** the ratio per deviation bin, with a CI. **P:** `P-L3-7: plan change from appearance ÷ plan change from seed, at d = 0 and d = 4 m: ____`. **K:** if A1.5 cannot be seeded on `e2e`, use the raw sampling spread as the floor and say so | 3.0–8.4 · $3–13 (all 441: 13.2–37.1 · $13–59) | +5–6 | same DECISIONS entry as L3-6; P-005; KC-L | agent; ★ Huy | Open: nothing published per call | F5, F6, F18, F59 |
| **L3-8** Closed-loop renderer agreement: minimal R2 (Harmonizer) | D3, I2, X2 | S2 extension · G3·R2 | (c) | **B:** one extra arm per policy (A1.5 BF16 and FP8): 441 scenes × seed 0, with the Harmonizer flipped relative to the pinned protocol. Report scene-level κ and paired score difference by deviation bin, whether S2's BF16-vs-FP8 verdict survives, and the throughput cost. **★:** κ and bootstrap, the invariance rule, diagnosis. **A:** κ with road-clustered CIs per bin, reported beside S2's seed test–retest κ in the same bin (the floor); the verdict under both states; rank agreement over 3 policies reported as INSUFFICIENT. **P:** `P-L3-8: share of pass/fail labels that flip between Harmonizer states, < 1 m vs ≥ 4 m: ____` (the v3 slot "τ for Harmonizer on vs off" is preserved). **K:** if κ in the < 1 m bin is below `____`, the renderer is a dominant noise source; report this before any S3 verdict | 22.7–56.3 · $23–90 (if the flipped arm is ON and the 212 ms is per frame, it adds 20.8–72.7 GPU-h per policy, by inference: $64–323 worst case in total) | +6–8 | DECISIONS (un-defer R2, scoped to the Harmonizer) + P-005 + KC-L | agent runs; ★ Huy | Partial race risk. NVIDIA ran challenge initial runs OFF and plans final runs ON, so it holds both datasets. Nothing is published; re-check the challenge reports in November | F3–F6 |
| **L3-9** Closed-loop traffic agreement: minimal R3 (log replay vs CAT-K) | I2, D3, X2, Q4 | S2 extension · G3·R3 | (c) | **B:** both arms on **one** SHA (`main@affc2eab`, or the post-10-31 release), with only `trafficsim` changed (`disabled` vs `catk`). Pass the 441 scene IDs from `e2e`'s `sim_suites_curated.csv` explicitly. Score both arms with one evaluator: port `e2e`'s corridor scorer, or re-evaluate the logs. Policy A1.5 BF16 (FP8 optional): replay × 1 and CAT-K × 3 repeats, because CAT-K is unseeded. Alternatively, upstream a seeding PR that threads `random_seed` into CAT-K sampling (#128 invited it), but only if NVIDIA's next release lacks it. **★:** agreement statistics, the invariance rule, a diagnosis of every flip. **A:** κ, flip rate per failure type and paired difference by bin, each reported beside S2's seed test–retest κ in the same bin; CAT-K's own repeatability; rank claims INSUFFICIENT at n ≤ 3. The replay arm runs once on `main`, so its repeatability is assumed equal to S2's `e2e` measurement, and the report says so. **P:** `P-L3-9a: change in at-fault collision rate, replay → CAT-K: ____`; `P-L3-9b: share of failure labels that flip (v3 slot): ____`. **K:** if CAT-K returns `FAILED_PRECONDITION` on more than `____`% of scenes, report coverage and stop | 45.4–112.7 · $45–180 (two policies: 90.7–225.4 · $91–361) | +10–13 | DECISIONS (un-defer R3) + P-005 + KC-L; opening a PR is Huy's outward action | agent port, PR code and runs; ★ Huy | Open for AlpaSim. The nuPlan precedent (IDM reactive agents) is CaRL Table 6. The seeding PR may collide with NVIDIA's "next release" (#224) | F20–F25, F28, F29, F32 |
| **L3-10** Performance levers with an equivalence guard (E4, E2, bundling) | I2, D2, X3, D3 | S2 (E4 and E2 already optional) · G7 | (c) | **B:** one variable per arm. (1) **E4:** a new 4-camera model-resolution camera config (no preset exists); an open-loop screen through L3-7's harness; then a one-seed closed-loop paired verdict 1080p vs model resolution through `verdict()`. (2) **E2:** A1.5 `max_batch_size` 1 → 2 or 4. (3) `render_bundling` NONE vs `BATCH_RENDER_RGB`, expected bit-identical (checked with L3-3). Run only the levers whose L3-5 Amdahl bound clears Huy's threshold. **★:** the equivalence margin, the cost model. **A:** speed-up with a CI, plus a paired equivalence verdict. **Premise note:** E4's "identical policy outputs" cannot hold bit for bit. **P:** E2 and E4 preserved verbatim; `P-L3-10: end-to-end sweep speed-up from E4, given L3-5's renderer share: ____`. **K:** E4's preserved kill; a lever is dropped if equivalence fails | 13.1–58.5 · $13–94 | +4–6 | P-005; KC-L. A DECISIONS note only records the premise note, because E4 is already in the S2 card | agent; ★ Huy | Open. NVIDIA requests 1080×1900 and splits 4 renderer : 4 driver GPUs; no resolution trade-off is published | F18, F53–F57; `04` §4, §7 |
| **L3-11** Single-node scaling curve with a capacity-model design doc | D2, I2, X3 (stands in for Q5) | S2 · G7 | (b) | **B:** a design doc: the capacity model from OPERATIONS.md's formulas plus L3-5's per-service times, predicting throughput for k renderer GPUs : m driver GPUs, compared with NVIDIA's PAI 4 : 4 topology. Plus a real measurement: throughput against `n_concurrent_rollouts` ∈ {1, 2, 4, 8} and NRE `--max-workers` ∈ {1, 2, 4}, on 20 scenes × 1 seed, 1–2 GPUs, with per-rollout outputs identical across concurrency levels. **★:** the capacity and cost model. **A:** measured curve vs model, residuals explained. **P:** `P-L3-11: concurrency at which one RTX PRO 6000 saturates: ____`. **K:** none (a budget line) | 3–15 · $3–24 | +4–5 | P-005. Multi-node work stays (c) only | ★ Huy model; agent sweep | Open: OPERATIONS.md gives formulas but no curve | F52, F53, F56 |

### 1a. Recommended minimum set and totals

| Tier | Rows | GPU-h | $ (at $1.00–1.60) | Hours | Gate |
|---|---|---|---|---|---|
| 0 — run S2 so that it measures the simulator | L3-1, L3-3, L3-4, L3-5 | ≈ 0–0.1 | ≈ $0 | +7–9 | none beyond S2's own (KC-L to publish) |
| 1 — measure **and** improve render fidelity cheaply | L3-2, L3-6 (100-scene Harmonizer subset) | 4.2–12.7 | $4–20 | +9–11 | one DECISIONS entry (partial R4/R2) + P-005 |
| 2 — optional cheap probes | L3-7 (100 scenes), L3-11 | 6.0–23.4 | $6–37 | +9–11 | same entry + P-005 |
| 3 — gated closed-loop arms | L3-8, L3-9 (1–2 policies), L3-10 | 81–340 | $81–545 (plus up to $233 if the flipped Harmonizer arm is ON and its cost scales with pixels) | +20–27 | DECISIONS (R2, R3) + P-005 + KC-L |

**Context.**
- S2's card budgets about 40 h and $136–544 for 4 sweeps, with no recorded slack. Tiers 0 and 1 add +16–20 h (≈ 40–50% of S2's hours) for ≤ $20.
- They can only be accepted at KC-1's re-forecast, because the weekly hours (P-003) are unknown.
- Tier 3 can roughly double S2's spend at the top end.
- If Tier 3 is taken, the cheapest arm with the most evidence value is **L3-9 on one policy** ($45–180). It is the only route to the I2 "traffic models" evidence, and it also informs S4's reward-hacking audit (see §4).

### 1b. The six candidate measurements (key question 6), with race status

| # | Candidate | Becomes | Already published by NVIDIA? (2026-10-27) |
|---|---|---|---|
| (i) | Seed reliability | L3-3 (decomposition), on top of S2's planned ICC/W | No. NVIDIA has only a qualitative note on NuRec run-to-run variation (#128). The 1.5 card's "±" is undefined |
| (ii) | Validity envelope by deviation bin | L3-4 | Partly. AR1 and RoaD **exclude** > 4 m. OmniDreams Fig. 14 plots NuRec FVD against mean deviation (no values in the text). The challenge **scores** a ≥ 4 m lateral exit as failure. Nobody publishes per-policy dose–response or a quality_score covariate |
| (iii) | Render-sensitivity probe per deviation bin | L3-6 (perception + KID), L3-7 (policy output) | No NuRec curve with values. DiffusionHarmonizer has one 2 m point; Cosmos3-Nano-Sim-Auto reports LiDAR detection F1 at 3 m, a different object |
| (iv) | Replay vs CAT-K score and ranking agreement | L3-9 (κ and verdict invariance; ranking INSUFFICIENT at n ≤ 3) | No. The nuPlan analogue is IDM-reactive, not learned (CaRL Table 6). The CAT-K seeding PR may collide with NVIDIA's next release |
| (v) | Render at model resolution with an output check | L3-10 (equivalence, not identity) | No |
| (vi) | NuRec vs Cosmos ranking agreement | Rejected for v4 (§5) | Yes in part: OmniDreams reports ranking preserved for n = 4 (qualitative). Cosmos3-Nano-Sim-Auto (10-20) targets closed-loop evaluation, so the race risk is high |

---

## 2. Facts ledger

All accessed 2026-10-27 unless marked "ledger" (meaning carried from `docs/research/2026-10-03-verification/` or `2026-10-25-teacher-to-thor/`, not re-researched). AlpaSim file links are pinned to the SHAs above.

### Q1. NuRec in 2026: renderer, fidelity limits, NVIDIA's metrics and repair tools, no-GT measurement

**Takeaway.**
- AlpaSim still renders with **NRE-GA 26.04** on both branches. The public scenes are **3DGUT** reconstructions.
- NVIDIA's repair line runs from **Difix3D+** to **Fixer** (Cosmos-Predict 0.6B, Nov 2025) to **Harmonizer / DiffusionHarmonizer** (CosmosPredict2 0.6B, temporal). NuRec docs now recommend the Harmonizer over both predecessors, and AlpaSim exposes it through one flag.
- No public number gives NuRec render quality as a function of offset. OmniDreams Fig. 14 is a curve without values in its text. DiffusionHarmonizer reports a single 2 m point on internal scenes.
- Two artifacts each NuRec clip already ships make a no-GT fidelity curve feasible:
  - the 3D labels (ClipGT obstacle and lane parquets) plus calibration;
  - the recorded front-wide MP4.

**Cited findings.**
- **F1.** The default renderer image is `nvcr.io/nvidia/nre/nre-ga:26.04` on `main` (base_config) and on `e2e_challenge` (base_config, `deploy/competition_ec2.yaml`, `e2e_challenge/dev.yaml`). It was bumped to 26.04 on 2026-06-11 (from 26.02 on 2026-03-30). Sources: [e2e base_config.yaml](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/base_config.yaml); [main CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md). P · H.
- **F2.** The renderer flags in `e2e` base_config are `--no-enable-nrend`, `--enable-harmonizer`, `--enable-editing-actors`, and `--enable-timing` (commented out, described as renderer-side timing). `main` base_config has the same flags **without** `--enable-harmonizer`. Sources: [e2e base_config.yaml L130–146](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/base_config.yaml); [main base_config.yaml](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/base_config.yaml). P · H. The meaning of `nrend` is undocumented (gap).
- **F3.** A comment in `e2e_challenge/ec2.yaml` says the developer base config enables the NRE Harmonizer, while the public competition evaluates the unharmonized renderer until separately announced final reruns. Source: [e2e ec2.yaml L32–35](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/e2e_challenge/ec2.yaml). P · H.
- **F4.** Issue #222 (2026-10-22, maintainer reply). Source: [issue #222](https://github.com/NVlabs/alpasim/issues/222). P · H.
  - Final runs will enable the Harmonizer.
  - The internal PAI test set is from a different, harder distribution than the public data.
  - On a renderer OOM, the scene is requeued and retried.
  - The reporting team reproduced locally with NRE 26.04 assets, no Harmonizer, `render_bundling=BATCH_RENDER_RGB` and 12-way concurrency, against the official `8gpu_32rollouts`. Their local val441 `scene_score` ran 0.10–0.15 above the leaderboard.
- **F5.** NuRec docs, "Use Harmonizer for Post-Processing" (last updated 2026-08-25). Sources: [NuRec Harmonizer](https://docs.nvidia.com/nurec/nurec/harmonizer.html) (26.04 archive: [link](https://docs.nvidia.com/nurec/archives/26.04/nurec/harmonizer.html)). P · H (via fetch).
  - It is a diffusion-based enhancer that runs downstream of reconstruction.
  - The backbone is CosmosPredict2, 0.6B.
  - It has a temporal run mode with a temporal total-variation loss against flicker.
  - It is enabled with `--enable-harmonizer` in the NRE gRPC server.
  - The docs recommend it instead of Difix and Fixer.
  - No cost or quality numbers are given, and the page does not say whether geometry is preserved.
- **F6.** DiffusionHarmonizer, arXiv [2602.24096](https://arxiv.org/abs/2602.24096) (v1 2026-02-27, v2 2026-03-05; Zhang, …, Fidler, Gojcic; project page on NVIDIA's lab site). Source: [HTML v2](https://arxiv.org/html/2602.24096v2). P · M-H (via fetch).
  - A single-step, temporally conditioned enhancer built on a 0.6B Cosmos text-to-image backbone.
  - **212 ms on one H100 at 1024×576.** The paper calls this "inference speed" without confirming it is per frame. L3-2 measures the real cost.
  - Novel trajectory at a **2 m** lateral shift (13 internal holdout scenes): FID 120.23 and FVD 470.11, against the best baseline at 134.98 / 506.86.
  - Waymo object insertion at a 3 m shift: FID 101.27, FVD 453.17.
  - The paper claims geometry preservation (DINO-structure 0.92) but does not measure it directly.
  - There is **no offset sweep, no raw-vs-enhanced comparison on the same inputs, and no perception or driving evaluation.**
- **F7.** Fixer (HF `nvidia/Fixer`). Source: [Fixer card](https://huggingface.co/nvidia/Fixer). P · H (via fetch).
  - A single-step image diffusion model (Linear DiT plus DC-AE, based on Cosmos-Predict 0.6B), trained on 3DGUT renders of internal AV images.
  - It has an offline mode (clean pseudo-views, then distil them into 3D) and an online mode (render time).
  - Resolution 576×1024. Latency: H100 26.5 ms, A100 50.6 ms, H20 61.7 ms, L40 87.8 ms.
  - Licence: NVIDIA Open Model License.
  - V2 (Cosmos) was released in Nov 2025; V1 (Stable Diffusion, `nvidia/difix`) in Jun 2025.
  - It is based on Difix3D+ ([arXiv 2503.01774](https://arxiv.org/abs/2503.01774); not fetched).
- **F8.** NVIDIA developer blog, 2025-12-03, "How to Enhance 3D Gaussian Reconstruction Quality for Simulation". Source: [blog](https://developer.nvidia.com/blog/how-to-enhance-3d-gaussian-reconstruction-quality-for-simulation/). P · H (via fetch).
  - It names 3DGS and 3DGUT and shows the Fixer online-mode workflow.
  - Its only number is one PSNR pair on one scene, 16.5809 → 16.6147.
  - A 3 m left-shift example is shown qualitatively only.
- **F9.** NuRec HF card. Source: [NuRec dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) (card only; no files fetched). P · H (via fetch).
  - Version 26.04 is the newest listed, with 1,607 clips. Earlier versions: 26.02; 26.01 (918 processed); 25.05. About 2.9 TB.
  - Reconstruction uses NVIDIA NuRec and cites 3DGUT (CVPR 2025).
  - Six cameras.
  - Each USDZ holds:
    - `checkpoint.ckpt`;
    - `clipgt/` parquets, including `obstacle`, `lane`, `lane line`, `road boundary`, `calibration estimate` and `egomotion estimate`;
    - meshes and `map.xodr`;
    - rig trajectories;
    - first-frame JPEGs for 4 cameras.
  - **The clip folder usually also holds `camera_front_wide_120fov.mp4`.**
  - `clip_ratings_XX.XX.csv` (from 26.02 on) has columns `clip_id`, `quality_score` ("perception based quality score (beta)", with no scale or method given), `time_of_day` and `ego_speed`.
  - Known challenges listed: night, fog, high ego speed, pedestrian legs, certain trailers.
  - Licence §1 internal use; §3 benchmarking results confidential; §4.6 no derivatives or distribution; §8 12-month term.
- **F10.** nurec-skills, per a secondary article. The `nre` skill trains **3DGUT and 3DGRT**, renders novel views locally or over gRPC, and is pinned to NRE `release_26.04`. `nurec-fixer` post-processes, evaluates or fine-tunes renders with DiffusionHarmonizer. The article's date is ambiguous (byline Aug 10, metadata Oct 27, 2026). Source: [radiancefields article](https://radiancefields.com/nvidia-publishes-six-agent-skills-that-drive-the-nurec-neural-reconstruction-stack). S · M.
- **F11.** The NuRec docs index (last updated 2026-08-25) lists no quality-metric tooling beyond a "Run NuRec Validation" link, which I could not reach (404 at the guessed URL). Source: [NuRec docs index](https://docs.nvidia.com/nurec/index.html). P · L-M (an absence).
- **F12.** OmniDreams, arXiv [2606.03159](https://arxiv.org/html/2606.03159), still **v3, 2026-09-23** on the fetched page. Source as linked. P · M-H (via fetch; figure values unreadable).
  - **Fig. 14:** closed-loop AlpaSim trajectories were frozen and replayed through both NuRec and OmniDreams. 4-camera clips were binned by mean distance from the recorded trajectory and scored by FVD against recorded video. NuRec is comparable near the path, then its FVD "rises rapidly". **No bin edges or values are in the text.**
  - **§9.4.1:** a 501-scene PAI NuRec subset (excluding OmniDreams-WAM training scenes), 20 s closed loop, replanning every 533 ms. Policies: WAM and A1.5 with 4, 2 and 1 cameras. The All-Incidents ranking is preserved. No statistics are given.
  - Incidents count only within a distance of GT. The extracted text says "44 m", most likely an extraction artifact for the ledger's 4 m (ledger 4.4).
  - Compute (GB300, 2-step model):
    - 4-view, 16 GPUs: 151 ms per 16-frame chunk, 105 FPS per camera.
    - 4-view, 1 GPU: 1,289 ms per chunk, 12 FPS.
    - Single view, 1 GPU: 118 ms per 8 frames, 68 FPS.
    - Single view, 8 GPUs: 78 ms, 103 FPS.
- **F13.** EUVS (ICCV 2025): translation cuts PSNR by 24.6% on average over 9 methods; **NuRec was not evaluated.** Ledger (`evaluation_release_gating.md` 3.5), [arXiv 2412.05256](https://arxiv.org/abs/2412.05256). P · H.
- **F14.** RoaD (CVPR 2026 Findings): reconstruction quality degrades with distance from the recorded trajectory; it uses the same 4 m exclusion as AR1 (AR1: > 4 m excluded, ledger 4.6). Source: [RoaD](https://openaccess.thecvf.com/content/CVPR2026F/papers/Garcia-Cobo_RoaD_Rollouts_as_Demonstrations_for_Closed-Loop_Supervised_Fine-Tuning_of_Autonomous_CVPRF_2026_paper.pdf). P · M (via search summary, not full text).
- **F16.** AlpaSim `docs/VIDEO_MODEL.md` (main). Source: [VIDEO_MODEL.md L169–190](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/VIDEO_MODEL.md). P · H.
  - OmniDreams runs through FlashDreams behind the same `renderer` endpoint.
  - **The documented recipe supports only single-view generation** (front-wide): `driver=alpamayo1_5_1cam +chunking=8frame`.
  - The video model emits 30 Hz and the driver subsamples by 3.
  - FlashDreams publishes Dockerfiles, not images.
- **F18.** Camera request for Alpamayo on `e2e`: 4 cameras (front-wide, front-tele, cross-left, cross-right) at 1080 × 1900, decoded width 1916 or 1920 because NRE preserves the source aspect ratio; 10 Hz; 30 ms shutter (`driver/alpamayo_configs.yaml` L10–37). Sources: [alpamayo_configs.yaml](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/driver/alpamayo_configs.yaml); [alpamayo_base.py](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/driver/src/alpasim_driver/models/alpamayo_base.py). P · H.
  - The A1.5 driver passes uint8 frames to the VL processor without resizing them itself (`_preprocess_images`).
  - The camera presets are 1cam, 2cam, 2cam_1080, 2cam_2168, 2cam_4k, 4cam_1080 and 6cam_1080. **There is no 320×576 preset.**
- **F60.** The no-GT literature uses FID/FVD at shifted views, because extrapolated-view ground truth does not exist:
  - DHGS uses FID for free-view renders;
  - WorldSplat shifts by ±1, ±2 and ±4 m and computes FID/FVD;
  - RGE-GS uses FID on Waymo cross-lane views;
  - FlexDrive notes that UniSim introduced lane shift.

  No source found validates FID against detector outputs. Sources: [DHGS](https://arxiv.org/pdf/2407.16600); [WorldSplat](https://arxiv.org/pdf/2509.23402); [RGE-GS](https://arxiv.org/pdf/2506.22800); [FlexDrive](https://arxiv.org/html/2502.21093v2). P · M (via search summary).

**Inferences.**
- **I-1. The Harmonizer may be the costliest renderer setting in the stack.** If the paper's 212 ms is per 1024×576 frame on an H100 (F6; unit unconfirmed), 4 cameras per 10 Hz tick add about 0.85 s per step if run sequentially on the same GPU. That rises to about 3 s per step if the cost scales with the 3.5× larger 1080×1900 frame. Both figures are unmeasured on RTX PRO 6000, and the NRE implementation (batching, internal resolution) is unknown. That is why L3-2 comes first.
- **I-2. Self-normalised detector recall controls for the detector's domain gap.** Recall is reported relative to d = 0 on the same rendered clip, and the 3D labels at any ego pose are the simulator's own state (ClipGT), so the novel-view label is known even though the novel-view image is not. Detector recall at d therefore measures how much the renderer degrades the evidence for objects the policy must see. That is closer to the policy's need than FVD.
- **I-3. KID at d = 0 measures reconstruction fit on training views; KID at d > 0 measures novel-view realism.** The difference isolates extrapolation. Scene-content shift at large d confounds the comparison, so KID is a proxy, as the literature acknowledges (F60).
- **I-4. A NuRec-vs-OmniDreams comparison of the deployed policy is not possible in open AlpaSim (F16).** Only the 1-camera A1.5 variant can run on OmniDreams there.

**Gaps.**
- NuRec quality against offset with numbers: OmniDreams Fig. 14 values are not in the text.
- `quality_score` scale and method.
- The meaning of `--no-enable-nrend`.
- Whether the Harmonizer runs per camera frame, and at what internal resolution, inside NRE.
- Whether any NRE release newer than 26.04 exists. 26.04 is the latest referenced, not verified as latest.
- The content of "Run NuRec Validation".

### Q2. Traffic models: AlpaSim's options, realism metrics, effects of reactivity on scores and rankings

**Takeaway.**
- AlpaSim has exactly two traffic modes: log replay (default) and CAT-K (`trafficsim=catk`, NVIDIA's own tokenized SMART-family model, CVPR 2025).
- At `main@affc2eab`, CAT-K is **still unseeded**: the seed is received and logged but never used by the sampler.
- CAT-K is **absent from `e2e_challenge`**, so the challenge protocol, and therefore S2's 441-scene protocol, is replay-only.
- WOSAC's 2025 meta-metric added traffic-light violation and capsule collisions; no 2026 edition was found.
- Published reactive-vs-non-reactive evidence is nuPlan-style, with IDM agents. Imitation-learned planners lose 6.9–8.5 CLS points under reactive traffic, against 0.7 for rule-based or RL planners, and ranks change.

**Cited findings.**
- **F20.** `main` CHANGELOG. Source: [main CHANGELOG L86–100, L137–140](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md). P · H.
  - "SMART/CATK trafficsim integration (29.06.26)", enabled with `trafficsim=catk`.
  - Traffic is explicitly disabled by default (`trafficsim=disabled`).
  - CATK batches requests and limits CPU threading.
  - A new `open_loop_collision` metric.
  - Force-GT warmup exclusion is anchored to the renderer handover.
- **F21. CAT-K is unseeded at code level (main@affc2eab).** Sources: [traffic.proto](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/grpc/alpasim_grpc/v0/traffic.proto); [servicer.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/trafficsim/alpasim_trafficsim/grpc/servicer.py); [rollout.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/trafficsim/alpasim_trafficsim/catk/smart/utils/rollout.py). P · H.
  - `traffic.proto` carries `fixed64 random_seed = 3`, and `servicer.py` logs `request.random_seed` (L213).
  - Token sampling calls `Categorical(logits=topk_logits / temp).sample()` (`catk/smart/utils/rollout.py` L175; also `tokens/token_processor.py` L140, L354).
  - `catk/scene_adapter.py` L519 calls `torch.randperm`.
  - None of these takes a seeded generator.
- **F22. CAT-K is absent on `e2e_challenge@e121e591`.** Sources: [e2e trafficsim dir](https://github.com/NVlabs/alpasim/tree/e121e59141dc391f6c295513d6dadc840a419149/src/trafficsim); [e2e scene_score.py](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/eval/src/eval/aggregation/scene_score.py); [main scene_score.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/aggregation/scene_score.py). P · H.
  - `src/trafficsim/` contains only `pyproject.toml`.
  - base_config's trafficsim command is a stub (`echo 'not yet included'`).
  - There is no wizard `trafficsim/` config.
  - The runtime still has trafficsim client code.
  - `traffic.proto` differs between branches (`scene_id` vs `map_id`, `label_class`, `get_available_scenes`).
  - Conversely, the curated 441 split (`data/scenes/sim_suites_curated.csv`) and the `left_corridor_laterally` failure in `scene_score.py` exist **only on `e2e`**. `main`'s `scene_score.py` lacks the corridor condition.
- **F23.** CAT-K weights `catk_v120` (`config.yaml`, `latest.ckpt`) and token vocabularies are Git LFS pointers (129–133 bytes) under `data/trafficsim-models/` on `main`. The AlpaSim repo is Apache-2.0. **The checkpoint's training data and licence are not stated in AlpaSim's docs.** Sources: [main data/trafficsim-models](https://github.com/NVlabs/alpasim/tree/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/trafficsim-models); [LICENSE](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/LICENSE). P · H for existence; provenance is a gap.
- **F24.** CAT-K paper, "Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models", arXiv [2412.05334](https://arxiv.org/abs/2412.05334) (v2 2025-03-14, CVPR 2025). Authors: Zhang, Karkus, Igl, Ding, Chen, Ivanovic, Pavone. Code at github.com/NVlabs/catk. P · H (via fetch).
  - A 7M tokenized policy fine-tuned with CAT-K beats a 102M one.
  - It held the top WOSAC leaderboard spot at submission.
  - The authors are NVIDIA AV researchers; Igl also maintains AlpaSim, per #198. The NVIDIA affiliation is my inference: the page lists none.
  - The NVlabs/catk licence was not checked (rate limit; gap).
- **F25.** The CAT-K service config on `main` uses 16 history steps, 15 prediction steps, `time_step` 0.1 s, a 100 m filter, and `predict_static: false`. Logged replay is used up to `handover_time_us`; after it, CAT-K must predict, or `simulate` returns `FAILED_PRECONDITION`. Sources: [catk.yaml](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/trafficsim/catk.yaml); [trafficsim README](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/trafficsim/README.md). P · H.
- **F26.** WOSAC 2025. Source: [Waymo Sim Agents 2025](https://waymo.com/open/challenges/2025/sim-agents/). P · H (via fetch). Also the WOSAC paper ([arXiv 2305.12032](https://arxiv.org/abs/2305.12032), ledger).
  - 32 rollouts × 8 s, conditioned on 1 s of history.
  - The meta-metric is a weighted mean of per-feature likelihood scores.
  - Features: linear and angular speed and acceleration; collision; distance to nearest object; TTC (vehicles); offroad; distance to road edge; **traffic-light violation** (vehicles).
  - 2025 changes: the traffic-light metric, a TTC vehicle filter, smoother kinematics, capsule-shaped collisions. Old leaderboards keep their scores.
  - **No 2026 edition is stated.**
- **F28. CaRL Table 6, nuPlan Val14, non-reactive / reactive CLS** (arXiv [2504.17838](https://arxiv.org/html/2504.17838) v3, 2025-08-20). P · H (via fetch).

  | Planner | Non-reactive | Reactive |
  |---|---|---|
  | Log replay | 93.5 | 80.3 |
  | PDM-Closed | 92.8 | 92.1 |
  | PLUTO | 92.6 | 89.7 |
  | PlanTF | 84.6 | 76.1 |
  | Diffusion Planner | 89.6 | 82.7 |
  | CaRL | 91.3 | 90.6 |

  The text says imitation-learned planners struggle with reactive IDM traffic.

  **Derived (my computation):**
  - Kendall τ(non-reactive, reactive) = 0.80 and Spearman 0.90 for n = 5 planners (one swap: PLUTO and CaRL).
  - τ = 0.33 and Spearman 0.37 for n = 6, including the log-replay ego, which falls from rank 1 to rank 5.
  - Reactive-minus-non-reactive deltas: −0.7 (PDM-Closed, CaRL), −2.9 (PLUTO), −6.9 (Diffusion Planner), −8.5 (PlanTF), −13.2 (log replay).
- **F29.** nuPlan-R (arXiv [2511.10403](https://arxiv.org/abs/2511.10403) v1, 2025-11-13) replaces IDM with noise-decoupled diffusion reactive agents. **The abstract gives no ranking numbers.** P · M (abstract only).
- **F30.** Ledger, unchanged:
  - Pseudo-simulation r = 0.89 (83 privileged planners; against nuPlan **reactive** closed loop; rendering not exercised).
  - Sim2Val ρ > 0.9 (per episode, 1 policy).
  - Bosch NAVSIM↔Bench2Drive ρ = 0.905 (n = 8).
  - HUGSIM and Bench2Drive-R give no external coefficient.

  Sources: [pseudo-sim](https://arxiv.org/html/2506.04218v3); [Sim2Val](https://arxiv.org/abs/2506.20553); [Bosch](https://arxiv.org/html/2605.00066v1); [HUGSIM](https://arxiv.org/abs/2412.01718); [Bench2Drive-R](https://arxiv.org/abs/2412.09647). P · H.
- **F31.** Issue #198. A stationary GT tail jitter turned forward overshoot into a "lateral" corridor exit and zeroed `scene_score`. Reported on 2026-10-08. The maintainer confirmed it on 10-09 (fix: extrapolate the final orientation from the last 1 m of movement), and it was fixed on 2026-10-22: metrics recalculated, v1 kept visible frozen. This is the #215 fix. Source: [issue #198](https://github.com/NVlabs/alpasim/issues/198). P · H.
- **F32.** Issue #224 (2026-10-23, maintainer): some simulation choices that affect model inference will be disclosed in the next release; closed as not actionable. Source: [issue #224](https://github.com/NVlabs/alpasim/issues/224). P · H.

**Inferences.**
- **I-5. Replay traffic cannot yield to the ego.** If the ego brakes or deviates, a logged follower drives its logged path. The challenge treats rear collisions as never at fault, so the at-fault endpoint partly hides this artifact, while `collision_any` (the S4 blog reward) does not.
- **I-6. Reactive traffic should move an imitation-learned VLA more than a rule-based planner.** CaRL Table 6 is an IDM-reactive precedent. Alpamayo is imitation-trained plus RL. This is a derivation input, not a prediction.
- **I-7. Any CAT-K arm run on `main` must be compared with a replay arm on the same SHA.** Comparing it with S2's `e2e` replay sweeps would change two variables at once: SHA and evaluator, plus traffic.

**Gaps.**
- The `catk_v120` training data and licence.
- Whether NVIDIA's post-10-31 release seeds CAT-K or brings CAT-K to the challenge branch.
- CAT-K's per-step GPU cost inside AlpaSim (unpublished).
- Realism of CAT-K on NuRec scenes: no WOSAC-style number is published for `catk_v120`.
- nuPlan-R's ranking numbers (full text not read).

### Q3. Cosmos world models as simulators

**Takeaway.**
- **Cosmos3-Nano-Sim-Auto**, released 2026-10-20 under OpenMDW-1.1, is NVIDIA's real-time, closed-loop, multi-sensor driving world model. It targets policy evaluation directly.
- It publishes video-quality metrics and FPS. It publishes **no closed-loop agreement with NuRec or real outcomes and no AlpaSim integration**.
- OmniDreams remains the only published NuRec-vs-world-model closed-loop comparison: ranking preserved, n = 4, no statistics.
- Cosmos-Drive-Dreams is a synthetic-data generator, not a closed-loop simulator.

**Cited findings.**
- **F40.** Cosmos3-Nano-Sim-Auto. Source: [HF card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto). P · H (via fetch).
  - Released 2026-10-20, v1.0.
  - 16B trainable parameters on the card; 15B in BF16 per HF metadata.
  - Use: closed-loop policy evaluation and verification, and synthetic data.
  - Inputs: a text prompt, plus World Scenario Map MP4s per camera (lane markings and cuboids) rendered from a policy's plan or a steering input. Optional first RGB frame and one LiDAR sweep.
  - Outputs: 1–11 Hyperion 8.1 views at 480p or 720p and 10–30 fps, plus LiDAR range and intensity.
  - FPS without LiDAR (vLLM-Omni, 4 steps, no CFG, 201 frames):

    | Views | Resolution | GB200 | H100 | H20 |
    |---|---|---|---|---|
    | 1 | 480p | 16.8 | 10.2 | – |
    | 1 | 720p | 10 | 4.7 | 4 |
    | 7 | 480p | 3 | 1.4 | – |
    | 7 | 720p | 1.3 | 0.58 | – |
    | 11 | 480p | 1.8 | 0.83 | – |
    | 11 | 720p | 0.74 | – | – |

  - Supported: Hopper and Blackwell (B200/B300/GB200/GB300). sm_120 is not mentioned.
  - Licence: OpenMDW 1.1, commercial use allowed.
  - Evaluation against Cosmos-Transfer2.5-2B Multiview (720p, 7 cameras): frame FID 48.6 vs 61.9 (T2V) and FVD 480 vs 772. The baseline wins on DOVER.
  - **No closed-loop policy agreement and no mention of AlpaSim or Alpamayo.** Multi-GPU optimisation is still underway.
- **F41.** The Cosmos Lab page lists the Cosmos3-Nano-Sim family (Bimanual, Transfer, Auto). Auto reaches 16.7 FPS at 480p per view, with 7 RGB views plus LiDAR, on **16 GB200s**. It reports no closed-loop evaluation and states no licence. The technical report is a PDF with no arXiv ID. Source: [Cosmos Lab](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/). P · M (via fetch).
- **F42.** A community repo says FlashAttention-4 has no sm_120 kernels (RTX PRO 6000, RTX 50 series), so Sim-Auto needs cosmos-framework#302 there. Source: [cosmos3-canvas](https://github.com/hakuturu583/cosmos3-canvas). S · L-M (via search summary). A community note records a naming discrepancy, Sim-Auto vs Transfer-Auto: [Robotics_Notebooks PR #2973](https://github.com/ImChong/Robotics_Notebooks/pull/2973) (S · L). An SGLang PR describes Sim-Auto as generating 0.4 s chunks: [sglang PR #39475](https://github.com/sgl-project/sglang/pull/39475) (S · L-M, via search summary).
- **F44.** Cosmos-Drive-Dreams (arXiv [2506.09042](https://arxiv.org/abs/2506.09042) v3, 2025-06-18) is a synthetic-data pipeline built on Cosmos-Drive models, for perception and policy training data. It open-sources a toolkit, dataset and weights. Closed-loop use is not stated. P · H (via fetch).
- **F45.** Ledger: OmniDreams is mid- and post-trained from Cosmos-Predict2.5-2B. Its weights `nvidia/omni-dreams-models` are auto-gated under the NVIDIA Open Model License. The Cosmos 3 report is OpenMDW-1.1. (`alpamayo_models_data.md` 8.8–8.10, 9.4.) P · H.

**Inferences.**
- **I-8. Cost per simulated step: roughly one to two orders of magnitude more GPU than NuRec.** For a 4-camera policy at 480p on an H100, interpolating Sim-Auto between 1 view (10.2 fps) and 7 views (1.4 fps) gives about 2–4 generated multi-view frames per second. At 10 Hz control that is about 2.5–5 GPU-seconds per simulated second, about 6–12 H100-hours per 441 × 20 s seed arm, or about $12–29 at $1.93–2.40 (excluding the policy). NuRec, at "100 FPS at 1080p", needs roughly 0.07–0.4 GPU-seconds per simulated second. Both are rough inferences.
- **I-9. Chunked generation changes the control semantics.** At 0.4 s chunks (Sim-Auto, S) or 533 ms (OmniDreams replanning), control does not run at NuRec's 10 Hz. An agreement study would confound renderer with control rate unless replanning is matched, as OmniDreams did.
- **I-10. NVIDIA built the OmniDreams → AlpaSim path itself (F16) and released Sim-Auto for closed-loop evaluation.** A Sim-Auto adapter for AlpaSim would likely race an NVIDIA artifact.

**Gaps.**
- Sim-Auto's measured FPS for 4 views on any GPU.
- Whether sm_120 works without the patch.
- Any NVIDIA plan for an AlpaSim adapter.
- Closed-loop agreement numbers for any Cosmos simulator.

### Q4. Sim-to-sim and sim-to-real agreement: methodology, and how much n a decidable claim needs

**Takeaway.**
- The published coefficients (pseudo-sim r = 0.89; Bosch ρ = 0.905, n = 8; Sim2Val ρ > 0.9 per episode, 1 policy) are all sim-to-sim or single-policy (ledger).
- **Policy-level rank agreement needs about 10 or more policies even at an observed ρ = 0.9, for a one-sided 95% lower bound of at least 0.6.** With S2's 3 policies it is undecidable.
- The decidable statistics at S2 scale are **scene-level**: κ on pass/fail and paired score differences across 441 scenes, road-clustered, by deviation bin. The gate-native check is **decision-level**: does S2's BF16-vs-FP8 verdict change when the renderer or traffic changes?

**Cited findings and derivations.** All derivations are mine (`l3_calc.py`), using Fisher-z approximations that are crude for n ≤ 10.
- **F70.** Under H0, P(τ = 1) = 1/n!: 0.167 (n = 3), 0.042 (n = 4), 0.0083 (n = 5), 0.0014 (n = 6). With 3 policies, perfect agreement is not even significant against "no association".
- **F71.** One-sided 95% lower bound for an observed Spearman ρ (Bonett–Wright, var z ≈ (1 + ρ²/2)/(n − 3)):

  | ρ observed | n = 6 | n = 8 | n = 10 | n = 15 | n = 20 | n = 30 |
  |---|---|---|---|---|---|---|
  | 0.8 | 0.01 | 0.25 | 0.37 | 0.50 | 0.57 | 0.63 |
  | 0.9 | 0.33 | 0.54 | 0.63 | 0.72 | 0.76 | 0.80 |
  | 0.95 | 0.60 | 0.74 | 0.79 | 0.85 | 0.87 | 0.90 |

  The minimum n for a lower bound of at least 0.6 is 15 at ρ = 0.85, 10 at 0.9 and 7 at 0.95. This is consistent with v3's n ≈ 28–34 for a decidable PASS/KILL at a 0.6 cut.
- **F72.** Kendall (Fieller, var z ≈ 0.437/(n − 4)): at τ observed = 0.8 the one-sided lower bound is 0.32 / 0.50 / 0.57 / 0.65 / 0.68 / 0.71 for n = 6 / 8 / 10 / 15 / 20 / 30.
- **F73.** Ledger 4.17: degraded variants of one parent (ladders) are not independent policies. They inflate ρ through range and share failure modes, so report within-family and between-family correlation separately. S3, S5 and S6 ladder checkpoints therefore do not rescue a rank-agreement n.
- **F74.** Ledger: the independence hierarchy, from least to most independent: the same rollouts, the same scenes with a different renderer (Harmonizer, OmniDreams), a different paradigm.

**Inferences.**
- **I-11. The three agreement statistics the v3 R2–R4 text already specified fit S2 directly:** κ by deviation bin, Bland–Altman, and τ with clustered CIs. Use them **scene-level**, and report "rank agreement: INSUFFICIENT (n = 3)" explicitly instead of a τ.

**Gaps.**
- The design effect of road clustering on κ's SE for the 441 split. It needs S2's road IDs: a ★ Huy derivation input.

### Q5. Simulation performance: levers, published figures, telemetry

**Takeaway.**
- **No AlpaSim throughput figure is published** (re-checked 2026-10-27). NuRec's "100 FPS at 1080p, 6 cameras" remains ambiguous.
- AlpaSim's levers are configuration-level:
  - renderer replicas and `--max-workers`;
  - `n_concurrent_rollouts`;
  - `render_bundling`;
  - driver `max_batch_size` (1 for A1.5);
  - the camera request (1080×1900 for Alpamayo);
  - the Harmonizer on or off.
- NVIDIA's own PAI challenge topology gives **4 GPUs to rendering and 4 to drivers** for 32 concurrent rollouts.
- Telemetry exists on both branches, under **different metric names**.

**Cited findings.**
- **F50.** TELEMETRY.md (**main only**). Source: [TELEMETRY.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/TELEMETRY.md). P · H.
  - Prometheus is on by default, with `metrics_plot.png` at the end of a run and data persisted under `prometheus/data`.
  - File-SD discovery for a central Prometheus.
  - On Slurm, `alpasim_gpu_workload_info{gpu}` joins to DCGM.
- **F51.** Metric names per branch. P · H.
  - `main` `telemetry_context.py` defines 18 series: `alpasim_rpc_duration_seconds{service,method,worker}`, `alpasim_rpc_blocking_seconds`, `alpasim_rpc_queue_depth_at_start_latest`, `alpasim_rollout_duration_seconds`, `alpasim_step_duration_seconds`, `alpasim_observation_barrier_seconds`, `alpasim_rollouts{status}`, `alpasim_renderer_active_rollouts`, `alpasim_renderer_rollouts_started{kind}`, `alpasim_renderer_scheduler_wait_seconds{kind}`, `alpasim_simulation_elapsed_seconds`, `alpasim_job_queue_depth`, three event-loop counters and three GC metrics.
  - `e2e` `telemetry_context.py` defines 14 **unprefixed** series, including `rpc_duration_seconds{service,method,tag}`, `step_duration_seconds`, `rollout_duration_seconds`, `rpc_queue_depth_at_start`, `simulation_seconds_per_rollout`, `simulation_total_seconds` and `simulation_rollout_count`, plus event-loop and GC metrics. It has no renderer-scheduler or observation-barrier series.
  - Sources: [main telemetry_context.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/runtime/alpasim_runtime/telemetry/telemetry_context.py); [e2e telemetry_context.py](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/runtime/alpasim_runtime/telemetry/telemetry_context.py).
- **F52.** OPERATIONS.md (main). Source: [OPERATIONS.md L39–102, L338–379](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md). P · H.
  - Capacity = `nr_gpus × replicas_per_container × n_concurrent_rollouts`.
  - Renderer capacity = `nr_gpus × max_workers × n_concurrent_rollouts`.
  - OSS configs prefer `replicas_per_container: 1` with a higher `--max-workers`.
  - Diagnostics: high queue depth means add replicas or concurrency; idle below 20% is good; idle above 80% points to a bottleneck.
  - `metrics_plot.png` shows idle fraction and simulated seconds per rollout.
- **F53.** The PAI EC2 topology `8gpu_32rollouts` is a "renderer-stability topology". Source: [8gpu_32rollouts.yaml](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/topology/8gpu_32rollouts.yaml). P · H.
  - Renderer: 4 GPUs × 8 replicas × 1 rollout.
  - Driver: 4 GPUs × 4 × 2.
  - Physics: 4 × 2 × 4.
  - Controller: a CPU service, 8 × 4.
  - Total: 32 concurrent rollouts; `nre_cache_size` 2.
- **F54.** `RenderBundling` takes the values `NONE` (default: one `render_rgb` per camera), `BATCH_RENDER_RGB` (NRE batched; used in PAI `ec2.yaml` L58) and `RENDER_AGGREGATED` (nuPlan). Sources: [runtime config.py L268–277](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/runtime/alpasim_runtime/config.py); [ec2.yaml](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/e2e_challenge/ec2.yaml). P · H.
- **F55.** Driver `max_batch_size` is 1 for A1.5 (commented "memory intensive"), 1 for A1 and 32 for VaVAM. Source: [alpamayo1_5.yaml](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/driver/alpamayo1_5.yaml). P · H.
- **F56.** The challenge README sets a 0.1 s target per `Drive` call, enforced through a per-track throughput budget. Submission status reports the observed throughput wall time. EC2 runs 16 replicas × 2 concurrent rollouts. Source: [e2e_challenge/README.md L160–173](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/e2e_challenge/README.md). P · H.
- **F57.** Ledger: NuRec's "100 FPS at 1080p, 6 cameras, RTX6000Pro" is ambiguous between per-camera frames and ticks ([NuRec scaling docs](https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html)). P · H.
- **F58.** No AlpaSim technical report and no official rollouts-per-GPU figure were found. AlpaGym says only that work is focused on throughput. A Japanese blog reports that A1.5 runs took about 5× longer than VaVAM. Sources: [AlpaGym](https://github.com/NVlabs/alpagym); [Classmethod](https://dev.classmethod.jp/articles/nvidia-alpasim/). P for the absence · M; S for the 5×.
- **F59.** Determinism on `main`. Sources: [schema.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/driver/src/alpasim_driver/schema.py); [main CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md). P · H.
  - The driver schema has `force_determinism: bool = False` ("Alpamayo models only").
  - The CHANGELOG says the public Alpamayo 1 driver can derive deterministic per-inference seeds from the session seed.
  - **`e2e_challenge` has no `force_determinism`.**

**Inferences.**
- **I-12. NVIDIA's 1 : 1 renderer-to-driver GPU split at 4 × 1080p implies rendering is not negligible** next to A1.5-class drivers. That makes E4 plausible as a lever, but L3-5 must measure it. E4's preserved prediction ("≥ 4× scenarios per renderer GPU") is a renderer-only claim; the end-to-end gain is Amdahl-limited.
- **I-13. Pixels do not bound a 3DGS frame's cost.** Projection and sorting scale with the number of visible Gaussians, so the renderer speed-up k from 11.25× fewer pixels is below 11.25.

**Gaps.**
- The measured renderer share at 4 × 1080p.
- CAT-K's per-step cost.
- The Harmonizer's per-step cost inside NRE.
- Whether NRE renders deterministically.
- Whether A1.5 accepts sub-320×576 renders. `base.py`'s `_resize_and_center_crop` raises when an image is too narrow, but the A1.5 path does not call it, so this is unverified.

### Q6. Cheap S2 measurements for D3/I2, and race checks

**Takeaway.** See §1b. Only (vi) is substantially raced. The rest are open as of 2026-10-27, but L3-8 (Harmonizer) and the CAT-K seeding PR in L3-9 carry a moderate risk that NVIDIA's post-challenge release or technical reports pre-empt them. Re-check on S2's start date (11-02).

**Cited findings.** F3, F4, F12, F16, F22, F31, F32, F40 and F53, as above.

**Gaps.** NVIDIA's post-10-31 release contents (#224) and the challenge technical reports (November).

---

## 3. Deltas against the earlier ledger (dated 2026-10-27)

1. **CAT-K seed (04 §5; 01 C4; ledger tooling #48). Still unseeded, now confirmed at code level.** `random_seed` is received and logged (servicer L213); sampling uses the unseeded global RNG (F21). `main` is unchanged since 2026-08-18, so the 10-03 and 10-25 statements hold. Source: F21.
2. **New: CAT-K is absent on `e2e_challenge` (e121e591, 2026-10-20).** The challenge, and S2's 441-scene protocol, is replay-only. The curated split and the corridor scorer exist only on `e2e`, and `traffic.proto` differs between branches. Earlier ledgers described CAT-K as "optional" without naming a branch. Source: F22.
3. **Harmonizer defaults.** Ledger `evaluation_release_gating.md` §1 said OFF for initial runs and ON for PAI final runs. Refinement: `e2e` base_config (developer default) is **ON**, `main` base_config is OFF, and the EC2 competition preset is OFF until the final reruns. The maintainer reconfirmed ON for the final runs on 2026-10-22 (#222). Sources: F2–F4.
4. **The repair model changed.** NuRec docs (2026-08-25) recommend the Harmonizer (CosmosPredict2 0.6B, temporal) instead of Difix and Fixer. DiffusionHarmonizer measured 212 ms per frame on an H100 at 1024×576 (arXiv 2602.24096 v2, 2026-03-05). Fixer, its predecessor, runs at 26.5 ms on an H100. Sources: F5–F7.
5. **Renderer version.** NRE-GA 26.04 is still the default on both branches. The NuRec card's newest listed version is still 26.04, and nurec-skills pins `release_26.04`. I did **not** verify that no newer NRE exists: "26.04 is the latest referenced". Sources: F1, F9, F10.
6. **NuRec card details beyond ledger 8.4.**
   - Each clip folder usually ships `camera_front_wide_120fov.mp4` (the recorded video), which enables a distributional fidelity metric.
   - `clip_ratings` columns are `quality_score` (perception-based, beta), `time_of_day` and `ego_speed`, present from 26.02 on.
   - Known hard cases: night, fog, high ego speed, pedestrian legs, trailers.

   Source: F9.
7. **OmniDreams in open AlpaSim is single-view only.** The documented recipe uses `alpamayo1_5_1cam`. The ledger said "OmniDreams behind the same renderer endpoint", which is true, but the deployed 4-camera A1.5 cannot be compared on it. OmniDreams is still v3 (2026-09-23). Sources: F12, F16.
8. **New model: Cosmos3-Nano-Sim-Auto (2026-10-20, OpenMDW-1.1).** It does closed-loop multi-view and LiDAR generation, runs about 1.4 fps at 7 views / 480p on an H100, and reports no closed-loop agreement. Sources: F40–F42.
9. **Telemetry.** TELEMETRY.md exists only on `main`. The metric names differ: `e2e` has 14 unprefixed series, `main` has 18 `alpasim_*` series. The ledger's "Prometheus default since 2026-06-30" stands. Dashboards for S2 (which runs on `e2e`) must use the `e2e` names. Source: F51.
10. **Throughput configuration facts (new).**
    - The PAI EC2 topology uses 4 renderer GPUs to 4 driver GPUs for 32 concurrent rollouts.
    - It sets `render_bundling: BATCH_RENDER_RGB`.
    - A1.5 runs at `max_batch_size: 1`.
    - Alpamayo requests 4 × 1080×1900, and there is no model-resolution camera preset.

    The ledger's "100 FPS" ambiguity remains. Sources: F18, F53–F55.
11. **Evaluator.** The corridor scorer bug #198 was fixed on 2026-10-22 (#215): metrics recalculated, the frozen v1 still visible. Validity-record semantics must name the evaluator SHA. Earlier ledgers cited #215 as a "mid-challenge fix" without its mechanism (GT tail jitter turning forward overshoot into a lateral exit). Source: F31.
12. **Determinism.** `force_determinism` (Alpamayo, per-inference seeds from the session seed) exists on `main` only. Whether the A1.5 driver on `e2e` is seeded by `DriveSessionRequest.random_seed` is unverified, which matters for S2's seed test–retest design. Source: F59.
13. **WOSAC 2025 metric changes** (traffic-light violation, TTC vehicle filter, smoother kinematics, capsule collisions); no 2026 edition found. The ledger cited only WOSAC 2023's likelihood framing. Source: F26.
14. **Reactive vs non-reactive evidence (new).** CaRL Table 6, nuPlan Val14: imitation-learned planners lose 6.9–8.5 CLS under IDM-reactive traffic, against 0.7 for PDM-Closed and CaRL. My derived τ is 0.80 (n = 5) and 0.33 (n = 6 including the log-replay ego). Source: F28.
15. **No contradiction, but a nuance on `04` §4** ("Alpamayo drivers reject images smaller than 320×576"). The rejection path exists in `base.py`, but the A1.5 path hands full frames to the VL processor (F18). The claim is unverified for A1.5 at 320×576, so E4 must test it.
16. **#224 wording (verification B3).** It is consistent: the maintainer's "next release" remark appears in #224 (a stationary-lead collision report), dated 2026-10-23. Source: F32.

---

## 4. First-principles rationale per finding

**L3-1, pin the full configuration.**
- **Mechanism.** A closed-loop score is a function of (policy, scenes, seeds, renderer image, renderer flags, traffic mode, camera request, evaluator). Today the S2 manifest pins only part of that tuple.
- **Evidence it bites.** The Harmonizer flag alone differs between the developer default, the competition preset and the final reruns (F2–F4). The evaluator changed on 10-22 (F31). #222 shows a 0.10–0.15 local-vs-leaderboard gap that came partly from distribution and partly from configuration (F4).
- **Why not the alternative.** Without the full tuple, every later agreement claim (L3-6 to L3-9) is confounded.
- **Cost.** One hour of manifest plumbing.
- **Decision input for Huy.** The Harmonizer state of the pinned protocol. ON gives parity with the challenge's final runs. OFF gives cost and comparability with team-zyt-style local runs. L3-2 supplies the price.

**L3-2, Harmonizer cost pilot.**
- **Mechanism.** If the paper's 212 ms is per frame on an H100 (F6; unit unconfirmed), the Harmonizer could cost as much per step as the A1.5 policy itself. If it is pinned ON for all 4 S2 sweeps, the inferred exposure is +249–873 GPU-h, or **$249–1,396**, against S2's $136–544.
- **Why this beats deciding a priori.** One ≤ $2 pilot with renderer `--enable-timing` replaces a factor-of-3 uncertainty, from per-frame vs batched execution and from resolution scaling, before the budget tier (P-005) is spent.
- **Alternative rejected.** Copying the paper's 212 ms: different GPU (H100 vs RTX PRO 6000), different resolution, and unknown NRE batching.

**L3-3, noise decomposition.**
- **Mechanism.** Seed-to-seed discordance has three possible sources:
  - (1) renderer nondeterminism (#128 admits "small" variation);
  - (2) policy sampling (flow-matching noise and CoC sampling);
  - (3) traffic stochasticity, which is zero under replay.

  Closed-loop dynamics amplify any tiny difference: once ego poses diverge, every later render differs. Two cheap probes separate the sources:
  - rendering the *same pose twice* tests (1) directly;
  - the divergence-onset tick in paired-seed logs shows *when* trajectories separate.
- **Why it beats ICC alone.** ICC(2,1) says how big the noise is, not where it comes from. A sim-team reviewer cares about the source, because the fix differs: deterministic kernels for the renderer, a seeded generator for the policy or for CAT-K.
- **Cost.** ≈ $0 if the probe runs inside a sweep job (the renderer already has the scene loaded), +2–3 h.
- **Derivation inputs for P-L3-3.** NuRec's "small" run-to-run variation (#128); replay traffic is deterministic; A1.5 seeding on `e2e` is unverified (F59).

**L3-4, envelope as a dose–response curve, with NVIDIA's own quality score.**
- **Mechanism.** If rendering degrades with deviation (OmniDreams Fig. 14, RoaD), failures in high bins mix policy error with renderer error.
- **What it adds to the deviation bins S2 already has:**
  - (a) the curve's *shape* per policy, which shows whether failure rises faster than the policy's own low-deviation failure rate;
  - (b) a covariate NVIDIA already computes, the per-clip perception-based `quality_score`. Whether it predicts validity exits and failures is a direct, cheap test of NVIDIA's own simulation-quality metric against closed-loop outcomes;
  - (c) an endpoint-overshoot class, so post-GT-end motion is not mislabelled (#198).
- **Why it beats re-tuning the 4 m cap.** It produces the evidence a cap needs. The cap itself stays Huy's tolerance.
- **Cost.** $0, +2 h.
- **Coverage caveat.** Ratings exist from 26.02 on, but the 441 split draws on 26.01 and 26.04, so some scenes may lack a rating. State the coverage.

**L3-5, critical-path profile and Amdahl bounds.**
- **Mechanism.** End-to-end speed-up from a lever that accelerates a fraction f of step time by k is S = 1/((1−f) + f/k).
  - E4 can at most remove the renderer's share; batching, the driver's; bundling, the per-camera RPC overheads.
  - AlpaSim already exports per-service RPC histograms (F51), so f per service can be measured on S2's own sweeps at no cost.
- **Why it beats running every lever.** Each lever arm costs a sweep, while the bound is free. The diagnosis is a ★ core: Huy's cost model.
- **Cost.** $0, +2–3 h.

**L3-6, render-fidelity curve without novel-view GT.**
- **Mechanism.** At a novel pose the true *image* is unknown, but the true *scene state* is known: AlpaSim's own 3D labels (ClipGT obstacles and lanes) and the camera calibration.
  - So the "evidence the policy needs" can be checked: can a fixed off-the-shelf detector still find the projected objects?
  - Normalising by the same detector at d = 0 cancels its real-vs-render domain gap.
  - KID against the recorded front-wide video adds a distributional view, anchored at d = 0, which measures reconstruction fit.
  - Running both with the Harmonizer OFF and ON measures whether NVIDIA's Cosmos-based enhancer **improves** fidelity where it matters, at large d. That is the I2 "improve … using … the Cosmos world model" evidence, obtained with one flag.
- **Why it beats the alternatives:**
  - held-out cameras need re-reconstruction from raw PAI-AV sensor data (licence §4.6, compute);
  - FVD/FID alone cannot say whether objects survive (F60 notes FID is not validated against detection);
  - PSNR at training views is fit, not generalisation.
- **Cost.** $4–18 with a 100-scene Harmonizer subset, +8–10 h.
- **Derivation inputs (not predictions):**
  - EUVS: −24.6% PSNR for lane-change-scale translation, averaged over 9 non-NuRec methods;
  - OmniDreams: NuRec FVD "rises rapidly" with deviation;
  - DiffusionHarmonizer: FID 120 at 2 m with the enhancer, no raw baseline;
  - geometry: the parallax angle to an object at range r under lateral offset d is atan(d/r). At r = 20 m that is about 2.9° (d = 1 m), 5.7° (2 m) and 11.3° (4 m), so near objects extrapolate first.

**L3-7, policy-sensitivity probe.**
- **Mechanism.** The question that matters for a gate is not whether the image looks different but whether the policy *decides* differently because of a render artifact.
  - Holding pose fixed and changing only appearance (Harmonizer ON vs OFF) isolates artifact sensitivity from legitimate geometric response.
  - Dividing by the policy's own seed-to-seed plan spread gives a dimensionless ratio. A ratio near or below 1 means the renderer's appearance is inside the policy's noise.
- **Why it beats closed-loop A/B first.** It costs about 1/10 of a closed-loop arm, and it shows *where* (which deviation bin) a closed-loop A/B would disagree.
- **Known approximation (inference).** The 4-frame history is rendered along a statically shifted log path, which approximates a real deviating trajectory.
- **Cost.** $3–13 for 100 scenes, +5–6 h.

**L3-8, minimal R2 (closed loop, Harmonizer).**
- **Mechanism.** The release gate is only as good as its invariance to renderer post-processing.
- **Decidable statistics:** κ across 441 scenes, by deviation bin, and decision invariance of S2's BF16-vs-FP8 verdict. Rank τ over 3 policies is not decidable (§2 F70).
- **The floor.** A flip rate between renderer states means nothing alone. It is read against S2's seed-vs-seed κ under one fixed configuration in the same bin, which S2 produces for free. Only disagreement beyond that floor is attributable to the Harmonizer.
- **Why only the Harmonizer, not OmniDreams.** The Harmonizer runs on the plan's RTX PRO 6000 and on the 4-camera policy. OmniDreams in open AlpaSim cannot run the 4-camera policy (F16).
- **Cost.** $23–90, plus the Harmonizer overhead if the flipped arm is ON (measured in L3-2).

**L3-9, minimal R3 (replay vs CAT-K).**
- **Mechanism.** Replay agents cannot respond to the ego. When the ego slows, deviates or stops, the scene becomes physically implausible: followers rear-end it and cut-ins ignore it. The nuPlan evidence (F28) shows that reactive traffic re-orders imitation-learned planners and costs them 6.9–8.5 CLS, against 0.7 for planners robust to reaction.
- **Design rules from the repo facts:**
  - **one SHA, one evaluator, only `trafficsim` toggled.** Otherwise SHA, evaluator and traffic change together (F22);
  - CAT-K is unseeded (F21), so either measure its own variability with 3 repeats or seed it upstream. The PR is small plumbing that NVIDIA itself invited (#128), but check the post-10-31 release first (#224).
  - **The floor.** As in L3-8, replay-vs-CAT-K flips are read against S2's seed-vs-seed κ per bin. The single replay arm on `main` borrows S2's `e2e` replay repeatability. That is an assumption, stated in the report, and it is checkable later if a second replay seed is affordable.
- **Why it matters beyond S2 (cross-lane note, inference).** S4's reward uses `collision_any` (−10). Under replay, rear-end collisions from non-yielding followers can penalise legitimate braking, a sim-artifact incentive that the S4 reward-hacking audit should know about. Running L3-9 before S4's reward ablation tells S4 whether this artifact is material.
- **Cost.** $45–180 for one policy, +10–13 h.

**L3-10, performance levers with an equivalence guard.**
- **Mechanism.** E4, E2 and render bundling are configuration changes that can save GPU time. Each can also change numerics:
  - rasterising at 320×576 is not the processor's resampling of a 1080p frame;
  - batched GEMMs differ from unbatched ones in floating point.

  So the acceptance test must be **paired equivalence through the gate** (`verdict()` under Huy's margin), not identity. This needs a premise note on the preserved E4 text; the text itself is not edited.
- **Ordering.** Run each lever only if L3-5's Amdahl bound says it can pay.
- **Why it impresses a sim-team reviewer.** "Faster" is only claimed together with "and the verdicts did not change". That is exactly the accuracy-and-performance coupling in I2.
- **Cost.** $13–94, +4–6 h.

**L3-11, single-node scaling with a capacity model (b).**
- **Mechanism.** OPERATIONS.md gives capacity formulas, and NVIDIA's own challenge topology encodes a 4 : 4 GPU split. A queueing and capacity model fitted to L3-5's per-service times predicts the best split and concurrency.
- **The stand-in measurement.** A 1–2 GPU sweep over concurrency and NRE workers validates the model. It replaces the multi-node scaling the plan cannot do; multi-node stays (c).
- **Built-in determinism test.** Per-rollout outputs must be identical across concurrency levels, which ties back to L3-3.
- **Cost.** $3–24, +4–5 h.

---

## 5. Rejected options, with reasons

1. **NuRec vs Cosmos (OmniDreams / Cosmos3-Nano-Sim-Auto) policy-ranking agreement, candidate (vi).**
   - *Raced and duplicative.* OmniDreams already published ranking preservation (n = 4; F12). Sim-Auto (2026-10-20) is built for closed-loop evaluation, and an AlpaSim adapter is an artifact NVIDIA is likely to ship (I-10).
   - *Infeasible on plan hardware.* Open AlpaSim's video-model path is single-view only (F16). Sim-Auto has no AlpaSim adapter, needs WSM control videos, and lacks sm_120 attention kernels without a patch (F42). It runs at about 1.4 fps for 7 views on an H100 (F40).
   - *Undecidable at n = 3.*

   Revisit only if NVIDIA ships a multi-view world-model renderer for AlpaSim **and** the panel reaches ≥ 10 independent policies (F71).
2. **Held-out-camera reconstruction fidelity.** Public NuRec scenes are trained on all 6 cameras (F9). Holding one out needs re-reconstruction from raw PAI-AV sensor data with NRE training: heavy GPU, a derivative work under licence §4.6, and unpublishable.
3. **Improving the scenes themselves (Fixer offline re-distillation, Instant NuRec re-reconstruction).** Same reasons as item 2: raw data, derivatives, compute. The *online* Harmonizer is the in-budget "improve" lever (L3-6, L3-8).
4. **WOSAC-style realism of `catk_v120` on NuRec scenes.** It would directly evidence "traffic model quality", but:
   - porting the WOSAC metric code to ClipGT is about 10–16 h;
   - `catk_v120`'s training data and licence are unknown (F23);
   - traffic realism is not the gate-relevant question (verdict invariance is).

   Keep it as an open option after L3-9.
5. **Policy-level τ/ρ between renderers or traffic models on S2's 3 policies.** Undecidable (F70–F71). Adding ladder checkpoints does not fix this (F73).
6. **Calibrating the 4 m cap to maximise agreement on the evaluated policies.** Circular: it tunes on the validation policies, which v3 forbade. The cap is Huy's pre-registered tolerance, informed by L3-4 and L3-6.
7. **Using the local-val441-vs-leaderboard gap (#222) as sim-to-sim evidence.** The internal test set is a different, harder distribution (F4), and the data are not paired.
8. **PSNR/SSIM on training views as "render fidelity".** It measures reconstruction fit, not novel-view generalisation. It is kept only as the d = 0 anchor of L3-6.
9. **LiDAR realism probes.** Alpamayo is camera-only. They are irrelevant to the gate's policies.
10. **Cross-paradigm R6 (28–34 checkpoints) and private set R5.** Deferred by D-012, collaborator-scale and outside S2.
11. **Upstream changes to NRE or the Harmonizer.** The renderer is a closed NGC container. Only AlpaSim accepts outside PRs, and none has merged since 2026-08-17.

---

## 6. Open questions, and what would resolve them

| # | Question | Why it matters | What resolves it |
|---|---|---|---|
| 1 | The Harmonizer's real per-step cost inside NRE at 4 × 1080×1900 on RTX PRO 6000. Per frame or batched? At what internal resolution? | Decides S2's pinned protocol and could move S2's cost by $249–1,396 (inference) | L3-2 pilot with `--enable-timing` (≤ $2) |
| 2 | Does NVIDIA's post-10-31 release unify `main` and `e2e` (CAT-K, corridor scorer, curated split), seed CAT-K, or disclose the withheld "simulation choices" (#224)? | Changes L3-9's porting hours and its race status; may change S2's protocol | Re-read AlpaSim on 11-02 (S2 already requires this) |
| 3 | `catk_v120` training data and licence (WOMD vs PAI-AV vs internal) | Publication terms of any CAT-K-derived number | Read the NVlabs/catk licence and AlpaSim docs after the rate limit resets; otherwise an AlpaSim issue question (Huy's outward action) |
| 4 | Is the A1.5 driver on `e2e` seeded by `DriveSessionRequest.random_seed`? There is no `force_determinism` on `e2e` | S2's seed test–retest semantics; L3-3 and L3-7's noise floor | Code read of the `e2e` A1.5 inference path at S2 start; L3-3's policy probe |
| 5 | Does NRE render deterministically for an identical request? | The renderer's variance component | L3-3 render-determinism probe |
| 6 | `quality_score`: scale, method, and coverage for 26.01-derived scenes in the 441 split | Usability of the L3-4 covariate | Inspect the CSV after S2's download. Ask NVIDIA only if needed |
| 7 | Does A1.5 accept model-resolution renders? Does NRE preserve aspect so that 320×576 comes back as about 320×568? | E4's feasibility | L3-10's open-loop screen; decoded sizes logged in the manifest |
| 8 | What does NuRec's "100 FPS at 1080p, 6 cameras" count? What is the renderer share of step time? | Sweep costs and every Amdahl bound | L3-5 telemetry (`rpc_duration_seconds{service="renderer"}`) |
| 9 | OmniDreams Fig. 14 bin values; the "44 m" in the §9.4.1 extraction (likely 4 m) | Precedent comparison for L3-6 | Read the PDF figure directly; not needed for L3-6's own measurement |
| 10 | Will NVIDIA publish Harmonizer ON-vs-OFF agreement from the challenge's initial vs final runs? | Race risk for L3-8 | Check the challenge results and technical reports in November 2026 |
| 11 | The meaning of NRE `--no-enable-nrend` | Manifest completeness (L3-1) | NRE gRPC API guide in the NuRec docs |
| 12 | Cosmos3-Nano-Sim-Auto: measured 4-view FPS; sm_120 status; any AlpaSim adapter plan | The condition for revisiting rejected item (vi) | Watch cosmos-framework #302 and the AlpaSim releases |
| 13 | nuPlan-R's numbers on learned-reactive vs IDM vs replay rankings | A second precedent for L3-9's derivation inputs | Read the nuPlan-R full text |
| 14 | Road-cluster design effect for κ on the 441 split | κ CIs in L3-8 and L3-9 | ★ Huy's derivation from S2's road IDs |
| 15 | Is any NRE newer than 26.04 available, and is the "Run NuRec Validation" tooling usable without NGC sign-in? | Renderer pin; extra fidelity metrics | Read the NuRec release notes (public docs) at S2 start; no sign-in |
| 16 | Is `camera_front_wide_120fov.mp4` really present for the 441 scenes? The card says only "usually" | L3-6's KID reference distribution | Verify at S2's download. If absent, fall back to the four first-frame JPEGs in each USDZ for KID's d = 0 anchor and say so |
