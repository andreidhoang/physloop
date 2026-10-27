# 02 — PhysLoop plan v4.1 (27 October 2026)

**Status:** authoritative (see `../DOC_AUTHORITY.md`). It amends v4 of 2026-10-25, which is archived in `archive/02_PHYSLOOP_PLAN_v4_2026-10-25.md`. Adopted by DECISIONS D-015; v4 itself was adopted by D-012. PhysLoop is a standalone, active project (D-009) and takes no hours, budget, deadline or decision from any other program.

**Evidence base:**
- `research/2026-10-27-role-alignment/`: seven research lanes (L1–L7), the synthesis report, the verification ledger and the arithmetic check behind v4.1;
- `research/2026-10-25-teacher-to-thor/`: the owner's decision input, the review, the verification ledger and the arithmetic check behind v4;
- `research/2026-10-03-verification/`: still valid wherever the later ledgers do not supersede it.

Inferences are labelled as inferences. The predictions, floors, margins and tolerances are Huy's: every slot marked `____` stays blank until he fills it in `../experiments/` before the run. A tag such as (L7-2) points to lane 7, row 2, of the 2026-10-27 research notes.

---

## 0. What changed

### 0.1 v4 → v4.1 (2026-10-27): corrections, savings, and evidence for the target role

**The request.** On 2026-10-27 Huy asked for the plan to be optimized so that, finished end to end, it is the strongest checkable evidence of exceptional engineering and research skill for NVIDIA's JR2010863, Senior Machine Learning and Simulation Engineer – Autonomous Vehicles (AV Simulation team). Years, degree and location are out of scope by design. Seven research lanes checked every stage against NVIDIA's 2026 stack. The line-by-line map from the posting to the stages is in `05_ROLE_MAPPING.md` §3.

**Three rules bound the amendment:**
1. **No new stage and no new framework.** Every addition is a measurement inside an existing stage, or a design document validated by a measurement. AlpaGym is used as shipped, through configuration only. Anything that adds spend beyond a stage's band, or needs more than one node, is a proposal (P-012 onward). Nothing is built only to signal.
2. **Corrections first, then savings, then role evidence.** The corrections change what the instrument measures. The savings pay for part of the additions.
3. **Ownership and gates are unchanged.** ★ cores, predictions, tolerances and margins stay Huy's. P-002, P-004 (KC-L, KC-L2), P-005, P-006 and KC-E bind exactly as in v4.

**Tiers.** Every v4.1 item carries one:
- **[C] core:** scheduled inside its stage;
- **[O] optional:** runs only if KC-1's re-forecast of weekly hours (P-003) allows; ranked within each stage;
- **[G] gated:** a proposal in `DECISIONS.md` (P-012 to P-019), decided by Huy.

**Corrections.** Each rests on primary sources read on 2026-10-27, mostly checked a second time independently (`research/2026-10-27-role-alignment/verification.md`). K11 is a derivation rather than a sourced fact.

| # | v4 said or assumed | What is true | v4.1 change |
|---|---|---|---|
| K1 | S2 runs "≥ 3 seeds", with seed test–retest and seed-paired verdicts | **`e2e_challenge`** (the branch S2 runs): the Alpamayo driver stores the session seed and never uses it. VLM sampling (top-p 0.98, temperature 0.6) and flow noise draw from the process-global RNG, whose state depends on call order across concurrent sessions. The branch has no determinism switch.<br>**`main`:** the driver reseeds every tick (session seed + inference count) only with `force_determinism: true`, which every shipped Alpamayo config sets to false.<br>**Elsewhere:** CAT-K ignores its seed. NVIDIA's maintainer recalled that NuRec "might" vary slightly from run to run (#128); that is unmeasured | S2 runs **replicates**. The seed mode is a pre-registered ★ choice. The negative control gains a same-policy re-run. Seed coupling is measured, never assumed (L4-1) |
| K2 | The contamination rule excludes clips in `alpamayo_aligned` | PAI-AV v26.10's `alpamayo_aligned.parquet` labels **every** clip: train 276,299 / test 28,317 / alpasim 1,536 = 306,152, per the public dataset card. So that rule excludes everything | The report and A1 key on the split label (S2; `nano_alpamayo` A1; L5-6) |
| K3 | S4 ★: "AlpaGym's clipped trajectory-level ratio … k3 KL" | **The ratio.** AlpaGym (`main` @ `972d160`, on cosmos-rl) clips a per-step-row **geometric-mean** ratio. Its log is the mean over T SDE steps × 64 waypoints × 2 controls of the per-element log-ratio, clamped to ±5, with T = 5 in the policy preset. It is then clipped to [0.8, 1.2].<br>**The KL.** It is the closed-form per-transition Gaussian KL: Flow-GRPO's expression under the reversed time convention, averaged over the same elements. It is off by default (`kl_beta: 0.0`).<br>**k3** belongs to the open-loop recipe | S4 ★ and Build (L2-2) |
| K4 | S4 arm 2: the blog reward "plus comfort/jerk" | Not configurable today. AlpaSim has no jerk or comfort scorer, and AlpaGym takes reward terms only from AlpaSim's aggregated metrics or distance to ground truth. AlpaSim's `utils_rs` already has a `Trajectory.jerk()` utility, so a jerk scorer is a small evaluation-side addition | The jerk arm waits for that scorer, built with the stall and oscillation scorers the audit needs (L2-3) |
| K5 | (implicit) AlpaGym's default preset trains | With the shipped defaults (`n_generation: 1`, `kl_beta: 0`), every advantage is 0, the gradient is 0, and the trainer skips the optimizer step: **the default config does not train.** NVIDIA's own presets use `n_generation` 2–6 | S4 Build requires `n_generation ≥ 2` |
| K6 | AlpaSim: no outside PR merged since 2026-08-17 | True of `main` only. On `e2e_challenge`, four PRs from accounts without collaborator status merged between 08-24 and 08-31, among them #167 (the 1,761/441 split). #167's author is an OPTED co-author who worked partly at NVIDIA Research, so these merges say little about unaffiliated contributors | §2, contract, `04` §5 (L1-16) |
| K7 | S2 waits for the 10-31 close because of #224 | #224 closed on 10-23. NVIDIA will hold the withheld simulation choices "until our next release" | S2 Timing (L1-16) |
| K8 | No closed-loop evaluation of any NVIDIA 2B student is published | TacitVLA (mostly NVIDIA authors, 2026-10-13) reports AlpaSim scores on the 441 public scenes for research variants built on a 2B backbone (Qwen3-VL-2B; 3.9B with its flow expert): no reasoning, explicit reasoning, tacit reasoning. The ± is the SD over three rollouts per scene; there is no paired test and no token-budget sweep. It is not a distilled student measured against its teacher | §0.2, §1, S3 and S5(c) wording (L1-2) |
| K9 | (implicit) reactive traffic is available on the S2 protocol | `e2e_challenge` has no CAT-K: its traffic service is a stub, and the production topology skips it, so traffic is log replay. CAT-K exists only on `main`, where its sampling ignores the seed | S2 manifest; P-013 (L3) |
| K10 | (implicit) the renderer setting is fixed | NVIDIA's Cosmos-based Harmonizer is on in the `e2e_challenge` developer default, and off in its competition preset ("unharmonized … until the … final reruns"). `main` passes no flag. NVIDIA "plans to enable" it for the final reruns (#222) | S2 pins it after a cost pilot (L3-1, L3-2) |
| K11 | `04` §7 E4 predicts "identical policy outputs" when rendering at model resolution | Rendering at 320 × 576 is not the processor's resampling of a 1080p frame, so outputs cannot be bit-identical | Premise note in `04` §7; the test becomes paired equivalence (L3-10) |
| K12 | KC-E fallback: a PR #42 student "at pilot scale" | Only PR #42's stage 1 needs ≥ 128 H100s. That is about 6,290 H100-h, inferred from NVIDIA's own 1.77 s/step on 128 GPUs (PR #44). A 10%-schedule pilot (≈ 629 H100-h, $1.1–1.5k) never reaches the recipe's on-policy phase, which starts at step 75,000 of 100,000 in the reproduced config (60,000 of 80,000 in the generic one). So it is not NVIDIA's student | KC-E re-scoped; the pilot becomes P-017 (L5-8) |
| K13 | S5: one 8 × H100 node at $15–32/h | List prices on 10-27 are $31.92–36.00/h (per-GPU list price × 8). vast.ai's one cheaper offer ($19.20/h) allowed only 0.2 days and had no NVLink | S5 cost; §5 (L1-15) |
| K14 | (implicit) one AlpaSim branch has everything S2 needs | **`e2e_challenge`** has the 441-scene curated protocol, the post-#215 evaluator and the `SLURM` (Pyxis) and docker-compose run methods.<br>**`main`** has the per-tick reseed, the 9-field DCGM counter list, the `alpasim_*` series including queue depth, the `SLURM_ENROOT` method, a CUDA MPS switch, the Slurm topology skill and the 8-GPU Alpamayo 1.5 evaluation topology. `main` has not changed since 2026-08-18 | S2 pins `e2e_challenge` for scores and ports or re-hosts what it needs: the reseed, a host-side DCGM exporter, throughput from its `rpc_duration_seconds` series. It switches to NVIDIA's next release if that merges the two |

**Savings.** These are inferences until measured. The measurements are scheduled.

| # | Change | Saving | Stage |
|---|---|---|---|
| V1 | NVIDIA's expert CUDA graph (alpamayo1.5 PR #36, merged 2026-08-29; off by default; never enabled in AlpaSim) behind a driver flag and a bitwise test. A PR comment reports 160 of 160 eager-vs-graph expert evaluations bitwise identical (batch 1–4). Its latency cuts, −15.5% (10B) and −31.5% (2B) per policy call, are preliminary: one paired AlpaGym sample per model | −11% to −14% GPU-h per Alpamayo 1.5 sweep, by inference; −85 to −266 GPU-h (−$85 to −$415) across S2–S5 | S2 → S3–S6 (L7-2) |
| V2 | S5's learner is PR #42's one-node stage-2 trainer, fed teacher labels through its `target_action` override | −4 to −8 h | S5 (L5-1) |
| V3 | While KC-E is open, S3 measures NVIDIA's deployment levers on the released teacher instead of training a pilot student | −10 to −20 h; about −$1.1–1.5k against a pilot | S3 (L5-8) |
| V4 | S3 reuses S2's teacher sweeps behind a concurrent drift sentinel | −54 to −135 GPU-h at an illustrative 20% sentinel | S3 (L4-3) |
| V5 | S7 uses AlpaSim's own external-driver mode and `planner_delay_us` delay buffer | −4 to −8 h | S7 (L7-8) |
| V6 | Conditional on Huy's choice of inference target: one replicate on more distinct scenes | −37.5% to −64.3% of rollouts at equal precision, for an illustrative within-scene correlation ρ = 0.3–0.9 | S3 (L4-2) |

**Role evidence added.** Every item is measured on one node inside a stage that already runs:
- **S2, the simulator as a measured system:**
  - a rollout-performance profile using AlpaSim's own throughput metric;
  - the expert-graph speed-up behind a bitwise test;
  - DCGM monitoring with a goodput waterfall;
  - an instrument-noise decomposition;
  - the validity envelope as a dose–response curve.
- **S4, the RL loop as a measured system:**
  - an estimator audit;
  - config-only reward arms with zero-weight hacking monitors;
  - a bottleneck analysis and scaling model of AlpaGym, validated from 1 to 8 GPUs on NVIDIA's own presets.
- **S5, the correction loop as a data pipeline:**
  - manifest-driven rounds, measured with NVIDIA's own training-efficiency definitions;
  - a 1→8-GPU scaling and resiliency ladder.
- **S3 and S6–S7, integration and C++:**
  - a batched AlpaSim driver for the 2B student architecture;
  - TensorRT-11-ready C++;
  - a production-grade C++ driver server.

**What it costs.** About +50 to +85 focused hours in [C], net of the savings: roughly 320 h becomes about 370–405 h (the sum of §3's stage rows). The synthesis report, which tiers some items differently, brackets the full range at +45 to +99 h. [O] adds up to about +40 h. These are planning inferences; KC-1 (11-02) re-forecasts them from S1's actual hours. **Cut order if hours run short:** S8 first, then [O] items, then [C] role items in reverse order of the per-stage lists. The corrections are never cut.

### 0.2 v3 → v4 (2026-10-25)

The owner's decision input (2026-10-25) made the teacher → student → Thor loop the #1 project. The review and verification kept that direction, amended its gates, and found that NVIDIA had moved first on the student. That activates the input's own contingency ("use Edge as the P2 baseline").

| Area | v3 (2026-10-03) | v4 | Why (evidence) |
|---|---|---|---|
| Headline | G3, the gate's own construct validity | The teacher → student → Thor loop, every closed-loop number measured by the gate | Owner's decision; review §1 |
| The gate | S1–S2, then validity studies | S1–S2 unchanged in substance; it is the instrument for every later stage | The input's gates (P0 noise floor, P1 effect size, P2 margin) are outputs of `verdict()` and the power tables (review §3) |
| Student | nanoAlpamayo trained by us | **NVIDIA's own 2B student** (Alpamayo 2 Edge, once released) is measured first. Our work is the closed-loop correction and the reasoning-budget curve on top of it | PR #42 (public, unmerged distillation recipe, ≥ 128 H100s recommended, no weights); Edge announced 10-21, download promised "later this month" (ledger §0) |
| Teacher RL | Not scheduled (AlpaGym fragile) | S4: first published closed-loop gains (or negative result) from AlpaGym, with a reward-hacking audit | AlpaGym has published no gains; a reasoning reward cannot move the frozen VLM there (ledger B1) |
| Thor | Collaborator tier only | S6 on Jetson AGX Thor if P-002 is approved; HIL in S7 | The input's choice; Jetson runs TensorRT 10.16.2, while DRIVE AGX Thor runs TensorRT 11.x (ledger C3) |
| Real car | Absent | S8 shadow mode in HCMC, **optional**, gated on P-006 | The input's choice. Shadow mode is open-loop measurement, not validation (§6) |
| Deferred, not deleted | — | G3 renderer and traffic agreement (R2–R4), cross-paradigm convergence (R6), private set (R5); G4 guardrail; the stand-alone G7 report | Scope. Reliability (R1) and the validity records stay in S2. Known-order ladders come free from S5–S6 (R7/R9). The v3 stage cards remain in `archive/02_PHYSLOOP_PLAN_v3_2026-10-03.md`. v4.1 offers minimal R2–R4 legs as gated proposals (P-013, P-014) |
| Publication | KC-L for numbers | KC-L, plus **KC-L2** for weights and labels derived from PAI-AV/NuRec | Data license §4.6 bans derivatives and distribution (ledger A8) |
| Manifest | Git hash, config, seeds, both AlpaSim SHAs | Adds the resolved lockfile, `n_sim_steps` requested and actual, the score definition, and the overlap with PAI-AV's `alpamayo_aligned` split. v4.1 adds the simulator configuration (S2) | AlpaSim issues #207 and #208; PAI-AV v26.10 (ledger B3, A8) |
| Calendar | D = 2026-10-05; S1 never started | D = **2026-10-26** | No `gate/`, `journal/` or experiment file existed on 2026-10-25 |
| Budget | $0 / ≤ $2.5k / ≤ $6k / collaborator | Same cloud tiers, plus two separate hardware lines (Thor, rig) | Review §3.5; prices re-verified 2026-10-25 and 2026-10-27 |

**What survived**, re-checked 2026-10-27:
- No upstream paired, stratified release verdict exists.
- No paired closed-loop evaluation of an NVIDIA 2B student against its teacher has been published. NVIDIA's distillation and Thor PRs report open-loop minADE and latency only. TacitVLA's unpaired AlpaSim scores are for 2B-backbone research variants, not for a distilled student measured against its teacher (K8).
- No published closed-loop RL gains from AlpaGym exist. Its `main` branch has had no code change since 2026-07-08. NVIDIA's public branches carry commits up to 2026-09-03 (a cosmos-rl 0.4.4 bump, rollout-transport hardening, humanoid and VLA integration). The alpamayo1.5 PR #36 thread shows 2B and 10B Alpamayo policies running inside AlpaGym. Treat both as race signals for S4 and S5.
- No 2B student has been shown at 10 Hz on Jetson AGX Thor. Edge's weights were still not public on 2026-10-27.

## 1. The one thing we build

**PhysLoop builds and measures the Alpamayo teacher → Thor-student path.** It has two parts.

1. **The gate (the instrument).** It runs paired, stratified, validity-recorded, reliability-measured closed-loop verdicts on AlpaSim. It answers one question about any two checkpoints: *did distillation, quantization or RL make the deployable model worse, and where?* It ships as thin layers that can be upstreamed into AlpaSim.
2. **PhysLoop-AV (the headline application).** It proceeds in six steps:
   - measure NVIDIA's own 2B student against its 10B teacher in closed loop;
   - post-train the teacher in closed loop;
   - correct the student on-policy;
   - map what runtime reasoning buys under a latency budget;
   - serve the student on Jetson AGX Thor and verify it in hardware-in-the-loop;
   - optionally, observe it in shadow mode in a human-driven car in Ho Chi Minh City.

**Research question.** Under a fixed on-device latency budget, how does closed-loop driving quality change with the runtime Chain-of-Causation (CoC) token budget?
- **What exists.** NVIDIA's TacitVLA (arXiv 2610.17540, 2026-10-13) measured three endpoints of a model on a 2B backbone in AlpaSim's 441 public scenes:
  - no reasoning: 46.4%;
  - explicit reasoning: 52.3%;
  - "tacit" reasoning, trained with reasoning but never decoded at inference: 52.5%.

  Each figure is ± the SD over three rollouts per scene, with no paired test. Explicit reasoning cost 269 ms against 127 ms on one H100.
- **What is open.** The curve between the endpoints, on the deployable student, at a fixed on-device latency, measured paired. That is the XPeng-vs-NVIDIA split ("language is a bottleneck" vs reasoning on the hot path) as a curve rather than a binary.
- **The second question.** What did NVIDIA trade to reach 10 Hz? Its own 2B students measure 171 ms (FP8) on DRIVE AGX Thor, yet Edge is announced at 10 Hz.

**Owner's targets** (from the decision input; each is re-stated on a named metric before pre-registration):
- the deployable student is non-inferior to its teacher within a margin of 10% of the teacher's score, paired, on one pinned run;
- it fits 16 GiB;
- it runs at p99 ≤ 100 ms with 4 cameras on Jetson AGX Thor;
- it sustains 10 Hz for 30 minutes.

### 1.1 What the finished plan evidences

The gate and the loop are also the evidence for the target role. Each stage card ends with a **Role evidence** line naming the JR2010863 lines it serves (IDs I1–I2, D1–D5, Q3–Q5, X1–X3, defined in `05` §3). Two limits are stated wherever the evidence is used:
- **Scale.** Every number comes from one rented node, unless a gated proposal (P-012) adds a measured multi-node point.
- **Origin.** PhysLoop is a solo reproduction on NVIDIA's open stack, not industry infrastructure experience (§6).

## 2. Deliverables, re-scoped against what upstream ships (re-checked 2026-10-27)

| G | Upstream already ships | Still open, our new work | Verdict | Stage |
|---|---|---|---|---|
| **G1 Gate v1** | Per-rollout scorers; `results-summary.json`; seeds on `main` (PR #149), honoured only with `force_determinism`; the leakage-aware 1,761/441 split (PR #167, an external PR merged on `e2e_challenge`); Drive-IRT panel ranking over unseeded replicates | Paired two-checkpoint verdict; stratification; replicate test–retest; per-stratum bounds; group-sequential looks; allowed-claims list | **Keep** | S1, S2 |
| **G2 Validity records** | `dist_to_gt_*` scorers; 4 m corridor scored as failure on `e2e_challenge` (stationary-tail false exits fixed in #215; leaderboard recomputed 10-22) | Per-rollout deviation bins; out-of-envelope rollouts reported separately; the envelope as a dose–response curve with NVIDIA's per-clip quality covariate | **Keep**, inside every report (no separate stage) | S2 onward |
| **G3 Gate validity** | Sim2Val (1 policy); OmniDreams (n = 4) | Reliability (R1) with an instrument-noise decomposition; known-group ladders from the pipeline (flow steps, bit-widths, CoC budget: R7/R9). Minimal R2–R4 legs only through P-013/P-014 | **Narrowed.** R2–R6 deferred | S2, S5, S6 |
| **G4 Guardrail** | `safety_monitor_safe` hook and scorer | Monitor, override path, policy+monitor scoring | **Deferred** | — |
| **G5 Student** | PR #42 recipe (open; no weights; open-loop only; stage 1 includes token-level on-policy distillation); PR #45 Thor inference (open; ~171 ms FP8 on DRIVE Thor); Edge announced, not released on 10-27; TacitVLA's unpaired research variants | Closed-loop paired verdict of NVIDIA's student against its teacher; closed-loop, state-visited correction; the CoC-budget curve | **Headline** | S3, S5 |
| **G6 Precision** | ModelOpt PTQ including the expert (0.43 pinned; 0.47 current); open-loop minADE only | Real-engine (TensorRT 10.16) per-format closed-loop verdicts on Jetson AGX Thor | **Keep**, moved to the device | S6 |
| **G7 Throughput and latency** | • AlpaSim Prometheus telemetry, with a DCGM counter list on `main` only;<br>• AlpaSim's Slurm run methods and `optimize-slurm-topology` procedure (`main`);<br>• AlpaGym's `perf` telemetry and Slurm presets;<br>• the expert CUDA graph (alpamayo1.5 PR #36);<br>• NVIDIA's training-efficiency measurement chapter (recipes PR #44, open);<br>• FlashRT Jetson numbers for the 10B teacher | The measured rollout profile and goodput of AlpaSim; the AlpaGym bottleneck and scaling model; the S5 pipeline metrics; the student latency profile on Jetson Thor; HIL with latency injection | **Keep**, split across stages, and widened to the training loop | S2, S4, S5, S6, S7 |
| **G8 Flywheel** | CoC autolabeler; Instant NuRec | Shadow-mode trigger data engine on HCMC traffic | **Keep, optional** | S8 |
| **G9 Teacher RL study** | AlpaGym (expert-only GRPO on cosmos-rl, VLM frozen):<br>• `main` has had no code change since 2026-07-08; NVIDIA's public branches carry commits to 2026-09-03;<br>• one-node and multi-node Slurm presets ship, "not expected to run out of the box";<br>• a 2B Alpamayo policy has been timed inside it (alpamayo1.5 PR #36) | First published closed-loop gains or negative result; reward-hacking audit; estimator audit; the bottleneck and scaling model | **New** | S4 |

## 3. Stages: one dependency graph

```
S1 ──► S2 ──┬──► S3 student verdict ──┬──► S5 closed-loop correction + CoC-budget curve ──┐
            │                         └──► S6 Jetson Thor serving (P-002) ◄───────────────┘
            └──► S4 teacher RL ────────────► (RL'd teacher labels S5 if KC-4 passes)
                                                    S6 ──► S7 HIL + latency injection ──► S8 shadow mode (optional, P-006)

KC-E: Edge weights public? Re-check 2026-10-31. If not public at S3's start, run S4 first (the input's order).
KC-L / KC-L2: no public PAI-AV/NuRec-derived number, weight or label without a written license answer (P-004).
Gated (P-012 … P-019): multi-node (S4/S5), traffic and renderer legs (S2), batched driver (S2→S5), third RL arm (S4),
PR #42 pilot (S3), few-step expert (S5), one C++ driver for two targets (S6).
```

**Stop points.** Each one ships a public artifact, subject to KC-L and KC-L2:
- after S3: the first paired closed-loop verdict on NVIDIA's student;
- after S4: the RL study (the input's candidate #2), with the RL-loop bottleneck and scaling report;
- after S6–S7: the Thor serving study (candidate #3).

S3 and S4 depend only on S2, so their order is reversible.

D is the start date of S1: **Monday 2026-10-26**. The windows below are v4's and assume 25–30 focused hours a week. That figure is unconfirmed (P-003). KC-1 re-forecasts every window from S1's actual hours, and decides which [O] items run. The effort figures are planning inferences.

| Stage | Serves | Window (v4) | Calendar (D = 2026-10-26) | Effort v4 → v4.1 [C] (h) | Cost at 10-27 prices (inference) | Kill check |
|---|---|---|---|---|---|---|
| **S1** Gate core on synthetic truth + power analysis | G1 | D+0 → D+1 | 10-26 → 11-02 | ~20 → ~24 | $0 | KC-1 |
| **S2** Reproduce + reliability on AlpaSim | G1, G2, G3·R1, G7 | D+1 → D+3 | 11-02 → 11-16 | ~40 → ~58–64 | $139–545 + storage | KC-2, KC-L |
| **S3** Closed-loop verdict on NVIDIA's 2B student | G5 | D+3 → D+5 | 11-16 → 11-30 | ~30 → ~28–34 | $106–540; up to $54–211 less if S2's teacher sweeps are reused | KC-3, KC-E |
| **S4** Closed-loop RL on the teacher | G9, G7 | D+5 → D+8 | 11-30 → 12-21 | ~50 → ~70–78 | $807–2,628 | KC-4 |
| **S5** On-policy correction + CoC-budget curve | G5, G3·R7, G7 | D+8 → D+11 | 12-21 → 2027-01-11 | ~60 → ~70–78 | $1,038–2,783 | KC-5, KC-L2 |
| **S6** Jetson AGX Thor serving | G6, G7, G3·R9 | D+6 → D+12 | 12-07 → 2027-01-18 | ~60 → ~60–62 | Thor $5,499 (P-002) + ~$102–395 | KC-6 |
| **S7** HIL + latency injection | G7 | D+12 → D+13 | 01-18 → 01-25 | ~20 → ~20–24 | ~$200 | KC-7 |
| **S8** Shadow mode, optional | G8 | D+13 → D+15 | 01-25 → 02-08 | ~40 | Rig ~$5–8k (P-006) | KC-8 |

The v4 total was about 320 focused hours. v4.1's [C] items bring it to about 370–405 h, roughly 12–16 weeks at 25–30 h/week, or about 2–3 weeks more than v4. [O] items add up to about 40 h. The overlapping windows assume S6's device work runs alongside S4–S5. If that is not realistic for one person, KC-1 re-forecasts the windows and applies the cut order in §0.1.

Each stage card below gives the Why as a verified symptom. ★ marks the cores Huy writes: the measurement instruments, statistics, RL and distillation loss math, verifier logic, predictions and diagnoses. Agents write plumbing, harnesses, scaffolds, failing tests, sweeps and plots. The derivations, known-answer exercises and traps for each stage are in `03_MASTERY_PLAN.md` §4.

### S1 — Gate decision core on synthetic truth, plus power analysis ($0, CPU)
- **Why.**
  - On 441 paired scenes at 4% discordance, 80% power detects only about 2.72 pp. The decision input's RL gate ("at-fault collisions down ≥ 20% relative") is 1.6 pp at an 8% base rate. It needs about 964 paired scenes at ψ = 4%, so on 441 its correct output is INSUFFICIENT (review §3.2).
  - **New in v4.1.** S2's repeats will be unseeded replicates unless S2 ports `main`'s reseed (K1). The power branch must therefore model within-scene correlation. Any interim look at S3–S5's sweeps must be designed now, before a sweep exists. Under the intersection–union rule, overall GO power is the product of per-stratum powers: four strata at 80% each give 41% overall.
- **Build:** `gate/`, as specified in v3:
  - the identity tuple and `is_pair()`;
  - `verdict()` returning GO / NO-GO / INSUFFICIENT, where **INSUFFICIENT blocks GO**;
  - a synthetic generator with known effects, road clusters and strata;
  - a 1,000-run Monte Carlo;
  - power tables ("what 441 scenes can certify"), now including:
    - scenes needed for a relative effect as a function of base rate and discordance;
    - the non-inferiority margin on both candidate score metrics. The challenge `scene_score` in [0, 1] is continuous, so its branch is MDE = (z_α + z_β)·σ_d/√n as a function of σ_d, the SD of paired differences (Tango is binary-only). The card-style km-between-events score is a rate ratio;
    - **[C] the replicate branch** (L4-2): Var(mean) = σ²/(nm)·[1 + (m − 1)ρ] for n scenes × m replicates. It gives the scenes-versus-replicates trade at equal precision;
    - **[C] intersection–union accounting** (L4-7): overall GO power as the product of per-stratum powers for road-disjoint strata, and the per-stratum power needed for 80% overall;
    - **[C] group-sequential tables** (L4-5): one-sided α = 0.05 and 0.025; K = 2 and 3 equally spaced looks; O'Brien–Fleming, Pocock and Lan–DeMets (O'Brien–Fleming-type) boundaries; the n_max inflation; expected n at δ = −Δ, midway, and 0;
  - known-answer tests.

  It consumes AlpaSim's per-rollout outputs keyed by scene ID, and needs no dataset license.
- **[C] Rule for interim looks in S3–S5,** written into `verdict()`'s specification:
  - looks come after pre-registered, randomized road batches, and every batch covers every stratum;
  - a stratum that has crossed its efficacy boundary stops receiving rollouts;
  - a futility crossing in any stratum ends the verdict as INSUFFICIENT. NO-GO comes only from the regression flags;
  - ≥ 4 m exits are tallied separately at every look.
- **[O] Provisional noise inputs** (L4-10, L4-8), only if S1's hours allow; otherwise they move to S2's first day:
  - base rates, replicate discordance, ICC, design effect and σ_d for reference pairs, read from the 8 reference policies the challenge ships (441 × 3 unseeded replicates; Alpamayo 1, not 1.5; renderer setting before the final runs);
  - IRT difficulty from Drive-IRT, used only to form strata or inclusion probabilities (Horvitz–Thompson weights), never as a fixed subset;
  - every result labelled provisional. Reading the LFS objects is Huy's go-ahead, and nothing derived is published before KC-L.
- **★ Huy:** `verdict()`, which includes
  - paired non-inferiority with a pre-registered margin (Tango score CI);
  - an intersection–union rule across strata;
  - Holm for regression flags;
  - cluster-adjusted McNemar (Durkalski) or a road-level paired bootstrap;
  - the one-sided zero-failure bound p_U = 1 − α^(1/n);
  - the power derivations, including the rate-ratio case and the replicate branch;
  - **the spending function, K, the futility rule and the stage-wise-ordering bounds after early stopping;**
  - **the allocation objective across strata;**
  - the predictions.
- **Acceptance.** As v4: the known-answer tests reproduce
  - 12 vs 6: one-sided 0.11894, two-sided 0.23788;
  - 0/441: 0.677% one-sided, 0.833% two-sided;
  - 0/1,000: 0.2991% one-sided, 0.3682% two-sided;
  - MDE 1.94 / 2.72 / 3.90 pp at ψ = 2.1 / 4 / 8%;
  - non-inferiority margins 2.67 / 3.77 pp;
  - power 0.22.

  The Monte Carlo must also show three things: INSUFFICIENT (never GO) for a 9-scene stratum; higher false-GO when stratification is removed; and false-GO ≤ α wherever power exists. One test from v4: Connor's n for a 1.6 pp effect at ψ = 4% (one-sided, 80%) is 964.

  **New in v4.1** (computed in `research/2026-10-27-role-alignment/notes/arith_check.py`):
  - O'Brien–Fleming boundaries for K = 2 at one-sided α = 0.025: 2.7965 and 1.9774; Pocock: 2.1783 at both looks. At α = 0.05: 2.3730 and 1.6780; Pocock 1.8754;
  - four strata at 0.80 give 0.4096 overall; 0.80 overall needs 0.9457 per stratum;
  - at m = 3 and ρ = 0.6, equal precision needs 1.3636 × the scenes with one replicate each, which is 54.5% fewer rollouts;
  - a sequential Monte Carlo on clustered synthetic truth holds false-GO ≤ α.
- **Predictions:**
  - P-S1-1, false-GO for a −30% effect on a 2% stratum at 7% base failure: ____
  - P-S1-2, share of INSUFFICIENT under the null: ____
  - P-S1-3, change in false-GO without stratification: ____
  - P-S1-4, the base at-fault rate at which a 20% relative cut becomes decidable on 441 scenes × 3 replicates: ____
  - P-S1-5, number of S3 strata that return INSUFFICIENT under the planned allocation: ____
- **Kill (KC-1, D+1).**
  - If `verdict()` cannot hold false-GO ≤ α on synthetic truth, fix the statistics before any rented GPU-hour.
  - No interim look is used in S3–S5 until the sequential Monte Carlo holds false-GO ≤ α.
  - If S1 runs past twice its estimate, ship the power-analysis half alone.
- **Role evidence (`05` §3):** X2, D3 (the measuring half).
- **Upstream value.** Publish the repository, then open an AlpaSim issue proposing the paired layer for `e2e_challenge/local_evaluation`.

### S2 — Reproduce and measure reliability on AlpaSim (the input's P0)
- **Why.**
  - NVIDIA's own Alpamayo 1.5 card moved 0.81 ± 0.01 → 1.37 ± 0.10 with unchanged weights, and never defines the score or "±".
  - The `e2e_challenge` evaluator changed mid-challenge: a stationary ground-truth tail produced false corridor exits (#198), fixed in #215 on 10-20, and the leaderboard was recomputed on 10-22.
  - One AlpaSim SHA resolved about 25 packages differently on two machines (#207).
  - **The repeats are not seeded** on the branch S2 runs (K1).
  - **The renderer has a protocol switch,** the Harmonizer (K10).
  - **Post-processing alone moves the score.** An NVIDIA-co-authored, Toronto-led paper (ECO, arXiv 2609.31383, 2026-09-25) raised AlpaSim scene scores by 123% (VaVAM) and 22% (DiffusionDrive) with no change to the policy.
  - **The protocol and the tooling sit on different branches** (K14).

  Without a pinned protocol and a measured noise floor, no later number means anything.
- **Timing.** S2 starts 11-02, after the challenge closes. First check for a new AlpaSim release or sync: #224 ties the withheld simulation choices to NVIDIA's next release. If none has landed, label S2's results provisional.
- **[C] Pre-registration before the first run** (★, in `experiments/`):
  - **The branch.** `e2e_challenge` for scores until a release merges the branches (K14).
  - **The seed mode.** Either port `main`'s per-tick reseed (and its deterministic-runtime setup) into the pinned `e2e_challenge` driver, or run unseeded replicates and say so.
  - **The Harmonizer state,** informed by the cost pilot below and by #222.
  - **The run method and topology.**
- **Build:** a G1 v1 report on three policies:
  - TransFuser (AlpaSim's provisional LTFv6 baseline driver), as a pipeline sanity check;
  - Alpamayo 1.5 in BF16;
  - Alpamayo 1.5 in FP8 (ModelOpt recipe).

  It runs 441 `nurec_curated_val` scenes × ≥ 3 **replicates**, which are seeded only if the manifest records a seeded mode. The report contains:
  - per-stratum paired verdicts with one-sided bounds;
  - replicate test–retest;
  - the negative controls:
    - a bit-identical re-export must not be flagged;
    - **[C]** a same-policy, same-seed re-run, which bounds NuRec's run-to-run variation (L4-1);
  - per-rollout validity records in deviation bins < 1, 1–2, 2–4 and ≥ 4 m, with `left_corridor_laterally` split out. **[C]** The envelope is also reported as a dose–response curve per policy, joined with NVIDIA's per-clip `quality_score` (beta), `time_of_day` and `ego_speed`, with an endpoint-overshoot class and black-frame render failures counted (L3-4);
  - the failure taxonomy mapped onto the NHTSA pre-crash typology (as v3), plus a candidate stratum for Huy: a stationary or slow obstacle in the ego's path. A Politecnico di Milano–led paper with an NVIDIA co-author documents this failure for Alpamayo 1.5 (arXiv 2610.28184): footprint intrusion falls from 58.1% to 14.5% with its guidance (L1-5);
  - **both score metrics**: the challenge `scene_score`, and the card-style km-between-events rate. The rate is labelled as a diagnostic under the paper's convention, which excludes > 4 m;
  - the numbers that S3–S5's power depends on: base rates, discordance, and σ_d of the paired `scene_score` differences between two policies. A 7 pp margin is decidable on 441 scenes only if σ_d ≲ 0.59 (review §3.3). **[C]** Seed coupling is part of this: Var(d) under matched versus label-permuted pairing of the same rollouts gives the within-scene correlation ρ_w for (1.5 BF16, 1.5 FP8) and for (TransFuser, 1.5) (L4-1);
  - **the contamination report (corrected, K2)**: the `alpamayo_aligned` split label (train, test or alpasim) of the source clip of every one of the 441 and 1,761 scenes. It is read with `physical_ai_av` ≥ 0.2.3 against PAI-AV revision `39414de6` (v26.10) (L5-6);
  - **[C] an instrument-noise decomposition** (L3-3):
    - renderer: a render-determinism probe at fixed poses, per Harmonizer state;
    - policy: identical inputs passed twice;
    - traffic: replay, which is deterministic;
    - the tick at which each replicate pair first diverges;
  - **[C] the rollout-performance profile**, which replaces "the first measured throughput" (L1-6, L3-5, L6-2, L7-1):
    - AlpaSim's own throughput definition: the stabilized full-run seconds per rollout. It is read only after the 30–45 minutes of stabilization that the `optimize-slurm-topology` procedure on `main` prescribes. On `e2e_challenge` it is computed from the `rpc_duration_seconds` series and the rollout timestamps. Reported beside it: rollouts per GPU-hour, per-service GPU utilization, RPC queue depth where exported, and DCGM tensor-active;
    - a per-service GPU-time split (driver, renderer, physics, controller, queue wait) from Prometheus plus `nsys` with NVTX;
    - a CUDA-event split of each driver call into encode, prefill, decode, expert and host residual;
    - GPU-hours from the **mean** wall clock, never the median, as NVIDIA's measurement chapter prescribes (recipes PR #44, open);
    - before any speed-up lever runs, its Amdahl bound S = 1/((1 − f) + f/k).

    The driver's share of step GPU-time is unmeasured; 73–87.5% is an inference. This profile re-baselines every cost in this plan;
  - **[C] NVIDIA's expert CUDA graph** (alpamayo1.5 PR #36) behind a driver flag (L7-2):
    - **The test.** Eager and graph driver outputs must be bitwise identical on every compared tick, with `force_determinism=true` and the same inference seeds, and the negative controls must still pass. Otherwise the flag stays off.
    - **Non-deterministic sweeps.** If sweeps run with `force_determinism=false`, the manifest says the bitwise claim was established in deterministic mode and assumed for the other.
    - **Pinning.** The driver's `alpamayo1_5` dependency is an unpinned git source, so the manifest records the resolved commit;
  - **[C] monitoring** (L6-3):
    - a DCGM exporter run on the host. Its counter list adds XID errors, power, total energy, SM clock, GPU temperature and PCIe replays to the 9 fields of `main`'s list. `e2e_challenge` ships no DCGM list, and the profiling fields need host access;
    - a per-sweep goodput waterfall: useful ÷ paid GPU-seconds, split into provisioning, startup, congestion, steady-state idle, retries, requeue loss, tail drain and teardown;
    - energy per completed rollout;
    - alert rules for XID ≠ 0, queue starvation and p95 divergence across workers.

    Retry and NaN-retry rates are computed after the run, because AlpaSim exports no retry counter;
  - **[C] the Harmonizer cost pilot** (L3-2): 5 scenes × 1 replicate, Alpamayo 1.5 BF16, Harmonizer off vs on, renderer timing on. It runs before the protocol is pinned;
  - **[C] the scorer known-answer test** (L1-3): #198's minimal reproduction becomes a failing test on the pinned evaluator. A stationary ground-truth tail gave 5.0 m of lateral error before the fix, and should give about 0 after it;
  - Prometheus dashboards, now joined with DCGM.
- **[O] in priority order:**
  1. **AlpaSim's own Slurm path on one rented node** (L6-1, L6-2):
     - every sweep goes through AlpaSim's `SLURM` run method (Pyxis + Enroot; the only Slurm method on `e2e_challenge`), submitted with its `submit.sh` as array jobs with `--requeue` and resume. Each sweep leaves `sacct` records and resume scripts under `evidence/`;
     - `main` adds `SLURM_ENROOT` (no Pyxis) and a CUDA MPS switch, so those arms need `main` or the next release;
     - run methods are never mixed across S2's policies;
     - add a run-method invariance check through the gate's own TOST, and one-variable CUDA MPS on vs off where available;
     - **Kill:** if a 1-GPU smoke run does not start within ____ (Huy) hours, revert to docker-compose and record the failure as a negative result;
  2. **Failure injection** on one sweep chunk (L6-4): a requeue, a killed driver replica, a stopped renderer and a time-limit requeue. For each, measure time to detect, time to recover, and lost and duplicated rollouts, checked by paired score equality against an uninterrupted reference;
  3. **Driver replicas per GPU,** 1 vs 2 (vs 3), with outputs checked bitwise on a subset (L7-3);
  4. **A capacity and service-balance design document** (L3-11, L7-11):
     - the balance N_drv·μ_drv ≈ N_rnd·μ_rnd;
     - renderer-to-driver GPU ratios against concurrency and NRE workers;
     - validated on 20 scenes × 1 replicate on 1–2 GPUs, against NVIDIA's own topologies;
  5. **Covariate and truncated-horizon correlations,** measured for free on S2's rollouts and adopted in S3–S5 only if pre-registered (L4-4).
- **[G]:**
  - P-013: log replay vs CAT-K;
  - P-014: renderer fidelity and Harmonizer agreement;
  - P-015: the batched driver.
- **Manifest:**
  - **From v4:** git hash, config, replicate count and seed mode, both AlpaSim SHAs, the **resolved lockfile**, `n_sim_steps` requested and actual, GPU edition, scene list, deviation convention, score definition.
  - **Added in v4.1:**
    - the branch;
    - the evaluator, at or after #215, with its scorer-file SHA;
    - `force_determinism`;
    - the Harmonizer state;
    - the renderer image (`nre-ga:26.04` unless re-pinned) and its rendering flags;
    - the traffic mode;
    - the camera request and the decoded size;
    - the run method, topology, concurrency, replicas and NRE workers;
    - the driver settings: expert graph, `max_batch_size`, and the resolved `alpamayo1_5` commit;
    - the controller and trajectory post-processing identity;
    - the renderer out-of-memory re-queue count per rollout;
    - the policy clock source.

  A rollout record without these fields is labelled provisional.
- **Card match.** Matching the card's 1.37 ± 0.1 is a **diagnostic, not a gate** (review §3.4). NVIDIA says its internal test set is harder than the public one (#222). If the gap persists, open an AlpaSim issue with logs and report relative results only.
- **★ Huy:**
  - from v4: the reliability statistics (ICC(2,1), Kendall's W, design effect 1 + (m − 1)ρ), the stratum definitions, the taxonomy mapping, the validity-record semantics, the contamination rule, and the diagnosis of every discordance;
  - added in v4.1:
    - the seed mode;
    - the ρ_w estimator, with its road-cluster CI;
    - the Harmonizer choice;
    - the variance-component model;
    - the stabilization criterion;
    - the goodput definition and alert thresholds;
    - the cost model and bottleneck diagnosis;
    - the semantics for corridor exits in the ground-truth terminal segment.
- **Acceptance:**
  - **As v4:** ICC and W reported with CIs, along with the noise floor and the negative-control flag rate; every rollout carries a validity record; every retry is logged: NaN retries (Alpamayo prefill intermittently emits all-NaN trajectories), rollout retries (`runtime.max_rollout_retries`, default 2) and renderer out-of-memory re-queues (#222). Rates are computed after the run, because AlpaSim exports no retry counter. The contamination labels are stated.
  - **New in v4.1:**
    - every rollout record carries the full manifest;
    - every throughput number names its preset, SHA, GPU edition and Harmonizer state;
    - the expert-graph bitwise test passes, or the graph stays off;
    - the goodput waterfall sums to the paid GPU-hours within ____ (Huy);
    - the scorer known-answer test passes.
- **Predictions** (preserved from v3, with "seeds" read as replicates):
  - (i) test–retest discordance rate of Alpamayo 1.5 across replicates: ____
  - (ii) % of rollouts beyond 2 m and 4 m, and whether it differs between good and bad policies: ____
  - the pre-registered reliability floor: ____
  - the at-fault collision base rate of Alpamayo 1.5 under the pinned protocol: ____
  - **New in v4.1:**
    - ρ_w for 1.5 BF16 vs FP8: ____; for TransFuser vs 1.5: ____
    - the renderer's share of replicate discordance: ____
    - the driver's share of step GPU-time: ____; the top driver stage: ____
    - sweep GPU-h saved by the expert graph: ____
    - S2 sweep goodput: ____; the largest loss component: ____
    - seconds the Harmonizer adds per step (4 × 1080 × 1900, RTX PRO 6000): ____
    - rank correlation between `quality_score` and the per-scene validity-exit rate: ____
  - Also preserved: `04` §7 E4 (render at model resolution), optional, under its new premise note. E1 is reconciled in S4.
- **Kill.**
  - **KC-2, D+3:** if reliability is below the pre-registered floor, or the noise floor exceeds the effects that S3–S5 must detect → stop and redesign.
  - **Renders not repeatable at matched poses:** report the renderer as a noise source before any verdict.
  - **Same-seed re-run drift:** if a re-run under the same seed drifts beyond ____ (Huy), the noise floor is that drift, not zero.
  - **KC-L:** no public number derived from PAI-AV or NuRec without a written answer on license §3 (P-004). The fallback is to publish code, scene-ID manifests and synthetic results.
- **Cost:**
  - **Sweeps:** 4 × 34–85 GPU-h = 136–340 GPU-h ⇒ **$136–530** on RTX PRO 6000 Workstation at $1.00–1.56 (vast.ai, 2026-10-27).
  - **Profiling and pilots:** ≈ $3–15.
  - **Storage:** $40–216/month for the ~2.9 TB NuRec release. PAI-AV/NuRec are first downloaded here, and the 12-month license clock starts then.
  - **[O] 1:** adds a Slurm-capable VM premium of $0–255 (P-005).
  - **Expert graph:** the inferred saving of −85 to −266 GPU-h across S2–S5 starts after its bitwise test.
- **Role evidence (`05` §3):** I2 (the simulation's accuracy and performance), D2, D3 (measure), D4, X2, X3. Q5 only through [O] 1.
- **Upstream value:**
  - the first public replicate test–retest and reliability numbers for AlpaSim, plus the manifest fields that issues #207 and #208 show are missing. #208's reporter offered a PR, so file no competing one: keep both fields in PhysLoop's manifest and switch to upstream's if it lands;
  - a PR exposing the expert-graph flag, filed only after re-checking NVIDIA's post-10-31 sync;
  - a PR adding the DCGM fields to `main`'s counter list;
  - issues for any autoresume or retry defect.

### S3 — Closed-loop verdict on NVIDIA's 2B student (the input's P2 gate, measured on NVIDIA's artifact)
- **Why.**
  - **Open-loop only.** NVIDIA's distillation PR #42 reports only open-loop minADE6: 0.8875 m for the 1.5-2B student vs 0.8298 m for the teacher, +6.95% worse. Its Thor PR #45 reports only latency. Edge is announced at 10 Hz on DRIVE AGX Thor with no published closed-loop numbers.
  - **Open-loop error does not predict closed-loop score.** It shows no significant correlation with it (Bosch, n = 7; `01` C3).
  - **The one closed-loop result is unpaired.** NVIDIA's TacitVLA reports AlpaSim scores for 2B-backbone research variants, with no paired test and no student-vs-teacher comparison (K8).

  So *did distillation make the deployable model worse in closed loop, and where?* is unanswered for NVIDIA's own student. It is exactly the gate's question.
- **Build.** Paired, stratified closed-loop verdicts on the S2 protocol:
  - Alpamayo 2 Edge (2B) vs Alpamayo 1.5 (10B), in BF16 and FP8;
  - Edge vs Alpamayo 2 Super on H100, if affordable;
  - Edge with runtime reasoning on vs off, if the release exposes the switch;
  - validity records and both score metrics;
  - known-order pairs where they exist, such as Edge's flow-step settings (G3·R7).
  - **[C] the 16 GiB check, extended to a batch curve** (L5-7):
    - a driver that serves AlpaSim's `EgodriverService` for the PR #42 student architecture and batches concurrent sessions. The architecture is the public Cosmos-Reason2-2B plus a randomly initialized compact expert, labelled "timing only" until real weights exist;
    - measured: ticks/s per GPU, p50/p99 per call and VRAM against B ∈ {1, 2, 4, 8, 16}, at a fixed CoC length;
    - Edge or a PR #42 checkpoint drops in when one exists;
    - verdict sweeps use the B = 1 path unless the batched path passes P-015's equivalence test;
    - the same driver is S5's collection driver;
    - the Cosmos-Reason2-2B license (NVIDIA Open Model License) is Huy's to accept.
  - **[C] teacher-arm reuse** (L4-3): S2's Alpamayo 1.5 BF16 and FP8 rollouts serve as the reference arm when every manifest field matches. A concurrent drift sentinel re-runs the teacher on a pre-registered random road sample; its fraction is ★. If the sentinel fails, run the full teacher sweep.
  - **[C] the inference target, a ★ decision before the run** (L4-2): these 441 scenes, or the population of roads. If it is the population, spend rollouts on more distinct scenes, one replicate each, drawn from the 2,202-scene pool minus contamination exclusions. A random subsample keeps ≥ 2 replicates to estimate within-scene variance.
  - **[C] interim looks and allocation** by S1's group-sequential tables and intersection–union accounting (L4-5, L4-7).
  - **[C] controller and post-processing** (L1-12): pinned and recorded, with per-rollout motion-limit-violation diagnostics. A paired difference then splits into plan geometry, controller tracking and decisions (ECO).

  The report states which teacher Edge was distilled from, or that it is undisclosed.
- **★ Huy:**
  - the margin Δ on the named metric (the owner's target is 10% of the teacher's score);
  - the strata;
  - the inference target;
  - the sentinel margin and rule;
  - the latency model behind the batch curve;
  - the diagnosis of every stratum where the student loses.
- **Acceptance:**
  - the pre-registered verdict, GO, NO-GO or INSUFFICIENT, applied as written;
  - the sentinel result printed beside the verdict;
  - batch-curve numbers labelled timing-only until real weights exist;
  - every claim uses `§6` wording.
- **Predictions:**
  - paired student–teacher difference on the named metric: ____
  - the strata where the student loses most: ____
  - validity-exit rate (≥ 4 m) of student vs teacher: ____
  - **New in v4.1:**
    - the student batch-curve knee: ____
    - paired teacher re-run minus S2 difference (sentinel): ____
    - the look at which S3 stops: ____
- **Kill.**
  - **KC-3:** the verdict is the result, whatever it says. If Edge does not run inside AlpaSim's driver container within 16 GiB, report that, and use the fallback.
  - **KC-E, checked 2026-10-31 and at S3's start.**
    - **Capture on release day:**
      - repo ID and revision;
      - license terms on evaluation, fine-tuning and publishing derivatives;
      - backbone, action expert and flow steps;
      - the runtime-CoC switch;
      - camera count. NVIDIA's AD-Memo paper (arXiv 2609.38641) builds on a 2B, single-camera-focused Alpamayo 2 backbone without an action expert, and also uses an 8B checkpoint. Neither is public, and any link to Edge is unverified;
      - disclosed teacher;
      - any NVIDIA closed-loop number.
    - **If Edge weights are not public, or its license forbids evaluation:** run S4 first (the input's order). While KC-E stays open, S3 measures NVIDIA's deployment levers on the released teacher as known-order pairs (L5-8):
      - K = 5 with fixed noise (PR #45's deployed setting) vs K = 10;
      - CoC capped vs uncapped;
      - FP8 vs BF16 from S2.
    - **The PR #42 pilot student is P-017.** At about 629 H100-h for a 10% schedule, it never reaches the recipe's on-policy phase, so it would not be NVIDIA's student.
- **Cost:**
  - 3–4 sweeps on RTX PRO 6000 ($102–527 at 10-27 prices);
  - minus $54–211 if the teacher arm is reused;
  - plus ≈ $4–13 for the batch curve;
  - plus an optional Alpamayo 2 Super sweep on H100 ($61–204).
- **Role evidence (`05` §3):** X2, D3, D5, I1.
- **Upstream value:** the first paired, stratified, student-vs-teacher closed-loop evaluation of an NVIDIA student, as a report and an AlpaSim issue. It is novel only if no one has published such a verdict by release day, so re-check then.

### S4 — Closed-loop RL on the teacher (the input's P1)
- **Why.**
  - AlpaGym has shipped since June with no published gains. Its `main` branch last changed code on 2026-07-08, it has had zero outside PRs, and it pins the unmerged recipes PR #18. NVIDIA's public branches carry commits to 2026-09-03, and a 2B Alpamayo policy has already been timed inside AlpaGym (alpamayo1.5 PR #36). So re-check novelty at every stop point.
  - Goodhart is live. In AR1, a reasoning-only reward raised the critic grade 3.1 → 4.5 while consistency fell 0.62 → 0.53. The open-loop recipe also rewarded worse reasoning until a sign fix on 10-06 (#40).
  - **New in v4.1:**
    - the S4 card in v4 described AlpaGym's estimator wrongly (K3);
    - its comfort/jerk arm is not configurable (K4);
    - its default preset trains nothing (K5);
    - NVIDIA's RL presets train at a different rate and horizon from the gate's evaluation protocol.

    Where the RL loop spends its time is the framework-design question NVIDIA's own 3- and 5-node presets argue about in their comments. v4 used the loop without measuring it.
- **Build:**
  - GRPO on Alpamayo 1.5 in AlpaGym, **expert only, with the VLM frozen as shipped** (ledger B1). Train on the 1,761 split; evaluate on the 441 held-out scenes. **Every run sets `n_generation ≥ 2`** (K5).
  - **[C] Protocol pin, a ★ decision** (L2-7).
    - **NVIDIA's two experiment presets** train at 5 Hz over 4.4 s: 22 steps after 1.6 s of forced ground truth, with policy input 320 × 512 and traffic off.
    - **The base default** is 10 Hz with 8 valid steps (0.8 s).
    - **The gate** evaluates at 10 Hz over 200 steps.

    Either train at the evaluation protocol, or keep a preset and record the mismatch as a manifest variable.
  - **Reward arms in AlpaGym: configuration only, one variable apart** (L2-3, corrected by K4):
    - **Arm A:** `progress_safety` as shipped, the blog reward: progress +1, `collision_any` −10, off-road −5, −0.01 × distance to GT.
    - **Arm B (proposed; Huy's ★ choice, not pre-registered):** arm A with `collision_any` replaced by front + lateral collisions, the at-fault definition the acceptance test uses. An episode with both front and lateral contact scores −20 under B and −10 under A; Huy records or caps it.
    - **The comfort/jerk arm waits for a jerk scorer** (K4).
      - The scorer is built with the stall and oscillation scorers below, on AlpaSim's existing `Trajectory.jerk()` utility.
      - The arm becomes configuration-only once AlpaGym's simulator checkout (a config value) points at an AlpaSim commit that has the scorer. That compatibility is unverified.
      - It runs as [O] 4 if S4's hours allow.
    - **First-principles note.** Group standardization makes the advantages invariant to any positive affine change of the reward. On a group where only collisions vary, −10 and −1 give identical updates. Only changes in which events vary together within a group can move the policy (`03` §4 S4).
  - The reasoning–action-consistency arm (AlpaJudge) can only run in the **open-loop** `alpamayo1_x_rl` recipe, after the sign fix. It is labelled open-loop and scored closed-loop by the gate.
  - **[C] Zero-weight monitors** (L2-4): logged at scale 0, never rewarded:
    - `progress_rel`;
    - distance travelled vs ground truth (stall);
    - `wrong_lane`;
    - distance to the lane boundary;
    - distance to the nearest obstacle (a close-encounter proxy);
    - `plan_deviation`;
    - `collision_rear`.
  - A reward-hacking audit: stalling or freezing to avoid collisions, progress gaming, oscillation. Upstream has no scorer for stall or oscillation, so these are new scorers. The audit is read together with S2's stationary-lead stratum, because stalling avoids stationary-lead collisions.
  - **[C] Estimator audit** (L2-2):
    - CPU known-answer tests from Huy's derivation of the estimator as AlpaGym implements it. The ratio is a per-step-row geometric mean over T × 64 × 2 elements (T = `inference_step`, 5 in the policy preset). Its log is clamped to ±5 before the exponential, and it is clipped to [0.8, 1.2]. The KL is the per-transition Gaussian KL, averaged over the same elements;
    - a first-step parity check with `allowed_outdated_steps = 0` and `mu_iterations = 1`: the ratio must equal 1 within ____ (Huy);
    - a parity failure blocks training runs until it is diagnosed.
  - **[C] Telemetry on every run** (L6-5, L7-10, L1-10):
    - `perf.enabled=true`, which reports trainer idle and rollout supply;
    - the merged AlpaSim telemetry;
    - PyTorch's Flight Recorder;
    - NCCL's RAS query on any stall;
    - NVIDIA's measurement definitions (recipes PR #44, open): GPU-hours from the mean wall clock; MFU (6N), HFU (8N) and OFU; severe stall, meaning an arrival spread above 2 s.

    This reconciles the preserved 22-Sep prediction E1: the stock 2-GPU loop leaves the trainer idle ≥ 40%.
  - **[C] The AlpaGym bottleneck analysis and one-node scaling model** (L2-1, L1-7, L6-5):
    - **The document.** A design document plus a sweep from 1 to 8 GPUs that uses shipped presets only: `local_colocated_1gpu`, `local_disaggregated_2gpu`, `slurm_partial_node_1_2_1` (4 GPUs) and `slurm_full_node_1_3_4` (8 GPUs: 1 policy, 3 rollout, 4 AlpaSim).
    - **Varied:** rollout replicas, the policy's `max_batch_size`, rollout batch × `n_generation`, and prefetch.
    - **The model.** Fit the balance N_gen·g ≈ N_train·t and an Amdahl/communication model on held-in configurations. Then predict the held-out ones, and NVIDIA's 3- and 5-node presets.
    - **Where.** The Slurm presets run on a single-node Slurm install with Pyxis and Enroot.
    - **Kill:** AlpaGym's README says the Slurm presets will not run out of the box. If they are not running after 8 agent-hours, keep the two local topologies and label the model "not validated beyond 2–4 GPUs".
  - At least 3 training seeds per arm.
  - **[O] in priority order:**
    1. **A successive-halving hyperparameter study,** carved out of the 225–450 H100-h (about 108 GPU-h; η = 3; about 9 configurations) (L2-5):
       - knobs: SDE noise level, `n_generation`, learning rate, KL β with its reference reset, staleness 0 vs 4, and inference steps 4 vs 5. Dr. GRPO's unbiased advantage is included only if a CPU dry run shows it can be set through the generated config;
       - the rung metric is measured on tuning scenes drawn from the 1,761, never the 441;
       - results are reported as IQM with interval estimates;
    2. **Anytime-valid monitoring** of each arm across saved checkpoints, with an e-process on a fixed monitoring subset, so a futile arm can stop early. The acceptance verdict stays fixed-n or group-sequential (L4-6);
    3. **Replicates allocated by predicted variance** for the rare at-fault endpoint, predicted only from data independent of the current sweep (L4-9);
    4. **The comfort/jerk arm** (K4): arm A plus a jerk term, one variable apart, once the jerk scorer exists.
  - **[G]:**
    - P-012: multi-node validation of the scaling model;
    - P-016: a third arm, either a Lagrangian collision constraint or reactive CAT-K traffic.
- **★ Huy:**
  - the reward design: arm B's term and any cap;
  - the GRPO estimator math **as AlpaGym implements it**: the per-step-row geometric-mean ratio and the Gaussian KL. This is set against AR1's unclipped softmax-weighted variant and the open-loop recipe's k3;
  - the hacking verifier;
  - the balance and scaling model;
  - the protocol choice;
  - the diagnosis.
- **Acceptance.** The owner's target, "at-fault collisions down ≥ 20% relative, with no regression in progress or off-road", becomes three tests on the held-out scenes, per stratum:
  - a one-sided paired verdict on at-fault collisions;
  - non-inferiority on progress;
  - non-inferiority on off-road.

  Its certifiability is read from S1's tables at S2's measured base rate. INSUFFICIENT is a reportable result. Mean ± CI over ≥ 3 training seeds.

  **New in v4.1:** parity is reported, and every estimator known-answer test passes. The scaling model states its prediction error on held-out configurations, with the window and step count behind every mean.
- **Predictions:**
  - relative change in at-fault collisions: ____
  - seed-to-seed SD of the closed-loop score: ____
  - **New in v4.1:**
    - the first hacking mode to appear, under arm A: ____ and under arm B: ____
    - rear-collision rate change from A to B: ____
    - the monitor that moves first: ____
    - parity deviation at step 1: ____
    - steady-state episodes per hour on the 8-GPU preset: ____
    - the share of rollout-worker time spent in the simulator step: ____
    - the fitted serial fraction: ____
    - the predicted speed-up of NVIDIA's 3-node preset: ____
    - the score change caused by the train/evaluation protocol mismatch: ____
  - Also preserved: `04` §7 headline 1 and E1, and headline 5 and E5, as written. Premise notes: the challenge has no close-encounter scorer, and AlpaGym trains only the expert.
- **Kill (KC-4).**
  - If the closed-loop score varies by more than 0.1 across training seeds after 3 training seeds, fall back to open-loop GRPO in alpamayo-recipes.
  - If AlpaGym will not run, skip S4. S5 then uses the released 1.5 as teacher.
  - **New in v4.1:** a first-step parity failure blocks training runs until it is diagnosed.
- **Cost:**
  - **Training:** 225–450 H100-h ⇒ **$403–1,080** at vast.ai $1.79–2.40/h (2026-10-27).
  - **Scaling sweep:** +64–96 GPU-h ⇒ **$64–230**, inside the S4 pilot tier (P-005).
  - **Evaluation:** up to 10 sweeps ($340–1,318).
- **Role evidence (`05` §3):**
  - I1;
  - D1, as a stand-in: a design document and measurement, with no new framework;
  - D3 (reward), D4 and Q3;
  - Q5, through the single-node Slurm presets;
  - X1 only as a stand-in.
- **Upstream value:** the first public closed-loop RL result on AlpaGym, positive or negative, with a hacking audit and a measured bottleneck and scaling model. It is filed as AlpaSim issues and a report, because AlpaGym accepts no PRs.

### S5 — On-policy correction of the student and the CoC-budget curve (the input's P2 b–d)
- **Why.**
  - **Offline distillation never sees the student's own states.** That is what PR #42 does: it trains the student only on the teacher's state distribution. Deployed in closed loop, the student's own small errors take it into states it never saw, so error compounds as O(T²ε). On-policy correction reduces this to O(Tε) (Ross & Bagnell).
  - **The evidence from NVIDIA's own on-policy papers is mixed** (verified 2026-10-27):
    - OPTED ran on one 8 × H100 node (two epochs took 17 h). On 441 held-out NuRec scenes, its LTFv6 student's scene score rose from 26.5 to 41.8 and its distance per infraction from 0.23 to 0.43 km. Yet its per-scene at-fault collision rate also rose, from 10.2% to 23.4%;
    - RoaD uses no teacher model: its guidance is the logged human trajectory;
    - the alpamayo1.5 PR #36 thread shows a 2B Alpamayo policy already running inside AlpaGym. Re-check novelty at release.
  - **What S5 can claim.** PR #42's stage 1 already distils on-policy at the token level, open-loop, on the student's own reasoning. S5's claim is therefore **closed-loop, state-visited** correction.
  - **The latency budget.** The arithmetic says a 2B reasoning student can afford only about 3–20 CoC tokens in 100 ms (review §3.1).
- **Build** (`../nano_alpamayo/README.md`):
  - **(a) DAgger/RoaD-style correction of the S3 student in AlpaSim.** The teacher labels the states the student visits; the teacher is the RL'd 1.5 if KC-4 passed, otherwise the released 1.5. One node.
    - **[C] The learner is PR #42's stage-2 expert-SFT trainer** (L5-1): VLM frozen, expert and action projections trained, with teacher actions entering through the model's `target_action` override. In order:
      - the pilot's first check: the stage-2 batch builder passes that key through;
      - then the sampler parity check;
      - then an overfit-one-batch check;
      - then the loss at init.
    - **[C] The loop runs as a manifest-driven data pipeline** (L5-5):
      - **The manifest, per round:** student and teacher hashes, AlpaSim SHA and lockfile, scene IDs, replicate or seed mode, label stride, `n_sim_steps`, and the trainer commit.
      - **Observations** come from AlpaSim's per-rollout `.asl` logs, which store the camera frames and egomotion the driver received.
      - **Teacher labels** are produced offline by replaying those logs, keyed by (session, timestamp).
      - **Data hygiene:** deduplication keys, and train and evaluation sets disjoint by road.
      - **Throughput** per stage: states collected, labelled and trained per GPU-hour.
      - **Metrics:** NVIDIA's measurement definitions (recipes PR #44, open).
      - **Reproducibility check:** re-running a round from its manifest reproduces its label shards, or the nondeterminism is documented.
    - **[C] The scaling and resiliency ladder,** on S5's 8 × H100 node (L6-6):
      - DDP and FSDP2 at 1, 2, 4 and 8 GPUs, with strong and weak scaling, 3 repeats after warm-up and device-synchronized timing;
      - `nccl-tests` as the communication ceiling;
      - MFU from Huy's FLOPs-per-sample derivation;
      - hang and crash drills under NVIDIA's Resiliency Extension, with the Flight Recorder on.
    - **[O] in priority order:**
      1. a RoaD-style control arm without a teacher, to separate the on-policy state distribution from the switch to teacher targets (L5-2);
      2. token-level on-policy distillation at closed-loop-visited states with PR #42's `on_policy.py`, as a separate variable (L5-9);
      3. a decode micro-benchmark, CPU vs NVDEC (L5-11).
  - **(b) [C] A flow-step ladder anchored on NVIDIA's deployed settings** (L5-3):
    - closed loop: K ∈ {10, 5, 4, 2}, plus FlashDrive's velocity-reuse schedule at equal network evaluations against uniform K = 4;
    - K = 8 and K = 1 run open-loop only;
    - every arm uses the same fixed noise file per scene and replicate (PR #45 ships one), so the arms are paired by common random numbers.
  - **(c) [C] The CoC-budget curve as one clean variable** (L5-4, L1-2):
    - measure the student's natural CoC length first;
    - set caps inside that range;
    - budget forcing on the same checkpoint; n = 0 means the trajectory token immediately;
    - log the realized length per tick;
    - pair caps by scene and noise;
    - latency comes from S6;
    - architecture fixed;
    - prefill reuse of history frames is a prerequisite for any reasoning arm; FlashDrive reports that it costs +2.7% minADE6.

    Label the n = 0 point by architecture: an expert reading an empty CoC is not a tacit-trained model (TacitVLA). An optional arm, a budget-randomized expert SFT, separates truncation shift from the value of the information. The design is Huy's.
  - **(d)** A short GRPO pass on the student, optional.
  - **[G]:** P-018, few-step distillation of the expert, only if the ladder shows a cliff above the K that S6's budget allows.
- **Prerequisites:**
  - the student's license permits fine-tuning. Edge's license is unknown; PR #42's student VLM, Cosmos-Reason2-2B, is under the NVIDIA Open Model License;
  - the AR1 Eq. 5 decoder passes its known-answer test (A5);
  - **a student exists.** If Edge is still not public at S5's start, Huy chooses between two paths: P-017, a pilot student with its limits stated; or running only S5(b)–(c), the ladder and the CoC curve, on the released teacher and labelling them so.
- **★ Huy:**
  - the correction loss and DAgger aggregation math;
  - the decoder math (AR1 Eq. 5);
  - the arm design;
  - the data split and the leakage audit (A1, corrected by K2);
  - FLOPs per sample and the efficiency definitions;
  - the caps.
- **Acceptance:**
  - criteria A1–A7 in `../nano_alpamayo/README.md`; A7 is new: every round reproduces from its manifest;
  - paired verdicts for corrected vs base student, and for corrected student vs teacher;
  - ladders ordered as predicted (G3·R7);
  - the realized-length distribution reported per cap.
- **Predictions:**
  - paired gain from correction: ____
  - shape of the score-vs-CoC-budget curve: ____
  - the K below which the score drops: ____
  - **New in v4.1:**
    - share of round time per pipeline stage: ____
    - strong-scaling efficiency at 8 GPUs, DDP / FSDP2: ____
    - MFU at 8 GPUs: ____
    - hang-detection latency: ____
  - Also preserved: `04` §7 E5.
- **Kill.**
  - **KC-5:** no paired gain after three ablation rounds → publish the negative result with its mechanism. A "round" is one ablation of the correction design, as defined in the experiment file; it is not one DAgger aggregation iteration.
  - **KC-L2:** no corrected weights trained on PAI-AV/NuRec-derived data are published without a written answer on license §4.6.
- **Cost:**
  - **The node:** one 8 × H100 node at $31.92–36.00/h (list prices, 2026-10-27) × 24–48 h ⇒ **$766–1,728**.
  - **Sweeps:** about 8, 6 plus 2 ladder anchors ($272–1,055).
  - **The scaling ladder** (16–24 H100-h) sits inside the pilot.
  - Run-hours are unmeasured; pilot first.
- **Role evidence (`05` §3):** I1, D2, D4, D5, X3.
- **Upstream value:** no public closed-loop, state-visited correction of an NVIDIA student exists, and no measured reasoning-budget curve on a deployable student exists.

### S6 — Serving on Jetson AGX Thor (the input's P3; needs P-002)
- **Why.**
  - NVIDIA staff said on 2026-09-07 that Alpamayo "is not available for AGX Thor currently".
  - The official 1.5 BF16 build runs at 3,322 ms per request on Jetson AGX Thor. FlashRT's best full request is 811 ms, so the 10B teacher cannot reach 10 Hz.
  - NVIDIA's 2B students run at 171 ms (FP8) on **DRIVE** AGX Thor through a forked TensorRT-Edge-LLM.
  - TensorRT-Edge-LLM does not export Alpamayo 1.5. It was still at v0.11.0 on 2026-10-27, and external PR #260 was still open.
  - Jetson runs TensorRT 10.16.2, whose support matrix lists native FP8 and FP4 on sm_110. The "fake quantization" reports concern ModelOpt PyTorch checkpoints, not engines.
  - **New in v4.1:** TensorRT 11.4 (support matrix of 2026-10-05) is current for DRIVE but does not support JetPack; NVIDIA refers to a future 11.x JetPack release. TensorRT 11 removes the weak-typing APIs, including the precision builder flags. `IPluginV2DynamicExt` has been deprecated since TensorRT 10.0 in favour of `IPluginV3`. So C++ written now should target strongly typed networks and `IPluginV3`.
- **Build** on Jetson AGX Thor (JetPack 7.2.1):
  - real (not emulated) engines: NVFP4/FP8 weights and FP8 prefill;
  - a static KV cache with CUDA graphs;
  - a fused flow-step kernel;
  - a camera-batched ViT. Before building it or history-frame reuse, check what TensorRT-Edge-LLM already ships: media-aware KV reuse since v0.10.0, FP8 ViT attention and a ViT-embedding cache since v0.10.1. Consume them rather than rebuild them, which would race NVIDIA (P-008's 2026-10-27 note);
  - streaming reuse of history frames;
  - per-stage latency at p50 and p99, a sustained 30-minute 10 Hz run with thermals, and energy per step;
  - per-format paired closed-loop verdicts (G6);
  - the expert-efficiency diagnosis (v3's S4b): why the flow expert runs at about 20% of its bandwidth bound. That figure is the 2026-10-03 ledger's own derivation, not FlashDrive's (P-007);
  - upstream PRs or issues for whatever breaks. The input's "empty-CoC Thor bug" was an Alpamayo 1 setup problem on DRIVE AGX Thor, fixed by 10-06, so it is not a target (ledger B6);
  - **[C] TensorRT-11-ready C++** (L7-7):
    - engines as strongly typed networks, with explicit Q/DQ from ModelOpt;
    - custom operations as `IPluginV3`;
    - CI builds against TensorRT 10.16 (Jetson) and TensorRT 11.x (x86);
  - **[C] the C++ driver server for S7,** built here (L7-9; see S7).

  P-007 (S6a on a rented RTX PRO 6000 first) and P-008 (a narrower build list) remain open; their 2026-10-27 evidence notes are in `DECISIONS.md`. Any expert fix found in S6a also speeds up S2–S5's sweeps.
- **★ Huy:** the latency model (roofline with KV reads), the fused kernel, the plugin design, the precision verdicts, the profiler diagnosis.
- **Acceptance.** The owner's targets:
  - p99 ≤ 100 ms with 4 cameras;
  - 10 Hz sustained for 30 minutes;
  - open-loop ADE drift ≤ 2% vs BF16, **plus** a closed-loop per-format paired verdict;
  - every precision number labelled "real engine" or "emulated";
  - every number names the device (Jetson AGX Thor, JetPack and TensorRT versions);
  - **new in v4.1:** the same C++ source builds against TensorRT 10.16 and 11.x.
- **Predictions:**
  - p99 per stage for the no-CoC and CoC arms: ____
  - expert efficiency vs its bound after fusion: ____
  - NVFP4 vs FP8 closed-loop verdict: ____
  - Also preserved: `04` §7 headline 4 and E7.
- **Kill (KC-6).** If p99 > 150 ms after S6, drop runtime CoC, cut to K = 2 and shrink the expert. If it is still over budget, kill the 10 Hz claim and report 5 Hz honestly. Without the device (P-002 declined), do the serving work on an RTX PRO 6000 with roofline scaling, labelled "not measured on Thor".
- **Cost:** the Jetson AGX Thor dev kit at $5,499 MSRP (unchanged on 2026-10-27), plus about 3 verdict sweeps ($102–395).
- **Role evidence (`05` §3):** Q4, X3 (algorithm optimization), I2 (performance, through P-007's expert diagnosis).
- **Upstream value:** no 2B Alpamayo student has been shown at 10 Hz on Jetson AGX Thor, and no closed-loop per-format verdict exists.

### S7 — Hardware-in-the-loop with latency injection (the input's P4)
- **Why.** Lockstep HIL scores the policy as if latency were zero. At 10 Hz, 139 ms of latency means each action lands 2 ticks late; at 15 m/s the car travels 2.1 m in that time.
- **Build:**
  - **[C] Through AlpaSim's own mechanisms** (L7-8), with no fork:
    - Thor serves the driver gRPC endpoint (`EgodriverService`) through AlpaSim's `driver_source=external_static` mode, while cloud AlpaSim runs in lockstep, so network delay never touches the score. On-device time is logged per step;
    - **latency injection:** constant-delay arms through `planner_delay_us`, at the measured model latency and at the glass-to-action latency (P-009 ii). Each shows how much score the real latency costs;
    - a per-step, trace-driven delay is added as an AlpaSim PR only if Huy's rule says the constant arms are insufficient.
  - **Delay is quantized to the control step.** At Δt = 100 ms, delays of 139, 171 and 200 ms all apply the plan 2 ticks late. This assumes the buffer is read at control ticks; check it.
  - **[C] A production-grade C++ driver server** (L7-9), built during S6:
    - nvJPEG decode into pre-allocated pinned and device buffers;
    - TensorRT on dedicated streams with a captured fixed-shape step;
    - per-stage CUDA-event timing plus a host monotonic clock for glass-to-action;
    - bounded lock-free single-producer/single-consumer queues;
    - the deterministic `inference_seed` contract;
    - Prometheus histograms;
    - tests: mock-engine unit tests, a golden bitwise replay, `compute-sanitizer` (memcheck, racecheck, synccheck, initcheck), ASan/UBSan/TSan host builds, and clang-tidy.
  - Real-time log replay from NVMe.
- **★ Huy:** lockstep semantics, the latency-injection design, equivalence statistics (TOST).
- **Acceptance:**
  - the Thor-in-loop score equals the cloud score within S2's noise floor (paired equivalence, Huy's margin);
  - the latency-injected score is reported with a CI;
  - **new in v4.1:** p50/p99/p99.9 per stage on Jetson AGX Thor, with the device, JetPack and TensorRT named and the "real engine" label; zero sanitizer findings; the golden replay bitwise.
- **Predictions:** score lost to the measured latency: ____
- **Kill (KC-7):** if lockstep changes policy semantics (timestamps, buffering), report open-loop replay only.
- **Cost:** about $200.
- **Role evidence (`05` §3):** Q4, X2, D3.

### S8 — Shadow mode in Ho Chi Minh City (the input's P5; optional; needs P-006)
- **Why.** HCMC traffic is out of distribution: PAI-AV covers the US and 24 European countries, and no Asian country (ledger A8). Mining where the student disagrees with a human driver builds an OOD set no US lab collects.
- **Gate before anything is mounted (KC-8):**
  - P-006 approved: the rig budget, plus Huy's confirmation of local rules for mounting recording equipment;
  - a privacy plan: notice, and anonymizing faces and plates before any publication.

  Vietnam's smart-vehicle permit (Law 36/2024 Art. 35(3); definition narrowed on 2026-07-01 to vehicles controlled without a driver) appears on its plain wording not to cover a passive rig. **That is an engineering reading, not legal advice; confirm local rules before driving.**
- **Build:**
  - The rig: 4 GMSL2 cameras matched to the training FOVs (120° front-wide, 30° front-tele, 2 × 120° cross), on Leopard's Thor-listed IMX728-9295 series. They connect through a Holoscan Sensor Bridge plus deserializers, because the Jetson dev kit has no native GMSL. PTP sync. RTK GNSS plus an IMU. Read-only CAN.
  - **No actuation path in hardware**, and the human driver is always in control.
  - Student predictions vs the driver's path.
  - A trigger classifier that mines disagreements into an HCMC OOD set (motorbike swarms, missing lane lines, wrong-way riders).
  - A **within-rig contrast**, wide arterials vs dense mixed traffic, to separate scene shift from sensor shift.
- **★ Huy:** the trigger logic, the disagreement metric, the OOD taxonomy, the contrast design.
- **Acceptance:** the owner's targets: ≥ 20 h logged, and the top-k disagreement clips reviewed and labelled.
- **Predictions:** disagreement rate in dense mixed traffic vs arterials: ____
- **Kill (KC-8):** any doubt about legality, privacy or safety → skip S8. S1–S7 stand on their own.
- **Cost:** the rig at ~$5–8k (estimate from verified parts; ledger C6–C7). Cameras $519–539 each; deserializer $388 per camera; HSB board $372.50 (out of stock on 10-25); ZED-X20P $300 plus an IMU; mounts, power and NVMe. The comma four ($999) is optional.
- **Role evidence:** weak for JR2010863 (D2, data engine). S8 is the first cut if hours run short.

### Releases
Each stop point ships an artifact: the open-source repository, a technical report, and (after S6–S8) a demo video with a Thor latency overlay. Each release is reproducible from a fresh clone and subject to KC-L and KC-L2. From v4.1, each S2–S5 report also carries its performance section: the rollout profile, goodput, and the scaling numbers, each with its device and scale named.

## 4. The first seven days: S1 ships a public artifact for $0

`physloop-gate` v0 is a paired release-gate core validated on synthetic truth, plus a 441-scene power analysis. It runs on CPU only and needs no dataset or license.

| Day | ★ Huy | Agents | Output |
|---|---|---|---|
| 1 (10-26) | Write `experiments/2026-10-26-s1-synthetic-truth.md` from the template and fill P-S1-1…5. Derive the exact McNemar null and the zero-failure bound by hand | Scaffold `gate/`, CI and the test harness | Experiment file with predictions |
| 2 | Write the `verdict()` core: paired non-inferiority (Tango CI), the intersection–union rule, the INSUFFICIENT state | Encode the known-answer tests | Core with tests going from failing to passing |
| 3 | Write the road-clustered paired bootstrap and the Durkalski adjustment; specify the generator | Build the generator to that specification | Generator with known truth |
| 4 | Diagnose every Monte Carlo deviation. Specify the group-sequential rule (spending function, K, futility) | Run 1,000-run sweeps (± stratification; null and −30%; 2% and larger strata), plus the sequential sweep, with plots | False-GO, false-NO-GO and INSUFFICIENT rates, fixed and sequential |
| 5 | Power analysis: MDE, NI margins, stratum power and its product across strata; the replicate branch; **scenes needed vs base rate and discordance for S4's relative gate; S3's margin on both score metrics**; boundary tables | Tables and figures; the boundary known-answer tests | "What 441 scenes can certify" |
| 6 | Reconcile prediction vs measurement and explain every miss causally; write the README's claims section. Re-check Edge (KC-E, due 10-31) with the release-day capture list | Packaging, docs | Reconciliation recorded |
| 7 (11-01) | Publish the repo (Huy's action); open the AlpaSim issue; write the journal; send the license letter if not yet sent (P-004). Pre-register S2's seed mode, Harmonizer state and run method | Release notes; draft S2's manifest template with the v4.1 fields | **Public artifact + upstream issue** |

## 5. Budget tiers (prices re-verified 2026-10-27; costs are inferences)

- **Price basis** (vast.ai, on-demand, 2026-10-27; 2026-10-25 in brackets):
  - RTX PRO 6000 Workstation: $1.00 cheapest, $1.56 median (33 offers) [$1.00 / $1.60, 49 offers].
  - RTX PRO 6000 Server: $1.27 / $1.34 [$1.24 / $1.47].
  - H100 SXM: $1.79 / $2.40 [$1.93 / $2.40].
  - B200: $6.49 / $10.25 [$7.50 / $9.38].
  - Method: per offer, `dph_total / num_gpus`; the cheapest, and the median over offers. Reproduce with `python3 -I stats.py` on `research/2026-10-27-role-alignment/notes/vast_2026-10-27/`.
  - An 8 × H100 node: $31.92–36.00/h. These are per-GPU list prices × 8 at Lambda ($3.99), RunPod ($3.99; its tier is unverifiable from the static page) and Nebius ($4.50), against one vast.ai 8-GPU offer at $32.00. Only that vast.ai offer was checked for a 24–48 h rental.
  - Multi-node: RunPod Instant Clusters are the smallest self-serve option (per-second billing, no minimum lease, Slurm supported; H100 cluster price not listed). Lambda's 1-Click Clusters need 16 GPUs for two weeks (about $33k), so they are out of reach.
- **One sweep** (441 scenes × 3 replicates × 200 steps) = 34–85 GPU-h, a single-stream upper bound. S2's measured profile replaces it.

| Tier | Unlocks | Estimate |
|---|---|---|
| **$0** | S1; every curriculum exercise; the AlpaSim issue; the license letter | CPU |
| **Cloud ≤ $2.5k** | S2 + S3, plus an S4 pilot (one reward arm × 3 training seeds, with the scaling sweep) | ≈ $0.6–2.3k plus storage |
| **Cloud ≤ $6k** | S2–S5 in full, plus S7's cloud side | ≈ $2.2–6.7k before the expert-graph saving, ≈ $2.2–6.3k after. The upper end exceeds the tier, so the cap rule below applies |
| **+ Jetson AGX Thor** | S6, S7 | $5,499 (P-002) |
| **+ Rig** | S8 | ~$5–8k (P-006) |
| **Gated additions (P-012 … P-019)** | Each proposal carries its own price; P-012, multi-node, is about $300–640 | Each needs Huy's decision and room in the recorded tier |
| **Collaborator** | PR #42-scale distillation (≥ 128 H100s recommended; ≈ 6,290 H100-h for stage 1 by inference); larger RL | NVIDIA Academic Grant (faculty PIs only); Nebius research credits (academic or non-profit); Inception (incorporated company) |

**Rules:**
- Stop at stage boundaries, never mid-stage.
- Re-price at every stage start.
- If cloud spend passes $4k, cap group sizes, cut the scene count, use interruptible RTX PRO 6000s for rollouts and H100s only for the optimizer.
- The tier choice is P-005, required before S2 spends anything.

## 6. Claims PhysLoop may and may not make

**May.** Every claim states n, CI, replicates and seed mode, manifest (including the resolved lockfile and the simulator configuration), AlpaSim SHA, scenario list, score definition, deviation convention and, for latency, the device.
- "Checkpoint B is not worse than A by more than Δ on every pre-specified stratum, at one-sided 95% confidence, paired and cluster-adjusted by road."
  - Δ is pre-registered on a named metric.
  - Underpowered strata are INSUFFICIENT, and INSUFFICIENT blocks GO.
  - With interim looks, the claim names the spending function and the look at which the verdict stopped.
- "NVIDIA's 2B student is / is not non-inferior to its 10B teacher within Δ in AlpaSim closed loop, under protocol P", or INSUFFICIENT.
- "RL changed at-fault collisions by X (one-sided bound) on n held-out scenes", or INSUFFICIENT.
- "The per-scenario failure rate in stratum S is ≤ X, by a **one-sided** 95% Clopper–Pearson bound."
- "On Jetson AGX Thor (JetPack 7.2.1, TensorRT 10.16.2), student S runs at p50/p99 = … ms per step with 4 cameras (real engine | emulated)."
- "In shadow mode, the student's trajectory disagreed with the human driver on X% of Y hours. This is an open-loop measurement."
- The G3 claims the narrowed protocol licenses: "gate rankings are reliable across replicates (ICC = …)"; "known-order checkpoints ordered correctly k/K".
- Performance claims name the device, the GPU count, the topology and the preset, for example: "On one 8-GPU node, AlpaGym's 1/3/4 preset ran at X episodes/h; the one-node model predicts Y for NVIDIA's 3-node preset (not measured)."

**Must not:**
- Per-mile crash rates, human parity, "above human", "safe" or "safe to deploy".
- Compliance with ISO 26262, ISO 21448, ISO/PAS 8800:2024, UL 4600, UN R157, UN R171, UN R[185] or the UN ADS GTR. "NATM compliance" is not meaningful either, because NATM is guidance.
- "Predictive validity", "validated against the real world", or any on-road correlation not measured.
- **Shadow mode as validation**, "on-road testing", "self-driving in HCMC" or "autonomous". It is open-loop and the human drives.
- **Jetson numbers as DRIVE numbers**, or the reverse. The silicon and bandwidth match, but the OS and the TensorRT major version do not.
- Fake-quantized or emulated precision presented as a speed-up.
- Card scores compared with numbers measured under another protocol.
- Counting ≥ 4 m exits as policy failures, or dropping them, without reporting them separately.
- "First" without a dated novelty check on release day.
- Publishing PAI-AV/NuRec-derived numbers before KC-L, or derived weights or labels before KC-L2. PAI-AV clips, NuRec scenes and auto-labels are never redistributed.
- Calling a PhysLoop monitor "Halos". Presenting paired release-to-release regression as what Waymo or any regulator requires.
- **Unseeded replicates called seeds,** or seed-paired results reported from a run whose manifest records no seeded mode.
- **Single-node results presented as multi-node or cluster scale,** a predicted multi-node number presented as a measured one, or this solo project presented as industry infrastructure experience.

## 7. Kill checks (dated for D = 2026-10-26; shift with D)

| Check | When | Rule |
|---|---|---|
| KC-1 | D+1 (11-02) | `verdict()` controls false-GO on synthetic truth, sequentially as well as at fixed n, and the S3/S4 power tables exist; otherwise fix before any GPU spend. Re-forecast all windows with actual hours; decide which [O] items run, and apply §0.1's cut order if needed |
| KC-E | 2026-10-31 and at S3's start | Edge weights public and licensed for evaluation → S3 as written, with the release-day capture list. Otherwise S4 first, and S3 measures NVIDIA's deployment levers on the released teacher; the PR #42 pilot is P-017 |
| KC-L | Before the first public PAI-AV/NuRec-derived number | A written answer on license §3, or the fallback (code, scene-ID manifests, synthetic results) |
| KC-L2 | Before any release of weights or labels trained on PAI-AV/NuRec-derived data | A written answer on license §4.6; otherwise release code and recipes only |
| KC-2 | D+3 (11-16) | Reliability ≥ Huy's floor and noise floor below the effects S3–S5 must detect; otherwise stop. S2's seed mode, Harmonizer state and run method were pre-registered before the first run |
| KC-3 | D+5 (11-30) | The student verdict is the result. Edge does not run in AlpaSim within 16 GiB → report and fall back |
| KC-4 | D+8 (12-21) | Closed-loop score varies by more than 0.1 across training seeds after 3 training seeds → open-loop GRPO fallback; AlpaGym does not run → skip. A first-step parity failure blocks training until diagnosed |
| KC-5 | D+11 (2027-01-11) | No paired correction gain after three ablation rounds → negative result with mechanism |
| KC-6 | D+12 (01-18) | p99 > 150 ms → no CoC, K = 2, smaller expert; still over → report 5 Hz |
| KC-7 | D+13 (01-25) | Lockstep alters policy semantics → open-loop replay only |
| KC-8 | Before S8 | Legal, privacy or safety doubt, or P-006 not approved → skip S8 |
| KC-$ | Any time | Cloud spend > $4k → the cap rule in §5 |

## 8. Risks

| Risk | Check | Response |
|---|---|---|
| Edge slips, ships without weights, or ships under a license that forbids evaluation or fine-tuning | KC-E, 10-31 | Run S4 first; S3 measures NVIDIA's deployment levers on the teacher; S5 needs a fine-tunable student |
| NVIDIA publishes a closed-loop evaluation of its own student | Every Edge, recipes or AlpaSim release | The gate's independent, paired, stratified verdict keeps its value; contribute the evidence upstream |
| NVIDIA publishes closed-loop RL or correction of a 2B Alpamayo policy. A 2B policy has already been timed inside AlpaGym (alpamayo1.5 PR #36), and AlpaGym's public branches moved until 2026-09-03 | Every stop point | Reword novelty to what stays open: the paired, stratified verdict, the measured loop, the hacking audit. Never claim "first" without the dated check |
| License §3 blocks publication of closed-loop numbers; §4.6 blocks weights | KC-L, KC-L2 | Send the letter now (P-004). Fallback: code, manifests, synthetic results; nuPlan/MTGS-track numbers if Alpamayo-family policies run there (unverified) |
| AlpaSim instability: a mid-challenge evaluator change; a SHA that does not pin the environment; simulation choices withheld until NVIDIA's next release; `main` unsynced since 08-18 | S2 | Start S2 after 10-31 and check for a release; pin the lockfile and the full simulator configuration; record `n_sim_steps`; report the instrument's noise |
| **Unseeded sampling on `e2e_challenge`** (K1) | S2 pre-registration | Run replicates and say so, or port `main`'s reseed and test it; measure seed coupling; never assume it |
| **The Harmonizer changes cost and possibly scores** (K10) | S2 cost pilot | Pin one state before the first run; a flipped-state arm only through P-014 |
| AlpaGym frozen, pinned to an unmerged PR, VLM frozen; its Slurm presets are said not to run out of the box | S4 | Build nothing on its internals; skip or fall back as in KC-4; cap Slurm bring-up at 8 agent-hours |
| **Role-alignment scope creep:** additions crowd out the headline results | KC-1 | §0.1's tiers and cut order. The corrections are never cut; the role items are cut before the S3/S4 verdicts slip |
| **Racing NVIDIA on small upstream pieces:** the expert-graph flag, the `n_sim_steps` field, CAT-K seeding, a Cosmos simulator adapter, a training-efficiency harness | Before any upstream PR | Re-check NVIDIA's post-10-31 sync first. #208's reporter has offered a PR. NVIDIA's PR #44 lists a harness and profiler as coming. No Cosmos3-Nano-Sim-Auto adapter is built |
| Contamination: evaluation scenes overlap the teacher's training clips | S2 contamination report | Report each scene's `alpamayo_aligned` label; exclude train-labelled scenes, or report verdicts with and without them |
| Thor toolchain: TensorRT-Edge-LLM lacks Alpamayo 1.5; PR #45 uses a fork and targets DRIVE; TensorRT 11 excludes Jetson today | S6 | Budget the porting work; label emulated numbers; write TensorRT-11-ready C++; fall back to 5 Hz honestly |
| Underpowered gates (S4's relative effect at low base rates) | KC-1, S2 | INSUFFICIENT is a result; Huy may change the endpoint or effect size before the run |
| Effort under-estimated (≈ 370–405 h after v4.1, unmeasured) | KC-1 | Re-forecast; cut S8 first, then [O], then the [C] role items, then S7 |
| The 12-month license term starts at first download | S2 | Record expiry = first download + 12 months |
| Price and supply moves (vast.ai H100 median $1.69 → $2.40 in three weeks; the RTX PRO 6000 Workstation offer count fell 49 → 33 in two days) | Each stage start | Re-price; interruptible RTX PRO 6000 for rollouts |
| Ownership drift: agents writing ★ cores | Every stage close | Owner proof (`03` §5); otherwise redo the core |
| Regulatory wording drifts | About 2027-01-30 (R[185]) | Update the claims wording; never claim compliance |

## 9. Open owner decisions

These decisions block specific stages. Recording each one closes its proposal in `DECISIONS.md`.

| Proposal | Decision | Blocks |
|---|---|---|
| P-002 | Buy a Jetson AGX Thor dev kit ($5,499 MSRP). Its numbers do not transfer 1:1 to DRIVE AGX Thor: same silicon and bandwidth, but JetPack/TensorRT 10.16 vs DriveOS 7.2/TensorRT 11.x | S6, S7 |
| P-003 | Weekly hours | Calendar accuracy; which [O] items run; re-forecast at KC-1 |
| P-004 | Send NVIDIA the written license request (Huy's outward action), now covering §3 (aggregate closed-loop results, scene-ID manifests) **and** §4.6 (weights and labels trained on PAI-AV/NuRec-derived data) | KC-L, KC-L2: every public number from S2 on |
| P-005 | Cloud budget tier ($0 / ≤ $2.5k / ≤ $6k / collaborator) | Any spend in S2 |
| P-006 | Shadow mode: rig budget (~$5–8k), Huy's confirmation of local rules, privacy plan | S8 |
| P-012 | One multi-node scaling measurement (2 × 8 H100 for S5's trainer, or AlpaGym's 3-node preset), after the one-node models exist | Multi-node evidence for D1/Q5 |
| P-013 | Minimal R3: log replay vs CAT-K on one SHA | Traffic-model evidence (I2) |
| P-014 | Minimal R2/R4: render fidelity without novel-view ground truth, a policy-sensitivity probe, a closed-loop Harmonizer-agreement arm | Renderer-quality evidence (I2, D3) |
| P-015 | A GPU-batched Alpamayo driver behind the P-009 ladder and a known-null paired verdict | Batched sweeps in S3–S5 |
| P-016 | A third S4 arm: a Lagrangian collision constraint or reactive traffic | — |
| P-017 | A PR #42 stage-1 pilot student as S3's fallback | — |
| P-018 | Few-step distillation of the flow expert, only if S5's ladder shows a cliff | — |
| P-019 | One C++ driver for two targets: S6a's runtime serves AlpaSim's student sweeps (needs P-007) | — |
| Per stage, pre-registered in `experiments/` | Floors and margins: reliability, Δ for S3 and S5 on a named metric, S4's endpoint and effect, S7's equivalence margin. **New in v4.1:** S2's seed mode, Harmonizer state and run method; S3's inference target and sentinel margin; S1's spending function and futility rule; S4's protocol choice and arm B's cap; S2's goodput definition | The matching KC |

## 10. Key primary sources

The full ledgers are `research/2026-10-27-role-alignment/verification.md`, `research/2026-10-25-teacher-to-thor/verification.md` and `research/2026-10-03-verification/report.md` §10.
- **Decision inputs and reviews:**
  - `research/2026-10-25-teacher-to-thor/decision_input_2026-10-25.md`, `review.md`, `notes/arith_check.py`;
  - `research/2026-10-27-role-alignment/report.md`, `notes/arith_check.py`.
- **Alpamayo:**
  - AR1 https://arxiv.org/abs/2511.00088;
  - cards https://huggingface.co/nvidia/Alpamayo-1.5-10B · https://huggingface.co/nvidia/Alpamayo2-Super;
  - Edge announcement https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion · https://blogs.nvidia.com/blog/open-models-europe-industry/;
  - expert CUDA graph https://github.com/NVlabs/alpamayo1.5/pull/36;
  - TacitVLA https://arxiv.org/abs/2610.17540.
- **NVIDIA repos:**
  - AlpaSim https://github.com/NVlabs/alpasim: issues #72, #128, #198, #207, #208, #222, #224; PRs #149, #167, #215;
  - AlpaGym https://github.com/NVlabs/alpagym and cosmos-rl https://github.com/nvidia-cosmos/cosmos-rl;
  - recipes https://github.com/NVlabs/alpamayo-recipes: PRs #18, #38, #40, #42, #44, #45;
  - AlpaJudge https://github.com/NVlabs/alpajudge;
  - TensorRT-Edge-LLM https://github.com/NVIDIA/TensorRT-Edge-LLM (issue #130, PR #260).
- **Cluster stack:**
  - Enroot https://github.com/NVIDIA/enroot · Pyxis https://github.com/NVIDIA/pyxis;
  - dcgm-exporter https://github.com/NVIDIA/dcgm-exporter;
  - NVIDIA Resiliency Extension https://github.com/NVIDIA/nvidia-resiliency-ext;
  - nccl-tests https://github.com/NVIDIA/nccl-tests.
- **Thor:**
  - Jetson Thor blog https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/;
  - JetPack 7.2.1 https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1;
  - TensorRT support matrix https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html;
  - FlashRT https://github.com/flashrt-project/FlashRT/pull/229;
  - FlashDrive https://arxiv.org/abs/2608.12932.
- **Data:** https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles · https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec · devkit https://github.com/NVlabs/physical_ai_av.
- **Post-training:**
  - RoaD https://arxiv.org/abs/2512.01993;
  - OPTED https://arxiv.org/abs/2609.20756;
  - DAgger https://arxiv.org/abs/1011.0686;
  - Flow-GRPO https://arxiv.org/abs/2505.05470;
  - closed-loop blog https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/.
- **Statistics:**
  - Jennison & Turnbull, *Group Sequential Methods* (2000);
  - Lan & DeMets (1983);
  - Howard et al., time-uniform confidence sequences (2021).
- **Regulation and law:**
  - NHTSA 91 FR 48485 https://www.govinfo.gov/content/pkg/FR-2026-07-31/html/2026-15483.htm;
  - C.N.288.2026 https://treaties.un.org/doc/Publication/CN/2026/CN.288.2026-Eng.pdf;
  - Vietnam Law 36/2024 (consolidated VBHN 55) https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf;
  - RAND https://www.rand.org/pubs/research_reports/RR1478.html.
