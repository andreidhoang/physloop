# 02 — PhysLoop plan v4 (25 October 2026)

**Status:** authoritative (see `../DOC_AUTHORITY.md`). It replaces v3 of 2026-10-03, which is archived in `archive/`. Adopted by DECISIONS D-012. PhysLoop is a standalone, active project (D-009) and takes no hours, budget, deadline or decision from any other program.

**Evidence base:**
- `research/2026-10-25-teacher-to-thor/`: the owner's decision input, the review, the verification ledger and the arithmetic check;
- `research/2026-10-03-verification/`: still valid wherever the 2026-10-25 ledger does not supersede it.

Inferences are labelled as inferences. The predictions, floors and margins are Huy's: every slot marked `____` stays blank until he fills it in `../experiments/` before the run.

---

## 0. What changed from v3

The owner's decision input (2026-10-25) made the teacher → student → Thor loop the #1 project. The review and verification kept that direction, amended its gates, and found that NVIDIA had moved first on the student. That activates the input's own contingency ("use Edge as the P2 baseline").

| Area | v3 (2026-10-03) | v4 | Why (evidence) |
|---|---|---|---|
| Headline | G3, the gate's own construct validity | The teacher → student → Thor loop, every closed-loop number measured by the gate | Owner's decision; review §1 |
| The gate | S1–S2, then validity studies | S1–S2 unchanged in substance; it is the instrument for every later stage | The input's gates (P0 noise floor, P1 effect size, P2 margin) are outputs of `verdict()` and the power tables (review §3) |
| Student | nanoAlpamayo trained by us | **NVIDIA's own 2B student** (Alpamayo 2 Edge, once released) is measured first. Our work is the closed-loop correction and the reasoning-budget curve on top of it | PR #42 (public, unmerged distillation recipe, ≥ 128 H100s recommended, no weights); Edge announced 10-21, download promised "later this month" (ledger §0) |
| Teacher RL | Not scheduled (AlpaGym fragile) | S4: first published closed-loop gains (or negative result) from AlpaGym, with a reward-hacking audit | AlpaGym has published no gains; a reasoning reward cannot move the frozen VLM there (ledger B1) |
| Thor | Collaborator tier only | S6 on Jetson AGX Thor if P-002 is approved; HIL in S7 | The input's choice; Jetson runs TensorRT 10.16.2, while DRIVE AGX Thor runs TensorRT 11.x (ledger C3) |
| Real car | Absent | S8 shadow mode in HCMC, **optional**, gated on P-006 | The input's choice. Shadow mode is open-loop measurement, not validation (§6) |
| Deferred, not deleted | — | G3 renderer and traffic agreement (R2–R4), cross-paradigm convergence (R6), private set (R5); G4 guardrail; the stand-alone G7 report | Scope. Reliability (R1) and the validity records stay in S2. Known-order ladders come free from S5–S6 (R7/R9). The v3 stage cards remain in `archive/02_PHYSLOOP_PLAN_v3_2026-10-03.md` |
| Publication | KC-L for numbers | KC-L, plus **KC-L2** for weights and labels derived from PAI-AV/NuRec | Data license §4.6 bans derivatives and distribution (ledger A8) |
| Manifest | Git hash, config, seeds, both AlpaSim SHAs | Adds the resolved lockfile, `n_sim_steps` requested and actual, the score definition, and the overlap with PAI-AV's `alpamayo_aligned` split | AlpaSim issues #207 and #208; PAI-AV v26.10 (ledger B3, A8) |
| Calendar | D = 2026-10-05; S1 never started | D = **2026-10-26** | No `gate/`, `journal/` or experiment file existed on 2026-10-25 |
| Budget | $0 / ≤ $2.5k / ≤ $6k / collaborator | Same cloud tiers, plus two separate hardware lines (Thor, rig) | Review §3.5; prices re-verified 2026-10-25 |

**What survived** (ledger §0, B3):
- No upstream paired, stratified release verdict exists.
- No closed-loop evaluation of any NVIDIA 2B student has been published; NVIDIA's PRs report open-loop minADE and latency only.
- No published closed-loop RL gains from AlpaGym exist.
- No 2B student has been shown at 10 Hz on Jetson AGX Thor.

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

**Research question.** Under a fixed on-device latency budget, how does closed-loop driving quality change with the runtime Chain-of-Causation (CoC) token budget? This is the XPeng-vs-NVIDIA split ("language is a bottleneck" vs reasoning on the hot path), measured as a curve rather than a binary. It also asks what NVIDIA traded to reach 10 Hz: its own 2B students measure 171 ms (FP8) on DRIVE AGX Thor, yet Edge is announced at 10 Hz.

**Owner's targets** (from the decision input; each is re-stated on a named metric before pre-registration):
- the deployable student is non-inferior to its teacher within a margin of 10% of the teacher's score, paired, on one pinned run;
- it fits 16 GiB;
- it runs at p99 ≤ 100 ms with 4 cameras on Jetson AGX Thor;
- it sustains 10 Hz for 30 minutes.

## 2. Deliverables, re-scoped against what upstream ships (2026-10-25)

| G | Upstream already ships | Still open, our new work | Verdict | Stage |
|---|---|---|---|---|
| **G1 Gate v1** | Per-rollout scorers; `results-summary.json`; seeds (PR #149); the leakage-aware 1,761/441 split (PR #167); Drive-IRT panel ranking | Paired two-checkpoint verdict; stratification; seed test–retest; per-stratum bounds; allowed-claims list | **Keep** | S1, S2 |
| **G2 Validity records** | `dist_to_gt_*` scorers; 4 m corridor scored as failure on `e2e_challenge` | Per-rollout deviation bins; out-of-envelope rollouts reported separately | **Keep**, inside every report (no separate stage) | S2 onward |
| **G3 Gate validity** | Sim2Val (1 policy); OmniDreams (n = 4) | Reliability (R1); known-group ladders from the pipeline (flow steps, bit-widths, CoC budget: R7/R9) | **Narrowed.** R2–R6 deferred | S2, S5, S6 |
| **G4 Guardrail** | `safety_monitor_safe` hook and scorer | Monitor, override path, policy+monitor scoring | **Deferred** | — |
| **G5 Student** | PR #42 recipe (open, no weights, open-loop only); PR #45 Thor inference (open, ~171 ms FP8 on DRIVE Thor); Edge announced | Closed-loop paired verdict of NVIDIA's student; on-policy correction; CoC-budget curve | **Headline** | S3, S5 |
| **G6 Precision** | ModelOpt 0.43 PTQ including the expert; open-loop minADE only | Real-engine (TensorRT 10.16) per-format closed-loop verdicts on Jetson AGX Thor | **Keep**, moved to the device | S6 |
| **G7 Throughput and latency** | AlpaSim Prometheus telemetry; FlashRT Jetson numbers for the 10B teacher | Measured AlpaSim throughput; student latency profile on Jetson Thor; HIL with latency injection | **Keep**, split across stages | S2, S6, S7 |
| **G8 Flywheel** | CoC autolabeler; Instant NuRec | Shadow-mode trigger data engine on HCMC traffic | **Keep, optional** | S8 |
| **G9 Teacher RL study** | AlpaGym (expert-only GRPO, VLM frozen, code-frozen since 2026-07-08) | First published closed-loop gains or negative result; reward-hacking audit | **New** | S4 |

## 3. Stages: one dependency graph

```
S1 ──► S2 ──┬──► S3 student verdict ──┬──► S5 closed-loop correction + CoC-budget curve ──┐
            │                         └──► S6 Jetson Thor serving (P-002) ◄───────────────┘
            └──► S4 teacher RL ────────────► (RL'd teacher labels S5 if KC-4 passes)
                                                    S6 ──► S7 HIL + latency injection ──► S8 shadow mode (optional, P-006)

KC-E: Edge weights public? Re-check 2026-10-31. If not public at S3's start, run S4 first (the input's order).
KC-L / KC-L2: no public PAI-AV/NuRec-derived number, weight or label without a written license answer (P-004).
```

**Stop points.** Each one ships a public artifact, subject to KC-L and KC-L2:
- after S3: the first closed-loop verdict on NVIDIA's student;
- after S4: the RL study (the input's candidate #2);
- after S6–S7: the Thor serving study (candidate #3).

S3 and S4 depend only on S2, so their order is reversible.

D is the start date of S1: **Monday 2026-10-26**. The windows assume 25–30 focused hours a week. That figure is unconfirmed (P-003), so re-forecast every window after S1's actuals at KC-1. The effort figures are planning inferences.

| Stage | Serves | Window | Calendar (D = 2026-10-26) | Effort (h) | Cost (inference) | Kill check |
|---|---|---|---|---|---|---|
| **S1** Gate core on synthetic truth + power analysis | G1 | D+0 → D+1 | 10-26 → 11-02 | ~20 | $0 | KC-1 |
| **S2** Reproduce + reliability on AlpaSim | G1, G2, G3·R1, G7 | D+1 → D+3 | 11-02 → 11-16 | ~40 | $136–544 + storage | KC-2, KC-L |
| **S3** Closed-loop verdict on NVIDIA's 2B student | G5 | D+3 → D+5 | 11-16 → 11-30 | ~30 | $102–748 | KC-3, KC-E |
| **S4** Closed-loop RL on the teacher | G9 | D+5 → D+8 | 11-30 → 12-21 | ~50 | $774–2,440 | KC-4 |
| **S5** On-policy correction + CoC-budget curve | G5, G3·R7 | D+8 → D+11 | 12-21 → 2027-01-11 | ~60 | $564–2,352 | KC-5, KC-L2 |
| **S6** Jetson AGX Thor serving | G6, G7, G3·R9 | D+6 → D+12 | 12-07 → 2027-01-18 | ~60 | Thor $5,499 (P-002) + ~$100–400 | KC-6 |
| **S7** HIL + latency injection | G7 | D+12 → D+13 | 01-18 → 01-25 | ~20 | ~$200 | KC-7 |
| **S8** Shadow mode, optional | G8 | D+13 → D+15 | 01-25 → 02-08 | ~40 | Rig ~$5–8k (P-006) | KC-8 |

The total is about 320 focused hours, roughly 11–13 weeks at 25–30 h/week. The overlapping windows assume S6's device work runs alongside S4–S5. If that is not realistic for one person, KC-1 re-forecasts the windows.

Each stage card below gives the Why as a verified symptom. ★ marks the cores Huy writes: the measurement instruments, statistics, RL and distillation loss math, verifier logic, predictions and diagnoses. Agents write plumbing, harnesses, scaffolds, failing tests, sweeps and plots. The derivations, known-answer exercises and traps for each stage are in `03_MASTERY_PLAN.md` §4.

### S1 — Gate decision core on synthetic truth, plus power analysis ($0, CPU)
- **Why.** On 441 paired scenes at 4% discordance, 80% power detects only about 2.72 pp. The decision input's RL gate ("at-fault collisions down ≥ 20% relative") is 1.6 pp at an 8% base rate. It needs about 964 paired scenes at ψ = 4%, so on 441 its correct output is INSUFFICIENT (review §3.2).
- **Build:** `gate/`, as specified in v3:
  - the identity tuple and `is_pair()`;
  - `verdict()` returning GO / NO-GO / INSUFFICIENT, where **INSUFFICIENT blocks GO**;
  - a synthetic generator with known effects, road clusters and strata;
  - a 1,000-run Monte Carlo;
  - power tables ("what 441 scenes can certify"), now including:
    - scenes needed for a relative effect as a function of base rate and discordance;
    - the non-inferiority margin on both candidate score metrics. The challenge `scene_score` in [0, 1] is continuous, so its branch is MDE = (z_α + z_β)·σ_d/√n as a function of σ_d, the SD of paired differences (Tango is binary-only). The card-style km-between-events score is a rate ratio;
  - known-answer tests.

  It consumes AlpaSim's per-rollout outputs keyed by scene ID, and needs no dataset license.
- **★ Huy:** `verdict()`, which includes
  - paired non-inferiority with a pre-registered margin (Tango score CI);
  - an intersection–union rule across strata;
  - Holm for regression flags;
  - cluster-adjusted McNemar (Durkalski) or a road-level paired bootstrap;
  - the one-sided zero-failure bound p_U = 1 − α^(1/n);
  - the power derivations, including the rate-ratio case;
  - the predictions.
- **Acceptance.** Unchanged from v3. The known-answer tests reproduce:
  - 12 vs 6: one-sided 0.11894, two-sided 0.23788;
  - 0/441: 0.677% one-sided, 0.833% two-sided;
  - 0/1,000: 0.2991% one-sided, 0.3682% two-sided;
  - MDE 1.94 / 2.72 / 3.90 pp at ψ = 2.1 / 4 / 8%;
  - non-inferiority margins 2.67 / 3.77 pp;
  - power 0.22.

  The Monte Carlo must also show three things: INSUFFICIENT (never GO) for a 9-scene stratum; higher false-GO when stratification is removed; and false-GO ≤ α wherever power exists. One new test: Connor's n for a 1.6 pp effect at ψ = 4% (one-sided, 80%) is 964.
- **Predictions:**
  - P-S1-1, false-GO for a −30% effect on a 2% stratum at 7% base failure: ____
  - P-S1-2, share of INSUFFICIENT under the null: ____
  - P-S1-3, change in false-GO without stratification: ____
  - P-S1-4, the base at-fault rate at which a 20% relative cut becomes decidable on 441 scenes × 3 seeds: ____
- **Kill (KC-1, D+1).** If `verdict()` cannot hold false-GO ≤ α on synthetic truth, fix the statistics before any rented GPU-hour. If S1 runs past twice its estimate, ship the power-analysis half alone.
- **Upstream value.** Publish the repository, then open an AlpaSim issue proposing the paired layer for `e2e_challenge/local_evaluation`.

### S2 — Reproduce and measure reliability on AlpaSim (the input's P0)
- **Why.**
  - NVIDIA's own Alpamayo 1.5 card moved 0.81 ± 0.01 → 1.37 ± 0.10 with unchanged weights, and never defines the score or "±".
  - The `e2e_challenge` evaluator changed mid-challenge (corridor fix #215, 10-20).
  - One AlpaSim SHA resolved about 25 packages differently on two machines (#207).
  - NVIDIA withholds some "simulation choices" until the challenge ends on 10-31 (#224).

  Without a pinned protocol and a measured noise floor, no later number means anything.
- **Timing.** S2 starts 11-02, after the challenge closes. First re-read AlpaSim for post-challenge disclosures. A run made before that is labelled provisional.
- **Build:** a G1 v1 report on three policies:
  - TransFuser (AlpaSim's provisional LTFv6 baseline driver), as a pipeline sanity check;
  - Alpamayo 1.5 in BF16;
  - Alpamayo 1.5 in FP8 (ModelOpt recipe).

  It runs 441 `nurec_curated_val` scenes × ≥ 3 seeds. The report contains:
  - per-stratum paired verdicts with one-sided bounds;
  - seed test–retest;
  - a negative control: a bit-identical re-export must not be flagged;
  - per-rollout validity records in deviation bins < 1, 1–2, 2–4 and ≥ 4 m, with `left_corridor_laterally` split out;
  - the failure taxonomy mapped onto the NHTSA pre-crash typology (as v3);
  - **both score metrics**: the challenge `scene_score`, and the card-style km-between-events rate. The rate is labelled as a diagnostic under the paper's convention, which excludes > 4 m;
  - the numbers that S3–S5's power depends on: base rates, discordance, and σ_d of the paired `scene_score` differences between two policies. A 7 pp margin is decidable on 441 scenes only if σ_d ≲ 0.59 (review §3.3);
  - **the contamination report**: overlap of the 441 and 1,761 scenes with PAI-AV's `splits/alpamayo_aligned.parquet`;
  - the **first measured throughput**, which re-baselines every cost in this plan;
  - Prometheus dashboards.
- **Manifest:** git hash, config, seeds, both AlpaSim SHAs, the **resolved lockfile**, `n_sim_steps` requested and actual, GPU edition, scene list, deviation convention, score definition.
- **Card match.** Matching the card's 1.37 ± 0.1 is a **diagnostic, not a gate** (review §3.4). If the gap persists, open an AlpaSim issue with logs and report relative results only.
- **★ Huy:** reliability statistics (ICC(2,1), Kendall's W, design effect 1 + (m − 1)ρ), stratum definitions, taxonomy mapping, validity-record semantics, the contamination rule, and the diagnosis of every discordance.
- **Acceptance.** ICC and W reported with CIs, along with the noise floor and the negative-control flag rate. Every rollout carries a validity record. Every retry is logged, because Alpamayo prefill intermittently emits all-NaN trajectories. The contamination overlap is stated.
- **Predictions** (preserved from v3):
  - (i) test–retest discordance rate of Alpamayo 1.5 across seeds: ____
  - (ii) % of rollouts beyond 2 m and 4 m, and whether it differs between good and bad policies: ____
  - the pre-registered reliability floor: ____
  - New: the at-fault collision base rate of Alpamayo 1.5 under the pinned protocol: ____
  - Also preserved: `04` §7 E4 (render at model resolution), optional.
- **Kill.**
  - **KC-2, D+3:** if reliability is below the pre-registered floor, or the noise floor exceeds the effects that S3–S5 must detect → stop and redesign.
  - **KC-L:** no public number derived from PAI-AV or NuRec without a written answer on license §3 (P-004). The fallback is to publish code, scene-ID manifests and synthetic results.
- **Cost:** 4 sweeps × 34–85 GPU-h = 136–340 GPU-h ⇒ **$136–544** on RTX PRO 6000 at $1.00–1.60/h. Add $40–216/month of storage for the ~2.9 TB NuRec release. PAI-AV/NuRec are first downloaded here, and the 12-month license clock starts then.
- **Upstream value:** the first public seed test–retest and reliability numbers for AlpaSim, plus the manifest fields that issues #207 and #208 show are missing.

### S3 — Closed-loop verdict on NVIDIA's 2B student (the input's P2 gate, measured on NVIDIA's artifact)
- **Why.** NVIDIA's distillation PR #42 reports only open-loop minADE6: 0.8875 m for the 1.5-2B student vs 0.8298 m for the teacher, +6.95% worse. Its Thor PR #45 reports only latency. Edge is announced at 10 Hz on DRIVE AGX Thor with no published closed-loop numbers. Open-loop error shows no significant correlation with closed-loop score (Bosch, n = 7; `01` C3). So *did distillation make the deployable model worse in closed loop, and where?* is unanswered for NVIDIA's own student. It is exactly the gate's question.
- **Build.** Paired, stratified closed-loop verdicts on the S2 protocol:
  - Alpamayo 2 Edge (2B) vs Alpamayo 1.5 (10B), in BF16 and FP8;
  - Edge vs Alpamayo 2 Super on H100, if affordable;
  - Edge with runtime reasoning on vs off, if the release exposes the switch;
  - a 16 GiB check;
  - validity records and both score metrics;
  - known-order pairs where they exist, such as Edge's flow-step settings (G3·R7).

  The report states which teacher Edge was distilled from, or that it is undisclosed.
- **★ Huy:** the margin Δ on the named metric (the owner's target is 10% of the teacher's score), the strata, and the diagnosis of every stratum where the student loses.
- **Acceptance:** the pre-registered verdict, GO, NO-GO or INSUFFICIENT, applied as written. Every claim uses `§6` wording.
- **Predictions:**
  - paired student–teacher difference on the named metric: ____
  - the strata where the student loses most: ____
  - validity-exit rate (≥ 4 m) of student vs teacher: ____
- **Kill.**
  - **KC-3:** the verdict is the result, whatever it says. If Edge does not run inside AlpaSim's driver container within 16 GiB, report that, and use the fallback.
  - **KC-E, checked 2026-10-31 and at S3's start:** if Edge weights are not public, or Edge's license forbids evaluation, run S4 first (the input's order). The S3 alternative is a PR #42 student trained at pilot scale, behind a cost cap Huy sets. PR #42 recommends at least 128 H100s, so full scale is collaborator-tier only.
- **Cost:** 3–4 sweeps on RTX PRO 6000 ($102–544), plus an optional Alpamayo 2 Super sweep on H100 ($66–204).
- **Upstream value:** the first closed-loop, paired, stratified evaluation of an NVIDIA student, as a report and an AlpaSim issue. It is novel only if no one has published it by release day, so re-check then.

### S4 — Closed-loop RL on the teacher (the input's P1)
- **Why.** AlpaGym has shipped since June with no published gains. Its last code change was 2026-07-08, it has had zero outside PRs, and it pins the unmerged recipes PR #18. Goodhart is live: in AR1, a reasoning-only reward raised the critic grade 3.1 → 4.5 while consistency fell 0.62 → 0.53. The open-loop recipe also rewarded worse reasoning until a sign fix on 10-06 (#40).
- **Build:**
  - GRPO on Alpamayo 1.5 in AlpaGym, **expert only, with the VLM frozen as shipped** (ledger B1). Train on the 1,761 split; evaluate on the 441 held-out scenes.
  - Reward ablation in AlpaGym:
    - the blog reward: progress +1, `collision_any` −10, off-road −5, −0.01 × distance to GT;
    - the same, plus comfort/jerk.
  - The reasoning–action-consistency arm (AlpaJudge) can only run in the **open-loop** `alpamayo1_x_rl` recipe, after the sign fix. It is labelled open-loop and scored closed-loop by the gate.
  - A reward-hacking audit: stalling or freezing to avoid collisions, progress gaming, oscillation. Upstream has no scorer for stall or oscillation, so these are new scorers.
  - At least 3 training seeds per arm.
- **★ Huy:** reward design, the GRPO estimator math (AlpaGym's clipped trajectory-level ratio vs AR1's unclipped softmax-weighted variant; k3 KL), the hacking verifier, and the diagnosis.
- **Acceptance.** The owner's target, "at-fault collisions down ≥ 20% relative, with no regression in progress or off-road", becomes three tests on the held-out scenes, per stratum:
  - a one-sided paired verdict on at-fault collisions;
  - non-inferiority on progress;
  - non-inferiority on off-road.

  Its certifiability is read from S1's tables at S2's measured base rate. INSUFFICIENT is a reportable result. Mean ± CI over ≥ 3 seeds.
- **Predictions:**
  - relative change in at-fault collisions: ____
  - the first hacking mode to appear: ____
  - seed-to-seed SD of the closed-loop score: ____
  - Also preserved: `04` §7 headline 5 and E5, as written. Premise notes: the challenge has no close-encounter scorer, and AlpaGym trains only the expert.
- **Kill (KC-4).**
  - If the closed-loop score varies by more than 0.1 across seeds after 3 seeds, fall back to open-loop GRPO in alpamayo-recipes.
  - If AlpaGym will not run, skip S4. S5 then uses the released 1.5 as teacher.
- **Cost:** training 225–450 H100-h ⇒ $434–1,080 at vast.ai $1.93–2.40/h, plus up to 10 evaluation sweeps ($340–1,360).
- **Upstream value:** the first public closed-loop RL result on AlpaGym, positive or negative, with a hacking audit. It is filed as AlpaSim issues and a report; AlpaGym accepts no PRs.

### S5 — On-policy correction of the student and the CoC-budget curve (the input's P2 b–d)
- **Why.** Offline distillation, which is what PR #42 does, trains the student only on the teacher's state distribution. Deployed in closed loop, its own small errors take it into states it never saw, so error compounds as O(T²ε). On-policy correction reduces this to O(Tε) (Ross & Bagnell; RoaD and OPTED report large sample-efficiency gains in AlpaSim). Separately, the latency arithmetic says a 2B reasoning student can afford only about 3–20 CoC tokens in 100 ms (review §3.1).
- **Build** (`../nano_alpamayo/README.md`):
  - (a) DAgger/RoaD-style correction of the S3 student in AlpaSim. The teacher labels the states the student visits; the teacher is the RL'd 1.5 if KC-4 passed, otherwise the released 1.5. One node.
  - (b) A flow-step ladder, K = 10 → 4 → 2.
  - (c) The **CoC-budget curve**: closed-loop score vs runtime token budget n ∈ {0, …}, with architecture fixed and latency taken from S6. The design is Huy's; prefill reuse of history frames is a prerequisite for any reasoning arm. FlashDrive reports that this reuse costs +2.7% minADE6.
  - (d) A short GRPO pass on the student, optional.
- **Prerequisites:** Edge's license permits fine-tuning; the AR1 Eq. 5 decoder passes its known-answer test (A5).
- **★ Huy:** the correction loss and DAgger aggregation math, decoder math (AR1 Eq. 5), the arm design, the data split and the leakage audit (exclude `alpamayo_aligned` clips from evaluation).
- **Acceptance:** criteria A1–A5 in `../nano_alpamayo/README.md`. Paired verdicts for corrected vs base student and corrected student vs teacher. Ladders ordered as predicted (G3·R7).
- **Predictions:**
  - paired gain from correction: ____
  - shape of the score-vs-CoC-budget curve: ____
  - the K below which the score drops: ____
  - Also preserved: `04` §7 E5.
- **Kill.**
  - **KC-5:** no paired gain after three ablation rounds → publish the negative result with its mechanism.
  - **KC-L2:** no corrected weights trained on PAI-AV/NuRec-derived data are published without a written answer on license §4.6.
- **Cost:** one 8×H100 node at $15–32/h × 24–48 h ⇒ $360–1,536, plus about 6 sweeps ($204–816). Run-hours are unmeasured; pilot first.
- **Upstream value:** no public closed-loop correction of an NVIDIA student exists, and no measured reasoning-budget curve exists.

### S6 — Serving on Jetson AGX Thor (the input's P3; needs P-002)
- **Why.**
  - NVIDIA staff said on 2026-09-07 that Alpamayo "is not available for AGX Thor currently".
  - The official 1.5 BF16 build runs at 3,322 ms per request on Jetson AGX Thor. FlashRT's best full request is 811 ms, so the 10B teacher cannot reach 10 Hz.
  - NVIDIA's 2B students run at 171 ms (FP8) on **DRIVE** AGX Thor through a forked TensorRT-Edge-LLM.
  - TensorRT-Edge-LLM does not export Alpamayo 1.5.
  - Jetson runs TensorRT 10.16.2, whose support matrix lists native FP8 and FP4 on sm_110. The "fake quantization" reports concern ModelOpt PyTorch checkpoints, not engines.
- **Build** on Jetson AGX Thor (JetPack 7.2.1):
  - real (not emulated) engines: NVFP4/FP8 weights and FP8 prefill;
  - a static KV cache with CUDA graphs;
  - a fused flow-step kernel;
  - a camera-batched ViT;
  - streaming reuse of history frames;
  - per-stage latency at p50 and p99, a sustained 30-minute 10 Hz run with thermals, and energy per step;
  - per-format paired closed-loop verdicts (G6);
  - the expert-efficiency diagnosis (v3's S4b): why the flow expert runs at about 20% of its bandwidth bound;
  - upstream PRs or issues for whatever breaks. The input's "empty-CoC Thor bug" was an Alpamayo 1 setup problem on DRIVE AGX Thor, fixed by 10-06, so it is not a target (ledger B6).
- **★ Huy:** the latency model (roofline with KV reads), the fused kernel, the precision verdicts, the profiler diagnosis.
- **Acceptance.** The owner's targets:
  - p99 ≤ 100 ms with 4 cameras;
  - 10 Hz sustained for 30 minutes;
  - open-loop ADE drift ≤ 2% vs BF16, **plus** a closed-loop per-format paired verdict;
  - every precision number labelled "real engine" or "emulated";
  - every number names the device (Jetson AGX Thor, JetPack and TensorRT versions).
- **Predictions:**
  - p99 per stage for the no-CoC and CoC arms: ____
  - expert efficiency vs its bound after fusion: ____
  - NVFP4 vs FP8 closed-loop verdict: ____
  - Also preserved: `04` §7 headline 4 and E7.
- **Kill (KC-6).** If p99 > 150 ms after S6, drop runtime CoC, cut to K = 2 and shrink the expert. If it is still over budget, kill the 10 Hz claim and report 5 Hz honestly. Without the device (P-002 declined), do the serving work on an RTX PRO 6000 with roofline scaling, labelled "not measured on Thor".
- **Cost:** the Jetson AGX Thor dev kit at $5,499 MSRP (NVIDIA FAQ; one retailer lists $6,269, out of stock), plus about 3 verdict sweeps ($102–408).
- **Upstream value:** no 2B Alpamayo student has been shown at 10 Hz on Jetson AGX Thor, and no closed-loop per-format verdict exists.

### S7 — Hardware-in-the-loop with latency injection (the input's P4)
- **Why.** Lockstep HIL scores the policy as if latency were zero. At 10 Hz, 139 ms of latency means each action lands 2 ticks late; at 15 m/s the car travels 2.1 m in that time.
- **Build:**
  - Thor serves the AlpaSim driver gRPC endpoint while cloud AlpaSim runs in lockstep, so network delay never touches the score. On-device time is logged per step.
  - **Latency injection:** AlpaSim applies each action after the measured on-device delay, which shows how much score the real latency costs.
  - Real-time log replay from NVMe.
- **★ Huy:** lockstep semantics, the latency-injection design, equivalence statistics (TOST).
- **Acceptance:**
  - the Thor-in-loop score equals the cloud score within S2's noise floor (paired equivalence, Huy's margin);
  - the latency-injected score is reported with a CI.
- **Predictions:** score lost to the measured latency: ____
- **Kill (KC-7):** if lockstep changes policy semantics (timestamps, buffering), report open-loop replay only.
- **Cost:** about $200.

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

### Releases
Each stop point ships an artifact: the open-source repository, a technical report, and (after S6–S8) a demo video with a Thor latency overlay. Each release is reproducible from a fresh clone and subject to KC-L and KC-L2.

## 4. The first seven days: S1 ships a public artifact for $0

`physloop-gate` v0 is a paired release-gate core validated on synthetic truth, plus a 441-scene power analysis. It runs on CPU only and needs no dataset or license.

| Day | ★ Huy | Agents | Output |
|---|---|---|---|
| 1 (10-26) | Write `experiments/2026-10-26-s1-synthetic-truth.md` from the template and fill P-S1-1…4. Derive the exact McNemar null and the zero-failure bound by hand | Scaffold `gate/`, CI and the test harness | Experiment file with predictions |
| 2 | Write the `verdict()` core: paired non-inferiority (Tango CI), the intersection–union rule, the INSUFFICIENT state | Encode the known-answer tests | Core with tests going from failing to passing |
| 3 | Write the road-clustered paired bootstrap and the Durkalski adjustment; specify the generator | Build the generator to that specification | Generator with known truth |
| 4 | Diagnose every Monte Carlo deviation | Run 1,000-run sweeps (± stratification; null and −30%; 2% and larger strata), with plots | False-GO, false-NO-GO and INSUFFICIENT rates |
| 5 | Power analysis: MDE, NI margins, stratum power; **scenes needed vs base rate and discordance for S4's relative gate; S3's margin on both score metrics** | Tables and figures | "What 441 scenes can certify" |
| 6 | Reconcile prediction vs measurement and explain every miss causally; write the README's claims section. Re-check Edge (KC-E, due 10-31) | Packaging, docs | Reconciliation recorded |
| 7 (11-01) | Publish the repo (Huy's action); open the AlpaSim issue; write the journal; send the license letter if not yet sent (P-004) | Release notes | **Public artifact + upstream issue** |

## 5. Budget tiers (prices verified 2026-10-25; costs are inferences)

- **Price basis** (vast.ai, on-demand, 2026-10-25):
  - RTX PRO 6000 Workstation: $1.00 cheapest, $1.60 median.
  - RTX PRO 6000 Server: $1.24 cheapest, $1.47 median.
  - H100 SXM: $1.93 cheapest, $2.40 median ($3.19 for verified hosts).
  - Elsewhere: Lambda H100 $3.99–4.29; RunPod H100 $2.69–3.99.
- **One sweep** (441 scenes × 3 seeds × 200 steps) = 34–85 GPU-h, a single-stream upper bound. S2's measured throughput replaces it.

| Tier | Unlocks | Estimate |
|---|---|---|
| **$0** | S1; every curriculum exercise; the AlpaSim issue; the license letter | CPU |
| **Cloud ≤ $2.5k** | S2 + S3, plus an S4 pilot (one reward arm × 3 seeds) | ≈ $0.5–2.2k plus storage |
| **Cloud ≤ $6k** | S2–S5 in full, plus S7's cloud side | ≈ $1.8–6.3k (stage costs in §3). The upper end exceeds the tier, so the cap rule below applies |
| **+ Jetson AGX Thor** | S6, S7 | $5,499 (P-002) |
| **+ Rig** | S8 | ~$5–8k (P-006) |
| **Collaborator** | PR #42-scale distillation (≥ 128 H100s recommended); larger RL | NVIDIA Academic Grant (faculty PIs only); Nebius research credits (academic or non-profit); Inception (incorporated company) |

**Rules:**
- Stop at stage boundaries, never mid-stage.
- Re-price at every stage start.
- If cloud spend passes $4k, cap group sizes, cut the scene count, use interruptible RTX PRO 6000s for rollouts and H100s only for the optimizer.
- The tier choice is P-005, required before S2 spends anything.

## 6. Claims PhysLoop may and may not make

**May.** Every claim states n, CI, seeds, manifest (including the resolved lockfile), AlpaSim SHA, scenario list, score definition, deviation convention and, for latency, the device.
- "Checkpoint B is not worse than A by more than Δ on every pre-specified stratum, at one-sided 95% confidence, paired and cluster-adjusted by road."
  - Δ is pre-registered on a named metric.
  - Underpowered strata are INSUFFICIENT, and INSUFFICIENT blocks GO.
- "NVIDIA's 2B student is / is not non-inferior to its 10B teacher within Δ in AlpaSim closed loop, under protocol P", or INSUFFICIENT.
- "RL changed at-fault collisions by X (one-sided bound) on n held-out scenes", or INSUFFICIENT.
- "The per-scenario failure rate in stratum S is ≤ X, by a **one-sided** 95% Clopper–Pearson bound."
- "On Jetson AGX Thor (JetPack 7.2.1, TensorRT 10.16.2), student S runs at p50/p99 = … ms per step with 4 cameras (real engine | emulated)."
- "In shadow mode, the student's trajectory disagreed with the human driver on X% of Y hours. This is an open-loop measurement."
- The G3 claims the narrowed protocol licenses: "gate rankings are reliable across seeds (ICC = …)"; "known-order checkpoints ordered correctly k/K".

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

## 7. Kill checks (dated for D = 2026-10-26; shift with D)

| Check | When | Rule |
|---|---|---|
| KC-1 | D+1 (11-02) | `verdict()` controls false-GO on synthetic truth, and the S3/S4 power tables exist; otherwise fix before any GPU spend. Re-forecast all windows with actual hours |
| KC-E | 2026-10-31 and at S3's start | Edge weights public and licensed for evaluation → S3 as written. Otherwise S4 first; S3 on a PR #42 pilot student under Huy's cap |
| KC-L | Before the first public PAI-AV/NuRec-derived number | A written answer on license §3, or the fallback (code, scene-ID manifests, synthetic results) |
| KC-L2 | Before any release of weights or labels trained on PAI-AV/NuRec-derived data | A written answer on license §4.6; otherwise release code and recipes only |
| KC-2 | D+3 (11-16) | Reliability ≥ Huy's floor and noise floor below the effects S3–S5 must detect; otherwise stop |
| KC-3 | D+5 (11-30) | The student verdict is the result. Edge does not run in AlpaSim within 16 GiB → report and fall back |
| KC-4 | D+8 (12-21) | Closed-loop score varies by more than 0.1 across seeds after 3 seeds → open-loop GRPO fallback; AlpaGym does not run → skip |
| KC-5 | D+11 (2027-01-11) | No paired correction gain after three rounds → negative result with mechanism |
| KC-6 | D+12 (01-18) | p99 > 150 ms → no CoC, K = 2, smaller expert; still over → report 5 Hz |
| KC-7 | D+13 (01-25) | Lockstep alters policy semantics → open-loop replay only |
| KC-8 | Before S8 | Legal, privacy or safety doubt, or P-006 not approved → skip S8 |
| KC-$ | Any time | Cloud spend > $4k → the cap rule in §5 |

## 8. Risks

| Risk | Check | Response |
|---|---|---|
| Edge slips, ships without weights, or ships under a license that forbids evaluation or fine-tuning | KC-E, 10-31 | Run S4 first; S3 on a PR #42 pilot; S5 needs a fine-tunable student |
| NVIDIA publishes a closed-loop evaluation of its own student | Every Edge, recipes or AlpaSim release | The gate's independent, paired, stratified verdict keeps its value; contribute the evidence upstream |
| License §3 blocks publication of closed-loop numbers; §4.6 blocks weights | KC-L, KC-L2 | Send the letter now (P-004). Fallback: code, manifests, synthetic results; nuPlan/MTGS-track numbers if Alpamayo-family policies run there (unverified) |
| AlpaSim instability: mid-challenge evaluator change; SHA does not pin the environment; withheld "simulation choices"; `main` unsynced since 08-11 | S2 | Start S2 after 10-31; pin the lockfile; record `n_sim_steps`; report the instrument's noise |
| AlpaGym frozen, pinned to an unmerged PR, VLM frozen | S4 | Build nothing on its internals; skip or fall back as in KC-4 |
| Contamination: evaluation scenes overlap the teacher's training clips | S2 contamination report | Exclude overlapping scenes, or report verdicts with and without them |
| Thor toolchain: TensorRT-Edge-LLM lacks Alpamayo 1.5; PR #45 uses a fork and targets DRIVE | S6 | Budget the porting work; label emulated numbers; fall back to 5 Hz honestly |
| Underpowered gates (S4's relative effect at low base rates) | KC-1, S2 | INSUFFICIENT is a result; Huy may change the endpoint or effect size before the run |
| Effort under-estimated (≈ 320 h, unmeasured) | KC-1 | Re-forecast; cut S8 first, then S7 |
| The 12-month license term starts at first download | S2 | Record expiry = first download + 12 months |
| Price and supply moves (vast.ai H100 median $1.69 → $2.40 in three weeks) | Each stage start | Re-price; interruptible RTX PRO 6000 for rollouts |
| Ownership drift: agents writing ★ cores | Every stage close | Owner proof (`03` §5); otherwise redo the core |
| Regulatory wording drifts | About 2027-01-30 (R[185]) | Update the claims wording; never claim compliance |

## 9. Open owner decisions

These decisions block specific stages. Recording each one closes its proposal in `DECISIONS.md`.

| Proposal | Decision | Blocks |
|---|---|---|
| P-002 | Buy a Jetson AGX Thor dev kit ($5,499 MSRP). Its numbers do not transfer 1:1 to DRIVE AGX Thor: same silicon and bandwidth, but JetPack/TensorRT 10.16 vs DriveOS 7.2/TensorRT 11.x | S6, S7 |
| P-003 | Weekly hours | Calendar accuracy; re-forecast at KC-1 |
| P-004 | Send NVIDIA the written license request (Huy's outward action), now covering §3 (aggregate closed-loop results, scene-ID manifests) **and** §4.6 (weights and labels trained on PAI-AV/NuRec-derived data) | KC-L, KC-L2: every public number from S2 on |
| P-005 | Cloud budget tier ($0 / ≤ $2.5k / ≤ $6k / collaborator) | Any spend in S2 |
| P-006 | Shadow mode: rig budget (~$5–8k), Huy's confirmation of local rules, privacy plan | S8 |
| Per stage, pre-registered in `experiments/` | Floors and margins: reliability, Δ for S3 and S5 on a named metric, S4's endpoint and effect, S7's equivalence margin | The matching KC |

## 10. Key primary sources

The full ledgers are `research/2026-10-25-teacher-to-thor/verification.md` and `research/2026-10-03-verification/report.md` §10.
- **Decision input and review:** `research/2026-10-25-teacher-to-thor/decision_input_2026-10-25.md`, `review.md`, `notes/arith_check.py`
- **Alpamayo:** AR1 https://arxiv.org/abs/2511.00088 · cards https://huggingface.co/nvidia/Alpamayo-1.5-10B · https://huggingface.co/nvidia/Alpamayo2-Super · Edge announcement https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion · https://blogs.nvidia.com/blog/open-models-europe-industry/
- **NVIDIA repos:**
  - AlpaSim https://github.com/NVlabs/alpasim (issues #207, #208, #224; PR #215);
  - AlpaGym https://github.com/NVlabs/alpagym;
  - recipes https://github.com/NVlabs/alpamayo-recipes (PRs #18, #40, #42, #45);
  - AlpaJudge https://github.com/NVlabs/alpajudge;
  - TensorRT-Edge-LLM https://github.com/NVIDIA/TensorRT-Edge-LLM (issue #130, PR #260).
- **Thor:** Jetson Thor blog https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/ · JetPack 7.2.1 https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1 · TensorRT support matrix https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html · FlashRT https://github.com/flashrt-project/FlashRT/pull/229 · FlashDrive https://arxiv.org/abs/2608.12932
- **Data:** https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles · https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
- **Post-training:** RoaD https://arxiv.org/abs/2512.01993 · OPTED https://arxiv.org/abs/2609.20756 · DAgger https://arxiv.org/abs/1011.0686 · closed-loop blog https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/
- **Regulation and law:** NHTSA 91 FR 48485 https://www.govinfo.gov/content/pkg/FR-2026-07-31/html/2026-15483.htm · C.N.288.2026 https://treaties.un.org/doc/Publication/CN/2026/CN.288.2026-Eng.pdf · Vietnam Law 36/2024 (consolidated VBHN 55) https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf · RAND https://www.rand.org/pubs/research_reports/RR1478.html
