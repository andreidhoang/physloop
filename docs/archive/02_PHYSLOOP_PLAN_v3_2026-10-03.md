# 02 — PhysLoop plan v3 (3 October 2026)

**Status:** authoritative (see `../DOC_AUTHORITY.md`). It replaces v2 of 2026-09-23, which is archived in `archive/`. PhysLoop is a standalone, active project (DECISIONS D-009). It has no dependency on any other program's hours, budget or deadlines.

**Evidence base:** `research/2026-10-03-verification/report.md`. Its §10 claim ledger covers every number below. Each number was checked against primary sources on 2026-10-03, and inferences are labelled as inferences. The predictions are Huy's: every slot marked `____` stays blank until he fills it in `../experiments/` before the run.

---

## 0. What the verification changed

Seven research lanes re-checked docs 01–05 against primary sources. Every arXiv ID on the reference shelf resolves to the claimed paper, and the worked statistics are arithmetically right. The problems are elsewhere: the framing of the headline, several upstream assumptions, and the economics.

| Area | Finding | Consequence |
|---|---|---|
| Headline | G3 as written could not be falsified. It named no reference. Its implied reference, Drive-IRT, re-fits the same rollouts the gate scores. "Predictive validity" needs on-road outcomes that only NVIDIA, Wayve and Waabi hold. And a ρ < 0.6 rule cannot be decided at n = 10: an observed 0.8 has a 95% CI of about [0.24, 0.96] | G3 becomes a **construct-validity** study with a pre-registered PASS / KILL / INSUFFICIENT rule at n ≈ 28–34 checkpoints (§2, S5) |
| Prior art | Sim2Val (CoRL 2025) reports per-episode Pearson ρ > 0.9 between AlpaSim and on-road logs. It covers one policy, two kinematic metrics and an 8 s horizon. OmniDreams preserved the ranking of 4 policies across renderers, with no coefficient | Novelty claim narrows to the "first open, multi-policy reliability and convergent-validity study of AlpaSim closed-loop scores" |
| Statistics | Exact McNemar ignores road clustering. "≤ 0.3% for 0/1,000" is a **one-sided** bound (two-sided: 0.368%). At 441 scenes and 4% discordance, 80% power detects only about 2.72 pp. A 2% stratum (about 9 scenes) has at most 0.2% power. Bosch's ρ = −0.36 does not reproduce (n = 7; recomputed −0.14) | Cluster-adjusted paired non-inferiority; INSUFFICIENT blocks GO; minimum stratum sizes |
| Challenge | Registration closes **2026-10-18** and needs an organization email. Teams get 3 official submissions per UTC month, ranked on private held-out scenes. 16 GiB per driver is a hard limit; 0.1 s per call is only a target. A lateral exit of ≥ 4 m scores as failure, whereas NVIDIA's papers *exclude* such events | No challenge path for an unaffiliated solo engineer. G2 reports validity exits separately, at both 2 m and 4 m |
| Upstream | AlpaSim has no paired verdict, no stratification, no seed test–retest, no renderer or traffic agreement, and no guardrail monitor (the hook exists). It merges external PRs within 0–6 days, with no CLA. AlpaGym has accepted zero external PRs and has been code-frozen since 2026-07-08 | G1, G2 and G4 become thin AlpaSim layers. G7 becomes a measurement report |
| Precision | NVIDIA's ModelOpt 0.43 recipe already quantizes the action expert, but evaluates open-loop minADE only and publishes no accuracy or latency numbers | G6 becomes closed-loop per-format verdicts with intervals (S4) |
| Distillation | No public recipe exists. NVIDIA has promised a 10B → ~2B script three times | G5 merges into nanoAlpamayo, with a pivot-to-evaluation kill (S7) |
| Instrument drift | NVIDIA's Alpamayo 1.5 card moved its AlpaSim score from 0.81 to 1.37 on 2026-08-07 with unchanged weights | Pin one protocol (scene list, AlpaSim SHA, seeds, deviation convention) for every number |
| Economics | The v2 cost model split stages wrongly. The expert runs 4–7× slower than its bound, rollout throughput is about 4 ticks/s per GPU (not 6–9), and W4A8 removes only 24.6 ms of FlashDrive's 717 → 151 ms. Prices moved: H100 $1.60–4.50/h, RTX PRO 6000 $1.40–2.19/h, B200 from $5.98/h | Re-priced budget tiers (§5). S2's first measured throughput replaces every estimate |
| Licenses | Model weights are OpenMDW-1.1. The PAI-AV and NuRec data licenses differ: internal AV development only (§1); benchmarking, regression and performance data treated as NVIDIA confidential (§3); no derivatives or distribution (§4.6); expiry 12 months after first download (§8). Access is automatic on Hugging Face once the license is accepted | A **publication gate** (KC-L) before any number derived from PAI-AV or NuRec. No dataset download before S2 |
| Regulation | The UN ADS GTR was adopted on 2026-06-24. UN R[185] is notified and enters into force around 2027-01-30 unless blocked. No regulation prescribes paired regression against the previous release. Waymo uses release-to-release regression only as a diagnostic; Zoox gates on one combined risk estimate | Paired regression is **PhysLoop's own design choice**. R[185] and the GTR join the no-compliance list (§6) |
| Roles | No US "Principal End-to-End" posting is live. NVIDIA's end-to-end, VLA and closed-loop-RL hiring is at Senior level, Levels 3–6; Role A's L6 band equals the Principal band | `05_ROLE_MAPPING.md` re-mapped to live posting IDs |

**What survived.** The negative claims that matter all held on 2026-10-03:
- No open go/no-go release rule exists.
- No public distillation procedure exists.
- No stage-level RL throughput profile has been published.
- No policy-plus-guardrail evaluation has been published.

The positive claims held too:
- NVIDIA positions Alpamayo as a teacher distilled into compact Thor models.
- DRIVE AV pairs end-to-end with a parallel classical safety stack (CLA, L4-ready S-Class).
- The memory arithmetic holds: 22.16 GB in BF16, about 11 GB in FP8.
- The pins hold: torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, ModelOpt 0.43.0.

## 1. The one thing we build

**PhysLoop is an open release gate for the Alpamayo teacher → student path.** It runs paired, stratified, validity-checked closed-loop evaluation on AlpaSim and measures its own reliability and construct validity. It answers one question: *did distillation, quantization or RL make the deployable model worse, and where?* It ships as thin layers that can be upstreamed into AlpaSim, the one NVIDIA repository that accepts outside work.

The second spine is **nanoAlpamayo**, a small driving VLA built and post-trained in this project. It doubles as the gate-measured student (G5). Every closed-loop number it produces comes from the gate.

## 2. Deliverables, re-scoped against what upstream ships

| G | Upstream already ships (2026-10-03) | Still open, our new work | Verdict | Stage |
|---|---|---|---|---|
| **G1 Gate v1** | Per-rollout scorers; `results-summary.json`; seeds (PR #149); the leakage-aware 1,761/441 split (PR #167); Drive-IRT panel ranking | Paired two-checkpoint verdict; stratification; seed test–retest; per-stratum bounds; extended taxonomy; allowed-claims list | **Keep**, as a thin AlpaSim layer | S1, S2 |
| **G2 Validity envelope** | `dist_to_gt_*` metrics; 4 m corridor scored as failure; one renderer endpoint (NuRec or OmniDreams); OmniDreams quality-vs-deviation precedent | Calibrated cap; separate reporting of out-of-envelope rollouts; renderer and traffic agreement | **Keep, narrowed**; its agreement parts are G3's reliability legs | S2, S3 |
| **G3 Gate validity** | Sim2Val (1 policy, per-episode); OmniDreams (n = 4, qualitative) | Multi-policy reliability and convergent-validity study | **Keep, restated** as construct validity (below) | S2–S5 |
| **G4 Guardrail** | `safety_monitor_safe` hook and scorer; MPC controllers | The monitor (C++), override path, paired policy-vs-policy+monitor scoring, intervention rate | **Keep**. The mechanism is PhysLoop's own design and is never called "Halos" | S6 |
| **G5 Distillation** | Teacher positioning; Alpamayo 2 Super SFT recipe; AlpaGym expert RL | Any public distillation procedure; paired student-vs-teacher verdict | **Merged into nanoAlpamayo**, with a pivot kill (KC-7) | S7 |
| **G6 Precision** | ModelOpt 0.43 PTQ including the expert; open-loop minADE only | Closed-loop per-format verdicts with intervals; published sensitivity | **Re-scoped** into S4's quantization ladder | S4 |
| **G7 Throughput** | AlpaGym `perf/`; AlpaSim Prometheus telemetry (default since 2026-06-30) | A public stage-level profile; batch-scaling curve; render-resolution experiment (a camera config, not a PR) | **Re-scoped** to a measurement report. PRs go to AlpaSim, not AlpaGym | S8 |
| **G8 Flywheel spec** | CoC autolabeler; Instant NuRec | Diagram and config | **Keep** as a document, low priority | S8 |

### G3 restated: construct validity with a pre-registered verdict

Measurement science prescribes this when no accepted criterion exists (Cronbach & Meehl 1955; Messick 1995; Kane 2013). References are ranked by independence, from R0 (Drive-IRT on the same rollouts) to R10 (on-road outcomes, attainable only inside NVIDIA, Wayve or Waabi).

1. **Reliability (R1).** At least 10 policies × 441 `nurec_curated_val` scenes × at least 3 seeds. S2 starts with 3 released policies; S4's ladder checkpoints complete the ≥ 10-policy reliability panel. They never count toward S5's convergent n. The metrology terms (JCGM 200) map as follows: seeds measure repeatability; GPU, driver and AlpaSim drift measure intermediate precision; another site or renderer measures reproducibility. Report ICC(2,1) with CI, Kendall's W and scenario-level discordance. No-op changes serve as negative controls. *Validity cannot exceed reliability; if reliability fails, stop.*
2. **Renderer and traffic agreement (R2–R4).** Compare Harmonizer on vs off, log replay vs CAT-K, and OmniDreams if accessible. Report Kendall τ and pairwise agreement with road-clustered bootstrap CIs, plus Cohen's κ by deviation bin (< 1, 1–2, 2–4, ≥ 4 m). This is reliability evidence (Campbell & Fiske 1959).
3. **Dose–response ladders (R7, R9).** At least 3 degradation types × at least 4 levels: camera dropout, injected latency, action noise, quantization bit-width, early checkpoints. Pre-register monotone decline per metric. Report pairwise accuracy with a Wilson CI, tie-aware accuracy, a trend test and a MAD outlier screen.
4. **Cross-paradigm convergence (R6, R9b).** 28–34 checkpoints of NAVSIM-family policies on AlpaSim's nuPlan/MTGS track, compared against NAVSIM EPDMS, with CARLA via `py123d_garage` as a third paradigm. Report Spearman/Kendall with Bonett–Wright CIs, and within-family vs between-family correlation. Never tune the cap, renderer or traffic model on the validation policies.
5. **Discriminant check (R8).** The gate should correlate only weakly with open-loop minADE6 on a published PAI-AV subset.
6. **Private set (R5).** At most 3 policies, only through a collaborator registered for a challenge. Not on the critical path.
7. **Verdict.** Three things are decidable here:
   - (a) reliability below Huy's floor;
   - (b) known-group pairwise-accuracy upper bound below Huy's floor;
   - (c) for cross-paradigm ρ: **PASS** if the one-sided 95% lower bound ≥ 0.6, **KILL** if the upper bound < 0.6, otherwise **INSUFFICIENT**. At n = 30, PASS needs an observed ρ ≳ 0.77–0.78 and KILL ≲ 0.35.

   Huy may re-set the 0.6 cut before pre-registering.

The strongest licensable conclusion is *a reliable, renderer- and traffic-robust instrument that responds correctly to known degradations and converges with independent simulation paradigms*. That is construct validity, **not** predictive validity.

## 3. Stages: one dependency graph

```
S1 ──► S2 ──┬──► S3 ──┬──► S4 ──┬──► S5   G3 verdict + preprint
            │         │         └──► S7   nanoAlpamayo = gate-measured student
            │         └──► S6             guardrail in C++
            └──► S8                       throughput report + flywheel spec (telemetry from S2–S4)

KC-L publication gate: no public number derived from PAI-AV/NuRec before a written answer on license §3,
or the fallback (code + scene-ID manifests + synthetic results).
```

D is the start date of S1, by default **Monday 2026-10-05**. The windows assume 25–30 focused hours a week. That figure is unconfirmed (proposal P-003), so re-forecast every window after S1's actuals at KC-1. Effort figures are planning inferences.

| Stage | Serves | Window | Calendar (D = 2026-10-05) | Effort (h) | Budget tier | Kill check |
|---|---|---|---|---|---|---|
| **S1** Gate core on synthetic truth + power analysis | G1 core | D+0 → D+1 wk | 10-05 → 10-12 | ~20 | $0 | KC-1 |
| **S2** Gate v1 on AlpaSim, reliability first | G1, G3·R1, G2 records | D+1 → D+3 | 10-12 → 10-26 | ~40 | ≤ $2.5k | KC-2, KC-L |
| **S3** Validity envelope + agreement | G2, G3·R2–R4 | D+3 → D+5 | 10-26 → 11-09 | ~30 | ≤ $2.5k | KC-3 |
| **S8** Throughput report + flywheel spec | G7, G8 | D+3 → D+8 | 10-26 → 11-30 | ~15 | ≈ $0 marginal | — |
| **S4** Known-group ladders + precision verdicts | G3·R7–R9, G6 | D+4 → D+7 | 11-02 → 11-23 | ~40 (+20 for S4b) | ≤ $2.5k | KC-4 |
| **S6** Guardrail-in-the-loop (C++) | G4 | D+6 → D+8 | 11-16 → 11-30 | ~35 | ≤ $2.5k | KC-6 |
| **S5** Cross-paradigm convergence + verdict | G3·R5, R6, R9b | D+8 → D+11 | 11-30 → 12-21 | ~60 | ≤ $6k or collaborator | KC-5 |
| **S7** nanoAlpamayo as student | G5 merged | D+11 → D+20+ | 12-21 → 2027-02-22+ | ~150 | ≤ $6k or collaborator | KC-7 |

Each stage card below gives the Why as a verified L5 symptom. ★ marks the cores Huy writes (the measurement instruments, statistics, RL loss math, verifier logic, predictions and diagnoses); agents write the plumbing, harnesses, tests, sweeps and plots. The derivations, known-answer exercises and traps for each stage are in `03_MASTERY_PLAN.md` §4. Costs are verified prices × derived hours, so they are inferences.

### S1 — Gate decision core on synthetic truth, plus power analysis ($0, CPU)
- **Why.** On 441 paired scenes at 4% discordance, 80% power detects only about 2.72 pp. The v2 example (12 vs 6) has power 0.22. A regression confined to a 2% stratum dilutes from 2.1 pp to 0.042 pp pooled, so it hides inside a passing aggregate.
- **Build:** `gate/`. It contains:
  - the identity tuple and `is_pair()`;
  - `verdict()` returning GO / NO-GO / INSUFFICIENT, where **INSUFFICIENT blocks GO**;
  - a synthetic generator with known effects, road clusters and strata;
  - a 1,000-run Monte Carlo;
  - power tables ("what 441 scenes can certify");
  - known-answer tests.

  It consumes what AlpaSim's evaluator emits: per-rollout pass/fail and scene scores keyed by scene ID. It needs no dataset license.
- **★ Huy:** `verdict()`, which includes
  - paired non-inferiority with a pre-registered margin (Tango score CI);
  - an intersection–union rule across strata;
  - Holm for regression flags;
  - cluster-adjusted McNemar (Durkalski) or a road-level paired bootstrap;
  - the zero-failure bound p_U = 1 − α^(1/n), one-sided;
  - the power derivations;
  - the predictions.
- **Acceptance.** The known-answer tests reproduce:
  - 12 vs 6: one-sided 0.11894, two-sided 0.23788;
  - 0/441: 0.677% one-sided, 0.833% two-sided;
  - 0/1,000: 0.2991% one-sided, 0.3682% two-sided;
  - MDE 1.94 / 2.72 / 3.90 pp at ψ = 2.1 / 4 / 8%;
  - non-inferiority margins 2.67 / 3.77 pp;
  - power 0.22.

  The Monte Carlo must also show three things: INSUFFICIENT (never GO) for a 9-scene stratum; higher false-GO when stratification is removed; and false-GO ≤ α wherever power exists.
- **Predictions:**
  - P-S1-1, false-GO for a −30% effect on a 2% stratum at 7% base failure: ____
  - P-S1-2, share of INSUFFICIENT under the null: ____
  - P-S1-3, change in false-GO without stratification: ____
- **Kill (KC-1, D+1):** if `verdict()` cannot hold false-GO ≤ α on synthetic truth, fix the statistics before any rented GPU-hour. If S1 runs past twice its estimate, ship the power-analysis half alone.
- **Upstream value.** No paired verdict, stratification or test–retest exists upstream. Challenge teams #2–#5 are statistically tied, so a certification table helps every participant. Delivery: publish the repository, then open an AlpaSim issue proposing the paired layer for `e2e_challenge/local_evaluation`.

### S2 — Gate v1 on AlpaSim, reliability first
- **Why.** NVIDIA's own Alpamayo 1.5 card moved from 0.81 ± 0.01 (910 scenarios) to 1.37 ± 0.10 (913) on 2026-08-07 with unchanged weights, and it never defines "±". An NVIDIA maintainer confirms that NuRec has small run-to-run variation and that the traffic model ignores the seed (AlpaSim issue #128).
- **Build:** a G1 v1 report on three released policies: Alpamayo 1, Alpamayo 1.5 BF16, and Alpamayo 1.5 FP8 (ModelOpt recipe). It runs 441 `nurec_curated_val` scenes × ≥ 3 seeds. The report contains:
  - per-stratum paired verdicts with one-sided bounds;
  - seed test–retest;
  - a negative control: a bit-identical re-export must not be flagged;
  - per-rollout validity records in deviation bins < 1, 1–2, 2–4 and ≥ 4 m, with `left_corridor_laterally` split out;
  - a failure taxonomy that maps AlpaSim's native scorers onto the NHTSA pre-crash typology: `collision_at_fault` (front or lateral), `collision_rear`, `offroad`, `wrong_lane`, `min_distance_to_obstacle_m`, `plan_deviation`, `safety_monitor_triggered`;
  - manifests (git hash, config, seeds, both AlpaSim SHAs, GPU edition);
  - the **first measured throughput**, which re-baselines every cost in this plan.
- **★ Huy:** reliability statistics (ICC(2,1), Kendall's W, design effect 1 + (m − 1)ρ), stratum definitions, taxonomy mapping, validity-record semantics, and the diagnosis of every discordance.
- **Acceptance.** ICC and W reported with CIs, along with the noise floor and the negative-control flag rate. Every rollout carries a validity record. Deviation is reported at both 2 m and 4 m. Every retry is logged, because Alpamayo prefill intermittently emits all-NaN trajectories.
- **Predictions** (preserved from mastery v1.1 Stage 1b, archived):
  - (i) test–retest discordance rate of Alpamayo 1.5 across seeds: ____
  - (ii) % of rollouts beyond 2 m and 4 m, and whether it differs between good and bad policies: ____
  - the pre-registered reliability floor: ____
- **Kill.**
  - **KC-2, D+3:** reliability below the pre-registered floor → stop.
  - **KC-L:** no public number derived from PAI-AV or NuRec without a written answer on license §3 (P-004). The fallback is to publish code, scene-ID manifests and synthetic results.
- **Cost:** 4 sweeps × 34–85 GPU-h ⇒ **$110–707** on RTX PRO 6000, from interruptible to secure rates. Add $40–216/month of storage for the ~2.9 TB NuRec 26.04 release.
- **Upstream value:** the first public seed test–retest and reliability numbers for AlpaSim.

### S3 — Validity envelope and inter-rater agreement
- **Why.** NuRec degrades as the ego leaves the capture path (OmniDreams Fig. 14). The challenge *scores* ≥ 4 m exits as failures, while NVIDIA's papers *exclude* them. A wandering policy is therefore penalized for renderer limits under one convention and excused under the other.
- **Build:**
  - a deviation-binned validity report;
  - renderer agreement (Harmonizer on/off; OmniDreams optional) and traffic agreement (replay vs CAT-K);
  - out-of-envelope results reported separately, never silently scored or dropped;
  - an upstream PR seeding the CAT-K traffic service, which an NVIDIA maintainer invited in issue #128.
- **★ Huy:** Bland–Altman limits, κ, τ and the clustered bootstrap; envelope semantics; the cap decision.
- **Acceptance:** κ and τ with CIs per deviation bin; the validity-exit rate per policy; agreement with and without rollouts ≥ 4 m.
- **Predictions:**
  - τ for Harmonizer on vs off: ____
  - κ in the ≥ 4 m bin vs the < 1 m bin: ____
  - share of failure labels that flip between replay and CAT-K: ____
- **Kill (KC-3, D+5).** If coverage inside the envelope falls below Huy's floor, tighten the cap and report the coverage loss. If renderer agreement fails, report the renderer as a dominant noise source.
- **Cost:** 6 sweeps ⇒ **$165–1,060**. The OmniDreams slice is optional: it needs GB300-class hardware and has no verified price.
- **Upstream value.** A calibrated cap, separate out-of-envelope reporting and agreement tables do not exist upstream. Regulators define a virtual tool's envelope through credibility assessment (NATM Annex III; EU 2022/1426). PhysLoop claims only an internally evidenced envelope.

### S4 — Known-group ladders, precision verdicts, discriminant check
- **Why.** FlashDrive ran a quantized, optimized Alpamayo 1.5 on 100 AlpaSim clips. Collision improved (0.19 → 0.15) and so did off-road (0.41 → 0.32), but **wrong-lane regressed (0.45 → 0.51)**. One aggregate would have called that a win. NVIDIA's PTQ recipe checks open-loop minADE only.
- **Build:**
  - a known-group report: pairwise accuracy, trend tests, ladders reported per metric;
  - per-format paired verdicts with intervals: BF16, FP8, NVFP4, W4A8 and AutoQuant, plus per-layer sensitivity from AutoQuant's `search_state`;
  - the discriminant check of the gate against open-loop minADE;
  - one release note written in the allowed-claims language of §6.
- **S4b (optional):** diagnose why the flow expert runs at about 20% of its bandwidth bound after CUDA graphs: 113.9 ms measured vs ≈ 22.3 ms bound over 8 steps. The cause is undiagnosed in public sources.
- **★ Huy:** ladder design and its pre-registered assumption, the pairwise statistics, the format verdicts, the S4b kernel diagnosis.
- **Acceptance:** k/K known pairs ordered correctly, with a Wilson CI; per-metric directions, never a single aggregate; per-format verdicts with intervals; the discriminant correlation with a CI.
- **Predictions:**
  - pairwise accuracy: ____
  - metrics that move monotonically on each ladder: ____
  - FP8 vs BF16 paired verdict: ____
  - NVFP4 sensitivity of the expert layers: ____
  - Also preserved: `04` §7 E7, as written.
- **Kill (KC-4, D+7):** if the 95% upper bound on known-group pairwise accuracy falls below Huy's pre-registered floor, downgrade the validity claims.
- **Cost:** ≥ 12 sweeps ⇒ **$331–2,120**. S4b adds a few RTX PRO 6000 hours.
- **Upstream value.** Closed-loop per-format verdicts and published sensitivity do not exist upstream. The ladders yield dozens of pairs with known order, without any on-road data.

### S5 — Cross-paradigm convergent validity and the pre-registered verdict (G3 headline)
- **Why.** The challenge's own 95% rank intervals leave PAI teams #2–#5 tied, with rank intervals spanning 3–10. A single aggregate rank is noise at this scale.
- **Build:**
  - the pre-registration file;
  - NAVSIM-family adapters (LTF, DiffusionDrive, GTRS) on AlpaSim's nuPlan/MTGS track;
  - NAVSIM EPDMS and `py123d_garage` (CARLA) runs;
  - the preprint: *first open, multi-policy reliability and convergent-validity study of AlpaSim closed-loop scores, with stated limits* (Bean et al. construct-validity checklist).
- **★ Huy:** the pre-registration, Fisher-z / Bonett–Wright CIs, within- vs between-family analysis, the verdict, the paper's argument.
- **Acceptance:** the pre-registered rule applied as written; n_families and n_checkpoints stated separately; results with and without outlier policies. Ladder variants never count as convergent n.
- **Predictions:**
  - cross-paradigm ρ: ____
  - pairwise accuracy: ____
  - within- minus between-family gap: ____
  - Also preserved: `01` §4 Q3 (gate vs Drive-IRT on 10 policies), re-labelled reference R0, an aggregation-robustness check.
- **Kill (KC-5, D+11).** Apply the three-way rule. KILL ⇒ publish the negative result with its mechanism and stop building on the gate's validity. INSUFFICIENT ⇒ report the interval and claim nothing.
- **Cost:** unmeasured per checkpoint, so pilot one first. The reference bracket is ~300 H100-h ⇒ **$507–1,350**.
- **Upstream value:** no multi-policy correlation of AlpaSim or NuRec scores with an independent reference exists. Drive-IRT has no external validation.

### S6 — Guardrail-in-the-loop in C++
- **Why.** DRIVE AV runs end-to-end alongside a classical safety stack on Halos, in both the L2 CLA and the L4-ready S-Class. Scoring the bare model scores something that never ships.
- **Build:**
  - a rule-based TTC and drivable-area monitor in C++, behind AlpaSim's `safety_monitor_safe` hook (upstream ships only consumers and tests);
  - paired verdicts for policy vs policy + monitor;
  - an intervention-rate report;
  - an upstream issue or PR for the monitor.
- **★ Huy:** the monitor logic (the verifier), using TTC and the RSS Lemma 2 safe distance; the thresholds; the intervention-rate statistics.
- **Acceptance:** C++ unit tests; the paired comparison; the intervention rate with a CI; a list of the failures the monitor catches and the ones it misses.
- **Predictions:**
  - intervention rate per policy: ____
  - share of at-fault collisions the monitor prevents: ____
- **Kill (KC-6, D+8):** if the monitor changes the semantics of the policy under test, report it as offline only.
- **Cost:** 2 sweeps ⇒ **$55–353**.
- **Upstream value:** the hook exists; the monitor, the override path and policy+monitor scoring do not.

### S7 — nanoAlpamayo as the gate-measured student (G5 merged)
- **Why.** In AR1, a reasoning-only reward raised the critic grade (3.1 → 4.5) while consistency fell (0.62 → 0.53) and ADE worsened (2.19 m). The proxy improved while behaviour degraded. RoaD documents the covariate shift that appears when open-loop BC is deployed in closed loop.
- **Build:** the details are in `../nano_alpamayo/README.md`.
  - A data pipeline on a PAI-AV subset with location-grouped splits. Publish code, clip-UUID manifests and aggregates only, per license §4.6.
  - A small VLM backbone (AR1's 0.5B is DINOv2 + Qwen2.5-0.5B).
  - 128 discrete trajectory tokens.
  - A flow expert reading the VLM KV cache.
  - Teacher labels from Alpamayo 1.5, the cheap teacher NVIDIA recommends, or from Alpamayo 2 Super (one H100 per instance).
- **Order:**
  1. data pipeline;
  2. overfit-one-batch and loss-at-init checks;
  3. SFT;
  4. CoC SFT on auto-labels;
  5. RoaD/OPTED-style closed-loop correction;
  6. GRPO on tokens;
  7. Flow-SDE through the expert last, as an ablation, with the expert-drift monitor.

  The order follows sample efficiency: RoaD and OPTED report orders-of-magnitude better efficiency than direct RL in AlpaSim.
- **★ Huy:** RL loss math (GRPO advantages, k3 KL, clipping; AR1's unclipped variant vs AlpaGym's clipped one), reward and verifier logic, decoder math (AR1 Eq. 5), the data split and leakage audit.
- **Acceptance:** criteria A1–A5 in `../nano_alpamayo/README.md`.
- **Predictions:**
  - minADE6 on the defined subset: ____
  - relative gain from RL: ____
  - paired student–teacher margin: ____
  - Also preserved: `04` §7 E5, as written.
- **Kill (KC-7).**
  - (a) NVIDIA releases its distillation script → pivot to evaluating NVIDIA's student on the gate.
  - (b) The student misses Huy's paired non-inferiority margin after three ablation rounds → shrink the subset and fix the pipeline. The margin can be no tighter than about 2.7 pp, what 441 scenes can certify.
  - (c) minADE misses after three rounds → same response.
- **Cost:** an 8×H100 node costs $13.4–36.0/h, so a 24 h run costs $322–864. The v2 G5 line (1,600 GPU-h) re-prices to **$960–7,200**. Run-hours are unmeasured.
- **Upstream value:** no public distillation recipe or student exists. The durable value is the measured paired comparison, not the student itself.

### S8 — Throughput measurement report and flywheel specification
- **Why.** The v2 cost model was off by up to about 2×, and budgets built on unmeasured throughput fail. AlpaGym's default is an 8-step smoke preset, not production.
- **Build:**
  - a stage-level AlpaSim profile (renderer, policy, controller, synchronization) from S2–S4 Prometheus telemetry at stated presets;
  - the render-resolution experiment (1080p vs 320×576, an 11.25× pixel ratio), run as a camera-config change;
  - a batch-scaling curve;
  - optionally, an AlpaGym trainer profile;
  - the G8 flywheel specification.
- **★ Huy:** the cost model (balance equation N_gen · g ≈ N_train · t; batching crossover with KV reads) and the bottleneck diagnosis.
- **Acceptance:** every number carries its preset, SHA and GPU edition.
- **Predictions:**
  - Preserved from mastery v1.1 Stage 1b (archived): (iii) scenarios/s per RTX PRO 6000 at 1080p vs 576×320: ____
  - Also preserved: `04` §7 E1, E2 and E4.
- **Kill:** budget line only.
- **Cost:** ≈ $0 marginal.
- **Upstream value:** the instrumentation is upstream, but no stage-level profile has been published. NVIDIA names closed-loop evaluation throughput as a product metric (JR2026365).

## 4. The first seven days: S1 ships a public artifact for $0

`physloop-gate` v0 is a paired release-gate core validated on synthetic truth, plus a 441-scene power analysis. It runs on CPU only, needs no dataset, and needs no license.

| Day | ★ Huy | Agents | Output |
|---|---|---|---|
| 1 | Write `experiments/2026-10-05-s1-synthetic-truth.md` from the template and fill P-S1-1…3. Derive the exact McNemar null and the zero-failure bound by hand | Scaffold `gate/`, CI and the test harness | Experiment file with predictions |
| 2 | Write the `verdict()` core: paired non-inferiority (Tango CI), the intersection–union rule, the INSUFFICIENT state | Encode the known-answer tests (S1 acceptance) | Core with tests going from failing to passing |
| 3 | Write the road-clustered paired bootstrap and the Durkalski adjustment; specify the generator | Build the generator to that specification | Generator with known truth |
| 4 | Diagnose every Monte Carlo deviation | Run 1,000-run sweeps (± stratification; null and −30%; 2% and larger strata), with plots | False-GO, false-NO-GO and INSUFFICIENT rates |
| 5 | Power analysis: MDE, NI margins, stratum power, G3 sample size and PASS/KILL thresholds at n = 10/20/30 | Tables and figures | "What 441 scenes can certify" |
| 6 | Reconcile prediction vs measurement and explain every miss causally; write the README's claims section | Packaging, docs | Reconciliation recorded |
| 7 | Publish the repo (Huy's action); open the AlpaSim issue; write the journal | Release notes | **Public artifact + upstream issue** |

## 5. Budget tiers (verified October 2026 prices; costs are inferences)

GPU-hours are derived from FlashDrive's measured AlpaSim step latency (463–1,150 ms) × 441 scenes × 3 seeds × 200 steps. That gives **34–85 GPU-h per policy sweep**, single-stream upper bound. S2's measured throughput replaces these figures.

| Tier | Unlocks | Price basis |
|---|---|---|
| **$0** | S1; every curriculum exercise; the AlpaSim issue; the license request | CPU |
| **≤ $2.5k** | S2, S3, S4, S6, S8: about 24 sweeps, ≈ 817–2,029 GPU-h ⇒ $660–1,890 at interruptible RTX PRO 6000 rates ($0.81–0.93/h); $1,140–2,840 at the vast.ai on-demand median ($1.40/h). Storage $40–216/month | RTX PRO 6000: vast.ai $1.40–1.47, RunPod $1.69–2.09, Nebius $1.80 (preemptible from $0.79) |
| **≤ $6k** | Adds S5 ($507–1,350 reference bracket) and an S7 pilot | H100: vast.ai median $1.69, RunPod $2.69–3.49, Lambda $3.99–4.29, Nebius $4.50 (since 2026-10-01); spot $0.60–0.79. 8×H100: $13.4–36.0/h |
| **Collaborator** | Full G3 including private-set checks; nanoAlpamayo SFT and RL at scale; possibly Thor hardware | NVIDIA Academic Grant (faculty PIs only); Nebius research credits (academic or non-profit); Inception (incorporated company) |

The B200 is not needed. Its floor is $5.98/h. The Jetson AGX Thor dev kit costs $5,499 (launch price $3,499) and belongs only to the collaborator tier. Rules: stop at stage boundaries, never mid-stage; re-price at every stage start. The tier choice is open as proposal P-005 and is required before S2 spends anything.

## 6. Claims the gate may and may not make

**May.** Every claim states n, CI, seeds, manifest, AlpaSim SHA, scenario list and deviation convention.
- "Checkpoint B is not worse than A by more than Δ on every pre-specified stratum, at one-sided 95% confidence, paired and cluster-adjusted by road."
  - Δ is pre-registered and no tighter than ≈ 2.7 pp at ψ = 4% on 441 scenes.
  - Underpowered strata are reported INSUFFICIENT, and INSUFFICIENT blocks GO.
- "The per-scenario failure rate in stratum S is ≤ X, by a **one-sided** 95% Clopper–Pearson bound." For 0/1,000 that is 0.30%, against 0.37% two-sided. n counts roads or logs when scene independence is doubtful.
- "These failure modes were found" (adversarial set, reported as findings).
- The G3 claims the protocol licenses:
  - "gate rankings are reliable across seeds (ICC = …)";
  - "robust to renderer post-processing and traffic model (τ = …)";
  - "responsive to controlled degradations (k/K known pairs ordered correctly)";
  - "converge with NAVSIM pseudo-simulation on n NAVSIM-family policies (ρ = … [CI]; n_families, n_checkpoints)";
  - "diverge from open-loop minADE as expected".
- "Gate rankings agree with reference R at ρ = … [CI] on n policies", always naming R's independence level (R0–R10).

**Must not:**
- Per-mile crash rates, human parity, "safe" or "safe to deploy". RAND: 275 million failure-free miles are needed to bound a fatality rate at 1.09 per 100 million miles.
- Compliance with ISO 26262, ISO 21448, ISO/PAS 8800:2024, UL 4600, UN R157, UN R171, **UN R[185]** (not in force before about 2027-01-30) or **the UN ADS GTR** (adopted 2026-06-24). "NATM compliance" is not meaningful either, because NATM is guidance.
- "Predictive validity", "predicts on-road outcomes", "validated against the real world", or any on-road correlation not measured.
- Regulatory credibility of AlpaSim as a virtual toolchain.
- Counting ≥ 4 m exits as policy failures without reporting them separately.
- Generalizing NAVSIM-family agreement to PAI-track VLAs without R1–R3 evidence on PAI.
- Calling the PhysLoop monitor "Halos".
- Presenting paired release-to-release regression as what Waymo or any regulator requires. It is PhysLoop's own design.

## 7. Kill checks (dated for D = 2026-10-05; shift with D)

| Check | When | Rule |
|---|---|---|
| KC-1 | D+1 (10-12) | `verdict()` controls false-GO on synthetic truth; otherwise fix before any GPU spend. Re-forecast all windows with actual hours |
| KC-L | Before the first public number derived from PAI-AV or NuRec (target D+3) | A written answer on license §3, or the fallback |
| KC-2 | D+3 (10-26) | Reliability ≥ Huy's pre-registered floor; otherwise stop |
| KC-3 | D+5 (11-09) | Coverage inside the envelope ≥ Huy's floor; otherwise tighten the cap and report the loss |
| KC-4 | D+7 (11-23) | Known-group upper bound ≥ Huy's floor; otherwise downgrade validity claims |
| KC-6 | D+8 (11-30) | The monitor fits in the loop without changing policy semantics; otherwise report it offline |
| KC-5 | D+11 (12-21) | Three-way ρ rule: PASS / KILL / INSUFFICIENT |
| KC-7 | D+20 (2027-02-22), or any NVIDIA distillation release | Student margin met, or pivot to evaluating NVIDIA's student |

## 8. Risks

| Risk | Check | Response |
|---|---|---|
| License §3 blocks publication of gate results | KC-L | Publish code, scene-ID manifests and synthetic results. Whether manifests and aggregates are covered is asked in the same request (P-004) |
| The 12-month license term starts at first download | Before S2 | No download before S2 starts; record expiry = first download + 12 months |
| NVIDIA ships its 10B → ~2B distillation script | GTC Berlin (2026-10-20 to 10-22); every recipes, AlpaGym or Edge-LLM release | KC-7(a) pivot |
| NVIDIA ships a paired gate or correlation study | Each AlpaSim sync (roughly monthly; the last `main` sync, 2026-08-11, is overdue) | Contribute G2–G3 evidence upstream; independent validation keeps its value |
| Too few independent policies for G3 | Pilot at D+8; KC-5 | State n_families and n_checkpoints; INSUFFICIENT is an acceptable outcome |
| AlpaGym fragility (unmerged PR #18 pin, frozen mirror) | Every release | Build nothing on AlpaGym internals; its profile stays optional |
| Nondeterminism (NuRec run-to-run variation; unseeded CAT-K) | S2 | Report it as a noise source; keep CAT-K disabled (the default) until seeded |
| Price and supply moves (Nebius H100 +17% on 2026-10-01; thin vast.ai H100 supply) | Each stage start | Re-price; fall back to interruptible RTX PRO 6000 |
| Budget overrun | Every stage boundary | Stop at the boundary |
| Hours assumption wrong (P-003) | KC-1 | Re-forecast every window |
| Regulatory wording drifts | About 2027-01-30 (R[185]) | Update the claims wording; never claim compliance |
| Ownership drift: agents writing ★ cores | Every stage close | Owner proof (`03` §5); otherwise redo the core |

## 9. Open owner decisions

These decisions block specific stages. Recording each one closes its proposal in `DECISIONS.md`.

| Proposal | Decision | Blocks |
|---|---|---|
| P-003 | Weekly hours | Calendar accuracy; re-forecast at KC-1 |
| P-004 | Send the written license §3 request to NVIDIA (Huy's outward action) | KC-L, before any public PAI-AV/NuRec-derived number |
| P-005 | Budget tier | Any spend in S2 |
| Per-stage, pre-registered in `experiments/` | Floors and margins: reliability, envelope coverage, known-group accuracy, Δ, the G3 cut | The matching KC |

## 10. Key primary sources

The full ledger is `research/2026-10-03-verification/report.md` §10.
- **Alpamayo-R1:** https://arxiv.org/abs/2511.00088 (v2, Tables 9–10, 14; §6.1)
- **Alpamayo cards:** https://huggingface.co/nvidia/Alpamayo-1.5-10B · https://huggingface.co/nvidia/Alpamayo2-Super
- **NVIDIA repos:** AlpaSim https://github.com/NVlabs/alpasim (branch `e2e_challenge`) · AlpaGym https://github.com/NVlabs/alpagym · recipes https://github.com/NVlabs/alpamayo-recipes
- **Challenge and Drive-IRT:** https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026 · https://github.com/kesai-labs/drive-irt
- **Data:** https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles · https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
- **Simulator-validity papers:** Sim2Val https://arxiv.org/abs/2506.20553 · OmniDreams https://arxiv.org/abs/2606.03159 · pseudo-simulation https://arxiv.org/abs/2506.04218 · Bosch https://arxiv.org/abs/2605.00066 · nuPlan https://arxiv.org/abs/2306.07962 · EUVS (ICCV 2025) https://arxiv.org/abs/2412.05256
- **Deployment and post-training:** FlashDrive https://arxiv.org/abs/2608.12932 · OPTED https://arxiv.org/abs/2609.20756 · RoaD https://arxiv.org/abs/2512.01993
- **Industry gating:** Waymo https://arxiv.org/abs/2505.09880 · Zoox https://zoox.com/assets/1786550724-zoox-safety-case-framework-2026.pdf
- **Regulation:** NHTSA 91 FR 48485 https://www.govinfo.gov/content/pkg/FR-2026-07-31/html/2026-15483.htm · C.N.288.2026 https://treaties.un.org/doc/Publication/CN/2026/CN.288.2026-Eng.pdf
- **RAND Driving to Safety:** https://www.rand.org/pubs/research_reports/RR1478.html
