# Review of the 2026-10-25 decision doc

**Status:** evidence record (`../../../DOC_AUTHORITY.md`). Live docs win where they differ.
**Input reviewed:** `decision_input_2026-10-25.md` ("Alpamayo-Grade AV Engineering: #1 Project Decision", 2026-10-25).
**Companion files:** `verification.md` (claim ledger, four research lanes, 2026-10-25) · `notes/arith_check.py` and its output (every number in §3, reproducible with `python3 -I notes/arith_check.py`).
**Reviewer's frame:** a principal physical-AI engineer deciding whether to stake the next quarter on this plan. Inferences are labelled as inferences.

---

## 1. Verdict

**Adopt the direction, amend the gates.** The doc's strategic read is right: the released Alpamayo checkpoints are cloud teachers, and the two widest open gaps in the ecosystem are the teacher → student distillation and real-time serving on Thor. Building the loop that closes those gaps is a stronger headline than plan v3's construct-validity study.

But the doc's acceptance gates cannot be decided without the instrument plan v3 already specifies. Its RL gate is underpowered on 441 scenes at realistic base rates (§3.2). Its distillation gate is an aggregate ratio on an instrument whose own card moved 69% with unchanged weights (§3.3). Its reproduction gate targets a number with an undefined "±" (§3.4). Its latency table puts the reasoning student at 139 ms against a 100 ms budget by its own formula (§3.1).

So the honest shape is: **the gate is the instrument, and teacher → student → Thor is the headline application it measures.** Neither replaces the other. That is what plan v4 records (`../../02_PHYSLOOP_PLAN.md`, `../../DECISIONS.md` D-012).

**Verification then activated the doc's own contingency** (`verification.md` §0). The doc's risk table says: if NVIDIA ships Edge or a distillation recipe, "use Edge as the P2 baseline". Two facts now point that way:
- **A distillation recipe is public.** alpamayo-recipes PR #42 (unmerged, 10-20) produces the same Cosmos-Reason2-2B student the doc planned to build. It ships no weights and recommends at least 128 H100s.
- **Edge is announced.** Alpamayo 2 Edge (2B, DRIVE AGX Thor, 10 Hz) was announced on 10-21, with download promised "later this month".

So v4 does not race NVIDIA to a 2B student. Its student stages do three things instead:
1. measure NVIDIA's own student in closed loop, paired against its teacher. Nobody has published that; NVIDIA reports open-loop minADE and latency only;
2. correct the student on-policy;
3. map what the runtime reasoning budget buys under a real latency budget.

NVIDIA's own 2B students run at 171 ms in FP8 on DRIVE AGX Thor (PR #45), while Edge claims 10 Hz. So "what did NVIDIA trade to reach 10 Hz?" is a measurable question that survives any release.

## 2. Where the doc conflicts with plan v3, and how v4 resolves each

| # | Topic | Plan v3 (2026-10-03) | Decision doc (2026-10-25) | v4 resolution |
|---|---|---|---|---|
| 1 | What PhysLoop is | An open release gate; nanoAlpamayo as the gate-measured student | Teacher RL → ≤ 2B student → 10 Hz on Jetson Thor → HCMC shadow mode | Gate = instrument (S1–S2). Teacher → Thor = headline application (S3–S7). Every closed-loop number still comes from the gate |
| 2 | AlpaGym | "Build nothing on AlpaGym internals"; code-frozen, no outside PRs, expert-only RL | P1 runs GRPO in AlpaGym, with an AlpaJudge reasoning–action reward | Use AlpaGym as a *user*, never as a contributor. A reasoning reward only counts if the VLM is trainable in that path (see `verification.md` B1). Fallback: open-loop GRPO in alpamayo-recipes |
| 3 | Thor hardware | Jetson AGX Thor ($5,499) is collaborator-tier only (P-002) | Buy it | Owner decision P-002. It blocks S5–S7, not S1–S4 |
| 4 | Budget | Tiers $0 / ≤ $2.5k / ≤ $6k / collaborator; P-005 open | $11–15k all-in | P-005 restated with v4 tiers (§3.5). No spend before it is recorded |
| 5 | Decision rules | Paired, cluster-adjusted non-inferiority; INSUFFICIENT blocks GO; one-sided bounds labelled | Relative thresholds on aggregates ("≥ 20%", "≥ 90%", "± 0.1") | Huy's thresholds are kept as his targets, restated in paired form before pre-registration; certifiability is checked against S1's power tables |
| 6 | Publication | KC-L: no public PAI-AV/NuRec-derived number before a written answer on license §3 | "Open-source repo, tech report, demo"; KC-L not mentioned | KC-L carried forward and extended (KC-L2): no student weights trained on PAI-AV-derived labels are published without a written answer on §4.6 (derivatives) |
| 7 | G3 construct validity (v3 S3, S5) and G4 guardrail (v3 S6) | Headline and core stages | Absent | Deferred, not deleted. Reliability and validity records stay inside S2. The student pipeline supplies known-order ladders (R7/R9) at near-zero marginal cost |
| 8 | Student | nanoAlpamayo from small parts (AR1-0.5B style) | Cosmos-Reason2-2B + 0.3–0.5B flow expert, distilled by us | NVIDIA's own 2B student (Edge once released; PR #42's recipe at pilot scale as the fallback). nanoAlpamayo becomes the closed-loop correction and CoC-budget work on that student. The AR1-0.5B layout stays as a small ablation |
| 9 | Calendar | D = 2026-10-05; KC-1 2026-10-12 | 12 weeks | S1 never started (no `gate/`, `journal/` or experiment file on 2026-10-25). D re-based |
| 10 | Vision tokens | 180 per image, 2,880 per step (derived, `04` §2) | ≈ 160 per image, 2,560 per step | 180 / 2,880 for Alpamayo 1.5 at 320×576; 160 is the AR1 paper's 448×280 setting |
| 11 | Cross-project priority | D-009: PhysLoop takes no decision from another project | Calls for a priority call against another project | Out of scope for PhysLoop docs (D-009). It is the owner's call elsewhere |

## 3. Arithmetic check (`notes/arith_check.py`)

### 3.1 The Thor latency table

The doc's formula is t ≈ 2·N_bb·T_vis / (η·F_peak) + n_tok·B_bb / BW + K·B_exp / BW, with F_peak = 517 TFLOPS dense FP8, BW = 273 GB/s and η = 0.4.

| Config (Jetson AGX Thor) | Doc total | Re-derived, T = 2,560 | Re-derived, T = 2,880 |
|---|---|---|---|
| 10B FP8, K = 10, 20 CoC tokens | ≈ 0.9 s | 888 ms | 913 ms |
| 10B, NVFP4 decode, FP8 expert | ≈ 0.7 s | 625 ms | 651 ms |
| 2B NVFP4 + 0.5B FP8 expert, K = 4, 20 tokens | ≈ 135 ms | **139 ms** | **145 ms** |
| 2B, no CoC, K = 2 | ≈ 55 ms | 53 ms | 59 ms |

The rows reproduce. Four consequences the doc does not draw:

1. **The CoC student misses 100 ms by the doc's own formula.** Only the no-CoC arm fits. NVIDIA's own 2B students measure 171 ms (FP8) and 221 ms (FP16) on DRIVE AGX Thor (PR #45), consistent with this floor.
2. **These are floors, not estimates.** On Alpamayo 1.5, FlashDrive measured decode at about 78% of its bandwidth bound and the flow expert at about 20% of it, even after CUDA graphs (`04` §4). With those efficiencies, the 2B student fits only **≈ 2.6 CoC tokens** in 100 ms.
3. **Streaming KV reuse of history frames is a precondition for any reasoning arm.** Re-encoding only the 4 new images of 16 cuts prefill about 4×. The budget then rises to ≈ 9.7 tokens at measured efficiencies, or ≈ 19.5 at the bound. FlashDrive reports that this reuse costs +2.7% minADE6 (0.770 → 0.791), so it has its own accuracy price.
4. **The research question is a curve, not a binary.** "CoC vs no-CoC at 100 ms" becomes: *closed-loop score as a function of the CoC token budget n ∈ {0, 4, 8, 16} at fixed architecture, with latency measured on the device*. The 100 ms point is read off that curve. Designing the experiment, and the prediction, are Huy's.

Minor: the doc's prose (11 GB ÷ 273 GB/s ≈ 40 ms/token) includes the expert's bytes. Decode reads the backbone only: 8.2 GB ÷ 273 GB/s ≈ 30 ms/token, which is what its table uses.

The physics claim stands: even at η = 1, prefill of a 10B model over 2,560 tokens takes ≈ 99 ms on Jetson Thor, so no 10B configuration can run at 10 Hz.

### 3.2 The RL gate: "at-fault collisions down ≥ 20% relative on held-out scenes"

A relative reduction is small in absolute terms when the base rate is low. Required paired scenes (Connor 1987, one-sided α = 0.05, power 0.8), against the 441-scene held-out split:

| Base at-fault rate | 20% relative = | Discordance ψ | Scenes needed | Decidable on 441? |
|---|---|---|---|---|
| 4% | 0.8 pp | 0.8% (RL only fixes, never breaks) | 771 | no |
| 8% | 1.6 pp | 1.6% (only fixes) | 385 | yes, but only with zero churn |
| 8% | 1.6 pp | 4% | 964 | no |
| 16% | 3.2 pp | 4% | 240 | yes |
| 16% | 3.2 pp | 8% | 481 | no on 441; yes on 441 × 3 seeds (n_eff ≈ 601 at design effect 2.2) |
| 19% | 3.8 pp | 8% | 341 | yes |

The 4% and 16% rows are the AR1 10B paper's closed-loop rates. The 19% row is FlashDrive's collision rate on 100 AlpaSim clips. The base rate under the pinned challenge protocol is unknown until S2 measures it.

**Consequence.** Whether P1's gate can be decided depends on two numbers that S2 measures (the base rate and the discordance) and on the tables S1 builds. At low base rates the gate's correct output is INSUFFICIENT, which is a reportable result, not a failure. Seeds help less than they appear to: 3 seeds at an intra-scene correlation of 0.6 are worth about 1.36 scenes each, not 3. Clustering by road shrinks the effective n further. Huy may instead choose a composite failure endpoint (collision or off-road), more held-out scenes, or a larger effect, but he must choose *before* the run.

### 3.3 The distillation gate: "student ≥ 90% of the teacher's AlpaSim score"

**Which "AlpaSim score" is meant decides the statistics** (`verification.md` A4). Two different scores exist:
- **The card score** (1.37). The AR1 paper defines its AlpaSim score as average km driven between events, with deviations beyond 4 m *excluded*. That is an unbounded rate, and the exclusion conflicts with PhysLoop's rule that ≥ 4 m exits are reported separately, never dropped.
- **The challenge `scene_score`**, in [0, 1]. It is clipped relative progress, set to 0 on an at-fault collision or off-road.

Under the challenge score, 10% of a teacher mean of 0.5–0.9 is a margin of 5–9 pp. Because `scene_score` is continuous, the binary McNemar MDE (≈ 2.7 pp at ψ = 4%) does not apply. The paired continuous MDE is (z_α + z_β)·σ_d / √n, where σ_d is the SD of per-scene teacher − student differences. The score is bimodal (0 on failure, near 1 otherwise), so σ_d is plausibly 0.3–0.5. At n = 441, one-sided, 80% power and before cluster adjustment, the MDE is 3.55 pp at σ_d = 0.3 and 5.92 pp at σ_d = 0.5.

**So the gate is decidable only if σ_d is small enough:**
- σ_d ≲ 0.59 for a 7 pp margin (teacher mean 0.7);
- σ_d ≲ 0.42 for 5 pp (teacher mean 0.5).

S2 measures σ_d; until then decidability is unknown (`notes/arith_check.py`).

Under the card-style rate, the test becomes a paired rate ratio over km. Its power depends on event counts, which S2 must measure first.

Restating the gate on one named metric is Huy's decision. S2 computes both, and labels the card-style number as a diagnostic under the paper's convention.

On either metric, teacher and student must be scored in the same pinned run. NVIDIA's own card moved 0.81 → 1.37 (+69%) with unchanged weights, 6.9× the margin. A ratio taken across runs or protocols measures the protocol, not the student.

### 3.4 The reproduction gate: "match the model-card AlpaSim score ± 0.1"

The two card values differ by 0.56, the card never defines "±", and the scene lists behind them (910 and 913 scenarios) are not the public 441. Matching is a protocol-identification problem, so it works as a **diagnostic**, not a gate. The gate is S2's: a pinned protocol with a measured seed test–retest noise floor. The doc's own fallback ("open a GitHub issue; report relative gains only") is the right response to a persistent gap.

### 3.5 Budget

v3 prices one evaluation sweep (441 scenes × 3 seeds × 200 steps) at 34–85 GPU-h. The doc implies about 24 sweeps:

| Phase | Sweeps | GPU-h |
|---|---|---|
| P0: TransFuser and Alpamayo 1.5 | 2 | 68–170 |
| P1: base + 3 reward variants × 3 training seeds | 10 | 340–850 |
| P2: teacher + 2 arms × 4 stages (KD, DAgger, GRPO, step cut) | 9 | 306–765 |
| P3: format check | 3 | 102–255 |
| **Total** | **24** | **816–2,040** |

At $1.00–1.60/h (the doc's RTX PRO 6000 prices), evaluation alone costs **$0.8–3.3k**. Adding RL ($434–1,080 at the doc's H100 prices) and distillation ($800–1,500), the cloud line is **≈ $2.1–5.9k**, against the doc's $2–3k. The doc's figure is the optimistic end. Hardware ($5.5k Thor; $3–8k rig) is on top.

### 3.6 Effort (inference)

Twelve weeks at the plan's assumed 25–30 h/week is 300–360 hours. Plan v3 alone estimated its student stage at ~150 h. The doc adds teacher RL, Thor kernels, HIL and a sensor rig, so the scope plausibly needs 400 h or more: 13–16 weeks or longer. P-003 (weekly hours) is still unrecorded. Re-forecast at KC-1.

## 4. What a principal engineer adds

1. **Shadow mode is open-loop evaluation.** The student never controls the car, so it never sees the consequences of its own errors. Covariate shift, the reason closed-loop exists, is invisible there. Disagreement with the driver is the same class of metric as open-loop L2, which shows no significant correlation with closed-loop score (Bosch, n = 7; `01` C3). Shadow mode is valuable for three things: mining out-of-distribution data, measuring real-sensor latency and robustness on Thor, and building a real-world open-loop benchmark. It is **not** validation of driving quality, and v4's claims list says so.
2. **The rig confounds scene shift with sensor shift.** Matching FOVs is not matching intrinsics, extrinsics, mounting height or ISP. A high disagreement rate on HCMC motorbike swarms mixes "HCMC is OOD" with "this rig is OOD". A within-rig contrast (wide arterials with lane lines vs dense mixed traffic) separates them.
3. **Recordings of public roads contain personal data** (faces, plates). Anonymize before anything is published. Vietnam's Personal Data Protection Law 91/2025/QH15, Art. 32, permits recording in public places in listed cases and requires notifying the people recorded (`verification.md` D7; an engineering reading, not legal advice).
4. **The pipeline produces known-order ladders for free.** Flow steps K = 10 → 4 → 2, bit-widths BF16 → FP8 → NVFP4, and the CoC token budget each give checkpoints with a predicted order. They are G3's known-group evidence (R7/R9) at almost no marginal cost, so v4 keeps a narrowed G3 without a separate stage.
5. **Latency injection is the HIL experiment NVIDIA would ask about.** Lockstep HIL scores the policy as if latency were zero. Replaying the measured on-device latency as action delay in AlpaSim shows how much score the real latency costs. The doc mentions it only in its Vietnamese notes; v4 makes it S6's headline.
6. **Execution is the binding constraint, not the plan.** S1 is about 20 hours, costs $0 and runs on a CPU. It had not started 20 days after its start date. Every gate in the new doc, from P0's noise floor through P1's effect size to P2's margin, is an output of S1's `verdict()` and power tables. Whatever the direction, the first milestone is S1.

## 5. Open questions this review cannot settle

- The base at-fault collision rate and discordance under the pinned protocol (S2 measures both).
- RL gain per GPU-hour in AlpaGym at the doc's 225–450 H100-h budget. NVIDIA's own production run reportedly used far more compute (`verification.md` A5). That is a prediction for Huy to write, not a fact.
- Whether a student distilled from teacher labels on PAI-AV clips is a "derivative" under the data license §4.6. Only NVIDIA can answer that (P-004, extended).
