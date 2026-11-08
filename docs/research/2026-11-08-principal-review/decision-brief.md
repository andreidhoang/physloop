# Decision brief: does the shipped expert-RL update improve held-out closed-loop driving?

Agent-authored principal review for Dang Huy, 2026-11-08. Serves S2.6 [C] and S4.1/S4.6 [C] (G1, G9).

It builds on three earlier reviews and re-derives only what was stale or missing:
- the [November 1 review](../2026-11-01-rl-direction-review/report.md);
- the [D-022 gap review](../2026-11-01-rl-direction-review/gap-review.md);
- the unadopted [P-032 contribution review](../2026-11-05-contribution-review/report.md).

**No GPU, rental, license, download, scientific run or protected-file edit occurred.** Every `____` stays Huy's.

Supporting material:
- Source re-check: [source-recheck.md](source-recheck.md).
- Computation: `evidence/principal-review-20261108-v1/`.
- Independent review: a separate agent re-derived the numbers and checked the logic. Its corrections are applied below.

## 1. The central question, after verification

> When NVIDIA's shipped AlpaGym update is applied to Alpamayo 1.5's continuous action expert in AlpaSim reconstructed scenes, does closed-loop driving improve on held-out scenes; and if not, can one diagnosed cause be changed to show an improvement?

Three of the question's premises needed correction ([source-recheck.md](source-recheck.md)):

- **"NVIDIA's RL update" is a shipped recipe, not a published result.**
  - No NVIDIA closed-loop effect of AlpaGym expert GRPO on Alpamayo 1.5 was found on 2026-11-08. This was an agent search that found nothing (NOT FOUND), not a confirmed absence.
  - The release (`nvidia/Alpamayo-1.5-10B`, revision `7aba829`) was RL post-trained "for reasoning/action consistency". That is a different stage.
  - The model card gives one AlpaSim score with no before/after comparison.
  - PhysLoop therefore tests the shipped code and reward at PhysLoop-registered settings, because no preset defines the run (gap review G4).
  - A null result is not a failure to reproduce NVIDIA, and a gain does not confirm NVIDIA.
- **The update trains the whole non-VLM action path.** That is the 2.3B expert transformer plus `action_in_proj` (including a LayerNorm) and `action_out_proj`. The visual and language weights stay frozen.
- **Training and evaluation are different closed loops.** In both, the expert emits 64 waypoints at 10 Hz as acceleration/curvature controls, and AlpaSim's MPC tracks them.

  | | Gym training | Challenge evaluation |
  |---|---|---|
  | Sampler | 5-step SDE | 10-step ODE, with CoC sampling |
  | Replanning | 5 Hz (smoke preset) | 10 Hz |
  | Controller | Linear MPC | Nonlinear MPC |
  | Driving time | About 4.4 s | About 18 s |

## 2. What exists: implemented, measured, proposed

| Level | What is true now | Source |
|---|---|---|
| Measured | S1 gate qualified on synthetic truth (D-016). Identical-policy controls GO 1000/1000; gross regression NO-GO; noisy-equivalent controls INSUFFICIENT about 100% of the time | [S1 handoff](../../S1_HANDOFF.md) |
| Measured | S2 CPU preparation passes 1,360 tests on frozen source. The clean lockfile rebuild has **not** passed: uv 0.8.17 panicked in the macOS SystemConfiguration lookup during a sandboxed agent run. It has not been tried outside the sandbox | `evidence/s2-prerental-hardening-20261107-v1/`, `evidence/s2-clean-rebuild-20261107-v1/` |
| Measured | One synthetic Vast host check: $0.60, instance destroyed (D-020). It qualifies no AV stack and no two-GPU training | [host report](../2026-11-01-vast-host-validation.md) |
| Source-verified | Gym's update path (see the list below the table) | [nvidia-stack.md](../2026-11-01-rl-direction-review/nvidia-stack.md), [source-recheck.md](source-recheck.md) |
| Source-verified | The training and evaluation paths compute different functions of the same weights, in six factors | P-032 |
| Algebra/CPU | A group of two gives an advantage of ±1 regardless of the reward gap. Non-inferiority (NI) needs 36/51/72/183 independent roads per stratum at margins .10/.07/.05/.02, even when every difference is zero. Common random numbers are the strong power lever | gap review G1, P-032 |
| Algebra/CPU | Superiority power, §3 below | this review |
| **Not measured** | Any project rollout of Alpamayo 1.5 in AlpaSim; the road count R; the noise coupling f; base rates; cost per rollout; training fit; any learning curve | — |

Gym's update path, as read in source:
- it trains the non-VLM action path with the VLM frozen;
- the reward is `P − 10C − 5O − 0.01D`;
- the group-standardized advantage is copied to every step;
- KL is off;
- staleness only produces a warning;
- failed rollouts are retried, then dropped.

The six factors that differ between the training and evaluation paths:
- image size;
- CoC;
- RoPE;
- SDE-5 vs ODE-10;
- 5 vs 10 Hz;
- linear vs nonlinear MPC.

## 3. Critical gaps, ranked by whether they can change the answer

### 1. No implemented rule can declare "improved"

The plan already says non-inferiority GO is not improvement (§6). What is new is that no rule for improvement exists.
- `verdict()` requires 0 < margin < 1, so it certifies non-inferiority and flags regression only.
- The bound needed is already computed. `verdict()` returns per-stratum `StratumResult.lower` and `.upper`, and the margin does not change them.
- `S1_CORRECTNESS.md` shows the upper endpoint covers the true difference with probability at least 1 − α.

So the rule "upper < 0 on every registered stratum, with the guards GO" is a valid one-sided level-α intersection-union test, under the S1 assumptions, at one fixed look. Three conditions apply:
- **Implementation:** decide from `verdict()`'s or `SequentialGate`'s own outputs. Do not re-implement its averaging, which would duplicate protected logic.
- **Looks:** with interim looks, use the registered spent α.
- **Runs:** with three runs, register how run-level verdicts combine (§4).

### 2. Showing improvement needs far more information than showing non-inferiority

The computation (`superiority_power_output.txt`) uses the locked bound at α = 0.05. Its inputs are illustrative, not measured. **R counts independent roads per stratum**, because every stratum must pass, so S strata leave at most 441/S each.

| Model | Quantity | R = 441, b = 0 | R = 441, b = 2% | R = 441, b = 5% |
|---|---|---|---|---|
| One binary outcome per road. A fraction b of roads is worse with the candidate (true regressions plus noise) | Net improvement for 80% power (exact) | ≈3.0 pp | ≈7 pp | ≈9.5 pp |
| Same | Power at the preserved target, 1.6 pp | 17% | 1% | ≈0% |
| Same | Smallest *observed* net that passes (≈50% power if that is the true value) | 2.3 pp | 5.4 pp | 7.3 pp |
| Continuous [0,1] road score, noise σ = 0.10 / 0.20 | Mean improvement for 80% power (±0.005 Monte Carlo) | 0.060 / 0.080 | — | — |

**What this means for the target.** The plan's preserved target, at-fault collisions down 20% relative, is 1.6 pp at a hypothetical 8% base rate. It has at most 17% power even in the ideal case of 441 independent roads in one stratum with no road getting worse.

**A note on b.** b cannot exceed the candidate's own failure rate.

**RoaD, as a magnitude only.** RoaD reported a collision change of 5.25% → 2.39% on 920 scenes.
- The comparison is across mismatched protocols: a different policy, closed-loop SFT, discarded ≥4 m rollouts, and per-scene rates averaged over rollouts.
- Read only as a magnitude, a 2.9 pp effect at R = 441 has about 80% power at b ≈ 0 and under 10% once 2% of roads are worse.

**A reference test.** The exact sign (McNemar) test on discordant roads tests the same null for one binary outcome per road. It needs about half the gate's threshold. Adopting it would be a new evaluator core needing KC-1 review. The P-032 audit already found empirical Bernstein looser than the locked bound.

**The levers,** as P-032 named them, now applied to superiority:
- more independent roads;
- lower discordance b through common random numbers;
- an endpoint with more information per road.

S2 measures R, f and σ. Until then, any S4 superiority forecast is a guess.

### 3. The evaluated policy path is undecided (P-032 item 1)

If S4.6 evaluates through the deployed `e2e_challenge` path while RL optimizes the Gym path, a null result cannot tell apart two explanations:
- RL does not generalize across scenes;
- the trained function is not the evaluated function.

This must be registered before the S2 pilot, because S2's noise, power and cost are measured on the path S4.6 reuses.

### 4. "Unseen" is not yet defined

What is missing:
- The S4 training scene set is a blank slot.
- The 441-scene list, its road identities and its strata are unregistered.
- The population audit (`audit_population`, S2 CPU-04) has not run, because road metadata needs the licensed dataset.

What is known about overlap:
- **Teacher-training clips.** P-032 inferred that at least 666 of 2,202 curated scenes carry an `alpamayo_aligned` label other than `alpasim`. So some evaluation scenes may come from clips used to train the teacher.
- **The curated split.** The 1,761/441 split groups road-sharing scenes into one split. It still allows other roads or directions at the same location to fall into different splits, and it says nothing about teacher training.
- **The `alpasim` label.** At v26.11 it covers 1,550 clips, fewer than the 2,202 curated scenes. The counts in `04` are stale.

What "unseen" can therefore mean:
- relative to PhysLoop's RL training, it can be defined exactly;
- relative to pretraining, it can only be bounded per scene from the label, with unknown provenance labelled `unverified`.

### 5. The training horizon is about a quarter of the evaluation horizon

Training drives about 4.4 s; evaluation drives about 18 s, with a different controller and replanning rate. A policy trained on short episodes can be judged on behavior it never optimized. Horizon and path shift must therefore be held fixed or measured before a null result is attributed to RL.

### 6. Reproducibility is not yet established

- The clean rebuild has not passed.
- The stock evaluation wrapper and Gym's download instructions both load the checkpoint without a revision, so the manifest must pin `7aba829`.

## 4. Proposed hypothesis (agent wording; Huy registers it)

**H₁ (checkpoint level).** For each of the B1–B3 checkpoints (§5), the claim is a comparison with the zero-update checkpoint, on road-disjoint held-out scenes:
- **Setup:** the checkpoint is produced by training the non-VLM action path with the shipped AlpaGym update, using reward `progress_safety`, the VLM frozen, settings `____` and budget `____`.
- **Primary:** it reduces the registered primary failure rate on every registered stratum.
- **Guards:** it does not regress beyond margin `____` on progress, offroad or the ≥4 m rate.

**How run-level verdicts combine:** `____`. The options:
- *all three runs pass*: an intersection-union test, valid at α;
- *at least k of 3 pass*: needs Holm or Bonferroni;
- *any run passes*: inflates the error rate to as much as 3α.

The result stays a statement about named checkpoints, not about the training method (plan §6).

**What "drives better" means.** These are recommended defaults; Huy decides.
- **Primary:** processed `collision_at_fault`, paired and road-adjusted. It counts front or lateral contact, episode-any, after warm-up, and is truncated at the first incident. Three things go with it:
  - **Shared apparatus.** It shares the scorer, ground truth and renderer with the training reward. It is a different function of the same apparatus, not an independent instrument.
  - **Known scorer defect.** The pinned `CollisionScorer` misclassifies simultaneous multi-actor contacts (AlpaSim PR #234 / issue #237). Test E24 must pin this behavior.
  - **Truncation bias.** Truncating at the first incident means a candidate that goes offroad or leaves the envelope earlier records fewer collisions. Register how ≥4 m rollouts count in the primary, and consider an "any incident" sensitivity endpoint.
- **Missing data:** a failed or missing candidate attempt counts as a failure for the superiority claim, so attrition cannot manufacture an improvement. Complete attempt accounting is kept alongside.
- **Always reported, never pooled into the primary:**
  - rear contacts and all contacts;
  - black frames and render failures;
  - stalls, oscillation and comfort;
  - infrastructure failures;
  - every attempted rollout.

**Four registered outcomes per stratum.** δ is the paired difference; positive means the candidate is worse. δ_min = `____` is Huy's smallest improvement worth having.

| Outcome | Condition |
|---|---|
| Improvement of at least δ_min not excluded | upper < 0 and lower ≤ −δ_min |
| Real but below δ_min | upper < 0 and lower > −δ_min |
| Meaningful improvement ruled out | upper ≥ 0 and lower > −δ_min. This is the falsifier |
| INSUFFICIENT | upper ≥ 0 and lower ≤ −δ_min |

**Multiplicity for the falsifier.**
- Reporting "ruled out" per stratum needs Holm across strata.
- Requiring it on all strata is valid at α, but it is not the complement of H₁.
- Mixed strata are reported as mixed.

**Do not expect the falsifier to fire at target scale.** At R = 441 with a true δ of zero, the lower bound clears −1.6 pp in only 0.4–2.5% of trials when discordance is 2–5% (the independent reviewer's simulation). Even with every difference zero, it needs R ≥ 229 per stratum. A negative result is informative only if S2 shows large R and small b. Otherwise INSUFFICIENT is the expected outcome.

**Keep three levels apart:**
1. the training reward rose;
2. a simulator metric moved;
3. the registered rule passed.

No level follows from the one before it.

## 5. Baseline and evaluation design (recommended defaults; Huy decides)

**Arms.**
- **A0** is the zero-update conversion of the release, evaluated through the registered path. It, not the release itself, is the control, because conversion can change behavior (E02).
- **B1–B3** are one checkpoint per independent training run, chosen by a rule frozen before training. The rule must be able to return step 0.
- **Optional reward-free control (Huy's choice, priced separately).** It runs at the same budget unit with advantages shuffled within groups, and separates "the reward signal" from "any gradient on these scenes". It costs one run and a small adapter.

**Data roles.** Training scenes come from the 1,761 pool. Development scenes are for diagnosis and checkpoint selection. Confirmation scenes are registered held-out scenes, used once.
- All three roles are road-disjoint under the registered overlap rule.
- Teacher-pretraining overlap is reported per scene from `alpamayo_aligned`, and the confirmation result is also shown on the `alpasim`-labelled subset.

**Pairing.** A0 and B use the same scenes and, once qualified, the same seed schedule (common random numbers; P-032). Report f separately for the re-export control and for B, because coupling weakens as the weights diverge.

**Uncertainty.** The road is the independent unit. Training-run variation is reported per run and never multiplied into the road count. Three runs give a checkpoint-level verdict per run plus a descriptive run spread, not a method-level interval.

**Fair budgets.** One primary budget unit is fixed before training: equal paid GPU-hours, equal simulated interaction or equal optimizer steps. Every update count names its unit.

## 6. Success and stop criteria (structure only; the numbers are Huy's)

| Point | Stop or branch when | Result to report |
|---|---|---|
| Correctness (KC-4) | E01–E04 parity, frozen-mask or version checks fail | Engineering defect; no learning claim |
| Learning signal | Training reward on train scenes is unchanged after `____` optimizer steps, or the informative-group fraction is below `____` | Optimization or signal failure, diagnosed before any transfer claim |
| Transfer | Reward rises on train scenes, but development outcomes do not move, or regress from step 0 | Transfer failure. Diagnose: reward/evaluation mismatch, path shift, horizon, drift or a simulator artifact |
| Confirmation | The §4 rule | One of the four outcomes per stratum, plus the registered run aggregation |
| Budget (KC-$) | The next unit plus teardown exceeds the remaining cap | Stop, export, destroy, verify absence |

## 7. Diagnosis and the one fix (conditional)

Only after development evidence:
1. Choose one mechanism from the plan's list (S4.5), plus P-032's path shift.
2. Register one change, a numeric prediction and a falsifier.
3. Compare against both A0 and the original RL arm, on fresh training runs at the same budget unit.

**Budget reality.**
- The gap review's provisional $650 S4 training line buys about 406–650 GPU-hours at $1.00–1.60/h.
- The whole $1,300 left after a maximal S2 buys about 810–1,300 GPU-hours.
- That $1,300 must also cover independent evaluation. One 441 × 3 sweep is historically 34–85 GPU-hours.
- No rendered-camera precedent shows learning at this scale.

Plan for one of three results: a baseline, an explained negative, or INSUFFICIENT. Treat the intervention arm as unfunded until S2 and the S4 engineering check price it.

## 8. Next executable experiment

The next experiment is the registered **S2 five-scene pilot** on the registered evaluation path, unchanged from the plan. Four items block it, in order:

1. **A clean rebuild of the frozen lockfile.** Retry outside the agent sandbox or with another installer version. This is an environment problem, not a code change.
2. **Huy's registrations:**
   - the evaluation-path estimand (P-032);
   - the 441-scene list with road identities and strata;
   - seed mode;
   - margins;
   - terminal and missing-data semantics;
   - the pilot prediction.
3. **Huy's license acceptance** for PAI-AV/NuRec, and for the Harmonizer assets if used.
4. **A concrete rental approval** under the $100 pilot cap.

**Built on 2026-11-08 at Huy's request (candidate, not registered).** `gate/superiority.py` and `gate/superiority_power.py`, outside the protected paths:
- They read `verdict()` and `SequentialGate` endpoints and re-implement no bound or averaging.
- Inputs that must be registered: strata, guards, δ_min, run ids and the aggregation rule. None of them has a default.
- The improvement and ruled-out claims are reported separately, each with its own level. The joint "below δ_min" statement is labelled at the summed level.
- Unadjusted flags mark a stratum the bound certifies as worse while the gate's regression flag is silent.
- Validation of rule v1:
  - 67 known-answer tests and 20 of 20 mutations killed;
  - full suite 1,427 passed, Ruff clean;
  - an independent review found no level-α bug, and its labelling, validation and seeding fixes are applied.
- Evidence for v1: `evidence/superiority-rule-20261108-v2/`.
- **Current rule v4.**
  - It adds registered guard and gating strata, a paired-observation check, guard-stratum worse-bound flags and comparability of guard definitions across runs.
  - It has had a second independent review: 94 tests, 29 of 30 mutants killed (the survivor is equivalent), full suite 1,454 passed.
  - Evidence: `evidence/strata-pricing-20261108-v2/`.
- Fixed-sample power through the real gate at R = 441 in one stratum, with illustrative inputs:
  - 3.0 pp net improvement: 0.85;
  - 1.6 pp: 0.19;
  - 9.5 pp with b = 5%: 0.85;
  - the same 441 roads split into two strata at 3.0 pp: **0.02**.

**Strata and guard pricing (2026-11-08): [strata-pricing.md](strata-pricing.md).**
- Do not register "improved in every stratum". With four 110-road strata, power is 0.000 even at Δ = 5 pp.
- Claim pooled improvement, with per-stratum non-inferiority guards at 0.10.
- The guard must pass in every gating stratum at once. At b = 2%, that means about 120 roads per stratum for two gating strata and about 150 for four.
- Each small stratum is your choice. Guarding it protects it: at b = 0, harm of +10 pp drops the claim rate from 0.333 to 0.016. At b = 2% it costs about half the power (0.451 → 0.239). Reporting it keeps the power (0.437 at b = 2%), but leaves its harm only visible in the report.
- Discordance b decides the outcome: pooled power at 3 pp is 0.85 at b = 0 and 0.09 at b = 2%.
- Replicates do not lower the detectable effect, because the locked bound ignores variance. Common random numbers are the lever.

**Still open for Huy:**
- whether the falsifier is per stratum (needs its own multiplicity control) or all strata;
- whether to name a "worse" outcome when the lower bound is above zero;
- whether the joint below-δ_min statement at the summed level is acceptable;
- fixed-sample or sequential analysis for superiority. Under `SequentialGate` a stratum closes at non-inferiority, so superiority is in effect tested at that look, and no sequential superiority power is computed yet.

## 9. What a result would and would not establish

**A supported result means:** the named checkpoints reduced the primary simulator endpoint, under this checkpoint, these recipe settings, this simulator path, non-reactive replay traffic and the registered held-out roads.

**It does not establish:**
- real-world safety;
- a property of the training algorithm beyond the three runs;
- behavior under reactive traffic;
- anything about Alpamayo 2.

**What should transfer to larger scale is the method:**
- the reward-to-evaluator reconciliation;
- the path-parity check;
- complete attempt accounting;
- the paired road-level rule.

**What larger studies must re-test:** whether the effect survives more roads, reactive traffic, the deployed path and independently trained runs.

## 10. Plan changes

**Applied as D-023 (`docs/DECISIONS.md`), in `02`:**
- **S2.6:** report superiority sensitivity alongside NI, per stratum.
- **S4.1:** the trainable set and the checkpoint revision pin.
- **S4.6:** an improvement claim needs a registered superiority rule, a falsifier and run aggregation.
- **New S4 registration slots,** all blank.
- **§6:** the claim name "the shipped AlpaGym update at registered settings".

**Left for Huy as proposals:**
- P-032's path estimand (§3 item 3);
- the primary endpoint and its ≥4 m convention;
- the four-outcome rule, δ_min and run aggregation (§4);
- the reward-free control (§5).
