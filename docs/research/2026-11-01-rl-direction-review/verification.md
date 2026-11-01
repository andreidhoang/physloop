# Verification record for plan v5

Date: 2026-11-01. Agent-authored research and document verification for S2 measurement preparation and the proposed S4/G9 focus. Huy explicitly authorized updating the plan after independent verification; D-021 records adoption. This record separates that completed review from unexecuted GPU/model experiments.

## Scope and reviewers

| Lane | Work | Independent review received |
|---|---|---|
| AV literature reviewer (`stage_review`) | 13 AV papers; inputs, privileged information, algorithms, benchmark and transfer limits | Root reviewed the ledger; this reviewer challenged synthesis, adjacent papers, plan and final context |
| Foundations reviewer (`test_review`) | 17 foundational/methodological papers; estimators, constraints, reward exploitation and statistical evidence | Root reviewed the ledger; this reviewer challenged protocol, plan, uncertainty and final corrections |
| NVIDIA source reviewer (`release_check`) | Official repositories, exact pins, training scope, hardware documentation and reward/runtime behavior | Root independently read critical code; this reviewer checked synthesis, tests, plan and technical context |
| Coordinating agent | CLEAR, Poutine, RefineDrive; original DAgger source; synthesis, candidate mechanisms, scope/budget and document integration | All three reviewers reviewed before adoption; all gave final readback passes after their corrections |

The reading set contains **34 paper sources**: 13 AV + 17 foundations/methodology + 3 adjacent VLA + the original DAgger source used directly in the synthesis. NVIDIA repositories/blogs are additional technical sources. Selected primary full texts/sections and official proceedings were inspected as identified in the ledgers; not every sentence or theorem in every paper was independently re-derived. This is not an exhaustive survey, citation-count ranking, replication of paper results or external peer review. The agents shared the project context and were tasked with separate lanes and adversarial cross-review.

The detailed sources and locators are in [AV literature](av-literature.md), [foundations](rl-foundations.md), [NVIDIA stack](nvidia-stack.md), [adjacent studies](adjacent-vla-studies.md) and [the synthesis](report.md). Conference and preprint status, empirical domain and implementation differences are retained instead of ranking papers by an invented influence score.

## Root checks of consequential code and paper claims

In addition to reading each lane, the coordinating agent inspected these implementation points at the pinned revisions:

- Gym's [rollout filter](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/rollout_filter.py): freshness sorting, warning behavior and discarded-artifact cleanup. A configured zero lag is not a verified hard fence.
- Gym-pinned AlpaSim's [scenario evaluator](https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/scenario_evaluator.py), [collision scorer](https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/scorers/collision.py), [ground-truth scorer](https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/scorers/ground_truth.py) and [processing](https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/aggregation/processing.py): raw aggregation, MAX contact, LAST progress and offline masking/truncation/derived metrics. These are different definitions; a difference alone is not proof of a bug.
- Gym's [trainer](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/trainer.py): zero reduced gradient skips optimizer.step, but the surrounding step still advances the learning-rate scheduler. Full optimizer/trainer-state immutability would have been a false test requirement.
- The primary CLEAR, Poutine and RefineDrive method/evaluation sections linked in the adjacent-study note: residual-control compute, trajectory-only versus CoT optimization, and original-NAVTEST versus full two-stage evaluation, respectively.

No claim of the exact final Alpamayo 2 Super reward recipe follows from the combined-model RL statement, a historical paper, a local example configuration or the AlpaJudge release. The model family, trainable distribution, scoring interface and evidence domain must be named separately.

## Findings resolved before handoff

| Review finding | Resolution |
|---|---|
| Initial wording presupposed reward transfer failure | The primary question is conditional; either improvement, an explained negative result or insufficient evidence is possible |
| Mean-density versus joint-density distinction too vague | Main plan and technical reference identify the joint ratio within one replay row; mean log difference is clamped before exponentiation, then PPO clipping |
| Constant reward test assumed all optimizer state unchanged | E07 requires skipped optimizer update and unchanged parameters/per-parameter moments/counters only in the pinned KL-off/no-auxiliary-gradient case; scheduler/LR progression is explicitly allowed |
| All groups required uniform versions | E09 limits uniform matching versions to strict parity; later asynchronous batches need accurate per-record versions under their own qualified contract |
| Version knob treated as proof | E04/E05 require actual identity and replayed states/masks/noise; later version/advantage accounting remains separate |
| Dense metrics assumed present in episode artifacts | Raw metric capture before discard/aggregation is an explicit qualification requirement |
| Frozen parameters treated as unchanged runtime | Record buffers and train/eval modes as well as parameter hashes |
| Poutine split and CoT scope ambiguous | Record 416 training + 63 held out from 479 preference examples; trajectory RL omits CoT tasks |
| New roads treated as enough for algorithm confirmation | Require fresh training runs after development selection or a reviewed selection-aware design; price them in addition to exploratory pilots |
| Three training runs implied sufficient inference | Treat three as a diagnostic minimum; distinguish road and training-run uncertainty and do not promise calibrated intervals |
| S4 admission required measured training cost before any qualifying update | Separate priced/capped engineering qualification from measured-forecast admission of the replicated scientific study |
| Legacy 0.1 variability trigger lost | Preserve its stop/review status where the score applies; Huy defines the spread statistic or explicitly replaces it; remove automatic open-loop fallback |
| Cash and resource spending conflated | Track deposits, unused credit, accrued usage and liabilities separately; require both cash and resource headroom and stop before any approved limit |
| S2 runbook retained broad matrix and Kendall-W requirement | Two full teacher/re-export sweeps, 2 × 441 × 3 = 2,646 planned rollouts at minimum replication; policy-panel ranking deferred; scientific drafts remain unadmitted |
| Group normalization described as exactly scale invariant in actual code | Qualify by the epsilon convention; approximate invariance requires epsilon negligible against original and rescaled variation |
| Mastery requested interpretation before results | Preregister interpretation/decision criteria; interpret measured outcomes afterward |
| Old context could re-open a completed rental or imply public rights | Both contracts explicitly close D-019; dated role maps and public artifacts remain scoped by v5 and KC-L/KC-L2 |

Final exact-readback outcomes from all three reviewers were **PASS** with no residual issue in the reviewed passages. This is a documentation/source-review pass, not a simulation, optimization or statistical-power pass.

## Revision integrity and reproducibility

The exact pre-amendment plan is preserved in [the v4.1 archive](../../archive/02_PHYSLOOP_PLAN_v4.1_2026-11-01.md). Its SHA256 is:

```text
d30751e4467803c8fa0db0c55f830c8f03ed0cf1c9cd2ef5661f4f5d219204d4
```

D-021 is appended to the decision log; previous decisions and the technical reference's September predictions remain historical. Both agent contracts were updated together with only their existing tooling differences. File guards were read before mutations; protected inference/hook/settings files were not changed by this review. Existing unrelated working-tree changes were preserved. No commit was made.

Documentation verification checks whitespace (`git diff --check`), local Markdown-link destinations, the archive checksum, stable plan sections, matching substantive agent contracts, source-count arithmetic and the active two-sweep arithmetic. These are engineering consistency checks. They do not rerun or extend the prior S1/S2 CPU-test results. A generated check record and file hashes are stored in `evidence/rl-plan-review-20261101-v1/verification.json` relative to the project root.

## Remaining unverified evidence

The following are prerequisites for later conclusions, not open tasks silently claimed complete here:

1. Scientific S2 admission, actual teacher/renderer/Harmonizer fit and behavior, complete raw-file extraction, deterministic diagnostic qualification and real reliability/power/cost.
2. S4 two-GPU or alternative-topology fit, conversions, replay parity, version fence, frozen-state audit, actual gradients/updates, retry capture and independent reward/evaluation reconciliation.
3. A replicated learning effect, a selected mechanism and ablation, fresh training-procedure confirmation where required, and useful precision on held-out roads.
4. Concrete rental/funding/access approvals, publication rights and Huy's scientific definitions, predictions and interpretation.

This review incurred **no new cloud spending**, accepted no license, downloaded no gated asset and launched no GPU/model run. D-019/D-020's earlier $0.60 synthetic-host usage and teardown remain separate evidence. The $100 pilot / $1,200 S2 / $2,500 project caps and protected-core ownership remain unchanged.
