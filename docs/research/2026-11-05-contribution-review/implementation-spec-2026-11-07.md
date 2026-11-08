# PhysLoop v5: engineering implementation contract

Agent-authored, 2026-11-07. **Specification, not implemented software or stage admission.**

This refines current S2 preparation under D-017/D-018, mapped to S2.1 and S2.3–S2.7, serving G1/G2/G3·R1/G7. Conditional sections refine P-032; they do not adopt it. S4 interfaces are downstream requirements for S2 evidence, not permission to implement the trainer. Plan v5/D-022 and recorded decisions remain authoritative. The supplied notebook update is a presentation update, not a scientific registration; its rendered artifact was not inspected in this review.

## 1. Findings and implementation priorities

The research question is coherent: does expert-only RL improve a named held-out driving outcome, and what explains transfer or failure? The immediate engineering gap is the chain from the registered policy and population to complete, auditable evaluation inputs. A rising reward curve cannot repair missing identities, invalid outcomes or insufficient information.

| Priority | Evidence in this checkout | Required implementation consequence |
|---|---|---|
| P0 | Active plan is v5/D-022; P-032 remains proposed | Separate current S2 work, proposed protocol changes, and later-stage work |
| P0 | Schema-v2 preflight checks declarations; adapter trusts sidecars | Build actual-file extraction and evidence verification before calling the instrument qualified |
| P0 | Source audit finds six train/evaluation path differences | Identify the complete policy path, not just checkpoint bytes; measure differences before claiming their behavioral effect |
| P0 | Full scene count is constrained but population/road mapping remains unregistered | Audit actual road support and overlap before cost/power conclusions |
| P0 | Proposed universal road floor is mathematically incorrect | Use a zero-difference diagnostic, not an unconditional admission rejection |
| P1 | Dense metrics and failed attempts can disappear upstream | Capture before truncation/cleanup; preserve every attempted job |
| P1 | Actual RNG coupling and renderer repeatability remain unmeasured | Separate noise schedule identity, consumed random draws and output repeatability |
| P1 | Full matrix narrowed to teacher/re-export | Remove four-sweep assumptions from active planning output while preserving dated historical evidence |

Source basis: [active plan](../../02_PHYSLOOP_PLAN.md), [S2 contract](../../S2_ENGINEERING_SPEC.md), [existing implementation plan](../../S2_IMPLEMENTATION_PLAN.md), [P-032 specification](refactor-spec.md), [scientific audit](scientific-audit.md), and [RL protocol](../2026-11-01-rl-direction-review/protocol-design.md). Upstream statements here refer to those pinned audits, not a fresh November 7 release check. Recheck releases before runtime integration or upstream contribution.

## 2. Corrections required before coding P-032

### 2.1 Road support: distinguish a null-control threshold from universal impossibility

The locked `paired_road_bounds` splits positive and negative road differences. For every difference equal to zero and a single look:

`U0(R, alpha) = 1 - (alpha/2) ** (1/R)`.

GO requires strict `U0 < margin`; hence the minimum integer R is strictly greater than `log(alpha/2)/log(1-margin)`. At alpha 0.05 the zero-difference thresholds are 36, 51, 72 and 183 for margins 0.10, 0.07, 0.05 and 0.02. Use the actual per-look alpha when a sequential schedule applies.

These are **not** lower bounds for every possible observed comparison. The existing function on five road differences all equal to −1 gives upper = 0.04364750021196295, below margin 0.05. Strong improvements can pass below the zero-difference threshold. The earlier audit's “no data can GO” claim and proposed universal schema-v3 rejection must not be implemented.

Contract: emit `zero_difference_control_not_certifiable` as a design diagnostic, with margin, alpha, R and U0. It becomes a study blocker only when an explicitly registered acceptance rule requires that identical-output control to certify. It never creates independent roads or justifies merging strata. Real power still requires sensitivity analysis using the locked method and the registered sampling assumptions.

### 2.2 Seed uniqueness is not independence or reproducibility

The proposed 32-bit generated seed is unsuitable for a million-key uniqueness check: under uniform hashing, expected colliding pairs are about 116.415. With 64 bits the corresponding expectation is about 2.71e−8; it is small, not zero. A collision-free finite test proves only that tested schedule has no duplicate seeds. Neither hashing nor equal seeds proves statistical independence of roads, unchanged policy marginals, or deterministic GPU outputs.

Keep four assertions separate: unique schedule keys; no detected seed collision; expected random draws actually consumed; outputs reproducible under the named numerical/runtime configuration.

### 2.3 E25 is a limited sensitivity study

An image × CoC × integrator 2³ design does not isolate all six differences: RoPE may change with prompt handling; policy frequency and MPC operate outside a single inference call. It is an exploratory diagnostic, not a single-variable causal confirmation or proof of trained-path equivalence. Register its estimands, contrasts, interactions, sample count and decision rule before running. Do not infer “no effect” merely because a point estimate is below within-path variability.

### 2.4 Other contract corrections

- Include explicit RoPE/position-ID behavior in the path descriptor; `prompt_mode` does not fully specify it.
- E26 must have an exact compatibility target. “Bitwise or same distribution” is not one acceptance rule. With matched complete RNG states, the unchanged stock branch must preserve outputs in the declared environment; statistical equivalence needs a separately registered margin and design.
- E23 seed permutation is a diagnostic, not a guaranteed return to an independent-noise discordance rate. Pairing common scenes still induces shared structure; zero-event samples can also make the discordance ratio undefined.
- Replay the same policy random choices to diagnose residual execution noise. Fresh draws answer a different question. Stable same-seed rankings alone do not prove a useful learning signal.
- A render failure and a nonfinite pose can coexist. Store orthogonal fields, not a mutually exclusive `render_status` enum encoding both.
- Missing-data substitutions depend on metric direction. For a higher-is-better score, adverse completion is A=1/B=0; for a higher-is-worse incident metric it is A=0/B=1. Raw missing values remain missing.

## 3. Minimal architecture and ownership

Reuse `gate/s2.py`, `gate/s2_preflight.py`, `gate/alpasim_adapter.py`, `gate/contracts.py` and the locked inference implementation. Add small modules only for demonstrated missing boundaries. Proposed names below are targets, not existing APIs.

| Boundary / proposed home | Inputs → outputs | Invariant | Scope |
|---|---|---|---|
| `gate/s2_identity.py` | resolved config + source/checkpoint/asset receipts → canonical identity hashes | Changed bytes/settings change identity; unknown never becomes a placeholder hash | S2.1 preparation |
| `gate/s2_ledger.py` | planned jobs + launch/runtime events → reconciled jobs and attempts | No successful-only denominator; duplicate/conflicting events block admission | S2.3 preparation |
| `gate/s2_extract.py` | source-pinned raw files + launch binding → observation ledger + issues | Evidence-derived coverage/completion; no zero-fill or guessed IDs | S2.3 preparation |
| Existing adapter | writer summary + identities + qualified observations → normalized records | Entire batch passes or no admitted records | S2.3 |
| `gate/s2_diagnostics.py` | complete pairs + registered settings → reliability/validity/power inputs | Road, scene, replicate and training-run axes remain distinct | S2.4–S2.6; scientific settings required |
| `gate/s2_cost.py` | usage events + liabilities + next-unit forecast → headroom report | Paid startup/retry/idle included; no authorization boolean inferred | S2.7 |
| `gate/s2_rng.py` and upstream driver variant | frozen RNG contract → per-stream draw receipts | Explicitly qualified streams only; no hidden global reseed | Conditional P-032 |

Do not create a new training framework, simulator, scorer, service mesh or generalized workflow engine. The first usable vertical slice is a synthetic raw-log fixture reaching the existing adapter with exact attempt accounting. The next is an admitted actual-file fixture, not an RL run.

## 4. Identity and schema contract

### 4.1 Separate identities to avoid circular hashes

`protocol_hash` identifies shared observation/action/scoring/sampling semantics, excluding the A/B checkpoint identities and generated schedule. `arm_hash` identifies each policy/checkpoint and export lineage. `schedule_hash` identifies the ordered jobs and RNG assignments. A comparison binds all three; A and B must share the registered protocol but normally have different arm identities.

Canonical encoding v1: UTF-8 JSON, sorted keys, compact separators, reject NaN/Infinity, integers for ticks/steps/seeds, explicit units. Hash original config bytes separately from the resolved canonical representation. Record serialization version; do not claim semantic equivalence from differently normalized files. Full commit IDs, dirty source archive/diff hashes, lockfile hashes and image digests remain separate fields.

The complete path descriptor contains:

- model/config/processor/tokenizer revisions and checkpoint tensor manifest;
- input camera order, frame timestamps/history, resize H/W, interpolation/antialiasing, pixel normalization and camera geometry handling;
- prompt template/hash, prompt mode, navigation/CFG settings and token-sampling settings;
- position-ID/RoPE implementation and offsets, with input/output traces for qualification;
- expert integrator, steps/time grid/noise, action representation/units/frame and trajectory post-processing;
- requested and effective policy/control/camera rates, controller implementation/config;
- dtype, graph flag, batch configuration, numerical determinism settings and RNG implementation.

Hashing is evidence integrity, not proof that the receipt is truthful. Runtime capture must compare declared settings with effective settings and loaded tensors. Same tensor values across different serialization bytes require a documented canonical tensor manifest: name, shape, dtype and bytes; a file hash alone is insufficient to prove or disprove tensor equality.

### 4.2 Schema migration

Keep schema-v2 readers and existing fixtures. Introduce an explicit v3 reader only after the P-032 schema choices are recorded; never silently upgrade a frozen registration. A migration is a draft with `frozen=false`, preserving source v2 hash and unresolved owner fields. It cannot emit launch/publication approval.

New proposed fields: `policy_arms`, `driver_path`, `population_manifest_hash`, `scene_assets`, `road_identity_source`, `strata_road_counts`, `rng_contract`, `missing_data_rule`, and `information_boundary`. Their location and types are versioned; unknown required modes fail closed.

The existing `Protocol.seed_mode` supports `synthetic_seeded`, `unseeded`, `seeded`; do not inject `crn_seeded` into it. Represent coupling separately in v3 (`pairing=common_random_numbers|independent|none`) and map to `seeded` only after qualification. Preserve unseeded legacy behavior. Pilot remains 5 scenes × 2 Harmonizer settings × 1 replicate; full remains 2 arms × 441 scenes × ≥3 replicates. Changing those counts requires a scientific amendment.

### 4.3 Scene and population receipts

Each scene binds catalog revision, source capture/clip/time interval, registered road/stratum, asset SHA-256, renderer/NRE version and role (pilot, instrument, development or confirmation). Asset metadata may be prepared before downloads; asset hashing waits for authorized access. A repeated scene ID with different content is an incompatibility, not a new independent scene. Do not infer road independence from unique filenames. Unknown overlap is reported explicitly.

## 5. Complete attempt accounting and extraction

Predeclare logical job `(study, phase, arm, road, scene, replicate, protocol_hash)`. Bind an upstream session/run/rollout UUID from actual launch evidence; retain the logical job if UUID creation fails. Retries get new attempt IDs plus parent and reason. They do not overwrite earlier attempts.

State transitions: planned → started → terminal (completed, failed, cancelled or incomplete). An attempt may remain started after a crash; reconciliation reports it as unresolved/incomplete. Event IDs are idempotent only when bytes match; conflicting duplicates block. Write receipts before destructive upstream cleanup. Flush/close receipts and seal each immutable export with hashes; a crash-torn final event is an integrity issue, never silently ignored.

Per observation record:

- requested, expected and actual horizon with timestamp coverage and terminal reason;
- full pre-truncation metrics/trajectory references and hashes; observed metric names/coverage;
- deviation maximum/bins, lateral corridor, overshoot and map coverage separately;
- `render_coverage=complete|partial|missing`, `black_frame_count`, `first_black_timestamp`, `pose_nonfinite_count`, `first_nonfinite_timestamp` (unknown is null);
- infrastructure status, scorer status, required field issues and source parser version;
- selection-for-analysis record under a frozen retry rule; no best-of-retry selection.

Do not attribute a black frame to the policy or renderer without additional evidence. Record observations before causal labels. Report the A-valid/B-valid 2×2 table using the registered validity definition and retain failure categories/denominators. ≥4 m observations remain visible; selecting only low-deviation outcomes would change the compared population.

A source-specific parser must verify timestamps, monotonicity, duplicate samples, coordinate frame and expected stream frequency. A legitimate recording-end clip requires registered terminal semantics and an independently tested expected-end calculation. The current exact-horizon adapter is not relaxed by this specification. Unsupported terminal cases remain blocked until that amendment is recorded.

Missing-data sensitivity is a separate derived artifact with explicit substitutions and source references, never fabricated observed `RolloutRecord`s. It cannot rescue broken provenance, unknown population or infrastructure-corrupted observation semantics. The official gate remains blocked under the current contract until Huy registers an applicable rule. Do not silently reconcile the older proposals' differing “both bounds agree” and “adverse bound passes” rules.

## 6. Conditional RNG and policy-path qualification

Implement only after the applicable P-032 choices/admission. Default stock driver stays on its unchanged branch; variant flags select a separately identified protocol.

Proposed deterministic key encoding:

`(rng_version, master_seed, phase, road, scene_asset_hash, replicate, simulation_tick, sample_index, stream_name, substep_index)`.

Use a named cryptographic hash and explicit byte order/truncation to a supported seed width; freeze that algorithm with golden vectors. Exclude arm for qualified common streams; include arm for independent streams. Distinguish `None` from integer zero. Domain-separate initial expert noise, SDE increments and token sampling. Keep the schedule identity outside its own seed derivation to avoid circularity. Detect collisions on the actual schedule and reject/revise before freezing; no outcome-informed rehashing.

Use explicit generators or injected noise tensors at every relevant stochastic operation. A per-session generator alone is insufficient when prompt lengths, branching, retry order or batch composition change random-draw consumption. Key expert draws to semantic indices rather than global call order. Record which streams are controlled and which renderer/Harmonizer/traffic streams remain uncontrolled. Different policy frequencies and different integrator grids do not have an automatically meaningful one-to-one noise coupling; define such coupling separately or label that contrast unpaired.

Qualification ladder:

1. CPU golden seed vectors, zero seed, reordered/interleaved sessions, domain separation and forced hash-collision rejection.
2. Fake model trace proves `traj_future` never invokes token generation and passes exact position IDs/masks; synthetic image fixture checks resize and processor geometry behavior.
3. Admitted fixed-input GPU replay compares raw noise, processed inputs, intermediate traces and action outputs in physical units; then batch/interleaving effects. Equal random tensors do not guarantee bitwise equal GEMMs.
4. E26 unchanged-stock replay with matched full RNG state and effective configuration; no distribution-only escape from a failed exact compatibility claim.
5. E27 Gym-path replay matches preprocessing, prompt/RoPE, action frames, integrator and sampled noise; tolerance comes from Huy's numerical ladder. One tick does not qualify controllers or closed-loop execution.
6. E22/E23 closed-loop diagnostics report first divergence and uncertainty; actual CRN benefit is measured separately for re-export, early RL snapshot and final checkpoint. A zero denominator yields `not_estimable`, not f=0.

E25 may follow as a separately registered path-sensitivity experiment. Frequency/controller effects require time-aligned control traces and closed-loop contrasts; they are not extra factors silently added to a single-tick experiment. A driver variant is optional if the deployed-path estimand is chosen and its limitations are explicitly accepted; do not make it a universal prerequisite by assertion.

## 7. Reliability, power and resource interfaces

Reliability output must include counts at each hierarchy level, paired differences, per-stratum road support, same-policy variation and absolute uncertainty. Huy freezes the repeatability/ICC/event definition and acceptance floor before the corresponding analysis. Do not reuse policy-panel Kendall W without a meaningful panel.

Power uses the locked `paired_road_bounds` and actual verdict aggregation, including all required strata and any registered alpha-spending schedule. Report Monte Carlo trial count, RNG, Monte Carlo uncertainty and assumed effect/discordance/road dependence. Same-policy variance does not identify post-training variance. Do not transplant the illustrative 50%→100% table into a forecast. More simulator replicates do not add roads; multiple training runs require a separate inference contract.

Cost output separates completed useful episodes, attempts, retries, paid startup/idle/warm-up/tail, storage/transfer/fees and unresolved liabilities. Count GPU-hours as device count × billed hours, not node-hours. Reserve confirmation and teardown before another work unit. New quotes and approvals are needed at admission; this document makes no November 7 price claim. Reuse the existing exact-ID watchdog and cleanup checks rather than creating another provisioning framework.

## 8. Test-first work packages

All rows below are acceptance specifications, not new test passes. Tests use synthetic/public-source fixtures until actual assets/runs are authorized. Evaluator cores require an independent reviewer before qualification; this document has not received that review.

| Order | Work package / dependency | Failing test to write first | Completion evidence |
|---|---|---|---|
| W0 | Current v5 scope reconciliation | Active cost/matrix output has 2 arms and 2,646 full jobs at 3 repeats; historical 4-sweep table is explicitly historical | Existing tests pass; drafts remain blocked, no invented registrations |
| W1 | Identity/population receipts | Same scene label with changed content rejects; modified RoPE/config changes path hash; unsupported source fails | Canonical encoding golden vectors and trace to source bytes |
| W2 | Attempt ledger, after W1 | Crash before UUID, duplicate event, retry after failure, conflicting terminal record | Every expected job reconciles; all attempts retained; no success-only join |
| W3 | Raw extraction, after W2 | Missing offroad field; black frame before contact; NaN pose; truncated metrics; legitimate vs unexplained terminal | Exact synthetic known answers through existing adapter; later actual-file qualification separate |
| W4 | Diagnostic/power harness | Zero-difference thresholds and five-road improvement counterexample; missing stratum; zero discordance denominator | Existing inference unchanged; scenarios explicitly labelled; no universal road-floor blocker |
| W5 | Cost/evidence bundle | Boundary at cap/deadline, missing charges, partial export, wrong resource ID | Refuse next unit; preserve/export evidence; no inferred cloud authority |
| C1 | Conditional v3 schema/RNG | Zero seed, reordered sessions, overlapping domains, forced collision, missing RNG consumer | Requires recorded P-032 choices; legacy mode unchanged |
| C2 | Conditional driver variant | E26/E27, prompt-generation spy, RoPE trace, batch trace | CPU structure first; admitted GPU numerical and closed-loop evidence later |
| C3 | S2 scientific pilot/full | Registered protocol and actual runtime gates | Measured acceptance, power and all-in forecast or INSUFFICIENT |

W0–W5 are sequenced engineering preparation within S2; unresolved scientific conventions remain blockers for dependent real-data analysis. This request delivers their specification, not their implementation. C1/C2 must not quietly become prerequisites for all v5 work. No S4 implementation is opened.

Suggested future test homes: `tests/test_s2_identity.py`, `tests/test_s2_ledger.py`, `tests/test_s2_extract.py`, `tests/test_s2_diagnostics.py`, `tests/test_s2_cost.py`, and conditional `tests/test_s2_rng.py`. Test adapters with independent literal oracles, not outputs calculated by the implementation being tested.

## 9. S4 handoff requirements, not a build authorization

The S2 bundle must let a later S4 checkpoint reuse the exact evaluation protocol, source assets, road population, required controls and information boundary. Reuse is invalid if those identities change; perform a new qualification or registered bridge study.

The later RL ledger needs group/generation identity, configured and actual behavior/current weight versions, retained zero-update checkpoint, SDE samples/masks, raw reward vector, advantages, replay rows, optimizer-step counts/skips, scheduler changes, conversion lineage and complete failed/dropped attempts. Capture dense metrics before upstream discards them. Enforce E01–E21 from the existing RL protocol after S4 admission.

Strict initial parity occurs before the first optimizer update, with real weight/sample identity, not merely `allowed_outdated_steps=0`. A KL-disabled log is labelled `not_computed`; offline drift probes use the zero-update checkpoint. Same-seed replay characterizes residual noise; an estimator/reward/anchor change remains Huy's research design. Preserve checkpoint-level versus method-level claims and reserve fresh training confirmation when selection requires it.

## 10. Decisions and completion boundary

No decision is demanded just to review or implement independent S2 preparation. Before dependent execution, Huy records population/road strata, estimand/path, margins/reliability floor, terminal/missing-data conventions, information boundary and scientific predictions. P-032 implementation additionally needs its path/seed/schema choices; scientific pilot still needs rights/access and concrete rental authority. S4 has its separate engineering and scientific admissions.

Do not request hashes, image digests or source facts from Huy: agents collect them. Do not fill his scientific values with convenient defaults. No protected files, hooks, settings or evidence archives are modified. No commit, download, rental, publication or external message is implied.

Review baseline: repository HEAD `3741d9233a0c01a27d3f02306311bb008a8ae3b6` with pre-existing uncommitted November 5 review files and decision entry. November 7 validation: read-only CPU calls to the locked gate reproduced the four zero-difference counts and the five-road counterexample; `2*441*3 = 2646`; seed collision arithmetic reproduced. These are deterministic engineering checks, not driving outcomes or a full-suite rerun.

Reproduce the key counterexample from repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -c 'from gate.verdict import paired_road_bounds; print(paired_road_bounds([-1.0]*5))'
```

Expected output: `(-1.0, 0.04364750021196295)`. Before implementation, add this and the zero-difference cases as regression tests in an unlocked test file; import the protected implementation read-only.
