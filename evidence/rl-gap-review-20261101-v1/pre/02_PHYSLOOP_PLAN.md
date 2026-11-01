# PhysLoop plan v5

**Authoritative, adopted 2026-11-01 under D-021.** Huy authorized the verified RL-first plan and context update after independent review. This replaces the active scope of v4.1, preserved in [the v4.1 archive](archive/02_PHYSLOOP_PLAN_v4.1_2026-11-01.md). Scientific definitions, numeric predictions and all execution gates remain separate from this planning revision.

## 0. Direction and authority

PhysLoop is an independent research project on **closed-loop post-training of driving action policies**, initially NVIDIA's Alpamayo 1.5 action expert in AlpaGym. The release gate remains the measuring instrument. The first focused path is **S1 → S2 → S4 → research conclusion**. G9 is the primary research deliverable.

The user requested independent multi-agent research using legitimate papers and official technical resources before updating the plan, then explicitly approved updating the main plan and context after verification. The evidence is [the November 1 research review](research/2026-11-01-rl-direction-review/report.md), including AV precedents, RL foundations, pinned source audit, engineering specifications and independent review resolutions. P-031 supplied the initial proposal; this revision incorporates corrections found during the deeper audit.

### What changes from v4.1

| Area | Active v5 commitment | Deferred from the first focused study |
|---|---|---|
| Research question | Whether and why expert-only RL changes independently measured driving outcomes | Student runtime-reasoning curve as the headline |
| S2 policy matrix | Frozen teacher, same-policy controls, required integration/reliability/validity checks | Full FP8 and TransFuser comparative sweeps; policy-panel ranking claims |
| S4 baseline | One shipped reward arm; at least three independent training runs if measured cost supports them | Broad hyperparameter search and unconditional second/third arms |
| S4 controlled study | One mechanism-driven intervention after baseline diagnosis and a funded protocol | Several simultaneous reward, exploration, horizon and environment changes |
| Systems work | Correctness, complete accounting, sufficient profiling and cost control on chosen topology | Mandatory 1→8-GPU scaling/model-fitting study and multi-node work |
| Other stages | Stage identifiers retained as future options | S3, S5, S6, S7 and S8 implementation; S8 was already optional |
| Claims | Scoped checkpoint evidence, or a separately qualified method-level result | Unconditional “first,” efficiency superiority, real-world or deployment claims |

This is a scope change, not an inference-method change. D-016's conservative gate, D-018's ownership/caps, protected files and Huy's retained scientific decisions remain. Existing S2 full-protocol drafts are not automatically a valid registration for the narrowed matrix. Revise and verify them before execution; do not weaken their admission checks to pass.

No new stage or general RL platform is added. Use NVIDIA's public stack where it supports the experiment. Configuration-only comparisons are preferred. A small evaluator term or capture adapter may be proposed when a demonstrated requirement is not expressible by configuration; qualify it independently. Model-loss, estimator or runtime-framework changes require a specific research design and scope decision, not an unrestricted fork.

### Current status

S1 is technically qualified. S2 free preparation and one bounded synthetic Vast host check are completed; actual scientific S2 and S4 remain unadmitted. The Vast allocation was destroyed and closed under D-020, with $0.60 recorded usage from existing credit. It qualifies only the tested host capabilities, not the AV stack, two-GPU training, licensed hosting or learning. The earliest scientific S2 admission remains November 2, after the separate registration and action approvals.

The $100 pilot, $1,200 S2 and $2,500 project cash ceilings are nested, all-in limits. They neither release funds nor authorize rental, funding, license acceptance or asset download. No future-stage implementation is opened by adopting this plan.

## 1. Research objective

**Primary question:** under one pinned reconstructed-scene protocol and compute envelope, when does the shipped continuous-expert update improve held-out driving outcomes, and when transfer fails, does correcting one demonstrated mechanism change it?

The initial path updates the continuous action expert with the visual/language backbone frozen. It does not claim language-reasoning training. The open-loop VLM/discrete-action recipe, learned-judge rewards and joint reasoner/expert training are separate future studies. Do not switch to Alpamayo 2 Super during an experiment because it is newer; doing so changes the model and resource problem.

**Contribution hierarchy:** first establish a reproducible baseline; then identify a concrete failure; then test one correction with an explicit falsifier. A controlled gain, a well-supported negative result and INSUFFICIENT are all valid outcomes. Installing AlpaGym or showing a rising reward curve is engineering progress, not the final research contribution.

**Competing explanation:** policy-visited-state correction or a privileged teacher may improve behavior more cheaply than direct sensor-policy RL. The AV literature includes these approaches. Our initial choice is motivated by the learning mechanism Huy wants to study. A comparative efficiency claim requires a matched alternative and complete accounting, not a citation to another paper's different experiment.

**Preserved owner target:** the earlier “at-fault collisions down at least 20% relative, with no regression in progress/off-road” remains a research target to restate on precise registered endpoints. It is not a promised effect, a fixed new prediction or automatically decidable on 441 scenes. Front/lateral contact is a simulator geometry proxy, not a legal fault determination. Huy owns its final scientific definition.

## 2. Deliverables

| ID | Deliverable | Active role and acceptance |
|---|---|---|
| G1 | Paired, stratified, road-adjusted gate | S1 software qualified; S2 must qualify real inputs, noise and operational power. GO/NO-GO/INSUFFICIENT retained |
| G2 | Validity and completeness records | Every attempted rollout accounted for; ≥4 m, render/map/infrastructure issues and terminal truncation visible |
| G3 | Instrument evidence | Active R1 reliability; other construct-validity legs remain gated/deferred. No predictive-validity claim |
| G4 | Guardrail | Deferred |
| G5 | Student verdict, correction and reasoning-budget curve | Deferred until a separately adopted extension after the RL conclusion |
| G6 | Jetson AGX Thor serving | Deferred; hardware purchase gate unchanged |
| G7 | Measurement and reproducibility | Active for S2/S4: throughput, useful versus paid work, lineage, failures, resource costs. Broad scaling/HIL reports deferred |
| G8 | Shadow-mode data work | Deferred and optional; no actuation path or rig authorization |
| G9 | Focused teacher RL study | Primary: baseline correctness, replicated learning evidence, one conditional intervention, held-out evaluation and mechanism report |

Reuse upstream scorers, simulator, training presets, conversion scripts and telemetry. PhysLoop adds the qualified evaluation/admission layer, missing lineage and diagnostics, specific tests, and the experimental comparison. Do not rebuild NVIDIA's trainer to create apparent infrastructure novelty.

## 3. Stage cards

Stage numbers remain stable. `[C]` is required for the active focused path. `[O]` is optional only after measured hours/cost justify it. `[G]` needs a recorded scientific/scope/action decision. An optional item never displaces required controls or independent evaluation. Weekly hours (P-003) remain unrecorded; old calendar windows are historical, not a new delivery promise.

### S1 — Qualified gate core on synthetic truth

**Serves G1.** The qualified implementation, conditional proof and reproduction commands are in [S1_IMPLEMENTATION](S1_IMPLEMENTATION.md), [S1_CORRECTNESS](S1_CORRECTNESS.md) and [S1_HANDOFF](S1_HANDOFF.md), under D-016.

The operational method is bounded-road conservative authorization with an equal-road estimand and finite-look alpha spending. Tango/road-bootstrap and canonical Gaussian methods remain labelled reference/planning paths; they do not authorize this v0. INSUFFICIENT blocks GO. Historical known answers, exact small-support checks, Monte Carlo and source-archive reproduction remain evidence, not unconditional correctness.

**Critical limitation:** noisy equivalent controls returned INSUFFICIENT in almost all registered synthetic trials. S2 must measure real variation and road support, and simulate power using the operational method before pricing an RL conclusion. A more convenient approximation does not bypass this limitation. Protected `gate/verdict.py` and `gate/stats/` remain read/test/review-only until Huy unlocks them.

No S1 rerun is required merely because the project focus changes. An inference-method amendment, if needed, is separately derived, reviewed, authorized and qualified.

### S2 — Qualify the instrument for an RL checkpoint comparison

**Serves G1, G2, G3·R1 and G7. Current stage; scientific admission pending.** This card replaces v4.1's broad mandatory policy sweep with the measurement needed for the focused study. [S2 engineering detail](S2_ENGINEERING_SPEC.md), [implementation plan](S2_IMPLEMENTATION_PLAN.md) and [rental packet](S2_RENTAL_APPROVAL.md) remain subordinate; stale full-sweep templates must be amended before use.

**[C] S2.1 Rights, revisions and registration.** Re-check NVIDIA's AlpaSim release/branch state at admission. Record model/data/renderer revisions, terms/access, hosting eligibility and first-download clock. Until a suitable sync is qualified, the inspected `e2e_challenge` protocol remains the candidate evaluation branch and results are labelled provisional where withheld settings matter. Huy registers target population, road/scene/stratum rules, metrics, margins, validity/terminal semantics, seed mode, Harmonizer, control horizon/frequency, run method/topology, reliability floors and scientific predictions.

The existing full preflight requires 441 scenes, and the five-scene pilot is unchanged. The review does not authorize a smaller convenient sample, a new stratum definition or a replacement estimator. If a road/source audit makes that sampling contract infeasible, stop for a documented scientific amendment. Templates are neither approval records nor launch permits.

**[C] S2.2 Bounded cost/protocol pilot.** Register the five-scene, one-replicate Harmonizer off/on contrast before running it. Measure the actual chosen guest, teacher/renderer/Harmonizer memory fit, full wall-clock cost, validity and failures. Pilot outcomes are excluded from full inference. Pilot scores cannot choose favorable scenes, margins or strata. Outcome-informed redesign is exploratory and needs an independent confirmation plan.

**[C] S2.3 Complete extraction and known answers.** Normalize raw outputs with exact identity, schedule, retry and completion checks. Preserve full trajectories and original metric arrays; summary scores alone cannot prove validity or completeness. Test stationary-ground-truth terminal behavior using the public pre-fix/fixed helper, validate the actual evaluator path, and distinguish a clipped available recording from an unexplained incomplete run. Do not fill absent offroad data, failed attempts or missing telemetry with zero.

**[C] S2.4 Frozen-teacher reliability and controls.** The full focused matrix contains the BF16 teacher and its bit-identical re-export control, with at least three simulator replicates under the registered schedule. Preserve same-policy repeatability probes, fixed-input policy probes and fixed-pose renderer probes. Call repetitions seeded only if the actual path consumes and preserves the seed. A deterministic diagnostic port, if needed, is separately qualified; adding a seed label does not make the stock challenge driver deterministic.

Full TransFuser/FP8 comparative sweeps and policy-panel ranking are deferred. A small upstream baseline may still be used when needed to diagnose an integration failure, but it is not a new comparative scientific sweep. Without a meaningful policy panel, do not claim Kendall-W panel ranking reliability. Freeze the suitable repeatability/ICC/event-noise analysis with Huy and report its assumptions and uncertainty.

**[C] S2.5 Validity and contamination.** Every rollout carries deviation bins `<1`, `1–2`, `2–4`, `≥4 m`, lateral corridor status, endpoint-overshoot/terminal status, render failures and map coverage where applicable. Join quality/time/speed metadata as diagnostics. Audit `alpamayo_aligned` split labels and source-road/clip/temporal overlap; presence in the parquet alone is not exclusion because it labels the entire dataset. Preserve all attempted scenes and failure denominators.

**[C] S2.6 Operational power and confirmation design.** Estimate same-policy variation and within-scene dependence; record actual independent road support. Compute sensitivity over plausible RL effects, discordance and per-stratum allocations with the existing operational method. Same-policy noise does not determine all post-training paired variances, so retain sensitivity ranges. Separate instrument estimation from confirmation-scene failure mining. Before S4, show what the available design can support and what would remain INSUFFICIENT.

**[C] S2.7 Cost and necessary telemetry.** Measure stabilized complete-rollout throughput, startup/idle/retry/tail losses, memory and the useful/paid time breakdown. Use the chosen upstream stabilization procedure and Huy's registered criterion; include paid warm-up. Device timings require warm-up and synchronization. Retain DCGM/Nsight/Prometheus evidence needed for diagnosis, actual GPU edition/driver and explicit missing-monitor statuses. The prior host's XID delivery gap still needs qualification or an explicit acceptance amendment.

NVIDIA's expert CUDA graph can be enabled only after bitwise parity under the declared seeded diagnostic and unchanged controls. Otherwise leave it off. Do not use FP8 or changed batching as an unqualified sweep speed-up. Detailed profiling serves a measured bottleneck, not a separate mandatory performance campaign.

**[O]** One-node Slurm/requeue, bounded failure injection, a service-balance probe and extended profiling may run only if they solve a current requirement and leave the evidence budget intact. **[G]** Reactive CAT-K (P-013), renderer-agreement expansion (P-014) and batching (P-015/P-022) remain separate. No 1→8-GPU scaling program is required for this focused path.

**Acceptance:** complete manifests and raw evidence; qualified extraction and negative controls; registered reliability floor satisfied; validity/contamination reported; operational power and all-in forecast sufficient for the selected next question, or an explicit INSUFFICIENT/stop result. Card-score matching is diagnostic only. A passing host smoke does not satisfy this acceptance.

**Huy's unresolved scientific inputs:** road population/strata; score and contact/validity/terminal conventions; margins and reliability floor; seed/Harmonizer/protocol choices; noise and cost predictions; information boundary between S2 and confirmation. Agents supply hashes, source audits, quotes and engineering evidence, not invented scientific answers.

**Kill KC-2:** unresolved extraction/control defects, reliability below the registered floor or inadequate power for the intended claim stop progression to a confirmatory RL study. Keep useful engineering results. KC-L blocks public licensed-data-derived numbers; KC-L2 separately blocks derived weights/labels.

### S4 — Focused continuous-expert RL study

**Serves G9 and G7. Planned; not admitted.** S2 is a prerequisite. The detailed [engineering/experiment specification](research/2026-11-01-rl-direction-review/protocol-design.md) supplies the required interfaces and E01–E18 checks; [the NVIDIA source audit](research/2026-11-01-rl-direction-review/nvidia-stack.md) identifies exact dependencies and limits.

**[C] S4.1 Freeze the implemented learning problem.** Start from one released Alpamayo 1.5 checkpoint and the inspected AlpaGym path. Pin the exact recipes PR dependency, Cosmos-RL and training AlpaSim; main branches are not interchangeable dependencies. Record trainable parameters, buffers/modes, action representation, conditioning inputs, SDE/noise settings, reward components, aggregation, group normalization, clipping, KL/reference settings and replay weighting.

The language/visual weights stay frozen as shipped; no reasoning-text optimization claim follows. At least two candidates are required for a nontrivial group-relative reward signal, but two candidates do not ensure reward diversity. The documented smoke uses two candidates and one update, and is not a learning study.

**[C] S4.2 Establish correctness before learning claims.** Qualify zero-update conversion/export parity and actual weight/load identities. Derive and independently test the implemented likelihood surrogate and KL. The geometric-mean ratio differs from the joint density ratio of the denoising/action-coordinate factors within a replay row, as well as from a whole-episode ratio. For strict initial parity, match actual behavior/current weight versions, recorded SDE samples, masks and noise convention, and use one replay iteration. `allowed_outdated_steps=0` alone is insufficient because the local filter warns rather than rejects stale kept samples. Later asynchronous operation needs its own version/advantage accounting.

Test equal-reward and reward-sign controls, finite valid gradients, exact frozen parameter hashes, declared buffer/mode behavior and intended trainable changes. On the pinned zero-gradient/KL-off stock path, the optimizer must be skipped. A one-update pass establishes execution correctness under that case, not behavioral improvement.

**[C] S4.3 Reconcile reward and evaluation semantics.** Trace raw event arrays into online reward and offline gate outputs. Online raw aggregation, warm-up, first-incident truncation, invalid-frame handling and derived contact metrics differ. Standard Gym artifacts leave dense metrics empty, so a qualified capture must preserve the needed per-timestep evidence before it is discarded. Capture all attempted groups, retries and exhausted/dropped payloads before upstream cleanup. Missing hard scenes cannot silently improve the training distribution or the reported score.

Retain the shipped `progress_safety` objective as baseline A. A front-or-lateral alternative, if chosen by Huy, must preserve time support and episode-any semantics; summing two indicators can double a penalty. A cap/OR is not currently a free configuration-only substitution. Do not adopt an arm until its exact evaluator term, interpretation and implementation scope are approved and tested. Keep all/rear contacts, progress, offroad, stalls, oscillation, comfort diagnostics and validity visible independently of the scalar reward.

**[C] S4.4 Measure a baseline pilot.** Run one shipped reward arm with at least three independent training runs only once the measured topology and complete evaluation allocation fit. Freeze the checkpoint-selection rule and use development data for diagnosis. Keep each run, failure, learning curve, parameter lineage and actual seed/replicate mode. Report action/reward diversity, all-tied groups, useful gradients, clipping/KL, actual rollout lag and paid useful-episode throughput. Measure full setup/conversion/export overhead as well as steady state.

A one-run engineering smoke can precede this pilot under a separately bounded admission. It does not replace the three-run baseline or support a training-method conclusion. If the minimum pilot cannot fit, stop at an engineering/feasibility result instead of silently reducing the claim's evidence.

**[G] S4.5 One controlled intervention.** After development evidence identifies a mechanism, Huy derives and registers one change, numeric prediction, falsifier, budget comparison and stopping rule. Candidate mechanisms are reward/evaluation mismatch, insufficient within-group signal, horizon mismatch or coarse temporal credit. A credit-estimator change is a larger research modification; it is not smuggled into a reward-only contrast.

Plan at least three independent training runs per arm. Three is a minimum diagnostic count, not proof of adequate uncertainty estimation. For a training-procedure claim after intervention selection, use fresh independent training runs or an explicitly reviewed selection-aware design. Reused exploratory pilot checkpoints can support a conditional checkpoint contrast, but new roads alone do not remove selection on their training randomness. Price exploratory and fresh confirmation runs separately; six study runs may be additional to the baseline pilot. Compare equal paid compute, equal simulated interaction or equal updates explicitly. Do not vary reward, noise, group size, horizon, precision, traffic and controller together. Protect a fixed independent confirmation budget before launching.

**[C] S4.6 Evaluate the locked checkpoint choices.** Export using the qualified path and evaluate against the frozen teacher under the S2 registration. Pair scenes and declared replicate schedules; adjust for roads and report each stratum and ≥4 m outcome separately. Report training-run variation separately from road/scene/replicate uncertainty. The checkpoint gate does not automatically establish an algorithm-level conclusion or adjust for best-of-many checkpoint selection. A method-level claim needs an independently reviewed run-level/crossed inference contract; otherwise report scoped checkpoint results.

**[C] S4.7 Research report and stop.** Deliver the falsifiable question, source/config/data lineage, measured cost, all training runs, selection rule, outcome vector, uncertainty, validity, failed controls, ablation and prediction-versus-measurement reconciliation. Huy owns scientific interpretation. The result can be improvement, regression, a negative mechanism result or INSUFFICIENT. Public artifacts remain subject to KC-L/KC-L2.

**Preserved predictions awaiting Huy:** relative collision change `____`; per-run variation `____`; mechanism-specific effect `____`; progress/offroad margins `____`; first hacking mode `____`; useful-group fraction `____`; parity tolerance and downstream justification `____`; stopping rule `____`; compute forecast error bound `____`. Historical v4.1/technical-reference predictions remain historical until explicitly registered for this study, not silently rewritten by agents.

**Kill KC-4:** parity/integrity failures block training; an unsupported learning signal, invalid-state exploit, excessive run variability or insufficient forecast blocks the controlled study until diagnosed. Preserve the legacy trigger, “closed-loop score varies by more than 0.1 across training seeds after 3 training seeds,” as a stop/review condition when the same score applies. The old wording does not define the spread statistic: Huy must fix that definition before scientific training, or explicitly replace the trigger if the score changes. A predicted SD is not a stopping rule. The former automatic switch to open-loop GRPO is removed: that changes model path, objective, hardware and research question. A fallback requires a separate reviewed proposal. Stop immediately at the approved cost/deadline boundary, export evidence and tear down; stage completion is never permission to exceed a cap.

**[O]** A small diagnostic profile or reproducibility/resume drill may address an observed bottleneck. **[G]** Reasoning RL, constrained objectives, reactive traffic, privileged-teacher comparisons, larger search, new model family and scaling beyond the chosen node require separate evidence and funding. None is required to call the initial focused study complete.

### Deferred stages S3, S5, S6, S7 and S8

The full prior cards, deliverables and owner targets remain in the archived v4.1 plan. They are not active implementation instructions. S3 student comparison and S5 correction may follow an RL result if a qualified student and resources exist. S6/S7 still require the Jetson decision and device-specific evidence. S8 remains optional, separately gated, with no actuation path. No deferred stage inherits a calendar or budget from this revision.

Re-entry requires a decision specifying the new question, prerequisite evidence, asset rights, protocol, cost and ownership. Edge availability no longer blocks the active S2/S4 path. Re-check its exact artifact/license at any future S3 admission.

## 4. Immediate work and research learning

The next execution is S2 qualification, not a training launch. Complete the source/scene/rights audit, revise the blocked full draft for this plan, obtain Huy's scientific registration and the concrete rental/license approvals, then run the bounded pilot. Use its actual measurements to release or reject the remaining S2 study. No automatic carry-over of D-019 rental authority remains.

Huy's learning order is: state-distribution shift and privileged information; PPO/GAE; group-relative advantages; diffusion/flow sampling and its two time axes; AV teacher/direct-RL precedents; the actual Gym estimator; reward exploitation; independent statistical evaluation. Specific papers and exercises are in [RL foundations](research/2026-11-01-rl-direction-review/rl-foundations.md) and [AV literature](research/2026-11-01-rl-direction-review/av-literature.md). `03_MASTERY_PLAN.md` owns mastery requirements; agent-created artifacts are not Huy's mastery evidence.

The recommended personal allocation after S2 remains 70% model hypotheses/design, 20% ablation/interpretation and 10% critical measurement review. This does not set P-003 weekly hours. Agents own evaluator engineering and operations within admitted stages; Huy retains scientific definitions, predictions, reward/loss/estimator research and conclusions.

## 5. Budget and resource decisions

**Adopted caps:** $100 pilot within $1,200 S2 within $2,500 project cloud, all-in. Hardware is separate and deferred. Recorded $0.60 host usage belongs inside S2; it is not subtracted twice. Reconcile deposits, accrued usage, unused credit and outstanding liability before each packet.

`remaining = 2500 − recorded project usage − remaining liabilities/reservations`, with a separately reconciled cash ledger so prepaid deposits are not double-counted as consumption. Each packet must satisfy both resource-consumption/liability headroom and the cumulative approved cash-outlay ceiling; unused credit is tracked separately. Available credit is not unlimited spending authorization. Every rental, account funding, license acceptance and asset transfer needs its applicable concrete approval.

At maximum S2 spending, only $1,300 remains. A provisional reservation of $650 training/qualification, $400 independent evaluation and $250 ancillary/failure costs illustrates that constraint. It is not a measured quote, released funding or a promise that six runs fit. Historical v4.1 estimates put full S4 at $807–2,628; they are archived context, not today's offer or the narrowed study's forecast.

The inspected Gym guide recommends two GPUs of at least 40 GB each for its documented route; actual 10B fit, interconnect, optimizer/rollout memory and renderer contention remain unmeasured here. The earlier 1×96 GB synthetic host check does not resolve that. The open-loop reasoning recipe's separate five-80-GB setup is not an automatic fallback. Re-price the exact topology at its stage admission.

Forecast provisioning, startup, conversion, training, development selection, confirmation, retries, idle time, retained storage, transfer, tax/fees and teardown. Reserve independent evaluation first. If the intended design does not fit, finish a smaller clearly scoped feasibility/baseline result or propose a separately priced expansion. Never reduce controls, discard failed seeds or mine confirmation to manufacture an affordable success.

**Hard cost rule:** do not start a unit whose forecast plus teardown/reserve exceeds the remaining approved envelope. Stop before its absolute deadline/cap even if the stage is unfinished. There is no $4,000 trigger under the adopted $2,500 cap; the old larger-tier rule is superseded. No automatic credit top-up, retry rental, storage retention or extension.

## 6. Allowed claims

Every empirical claim names the model/checkpoint and source identities, target population, road/scene counts, training runs, simulator replicates and actual seed mode, config, lockfile, renderer/traffic/controller, requested/actual horizon, selection rule, endpoint, deviation convention, uncertainty method and device/topology for performance. Uncommitted source requires archived diff/source hashes in addition to git SHA.

**Allowed when supported:**

- Checkpoint B is non-inferior to A within the registered margin on every pre-specified stratum, paired and road-adjusted, with the one-sided bound and stopping rule named; or NO-GO/INSUFFICIENT.
- RL changed a precisely defined simulator incident endpoint by a measured amount under protocol P; progress, offroad, stalls, validity and infrastructure outcomes reported alongside it. A superiority claim needs evidence for improvement, not merely NI GO.
- A specified intervention changed behavior under a controlled contrast, with the scope of the causal attribution and training-run evidence stated.
- A mechanism failed or a comparison remained INSUFFICIENT under the registered design.
- On the named device/node, the qualified loop achieved a measured useful-episode rate and all-in cost. Comparative efficiency requires a matched baseline and complete accounting.
- Software passed specified known-answer, negative-control and reproduction checks under stated assumptions. Source inspection is labelled source inspection, not runtime proof.

**Not allowed:** “safe,” real-world validation, per-mile crash claims, human parity/above-human claims, predictive validity, regulatory/standards compliance, or deployment assurance from these simulations. Replay-only results are not evidence of social interaction with reactive traffic. Judge scores are not collision probabilities or proof of causal reasoning. GO is not necessarily improvement. Three training runs are not three times the independent road count. A checkpoint comparison is not automatically a training-algorithm result.

Do not call public-recipe coefficients Super's final training reward, equate a paper's research variant with released weights, claim exact likelihood/unbiasedness merely from a PPO/GRPO label, or describe the current smoke as the one-candidate base default. Never assert “first” without a dated release-time novelty check. Do not compare card/benchmark scores across mismatched protocols.

No public PAI-AV/NuRec-derived numbers before KC-L; no derived weights/labels before KC-L2; no redistribution of restricted source assets. Shadow mode is open-loop. Jetson and DRIVE AGX Thor results are distinct. Emulated precision is not measured engine speed. Single-node work is not cluster/production-scale evidence or industry infrastructure experience. Huy's mastery is not inferred from agent delivery.

## 7. Admission and kill checks

| Check | Condition | Result if unmet |
|---|---|---|
| KC-1 | S1 calibration/known answers and operational power machinery qualified | Already technically qualified; any new core method needs renewed review |
| S2 admission | November 2 threshold, current release check, owner registration/predictions, asset rights and concrete action approvals | No scientific pilot |
| KC-2 | Complete reliable instrument with valid sampling/power for the intended question | Diagnose, narrow with explicit registration, or INSUFFICIENT; no confirmatory S4 claim |
| S4 engineering admission | S2 accepted, rights/lineage, exact priced topology, conservative cap/deadline, known-answer acceptance contract and concrete action approval | No qualification allocation or update |
| S4 scientific admission | Engineering qualification supplies measured fit/cost; Huy freezes the scientific registration/predictions and releases the priced replicated study | No scientific baseline or controlled training study |
| KC-4 correctness | Actual identity/likelihood/export parity, frozen/trainable masks, reward semantics and complete attempted-data accounting | No learning claim or scale-up |
| KC-4 study | Useful learning signal and funds for replication plus independent evaluation | Baseline/feasibility conclusion or separately priced amendment |
| KC-L | Written resolution of publication rights for derived benchmark numbers | Publish only permitted code/synthetic material |
| KC-L2 | Written resolution of derived weight/label publication rights | No derived model/label release |
| KC-$ | Remaining approved cost and deadline sufficient for next unit plus cleanup | Stop/export/destroy/verify absence immediately |
| Deferred stage | Explicit re-entry decision and its original hardware/asset/scientific gates | No implementation |

Numeric scientific floors/predictions remain Huy's. Dates for completion require P-003 and measured throughput; no obsolete v4.1 date is presented as a live deadline. D remains 2026-10-26 for historical records.

## 8. Principal risks and responses

| Risk | Evidence to seek | Response |
|---|---|---|
| Reward rises while behavior stagnates | Vector rewards, raw events, frozen outcome measures | Select one demonstrated mechanism; keep negative results |
| Tied groups or poor exploration | Per-group action/return diversity and valid gradient | Diagnose before coefficient tuning |
| Short-horizon or controller mismatch | Controlled trace comparison and time-resolved failures | Freeze protocol; isolate one change |
| Stale or selectively surviving data | Weight/group/attempt ledger before upstream drops | Fail identity checks; quantify effective sampling; no silent deletion |
| Reconstruction/traffic exploitation | Validity exits and independent diagnostic outcomes | Limit claims; later environment change is separate |
| Statistical underpower | Actual road support, operational power, run variation | INSUFFICIENT or explicitly revised study before outcomes |
| Budget consumed before confirmation | Measured all-in forecast and reserved evaluation | Stop at the baseline or seek a separate extension |
| Stronger correction baseline or new upstream release | Dated literature/release review | Narrow novelty; compare only when feasible, never race an offered artifact |
| Publication restrictions | KC-L/KC-L2 records | Resolve early; code/synthetic fallback |
| Scope returns to eight simultaneous stages | Stage/deliverable ownership check | Keep S1/S2/S4 active; propose extensions after the focused conclusion |

## 9. Open decisions and extensions

P-030/D-018 evaluator ownership and cash ceilings remain adopted. P-003 weekly hours and P-004 license/publication questions remain open. P-031's focus is incorporated only through this reviewed plan revision and its adoption record; its initial wording is not a frozen experiment.

Before the next paid step: Huy records the S2 scientific definitions/prediction and actual asset/rental actions. Agents prepare exact current quotes, hashes, source and hosting facts, cost/cleanup controls and the blocked registration's engineering fields. Before S4: choose the question using permitted development evidence, fix the prediction/metric-to-decision rule, and price the actual required replication/evaluation.

P-002/P-006 hardware/rig and P-012–P-029 extensions remain unadopted except their explicitly recorded historical adoptions. Nothing in this plan funds them or unlocks a protected path. Future reasoning RL, reactive traffic, constrained RL, student correction or runtime work must earn a separate question and resource case after the first focused conclusion.
