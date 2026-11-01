# Engineering and experiment specification for focused RL

Agent-authored design, 2026-11-01, serving S2/G1–G3·R1 and S4/G9/G7. This specifies future acceptance evidence. It does not report those tests as executed, fill Huy's scientific prediction slots, authorize S4 implementation before admission, or unlock protected files. The main plan owns sequence and scope; this document refines the research review.

## 1. Objects that must be versioned

Treat a run as a graph of immutable identities rather than a model name. The baseline model revision, tokenizer/processor, inference checkpoint, converted trainer checkpoint, exported checkpoint, optimizer state and reference policy must be separately identifiable. Each conversion records its input hash, output hash, code revision and configuration. A repository SHA without the actual working-tree diff does not identify modified source.

The environment record includes Gym, Cosmos-RL, its recipes dependency, training AlpaSim, evaluation AlpaSim/evaluator, lockfiles, container digests, GPU edition/driver, renderer/Harmonizer, traffic, controller and trajectory post-processing. Training and evaluation may intentionally differ; differences are explicit fields, not an overwritten common config.

Every rollout binds:

- source dataset revision, scene/road/source-clip IDs, split role and permitted use;
- group ID, generation index, training-run identity, simulator replicate identity, actual RNG mode and configured/observed policy weight version;
- requested horizon, warm-up, actual valid action count, camera/control rates, actual timestamps, termination reason and retry lineage;
- raw reward components, aggregate reward, masks, group advantage and the version of the code computing each;
- current/reference/behavior policy identities (with KL off, the reference is the saved zero-update checkpoint, compared offline), log densities, reduction convention, clip indicators, KL term and valid replay denominator;
- original trajectory and metric arrays, deviation bins, black-image/render status, map coverage and infrastructure failures;
- elapsed paid time, resource count and the artifact needed to reproduce export and evaluation.

The standard Gym episode object sets its dense-metric field to an empty dictionary. The required event-level audit therefore needs a qualified capture of returned timestep metrics or simulator-side originals, joined by run/scene/timestamp before aggregation or cleanup. Do not assume ordinary saved episode artifacts already contain that evidence. Record model train/eval mode and non-parameter buffers as well as weights; frozen weights alone do not imply unchanged runtime behavior.

An incomplete or contradictory identity blocks admission. Missing values are not zeros. A truncated recording is neither automatically a crash nor automatically a completed clean episode. Expected terminal semantics are a scientific definition to register and test.

## 2. Separation of data and information

Use distinct training, development and confirmation roles. Audit source-road, source-clip and temporal overlap across roles, plus known model-training contamination. Document unknown teacher pretraining provenance. Scene IDs alone do not establish independence.

The existing 1,761/441 split is an upstream starting partition, not proof of the sampling assumptions. The current preflight requires 441 full scenes. If source overlap or available road support forces a change, return a documented protocol change for Huy; do not bypass the validator or replace excluded scenes opportunistically. No new split is adopted by the literature review.

Development outcomes may select the intervention and checkpoint rule. Confirmation outcomes may not. S2's instrument measurements need an explicit information policy: noise estimation under the registration is not permission to mine confirmation failures for a reward change. If the same scenes have already informed the intervention, obtain independent confirmation or report an exploratory result.

Privileged ground truth can be used by an allowed scorer. It must not enter the policy observation through future ego actions, labels, prompt fields, filenames or selection shortcuts. Validate this by schema tests and adversarial canaries. Changing the privileged input to a judge changes its task; label and assess that change independently.

## 3. Baselines and contrasts

Three baselines answer different questions:

| Baseline | Purpose | Required equality or difference |
|---|---|---|
| Released frozen teacher | Reference driving behavior | Registered evaluation protocol |
| Zero-update conversion/export | Detect behavior changed by format, sampler, defaults or conversion | Same weights and qualified inference semantics; deviations diagnosed before learning claims |
| Shipped RL arm | Establish the actual learning process | Frozen implementation/config, including the registered learning-run settings that no shipped preset supplies as one two-GPU multi-scene run; all runs and selection costs retained |

The first intervention changes one defined mechanism. Examples are the collision predicate, controlled horizon or one sampling parameter. Do not also switch checkpoint family, precision, image size, control frequency, controller, traffic or estimator. A bug correction that changes the baseline invalidates earlier causal comparisons; preserve old results as diagnostics and restart the affected comparison with the corrected baseline.

When comparing a longer horizon or larger group, the number of samples and updates within a fixed cash budget will differ. At fixed `mini_batch`, a longer horizon also adds optimizer steps per rollout, because replay rows are per driving step. State whether the estimand is equal paid compute, equal simulated interaction or equal optimizer updates. Choose one primary comparison; secondary budgets are separately priced. A method that wins per update but costs several times more per update has not demonstrated compute efficiency. For training-procedure confirmation after mechanism selection, reserve fresh independent training runs or a reviewed selection-aware design. Reused exploratory checkpoints support only a conditional checkpoint comparison; new roads do not remove selection on the training runs.

## 4. Acceptance tests before a training conclusion

These are specifications, not current passes. Huy retains the RL derivation and numeric tolerances; agents may construct independent fixtures and verify them after admission.

| ID | Input or perturbation | Required outcome and reason |
|---|---|---|
| E01 | Identical retained weights and identical inference inputs/noise | Repeatability under the declared mode; hashes/config identify any exception |
| E02 | Trainer conversion followed by export with zero optimization | Parity through the numerical ladder; a change here cannot count as learning |
| E03 | Snapshot all named parameters before/after one trainer step | Frozen VLM/visual weights unchanged; only declared trainable parameters may change |
| E04 | Replay identical sampled SDE states, masks, noise convention and behavior weights before update | Old/new log-density difference zero within a registered tolerance; ratio near one |
| E05 | Supply a stale rollout while setting zero allowed lag | Qualification rejects or stops on observed mismatch; a warning is insufficient; test the actual dispatcher path |
| E06 | Permute sample order across workers and reconstruct a group | Same group membership, rewards and advantages under the registered reduction; rank-local subsets cannot redefine the global group |
| E07 | Equal rewards for every group member | Zero reward-based centered advantage. In the pinned stock path with KL=0, no auxiliary gradient and zero reduced gradient, optimizer.step must be skipped: parameters and per-parameter optimizer moments/step counters stay unchanged at that step. The scheduler still advances in the pinned trainer, so learning-rate param groups and all trainer state need not remain unchanged; test auxiliary/KL cases separately |
| E08 | Reverse the reward ordering with other inputs held fixed | Advantage ordering reverses; independent test detects a sign inversion |
| E09 | Duplicate/omit records, misreport behavior version, or mix versions within one recorded action trace | Reject integrity violations; strict parity cohort requires uniform matching versions. A later async batch may contain different correctly disclosed behavior versions under its qualified contract; no silent best-of-retry selection |
| E10 | Synthetic contact before warm-up, after first incident and at final timestamp | Online and evaluation traces match their separately documented masks and time domains; mismatch is visible |
| E11 | Both front and lateral contact in one episode | Any-contact indicator remains one; an additive two-indicator objective is explicitly different |
| E12 | Identical path represented at different timestamps, plus misaligned timestamps | Reference deviation uses the specified interpolation/coordinates; bad time ordering fails |
| E13 | Stationary trajectory, oscillation, legitimate emergency stop, long detour | Each diagnostic records the intended phenomenon; progress cannot silently stand in for all outcomes |
| E14 | Black frame, ≥4 m exit, missing metric, NaN active log density | Preserve validity/infrastructure status; fail admission where required; do not convert missing evidence to successful driving |
| E15 | Duplicate launch, process failure, resume at wrong weight/config | Detect mismatched run identity and duplicate attempts; preserve all failures and exact resume state |
| E16 | Same locked checkpoint evaluated through train and gate adapters | Differences explained by a configuration/adapter audit before attributing them to the model |
| E17 | Distinct training runs evaluated on shared roads | Report both axes of uncertainty; never count rollout multiplicity as independent road support |
| E18 | Cost or deadline boundary reached | Export evidence, terminate task resources, verify absence; no automatic extension |
| E19 | Groups of two with reward gaps of 0.001, 5 and 10; a group of six with one collision; a group of six differing only by noise | Reproduce the pinned advantage values: about ±1 for every group of two, −√5 for the collision, unit rms for noise. This records that magnitude is lost; it is not a learning pass condition |
| E20 | One trainer step with known rows, `mini_batch` and iterations | Counted optimizer steps plus logged zero-gradient skips equal the pinned formula; parity uses values from before the first step; every reported update count names its unit |
| E21 | Stock run with `kl_beta: 0` and a saved zero-update checkpoint | Logged KL is recorded as not computed, never as zero divergence; distance from the zero-update checkpoint is produced offline; the early checkpoints it needs survive retention |

For E04, a single loose `allclose` is inadequate. First test mathematical identities on exact small inputs; then inspect the dtype, reduction order and per-element error; then test the actual sampled-state replay with frozen versions. A tolerance is justified by downstream effect and numerical scale, not widened after a failure. Action-level and closed-loop checks remain necessary even if log-density parity passes.

E05 must establish a real version fence. Filtering after advantages were computed can change the effective estimator and must not be introduced casually. A fail-fast assertion or supported synchronous qualification mode may be sufficient for initial diagnosis; an algorithm/runtime modification requires its own reviewed design and scope decision. No new asynchronous framework is required.

## 5. Small mathematical oracles

The following examples specify reasoning exercises and independent test expectations, not a replacement RL implementation.

**Centered relative signal.** With all rewards equal, subtracting the group mean gives zero for every sample. With finite positive denominator epsilon, standardized advantages are zero. For two different rewards, verify the actual code's sample-versus-population standard deviation and epsilon before asserting a numerical advantage value. Rescaling all rewards by a positive number preserves ordering; exact standardized magnitudes are invariant only if the epsilon convention permits it. Scaling just one component can change rankings.

**Group of two.** With population standard deviation, `std = |Δ|/2`, so the pinned advantage is `±|Δ|/(|Δ| + 2ε)`. At ε = 1e-6, a reward gap of 0.001 gives ±0.998 and a gap of 10 gives ±1.000. In a group of six with one collision and all else equal, the collided candidate gets −√5 whatever the penalty. Any untied group has advantage energy near its group size, however small its differences. At group size two, coefficients therefore act through ordering only. See [the gap review](gap-review.md), G1.

**Joint versus mean density ratio.** If two independent factors have likelihood ratios 4 and 1, their joint ratio is 4; the geometric mean is 2. Thus replacing a sum of log ratios by their mean changes the objective. At identical policies both equal one, so a parity test alone cannot identify the intended estimator. Add a deliberately non-identical analytic case.

**Gaussian KL.** For univariate unit-variance normal distributions whose means differ by 0.1, the directed KL is 0.005 in either direction. Changing variances makes direction matter. Test both order and reduction, not just a zero-KL identity.

**Clipped objective.** For positive and negative advantages, verify the minimum of clipped and unclipped terms separately. A clipped surrogate is not a hard bound on all post-update probabilities, parameter change or KL. Observe actual clip fraction and measured divergence.

**Reward vector.** An episode with both front and lateral contact can have `max(any)=1` and `max(front)+max(lateral)=2`. Replacing any-contact with their sum changes magnitude as well as predicate. `min(max(front)+max(lateral),1)` is a different explicitly defined objective and may require an evaluator extension. Its adoption belongs to Huy, with event timing preserved.

**Power and units.** A 20% relative decrease at an 8% base rate is a 1.6 percentage-point absolute change. The base rate is hypothetical until measured. Neither 441 scenes nor three training runs automatically makes that change detectable under road dependence and all-strata requirements.

## 6. Training diagnostics and falsifiers

Keep scalar return alongside every component. Track group reward variance, all-tied/all-fail/mixed groups, the reward component that decided each group's order, the share of groups whose order flips on re-simulation, action diversity, valid sampled transitions, the share of replay rows after the first hard event, gradient norm, changed parameter norms per module, old/current/reference policy differences, clip fraction, KL where it is actually computed, rollout lag and useful samples per paid GPU-hour. A nonzero gradient is not proof that the intended action choices received useful credit.

Group candidates are themselves same-policy replicates, so their spread alone cannot separate signal from noise. Re-simulate groups under the declared seed mode; if the path cannot repeat, record that limit.

| Candidate mechanism | Evidence needed on train/development data | One possible intervention | Evidence that weakens it |
|---|---|---|---|
| Reward/evaluation semantics differ materially | Reconstruct specific episodes where event time/predicate changes ordering | Correct one approved mask, predicate or termination rule | Reconciled ordering changes but held-out outcomes do not, or another component explains the effect |
| Insufficient within-group alternatives | Quantified tied rewards and near-identical sampled actions | One noise or group-size setting | Diversity rises without useful ranking/learning; extra cost explains apparent gain |
| Noise-ranked or magnitude-blind groups | Group order flips on re-simulation, or a small continuous term decides order rather than the incident term | One group-size or normalization change, with estimator scope decided by Huy | Order is stable and incident-decided, yet held-out outcomes do not move |
| Unanchored drift from the pretrained expert | Distance from the zero-update checkpoint grows while development outcomes fall from step 0, on fixed-input probes and early snapshots | One anchor or learning-rate change, as Huy's objective decision | Outcomes match a sham run that permutes advantages within groups, or regression persists with the policy held near the teacher |
| Horizon mismatch | Failure timing and controlled short/long evaluation profiles | Training horizon alone | Improvement disappears at matched compute or is explained by changed exposure/termination |
| Coarse temporal credit | Same episode advantage spreads across decisions with different causal roles | Separately designed credit estimator | No robust advantage over baseline under equal data/compute; parity or density assumptions fail |
| Proxy exploitation | Return increases with stalling, invalid episodes or ungrounded scoring | One targeted objective correction | Independent outcomes remain unchanged or regress elsewhere |

The falsifiers are qualitative design requirements. Huy fills the quantitative effect, tolerance, stopping and decision rules before the dependent run. Choosing a mechanism from development data is legitimate if the final confirmation remains independent and the selection process is disclosed. A sham run that permutes advantages within groups keeps update magnitudes and removes signal, which separates erosion from learning. It costs one run and a small adapter, so it is optional and needs its own scope decision.

## 7. Statistical contract

Define a checkpoint estimand before calculating it. For example, retain the gate's equal-road target: sample a road from the registered population, then average its eligible scenes and replicates according to the frozen rule. Per-stratum support is counted at the road level. Register how a road spanning strata is handled; do not assume strata are independent merely because they have different names.

An algorithm claim adds a random variable: the independently trained model. Its uncertainty is crossed with evaluation roads, since the same model is evaluated on many roads and multiple models share them. The existing gate does not automatically supply that algorithm-level interval. A reviewed hierarchical or run-level contract is required; with very few runs, report descriptive run variation and scoped checkpoint verdicts instead of manufacturing a precise method-level bound.

Three independent training runs per arm are the minimum planned diagnostic replication. They do not guarantee power or reliable bootstrap coverage. Training seeds do not guarantee full runtime determinism; record the actual sources of randomness and scheduling. Simulator replicates remain unseeded unless their path demonstrably consumes and preserves the registered seeds.

Use the actual bounded-road operational method for authorization power. Historical Tango, bootstrap and Gaussian tables remain diagnostics/planning references. Margins and alpha spending do not change after observing results. Checkpoint selection, multiple arms and repeated confirmation use must be included in the inferential contract. If no sufficiently powered design fits, finish with an exploratory diagnosis or INSUFFICIENT.

## 8. Cost and admission packet

The forecast is `setup + training + development selection + locked confirmation + storage + transfer + fees + failure reserve`. Count all replicas and all failed/rerun jobs. Separate cash deposited, accrued usage, unused credit and remaining liabilities; avoid double-counting prepaid usage.

Measure setup time, complete episode wall time, valid samples per update, optimizer time, utilization, memory high-water marks, retry loss and export/evaluation overhead on the actual approved topology. Multiply by the number of training runs and selected checkpoints. Count fresh algorithm-confirmation runs in addition to exploratory pilots; reuse of exploratory baseline checkpoints does not provide independent training-procedure confirmation without a qualified selection-aware design.

S4 has two admissions. After S2, a separately priced, explicitly capped engineering qualification may measure fit, correctness and throughput under deterministic expected outcomes; it needs an approved host/asset package and hard stop but does not require an already measured training forecast. Before the replicated scientific study, the packet must contain that measured forecast, independent-evaluation reservation, exact machine/region/resources, verified guest capabilities, license/access status, asset-retention rules, external kill deadline, teardown evidence procedure, and Huy's experiment registration. The previously destroyed single-GPU Vast VM does not qualify the two-GPU trainer or grant another rental.

If two arms do not fit, do not reduce confirmation or hide seed failures. Finish the baseline diagnosis or propose a separately priced extension. If licensed results cannot be published, retain code, synthetic tests and permitted artifacts; do not infer publication rights from public source-code licenses.

## 9. Deliverables and ownership

Agents prepare manifests, data audits, source/semantic reconciliation, independent test scaffolds, launch controls, logs, cost forecasts and verification reports. Huy owns the scientific objective, selected intervention, reward/loss/estimator research design, numeric predictions, margins, strata, interpretation and claims. Evaluator implementation remains delegated under D-018, subject to protected-file guards and independent review.

The next scientific artifact is a completed experiment registration, not a training launch. The next runtime evidence is S2 instrument qualification under its separately approved protocol. This specification keeps every future dependency visible without presenting future work as already executed.
