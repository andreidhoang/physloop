# Proposed focused RL research path

**Historical proposal, superseded with corrections:** Huy authorized adoption after independent verification; D-021 adopts plan v5. Use [the completed review](2026-11-01-rl-direction-review/report.md) and the current main plan. The proposal text below records the earlier decision input, not current execution authority.

Agent-authored recommendation for Huy, 2026-11-01. P-031 is a proposal, not an amendment to plan v4.1 or admission to S4. It serves the S2 prerequisites for G1/G2/G3 and the S4 teacher-RL study G9/G7. No scientific prediction, rental, asset access or training run is authorized by this document.

Make closed-loop AV post-training the research focus: establish when an RL update improves held-out driving outcomes, explain the failures of that transfer, and test one correction. Keep the existing gate as the measuring instrument. This fits Huy's stated interest in model research more directly than pursuing evaluation, distillation, serving and hardware simultaneously. It sacrifices breadth of evidence for the existing simulation-engineering role target; it does not guarantee greater impact or a particular hiring outcome.

The proposed first research question is: **When does expert-only closed-loop RL improve held-out collision outcomes while preserving progress and off-road performance, and when does its reward increase come from stalling, proxy exploitation or simulator artifacts?** A first reproducible baseline establishes the research substrate. A controlled intervention and a supported mechanism establish a contribution beyond running NVIDIA's tutorial.

RL is a plausible long-term specialization because the learned policy changes the states it encounters. Logged imitation supplies examples on a different state distribution; closed-loop post-training can expose consequences of the policy's own mistakes. However, RL still depends on the pretrained representation, exploration coverage, reward validity and simulator fidelity. If the model cannot perceive an obstacle, never samples a useful response, or receives incorrect feedback, more optimizer steps alone may not help. Existing [DPPO](https://arxiv.org/abs/2409.00588) and [Flow-GRPO](https://arxiv.org/abs/2505.05470) also mean that applying policy gradients to a diffusion or flow model is established work. Novelty must be checked for the specific mechanism and experiment.

## Scope changes proposed for adoption

The dependency order becomes S1 → S2 → S4, with an explicit stop after an RL study. S3 and S4 are already reversible in plan §3; promoting G9 above G5 as the headline and deferring later work require adoption of this proposal.

| Work | Proposed priority | Reason |
|---|---|---|
| S1 gate | Retain completed instrument and its limitations | A correct underpowered result remains INSUFFICIENT |
| S2 real instrument qualification | Complete before training claims | Pin semantics, measure variability, expose invalid rollouts, estimate useful throughput and cost |
| S2 broad BF16/FP8 policy comparison | Defer the full comparative study if an amended S2 protocol can establish its required reliability without it | Precision research need not precede an expert-RL question |
| S2 TransFuser baseline | Retain the integration sanity check; propose deferring a full comparative sweep | Separate pipeline defects from the teacher's behavior |
| S4 expert-only RL | Primary research milestone | Direct model intervention with an independently measured consequence |
| S4 one-to-eight-GPU scaling sweep | Defer; retain measured utilization, goodput and failure diagnosis on the chosen topology | Spend scarce compute on replicated behavioral evidence first |
| S3 student verdict and S5 correction | Defer until the first RL conclusion | A missing student must not block the teacher study |
| S6–S8 device, HIL and shadow mode | Defer from this focused path | They answer different questions and require separate resources |

Where this table defers currently required work, adoption must explicitly amend that requirement; S8 remains optional under its existing gates. S2's scientific acceptance conditions must be re-registered explicitly if its policy matrix changes. Full provenance, estimator validity, known-answer and negative controls, contamination checks, power analysis, validity accounting and actual cost measurement stay mandatory. No replacement sampling population, margin or method is chosen here.

## Three distinct optimization paths

| Path | What the inspected implementation trains | Implication |
|---|---|---|
| AlpaGym with Alpamayo 1.5 | Continuous action expert; language and visual parameters frozen | Best first path under this project; optimize consequences of sampled actions |
| `alpamayo1_x_rl` | Open-loop VLM with discrete action tokens; reasoning-grader example | A separate reasoning research lane with different compute and evaluation requirements |
| Alpamayo 2 Super | NVIDIA describes the combined model as RL-post-trained | The inspected public description does not specify the final component-by-component reward recipe |

At AlpaGym `972d160eed0e23d388497851504a3a233fec5879`, the wrapper comes from recipes PR #18 at `1d99fc50370c96637157455da386323a44663c9d`; it is not recipes main. The [wrapper freezes the visual and language parameters](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/models/expert_model/cosmos_wrapper.py#L43). The [shipped inference adapter](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/inference_model.py#L35) uses `traj_future`, skipping autoregressive reasoning generation. A text-reasoning RL claim therefore needs a separately qualified path.

The [AlpaGym onboarding guide](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md) recommends two GPUs with at least 40 GB each for the documented setup. The [open-loop recipe](https://github.com/NVlabs/alpamayo-recipes/blob/2e907dab31e5353f42b2c8a2f79d7e39b6e0d42c/recipes/alpamayo1_x_rl/README.md) documents at least five 80 GB GPUs. These are documented configurations, not a proof of minimum possible memory use. Our completed one-GPU Vast synthetic test does not qualify either training topology. Checkpoint conversion, optimizer/rollout memory, renderer concurrency, communication and export parity still need tests under a new approved rental.

The [Super announcement](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) supports the model-level RL statement. Do not infer its exact final rewards, component training schedule or judge from that statement, and do not change the initial teacher during a reward comparison.

## Why reward design alone is incomplete

Four mechanisms must work together: the policy must sample different useful behaviors; the reward must rank them sensibly; the estimator must assign the feedback to the trainable decisions; and those updates must transfer beyond the training scenes and simulator settings.

For the schematic group advantage `A_i = (r_i - mean(r)) / (std(r) + epsilon)`, equal rewards give zero reward-based advantage. That does not guarantee zero parameter changes: auxiliary losses or optimizer weight decay can still act. Log zero-variance groups, distinct action samples and reward-component variation. All failed samples can be uninformative even when their common penalty is very large.

Positive affine scaling leaves standardized advantages unchanged when the numerical epsilon can be neglected. If collisions are the only varying component, increasing their penalty from 10 to 100 supplies no new ordering. When progress and other components also vary, changing the collision weight can change the ordering; that is a different case. Inspect within-group rankings and tradeoffs before spending on a coefficient sweep.

The current [advertised smoke configuration](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/experiment/alpamayo_1_5_local_2gpu_smoke.yaml) uses `n_generation=2`. The one-candidate setting belongs to the base default; describing the current quick start as necessarily producing zero advantage would be wrong. Nonzero reward variance is necessary for this relative signal but does not prove useful learning.

There are two different time axes: driving decisions through an episode, and denoising transitions inside each sampled trajectory. AlpaGym's inspected [density reduction](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/diffusion/flow_matching.py#L538) and [replay objective](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/replay_objective.py#L44) yield a geometric-mean likelihood surrogate, not the exact probability ratio of an entire driving episode. The [trainer](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/trainer.py#L267) applies an episode advantage across valid replay steps. Improving temporal credit assignment is a possible later research problem; changing the estimator before reproducing this one would obscure the baseline.

The smoke run controls 22 steps at 5 Hz after ground-truth warm-up; our planned gate runs 200 steps at 10 Hz. A gain over a short training horizon might not survive the longer evaluation. That is a testable concern, not an observed failure. Freeze and report each protocol. If horizon becomes the experimental variable, hold policy frequency, warm-up semantics, traffic, reward aggregation and all other settings fixed across its training arms.

## Milestones and acceptance evidence

These are subdivisions of S2/S4, not new authorized stages. Durations depend on P-003 and the measured pilot; do not turn the old calendar into a delivery promise.

| Milestone | Deliverable | Required evidence before proceeding |
|---|---|---|
| S2 instrument | Frozen baseline, valid metric extraction, reliability and cost report | Same-policy controls, complete manifests, separate infrastructure/validity outcomes, measured power for the proposed question |
| S4 training correctness | One working exportable update on approved development data | Frozen-parameter hashes; finite, expected trainable gradients; old/new likelihood parity before an update; known-answer reward and estimator tests; zero-update export parity; reproducible configuration |
| S4 baseline pilot | One shipped reward arm, three independent training runs, frozen teacher reference | Seed-specific curves and development outcomes, useful learning signal, cost forecast, no unexplained gains or invalid-state exploit |
| S4 controlled study | Baseline reward versus one mechanism-driven change, at least three training runs per arm | One variable changed, fair budget accounting, fixed checkpoint selection, independent confirmation and anti-hacking outcomes |
| S4 report | Supported behavioral gain, scoped negative result, or INSUFFICIENT | Reproducible result and mechanism evidence; publication permission for licensed-data-derived results |

The baseline pilot can end in a valuable diagnosis. It is not a causal comparison of reward designs. A two-arm study requires six training runs plus reference evaluation and confirmation; it cannot be smuggled into the one-arm pilot budget.

For training correctness, check noise samples, masks, SDE settings and weight versions when comparing old/new likelihoods. Test a zero-update round trip from inference checkpoint to trainer format and back before attributing changed behavior to learning. Start parity checks with no allowed weight staleness and one replay iteration. A frozen backbone can produce different activations as the vehicle reaches new states; its weights, rather than all activations, must remain unchanged.

Keep the complete reward vector alongside the scalar: progress, all/front/lateral/rear contacts, off-road, executed reference deviation, and diagnostic stall/oscillation measures. Log sample group identity, training step, rollout weight version, policy noise, retries, valid horizon and denominator. A stopped or invalid episode must not silently receive the same meaning as a completed collision-free drive.

## Selecting the first intervention

Use training/development failures to choose one intervention. The strongest starting candidate within the existing S4 card is reward alignment, but the mechanism must be observed before Huy records its numeric prediction.

| Development finding | Candidate single intervention | Main falsifier or confound |
|---|---|---|
| Replay followers cause rear contacts that penalize otherwise useful braking | Change the collision predicate while preserving its temporal aggregation, scaling and other reward terms | The policy may learn aggressive braking or increase rear collisions; geometry is not a legal fault determination |
| Improvement disappears after the short training horizon | Change controlled episode horizon only | Longer episodes cost more and change the number of updates under a compute cap; report both sample and wall-clock efficiency |
| Groups frequently have indistinguishable rewards/actions | Change one exploration or group-size parameter | Extra samples cost compute; dropping uninformative groups can change the effective scene distribution |
| Reward improves by stopping, oscillating or exploiting invalid renders | Diagnose the loophole and propose one targeted correction | A larger collection of penalties can hide the cause; never drop invalid episodes to make the result improve |

For the collision candidate, retain the shipped `collision_any` arm as A. A possible B uses front-or-lateral contact, capped to prevent counting both twice, **only if** the source scorer's temporal aggregation supports that comparison. Inspect whether the existing term is episode-any, fraction-of-steps or another aggregate before writing the reward; otherwise changing both the predicate and its aggregation creates two interventions. The alternative needs explicit approval and implementation qualification if the shipped config cannot express it. Call it a contact-geometry proxy, preserve all/rear-contact monitors, and never present it as proven legal or causal fault.

The [starter reward](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/reward/progress_safety.yaml) is `P - 10 C_any - 5 O - 0.01 D`; the [distance term](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/rewards/distance_to_gt.py) is executed-position XY RMSE. Its terms are a reproducible starting point, not a disclosure of Super's checkpoint recipe or universal driving priorities.

At equal total compute, longer episodes or larger groups may allow fewer updates. Report that difference rather than pretending equal steps imply equal cost. A secondary matched-update comparison is useful only if separately funded and pre-registered. Do not change reward, horizon, traffic, checkpoint and sampling strategy in the same contrast.

## Confirmation and claims

Split train, development/checkpoint selection, and confirmation by source road/clip, with adjacent-source and known training-overlap audits. Audit the existing 1,761/441 split rather than assuming different scene IDs establish independence. If strict road separation requires removing overlaps, record the smaller sample and recompute power. The pretrained teacher's complete training history may remain unknown; describe that limitation.

S2 may estimate instrument noise under its registration. It must not become a source of confirmation-scene failure mining for reward design. If outcomes from the 441 scenes have informed interventions, those outcomes are development evidence and the claim needs independent confirmation or an explicitly exploratory label. A public benchmark is not automatically an untouched holdout.

Select checkpoints on development data under a rule fixed before training, then evaluate the locked selection. Retain the fixed teacher, evaluate corresponding arms on the same scenes, and check any reuse against the complete protocol plus a concurrent drift sentinel. Training seeds and simulator replicates are different sources of uncertainty; three training seeds do not multiply the count of independent roads. AlpaGym supplies session/step generators, whereas the currently planned challenge driver ignores its session seed; qualify each path separately and name any unseeded replicates honestly.

Report per-seed results and variation across training runs alongside paired road-cluster bounds, strata and validity exits. The existing gate qualifies a defined checkpoint comparison; it does not automatically establish performance of an entire training algorithm or adjust for searching many checkpoints. A method-level claim needs an explicit hierarchical or run-level inferential contract. No new statistical core or method is adopted here.

Success requires a supported change on the registered collision endpoint and non-inferiority on progress/off-road, with stalls, rear contacts, comfort diagnostics, >=4 m exits and infrastructure failures visible. Judge scores and training return remain diagnostics. The existing 20% relative collision target is a research target, not a promised detectable effect. S1's operational bounded-road gate was conservative enough to return INSUFFICIENT on almost all noisy-equivalent synthetic trials; Gaussian planning approximations cannot override that limitation. Measure actual power before buying the experiment. More independent roads, a narrower justified claim, or an explicitly reviewed inference amendment are possible responses; weakening criteria after seeing results is not.

Replay traffic cannot establish learned social interaction with responsive drivers. Closed-loop ego control does not make the surrounding vehicles reactive. Moving to CAT-K is a separate future intervention with its own validity and seeding checks.

## Reasoning RL as the second research lane

AlpaJudge is useful first as a diagnostic of explanations on held-out development examples. Its [RF input contract](https://github.com/NVlabs/alpajudge/blob/607ae9d2c56e08a9f55a1093ce625af43a53b114/src/alpajudge/reference_free/processing.py#L13-L68) uses images and ground-truth action history/future. Replacing that future with policy-generated actions changes the scoring task. Validate that adaptation against independent labels before using it as a reward; keep privileged future information outside the policy input.

Its [soft aggregation](https://github.com/NVlabs/alpajudge/blob/607ae9d2c56e08a9f55a1093ce625af43a53b114/src/alpajudge/reference_free/model.py#L280-L319) can equal 0.7 when the safety head is zero and the other heads are one. That is an algebraic property, not an observed frequency of dangerous behavior or a calibrated failure probability. Learned grounding/causality labels do not prove causal faithfulness.

A frozen judge is compatible with policy-gradient RL; differentiation through it is unnecessary. The relevant requirement is that its reward vary with the trainable policy's choices. A fixed rationale scored with fixed images and fixed action references gives the same score for every expert sample, so its within-group advantage contribution cancels. Across closed-loop states an expert can indirectly change inputs; such a reward must be tested for what it actually encourages. This does not become language-reasoning training while the language parameters remain frozen.

Only after the first expert study, evaluate a separate reasoning lane: an appropriate trainable reasoning model, a validated judge on its outputs, one fixed added term, and independent closed-loop outcomes. Joint reasoner/expert updates, adaptive constraints, reactive traffic, multi-node scaling and student transfer are later decisions, not simultaneous first experiments. A path can grow into those topics without funding them all now.

## Cash discipline

The adopted $100 pilot / $1,200 S2 / $2,500 project caps remain nested ceilings. At maximum S2 spending, $1,300 remains for everything after S2. The recorded $0.60 host usage is within S2 and must not be deducted a second time. Historical plan estimates put full S4 at $807–2,628 before combining it with S2; they are not current quotes or measured training cost.

One proposed allocation of that $1,300 remainder is $650 for training and training qualification, $400 reserved for independent evaluation, and $250 for storage, transfer, fees and failures. This is a planning partition, not a claim that six training runs or the required evaluation fit. Evaluation's historical full-S4 range alone was $340–1,318. Measure the chosen two-GPU topology before promising a study; if the forecast fails, retain a one-arm pilot/negative result or request a separately priced expansion. Do not silently cut seeds, strata or confirmation to rescue a claim.

Use `remaining = project_cap - recorded_S2_cost - other_project_cost`, with deposits, accrued usage and remaining liabilities reconciled under D-018. Forecast all provisioning, idle time, model conversion, useful training, evaluation, storage retention, transfer and teardown. Protect evaluation funds before launching training. A dollar ceiling is not a release of funds or permission to accept licenses. KC-L and KC-L2 remain central to whether the result or trained artifact can become a public contribution.

## Huy and agent responsibilities

Keep the adopted ownership arrangement and the recommended 70% research/design, 20% ablation/interpretation, 10% measurement review allocation for Huy after S2. The 70% would concentrate on reward semantics, exploration, flow-policy likelihoods, credit assignment and one implementable intervention. Huy owns scientific predictions, model-research loss/reward/estimator design, margins, strata and conclusions. Agents own evaluator implementation, data/split/lineage audits, orchestration, cost controls, telemetry, engineering defect diagnosis and independent verification, within admitted stages and file protections.

Before the first RL run, Huy should be able to explain five things: which parameters change; what randomness creates alternatives; why the reward can distinguish them; what the actual estimator approximates; and what outcome would refute the intervention. Follow with the two time axes, state-distribution shift, proxy exploitation and the difference between checkpoint and algorithm-level evidence. Running commands or reading a rising reward curve is insufficient evidence of those skills.

The immediate decision is whether to adopt this focused scope and revise the relevant stage cards. The next eligible execution is the S2 instrument pilot, once its admission, concrete protocol, prediction, license/access and rental approvals are recorded. The next research artifact is a written S4 question and an experiment contract whose numeric prediction is Huy's, informed only by permitted development evidence. No future-stage implementation begins merely because this proposal exists.
