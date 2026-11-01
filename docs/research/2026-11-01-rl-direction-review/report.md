# Research review for the focused RL direction

Research date: 2026-11-01. Agent-authored analysis for Dang Huy. The user authorized updating the main plan after independent verification. This report supports that revision; it does not record a training result or open a spending, license or scientific-experiment gate.

## Recommendation

Make **closed-loop post-training of driving action policies** the primary research path. Start with the released Alpamayo 1.5 expert in AlpaGym and study the relationship between its learning signal and independently measured driving outcomes. Keep the gate as the instrument. Finish S2 qualification, conduct one reproducible S4 baseline, and select one intervention from observed development failures.

RL fits Huy's stated interests in model learning more directly than simultaneously pursuing a student, a runtime, a device and a shadow-mode rig. The literature supports the importance of learning on the states a policy visits. It does **not** establish that direct RL is always the most economical improvement method, that this particular setup will improve, or that applying GRPO to an AV policy is novel.

The defensible initial question is:

> Under one pinned simulator and compute envelope, when does the shipped expert-training update improve held-out driving outcomes, and when transfer fails, does correcting one demonstrated mechanism change it?

The first deliverable can be an explained negative result. A positive contribution needs a reproducible behavioral effect, an informative ablation and a mechanism that survives independent evaluation. An operational defect report is useful engineering evidence, but it must not be relabelled an algorithmic discovery.

## Review method and evidence levels

Three independent research agents covered AV literature, RL foundations and NVIDIA's executable stack. The coordinating agent inspected adjacent VLA post-training papers, cross-checked critical code, synthesized the plan and requested adversarial review before adoption. The lane files retain their primary links, versions, locators and qualifications:

- [AV literature](av-literature.md): driving inputs, privileged information, action spaces, evaluation environments and transfer limits.
- [RL foundations](rl-foundations.md): policy gradients, diffusion/flow estimators, constraints, reward exploitation and statistical evaluation.
- [NVIDIA stack](nvidia-stack.md): actual dependency pins, online reward semantics, rollout freshness, training scope and runtime acceptance requirements.
- [Adjacent VLA studies](adjacent-vla-studies.md): CLEAR, Poutine and RefineDrive, with distinctions that prevent misleading comparisons.
- [Engineering and experiment specification](protocol-design.md): inputs, outputs, tests, falsifiers and owner decisions.
- [Verification record](verification.md): reviewers, findings, resolutions and document checks.

“Influential” here means a foundational method, an established benchmark, or a directly relevant research lineage. It is not a claimed citation-count ranking. Recent preprints are labelled as such; a preprint's reported result remains the authors' result. Neither a prominent venue nor several agents agreeing constitutes an independent replication.

| Evidence | What was established in this review | What remains outside it |
|---|---|---|
| Primary publication | Method, reported protocol, reported result and stated limits | Independent reproduction of the result |
| Pinned source inspection | Implemented computations and configuration at the inspected revision | Successful execution of the complete stack |
| Algebra and counterexamples | Conditional properties of a specified formula | All numerical/runtime cases or useful learning |
| Existing local evidence | S1 synthetic qualification and one completed synthetic Vast host check | S2 scientific qualification or any S4 training gain |
| Proposed acceptance test | The observation required to admit the next step | A test already passed |

The review searched AV RL, camera/VLA post-training, flow/diffusion RL, simulator validity and evaluation. It followed primary-paper references into earlier work and examined implementation dependencies rather than assuming a repository's default branch is the running dependency. This is a targeted decision review, not an exhaustive systematic survey or a proof that no related paper exists.

## What changes the research decision

### Direct RL has a strong rationale and serious competing baselines

Behavior cloning learns from supplied demonstrations; closed-loop learning can expose the policy to consequences and recovery states created by its own choices. But imitation on policy-visited states can also address distribution shift. The original [DAgger paper, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html), supplies a no-regret reduction under additional assumptions; it is not a reward-based RL algorithm or an unconditional driving guarantee. The review includes privileged RL teachers, World on Rails, Roach and OPTED alongside direct RL.

OPTED is especially relevant because it uses AlpaSim for sensor-policy experiments. Its privileged teacher and supervised student correction are an efficiency competitor to direct sensor-policy RL. Its much larger interaction-efficiency ratio belongs to a separate controlled experiment, not a demonstrated Alpamayo-versus-AlpaGym comparison. See the protocol-specific discussion in [the AV review](av-literature.md).

**Decision:** use direct expert RL because Huy wants to investigate its learning mechanism on the NVIDIA stack. Do not claim it is the best improvement-per-dollar path without an adequately controlled correction baseline. A future efficiency study must include a credible alternative and all label, simulator, training and evaluation cost.

### Closed-loop VLA RL is already an active field

CLEAR uses a frozen visual/language backbone with a learned residual waypoint policy and substantial simulation resources. Poutine uses preference-derived trajectory rewards in a logged-data setting. RefineDrive combines failure correction and GRPO under NAVSIM trajectory evaluation. These are different learning and evaluation regimes, not interchangeable demonstrations of a single technique. [Details and primary sources](adjacent-vla-studies.md).

**Decision:** remove an unconditional “first closed-loop RL result” headline. The novelty target is a narrow, evidenced mechanism or boundary of transfer, accompanied by a paired evaluation and artifact. Re-check related work at intervention selection and publication; the literature review is not a permanent novelty certificate.

### The training reward and the evaluator are different programs

The NVIDIA audit traces online rewards to the AlpaGym-pinned simulator. Raw online aggregates and offline reporting differ in warm-up masking, first-incident truncation, invalid-image handling and derived metrics. A reward name is insufficient to establish what happened to the ego vehicle or what time interval it covers. [Pinned source chain](nvidia-stack.md).

This creates three distinct possibilities when reward rises: useful driving improvement, optimization of a different but legitimate objective, or exploitation of a scoring defect. Those explanations need different responses. Merely replacing a coefficient cannot distinguish them.

**Decision:** make raw-event-to-online-reward-to-offline-score reconciliation a mandatory S4 entry artifact. Keep metric-vector logging and synthetic trajectories with events before warm-up, after the first incident, at the final step and in invalid intervals. Preserve failures; never make train and evaluation agree by silently dropping inconvenient intervals.

### Configuration labels do not enforce training correctness

The inspected Gym rollout filter warns about stale kept samples; the tolerance setting is not, by itself, a hard exclusion rule. Cosmos-RL dispatch behavior and actual recorded versions also matter. The root reviewer independently inspected this function. [Implementation](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/rollout_filter.py).

**Decision:** qualification asserts actual weight-version identity and likelihood parity before the first update. Preserve group membership and the point at which advantages are computed. Do not discard individual samples after normalization and assume the estimator remains unchanged. A failure produces a diagnostic stop, not a hidden change to the algorithm.

The wider lesson is that “PPO,” “GRPO” and “Flow-GRPO” name families. The actual density reduction, normalization, masks, replay weighting, clipping and reference policy determine the implemented estimator. The plan requires that derivation without assuming a general theorem automatically covers the implementation.

### Reasoning RL is a separate research lane

The shipped Gym path trains the continuous expert with visual/language weights frozen. The public open-loop reasoning recipe trains a different representation with different compute requirements. AlpaJudge can be a useful diagnostic, but its scores do not turn a frozen language model into a trained reasoner. Ground-truth future action information used for scoring must remain outside the policy observation path. [Implementation and input contracts](nvidia-stack.md).

**Decision:** reasoner training, learned-judge reward adaptation and joint language/action optimization wait until a separately designed study. A constant score within a sampled group provides no relative reward signal. A state-dependent judge can still influence expert learning indirectly, so the precise failure is conditional constancy, not that frozen judges are unusable in policy gradients.

### Evaluation power and seed variation can be the limiting resource

The existing gate has a conservative equal-road contract. Its synthetic noisy-equivalent controls were almost always INSUFFICIENT. A Gaussian planning calculation cannot override the actual operational method. Moreover, checkpoint uncertainty over roads is different from variation over independently trained models. Three runs are a useful minimum diagnostic requirement, not an automatic basis for a precise algorithm-level confidence interval. [S1 handoff](../../S1_HANDOFF.md), [statistical review](rl-foundations.md).

**Decision:** protect independent evaluation funds before training. Report each training run. Use a reviewed run-level or crossed design for algorithm claims, if one is feasible; otherwise state checkpoint-level evidence. Never treat three seeds times 441 scenes as 1,323 independent roads, and never call non-inferiority GO a proof of improvement.

## Ranked research directions

Ranking is an engineering judgment for this project, not a prediction of publication impact. “Start” means after the required stage admission.

| Direction | Potential contribution | Cost and confounds | Priority |
|---|---|---|---|
| Reward/evaluator semantics and outcome transfer | Show which event definition, masking or termination choice changes learning and why | First resolve implementation correctness; historical reward-alignment ideas are not novel by themselves | Start with the audit; intervene only on demonstrated mismatch |
| Learning-signal availability | Explain all-tied groups, action diversity and useful gradient per paid rollout | Group size/noise changes also change cost and potentially state distribution | Strong candidate if the baseline is signal-starved |
| Horizon and temporal credit | Explain short-training/long-evaluation failure; test one horizon or credit change | Driving and denoising time axes differ; changing both estimator and horizon confounds attribution | Next candidate after baseline correctness |
| Constraint-aware objectives | Separate progress optimization from explicit incident constraints | A scalar penalty is not a constraint guarantee; rare-event estimation and multipliers need resources | Later, separately designed |
| Judge-based reasoning/action learning | Ground explanations and link them to actions with independent tests | Wrong trainable path, privileged scoring inputs, domain shift and reward exploitation | Second research lane |
| Reactive traffic and simulator validity | Test whether an improvement survives a different interaction model | Changes the environment; requires another validity and seed audit | Later robustness extension |
| Privileged teacher and on-policy correction | Compare economical improvement strategies | Needs comparable student/model/labels and a fair total-cost baseline | Alternative path if direct RL is uneconomical; no automatic switch |

No intervention wins solely from this ranking. The first study selects one mechanism using training/development evidence, documents the selection, freezes the hypothesis and prediction, and then evaluates independently.

## Research sequence and stopping points

1. **S2 instrument qualification.** Establish actual data eligibility, sampling support, score extraction, reliability, validity, topology fit and all-in cost. Keep confirmations separate from failure mining.
2. **S4 correctness.** Prove small specified identities and pass known-answer tests; establish frozen parameters, likelihood/version parity, zero-update export parity and one finite, attributable update.
3. **S4 baseline.** One shipped reward arm with at least three independent training runs if the measured envelope funds them. Frozen teacher reference; development diagnostics; all runs retained.
4. **S4 intervention.** One change with an explicit falsifier. At least three runs per arm is the minimum planned replication, not a promise of precision. For a training-procedure claim, reserve fresh independent training runs after intervention selection, or qualify a selection-aware design. Reusing exploratory pilot checkpoints on new roads supports a conditional checkpoint comparison; independent roads alone do not remove selection on those runs' training randomness. Price fresh confirmation runs separately from exploratory pilots.
5. **S4 conclusion.** Supported improvement, supported regression, an explained failure, or INSUFFICIENT. Stop the focused project here before deciding whether student transfer, reasoning RL, hardware or broader scaling is justified.

The current budget can fund some evidence, but this review cannot establish that it funds the complete two-arm study. At the maximum $1,200 S2 expenditure, $1,300 remains under the $2,500 project cap. Training, confirmation, failed setup, storage and transfer all consume it. The proposed $650/$400/$250 reservation is planning arithmetic, not a quote or funding release. Avoid promising a calendar until Huy records weekly hours and measured throughput supports a forecast.

## What Huy should own and learn

The focused path increases Huy's time on policy distributions, objective design, exploration, estimator derivation and causal interpretation. Agents own measurement implementation, source and data lineage, orchestration, cost controls, test scaffolds and engineering defect diagnosis under D-018. Protected file entries remain unchanged.

Read in this order: decision and state-distribution basics; PPO/GAE; group-relative advantages; diffusion/flow policies and their two time axes; AV teacher-versus-direct-RL precedents; the exact Gym computation; reward exploitation and independent evaluation. The lane reviews provide specific readings and exercises. The skill is demonstrated by deriving what the code optimizes and predicting a failure before seeing its outcome, not by recalling paper names.

Before an experiment, Huy should be able to explain which parameters change, which stochastic choices generate alternatives, which reward variation ranks them, how the estimator assigns credit, what information is privileged, and what held-out observation would refute the proposed mechanism. Neither the agents' literature review nor agent-written tests constitute Huy's mastery evidence.

## Validation boundary

The research direction is supported as a focused, technically grounded program. Source-level dependencies and important failure modes have been identified. **End-to-end execution is not yet validated.** The real AlpaSim qualification, licensed assets, training topology, actual update behavior, statistical power and public-release permissions remain required evidence. The implementation specification names those gaps as tests and decisions so that a source review cannot accidentally become a claim of a completed experiment.
