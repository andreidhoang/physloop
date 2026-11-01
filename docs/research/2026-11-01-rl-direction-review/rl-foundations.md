# RL foundations, estimator scrutiny and evaluation evidence

**Author:** Codex research reviewer, 2026-11-01. Agent-authored literature analysis and recommendations; not Huy's mastery evidence, scientific prediction or experiment registration.

**Purpose and scope:** evidence for the proposed focused S4/G9 teacher-RL direction and the S2/G1–G3 measurement prerequisites. This note elaborates the S4 estimator/reward-hacking audit and the evidence needed to change the plan. It does not admit training, change protected inference code, define a scientific margin, or authorize spending or asset access. The owner requested verification before the plan/context update. Adoption and execution remain separate decisions.

**How sources were checked:** primary full texts, selected method/theorem/experiment sections, and official proceedings. Locators below refer to the linked version; a preprint's numbering need not match its conference revision. This is a selected learning chain, not an exhaustive survey or a citation-count ranking. “Foundational” means a result supplies a mathematical ingredient used by later methods; “directly relevant” means it studies a policy parameterization or experimental problem close to ours. Conference publication is not a guarantee that a claim transfers to driving.

## 1. The scientific choice this evidence supports

The defensible research question is about **when a specific, audited training procedure improves independently measured closed-loop behavior at an affordable cost**, and when reward gains instead reflect insufficient exploration, poor temporal credit, proxy exploitation or a measurement mismatch. “Use RL on a VLA” is an implementation category, not a sufficiently novel claim.

Start by distinguishing five objects:

| Object | Required meaning in PhysLoop | What it does not establish |
|---|---|---|
| Training reward | The exact quantity consumed by the optimizer, including aggregation, masks and episode termination | The held-out evaluation endpoint |
| Advantage estimator | How sampled outcomes become weights on log-probability gradients | A proof that the reward is correct |
| Policy distribution | The sampled random variable and its actual density, including latent denoising transitions | The marginal likelihood of the final action merely because conditional densities are available |
| Evaluation protocol | Fixed checkpoints, roads, traffic, renderer, scorer, validity convention and uncertainty treatment | Real-world validity or causal reasoning ability |
| Research result | An effect and mechanism supported by controls, uncertainty and honest limits | The maximum training return or the best of many unreported seeds |

RL is a reasonable research focus because the released stack exposes testable estimator, exploration, reward and train/evaluation mismatches. Whether it is the best affordable empirical direction is conditional on actual throughput, fit, reward signal and evaluation power. Neither a paper's GPU count nor our completed synthetic host test answers those questions.

## 2. Primary reading ledger

### F1. Policy gradients: the causal path does not require differentiating the reward

**Sutton, McAllester, Singh and Mansour, _Policy Gradient Methods for Reinforcement Learning with Function Approximation_, NeurIPS 1999.** Read §1, Theorem 1; §2, Theorem 2; §3's state-dependent baseline discussion. [Official proceedings PDF](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf).

The policy-gradient theorem expresses the gradient using policy derivatives and action values, without differentiating the state-visitation distribution. Compatible function approximation supplies a conditional result, not permission to substitute an arbitrary critic without error. A state-only baseline can reduce variance without changing the expected gradient under the stated sampling assumptions.

**Why read it:** it explains why a detached simulator or frozen judge can train a stochastic policy. The necessary path is sampled output → outcome variation → advantage → log probability of the trainable policy. A judge need not be differentiable. A reward that is identical for every candidate within a centered group supplies no reward-ranking signal to that group.

**Limit:** the result does not establish global convergence of an arbitrary neural actor-critic implementation or validate a learned reward.

### F2. GAE: temporal credit is a bias–variance decision

**Schulman et al., _High-Dimensional Continuous Control Using Generalized Advantage Estimation_, ICLR 2016; inspected arXiv v6.** Read §2–3, especially Eqs. 9–18. [Versioned full text](https://arxiv.org/html/1506.02438v6).

The estimator combines temporal-difference residuals: `delta_t = r_t + gamma V(s_{t+1}) - V(s_t)` and `A_t = sum_l (gamma lambda)^l delta_{t+l}`. At lambda zero it is one-step TD; at lambda one the full-return form removes the value-approximation bias relative to the discounted gradient under the paper's conditions. Discounting and lambda play different roles: reducing gamma changes the objective relative to undiscounted return, while reducing lambda can add value-approximation bias.

**Why read it:** delayed collision outcomes, short training episodes and action chunks make credit assignment concrete. A single episode score attached to every driving tick is not automatically GAE.

**Limit:** finite-horizon truncation, incorrect bootstrap masks and inaccurate values can alter the estimator. GAE cannot discover a useful action that exploration never samples.

### F3. TRPO: distinguish the bound from its approximation

**Schulman et al., _Trust Region Policy Optimization_, ICML 2015.** Read §2–4, Theorem 1 and the practical average-KL constraint. [Proceedings](https://proceedings.mlr.press/v37/schulman15.html), [PDF](https://proceedings.mlr.press/v37/schulman15.pdf).

The theoretical improvement bound uses a worst-state divergence and exact quantities. The practical algorithm substitutes a sampled surrogate and an average-KL trust region, with additional optimization approximations. The practical average and the theorem's maximum are different objects.

**Why read it:** it supplies the local-policy-update rationale behind PPO and helps separate a theoretical guarantee from an engineering approximation.

**Limit:** do not describe sampled neural TRPO, PPO, or AlpaGym as guaranteeing monotonic true return, much less collision reduction. A small average divergence can coexist with a large change in a rare state relevant to the driving endpoint.

### F4. PPO: clipping changes the objective; it is not a projection

**Schulman et al., _Proximal Policy Optimization Algorithms_, arXiv:1707.06347 (2017).** Read §2.2, §3 Eq. 7 and §5 Algorithm 1. [Paper PDF](https://arxiv.org/pdf/1707.06347).

The clipped objective takes the smaller of `ratio × advantage` and its clipped counterpart. It is pessimistic relative to the unclipped surrogate at that sample; it does not hard-constrain the ratio, parameter step or KL. Shared parameters and repeated optimization can move probabilities beyond the clip interval.

**Why read it:** this is the practical first-order policy-gradient reference inherited by many later methods, including GRPO and diffusion-policy fine-tuning. Understanding ratio and advantage signs is more useful than memorizing a default clip value.

**Limit:** empirical robustness on the paper's tasks is not a universal optimizer guarantee. Log-ratio clamping, ratio clipping, gradient clipping and a KL penalty are separate operations and should be audited separately.

### F5. Implementation Matters: the baseline is more than the algorithm name

**Engstrom et al., _Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO_, inspected arXiv v1, 2020.** Read §3–5 and Appendix A.2. [Versioned full text](https://arxiv.org/html/2005.12729v1), [authors' code](https://github.com/MadryLab/implementation-matters). This entry cites the inspected preprint rather than inferring a venue from an older OpenReview PDF header.

The study separates core objectives from implementation additions: value clipping, reward scaling, initialization, learning-rate schedules, observation transforms and gradient clipping. Its experiments show large effects on both reward and effective trust-region behavior in the tested locomotion settings.

**Why read it:** changing a reward while also changing normalization, sampler noise, rollout length, optimizer or checkpoint selection does not isolate a reward mechanism.

**Limit:** the paper's relative effect sizes are not estimates for Alpamayo. Its lesson is to pin and expose implementation choices, not to assume the same optimization wins everywhere.

### F6. Original GRPO: group comparison, originally for mathematical text

**Shao et al., _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_, arXiv:2402.03300v3, 2024.** Read §4.1.1–4.1.4, Eqs. 3–4 and Algorithm 1. [Versioned full text](https://arxiv.org/html/2402.03300v3), [official repository](https://github.com/deepseek-ai/DeepSeek-Math).

GRPO replaces a learned value baseline with rewards from a group of outputs for the same question. The outcome-supervised variant gives each output a group-standardized reward advantage shared across its tokens, with clipped token ratios and a reference-policy KL term. The paper also discusses process supervision and iterative variants. GRPO was not first introduced by DeepSeek-R1, and its original formulation is not inherently limited to rule-verified rewards.

**Why read it:** it explains the value-model memory tradeoff and why group composition, reward variation and normalization matter.

**Limit:** mathematical answer generation supplies neither a driving-time critic nor a closed-loop control guarantee. Same-question grouping must be translated deliberately to same-initial-condition rollouts, not arbitrary different scenes.

### F7. DPPO: explicitly model both environment and denoising time

**Ren et al., _Diffusion Policy Policy Optimization_, ICLR 2025; inspected arXiv v3, 2024-12-09.** Read §4.1–4.3, §5.4–6 and Appendix B. [Versioned text](https://arxiv.org/html/2409.00588v3), [official conference record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html).

DPPO uses an augmented MDP with environment steps and denoising steps. Conditional Gaussian denoising transitions provide tractable policy-gradient terms. Its advantage construction combines environment-level information with denoising weighting; Appendix B uses environment-step GAE. The study investigates control/manipulation benchmarks and a specific furniture-assembly sim-to-real transfer, rather than driving.

**Why read it:** it is a direct bridge from PPO to multi-step generative action policies and a reference against which an adaptation must list its changes.

**Limit:** empirical structured exploration and transfer on these tasks do not establish AV transfer. AlpaGym's group episode advantages and geometric-mean ratio should not be called exact DPPO without an equivalence argument. Conditional transition densities are not automatically the marginal final-action density.

### F8. Flow-GRPO: useful sampler mathematics, image-generation evidence

**Liu et al., _Flow-GRPO: Training Flow Matching Models via Online RL_, NeurIPS 2025.** Inspected v3 and checked the corresponding method and resource sections in v5, 2025-10-27. Read §3–4, §5.3, Appendix A and B.4. [v5 full text](https://arxiv.org/html/2505.05470v5), [conference record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a10c46572628d58cb44fb705f25cbbf-Abstract-Conference.html).

The paper converts an ODE sampler to a stochastic sampler with a continuous-time marginal-preservation derivation, and applies group-relative updates to generated images. “Online” here includes drawing new images and scoring them. It does not mean interaction with a physical control environment. Its experiments use text-to-image tasks; Appendix B.4 reports 24 A800 GPUs.

**Why read it:** it exposes the role of stochastic denoising transitions, group-relative reward and reduced training denoising steps.

**Limit:** adding arbitrary noise does not inherit the derivation. Marginal preservation does not imply identical sample paths, bitwise equivalence or exact finite-step learned-sampler behavior. Its KL ablations are task-specific evidence, not a guarantee against reward exploitation.

### F9. ReinFlow: a closer flow-policy control precedent

**Zhang et al., _ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning_, NeurIPS 2025.** Read §4.1–4.4, Algorithms 1–2, §5 and Appendix D.2. [Official PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/98d6f928497d9eaad9f320fb2db3040d-Paper-Conference.pdf).

ReinFlow adds learned Gaussian noise to flow-policy transitions and optimizes the resulting discrete denoising-path distribution. Its practical implementation uses PPO and can use GAE. Experiments cover simulated locomotion and manipulation, including pixel observations and few denoising steps. This is a more direct continuous-control precedent than image-only Flow-GRPO.

**Why read it:** “apply PPO to a stochastic flow action policy” is already an established methodological direction. A study must identify its additional mechanism or evidence contribution.

**Limit:** tractability concerns the augmented denoising path, not automatically the terminal action's marginal. The paper removes the learned noise network after fine-tuning; this training/evaluation sampler distinction must be understood. Its robot-scale cost and performance comparisons do not forecast Alpamayo/renderer cost or establish driving outcomes.

### F10. Reward shaping: changing learning speed versus changing the task

**Ng, Harada and Russell, _Policy invariance under reward transformations: Theory and application to reward shaping_, ICML 1999.** Read §2–3, Theorem 1 and Corollary 2. [Author-hosted paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

Potential-based shaping uses `F(s,a,s') = gamma Phi(s') - Phi(s)` and preserves optimal policies under the paper's assumptions. The necessity result is about a guarantee across arbitrary MDPs without additional knowledge, not a claim that every other shaping reward fails on every task. Boundedness and absorbing/terminal conventions matter.

**Why read it:** dense progress or comfort rewards may deliberately change the desired tradeoff, rather than merely accelerate learning of the original endpoint.

**Limit:** do not call an arbitrary distance penalty policy-invariant. A per-step offset in a variable-length episode can change duration incentives. Finite truncation leaves a terminal potential term unless handled consistently. This theorem does not validate the chosen base reward.

### F11. Reward overoptimization: independent assessment is necessary

**Gao, Schulman and Hilton, _Scaling Laws for Reward Model Overoptimization_, ICML 2023.** Read §2, §3.6 and §4.5. [Official PDF](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf).

In language-model experiments, a proxy reward can improve while a stronger “gold” reward deteriorates. That gold measure is itself a learned model, not perfect human truth. The study compares PPO and best-of-N and examines optimization pressure and reward-model size. Its KL findings do not support a universal claim that KL prevents overoptimization.

**Why read it:** a learned judge should not simultaneously be the sole training objective, checkpoint selector and final evidence of improvement.

**Limit:** this does not prove AlpaJudge will be hacked in a particular way, nor quantify that risk in driving. It motivates independent behavioral outcomes and deliberate controls. Expert agreement, evaluator agreement and actual closed-loop task performance are separate quantities.

### F12. CPO: expected constraints are different from fixed penalties

**Achiam et al., _Constrained Policy Optimization_, ICML 2017.** Read §4–6, Eqs. 10–14 and Proposition 2. [Proceedings](https://proceedings.mlr.press/v70/achiam17a.html), [PDF](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf).

A constrained MDP optimizes expected return subject to expected discounted cost limits. CPO derives conservative update bounds, then uses sampled and local approximations in practice. Proposition 2 includes an error term; §6 explicitly discusses sampling/approximation error and infeasibility recovery.

**Why read it:** `reward - lambda × collision_cost` defines a scalar tradeoff. A chosen fixed lambda does not demonstrate that a collision constraint is satisfied. Adaptive Lagrangian optimization and CPO are related alternatives but not the same algorithm.

**Limit:** expected-cost constraints are not per-trajectory guarantees. They do not certify no collision, safe deployment or compliance. A constraint method adds estimator, feasibility and tuning requirements; it should not become a third arm merely because it sounds safer.

### F13. Offline RL: define the available information before picking a method

**Levine, Kumar, Tucker and Fu, _Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems_, arXiv:2005.01643v3, 2020.** Read §2.2–2.4 and §3–4 selectively. [Versioned full text](https://arxiv.org/html/2005.01643v3).

Offline RL learns from a fixed dataset of transitions without collecting new environment transitions. Off-policy methods can still collect new data, so the terms are not interchangeable. The central problem is distribution shift when the learned policy asks counterfactual questions not well supported by the data.

**Why read it:** prerecorded videos, scene reconstructions or reference trajectories do not automatically constitute a suitable offline RL transition dataset. Generating new candidates for a fixed prompt and scoring them can be online conditional-generation optimization while remaining open-loop with respect to driving dynamics.

**Limit:** offline RL cannot repair missing exploration by collecting new transitions. A favorable offline value estimate does not by itself qualify the policy under a closed-loop driving protocol.

### F14. CQL: conservative value estimates are an alternative, not a free guarantee

**Kumar et al., _Conservative Q-Learning for Offline Reinforcement Learning_, NeurIPS 2020; inspected arXiv v3.** Read §3.1–3.3 and §4. [Versioned full text](https://arxiv.org/html/2006.04779v3).

CQL addresses overestimated values for actions outside the data by regularizing learned Q values. Its lower-bound and improvement results depend on stated assumptions, including the form of the estimator and policy updates. The guarantees concern values/returns in that model, not a collision-free vehicle.

**Why read it:** it makes the danger of maximizing an imperfect evaluator concrete. It is a useful branch if a suitable logged transition corpus exists.

**Limit:** introducing a critic, dataset-support problem and flow-policy extraction is additional work. CQL is not a drop-in replacement for AlpaGym's group-relative update or a demonstrated cheaper route for this repository.

### F15. IQL: avoid unseen-action queries during value learning

**Kostrikov, Nair and Levine, _Offline Reinforcement Learning with Implicit Q-Learning_, ICLR 2022; inspected arXiv v1, 2021-10-12.** Read §4.1–4.4, Eqs. 5–7 and Algorithm 1. [Paper PDF](https://arxiv.org/pdf/2110.06169).

IQL learns an upper-expectile state value from dataset actions, backs it up into a Q function and extracts a policy with advantage-weighted behavioral cloning. Separating the state-value expectile from dynamics averaging avoids treating a lucky transition as evidence of an unusually good action.

**Why read it:** it offers a concrete alternative to online exploration when data support is credible, and connects value learning to weighted imitation.

**Limit:** avoiding out-of-dataset action queries in the training objective does not guarantee that a deployed neural policy stays in-distribution. Its benchmark results do not establish a workable Alpamayo integration or independent driving gains.

### F16. Deep RL That Matters: implementation and training randomness are evidence dimensions

**Henderson et al., _Deep Reinforcement Learning That Matters_, AAAI 2018; inspected arXiv paper.** Read “Random Seeds and Trials,” “Environments,” and “Discussion and Conclusion.” [Paper PDF](https://arxiv.org/pdf/1709.06560).

The paper documents sensitivity to random seeds, implementation, reward scaling and hyperparameters. It illustrates how apparently comparable sets of few runs can produce misleadingly different conclusions. It does not prescribe a universal sufficient trial count.

**Why read it:** one successful run is a checkpoint result, not a reliable estimate of the training procedure's performance distribution.

**Limit:** its historical statistical examples should not be copied without checking assumptions. Neither overlapping error bars nor a nonsignificant difference establish equivalence. Report unsuccessful runs and the selection procedure, not just a favorable mean.

### F17. Rliable: useful evaluation principles, not a ready-made road-cluster test

**Agarwal et al., _Deep Reinforcement Learning at the Edge of the Statistical Precipice_, NeurIPS 2021.** Read §4.1–4.3, Figure 6 and Table 1. [Official PDF](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf), [authors' project](https://agarwl.github.io/rliable/).

The paper recommends uncertainty intervals, performance profiles and robust summaries for multi-task RL benchmarks. Its stratified bootstrap resamples runs independently within tasks. Figure 6 explicitly shows undercoverage with three runs in the studied setting; improved coverage at ten runs there is not a universal ten-seed rule. Probability of improvement does not convey effect size.

**Why read it:** a bootstrap does not create information absent from a few training runs. A robust aggregate can supplement a benchmark report.

**Limit:** PhysLoop's shared roads crossed with training runs are not automatically the paper's sampling design. IQM trims distribution tails; it must not replace collision endpoints, adverse strata or the gate's explicit estimand.

## 3. Derivations and implementation questions specific to this project

The following are reviewer analysis and proposed audit requirements, not a new estimator design or an adopted scientific experiment.

### 3.1 Group normalization can erase the intended intervention

Write the actual implemented expression, including whether it is `(R_i - mean(R))/(std(R) + epsilon)` or uses `sqrt(var + epsilon)`, which variance convention it uses, and how invalid candidates are handled.

For the first expression, changing all group rewards to `a R_i + b`, with `a > 0`, gives `(R_i - mean(R))/(std(R) + epsilon/a)`. Therefore translation cancels exactly in real arithmetic, while scale invariance is exact only when epsilon is absent or scaled accordingly. It is approximate when epsilon is negligible. Floating-point behavior adds another qualification. A constant added equally to all candidates cancels; a context-only judge score cannot rank that group's actions.

All equal rewards imply a zero numerator. If the computed standard deviation is finite and zero, a positive stabilizer yields zero reward advantage; without it, division is undefined. A size-one group with an unbiased sample-variance implementation can instead produce NaN, which adding epsilon does not repair. This is not a claim of zero total parameter update: KL, entropy, weight decay, optimizer state or other losses may still act. All-negative but unequal rewards can supply a valid relative signal.

Changing one reward coefficient can affect learning when other components vary, clipping/thresholds change, epsilon matters, or normalization/reduction is different. Changing a lone binary penalty's magnitude under otherwise identical nondegenerate group normalization may barely change its reward gradient. Measure per-component within-group variation before selecting that experiment.

For a common context and independent Bernoulli candidate successes of probability `p`, the probability that a size-`G` group contains both successes and failures is `1 - p^G - (1-p)^G`. This is an illustrative calculation, not an empirical model of correlated driving rollouts. It demonstrates why group size two can yield little binary reward contrast when success is near zero or one. Larger groups cost more; they are not automatically the best use of a fixed compute budget.

### 3.2 Two time axes and three possible probability objects

Use separate symbols for driving tick `t` and denoising transition `k`. A conditional transition density `p_theta(z_{k-1} | z_k, context_t)` is distinct from the joint denoising path density and from the marginal distribution of the terminal action. Integrating out the latent path is generally different from evaluating its sampled conditional factors.

For independent scalar Gaussian coordinates, joint log density is the **sum** of coordinate log densities. Replacing a sum of log-ratio terms with their mean yields a geometric-mean ratio. If there are `D` terms, the joint ratio is the mean-ratio raised to `D`; clipping those two ratios at the same numerical interval defines different objectives. A normalization may be a useful engineering choice, but its meaning must be explicit. A clamp before exponentiation changes the objective again.

DPPO's augmented process, Flow-GRPO's per-denoising-step objective, ReinFlow's joint-path construction and AlpaGym's current reduction are related but cannot be treated as equivalent merely because each uses Gaussian log probabilities and a clipped update. The correct reference is the **actual sampled process and implemented loss**, not an algorithm label.

For credit assignment, state which environmental reward/return is attached to each denoising transition, whether there is a driving-time critic, and whether only the executed prefix of an action chunk affects the environment. The predicted action horizon, executed control horizon and total rollout duration are separate quantities. Training at 5 Hz for a short episode and evaluating at 10 Hz for a long episode changes exposure, discount interpretation and the events the policy can learn from.

### 3.3 Staleness and parity are observed properties

A config value requesting zero staleness is insufficient if the code only warns about old rollouts. The [NVIDIA source audit](nvidia-stack.md) records that behavior and its exact pinned function. Log the rollout worker's actual parameter/version identity, the trainer's old-policy identity and the update that consumes each sample. Define whether mismatches are rejected, regenerated or deliberately corrected. A quiet warning log is not evidence that weights matched.

At zero update, replay exactly the same latent transitions with the same masks, noise scale, scheduler, conditioning and numerical settings. The intended ratio should be one under the chosen reduction. Distinguish rollout/learner precision differences from mathematical errors. Huy must define the meaningful tolerance before the test; do not infer it from a convenient observed error.

Parity is necessary, not sufficient: two implementations can agree on the same incorrect likelihood or reward. Include independent small known-answer distributions, ratio-sign controls and a deliberately mismatched policy identity.

### 3.4 Reward provenance is not evaluation provenance

The [NVIDIA implementation audit](nvidia-stack.md) in this review found that online training aggregates raw metrics using choices such as maximum contact indicators and last progress values, while the challenge/offline evaluator has masks, truncation and derived at-fault logic. It also identified an older pinned simulator dependency. These findings must be traced to that audit's exact files/commits before implementation; this foundations note is not an independent source-line verification of them.

Consequently, a training field called “collision” or “progress” cannot be assumed to equal the gate endpoint. Build a field-by-field mapping: raw source, aggregation window, active mask, contact definition, timeout behavior, invalid-rollout convention, unit, scale and final reward contribution. Test cases should include a rear contact, multiple contact types, a validity exit before a later event, a stationary ground-truth tail, and an infrastructure failure. Differences can be intentional, but must be visible.

## 4. The evaluation design must answer the claim actually being made

### 4.1 Fixed checkpoint and training procedure are different estimands

One question is whether **this saved checkpoint** improves on the frozen teacher over the target road population. Another is whether **this training procedure**, with its randomness and checkpoint-selection rule, usually produces better policies. The latter requires independent training runs, not merely more evaluations of one run.

Let `D[j,r]` denote the paired candidate-minus-reference road outcome for training run `j` and road `r`, after the declared within-road scene/replicate aggregation. Runs share roads; scenes may share a road; simulator replicates may share all those inputs. These observations are not a flat independent sample. Three training runs on 441 scenes do not become 1,323 independent training runs or independent roads.

Keep the conditional per-checkpoint verdict and a separate algorithm-level summary. A proposed crossed/hierarchical inference method must preserve road pairing and account for independent training-run variation. It needs its own assumptions, tests and qualification; this note does not substitute a new bootstrap into the protected gate. With only three runs, display each run and treat uncertainty about the training distribution as weak. The number three is an initial replication floor, not a power or coverage guarantee.

Simulation replicates and training seeds must be named separately. A matching integer seed is not evidence of common-random-number coupling when the driver ignores that seed or diverging control paths consume randomness differently. Preserve the actual seed mode and divergence diagnostics.

### 4.2 Development, selection and confirmation

Choose the first mechanism using development data and feasibility measurements. That adaptive choice makes the development comparison exploratory. Freeze the selected arm, training budget, checkpoint-selection rule, endpoints, margins, exclusions and inferential procedure **before** examining independent confirmation outcomes.

Independent confirmation must match both sources of randomness in the claim. For a training-procedure claim, prefer fresh independent training runs after intervention selection, as well as independent roads; otherwise use an explicitly selection-aware design. Reusing the exploratory baseline runs on new roads can support a conditional checkpoint comparison, but new roads alone do not remove selection on those runs' training randomness.

Use road/source lineage to separate training, tuning and confirmation. Different clip identifiers from the same road or source sequence do not automatically provide independence. “Held out from this RL run” does not imply “unseen in foundation-model training”; keep both contamination labels.

Repeatedly evaluating saved checkpoints on the confirmation set and picking the winner consumes that set for selection. Options are a fixed checkpoint chosen by a prior rule, selection on a development set followed by untouched confirmation, or a separately qualified sequential/selection-aware design. Do not infer permission for repeated peeking from the existence of S1 group-sequential code.

### 4.3 Endpoint and power discipline

Report the raw collision, progress, off-road and validity outcomes alongside training reward. A composite reward can rise by improving an easy component while worsening a rare component. Preserve per-stratum outcomes and the registered equal-road estimand. A scene-weighted visualization may be useful but should be labelled separately.

For rare outcomes, an absence of observed regression does not establish non-inferiority. Use S2's independent base-rate/discordance/variance evidence to determine whether the proposed confirmation can answer the desired effect. A claim may remain INSUFFICIENT even when the mean reward is substantially higher. Do not relax a scientific margin after seeing the result to obtain GO.

Evaluate failure mechanisms with labels independent of the reward implementation where possible. Diagnostics can include stall duration, distance travelled, stationary-lead behavior, repeated steering oscillation, contacts by type and validity exits. The diagnostic name alone does not validate its measurement, and association with reward is not causal proof of a policy's reasoning.

## 5. Fair controls and realistic compute comparisons

**Minimum controls for the first empirical study, subject to admission:** the pinned released teacher; a zero-update load/export path; the reproduced shipped training arm; every independent training run; and a fixed independent evaluator. The no-training reference is necessary even if a prior paper reports a teacher score, because the protocol may differ.

Changing one factor means all other consequential settings remain pinned: starting checkpoint, trainable parameters, data/scene sampler, group size, simulator, renderer, traffic, episode horizon, action frequency, reward aggregation, optimizer, normalization, stochastic sampler, KL reference, update count or budget stopping rule, and selection method. If a bug fix is necessary, apply it to the common baseline before the research contrast; do not market a broken-versus-correct implementation comparison as the superiority of a new research algorithm.

Predeclare the comparison budget. Equal optimizer steps can mean unequal rendered frames, generated candidates, denoising transitions and GPU-hours. Equal successful rollouts can hide wasted failures. Report at least total compute/time, environment transitions, accepted and rejected candidates, updates, denoising steps, evaluation cost and unsuccessful/aborted runs. Hardware edition and topology matter. Hyperparameter search and checkpoint evaluation are part of the method's cost, not free overhead to omit.

A primary fixed-compute comparison can answer affordability; a secondary fixed-sample comparison can help explain learning efficiency if it is affordable and predeclared. Neither is universally the right primary estimand. Reporting both after choosing whichever favors the method is not a fair design.

Do not borrow Flow-GRPO's image training scale or another AV paper's reported wall time as our forecast. CLEAR's reported large multi-GPU study and render-free privileged alternatives such as OPTED belong in the closest-work comparison. Their exact claims and costs are reviewed in the AV lane. They imply that neither “frozen vision/LLM with RL” nor “cheap closed-loop policy improvement” is an unoccupied category. A direct full-scale reproduction may be unaffordable; a precisely scoped mechanism comparison can still be useful.

## 6. Prioritized learning path and concrete exercises

The exercises are proposed learning artifacts. They are not claims that Huy completed them. CPU known-answer work and paper derivations can precede any paid model run; new model loss/estimator design remains Huy's scientific work.

| Order | Read | Produce before proceeding | What a correct artifact must explain |
|---|---|---|---|
| 1 | F1, then the problem's actual policy/reward interface | A one-page random-variable and causal-path diagram | Which parameters change; what is sampled; where reward variation enters; why a detached judge can work and a group-constant reward cannot rank candidates |
| 2 | F2–F4 | Hand-computed tiny-MDP advantages and ratio-sign cases | Return versus TD versus GAE; terminated versus truncated masks; positive/negative advantages; clipping versus an actual constraint |
| 3 | F5–F6 | A complete estimator specification and group table | Sum/mean reductions, epsilon, sample standard deviation convention, group size one, equal rewards, negative rewards, affine changes, KL-only updates |
| 4 | F7–F9 | A two-time-axis density derivation | Conditional Gaussian factors versus joint paths versus marginal final actions; sampler support; finite discretization; implemented AlpaGym deviations |
| 5 | F10–F12 | A reward-to-endpoint map and two counterexamples | Why a progress proxy can change the task; a stalling optimum; a scalar penalty that fails an expected-cost limit; the assumptions of a potential-based alternative |
| 6 | F16–F17, reread before claiming results | A sampling/selection diagram and mock report | Fixed checkpoint versus training algorithm; training runs crossed with roads; actual units of independence; all runs shown; confirmation remains untouched |
| 7 | F13–F15, conditional branch | A data-support inventory and alternative-method decision | Whether transitions/rewards/actions exist; what a critic must extrapolate; whether offline RL reduces total work for this exact data/model stack |
| 8 | Closest AV work and the pinned NVIDIA source audit | A nearest-neighbor contribution table | What the work adds beyond existing AV RL, existing flow-policy RL and upstream's own implementation |

Useful deterministic counterexamples include: a group of identical failure rewards; a constant context score plus changing action rewards; a tiny nonzero standard deviation comparable to epsilon; two Gaussian coordinates whose mean-ratio and joint-ratio clip decisions differ; a delayed failure after the training horizon; a terminal versus time-limit bootstrap; a stale rollout whose initial configuration says zero staleness; and a successful-looking aggregate produced solely by selecting the best training run. They test concepts before introducing expensive simulator noise.

## 7. Review checklist for the integrated plan

1. **Question and novelty:** names one mechanism and its closest existing alternatives; contains no unsupported “first,” “best,” or citation ranking. A narrower negative result is acceptable.
2. **Identity:** records trainable/frozen modules, starting and final weight hashes, data lineage, exact code/dependencies, stochastic sampler and actual rollout/learner policy versions.
3. **Objective:** distinguishes training reward from held-out endpoints; specifies units, aggregation, masking, terminal/truncated handling, horizon and action execution frequency.
4. **Estimator:** identifies the sampled density, every reduction and clamp, advantage definition, epsilon, reference update and staleness enforcement. Parity and independent known answers precede long training.
5. **Exploration:** reports group composition, duplicate actions/trajectories, component-wise within-group variance, degenerate-group fraction, reward noise and informative groups per unit compute.
6. **Credit:** separates environment time from denoising time; distinguishes an episode scalar from GAE; explains which actions can affect delayed outcomes within the horizon.
7. **One-variable contrast:** preserves the common baseline and identifies any unavoidable co-change. Engineering repairs are common prerequisites, not hidden treatment differences.
8. **Selection:** distinguishes exploration/tuning from confirmation; freezes the checkpoint rule and independent evaluation before examining confirmation outcomes.
9. **Uncertainty:** separates training randomness, road sampling and simulator replicates; treats shared roads as paired clusters; does not present a three-seed bootstrap as automatically calibrated.
10. **Power and claims:** predeclares meaningful effects/margins; permits INSUFFICIENT; keeps invalid/out-of-envelope cases visible; never promotes simulator reward, judge preference or a constrained-RL bound to a real-world safety claim.
11. **Compute:** includes rollout/rendering, failed work, training, selection/evaluation and search costs; labels matched compute versus matched samples; uses measured feasibility before promising a second arm.
12. **Authority:** preserves Huy's scientific definitions/predictions, file locks, stage admission and concrete rental/license approval. A literature-based recommendation is not execution authority.

## 8. Recommended first decision after the audit

Retain expert-only closed-loop RL as the focused candidate direction, with a **single reproduced shipped arm against the frozen teacher** as the initial feasibility and signal study. Its purpose is to establish whether a correct, affordable learning signal exists under a measured protocol. It is not sufficient by itself for a strong algorithmic novelty claim.

Select a one-variable second experiment only if independent development evidence supports a specific bottleneck: sparse within-group variation, train/evaluation horizon mismatch, a validated reward/endpoint mismatch, or a measured exploration failure. A compute-efficient privileged-policy/distillation route is a real alternative to compare conceptually, not an extra implementation to add automatically. If the budget cannot support independent confirmation, narrow the claim to an engineering reproduction/diagnosis or measured negative result; do not replace confirmation with the training reward.

An AlpaJudge reasoning reward remains a separate research path unless the trainable variables, candidate-dependent inputs and validity of its adapted input distribution are demonstrated. A frozen judge can provide a policy-gradient reward; a frozen reasoning output shared by all expert candidates cannot provide action-specific credit by itself. Improvements in judged reasoning require their own independent assessment and do not establish causal reasoning or closed-loop driving improvement.
