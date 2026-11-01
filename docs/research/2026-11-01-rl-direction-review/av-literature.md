# AV literature review for the proposed RL-primary direction

Independent agent review, 2026-11-01. Documentation only. This review informs P-031; it does not adopt a scientific protocol, prediction, estimator, rental or model change. The existing S2 qualification dependency, owner decisions and license/publication gates remain. Sources below are original papers, conference proceedings and author-maintained resources. No experiment was replicated for this review.

## Assessment

An RL-primary research path is defensible as a specialization in sequential decision-making and policy improvement. The literature does not establish that direct sensor-policy RL is always the cheapest or strongest way to improve an existing driving model. Privileged-teacher imitation, on-policy correction, latent-model RL and direct sensor-policy RL are distinct approaches with different assumptions and costs.

P-031 should be a mechanism study on a pinned action expert, with independent outcome evaluation. Merely running closed-loop RL, adding a collision penalty, using a world model, or transferring an RL teacher to a sensor student is established work. A useful contribution needs a reproducible failure, a specific explanation, a controlled intervention and a falsifier. A carefully scoped negative result can satisfy that research objective.

“Influential” below means foundational to a relevant method or evaluation lineage, or directly consequential to this project's comparison set. It is not a citation-count ranking. Older peer-reviewed works and recent unreviewed preprints are identified separately. This is a selected literature review, not an exhaustive novelty search.

## Primary-paper ledger

### 1. CARLA: An Open Urban Driving Simulator

**Status/source:** CoRL 2017, PMLR 78, proceedings version. [Paper](https://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf), especially §§2–5 and supplement §S.2.3.

- **Method and interface:** A benchmark/simulator paper comparing modular, imitation and A3C agents. The RL agent observes camera frames and measurements/navigation commands and acts through vehicle controls. It learns online from simulator rewards, without expert driving traces; this is not a teacher/student VLA.
- **Evidence:** Controlled navigation tasks distinguish town/weather generalization and include a dynamic-obstacle condition. The original tasks permit ignoring traffic lights and speed limits. The reward mixes progress/speed with collision and road-violation terms.
- **Why read / limit / open issue:** The foundational controlled-benchmark example shows why environment version and task definition belong to every result. Its original benchmark cannot certify modern interactive driving or real-world behavior. Transfer the protocol discipline, not historical leaderboard claims.

### 2. Learning by Cheating (LBC)

**Status/source:** CoRL 2019; PMLR 100 publication in 2020. [Proceedings](https://proceedings.mlr.press/v100/chen20a.html), [paper](https://proceedings.mlr.press/v100/chen20a/chen20a.pdf), §§2–3 and the simulator-version discussion.

- **Method and interface:** Two-stage imitation, not RL. A privileged BEV/map/actor-state teacher first imitates expert demonstrations. A monocular RGB/speed/navigation student learns waypoint outputs, executed by a PID controller, including supervision on states reached by the student.
- **Evidence:** CARLA/NoCrash experiments and ablations separate multi-command supervision from student-rollout supervision. The paper explicitly warns that changed simulator rendering and pedestrian behavior make versions incomparable.
- **Why read / limit / open issue:** Foundational teacher/student and on-policy correction design. Privileged training information does not imply privileged deployment input. Its expert availability and perception-transfer assumptions differ from direct expert-only GRPO. The remaining question for PhysLoop is which visited-state errors need reward optimization rather than an accessible corrective teacher.

### 3. Roach: End-to-End Urban Driving by Imitating a Reinforcement Learning Coach

**Status/source:** ICCV 2021. [Author paper](https://arxiv.org/abs/2108.08265), [PDF](https://arxiv.org/pdf/2108.08265), §§3–4; [official code](https://github.com/zhejz/carla-roach).

- **Method and interface:** A privileged semantic-BEV RL coach maps simulator state and ego measurements to bounded continuous steering/acceleration distributions. A monocular sensor student is trained by imitation, benefiting from action and auxiliary coach supervision. The camera student is not itself evidence of direct camera-based RL.
- **Evidence:** Closed-loop CARLA/NoCrash tests with simulated traffic, including held-out town/weather, assess coach quality and transfer to the student.
- **Why read / limit / open issue:** Establishes that an RL teacher can provide richer on-policy supervision than a hand-coded expert. Teacher strength, privileged-state access, student capacity and input asymmetry all affect transfer. “RL teacher improves driving student” is not a new thesis; identifying a failure of that transfer could be.

### 4. Learning to Drive from a World on Rails

**Status/source:** ICCV 2021 oral; inspected arXiv v3, 2021-10-02. [Author project](https://dotchen.github.io/world_on_rails/), [paper](https://arxiv.org/pdf/2105.00636), §§3–5.

- **Method and interface:** Model-based learning from offline logs. Factorized ego dynamics and a nonreactive logged world permit Bellman dynamic programming over candidate ego states/actions. Those values supervise an RGB/speed/navigation policy with discrete steering/throttle/brake actions. Privileged logs support training; the final policy uses sensor inputs.
- **Evidence:** CARLA leaderboard and NoCrash evaluation executes the learned policy in a dynamic environment.
- **Why read / limit / open issue:** The critical assumption is that ego actions do not alter the surrounding world's evolution during training. Successful evaluation does not make that assumption universally valid. This is especially relevant to AlpaSim replay: ego closed loop and socially reactive traffic are separate properties. Study transfer limits rather than calling every replay rollout a multi-agent interaction.

### 5. Nocturne

**Status/source:** NeurIPS 2022 Datasets and Benchmarks. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/191e9e721a2748a860714fb23aaf7c5d-Abstract-Datasets_and_Benchmarks.html), [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/191e9e721a2748a860714fb23aaf7c5d-Paper-Datasets_and_Benchmarks.pdf), §§3.2–3.3 and experiments; [code](https://github.com/facebookresearch/nocturne).

- **Method and interface:** A 2D multi-agent simulator and benchmark. It exposes occlusion-limited vectorized objects/road geometry rather than camera pixels. Agents use acceleration, steering and view/head-direction controls; the paper discretizes them. Real driving logs supply scenes and imitation examples. Controlled agents can interact, while logged agents can be replayed.
- **Evidence:** RL and imitation baselines test coordination under partial observability.
- **Why read / limit / open issue:** Separates the challenge of interaction from visual representation learning. Simplified geometry and visibility are not a camera-system qualification. It supports studying cheap behavioral hypotheses, but replacing AlpaSim with it would be a new environment and inference target, not an interchangeable performance measurement.

### 6. MILE: Model-Based Imitation Learning for Urban Driving

**Status/source:** NeurIPS 2022 main track. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/827cb489449ea216e4a257c47e407d18-Abstract-Conference.html), [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/827cb489449ea216e4a257c47e407d18-Paper-Conference.pdf), method and experiments; [Wayve explanation](https://wayve.ai/thinking/learning-a-world-model-and-a-driving-policy/).

- **Method and interface:** Offline model-based imitation, not reward-based RL. Camera observations, expert controls and semantic supervision train an action-conditioned latent dynamics model and vehicle-control policy. A learned world model is not sufficient to classify a method as RL.
- **Evidence:** Closed-loop CARLA evaluation includes new town/weather and execution using imagined plans. The model represents both scene dynamics and ego behavior.
- **Why read / limit / open issue:** A concrete alternative to the premise that interactive optimization is necessary for every gain. Plausible predicted futures do not establish accurate counterfactual responses outside the logged data distribution. PhysLoop should separate representation/world-model quality from the effect of the RL update.

### 7. Waymax

**Status/source:** NeurIPS 2023 Datasets and Benchmarks. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1838feeb71c4b4ea524d0df2f7074245-Abstract.html), [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1838feeb71c4b4ea524d0df2f7074245-Paper-Datasets_and_Benchmarks.pdf), §§3.2–3.5, 4.3 and 5.2–5.3.

- **Method and interface:** Accelerated, log-initialized, vector-state simulation. It supports pose-delta and acceleration/curvature dynamics, replay traffic, rule-based agents and learned agents. Reference experiments include behavior cloning and DQN, not a sensor VLA. Observations can be partially observable, but there is no equivalent rendered-camera burden.
- **Evidence:** Baselines compare action representations, route conditioning and traffic models; trained baselines report variation across seeds. The study documents RL overfitting to simulated-agent behavior.
- **Why read / limit / open issue:** Direct evidence that better training return can mean exploiting a simulator counterpart. A change in traffic policy changes the experiment. PhysLoop can borrow the held-out-environment falsifier, while making no claim that replay-only gains establish social generalization.

### 8. Think2Drive

**Status/source:** ECCV 2024 proceedings. [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06129.pdf), §§3, 4.7 and 5; [supplement](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06129-supp.pdf).

- **Method and interface:** Online model-based RL with a learned latent world model and actor/critic. Inputs are privileged semantic BEV masks and ego state; the paper uses 30 discrete vehicle-control actions. It is a privileged neural planner, not raw-camera VLA post-training. CARLA supplies interactive scenario dynamics.
- **Evidence:** CARLA Leaderboard-v2 scenarios and ablations examine reset/warm-up, replay priorities, training ratio and steering stabilization.
- **Why read / limit / open issue:** Shows that successful driving RL depends on exploration, initialization, data selection and control smoothness, not just a scalar collision penalty. Privileged compact observations and scenario engineering make its compute story incomparable with a rendered 10B policy. The open issue is which mechanism actually limits the existing pretrained expert.

### 9. Bench2Drive

**Status/source:** NeurIPS 2024 Datasets and Benchmarks. [Proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/017761f94a1cd66d01c041aff85492c4-Paper-Datasets_and_Benchmarks_Track.pdf), §§3–4; [official repository](https://github.com/Thinklab-SJTU/Bench2Drive).

- **Method and interface:** Dataset and closed-loop evaluation benchmark, not an RL algorithm. A privileged Think2Drive teacher produces training demonstrations; evaluated sensor models use their own perception and planning architectures. CARLA evaluation organizes 44 interactive scenario types over 220 routes.
- **Evidence:** Multiple sensor baselines, ability-specific evaluation and separate efficiency/smoothness measurements expose weaknesses hidden by aggregate scores.
- **Why read / limit / open issue:** Relevant precedent for per-condition diagnosis and fixed training data. The repository documents protocol/metric changes and a camera-extrinsic evaluation bug; version pinning is essential. Its scenarios and rendering differ from AlpaSim reconstruction, so scores cannot be pooled or transferred. A granular evaluation design is already established; a new learner mechanism must add more.

### 10. Raw2Drive

**Status/source:** NeurIPS 2025 main track, proceedings version. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/c2915bc5961edb04e209a524ec167522-Paper-Conference.pdf), §3, experiments and Appendix C.1–C.4; [official code](https://github.com/Thinklab-SJTU/Raw2Drive).

- **Method and interface:** Dual-stream model-based RL. A privileged BEV world model/planner guides a surround-camera world model and policy through latent-rollout/head alignment. Deployment uses raw sensor inputs; the action space contains 39 discrete vehicle-control combinations. This is stronger evidence for sensor-policy RL than simply citing a privileged planner.
- **Evidence:** CARLA-v2/Bench2Drive comparisons and component ablations.
- **Why read / limit / open issue:** Establishes a training bridge from privileged simulation to sensor RL. Its learned-world-model pipeline and large training allocation are not an immediately affordable replacement for AlpaGym. Replicating its broad claim is unnecessary; borrow the distinction between representation error, dynamics error and policy-optimization error.

### 11. Alpamayo-R1

**Status/source:** NVIDIA technical paper, arXiv v2 dated 2026-01-07; v1 submitted 2025-10-30. [Versioned paper](https://arxiv.org/html/2511.00088v2), §§3, 5.3, 6.1–6.3.

- **Method and interface:** Camera/ego-motion VLA with language reasoning and a continuous trajectory decoder. Its staged training includes supervised reasoning/action learning and RL rewards for reasoning, consistency and trajectory quality. The paper's reasoning-token post-training and public AlpaGym's frozen-language continuous-expert training are distinct paths.
- **Evidence:** Separate open-loop trajectory/critic evaluation, closed-loop AlpaSim tests, architectural ablations and reported on-vehicle demonstrations. Its private data and tested model variants limit direct reproduction from public weights.
- **Why read / limit / open issue:** Architecture and training semantics are prerequisites for designing an intervention. Critic alignment is not proof of causal explanation, and paper-level improvements cannot be attributed wholesale to a public checkpoint or one training component. P-031 should target the actual trainable expert and independently observed behavior.

### 12. OPTED

**Status/source:** Preprint, arXiv v1, 2026-09-17; no peer-reviewed venue established here. [Versioned paper](https://arxiv.org/html/2609.20756v1), §§III-B–III-C, IV-A–IV-B.

- **Method and interface:** A render-free, vector-input RL teacher labels states visited by camera students. The student's update is supervised on-policy distillation, not direct sensor-policy RL. Teacher actions are converted to the student's output representation when necessary.
- **Evidence:** TransFuser/VaVAM experiments use AlpaSim nonreactive replay, 20-second scenes at 10 Hz, 1,669 training/441 evaluation scenes, three training seeds and final-checkpoint evaluation. The approximately 1,000-fold interaction comparison is a separate vectorized PufferDrive/WOMD study, not an Alpamayo sensor comparison.
- **Why read / limit / open issue:** A direct efficiency competitor. Privileged-teacher cost, representation conversion and controller behavior matter. It supports considering correction baselines, not assuming universal savings. An efficiency claim for PhysLoop would need a matched on-policy-correction comparison; otherwise make a mechanism claim.

### 13. RoXDrive

**Status/source:** Preprint, arXiv v2, 2026-09-30; v1 2026-09-29. [Versioned paper](https://arxiv.org/html/2609.36851v2), §§4–5 and Appendices B/I.

- **Method and interface:** Scene-level grouped policy optimization over sampled trajectory candidates in an action-conditioned video world model. An inverse-dynamics evaluator filters action/vision-inconsistent rollouts; hard scenes receive an additional expert-guided update. This combines several mechanisms, rather than isolating one reward change.
- **Evidence:** Public nuScenes and private-data experiments report open- and generated-world closed-loop outcomes. The simulator is closed-source X-World; private-data results are not fully reproducible from the public dataset.
- **Why read / limit / open issue:** A close novelty threat to generic “validity-aware world-model RL.” The authors explicitly separate ego-motion consistency from whole-scene physics and realistic other-agent responses. Do not treat visual plausibility, action-faithfulness or filtered-rollout gains as independent real-world validation. Preserving rejected/invalid outcomes remains essential for PhysLoop's claims.

## Verified technical resources and exclusions

- **DriveIRT** is verified as author-maintained evaluation software, not established here as an RL method or peer-reviewed paper. Its [repository](https://github.com/kesai-labs/drive-irt) accepts a policy-by-scenario score matrix and fits an item-response model. The [authors' common-task discussion](https://kesai.eu/blog/2026-09-08-common-task-framework/) explains the benchmarking motivation. It may support diagnostic panel/ranking work; it does not replace the release gate's road-aware paired inference or authorize adaptive confirmation-set selection.
- **DynaDrive:** the requested name did not resolve to a verifiable primary AV paper in this search. It is excluded rather than assigned a guessed citation. World on Rails and MILE supply verified offline-world alternatives.
- **CARLA, Roach, Nocturne, Raw2Drive and Bench2Drive repositories** above are technical resources for understanding interfaces and protocols. Their availability does not establish current compatibility, runtime cost, license fit for project assets or permission to migrate the project.
- **P-031's pinned AlpaGym/recipes implementation audit** remains the immediate engineering source. A historical paper's training loss, released model, current recipe and specific checkpoint are not interchangeable. Root's separate review covers CLEAR/Poutine/RefineDrive; this ledger does not duplicate those assessments.

## Consequences for the proposed study

### Recommended first question

Use the existing expert-only route to ask: **under a fixed reconstructed-scene protocol, which measured learning mechanism prevents or enables held-out behavioral improvement?** The first shipped-reward pilot establishes a working, exportable update and its limits. It is not itself evidence of a new RL algorithm.

Choose the first intervention only from training/development evidence:

| Observed development mechanism | Narrow candidate contrast | Evidence that could refute the explanation |
|---|---|---|
| Short episodes omit the consequences that dominate longer evaluation | Controlled training horizon, keeping policy frequency, warm-up and reward aggregation explicit | Gains disappear at the fixed confirmation horizon, or are explained by different samples/updates rather than horizon |
| The reward ranks braking choices poorly because replay contacts disagree with the intended endpoint | Collision predicate only, preserving temporal aggregation and penalty scale | Better selected contact metric comes with worse all-contact, rear-contact, progress or invalidity outcomes |
| Groups do not contain distinct useful actions/returns | One exploration or group-size parameter | More diversity produces no informative ranking or no held-out gain at the declared compute budget |

These are candidate investigations, not claims that the mechanisms occur or that the corrections will work. Huy supplies the mechanism derivation, numeric prediction, scientific semantics, tolerances and decision rule. No novelty claim is established by this review.

Do not call replay-only improvement social-interaction learning. The World on Rails/Waymax lineage makes the environment-response assumption explicit. A later reactive-traffic test could challenge transfer, but would require its own admission and protocol; it is not silently included in the first budget.

An improvement claim needs frozen-base comparison, independent development/confirmation selection, training-seed variation, paired road-level evidence, complete invalidity accounting and the declared metric. A causal reward-design claim additionally needs two trained arms. A better-efficiency claim additionally needs a relevant competing training route and complete teacher/setup/evaluation cost. These are different research claims with different minimum experiments.

The existing operational gate is conservative; non-inferiority GO alone is not a superiority result. Measure real-data variability and certify that the available confirmation sample can answer the selected question before buying the full training study. None of the cited papers supplies the missing PhysLoop power measurement.

## Learning priorities for Huy

1. **Visited-state distribution and teacher/student asymmetry:** read LBC and Roach. Explain why on-policy supervision is not synonymous with policy-gradient RL, and distinguish the privileged teacher's information from deployable inputs.
2. **What closes the loop:** read World on Rails and Waymax, then Nocturne. Explain separately ego feedback, traffic reactivity, partial observability, learned dynamics and validity outside logged support.
3. **How a driving reward produces useful optimization:** read Think2Drive's ablations and the actual AlpaGym objective. Derive the grouped advantage, zero-variance case, affine-scaling limitation and the actual likelihood surrogate; do not import a paper's different estimator by name.
4. **Representation and simulation bottlenecks:** contrast MILE with Raw2Drive. Explain why frozen perception, imperfect state representation and world-model error can limit a perfectly implemented optimizer.
5. **Modern comparison discipline:** read Alpamayo's training sections, OPTED's two experimental families and RoXDrive's limitations. Explain why an action-expert update, a reasoning-token update, correction by a teacher and filtering synthetic rollouts support different claims.
6. **Before scaling:** write the trainable-parameter map, action representation, reward-component table, traffic model, update/export parity checks, selection rule, confirmation estimand and compute ledger. The deliverable is understanding and a falsifiable experiment contract, not reproducing every simulator in this reading list.

The appropriate first stop is a reproducible, bounded study with an interpretable result. A broad new world-model stack, multi-agent platform, language-reasoning lane or student transfer should follow an identified limitation and a separately measured resource case.
