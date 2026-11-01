# Adjacent VLA post-training studies

Primary sources inspected on 2026-11-01 by the coordinating agent. These recent studies constrain novelty and transfer claims; they are not ranked as the most-cited works. Publication status below refers to the inspected arXiv version, without asserting a later peer-reviewed venue. Reported outcomes are the authors' measurements.

## CLEAR

[CLEAR, arXiv 2607.02841v1, 3 July 2026](https://arxiv.org/html/2607.02841v1). Read §3.3–3.5, §4.1 and §5. The method learns residual waypoint actions with PPO around a pretrained VLA prior. It freezes the visual/language backbone and disables autoregressive language generation during RL. Its CARLA training uses 64 environments on 32 V100s with an 8-H100 learner, approximately six days; this is not evidence that a two-GPU Alpamayo experiment fits our cap. It reports CARLA longest6/Bench2Drive results, with separate nuScenes evaluation. Its existence rules out treating frozen-backbone, closed-loop VLA action training as new by itself. For PhysLoop, residual policies are a later alternative architecture; changing to one in the first reward comparison would add a confound.

## Poutine

[Poutine, arXiv 2506.11234v1, 12 June 2025](https://arxiv.org/html/2506.11234v1). Read §2.3 and §3. The 3B VLM uses GRPO with a trajectory preference-derived rater-feedback reward and a format reward. It uses 416 training examples plus 63 held out from 479 preference-labelled validation examples; the reported RL stage is 12 hours on four A100s. That stage optimizes future-trajectory prediction and omits the structured CoT tasks. Its results therefore do not establish improved reasoning from RL. This is logged-context trajectory optimization, not repeated sensor feedback in an interactive simulator. Rater-feedback score cannot be translated into collision reduction. Adopting it would change model, reward, data and evaluation together. Treat it as a separate possible lane, not an automatic substitute when AlpaGym training is unstable.

## RefineDrive

[RefineDrive, arXiv 2609.35078v1, 28 September 2026](https://arxiv.org/html/2609.35078v1). Read §3.2–3.5 and §4.1–4.3. It derives failure diagnoses from simulator states, retrieves feasible nearby trajectory corrections, applies correction SFT and then safety-layered GRPO. Its Qwen3.5-4B policy predicts four-second trajectories. Crucially, evaluation under NAVSIM v2 metrics uses the original NAVTEST scenes only; the paper explicitly excludes second-stage synthetic observations and two-stage pseudo-simulation aggregation. This is relevant prior art for failure-driven correction and graded feedback among failed trajectories. It does not establish repeated closed-loop camera-policy behavior in AlpaSim. A similar reward idea would need a sharper mechanism and a protocol-specific contribution; “use failure feedback” is too broad a novelty claim.

## Cross-study comparison rule

Compare methods only after naming the observation distribution, trainable parameters, action representation, privileged inputs, traffic response, time horizon, controller and scoring procedure. A benchmark may simulate a proposed trajectory without feeding fresh sensor observations back to the policy. Such evaluation answers a useful but different question from recurrent closed-loop driving. Hardware counts establish a reported setup, not a lower bound on possible implementation cost.

These distinctions are the coordinating agent's methodological assessment of the papers. No reported performance is transferred numerically to PhysLoop, and no paper's model is adopted by this review.
