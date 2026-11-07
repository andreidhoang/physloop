# Sources not yet covered by the RL direction review (2026-11-07)

This is an agent-authored note for Dang Huy, written for the S4 admission review. It records sources found on 2026-11-07 while researching the industry and posting comparison (`2026-11-07-posting-refresh.md`). None of them appears in `docs/` before this note. **It changes no plan text, stage, budget, gate or decision.** Scientific definitions, margins, predictions and the choice of any S4 intervention remain Huy's.

## Change rule applied

The rule is the one used for the D-022 gap review. A source changes plan v5 only if it does one of these:
- falsifies a fact the plan relies on;
- makes a `[C]` commitment infeasible or ill-posed;
- supplies an established mechanism the plan is missing.

A source merely existing, being newer or being relevant does not qualify.

**Result: no source below qualifies.** Plan v5 stays unchanged.

## Verification levels

- **Root-read:** the coordinating agent read the primary page itself on 2026-11-07.
- **Agent-read:** a research subagent read the source; the coordinating agent did not re-read it.

## Sources

### RoaD: Rollouts as Demonstrations (NVIDIA, preprint)

[arXiv 2512.01993 v1](https://arxiv.org/abs/2512.01993), 2025-12-01, by Garcia-Cobo, Igl, Karkus, Zhang, Watson, Chen, Ivanovic and Pavone. **Root-read** (abstract page, and the HTML paper §§1, 4–6 and Appendices A–B).

**What it does.**
- **Model.** Closed-loop supervised fine-tuning of a VLA policy described as structured similarly to Alpamayo-R1. It sees two cameras (wide front and telephoto) plus 1.6 s of ego history, outputs 6.4 s trajectories, and is fine-tuned with its encoders frozen.
- **How the training data is made.** The policy samples 64 trajectories at temperature 0.8. The rollout executes the one closest to the logged trajectory over the first 2 s. Recovery mode starts at 3 m and blends toward the log over 30 steps. Rollouts that deviate more than 4 m are discarded.
- **Data.** 8,251 reconstructed scenes for training, 3 rollouts each, 20 s per rollout.

**Reported AlpaSim result** (920 NuRec scenes, 3 rollouts each; Table 3):

| Metric | Base | RoaD |
|---|---|---|
| Driving score | 0.4443 | 0.6300 |
| Collision rate | 0.0525 | 0.0239 |
| Offroad rate | 0.2833 | 0.2098 |

**Reported limits.**
- The paper reports no training seeds. Its error bars come only from rollout variation, because the full pipeline was too expensive to repeat.
- The main driving-score uncertainty is given as ±0.0057 in §5.2.1 but ±0.0090 in Table 3.
- In a 75-scenario sim-to-sim check (Table 5), the gain shrinks when the renderer moves from 3DGS to NeRF.
- Fine-tuning for too many steps slightly hurts performance.
- **No RL baseline is measured.** The claim that it needs orders of magnitude less data than RL rests on citation and argument (§1, Appendix A).

**Bearing on plan v5.** It does not qualify under the rule:
- Plan §1's "Competing explanation" already names policy-visited-state correction as possibly cheaper than direct RL.
- RoaD measures no RL comparison, so it falsifies no S4 fact.

**Relevance to S4.** It is the strongest same-simulator, sibling-model precedent for that competitor:
- **At admission:** the S4 conclusion will be read against RoaD's reported gain. Plan §1's rule already applies: a comparative or efficiency claim needs a matched alternative and complete accounting.
- **As a contrast:** RoaD discards ≥4 m rollouts, reports one training run and has inconsistent uncertainty figures. PhysLoop's G2 completeness, ≥4 m reporting and training-run rules address exactly these points. That is a framing observation, not a claim about RoaD's correctness.
- **Mechanism:** RoaD's guidance uses the logged trajectory, which is privileged information during data generation. A RoaD-style arm would therefore be a privileged-teacher correction baseline, not a reward-only method.

### RAD and RAD-2 (HUST and Horizon Robotics)

**Agent-read**; affiliation and details of RAD-2 were not re-checked.
- **RAD** ([arXiv 2502.13144](https://arxiv.org/abs/2502.13144), NeurIPS 2025, [code](https://github.com/hustvl/rad)): RL of a camera end-to-end policy in 3DGS-reconstructed scenes. It reports about three times lower collision rate than imitation.
- **RAD-2** ([arXiv 2604.15308](https://arxiv.org/html/2604.15308v1), 2026-04): discriminator-guided RL on BEV features.

**Bearing on plan v5.** It does not qualify. Plan v5 makes no novelty claim for RL in reconstructed scenes and never asserts "first". It adds to the evidence that closed-loop RL of a sensor policy can work, which is not in dispute.

### Wayve, "Validating the AI Driver" (Validation 2.0)

[Wayve blog](https://wayve.ai/thinking/validating-the-ai-driver/), 2026-11-05. **Root-read.**

**What it describes.**
- **Evidence pooling:** combines evidence across releases, platforms and test types (simulation, on-road, rig, dashcam, fleet) using Bayesian power priors, weighting past data by its similarity to the new release.
- **Reported effect:** in one internal analysis, adding fleet data cut the 95% upper bound on an event rate by about three times and the new validation data needed by more than 20 times.
- **GRADER:** names an offline reward model, "Generalizable Reward for Autonomous Driving Evaluation and RL", without describing an RL training method.

**Bearing on plan v5.** It does not qualify:
- **Not established for this case.** It is a company blog with an internal analysis. The analysis concerns event-rate bounds, not paired checkpoint comparison.
- **Wrong assumption for the gate.** Power priors assume the new release behaves like earlier ones, which is the property the gate tests.
- **Ownership.** D-016's conservative method is qualified and protected, and any inference amendment must be separately derived, reviewed and authorized. Huy owns estimands and margins.

**Possible use.** It is relevant only if S2 shows the operational gate lacks power on real data (the S1 limitation). In that case it is one reference among others for a separately derived amendment, not a default.

### Industry direction (context only)

- **Waymo, "Demonstrably safe AI for autonomous driving"** ([blog](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/), 2025-12-09, **root-read**). The inner learning loop trains the Driver with RL in simulation using a Simulator and Critic. An outer loop uses real-world driving to flag problems and checks fixes in simulation. Large teachers are distilled into onboard students. The post gives no reward, scale, input representation or results.
- **Apple Gigaflow** ([arXiv 2502.03349](https://arxiv.org/abs/2502.03349), 2025-02, **agent-read**, affiliation confirmed in the PDF by the subagent). Self-play RL over 1.6 billion km on vector state, not sensors.

**Bearing on plan v5.** It does not qualify. These sources support the direction already adopted under D-021 and supply no S4 fact, mechanism or commitment change.

## Action for the S4 admission review

1. **Read RoaD's §5.2 and Appendix B** before Huy finalizes the S4 claim wording and the competing-explanation discussion.
2. **Decide on a RoaD-style baseline.** If the S4 baseline fails to transfer and an alternative path is considered (`report.md`, alternative row), the choice of whether to use a RoaD-style closed-loop SFT baseline is Huy's. It needs its own matched design and accounting; nothing here adopts it.
3. **Keep Wayve's power-prior pooling in reserve.** Revisit it only if S2 power measurement fails, under a separately derived inference amendment.
