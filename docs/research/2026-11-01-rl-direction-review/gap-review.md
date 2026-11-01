# Gap review of plan v5

Agent-authored review by Claude Code, 2026-11-01, written after the Codex review in this directory. Huy asked for independent verification against papers and official technical sources before any plan update, no repetition of finished work, and no change if the plan was already correct. This serves S2/S4 planning for G9. It is documentation only: no training, rental, license acceptance, gated download or spending occurred, and no scientific prediction, margin, tolerance, reward or estimator was chosen.

## Verdict

**The direction stands.** No source contradicts S1 → S2 → S4, direct expert RL as the first baseline, or the shipped reward as baseline A. The next milestone is still the S2 scientific pilot. S2, budget caps, admissions and ownership are unchanged.

**The S4 card did not stand as written.** Five defects met the change rule below. Two independent agents reached G1 and G2 separately, one from AV precedents and one from RL fine-tuning. Plan v5 was amended under D-022 for these five only.

| ID | Rule | Defect | Change made |
|---|---|---|---|
| G1 | (c) | At group size two the stock advantage is a sign, so a 0.1 m gap and a collision get the same update | Named mechanism, diagnostics, E19, risk row, KC-4 wording |
| G2 | (c) | Nothing on the stock path pulls the policy toward the teacher, and the plan named no such failure | Named mechanism, diagnostics, E21, risk row, KC-4 wording |
| G3 | (a) | "One optimizer step" is false; one trainer step is many | Corrected sentence, update unit defined, E20 |
| G4 | (b) | "Shipped" does not define one learning run | Registration fields left blank for Huy |
| G5 | (b) | Required KL and reference reporting cannot exist on baseline A | Reference defined as the saved zero-update checkpoint, retention registered |

Neither G1 nor G2 has been measured in driving. They are named failure modes with diagnostics. They do not establish that the stock path will fail, and they adopt no fix.

## Change rule

Fixed before the agents ran. A finding changes the plan only if it (a) falsifies a stated fact in `02` or the review, (b) makes a `[C]` commitment infeasible or ill-posed, or (c) names a well-established failure mechanism of the primary path that is absent from `02` §1, S4.5, §8/KC-4 and `protocol-design.md` §6.

A paper's mere existence does not qualify: the plan already defers the novelty re-check to intervention selection. Those are recorded here as precedent only and changed nothing.

## Method and scope

The grep of live documents showed the v5 review cited none of RAD, CaRL, BC-SAC, Gigaflow, TrajHF, Reward (Mis)design, DSRL or the fine-tuning failure literature. Two agents covered those gaps, with the 34 reviewed papers excluded. The coordinating agent then re-checked the consequential claims against primary sources and pinned code, and independently re-ran the worked numbers.

Upstream state was re-read on 2026-11-01. AlpaGym `972d160`, AlpaSim `affc2ea`, recipes `2e907da`, alpamayo1.5 `36aeb4c`, alpamayo2 `5977692` and release v2026.5 all still match the snapshot in [the NVIDIA audit](nvidia-stack.md). Recipes PR 18 is still open at the pinned head.

Code statements below are source inspection at those pins, not execution. Evidence, including the pre-amendment documents, fetched pinned files, the toy script and its output, is in `evidence/rl-gap-review-20261101-v1/` with `SHA256SUMS`.

## G1. Sign-only advantages and noise-ranked groups

**Source.** Cosmos-RL computes `(r − mean)/(np.std(r) + 1e-6)` with population standard deviation ([`grpo.py` lines 38–47](https://github.com/nvidia-cosmos/cosmos-rl/blob/d2a2c57c4bd6496482bc42da19a59b4432705eda/cosmos_rl/dispatcher/algo/grpo.py#L38)). One advantage per rollout is copied to every valid driving step. Group size two is not only the smoke: `public_2507_1epoch` uses it, and `config_validation.py` enforces it for the 2/3-node shared-Cosmos topology. Only `alpamayo_1_5_clrl_test_run` uses six.

**Derivation.** With G = 2, `std = |Δ|/2`, so the advantage is `±|Δ|/(|Δ| + 2ε)`.

| Rewards | Cause | Advantage |
|---|---|---|
| −0.213, −0.214 | 0.1 m tracking gap through the −0.01 term | ±0.998 |
| 1, −9 | One candidate collided | ±1.000 |
| Gap 1e-4 | Any tiny difference | ±0.980 |
| Gap 1e-5 | Any tiny difference | ±0.833 |
| Six candidates, one collision | | −√5 = −2.236, whatever the penalty |
| Six candidates, noise only | | rms 0.994, max 1.455 |

Three consequences follow:

- At G = 2 the −10, −5 and −0.01 coefficients act through ordering only. A clean group weighs as much as a collision group.
- `progress_safety` has continuous terms, so exact ties are improbable. The plan's "all-tied" count can read near zero on a run ranked by noise. The signal is plentiful and unweighted, not scarce.
- The candidates in a group are themselves same-policy replicates, so their spread cannot separate signal from noise. S2.4 measures reliability on the evaluation protocol, not on the training protocol.

**Toy check, not a prediction.** One-dimensional Gaussian policy, collision above a threshold, reward −10·C plus noise. Rate 0.08 reuses the hypothetical in `protocol-design.md` §5. Groups needed for |z| = 2 on the collision direction, from 2M simulated groups at G = 2 and about 667k at G = 6:

| Rate, G | Centered | Standardized |
|---|---|---|
| 0.08, 2 | 29 | 87–88 |
| 0.02, 2 | 115 | about 800–850 |
| 0.08, 6 | 7 | 10 |
| 0.02, 6 | 33 | 80 |

Exact ties restore 29 at rate 0.08. The standardized count barely changes with noise between 0.001 and 0.1, because any noise breaks the ties. The agent's independent run gave 86, 792, 10 and 79, so the rate 0.02, G = 2 cell is soft. Real driving groups are correlated, so transfer the direction, not the numbers.

**Literature.** [Flow-GRPO v5](https://arxiv.org/html/2505.05470v5) §5.3 reports G = 12 and G = 6 unstable and collapsing, G = 24 stable, attributing this to inaccurate advantage estimates; the task is text-to-image generation. [Dr. GRPO v2](https://arxiv.org/html/2503.20783v2) §3 shows standard-deviation division overweights low-variance groups, and variants without it did better on math text at G = 8, though the paper does not isolate that term from length normalization. Both are outside driving. `docs/research/2026-10-27-role-alignment/notes/L2_rl_framework.md` had recorded the group-size point; plan v5 did not carry it.

**Limits.** Exogenous simulator noise on the training path is unmeasured and may be small; the sign-only property holds regardless. Rank updates need not hurt typical returns. The expected cost falls on rare-event targets, which is the owner target at `02` §1. `unbiased_advantage` exists in Cosmos-RL but is absent from AlpaGym's `CosmosRLTrainPolicyConfig`, so removing the division is an estimator change, not configuration.

## G2. Unanchored drift from the pretrained expert

**Source.** `kl_beta` is 0.0 in the base defaults and both Alpamayo 1.5 presets, and `reference_reset_interval` is 0. `_ensure_reference_model` returns before building a reference when `kl_beta` is not positive, and the loss has no imitation term. Cosmos-RL's own field text warns that the KL-off setting may be numerically unstable for long runs. What remains does not bound distance from the teacher:

- The ratio clip acts per row against rollout-time log densities, not against the initial weights. Log density is averaged over all non-batch dimensions, about 640 at `inference_step: 5` on the agent's count of 64 × 2 × 5, so a geometric-mean ratio of 1.2 would correspond to a joint log-ratio near 117 nats at that sample.
- AdamW weight decay of 0.01 pulls toward zero, not toward the teacher.
- Adam's step size is scale-free, so gradients with little signal still move weights.

**Precedent.** Methods that fine-tune a pretrained driving policy and state their objective carry an anchor: RAD (one IL round per four RL rounds), ReCogDrive (a BC term), TrajHF and DriveDPO. NVIDIA's own expert-RL recipe in [recipes PR #42](https://github.com/NVlabs/alpamayo-recipes/pull/42), still open, uses `--anchor-beta 1.0` toward the init weights at lr 3e-6. Its README says training converges within 200 optimizer steps then oscillates, and tells users to scan early snapshots. That recipe is a 2B student with a differentiable open-loop objective, not closed-loop GRPO.

NVIDIA's own [Alpamayo-R1 paper v2](https://arxiv.org/html/2511.00088v2) §5.3 states a KL term against a reference policy, typically the SFT model, to prevent over-optimization on noisy or biased rewards and to preserve pre-training priors. The review's ledger did not record this. That passage describes the paper's model-level objective, which is not the released Gym expert path, so it does not prove the Gym path needs one. It does mean the released presets set to zero a term NVIDIA's paper describes for this purpose.

**Measured evidence is thinner.** [RAD v2](https://arxiv.org/html/2502.13144v2) Table 1, reconstructed scenes with cameras, 337 evaluation clips, no seeds reported:

| Arm | Collision rate | Deviation (ADD) |
|---|---|---|
| IL | 0.229 | 0.238 |
| RL | 0.143 | 0.345 |
| RL + IL | 0.089 | 0.257 |

RL alone did not collapse. It cut collisions, deviated more and gained less than RL with IL. The paper does not say whether the RL-only arm starts from the pretrained stage. [ReCogDrive v2](https://arxiv.org/html/2506.08052v2) asserts collapse without its BC term, but no zero-weight arm exists, and its weakest weight scored best (PDMS 90.8 at 0.001, 89.3 at 0.1). **So an anchor is not a documented prerequisite**, and no driving paper measured collapse without one.

Outside driving, [Wołczyk et al., ICML 2024](https://proceedings.mlr.press/v235/wolczyk24a.html) measured forgetting of pretrained capabilities under APPO and PPO as well as SAC, naming interference in the function approximator. Its imperfect-cloning case resembles a behaviour-cloned start. Flow-GRPO §5.3 reports that omitting KL collapsed visual diversity or quality.

**The classic dip does not transfer.** Cal-QL, WSRL and AWAC attribute their dips to critic mismatch, bootstrap error or pessimistic targets, and the stock path has no critic. What transfers is interference and noise-driven erosion. That is a different mechanism with no driving measurement.

**Why existing checks miss it.** S4.6 catches regression only after the spend. Uniform erosion would not trip "excessive run variability". "Proxy exploitation" requires rising return, and drift does not.

## G3. One trainer step is many optimizer steps

[`nvidia-stack.md`](nvidia-stack.md) stated "Smoke: two candidates, one optimizer step". The pinned loop runs `grpo_optimization_iterations × num_mini_batches` updates, and `mini_batch` is a row count (`mini_batch_size = min(self._mini_batch, num_steps)`). `self.optimizers.step()` runs per minibatch, except where the reduced gradient is zero, so that product is an upper bound. The smoke has `mini_batch: 1` and 22 valid rows, so up to 22 optimizer steps in one trainer step, all one sign.

Consequences for the plan:

- Strict parity holds only before the first optimizer step of a pass.
- "Equal updates" was undefined.
- A horizon change at fixed `mini_batch` also changes optimizer steps per rollout, so "training horizon alone" is not one variable.

## G4. "Shipped" defines no learning run

| Preset | Group size | Learning rate | Scenes | Steps | Reward | Topology |
|---|---|---|---|---|---|---|
| Base default | 1 | 1e-6 | 2 | 1 | `distance_to_gt` | Required |
| `alpamayo_1_5_local_2gpu_smoke` | 2 | 1e-4 | 1 | 1 | `distance_to_gt` | Local 2-GPU |
| `alpamayo_1_5_clrl_test_run` | 6 | 1e-4 | 1 | 50 | `progress_safety` | Slurm full node |
| `public_2507_1epoch` | 2 | 1e-6 | Suite | 1 epoch | `distance_to_gt` | Sized by its comment for 16 rollout and 4 trainer replicas |

No preset combines multiple scenes with `progress_safety` or lr 1e-4, and only one trains with `progress_safety` at all. `02` S4.4 is runnable but not pinned. S4.1's record list and the do-not-vary list both omitted learning rate. Precedents treat it as first-order: RAD and ReconDreamer-RL use 5e-6 against 1e-4 pretraining, PR #42 uses 3e-6, and these presets differ by a factor of 100. A flat result and a divergence would both land in KC-4 as an unsupported signal, with opposite causes.

Choosing these values is a scientific input. The amended plan leaves them blank for Huy.

## G5. Required reporting cannot exist on baseline A

`02` S4.4 required clipping/KL reporting, and `protocol-design.md` §6 required reference policy differences, with §1 binding reference identity and blocking admission when identity is incomplete. On the stock path no reference exists and logged KL is zero by construction, which reads as false reassurance. Separately, `save_freq: 20` and `max_keep: 5` could prune the early checkpoints that PR #42's guidance says matter, though that pruning was not observed at runtime.

The fix defines the reference as the saved zero-update checkpoint, compared offline, and registers retention.

## What was confirmed, so left alone

- **Direct update as first baseline.** DSRL and residual methods are reported as more sample-efficient, and direct on-policy updates are the slow class. But `02` §1 asks about the shipped update, and Gym has no critic, replay or latent actor, so either alternative is a build gated by `02` §0. If G2 appears, a frozen-weight arm is a natural later control, not a swap.
- **The shipped reward as baseline.** Neither CaRL nor Reward (Mis)design says otherwise. By inspection a stall outranks a crash, so the reward passes Knox's idle-versus-crash check, which most published rewards the paper examined failed.
- **Budget caution.** `02` §5 already refuses to promise that six runs fit, and `02` S4.4 permits a feasibility stop. The evidence supports that caution; it does not make S4 infeasible.
- **Load-bearing Codex claims re-checked.** Warn-not-reject staleness, zero-gradient skip, unconditional scheduler step, population standard deviation, the `cotrain_vlm` freeze, the two-GPU 40 GB recommendation, the dependency pins and KL off in shipped presets. All held at the pins.
- **Post-event rows.** RAD ends episodes on collision or >2.0 m because post-event sensor data harms training. The stock path requires a fixed 22 valid steps and copies one advantage to each. `02` S4.3 and the temporal-credit row already cover this, so only one diagnostic was added.

### Budget in plain numbers

The provisional $650 line buys about 406–650 GPU-hours at the $1.00–1.60 October price basis in `04`. That is 203–325 wall-clock hours on two GPUs, or 45–72 GPU-hours per run across nine runs.

Only RAD, ReconDreamer-RL and partly CLEAR are rendered-camera closed-loop results, and none states total samples, so no sample ratio exists. CLEAR's reported 32 V100 + 8 H100 for six days is about 5,760 GPU-hours, across different GPU generations. RAD used 32 RTX 4090s for an unstated duration. Hardware counts are reported setups, not lower bounds. **No rendered precedent supports a learning result at this scale, so S4.5 should not be assumed reachable.** A baseline, feasibility or explained negative result is the outcome to plan for.

## Sources outside the v5 ledger

"Root" means the coordinating agent re-read the cited passage or code. "Agent" means one research agent read it and root did not. Status is as read; none of these was replicated.

| Source | Status, version read | Checked by | Role and transfer limit |
|---|---|---|---|
| [RAD](https://arxiv.org/html/2502.13144v2) | NeurIPS 2025 per repo; arXiv v2 | Root | Closest precedent: PPO with IL in 3DGS scenes, 32 RTX 4090 |
| [ReCogDrive](https://arxiv.org/html/2506.08052v2) | arXiv v2; venue not shown on the page | Root | Single-step NAVSIM, not closed loop |
| [Dr. GRPO](https://arxiv.org/html/2503.20783v2) | Preprint v2 | Root | Math text |
| [Wołczyk et al.](https://proceedings.mlr.press/v235/wolczyk24a.html) | ICML 2024, PMLR 235 | Root | NetHack, Montezuma's Revenge, robotics |
| [Recipes PR #42](https://github.com/NVlabs/alpamayo-recipes/pull/42) | Open at `5ade7d0` | Root | Open-loop 2B student |
| [TrajHF](https://arxiv.org/html/2503.10434v2) | Preprint v2 | Agent | Open-loop |
| [DriveDPO](https://arxiv.org/html/2509.17940v1) | NeurIPS 2025 per proceedings | Agent | NAVSIM |
| [ReconDreamer-RL](https://arxiv.org/html/2508.08170v2) | Preprint v2 | Agent | Simulator change; RL objective unstated |
| [RAD-2](https://arxiv.org/html/2604.15308v1) | Preprint v1 | Agent | Discriminator RL on BEV features |
| [CaRL](https://arxiv.org/html/2504.17838v3) | CoRL 2025, v3 | Agent | Privileged BEV, from scratch |
| [BC-SAC](https://arxiv.org/html/2212.11419v2) | arXiv v2 | Agent | Vector, non-reactive replay, no pretrained prior |
| [Gigaflow](https://arxiv.org/html/2502.03349v1) | Preprint v1 | Agent | Batched vector simulation |
| [Reward (Mis)design](https://ar5iv.labs.arxiv.org/html/2104.13906) | Artificial Intelligence, 2023 | Agent | Reward audit, read through a summarizer |
| [CarPlanner](https://arxiv.org/html/2502.19908v3) | CVPR 2025, v3 | Agent | nuPlan vector |
| [Learning to Drive in a Day](https://arxiv.org/html/1807.00412v2) | Preprint v2 | Agent | Lane following from random weights |
| [ResiP](https://arxiv.org/html/2407.16677v4) | Preprint v4 | Agent | Simulated assembly; reports KL needed to avoid collapse under direct PPO |
| [DSRL](https://arxiv.org/html/2506.15799v2) | Preprint v2 | Agent | Frozen base; robotics; actor-critic |
| [Cal-QL](https://arxiv.org/html/2303.05479v4), [WSRL](https://arxiv.org/html/2412.07762v1), [AWAC](https://ar5iv.labs.arxiv.org/html/2006.09359), [JSRL](https://arxiv.org/abs/2204.02372) | NeurIPS 2023, ICLR 2025, preprints | Agent | Critic-based dips; do not transfer |
| [Policy Decorator](https://arxiv.org/html/2412.13630v1), [DAPG](https://ar5iv.labs.arxiv.org/html/1709.10087), [FQL](https://arxiv.org/abs/2502.02538) | As listed in the agent report | Agent | Each carries an anchor; precedent only |
| [DiffusionNFT](https://arxiv.org/html/2509.16117v1) | ICLR 2026 Oral, v1 | Agent | Likelihood-free; in Cosmos-RL for LoRA pipelines, not wired to AlpaGym |
| Waymo blogs, [Dec 2025](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/), [Aug 2026](https://waymo.com/blog/2026/08/10ailessons/) | Official | Agent, through a summarizer | State RL trains the Driver; give no reward or scale |
| [NVIDIA closed-loop tutorial](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/) | Official, 2026-05-31 | Root | Names GRPO; 10B does not fit one GPU; does not state what is frozen, so that claim rests on code |

One prior-note error was reported and not re-checked by root: `L2_rl_framework.md` says CaRL's clip 0.1 beat 0.2, while the CaRL v3 text gives 0.1 only as a hyperparameter.

## Not verified

- Any of G1 or G2 on the real stack. Nothing executed. Expert weight scale, simulator noise on the training protocol, `dp_shard_size` row splitting and runtime checkpoint pruning are unmeasured.
- Whether RAD's RL-only arm started from pretraining, RAD's wall time and sample count, and ADD's unit.
- ReconDreamer-RL's RL-stage objective and compute.
- Whether AlpaSim itself can end an episode on collision.
- Whether Flow-GRPO's group-size runs used KL.
- Section numbers for papers marked "Agent" came from text extraction, and this remains a targeted gap review, not an exhaustive survey. The agents shared the project context, so their agreement on G1 and G2 is corroboration, not independent replication.
