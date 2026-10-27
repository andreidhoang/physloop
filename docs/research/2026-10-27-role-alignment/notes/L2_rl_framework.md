# L2: RL framework, GRPO estimator choices and reward design for S4 (closed-loop GRPO on Alpamayo 1.5 in AlpaGym), mapped to JR2010863 I1, I2, D1, D3 (reward half), D5, Q3, X1

Lane L2, research only, as of **2026-10-27**. Every source below was accessed on 2026-10-27 unless it is marked as coming from the earlier ledger (2026-10-03 or 2026-10-25). P = primary source, S = secondary. Confidence: H = high (read in code or the paper itself), M = medium (abstract or snippet only, or an inference from code structure), L = low. Commit SHAs are pinned wherever code was read. Quotes are at most a few words; everything else is paraphrased. **No prediction, tolerance or margin values are proposed.** Every slot written `____` is Huy's. Code identifiers (config keys, function names) come from Apache-2.0 repositories and are quoted as identifiers.

---

## 1. Findings table

Price basis is the plan's §5, vast.ai on-demand prices of 2026-10-25: H100 SXM $1.93 cheapest / $2.40 median, and RTX PRO 6000 Workstation $1.00 / $1.60. An 8-GPU node therefore costs $15.44–19.20/h on H100 and $8.00–12.80/h on RTX PRO 6000 WS. The plan prices S4 at 225–450 H100-h of training for 2 reward arms × 3 seeds, i.e. 6 runs of about 37.5–75 H100-h each, plus up to 10 evaluation sweeps; S4 has about 50 focused hours. Throughput anchors are **low confidence**: the plan's estimate of about 4 ticks/s per GPU (`04` §4, an inference), and Classmethod's community measurement of 1.5–3.5 min per trainer step on 2×L40S (S, ledger 2026-10-03).

| ID | JR2010863 line(s) | Stage / deliverable | Kind | What changes in the S4 stage card | GPU-h and $ delta (2026-10-25 prices) | Hours delta | Gate | Owner | Sources |
|---|---|---|---|---|---|---|---|---|---|
| **L2-1** AlpaGym bottleneck analysis and one-node scaling model | D1, I2 (framework performance), X1 (stand-in), D4, Q5 (single-node Slurm presets only) | S4 / G9, G7 | **b**, gated as c because it adds spend | **Build:** a design doc plus a 1→8-GPU sweep that uses only shipped presets and replica overrides: `local_colocated_1gpu` (2 GPUs by my count, 1 colocated Cosmos GPU plus `alpasim_gpus: 1`; the preset states no total), `slurm_partial_node_1_2_1` (4 GPU), `slurm_full_node_1_3_4` (8 GPU), varying `rollout_replicas` 1/2/3, `policy.inference.max_batch_size`, `rollout.batch_size`×`n_generation`, and `prefetch_rollout` on/off. Instruments: `perf.enabled=true`, AlpaSim Prometheus, PR #44 metrics (mean wall clock → GPU-h, stall spread, MFU/HFU on 6N/8N, OFU from `nvidia-smi dmon --gpm-metrics`), and a `weight_version` lag histogram. Fit the balance and Amdahl/communication model on held-in configurations, then predict the held-out ones. **★:** the scaling model and its diagnosis. **Acceptance:** stated prediction error on the held-out configurations; the window and step count behind every mean are stated. **Prediction slots:** steady-state episodes/h at 1_3_4 `____`; share of rollout-worker wall time inside `rollout/sim_step_rpc` `____`; fitted serial fraction f `____`; predicted speedup for NVIDIA's 3-node preset `____`. **Kill:** if NCCL transport or Slurm does not come up on a rented single node, fall back to the two local topologies and label the model "not validated beyond 2–4 GPUs". **Cost:** as in the next column | +64–96 GPU-h: **H100 $124–230**, or **RTX PRO 6000 WS $64–154**. About half overlaps the S4 pilot if 1_3_4 becomes the production topology | +12–18 h (Huy about 10–14 for the model, predictions and doc; supervision about 3–4) | P-005 (inside the ≤ $2.5k pilot or ≤ $6k tier); KC-L before publishing any number measured on NuRec scenes (conservative reading of licence §3) | ★ Huy (model, predictions); agent (harness, sweep, plots) | Q1 1.4–1.9, 1.22–1.23; Q7 7.1–7.4; `04` §4 |
| **L2-2** Estimator audit: known-answer suite and log-prob parity | Q3, D5, D1, X1 (infrastructure correctness) | S4 / G9 | **a** | **★ row corrected:** AlpaGym's estimator is a per-step-row PPO clip on a **geometric-mean (GSPO-style) chain ratio**, exp(mean over T×64×2 of per-element log-ratios), clamped to ±5 in log space and clipped to [0.8, 1.2]. Its KL is the **closed-form per-transition Gaussian KL** (the Flow-GRPO form), averaged over the same elements. It is **not k3** (§3, deltas D-4 and D-5). The advantage is cosmos-rl's group-standardized episode reward, broadcast to every valid step row. **Build:** the known-answer tests KA-1…KA-7 (§4), and a first-step parity check with `allowed_outdated_steps=0` and `mu_iterations=1`, where the ratio must equal 1 within Huy's tolerance. This follows the bit-equal sampler parity check that NVIDIA's PR #42 performs. **Acceptance:** every KA passes; parity is reported. **Prediction slot:** parity deviation at step 1 `____`. **Kill:** a parity failure blocks training runs until it is diagnosed | ≈ 0 (CPU tests; parity runs inside the first real run, at most about 1 GPU-h, ≤ $2.40) | +4–6 h | none | ★ Huy (math, tolerances); agent (failing tests from the KA inputs) | Q1 1.13–1.19; Q4 4.1, 4.4, 4.5, 4.9; Q2 table (PR #42 parity check); KA-1…KA-8 |
| **L2-3** Reward ablation re-specified to config-only, one-variable arms | D3 (reward), Q3, I1 | S4 / G9 | **a** | **Build:** arm A is `reward=progress_safety` as shipped. Arm B changes one variable: `collision_any` (−10) is replaced by `collision_front` + `collision_lateral`, the at-fault definition of the challenge and the S4 acceptance metric. Note for the ★ reward row: an episode that is both front and lateral scores −20 here, against −10 under `collision_any`. Under group standardization this rarely changes the ranking, but it is a second difference between the arms that must be recorded or capped. **The planned "plus comfort/jerk" arm is not config-only**: AlpaSim at the pinned `46b401be` has no jerk or comfort scorer. Defer it to a gated AlpaSim scorer, in the same row as the plan's stall/oscillation scorers. **★:** term choice and scales. **Acceptance:** unchanged (gate verdicts on the 441 held-out scenes). **Prediction slots:** first hacking mode under arm A `____` and under arm B `____`; rear-collision rate change from A to B `____` | 0 (still 2 arms × 3 seeds) | −6 to +2 h. The low end applies if the plan's 50 h included building a jerk scorer, which the plan does not state | none beyond the existing P-005; KC-L2 before releasing any RL'd checkpoint | ★ Huy | Q1 1.20–1.21; Q5 5.1–5.5 and mechanism; KA-1, KA-2, KA-4, KA-8 |
| **L2-4** Training-time hacking monitors from existing AlpaSim metrics | D3, D4 | S4 / G9 | **a** | **Build:** add zero-scale reward terms (`scale: 0.0`) so they are logged without changing the reward: `progress_rel` (MIN-aggregated), `dist_traveled_m` vs `gt_dist_traveled_m` (stall), `wrong_lane`, `min_distance_to_lane_boundary_m`, `min_distance_to_obstacle_m` (a close-encounter proxy), `plan_deviation`, `collision_rear` (being rear-ended under non-reactive replay). Oscillation still needs a new scorer. **Acceptance:** each monitor appears in the per-episode logs. **Prediction slot:** the monitor that moves first `____` | 0 | +1–2 h | none | agent (config), ★ Huy (verifier logic) | Q1 1.20–1.21, 1.24; Q5 mechanism; O-2 |
| **L2-5** Budgeted hyperparameter study (successive halving with separate tuning seeds and scenes) | Q3 (tuning), D1 | S4 / G9 | **a** if carved out of the 225–450 H100-h; **c** if additive | **Build:** about 9 configurations, Huy's fractional design over the knobs that can be reached from config: SDE `noise_level` (0.2 shipped; π_RL found 0.2 unstable), `n_generation` (1/2/6 in presets), `optm_lr` (1e-6 default vs 1e-4 in NVIDIA's test run), `kl_beta` with `reference_reset_interval`, `allowed_outdated_steps` (0 vs 4), `inference_step` (4 or 5), and optionally `unbiased_advantage` (Dr. GRPO) through the generated-TOML route if the dry run in O-1 confirms it. Use η = 3 over 3 rungs. The rung metric is training reward or a cheap closed-loop metric on **tuning** scenes taken from the 1,761 training split, never the 441 test scenes. Report with IQM and interval estimates (rliable); the final claims still come only from the gate. **Prediction slots:** the knob with the largest rung-0 effect `____`; the surviving configuration `____` | 108 GPU-h at n0 = 9, η = 3, r0 = 4 GPU-h: **H100 $208–259** or **RTX PRO $108–173**. Zero delta if carved out (it then consumes about 108 of the 225–450 H100-h) | +6–8 h | P-005 if additive; KC-L2 before releasing any RL'd checkpoint | ★ Huy (design, rung metric); agent (sweep) | Q1 1.9–1.11; Q3 3.2; Q4 4.1–4.2; Q6 6.1–6.5; KA-9 |
| **L2-6** Multi-node validation of the L2-1 model on NVIDIA's own 3-node preset | D1, Q5, X1 | S4 / G9 (gated) | **c** | **Build:** run `slurm_distributed_1_1_1` (3 × 8 GPUs: one policy, one rollout, one AlpaSim node; NCCL transport) for about 3–4 h. **Acceptance:** prediction error of the one-node model at 3 nodes. **Prediction slot:** 3-node speedup `____`. **Kill:** if no cross-node fabric (IB or EFA) is available, keep the prediction unvalidated | +72–96 GPU-h: $139–230 at vast H100 prices *if* a multi-node offer exists; $287–412 at Lambda's $3.99–4.29 (multi-node availability UNVERIFIED) | +6–10 h | P-005 plus a new owner decision; collaborator tier preferred | ★ Huy | Q1 1.5–1.6, 1.19; Q7 (design-doc item 3) |
| **L2-7** Pin the training protocol to the evaluation protocol, or record the mismatch as a variable | D3 (simulation validity), I2 | S4 / G9 | **a** | NVIDIA's RL presets train at `control_timestep_us=200000` (5 Hz), with 22 closed-loop steps (4.4 s after 1.6 s of force-GT), `+cameras=4cam_1080`, and policy `input_size: [320, 512]`. The gate evaluates at 10 Hz over 200 steps. **Build:** Huy chooses either to train at 10 Hz (and decides `expected_valid_steps`) or to keep 5 Hz and log the mismatch in the manifest. **Prediction slot:** score change from the train/eval rate mismatch `____` | 0 if recorded. At 10 Hz over the same simulated seconds, policy steps and rendered frames per episode double, so rollout GPU-h rises by up to 2× for the rollout share | +1–2 h | none (P-005 if 10 Hz raises spend) | ★ Huy decides; agent records | Q1 1.10–1.11; D-15; O-7 |
| **L2-8** Constrained (Lagrangian) penalty on collisions as an alternative to fixed −10 | Q3, D3 | S4 / G9 (gated) | **c** | **Build:** a third arm. An outer loop updates the collision scale at checkpoint boundaries (dual ascent on the measured violation rate) through config-only restarts from the exported checkpoint. The optimizer state resets at each boundary, which must be recorded. **★:** the multiplier update and the constraint level d `____` | +1 arm × 3 seeds = +112–225 H100-h: **$217–540** | +8–12 h | P-005; KC-L2 before releasing any RL'd checkpoint | ★ Huy | Q5 5.6–5.7; Q4 4.11; KA-4 |
| **L2-9** Reactive-traffic arm (CAT-K on) instead of log replay | I2 (traffic models), D3 | S4 / G9 (gated) | **c** | AlpaGym's AlpaSim topologies set `trafficsim: skip: true`, so RL runs against non-reactive replay agents. One arm would re-enable CAT-K through `alpasim.wizard_args.extra_overrides`; that this works is UNVERIFIED. **Prediction slot:** rear-collision and stall rates under reactive vs replay agents `____` | +112–225 H100-h ($217–540), plus CAT-K GPU needs (unknown) | +6–10 h | P-005; KC-L2 before releasing any RL'd checkpoint | ★ Huy | Q1 1.8; Q5 mechanism; D-16; O-12 |

Notes on the table:
- **Spend.** L2-2, L2-3, L2-4 and L2-7 cost $0 and should go in regardless.
- **Hours.** L2-1 and L2-5 together add about 18–26 h to S4's about 50 h. That pushes S4 toward 70 h unless the comfort/jerk arm is dropped (L2-3), which saves time.
- **Kind (c) rows.** L2-6, L2-8 and L2-9 are kind (c) and should wait for the S4 pilot.
- **Licence gates.** KC-L applies to any published number measured on NuRec scenes, including throughput (conservative reading of licence §3). KC-L2 applies to any release of an RL'd checkpoint from any arm, because every S4 arm trains on NuRec-derived rollouts.
- **Source column.** It points to rows of the §2 Facts ledger (Q1 1.x … Q7 7.x), the known-answer list (KA-n, under Q4), the deltas (D-n, §3) and the open questions (O-n, §6). Every URL is in the ledger rows.

---

## 2. Facts ledger (grouped by key question)

### Q1. AlpaGym as shipped: architecture, estimator, rewards, hardware, scaling, levers

**Takeaway.** AlpaGym is still frozen at commit `972d160` (2026-07-21; last code sync 2026-07-08).
- **Stack.** It runs on cosmos-rl `d2a2c57` (2026-05-27), recipes `1d99fc5` (head of the unmerged PR #18) and AlpaSim `46b401be` (2026-06-15).
- **Scaling.** Presets already cover a full 8-GPU node (1 policy / 3 rollout / 4 AlpaSim GPUs) and 3- and 5-node Slurm runs with NCCL transport, so "can it use 8 GPUs / multi-node" is a **config-level yes** (Slurm only).
- **Estimator.** The objective is a PPO clip on a per-step geometric-mean chain ratio. Advantages are standardized within a group of episodes. The KL is a closed-form Gaussian KL that is off by default.
- **Rewards.** The reward is an episode-level sum of AlpaSim aggregated metrics.
- **Default trap.** The default `n_generation: 1` gives every advantage a value of 0.

**Cited findings**

| # | Fact | Source (P/S, conf.) |
|---|---|---|
| 1.1 | AlpaGym HEAD is `972d160` (2026-07-21, docs). The last code change is PR #4 "Manual sync of changes upto 06 26 2026" (merged 2026-07-08), which added Slurm autoresume, NCCL for the remaining presets, batched deterministic rollouts, AlpaSim fetched by SHA, and a recipes pin update. No commit since, as of 2026-10-27 | [commits](https://github.com/NVlabs/alpagym/commits/main) (P, H) |
| 1.2 | Pins: cosmos-rl `d2a2c57…` (overriding the recipe's `747d1bd`, "~88 commits ahead"); alpamayo-recipes `1d99fc5…`; `alpasim-grpc` `46b401be…`; alpamayo `4cda35d…`; torch from the cu128 index; flash-attn 2.8.3 (cu12, torch 2.8) | [pyproject.toml @972d160](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/pyproject.toml) (P, H) |
| 1.3 | Pin dates: cosmos-rl `d2a2c57` is 2026-05-27 ("gym example trainer", #689, which adds the trajectory-iteration contract). AlpaSim `46b401be` is 2026-06-15 | [cosmos-rl commit](https://github.com/nvidia-cosmos/cosmos-rl/commit/d2a2c57c4bd6496482bc42da19a59b4432705eda); [alpasim commit](https://github.com/NVlabs/alpasim/commit/46b401beaec27977cbb86826df8e39cec2d49e49) (P, H) |
| 1.4 | Architecture: Cosmos-RL asks the `AlpagymRollout` backend for payloads. The rollout worker issues one AlpaSim `simulate(1)` call per rollout, through a streaming pool of up to `max_concurrent_rollouts`. AlpaSim calls back into the worker's `EgodriverService` on every tick. Policies on one worker share an `InferenceEngine` thread that batches forwards up to `max_batch_size`. Completed episodes go to Cosmos over disk (JSON) or NCCL (tensor payloads, `TCPStore` rendezvous). The reward is computed on the rollout worker | [runtime README @972d160](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/README.md) (P, H) |
| 1.5 | Trainer backend: `AlpagymGRPOTrainer` subclasses cosmos-rl's `GRPOTrainer` and replaces the text loss with replay training. Policy parallelism keys are `tp/cp/ep/dp_shard/pp/dp_replicate` (all 1 by default). The NCCL transport preset says its env vars feed both AlpaGym's data plane and cosmos-rl's native policy→rollout **weight-sync mesh over NCCL** | [trainer.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/trainer.py); [nccl.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/transport/nccl.yaml) (P, H) |
| 1.6 | Topology presets: `local_colocated_1gpu`; `local_disaggregated_2gpu`; `slurm_partial_node_1_2_1` (4 GPUs: 1 policy, 2 rollout, 1 AlpaSim); **`slurm_full_node_1_3_4`** (one 8-GPU node: 1 policy, 3 rollout, 4 AlpaSim); **`slurm_distributed_1_1_1`** (3 nodes; policy_replicas 4, rollout_replicas 8, dp_shard 2); **`slurm_distributed_1_2_2`** (5 nodes; rollout_replicas 16); **`slurm_distributed_shared_cosmos_2_3`** (5 nodes: 2 shared Cosmos nodes plus 3 AlpaSim nodes). Slurm execution requires `cosmos.mode=disaggregated`. No Kubernetes preset exists | [topology dir](https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/topology); [config_validation.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/config_validation.py) (P, H) |
| 1.7 | NVIDIA's own balance reasoning, in the `shared_cosmos_2_3` comments. It moves a node to AlpaSim, calling AlpaSim the per-node render bottleneck. It treats concurrent sim sessions = batch_size × n_generation × prefetch, needing an off-policy buffer C = 5×12×4 = 240 > 72 slots. It says that dropping any of five coupled settings makes the third AlpaSim node "starve back to the 0.42 baseline" (**units undefined**). It trains 22 replay rows per episode as step-level samples (internal MR-2266) | same dir, `slurm_distributed_shared_cosmos_2_3.yaml` (P, H for the text; the meaning of 0.42 is UNVERIFIED) |
| 1.8 | AlpaSim capacity per AlpaGym topology: `alpagym_4gpu` is renderer 4 GPUs × 1 replica × 3 concurrent rollouts = 12, with physics 12 and controller 6 × 2 = 12 on CPU. `alpagym_8gpu` gives 24. The driver is `null` (AlpaGym is the driver). **`trafficsim: skip: true`**, so RL runs against non-reactive log-replay agents | [alpasim_configs topology](https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879/packages/alpasim_configs/src/alpagym_alpasim_configs/configs/topology) (P, H) |
| 1.9 | Defaults (`default.yaml`): `cosmos.mode: colocated`; policy_replicas 1, rollout_replicas 1; `train_batch_per_replica: 1`; `optm_lr: 1.0e-6`; `optm_warmup_steps: 20`; train_policy `allowed_outdated_steps: 4`, `on_policy: false`, `mini_batch: 1`, `grpo_ratio_clip_low/high: 0.2`, `grpo_optimization_iterations: 1`, `kl_beta: 0.0`, `reference_reset_interval: 0`; rollout `n_generation: 1`, `batch_size: 1`, `prefetch_rollout: true` (needs `dp_shard_size == 1`); `expected_valid_steps: 8`; force-GT 1.7 s; control step 100 ms; `perf.enabled: false`; Slurm defaults 1 node × 8 GPUs | [default.yaml @972d160](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml) (P, H) |
| 1.10 | Policy preset `alpamayo_r1.yaml`: 4 cameras, `input_size: [320, 512]`, 4 context frames, `max_batch_size: 4`, VLM sampling temperature 0.6 and top_p 0.98, `force_determinism: true`, `retry_on_nonfinite: true`, **`int_method: sde`, `noise_level: 0.2`, `inference_step: 5`** | [policy yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml) (P, H) |
| 1.11 | NVIDIA's experiment presets. `alpamayo_1_5_clrl_test_run`: topology `slurm_full_node_1_3_4` with `rollout_replicas: 2` and `dp_shard_size: 2`; `reward: progress_safety`; **control 200 ms, force-GT 1.6 s, `n_sim_steps: 30` → 22 valid steps**; `+cameras=4cam_1080`; `max_num_steps: 50`; `train_batch_per_replica: 6`; **`optm_lr: 1.0e-4`**; warmup 1; **`n_generation: 6`**; prefetch off; KL 0. `alpamayo_1_5_local_2gpu_smoke`: disaggregated, `inference_step: 4`, `n_generation: 2`, lr 1e-4 | [experiment dir](https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/experiment) (P, H) |
| 1.12 | Run preset `public_2507_1epoch`: test suite `public_2507`; `n_generation: 2`, `batch_size: 1`, `train_batch_per_replica: 8`, `mini_batch: 4`. A comment sizes it for 16 rollout replicas and 4 trainers | [run preset](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/run/public_2507_1epoch.yaml) (P, H) |
| 1.13 | **Objective**: `compute_ppo_surrogate`. log_ratio = new − old, clamped to [−5, 5]; ratio = exp; PPO min with clip [1−low, 1+high]; mean over valid (non-padding) rows. The KL penalty is `kl_div` mean × `kl_beta` (0 means off) | [replay_objective.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/replay_objective.py) (P, H) |
| 1.14 | **Advantage**: computed by Cosmos-RL before the trainer sees the data, one per rollout (episode), and **broadcast to every valid step row**; padding rows get 0. Group size = `rollout.n_generation`, i.e. sibling closed-loop episodes of the same scene payload | trainer.py `_prepare_training_data`; [rollout_filter.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/rollout_filter.py) (P, H) |
| 1.15 | cosmos-rl GRPO advantage at `d2a2c57`: (r − mean)/(np.std(r) + eps), with population std, `eps = 1e-6`, **or** r − mean when `unbiased` is set. **Inference: with `n_generation: 1`, every advantage is exactly 0**, so the default preset updates nothing through the policy loss | [grpo.py @d2a2c57](https://github.com/nvidia-cosmos/cosmos-rl/blob/d2a2c57c4bd6496482bc42da19a59b4432705eda/cosmos_rl/dispatcher/algo/grpo.py) (P, H; the zero-advantage consequence is inferred) |
| 1.16 | **Log-prob**: `_batched_sde_logprob` takes `samples_list [B, T+1, Tf, D]` (T SDE steps; for the expert, Tf × D = 64 × 2). It computes the Gaussian per-element log-density of each transition, then takes the **mean over steps and elements**, giving one scalar per sample. The per-step mean is x·(1 − s²/(2σ)·dt) + v·(1 + s²(1−σ)/(2σ))·dt, with σ = 1 − t and s = a·√(σ/(1 − min(σ, σ_max))). The time grid is linspace(0, 1, T+1) | [flow_matching.py @1d99fc5](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/diffusion/flow_matching.py); [expert model.py](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/models/expert_model/model.py) (P, H) |
| 1.17 | **KL**: with a reference model, kl_div = (dt/2)·(s·t/(2(1−t)) + 1/s)²·(v_ref − v_θ)², averaged over steps and elements. This is the closed-form KL between equal-variance Gaussian transitions, i.e. Flow-GRPO's expression; it is **not a k1/k2/k3 sample estimator**. The reference is a frozen deep copy, optionally refreshed every `reference_reset_interval` steps | same files; trainer.py `_ensure_reference_model` (P, H) |
| 1.18 | `sde_type="cps"` raises NotImplementedError (unchanged). The code default `noise_level` is 0.7; AlpaGym sets 0.2. The replay trace stores `noise_level` so the trainer recomputes with the rollout's value | flow_matching.py; model.py (P, H) |
| 1.19 | **Staleness**: AlpaGym's trainer filter keeps the freshest `train_batch_per_replica` rollouts and only **warns** about kept rollouts older than `allowed_outdated_steps`. On cosmos-rl `main`, the dispatcher (`status.py`) filters by staleness and the rollout worker caps prompts at current weight version + k | rollout_filter.py (P, H); [cosmos-rl status.py / rollout_control.py on main e82b56f](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/dispatcher/status.py) (P, H for main; UNVERIFIED at the pin `d2a2c57`) |
| 1.20 | **Rewards** are episode-level: a sum over terms of scale × `episode.metrics.aggregated[name]` (`kind: metric`), plus `kind: distance_to_gt` = scale × XY RMSE in metres between executed and GT poses. Presets: `distance_to_gt` (−0.01, the default); `metrics` (progress +1, `collision_any` −10); `progress_safety` (progress +1, `collision_any` −10, `offroad` −5, distance_to_gt −0.01). Validation does not forbid `scale: 0.0` | [reward presets](https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/reward); [rewards/compute.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/rewards/compute.py); [config.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/config.py) (P, H) |
| 1.21 | AlpaSim metrics at the pin `46b401be` that can serve as reward terms (with time aggregation): `collision_front/rear/lateral/any` (MAX); `offroad`, `wrong_lane` (MAX); `min_distance_to_lane_boundary_m` (MIN); `progress`, `progress_rel_to_total` (LAST; fraction of the full GT path); `progress_rel` (MIN); `dist_to_gt_trajectory/location` (MAX); `dist_traveled_m`, `gt_dist_traveled_m` (LAST); `min_distance_to_obstacle_m`; `plan_deviation`. **There is no `collision_at_fault` metric and no jerk or comfort scorer** (scorer files: collision, ground_truth, image, minADE, min_distance_to_obstacle, offroad, plan_deviation, safety; code search for "jerk" in AlpaSim eval returns 0) | [scorers @46b401be](https://github.com/NVlabs/alpasim/tree/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/scorers) (P, H) |
| 1.22 | Hardware: 2 GPUs with ≥ 40 GB (ONBOARDING; unchanged). Classmethod saw about 45 GB per GPU on 2×L40S; a 10-step run took about 32 min; weight sync took about 45 s initially and about 3 s per step for about 8.7 GB | ledger 2026-10-03 (P for ONBOARDING; S for Classmethod) |
| 1.23 | `perf/` scopes found in code: `driver/drive`, `driver/policy_step` (GPU wall), `rollout/episode`, `rollout/sim_step_rpc`, `rollout/sim_request_build`, `rollout/reward_compute`, `rollout/engine_init`, `trainer/step`, `trainer/artifact_load(/disk_read)`. There are also periodic CPU/GPU utilization samplers (psutil/pynvml) and a "Simulator internals" block parsed from AlpaSim Prometheus (render/physics/drive RPC sums, which overlap and include queue wait). Categories: orchestration, external_rpc, compute_cpu, compute_gpu_wall, io, synchronization_wait. **Not instrumented:** weight-sync time, trainer MFU, the staleness histogram, CUDA-event forward/backward split | [perf README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/perf/README.md) plus grep of the scope names (P, H for scopes found; the "not instrumented" list is M) |
| 1.24 | Only 8 GRPO fields are exposed in the typed schema: `allowed_outdated_steps, on_policy, mini_batch, grpo_ratio_clip_low/high, grpo_optimization_iterations, kl_beta, reference_reset_interval`. The cosmos TOML is generated from these dataclasses (`run_artifacts._build_cosmos_config`). There is no pass-through for other cosmos keys. The CLI skips writing artifacts when `execution.resolved_config_path` is set, so editing a generated run directory is a possible route; it is untested | [run_artifacts.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/run_artifacts.py); [cli.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/cli.py) (P, H for the code; whether the route works is UNVERIFIED) |
| 1.25 | cosmos-rl `GrpoConfig` at `d2a2c57` has knobs AlpaGym does not expose, including `variant` (grpo/gspo/dapo), `filter_reward_metric` (DAPO dynamic sampling), `advantage_low/high` (±5), `lower_bound_ratio` (dual clip), `loss_type`, `unbiased_loss_max_tokens`, **`unbiased_advantage`** (no std division), `aipo_rho`, `entropy_coeff`, `outdated_rollout_fetch_batch_size`, `use_decoupled_loss` and `behav_imp_weight_cap`. Loss-side knobs are irrelevant because AlpaGym replaces the loss. Dispatcher-side `unbiased_advantage` is wired into the GRPO advantage on main | [config @d2a2c57](https://github.com/nvidia-cosmos/cosmos-rl/blob/d2a2c57c4bd6496482bc42da19a59b4432705eda/cosmos_rl/policy/config/__init__.py); [local_calculator.py on main](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/reward/local_calculator.py) (P, H) |
| 1.26 | Batch-geometry validation: `train_batch_per_replica` must be divisible by `dp_shard_size × mini_batch`. NCCL transport requires replicas ≥ 1. `expected_valid_steps` must equal `n_sim_steps` − force-GT warmup steps | config_validation.py (P, H) |

**Every lever a user can change through configuration alone** (Hydra overrides at `972d160`):
- **Policy (`policy.*`).**
  - model I/O: `use_cameras`, `input_size`, `num_context_frames`, `dtype`;
  - inference: `max_batch_size`, `return_trace_for_rl` (must be true);
  - sampling: `temperature`, `top_p`, `top_k`, `num_traj_samples`, `num_traj_sets`, `max_generation_length`, `force_determinism`, `retry_on_nonfinite`;
  - `diffusion_kwargs`: `int_method`, `noise_level`, `inference_step`, `temperature`;
  - `trajectory_selector` (`identity` or `closest_to_previous`).
- **Reward.** `reward.terms`: any AlpaSim aggregated metric × scale, plus distance_to_gt × scale.
- **Cosmos (`cosmos.*`).**
  - `mode` (colocated or disaggregated);
  - `launch.{policy_replicas, rollout_replicas}`;
  - `train.{train_batch_per_replica, max_num_steps, num_epochs, optm_lr, optm_warmup_steps, optm_decay_type, optm_decay_ratio, optm_min_lr_factor, ckpt.*}`;
  - the 8 `train_policy` fields;
  - `policy.parallelism.{tp,cp,ep,dp_shard,pp,pp_micro_batch,dp_replicate}`;
  - `rollout.{n_generation, batch_size, parallelism.{tp,pp}, backend, prefetch_rollout}`;
  - `logging.*`.
- **Transport.** `disk` or `nccl`, with `nccl_env`.
- **Execution.** `backend`, `slurm.{nodes, gpus_per_node, topology.{alpasim_gpus | cosmos_nodes, alpasim_nodes}, time, autoresume, …}`.
- **AlpaSim (`alpasim.wizard_args`).** `topology` (`1gpu`, `alpagym_gpu1`, `alpagym_gpu_id3`, `alpagym_4gpu`, `alpagym_8gpu`), `force_gt_duration_us`, `control_timestep_us`, `n_sim_steps` (derived), `renderer` (an AlpaSim renderer config group, which is the hook for OmniDreams), and `extra_overrides` (free-form AlpaSim Hydra overrides such as cameras or simulation_config).
- **Other.** `expected_valid_steps`; `dataset.{scene_ids | test_suite_id}`; `perf.*`.

**Inferences**
- **The ratio is length-normalized.** The "trajectory-level clipped ratio" in `02`/`03`/`04` is a length-normalized chain ratio. For T = 5 there are N = 640 elements, so the clip at 1.2 on the geometric mean corresponds to exp(116.69) on the joint chain ratio; on the joint scale it is effectively unclipped. Gradients are scaled by 1/N relative to a sum-log-prob objective, which plausibly explains NVIDIA's lr of 1e-4 in the test run against the 1e-6 default. That explanation is an inference.
- **Episode-level credit.** Each closed-loop step is a separate PPO sample carrying the episode advantage, i.e. episode-level credit assignment without a critic.
- **Training rate vs evaluation rate.** NVIDIA's own RL presets act at 5 Hz over a 4.4 s closed-loop window. Gate results at 10 Hz over 20 s therefore measure transfer across control rate and horizon unless S4 pins them (L2-7).
- **Input size may not match.** The 320×512 input of the policy preset differs from the 320×576 tensor contract in `04` §2. Whether the gate's AlpaSim driver for 1.5 uses 320×576 must be checked before paired verdicts (O-7).

**Gaps**
- The meaning of "0.42 baseline" is unknown.
- Whether cosmos-rl at `d2a2c57` enforces staleness exactly as `main` does is unverified.
- Whether zero-scale reward terms reach the trainer's logged metrics is unverified (the reward is computed on the rollout worker).
- Whether a new AlpaSim scorer's metric reaches `episode.metrics.aggregated` without AlpaGym code changes is unverified. The runtime README says simulator metrics are attached "when the runtime can assign them unambiguously".
- Weight-sync bytes per update have not been measured at the shipped configuration. Only Classmethod's 8.7 GB is known (S).

### Q2. NVIDIA's open RL and post-training infrastructure (late 2026) relevant to VLA and physical AI

**Takeaway.**
- **Alpamayo's RL framework.** NVIDIA's Alpamayo RL runs on **cosmos-rl**: the open-loop 1.5 post-training, AlpaGym's closed loop, and its own π0.5/OpenVLA VLA GRPO examples.
- **cosmos-rl status is mixed.** The README carries a "limited maintenance, migrate to Cosmos 3" banner, yet v0.4.7 shipped on 2026-10-22, and October commits rework async dispatch, payload lookahead across weight sync, and UCXX/NCCL transport.
- **NeMo-RL** is the LLM/VLM post-training stack: Ray, Megatron-Bridge or Automodel, vLLM/SGLang, GRPO/DAPO/GSPO/GDPO/PPO. No VLA or flow-policy path was found.
- **Isaac Lab** points users to third-party **RLinf** for VLA RL post-training (GR00T).
- **Which framework trained Alpamayo.** The 512-policy + 128-rollout-GPU run that post-trained Alpamayo 1.5 is the cosmos-rl recipe `alpamayo1_x_rl`: open-loop GRPO on tokens, with the expert not trained. Primary: the recipes README, verified 2026-10-03 and 2026-10-25. AR1 describes a customized Cosmos-RL with vLLM rollouts.

| Framework (version, date) | Licence | VLM / VLA / flow | Algorithms | Async, disaggregation, weight sync | Multi-node and fault tolerance | Activity | Source (P/S, conf.) |
|---|---|---|---|---|---|---|---|
| **cosmos-rl** (PyPI 0.4.0 on 2026-02-06 … 0.4.4 on 07-31, 0.4.5 on 09-04, 0.4.6 on 09-16, **0.4.7 on 2026-10-22**) | Apache-2.0 code | VLA: OpenVLA-OFT 7B configs ("8p8r", colocated variants, LIBERO, RoboTwin); **π0.5 SFT (2026-02-05) and GRPO** (LIBERO-10, BEHAVIOR-1K; `noise_method` default `flow_sde`, `noise_level` 0.5, noise annealing params); Cosmos Policy + LIBERO (2026-03-06); WFM RL via DiffusionNFT (Cosmos-Predict2.5, SD3.5, Sana) | GRPO variants grpo/gspo/dapo, AIPO, decoupled loss, on-policy distillation (teacher worker), SFT. **Token KL is k3** (exp(ref−cur) − (ref−cur) − 1), with an optional IS-weighted variant | "Fully asynchronous" policy (consumer) and rollout (producer) replicas; colocated or disaggregated modes; single controller (Redis); NCCL weight sync; UCXX/NCCL payload transport; payload lookahead across weight sync (#768, 2026-10-17); FP8 training, FP8/FP4 rollout; vLLM and TRT-LLM rollout | README: dynamic NCCL process groups "for … fault-tolerant and elastic" training; Slurm launch scripts (`tools/slurm`). Kubernetes: not found | README banner: no longer under active development, limited maintenance. Commits #763–#781 dated 2026-10-17 to 10-22 | [README](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/README.md); [PyPI](https://pypi.org/project/cosmos-rl/#history); [commits](https://github.com/nvidia-cosmos/cosmos-rl/commits/main); [pi05 config](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/configs/pi05/pi05-libero10-grpo.toml); [pi05 model](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/policy/model/pi05/__init__.py); [grpo_trainer @d2a2c57](https://github.com/nvidia-cosmos/cosmos-rl/blob/d2a2c57c4bd6496482bc42da19a59b4432705eda/cosmos_rl/policy/trainer/llm_trainer/grpo_trainer.py) (P, H) |
| **NeMo-RL** (v0.7.0 on 2026-07-25; README news through 2026-10-22) | Apache-2.0 | LLM and VLM (e.g. Qwen3.5 VLM GRPO; Nemotron 3.5 Super VL image GRPO). **No VLA, flow-matching or diffusion policy path found** (code search for `flow_matching`: 0 hits) | GRPO/GSPO/DAPO, GDPO (2026-03-12), **PPO (v0.7.0)**, CISPO, DPO, SFT, RM, on-policy distillation (MOPD), LoRA | Async RL with replay buffers ("fully asynchronous GRPO"); "refit" weight sync, including NeMo-DCR (delta-compressed, bitwise-exact sparse sync; preprint 2026-10-07); backends Automodel (FSDP2/TP/CP/EP) and Megatron through **Megatron-Bridge**; generation on vLLM, SGLang or Megatron inference | Ray. Cluster docs cover Slurm (`ray.sub`) and mention Kubernetes. "Resiliency" (fault tolerance, autoscaling) is listed as upcoming | Very active (model day-0 support, NGC container) | [README](https://github.com/NVIDIA-NeMo/RL/blob/main/README.md) (P, H); [cluster docs 0.7.0](https://docs.nvidia.com/nemo/rl/0.7.0/cluster.html) (P via search snippet, M) |
| **Megatron-Bridge** | (not checked) | Megatron backend for NeMo-RL; HF↔Megatron checkpoint conversion | — | — | — | — | NeMo-RL README (P, M) |
| **Isaac Lab** (main, page updated 2026-10-27; versions up to v3.0.0) | (not checked) | "RL Post-Training for VLA Models" is an experimental feature that uses **RLinf** (third party) for GR00T N1.5/N1.7 and OpenVLA on Isaac Lab tasks | PPO / actor-critic / SAC | Actor (FSDP), Rollout (HF inference), Env (Isaac Lab) workers on Ray | Across GPUs and nodes with FSDP; pins RLinf 0.2 commit `0f9ea98c…` | — | [Isaac Lab page](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/rlinf_vla_posttraining.html) (P, H) |
| **AlpaGym** (`972d160`) | Apache-2.0 (ledger) | Alpamayo 1.5 expert only; VLM frozen | PPO-clip on a geometric-mean chain ratio; GRPO group advantage; closed-form Gaussian KL | Colocated or disaggregated; NCCL rollout transport; cosmos-rl weight sync; staleness ≤ 4 | Slurm 1–5 nodes (presets); Slurm autoresume plus checkpoint requeue. No Kubernetes | Code frozen since 2026-07-08 | Q1 above (P, H) |
| **alpamayo-recipes** | Apache-2.0 code (ledger) | Open-loop GRPO on the VLM (tokens) with cosmos-rl, 512 policy + 128 rollout GPUs for 1.5. **PR #42** (open, 2026-10-20) stage 3: **DRaFT** (truncated, arXiv 2309.17400) of the expert: soft-min ADE backpropagated through the 10-step Euler sampler, frozen VLM, velocity-field anchor to the initial policy, 8×H100, converges in under 200 optimizer steps and "oscillates afterwards", with a one-time bit-equal parity check of the differentiable sampler against the product sampler. **PR #44** (open, 2026-10-21): training-efficiency chapter 05 "Measurement" | — | — | — | Recipes PRs #42–#45 opened 2026-10-20/21; no RL PR since | [PR #42](https://github.com/NVlabs/alpamayo-recipes/pull/42); [PR #44](https://github.com/NVlabs/alpamayo-recipes/pull/44); [RL README](https://github.com/NVlabs/alpamayo-recipes/blob/main/recipes/alpamayo1_x_rl/README.md) (ledger) (P, H) |

**Inferences**
- **The skill that transfers is cosmos-rl.** A reviewer on the AV Simulation team would recognise its replica, weight-sync, staleness and dispatch semantics as the team's own stack. Measuring them through AlpaGym (L2-1) is the most direct "framework" evidence available without building one.
- **NeMo-RL** adds no capability S4 can use: no AlpaSim, no flow policy, no continuous-action environment.
- **The banner and the October commits do not conflict.** The "limited maintenance" banner covers model, feature and community work. The October commits are infrastructure hardening (dispatch accounting, lookahead, transport) by NVIDIA engineers, so the async core is still maintained.

**Gaps**
- Whether Alpamayo 1.5's production model received any expert RL is not stated anywhere read.
- The NeMo-RL Kubernetes section was not read.
- Megatron-Bridge and Isaac Lab licences were not checked.
- Whether cosmos-rl supports Kubernetes was not checked.

### Q3. Non-NVIDIA frameworks for RL on flow-matching VLAs: rollout/trainer balance and simulator-bound throughput

**Takeaway.**
- **Shared recipe.** The open recipes for flow VLAs converge on **a frozen VLM, RL on a small action expert, ODE→SDE (or a learned noise) for log-probs, and colocated or pipelined placement**.
- **The bottleneck depends on the simulator.** With per-environment rendering or physics simulators (LIBERO, ManiSkill, AlpaSim), environment time dominates and pipelining or disaggregation buys 1.6–2.4×. With batched or render-free simulators (GigaFlow, CaRL), throughput is learner-bound and large batches matter.

| # | Fact | Source (P/S, conf.) |
|---|---|---|
| 3.1 | π_RL (RLinf team; v3): Flow-Noise uses a learnable noise network and the joint chain log-likelihood. Flow-SDE converts the ODE to an SDE with σ = a·√(τ/(1−τ)), makes **one** denoising step stochastic per environment step, and takes the ratio per denoising step. A two-layer MDP gives about a 2× speedup. Run on **8×H100, colocated, serial** env/rollout/actor on the same GPU; VLM frozen, about 300M expert trained; PPO + GAE with the critic on the VLM output for π0.5 | [arXiv 2510.25889v3](https://arxiv.org/abs/2510.25889) (P, H) |
| 3.2 | π_RL ablations (π0 Flow-SDE, LIBERO-Spatial, 100 steps). Noise a = 0.2/0.5/0.8/1.0 gives RL eval 73.1/94.5/**98.1**/88.5. The text says a = 0.2 was unstable with a higher clip fraction. Denoise steps K = 2/4/8 gives RL eval 97.0/94.5/86.7. Headline: π0.5 ManiSkill 40.1 → 90.9 (Flow-SDE) | same (P, H) |
| 3.3 | RLinf-VLA (RSS 2026; v3 of 2026-08-12): a hybrid fine-grained pipeline gives a 1.61–1.88× training speedup on ManiSkill; 98.11% on 130 LIBERO tasks. System design is in arXiv 2509.15965. Search snippets of the paper: 64/128/256 parallel environments for 8/16/32 GPUs on LIBERO, and SimpleVLA-RL supports only colocated mode | [arXiv 2510.06710](https://arxiv.org/abs/2510.06710) (P, H for the abstract; snippet facts M) |
| 3.4 | RLinf README claims up to 2.434× for hybrid mode against existing frameworks (v0.1, 2025-12-17) | ledger 2026-10-03 B1 (P, H) |
| 3.5 | SimpleVLA-RL (2025-09-11): built on veRL, adding VLA trajectory sampling and parallel multi-environment rendering. It stresses that VLA rollouts need multi-round environment interaction, which makes them slower and costlier than LLM rollouts | [arXiv 2509.09674](https://arxiv.org/abs/2509.09674) (S via search snippet, M) |
| 3.6 | veRL rollout-correction (TIS/MIS/Geo-RS; IcePop 0.5/5.0 is an option, not the default) and AReaL (2.77× async, `max_head_offpolicyness`) | ledger 2026-10-03 B9–B11 (P, H) |
| 3.7 | GigaFlow ("Robust Autonomy Emerges from Self-Play", 2025-02-05): batched simulator, **1.6 billion km**, "42 years of subjective driving experience per hour" on **a single 8-GPU node**, no human data, about 17.5 years between incidents in simulation. The abstract does not name the RL algorithm | [arXiv 2502.03349](https://arxiv.org/abs/2502.03349) (P, H) |
| 3.8 | CaRL: 300M samples on 8×A100 (40 GB) plus 108 CPU cores for 1 week (CARLA); 500M samples on nuPlan at 14 ms per step; 128 environments | [arXiv 2504.17838v3](https://arxiv.org/abs/2504.17838) (P, H) |

**Inferences**
- **AlpaSim is the expensive kind.** It is a per-session gRPC simulator with a 3DGS renderer, so S4 is simulator-bound at small trainer scale. NVIDIA's own 5-node preset moves a node from Cosmos to AlpaSim (1.7). L2-1 should therefore model AlpaSim capacity C_sim = GPUs × replicas × concurrent rollouts as the first-order term.
- **No public framework drives AlpaSim.** None of these frameworks runs AlpaSim or Alpamayo. Their value to S4 is as **estimator and placement references**, not as replacements (§5).

**Gaps**
- VLA-RL (arXiv 2505.18719) and OpenRLHF were not fetched in this pass. No numbers are given for them.

### Q4. GRPO estimator math for flow-matching experts, and what AlpaGym implements

**Takeaway.** Among the variants below, AlpaGym implements **Flow-GRPO's ODE→SDE sampler and closed-form KL**, with three differences:
- **A GSPO-like length-normalized ratio.** It takes the mean of per-element log-ratios over all SDE steps and action dimensions, not Flow-GRPO's per-step ratio.
- **A lower noise level.** a = 0.2, against Flow-GRPO's 0.7 and π_RL's best of 0.5–0.8.
- **Small groups.** G = 1, 2 or 6, against Flow-GRPO's 24, where G = 6 and 12 collapsed.

Config cannot switch the ratio granularity, add a critic, use Flow-Noise or apply GDPO. It can reach noise, steps, G, lr, KL, staleness and clip, and possibly `unbiased_advantage` through the TOML route.

| # | Fact | Source (P/S, conf.) |
|---|---|---|
| 4.1 | Flow-GRPO (2025-05-08): ODE→SDE update x_{t+Δt} = x_t + [v + σ_t²/(2t)·(x_t + (1−t)v)]Δt + σ_t√Δt·ε with σ_t = a√(t/(1−t)), **default a = 0.7**. Clipped surrogate with a **per-denoising-step ratio**, group-normalized advantage shared across steps, and a **closed-form KL** between isotropic Gaussians. β = 0.04 (GenEval, OCR) or 0.01 (PickScore). **G = 24 by default; G = 12 and G = 6 were unstable and eventually collapsed.** Training uses T = 10 steps against 40 at inference (over 4× speedup); 5 steps was not consistently faster | [arXiv 2505.05470](https://arxiv.org/abs/2505.05470) (P, H) |
| 4.2 | Flow-GRPO without KL: quality metrics drop (GenEval-task DeQA 2.77 vs 4.01 with KL) and style or diversity collapses on the human-preference task. Occasional reward hacking persists even with KL | same (P, H) |
| 4.3 | π_RL Flow-Noise vs Flow-SDE: see 3.1–3.2 | (P, H) |
| 4.4 | GSPO (2025-07-24): s_i = (π_θ(y_i)/π_old(y_i))^{1/\|y_i\|} = exp(mean token log-ratio), with a clipped sequence-level objective and group-normalized advantage. Clip ranges 3e-4 / 4e-4, against GRPO's 0.2 / 0.27. Token-level ratios accumulate high-variance noise. Clip fractions differ by about two orders of magnitude | [arXiv 2507.18071](https://arxiv.org/abs/2507.18071) (P, H) |
| 4.5 | Dr. GRPO (2025-03-26; v2 2025-10-06) removes (a) the 1/\|o_i\| length normalization ("response-level length bias") and (b) the std normalization, which gives a "question-level difficulty bias" by over-weighting low-variance groups. Its advantage is r − mean | [arXiv 2503.20783](https://arxiv.org/abs/2503.20783) (P, H) |
| 4.6 | FPO "Flow Matching Policy Gradients" (2025-07-28; v2 2025-08-01): an advantage-weighted ratio built from the conditional flow-matching loss, compatible with PPO-clip, avoiding exact likelihoods | [arXiv 2507.21053](https://arxiv.org/abs/2507.21053) (P, M: abstract only) |
| 4.7 | ReinFlow (learnable noise injection; NeurIPS 2025), DSRL (latent-noise RL), DiffusionNFT (likelihood-free, ICLR 2026 Oral; cosmos-rl ships NFT configs for WFMs), DPPO | ledger 2026-10-03 B13–B15 (P, H). DPPO not re-fetched (gap) |
| 4.8 | KL estimators. Schulman's note defines k1 = −log r, k2 = ½(log r)² and k3 = (r − 1) − log r, with r = p/q and x ~ q; k1 and k3 are unbiased, and k3 has low variance and is non-negative. Liu et al. (arXiv 2510.01555) argue that **k3 used as a loss gives a biased gradient** (a first-order approximation); that k1 placed in the reward is the principled form for reverse KL; that k2 as a loss is gradient-equivalent on-policy; and that off-policy k_n losses need importance weights | [Schulman note](http://joschu.net/blog/kl-approx.html) (P; **the fetch failed, so the definitions are standard but the numeric table was not re-verified**); [arXiv 2510.01555](https://arxiv.org/abs/2510.01555) (P via search snippet, M) |
| 4.9 | cosmos-rl token GRPO (pinned `d2a2c57`) uses **k3** on log-ratio ref − cur (clamped ±20), optionally IS-weighted. This is the KL of the **open-loop** text recipe. The expert path in AlpaGym uses the closed-form Gaussian KL (1.17) | grpo_trainer @d2a2c57 (P, H) |
| 4.10 | AR1's own GRPO (Eq. 10) is an unclipped softmax weight exp(βA_i)/Σexp(βA_j) on a mean-centred advantage, plus KL | ledger 2026-10-03 B20 (P, H) |
| 4.11 | GDPO (NVIDIA; ICML 2026): normalize **each reward term separately** within the group, sum the results (optionally weighted), then normalize batch-wise. Example with two binary rewards and G = 2: GRPO maps (0,1), (0,2) and (1,2) all to (−0.7071, 0.7071), whereas GDPO separates them (−1.4142, 1.4142 for (0,2)). Shipped in NeMo-RL on 2026-03-12; not found in cosmos-rl or AlpaGym | [arXiv 2601.05242](https://arxiv.org/abs/2601.05242) (P, H); NeMo-RL README (P, H) |

**Derivation inputs for the ★ estimator row**, CPU-checkable. These are inputs and formulas, not code; the numbers were computed in this session with numpy, double precision.
- **KA-1. Group advantage.** cosmos-rl `d2a2c57`: A = (r − r̄)/(σ_pop(r) + 1e-6), or r − r̄ when `unbiased`.
  - [−10, 0, 0, 0] → **[−1.732051, 0.577350, 0.577350, 0.577350]**. [−5, 0, 0, 0] gives the same.
  - Mean-centred: [−7.5, 2.5, 2.5, 2.5] and [−3.75, 1.25, 1.25, 1.25].
  - [1, 0, 0, 1, 1, 0, 0, 0] → +1.290994 / −0.774597 (matches `03` §4).
- **KA-2. Affine invariance.** A(a·r + b) = A(r) for a > 0. The all-collide group [x − 10 for x in safe] gives exactly the advantages of `safe`.
- **KA-3. Group size 1.** G = 1 → A = 0 for any reward, so the default `n_generation: 1` trains nothing.
- **KA-4. Mixed groups.** P(a group mixes collision and no-collision) = 1 − p^G − (1−p)^G.
  - p = 0.08: G = 2 → 0.1472; 6 → 0.3936; 8 → 0.4868; 16 → 0.7366; 24 → 0.8648.
  - p = 0.04: G = 6 → 0.2172; 24 → 0.6246.
- **KA-5. Geometric-mean ratio.** N = T × 64 × 2 elements; r_norm = exp(m) with m the mean per-element log-ratio; r_joint = exp(N·m).
  - T = 5 → N = 640; one element with log-ratio ln 2 → r_norm = **1.0010836**, r_joint = 2.
  - The mean-ratio clip at 1.2 or 0.8 equals joint exp(**116.6858**) or exp(**−142.8119**).
  - Gradient scale relative to a sum objective = 1/640 = 0.0015625.
  - T = 4 (smoke preset) → N = 512 and exp(93.3486).
- **KA-6. Closed-form per-element KL** = c(t)·Δv², with c(t) = (dt/2)·(s·t/(2(1−t)) + 1/s)², s = a·√(σ/(1 − min(σ, σ_max))), σ = 1 − t, grid linspace(0, 1, T+1).
  - a = 0.2, T = 5, t = 0/0.2/0.4/0.6/0.8: s = 0.447214/0.4/0.244949/0.163299/0.1 and **c = 0.5/0.65025/1.734/3.9015/10.404**.
  - a = 0.5: c = 0.08/0.126562/0.3375/0.759375/2.025. a = 0.7: c = 0.040816/0.079083/0.210888/0.474497/1.265327.
  - Example: a = 0.2, t = 0.4, Δv = 0.1 → **0.01734**.
  - Derivation: equal-variance Gaussians N(μ₁, s²dt) and N(μ₂, s²dt) have KL = (μ₁−μ₂)²/(2s²dt), with μ₁ − μ₂ = Δv·(1 + s²(1−σ)/(2σ))·dt.
- **KA-7. KL estimators.**
  - k3 at r = 0.8 → 0.023144; at r = 1.25 → 0.026856 (as in `03`).
  - Analytic targets for the unbiasedness check: KL(N(0,1)‖N(μ,1)) = μ²/2 → 0.005 at μ = 0.1 and 0.5 at μ = 1.
- **KA-8. Std amplification in all-safe groups.**
  - Progress [0.650, 0.655, 0.660, 0.645, 0.652, 0.648] has σ = 0.004853, giving advantages as large as **±1.7167**.
  - The same group with one collision (−10) gives **−2.2361 (= −√5)** for the collider and +0.445…0.449 for the rest.
  - Mean-centred, that group gives −8.335 vs +1.66–1.675, while the all-safe group gives about ±0.008.
- **KA-9. Successive-halving cost** (for L2-5). Total = Σ_k (n0/η^k)·(r0·η^k) = (K+1)·n0·r0 when η divides evenly.
  - n0 = 9, η = 3, K = 2, r0 = 4 GPU-h: rungs 9×4, 3×12, 1×36 → **108 GPU-h**, i.e. $208–259 on H100 ($1.93–2.40) or $108–173 on RTX PRO 6000 WS ($1.00–1.60).
  - With r0 = 8: 216 GPU-h ($417–518 H100). With n0 = 8, η = 2, K = 3, r0 = 4: 128 GPU-h ($247–307 H100).

**Inferences**
- **Noise level.** AlpaGym's a = 0.2 sits where π_RL saw instability and the smallest RL gain. The value of a also sets the KL scale (KA-6: c(t) falls about 6–12× from a = 0.2 to a = 0.7), so a and β are coupled knobs.
- **Clip width.** A clip of 0.2 on a 640-element geometric mean is about 500–670× wider than GSPO's 3e-4/4e-4 on its comparable normalized ratio. That analogy is loose (continuous Gaussian elements vs discrete tokens) and is offered as a derivation input only.
- **Group size.** n_generation ≤ 6 combines small groups (Flow-GRPO collapsed at G ≤ 12) with closed-loop episodes that cost far more than image samples. G and lr are therefore first-order tuning knobs (L2-5).

**Gaps**
- FPO's exact ratio construction, DPPO, and the numeric bias/variance table of the KL estimators were not re-verified (fetch failure).
- No source tests the GSPO-style chain ratio on a flow expert in closed loop.

### Q5. Reward design and reward hacking in closed-loop driving RL; how penalty magnitudes interact with GRPO normalization

**Takeaway.** Under GRPO's per-group standardization, **only within-group differences of the summed reward matter, and only their ratios**.
- Scaling the whole reward (−10 → −100 on every term) changes nothing.
- A constant shift changes nothing.
- The collision term contributes only in **mixed** groups (KA-2, KA-4).
- In all-safe groups, which are the majority at low collision rates, the advantage is driven by progress and distance-to-GT differences of a few thousandths, **amplified to unit variance** (KA-8). This is a mechanistic route to progress gaming.
- Penalty magnitudes therefore matter only relative to the other terms that vary inside the same groups, and the advantage normalization (std vs mean-centred, or per-term GDPO) decides how safe and unsafe groups are weighted against each other.

| # | Fact | Source (P/S, conf.) |
|---|---|---|
| 5.1 | CaRL (CoRL 2025, Tübingen with NVIDIA-affiliated authors; listed on NVIDIA research pages): the reward is r_t = RC_t·Π p_t − T, i.e. route-completion increments times soft penalty factors in [0, 1] (comfort factors must stay > 0), with hard infractions terminating the episode (T = 1 for collisions and red lights). A Roach-style summed reward **collapsed when the mini-batch grew** 256 → 1024 (DS 34 ± 7 → 2 ± 2) while CaRL's improved (21 ± 11 → 38 ± 3). Observed hacking under the Roach reward: waiting at red lights to collect speed reward at zero speed, then driving slowly. nuPlan training adds a constant survival bonus per frame (fixed 15 s episodes). PPO settings: lr 2.5e-4, clip 0.1, mini-batch 16384, 3 epochs | [arXiv 2504.17838v3](https://arxiv.org/abs/2504.17838); [NVIDIA research page](https://research.nvidia.com/labs/avg/publication/jaeger.dauner.etal.corl2025) (P, H) |
| 5.2 | RAD (3DGS RL; NeurIPS 2025): rewards for dynamic collision, static collision, position deviation > 2.0 m and heading deviation > 40°, **each terminating the episode** because 3DGS renders get noisy after such events. PPO with separate lateral/longitudinal clipping (0.1 / 0.2), GAE γ = 0.9 and λ = 0.95; RL:IL rounds 4:1; 4,305 scenes (3,968 for RL); 32 RTX 4090s. Removing the dynamic-collision reward gives the worst collision rate (0.238). No reward hacking is reported | [arXiv 2502.13144](https://arxiv.org/abs/2502.13144) (P, H) |
| 5.3 | NVIDIA's closed-loop blog reward = AlpaGym `progress_safety` (`collision_any`, not at-fault); no results or throughput numbers published | ledger 2026-10-25 A9; 2026-10-03 §4 (P, H) |
| 5.4 | AR1's open-loop RL reward = λ_L2‖x − x*‖² + λ_coll·I[collision] + λ_jerk·J. Reward hacking on the reasoning side: a reasoning-only reward raised the critic grade while consistency fell 0.62 → 0.53. The sign bug was fixed in #40 (2026-10-06) | ledger 2026-10-03 (P, H) |
| 5.5 | AlpaGym rewards are **episode-level** with MAX-aggregated binary events and a LAST-aggregated progress fraction (Q1 1.20–1.21). `collision_any` includes rear collisions; the challenge treats rear collisions as never at fault | Q1; ledger `04` §5 (P, H) |
| 5.6 | GDPO (Q4 4.11): summed-reward GRPO collapses distinct reward combinations into identical advantages, and per-term normalization fixes this | [arXiv 2601.05242](https://arxiv.org/abs/2601.05242) (P, H) |
| 5.7 | RCPO (ICLR 2019): constrained policy optimization on multiple timescales with a penalty signal and a convergence proof, as an alternative to hand-set penalty weights. The update rule was not read (abstract only) | [arXiv 1805.11074](https://arxiv.org/abs/1805.11074) (P, M) |
| 5.8 | Flow-GRPO: without KL, a single style collapses (diversity loss), and some per-prompt hacking remains even with KL | (P, H) |
| 5.9 | TacitVLA (NVIDIA with Berkeley, Stanford and JHU coauthors, per an aggregator): reasoning text is used as a training signal but not read by the action head at inference; the aggregator reports 52.5% on the public PAI-AV closed-loop challenge split and up to 3.4× lower latency. **No arXiv or primary page found** | [AI Weekly](https://aiweekly.co/alerts/nvidia-tacitvla-matches-vla-baselines-at-34x-lower-latency) (S, L) |

**Mechanism (first principles, an inference from KA-1, KA-2, KA-4 and KA-8 together with 5.5).**
- **Penalty magnitude.** With std normalization, a single collision in a group of G gives that rollout −√(G−1) whatever the penalty magnitude, as long as the penalty dominates the other within-group differences. −10 vs −5 changes nothing there. It matters only when a group mixes event types: off-road vs collision, or collision with high progress vs no collision with low progress.
- **Weight of clean groups.** With per-group std normalization, an all-safe group receives the same total gradient weight as a collision group, so a run at low collision rates is trained mostly on progress and RMSE ranking. That is the Dr. GRPO "difficulty bias" in a driving setting.
- **Mean-centred alternative.** With mean-centring (`unbiased_advantage`), clean groups contribute about 1/1000 of a collision group's signal. That changes which behaviour is optimized at all.
- **Hacking mode depends on the collision definition.** Under **non-reactive replay agents** (`trafficsim` skipped), `collision_any` penalizes being rear-ended after braking, which pushes against legitimate braking. An at-fault-only penalty removes that pressure but makes a stall that triggers rear-ends free. This is the structural reason L2-3 makes the collision definition the single variable, and L2-4 logs `collision_rear` and the stall monitors.

**Gaps**
- No primary Waymo or Wayve closed-loop RL reward-hacking paper was fetched in this pass.
- RCPO's update rule was not read.
- Neither OPTED's teacher reward nor RoaD's details were re-read (both are closed-loop supervised, per the ledger).

### Q6. Hyperparameter tuning on a small budget

**Takeaway.** Knobs, in the order the evidence suggests:
- SDE noise level a (π_RL: a = 0.2 was unstable and 0.5–0.8 best; Flow-GRPO default 0.7);
- group size G (Flow-GRPO: G ≤ 12 collapsed);
- learning rate (NVIDIA's own presets differ by 100×; the geometric-mean ratio scales gradients by 1/N);
- KL β coupled to a (KA-6);
- denoising steps (π_RL: fewer steps raised RL eval; Flow-GRPO: 10 for training vs 40 for inference);
- staleness k (0 vs 4);
- clip (CaRL: 0.1 beat 0.2 for driving PPO).

Successive halving with separate tuning seeds and scenes (ASHA and Eimer), reported with IQM and intervals (rliable), fits in about 108 GPU-h (cost model KA-9, listed with the Q4 known answers).

| # | Fact | Source (P/S, conf.) |
|---|---|---|
| 6.1 | ASHA, "A System for Massively Parallel Hyperparameter Tuning" (MLSys 2020): parallel aggressive early stopping that scales linearly with workers | [arXiv 1810.05934](https://arxiv.org/abs/1810.05934) (P, M: abstract) |
| 6.2 | Eimer, Lindauer, Raileanu (2023): RL hyperparameters strongly affect final performance and sample efficiency; the best configuration depends on the tuning seed (risk of overfitting); HPO tools beat hand tuning at lower cost; **separate tuning seeds from test seeds** | [arXiv 2306.01324](https://arxiv.org/abs/2306.01324) (P, M) |
| 6.3 | Agarwal et al. (NeurIPS 2021 Outstanding Paper): with few runs, report interval estimates, performance profiles and IQM (rliable library) | [arXiv 2108.13264](https://arxiv.org/abs/2108.13264) (P, M) |
| 6.4 | CaRL: driving PPO is sensitive to lr (Atari lr with Roach settings dropped DS 22 → 2), epochs (4 → 3 helped) and clip (0.1) | [arXiv 2504.17838v3](https://arxiv.org/abs/2504.17838) (P, H) |
| 6.5 | AlpaGym's shipped values (Q1): a 0.2, inference_step 5 (4 in the smoke preset), G 1/2/6, lr 1e-6 (default) vs 1e-4 (test run), clip 0.2, KL 0, k 4 | (P, H) |

**Inferences**
- **Rung metric.** Use a per-group statistic that cannot be gamed, logged alongside the training reward, such as the collision rate on tuning scenes, not the reward alone. A reward-only rung would select the hacking configuration.
- **Seed count.** Three seeds per final arm (the plan's rule) is below what Eimer and rliable consider safe for ranking configurations. That is why tuning must not reuse the 441 test scenes or the final seeds.

**Gaps**
- No published tuning study exists for GRPO on a flow expert in closed-loop driving.

### Q7. Systems measurement that stands in for "leads RL framework design"

**Takeaway.** NVIDIA's own artifacts define what to measure, without building anything:
- AlpaGym's `perf/` scopes and the AlpaSim Prometheus "Simulator internals";
- the topology presets' capacity and staleness reasoning (1.7–1.8);
- the Alpamayo team's open PR #44 measurement chapter, which gives GPU-hours from the **mean** wall clock (never the median), the per-rank **severe stall** (arrival spread > 2 s), clean steps and the stall gap, **MFU/HFU (6N/8N)** and measured **OFU** (`nvidia-smi dmon --gpm-metrics`), and a CUDA-event split into arrival/forward/backward/host residual.

A credible bottleneck analysis built on these, with a balance and Amdahl/communication model validated from 1 to 8 GPUs on one node, costs about 64–96 GPU-h.

| # | Fact | Source (P/S, conf.) |
|---|---|---|
| 7.1 | PR #44 (open, 2026-10-21) defines the metrics above; its internal reference data is a 2B model on 16 nodes / 128 GPUs. It says to drop the first ~20 steps and to state the averaging window | [PR #44 file at head 35ea6cc](https://github.com/annz96/alpamayo-recipes/blob/35ea6cca4eb4545d2f8b32eae06a5886909ba02d/docs/training_efficiency/05_Measurement.md) (P, H) |
| 7.2 | Balance equation N_gen·g ≈ N_train·t; about 4 ticks/s per GPU estimated for batched rollout; the expert is compute-bound at B = 32 (22.4 ms per step) | `04` §4 (ledger; estimates, L–M) |
| 7.3 | AlpaSim capacity formula as printed in AlpaGym's topology YAML: total per service = GPUs × replicas per container × concurrent rollouts (12 for 4 GPUs; 24 for 8) | (P, H) |
| 7.4 | Sessions per rollout worker = batch_size × n_generation × prefetch, and the off-policy buffer must exceed the slot count (NVIDIA comment) | (P, H) |
| 7.5 | Earlier `04` §7 E1: open-loop predictions about trainer idle time (preserved) | ledger (P) |

**What the design doc must contain** (inference; the model itself is Huy's ★):
1. **System map at the pins.** Controller, policy replicas, rollout replicas with the simulate pool and shared InferenceEngine (`max_batch_size`), AlpaSim services (renderer, physics, CPU controllers, trafficsim off), transport (disk or NCCL), weight sync (cosmos-rl NCCL), and where the reward is computed.
2. **Throughput model.**
   - t_tick = t_render + t_physics + t_controller + t_policy(B) + RPC + queue.
   - T_episode = (warm-up ticks + `expected_valid_steps`) × t_tick + session overhead.
   - Episode production X_gen = min(S_total, C_sim)/T_episode.
   - Trainer consumption X_train = policy_replicas × train_batch_per_replica / t_update.
   - Balance X_gen ≈ X_train.
   - Staleness lag ≈ queue length / X_train, bounded by k.
3. **Scaling model.** Amdahl S(N) = 1/((1 − f) + f/N), with f fitted from the 2, 4 and 8 GPU points. Communication t_sync ≈ bytes/B_eff per update: the expert in BF16 is about 4.6 GB at 2.3B params (an inference), while Classmethod measured about 8.7 GB (S). Rollout payload bytes per episode come over NCCL. Predictions for the 3- and 5-node presets are stated as **unvalidated**.
4. **Validation.** Predict the held-out single-node configurations and report the error. Use PR #44 metrics and the perf scopes. Report means with their windows; GPU-h from the mean.
5. **Deliverables.** A report and figures, plus an **AlpaSim issue** with the measured simulator-side numbers. AlpaGym accepts no PRs; whether it now accepts issues was not checked.

Cost: see L2-1.

**Gaps**
- No public stage-level AlpaGym profile exists (the ledger finding still holds).
- The 0.42 figure is undefined.
- The bytes per NCCL rollout payload are unmeasured.

---

## 3. Deltas against the earlier ledger (dated, with sources)

| ID | Earlier statement (where) | Delta as of 2026-10-27 | Source |
|---|---|---|---|
| D-1 | AlpaGym code-frozen since 2026-07-08 (`02`, verification B1) | **Confirmed**: HEAD is still `972d160` (2026-07-21); no commit since | [commits](https://github.com/NVlabs/alpagym/commits/main) |
| D-2 | Not established whether any Slurm preset raises scale (2026-10-03 tooling notes, Gaps) | **Resolved.** Presets cover 1 node × 8 GPUs (1/3/4) and 3- and 5-node disaggregated Slurm runs with NCCL transport. Multi-node is config-level (Slurm only). No Kubernetes | Q1 1.6 |
| D-3 | "n_generation 1 (default), 2, 6" (tooling notes) | **Added consequence**: with G = 1 the cosmos-rl advantage is identically 0, so the default trains nothing; real runs need G ≥ 2 | Q1 1.15; KA-3 |
| D-4 | "Trajectory-level clipped ratio" (`02` §3 S4 ★; `04` §2; `03` §4) | **Refined**: exp(mean over T×64×2 per-element log-ratios) per step row, a GSPO-style length-normalized chain ratio. The 1/N gradient scale and the effectively unclipped joint ratio follow | Q1 1.13, 1.16; KA-5 |
| D-5 | "k3 KL" as S4's ★ (`02` §3 S4; `03` §4 S4 row) | **Corrected for the expert path**: AlpaGym's KL is the closed-form Gaussian per-transition KL (the Flow-GRPO form), averaged, and off by default. k3 is the **open-loop token** recipe's KL (cosmos-rl `d2a2c57`). The `03` k3 known answer stays valid for that recipe only | Q1 1.17; Q4 4.9 |
| D-6 | "Versions older than v_k − k dropped" (2026-10-03 KQ1 B21) | **Nuanced**: AlpaGym's trainer filter keeps the freshest batch and only warns about stale rollouts. Dropping happens in the cosmos-rl dispatcher and rollout control (verified on `main` `e82b56f`; unverified at the pin) | Q1 1.19 |
| D-7 | S4 Build "the same, plus comfort/jerk" (`02` §3) | **Not config-only**: AlpaSim `46b401be` has no jerk or comfort scorer, and AlpaGym reward kinds are only `metric` and `distance_to_gt`. It needs an AlpaSim scorer (hours delta) | Q1 1.21 |
| D-8 | "Collision term is `collision_any`, not at-fault" (verification A9) | **Extended**: no `collision_at_fault` metric exists at the pin. At-fault is configurable as `collision_front` + `collision_lateral`. Progress is a LAST-aggregated fraction of the full GT path, and distance-to-GT is an RMSE in metres | Q1 1.20–1.21 |
| D-9 | cosmos-rl "maintenance mode" VERIFIED (KQ1 B7) vs UNVERIFIABLE (tooling notes #82) | **Conflict resolved**: the README banner says no longer under active development / limited maintenance, **and** v0.4.7 shipped on 2026-10-22 with async, dispatch and transport commits through 2026-10-22. AlpaGym's pin `d2a2c57` (2026-05-27) predates 0.4.4 | Q2 table |
| D-10 | cosmos-rl as "AlpaGym's backend" only | **New**: cosmos-rl now ships VLA GRPO (π0.5 with `flow_sde` by default, OpenVLA-OFT, Cosmos Policy), WFM NFT and on-policy distillation | Q2 table |
| D-11 | (not covered) | **New**: NeMo-RL v0.7.0 (2026-07-25) added PPO, GDPO (2026-03) and NeMo-DCR (2026-10-07). No VLA or flow path | Q2 table |
| D-12 | (not covered) | **New**: Isaac Lab documents RLinf, a third party, as its VLA RL post-training route (GR00T N1.5/N1.7; page updated 2026-10-27) | Q2 table |
| D-13 | PR #42 "DRaFT RL on the expert" (verification §0) | **Detailed**: truncated DRaFT, soft-min ADE through the 10-step Euler sampler, frozen VLM, velocity anchor, 8×H100, converges in under 200 steps, plus a bit-equal parity check. Open-loop | Q2 table |
| D-14 | (not covered) | **New**: PR #44 (2026-10-21), NVIDIA's measurement vocabulary (mean-based GPU-h, severe stall, MFU/HFU/OFU, CUDA-event split) | Q7 7.1 |
| D-15 | AlpaGym default preset "8 policy steps, distance_to_gt" (`04` §2) | **Extended**: NVIDIA's test preset trains at **5 Hz** (200 ms), with 22 valid steps, 4cam_1080, lr **1e-4** and G = 6. The policy `input_size` is [320, 512], against 320×576 in `04` §2 | Q1 1.10–1.11 |
| D-16 | Prediction 3 premise ("AlpaGym already runs staleness up to 4") | **Added**: prefetch requires `dp_shard_size == 1`, and NVIDIA's test run turns prefetch off when it uses dp_shard 2. RL presets run AlpaSim with **trafficsim skipped** (non-reactive agents) | Q1 1.8–1.11 |
| D-17 | Classmethod wall-clock as the only public AlpaGym numbers | **Unchanged**: still secondary and the only numbers. NVIDIA's "0.42 baseline" comment is the only first-party performance figure, and its units are undefined | Q1 1.7, 1.22 |

---

## 4. First-principles rationale per finding

**L2-1, bottleneck analysis and scaling model.**
- **Mechanism.** JR2010863 D1 and X1 ask for evidence of designing and running RL infrastructure at scale; a solo budget cannot buy scale. What can be checked is whether Huy can **predict** the throughput of NVIDIA's own framework under configurations he has not run. A model fitted on 2- and 4-GPU points that predicts the 8-GPU split, and that writes down falsifiable numbers for NVIDIA's 3- and 5-node presets, is exactly the reasoning in NVIDIA's own preset comments (1.7–1.8). A reviewer on the AV Simulation team can check it against their internal runs.
- **Why it beats the alternatives.** Building a new engine is ruled out. Profiling one default configuration says nothing about design. A paper survey produces no measurement.
- **Cost.** 64–96 GPU-h and 12–18 h.
- **Risk.** Single-node results may not transfer to the multi-node regime, where NCCL, cross-node payloads and weight sync dominate. That is why the multi-node numbers are labelled predictions and L2-6 is gated.

**L2-2, estimator audit.**
- **Mechanism.** The S4 ★ row names an estimator that differs from the code (D-4, D-5). An RL result is uninterpretable if the estimator is misdescribed. The known-answer tests turn each property of the code into a checkable number: zero advantage at G = 1, the 1/N gradient scale, the KL coefficient schedule.
- **Parity check.** The ratio must be 1 at k = 0 and μ = 1. This is the standard train/inference-mismatch detector (veRL's motivation for rollout correction, ledger B9) and mirrors PR #42's bit-equal sampler check.
- **Cost.** It costs nothing on GPUs and blocks every class of silent failure the plan's rule "a number that beats its prediction is investigated as a bug" is meant to catch.

**L2-3, config-only, one-variable reward arms.**
- **Mechanism.** Under group standardization, only within-group differences and ratios count (KA-1, KA-2). So the informative ablation changes **what varies within groups**, not its magnitude. Switching `collision_any` for at-fault terms changes which behaviours are penalized under non-reactive agents (rear-end exposure), which a magnitude change would not.
- **Why it beats the alternatives.** The plan's comfort/jerk arm needs new scorer code (D-7) and confounds two variables if added with weights.
- **Cost.** Zero GPU delta.

**L2-4, hacking monitors.**
- **Mechanism.** Hacking shows up first as a divergence between the optimized proxy and quantities not in the reward. Logging them at zero scale is a free control channel that turns the planned post-hoc audit into a time series, and it detects the KA-8 progress-gaming route as it happens.
- **Why it beats the alternatives.** Post-hoc audits alone cannot tell when hacking began.
- **Cost.** 1–2 h.

**L2-5, budgeted tuning.**
- **Mechanism.** NVIDIA's own presets disagree with each other (lr 1e-6 vs 1e-4) and with the literature (a = 0.2 vs 0.5–0.8; G ≤ 6 vs G = 24). So the defaults are not a principled baseline. Successive halving spends compute in proportion to evidence. Separate tuning seeds and scenes keep the 441-scene verdict uncontaminated (Eimer).
- **Evidence it produces.** Q3's "hyperparameter tuning" with a recorded design, rather than an anecdote.
- **Cost.** About 108 GPU-h, which should be carved out of S4's training budget.

**L2-6, multi-node validation.**
- **Mechanism.** It is the only direct test of the communication term. It is (c) because it needs multi-node fabric and spend beyond S4's tier.

**L2-7, protocol pinning.**
- **Mechanism.** RL optimizes whatever the training simulator presents: 5 Hz, a 4.4 s horizon, 320×512. A gate verdict at 10 Hz and 20 s then mixes the RL effect with a rate and horizon shift. Recording or pinning it keeps "one variable per experiment" true.
- **Cost.** Zero if the mismatch is recorded.

**L2-8, Lagrangian penalty.**
- **Mechanism.** It replaces the fixed −10 with a multiplier updated from the measured constraint violation. The constraint then targets a rate rather than a hand-set exchange rate with progress.
- **Limits.** Under std normalization the multiplier only bites in mixed groups (KA-4). The checkpoint-boundary approximation also resets optimizer state.
- **Cost.** (c) because it adds an arm.

**L2-9, reactive traffic.**
- **Mechanism.** It is the direct test of the non-reactive-agent hacking mechanism (Q5), and it exercises JR2010863 I2's "traffic models".
- **Cost.** (c) because it adds an arm and needs CAT-K compute, which is unknown.

---

## 5. Rejected options

| Option | Reason |
|---|---|
| Switch S4 to **NeMo-RL** | It has no AlpaSim or closed-loop continuous-control environment and no flow-matching policy path (code search 0 hits). Using it would mean building an environment and model integration, i.e. a new framework |
| Switch S4 to **RLinf / π_RL** (PPO with critic, Flow-Noise) | It supports flow VLAs and PPO, and Isaac Lab endorses it, but there is no AlpaSim connector and no Alpamayo port. Porting is forking and building. Keep it as the estimator reference |
| Bypass AlpaGym with **cosmos-rl's VLA trainer** directly | It duplicates AlpaGym on the same backend and needs a new rollout backend for AlpaSim, i.e. building |
| **Patch AlpaGym** to use per-step ratios, Flow-Noise, a critic, GDPO normalization or DAPO dynamic sampling | Modifying AlpaGym internals is forbidden, and AlpaGym is code-frozen and accepts no PRs. Each item can only be stated as an open question for NVIDIA |
| **PPO with a critic** inside S4 (to cover Q3's "PPO") | It is not reachable through config. Q3's PPO stays unexercised, and the honest coverage is GRPO only |
| **DRaFT** (PR #42 style) in closed loop | AlpaSim (renderer and physics) is not differentiable, and DRaFT in PR #42 is open-loop ADE. Only its parity-check practice is adopted (L2-2) |
| **Joint VLM + expert** closed-loop RL (`cotrain_vlm`) | It is hard-set to False in AlpaGym's converter, so it needs code changes |
| Closed-loop **reasoning reward** | The rollouts generate no reasoning text, and the VLM is frozen (ledger B1) |
| **Multi-node runs inside S4's base budget** | Kind (c) only (L2-6) |
| **Staleness study E6 at scale** | Not scheduled and scope creep. The throughput side is measured inside L2-1, and k is one knob in L2-5 |
| veRL / OpenRLHF / SimpleVLA-RL as substrates | They are LLM or VLA frameworks without AlpaSim, and SimpleVLA-RL is colocated only |
| Citing **TacitVLA** as evidence | Secondary sources only; not load-bearing for S4 |
| **OmniDreams renderer** via `alpasim.wizard_args.renderer` in S4 | Possible at config level, which is relevant to I2's Cosmos world model. But its cost and GPU needs are unknown and it adds a second simulator variable. Defer to a proposal |

---

## 6. Open questions and what would resolve them

| ID | Question | Resolution (cost) |
|---|---|---|
| O-1 | Can cosmos-rl knobs AlpaGym does not expose (`unbiased_advantage`, `advantage_low/high`) be set by editing the generated `cosmos_config.toml` and launching with `execution.resolved_config_path`? Does cosmos-rl `d2a2c57` wire `unbiased_advantage` as `main` does? | A CPU-only AlpaGym config resolution, a TOML diff, and reading `cosmos_rl/reward/*` at `d2a2c57` ($0, about 1 h agent) |
| O-2 | Do zero-scale reward terms reach the trainer and logger metrics? | Inspect the `report_metrics` path in AlpaGym, then confirm in the first real run ($0) |
| O-3 | Is staleness enforcement at `d2a2c57` the same as on `main` (`status.py`, `rollout_control.py`)? | Read the files at the pinned SHA ($0) |
| O-4 | What is the "0.42 baseline" (utilization? episodes/s?) in `shared_cosmos_2_3`? | Measure the AlpaSim GPU utilization ratio in L2-1. Optionally Huy files an AlpaGym issue (an outward action; templates exist since 2026-07-21) |
| O-5 | Does a new AlpaSim scorer (jerk, oscillation, stall) appear in `episode.metrics.aggregated` with no AlpaGym change? | Read AlpaGym `driver_server.py` / `proto_conversion.py` and AlpaSim's in-runtime eval config ($0) |
| O-6 | Are there 8×RTX PRO 6000 or 8×H100 single-node offers with working NCCL P2P on vast.ai at S4 start? Does the NuRec renderer plus trainer fit and run on RTX PRO 6000? | Re-price at S4 start (plan rule). AlpaGym ships `install/check_nccl_p2p.py` |
| O-7 | Does the gate's AlpaSim driver for 1.5 feed 320×576 while AlpaGym trains at 320×512? | Read the AlpaSim driver config at the pinned challenge SHA ($0) |
| O-8 | Did Alpamayo 1.5 receive any closed-loop or expert RL before release? | No primary source found. Only NVIDIA can answer; the recipes README states only the open-loop 640-GPU recipe |
| O-9 | What precision does the rollout worker use vs the trainer recompute, which sets the parity floor? | Read `inference_model.py`, then measure at step 1 (L2-2) |
| O-10 | What is TacitVLA's primary source, and does it use RL? | Find the arXiv listing when it is indexed |
| O-11 | Does NeMo-RL run on Kubernetes, and does cosmos-rl? | Read the NeMo-RL cluster docs K8s section and the cosmos-rl docs ($0) |
| O-12 | Can CAT-K be enabled inside AlpaGym runs through `extra_overrides`, and what GPUs does it need? | Read the AlpaSim trafficsim config at `46b401be` ($0) |
