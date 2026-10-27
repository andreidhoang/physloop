# NVIDIA open AV tooling (AlpaSim, AlpaGym, alpamayo-recipes, TensorRT-Edge-LLM, NuRec) and the production DRIVE AV stack — verified state as of 2026-10-03

Scope and method. I re-checked every claim the PhysLoop docs make about these repos and the production stack against primary sources: repo files, commit history, PRs/issues, NVIDIA docs/blogs/newsroom, NVIDIA forums, Uber IR. I read the docs only (02 §0/§2/§6, 04 "State of the art"/"Renderer bound"/"Trainer bound"/"Tensor contract"/"Path to the real world", 01 §3 C4/C9/C10) and did not edit them. All access dates are **2026-10-03**.

Repository snapshots used (cite these SHAs; GitHub code search indexes only default branches):
- AlpaGym `main` @ `972d160` (last commit 2026-07-21) — https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879
- AlpaSim `main` @ `affc2eab` (last commit 2026-08-18) — https://github.com/NVlabs/alpasim/tree/affc2eab209fa43bdfa2f26c0f8d437922d78a68
- AlpaSim `e2e_challenge` @ `2023534` (last commit 2026-09-18) — https://github.com/NVlabs/alpasim/tree/20235345d7ea9eed5d5156099cf219cd35b2e4dc
- alpamayo-recipes `main` @ `c29dc61` (last commit 2026-09-30) — https://github.com/NVlabs/alpamayo-recipes/tree/c29dc61b7318169d2439f3e9d88e547242e301de
- alpamayo-recipes commit `1d99fc5` (2026-06-30). AlpaGym pins this commit. It is the head of **unmerged** PR #18. https://github.com/NVlabs/alpamayo-recipes/tree/1d99fc50370c96637157455da386323a44663c9d

Status conventions:
- VERIFIED: the primary source says it.
- CORRECTED: the primary source contradicts the claim or materially changes it.
- NOT FOUND: the feature or artifact does not exist upstream in the sources searched.
- UNVERIFIABLE: not checkable from primary sources within this task, or not checked because it is outside scope.
- Negative claims ("nobody has X"): VERIFIED means no counter-evidence in the sources searched as of 2026-10-03.

Quoting: Apache-2.0 repo text and config lines are quoted verbatim. Copyrighted web prose (blogs, press releases, forum posts) is paraphrased (marked ≈), with one short verbatim forum quote where the exact wording is the claim.

---

## 1. AlpaGym (github.com/NVlabs/alpagym): dates, algorithm, SDE config, staleness, prefetch, perf/Prometheus, hardware, reward, VLM training, tick subsampling, published throughput

### Takeaway
AlpaGym does run closed-loop GRPO-style (PPO-clip) updates on Alpamayo 1.5's flow-matching action expert. It uses a per-sample SDE log-prob, staleness of up to 4 steps, rollout prefetch and an opt-in `perf/` module. Four claims need correcting:
- `sde_type` has only one working value (`cps` raises NotImplementedError) and AlpaGym does not expose it.
- The VLM is frozen by default (only the expert trains).
- The expert-RL model code lives on an **unmerged** alpamayo-recipes PR branch.
- The public repo has been code-frozen since the 2026-07-08 sync.

A community blog (Classmethod, 2026-07-19) published coarse wall-clock numbers. No stage-level throughput profile is public.

### Cited Findings
- First public commit is 2026-06-16T16:04:52Z by chenyx09: "Introducing AlpaGym, an E2E closed-loop RL pipeline for AV policies, developed by Nvidia AV research team" (commit `92ba5ce`). [commit](https://github.com/NVlabs/alpagym/commit/92ba5cea2234e5407d0977f7b96a7c854f32e039)
- NVIDIA announced AlpaGym two weeks earlier: Alpamayo 2 Super press release (GTC Taipei, 2026-05-31) and the closed-loop blog (2026-05-31, Ivanovic and Pavone). [press release](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis); [blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
- The public repo is a manual sync of an internal GitLab monorepo.
  - Commit `1327bd6` says "See merge request alpamayo/alpamayo!2422 (cherry picked from alpamayo 0232e34c5)".
  - PR #4 is titled "Manual sync of changes upto 06 26 2026" (merged 2026-07-08).
  - The only later commit is #5 (docs/issue templates, 2026-07-21).
  - [commits](https://github.com/NVlabs/alpagym/commits/main)
- NVIDIA staff opened PRs #6 and #7 ("Make AlpaGym training work E2E on IAD", 2026-08-25) and #8 ("Bump cosmos-rl to 0.4.4", 2026-09-03). All were closed within minutes and none reached `main`. [PR list](https://github.com/NVlabs/alpagym/pulls?q=is%3Apr)
- README: AlpaGym "currently supports the Alpamayo 1.5 model with 10b parameters" and "Current work focuses on throughput and scaling". Cosmos-RL is the trainer; AlpaSim (driven over gRPC) is the environment. [README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/README.md)
- SDE sampling config, verified in `packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml`:
  - `int_method: sde`, `noise_level: 0.2`, `inference_step: 5`, `max_batch_size: 4`, `num_traj_samples: 1`.
  - `input_size: [320, 512]`, 4 cameras, `num_context_frames: 4`, `num_historical_waypoints: 16`, `num_future_waypoints: 64`, `step_dt_us: 100000`.
  - It also notes the VLM prefill "intermittently emits an all-NaN trajectory" and enables retries.
  - [file](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml)
- `sde_type` returns zero code-search hits in AlpaGym and in the whole NVlabs org on default branches. It exists only in `recipes/alpamayo1_x_rl/diffusion/flow_matching.py` at the pinned commit `1d99fc5`:
  - `int_method: Literal["euler", "midpoint", "sde"]`
  - `_sde(..., sde_type="sde")`
  - `elif sde_type == "cps": raise NotImplementedError("CPS is not implemented")`
  - The per-step Gaussian log-probs are averaged over dimensions and over steps (`torch.stack(log_prob_list, dim=1).mean(dim=1)`), giving one log-prob per sample.
  - [file](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/diffusion/flow_matching.py)
- Where the expert-RL code lives:
  - AlpaGym pins `alpamayo1-x-rl` from alpamayo-recipes at `rev = "1d99fc50..."` (`packages/policies/alpamayo_r1/pyproject.toml`). That commit is the last commit of recipes PR #18, "Support RL in AlpaGym", open since 2026-06-15.
  - Recipes `main` has no `models/expert_model/` (only `reasoning_vla`).
  - PR #18's commits port `ExpertModelCosmos`, `ExpertModelRL` and the "SDE log-prob kernel used by diffusion-trajectory GRPO" from AlpaGym.
  - [pyproject](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/pyproject.toml); [PR #18](https://github.com/NVlabs/alpamayo-recipes/pull/18)
- Training objective:
  - `bundle.py`: the trainer consumes "the per-sample CFM SDE log-probability". CoT/reasoning-token replay raises `NotImplementedError("Alpamayo R1 CoT replay is not supported yet...")`.
  - `replay_objective.py` implements `compute_ppo_surrogate` (clipped ratio, mean over valid non-padding rows) plus an optional KL penalty.
  - [bundle.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py); [replay_objective.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/replay_objective.py)
- **The VLM is not trained by default.**
  - `ExpertModelConfig` (pinned commit): `cotrain_vlm: bool = False` ("When False, freeze the VLM") and `stop_grad_from_vlm: bool = True` ("detach VLM outputs before they feed the expert head").
  - AlpaGym's converter hard-sets `EXPERT_DEFAULTS = {"cotrain_vlm": False, "stop_grad_from_vlm": True, ...}`.
  - [config.py](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/models/expert_model/config.py); [converter](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py)
- Staleness, prefetch and other defaults in `packages/host/src/alpagym_host/conf/default.yaml`:
  - `allowed_outdated_steps: 4`, `on_policy: false`.
  - `prefetch_rollout: true` (comment: runs cosmos-rl's `_prefetch_loop`; "Requires dp_shard_size == 1").
  - `grpo_ratio_clip_low/high: 0.2`, `kl_beta: 0.0`, `optm_lr: 1.0e-6`.
  - `rollout.n_generation: 1`, `batch_size: 1`.
  - `expected_valid_steps: 8` ("Per-episode step budget the trainer's packer pads every rollout to").
  - AlpaSim `force_gt_duration_us: 1_700_000` and `control_timestep_us: 100_000`, so `n_sim_steps` = 8 + 17 warmup steps.
  - [default.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml)
- Tick subsampling: undocumented.
  - The PPO surrogate runs over B×T step rows. The packer pads every rollout to `expected_valid_steps` (`T_pack`), and padding rows are masked.
  - That structure suggests every valid policy step is trained. This is an inference from code structure; I did not read `packer.py` end to end.
- `perf/` module (`packages/runtime/src/alpagym_runtime/perf/`):
  - Split into `instrument/` (scopes, markers, CPU/GPU readers, periodic monitors) and `analysis/` (CLI). It is off by default (`perf.enabled: false`).
  - Each worker writes `{run_dir}/perf/alpagym_perf_{role}_{rank}_{pid}.json`.
  - The CLI "auto-discovers AlpaSim's Prometheus telemetry (`metrics_worker_*.prom`...)" and prints it as the "Simulator internals" block.
  - AlpaGym does not itself export Prometheus; it parses AlpaSim's files.
  - The example numbers in the README are illustrative, not measurements.
  - [perf/README.md](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/perf/README.md)
- Hardware (ONBOARDING): recommends "two ... CUDA GPU with at least 40 GB of VRAM (e.g. A6000)" and ~100–150 GB disk. Tested "on 2x50 GB GPUs (RTX 6000 Ada)"; "The 10B model requires two GPUs". [ONBOARDING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md)
- Distillation: ONBOARDING says "In the future, we are planning to release a distillation script that can convert the 10B Alpamyo model to a smaller checkpoint (e.g. 2B)". [ONBOARDING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md)
- Reward presets:
  - `conf/reward/progress_safety.yaml`: `progress` 1.0, `collision_any` −10.0, `offroad` −5.0, `distance_to_gt` −0.01.
  - **The default** is `reward: distance_to_gt` (only `distance_to_gt` −0.01). The README quick-start passes `reward=progress_safety`.
  - [progress_safety.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/reward/progress_safety.yaml)
- Published numbers:
  - NVIDIA's closed-loop blog (2026-05-31) and HF blog (2026-06-01) give no throughput, GPU-hour or result numbers (≈ per fetch). [blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/); [HF blog](https://huggingface.co/blog/drmapavone/nvidia-alpamayo-2)
  - Community numbers (Classmethod, Kanno, 2026-07-19), on AWS g6e.12xlarge using 2 of 4 L40S 48 GB:
    - ~45 GB VRAM per GPU (policy and rollout).
    - Smoke test (1 step) ~10 min; 10-step run ~32 min (≈1.5–3.5 min/step).
    - Initial weight sync ~45 s for ~42 GB; per-step sync ~3 s for ~8.7 GB.
    - ~117 GB disk per run; rewards ≈0.63–0.67 for safe episodes vs ≈−9.5 for collisions.
    - [Classmethod](https://dev.classmethod.jp/articles/nvidia-alpagym-on-aws/)
- AlpaGym's README lists batched GPU inference (`inference/`). The 06-26 sync added "Batched deterministic rollouts". [README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/README.md); [PR #4 commit](https://github.com/NVlabs/alpagym/commit/a092900de0322ce3e5edc21c22158286e00340f5)

### Inferences
- The claim that RL through the expert is "largely done by NVIDIA" (02 §0) stands. The shipped default trains the expert **only**, with VLM gradients stopped. Joint VLM+expert closed-loop RL is a config flag (`cotrain_vlm`) that is neither documented nor exercised in the public config. The open-loop recipe trains only the VLM. So the expert-conditioning drift hazard in 04 applies to the open-loop recipe, not to AlpaGym's default.
- The log-prob is averaged over 64×2 action dimensions and over SDE steps. Any PhysLoop parity test (04: "per-step Gaussian log-density parity within 1e-4") must mirror that convention.
- The 200-tick backprop cost argument in 04 does not describe AlpaGym's default. The default is 8 policy steps per episode, 1 generation per prompt, batch 1 — a smoke configuration. Profiling (E1/G7) must state which preset it measured.
- AlpaGym's expert path depends on an unmerged NVIDIA PR branch plus a code-frozen public mirror. Building on it carries fragility risk: pins can move, or the branch can be rewritten, at the next internal sync.

### Gaps
- Whether internal AlpaGym (GitLab) already has newer throughput work: UNVERIFIABLE (not public).
- Whether any Slurm/IAD preset raises `n_generation`, episode length or batch: I did not read `conf/topology/*` or `run/*` presets.
- `packer.py` and `trainer.py` were not read in full, so the "no subsampling" conclusion is structural, not line-by-line.

---

## 2. AlpaSim (github.com/NVlabs/alpasim): architecture, RolloutSpec/seeds, e2e_challenge, CAT-K, Slurm/enroot, scorers — and does it already provide paired comparison, stratification, test–retest variance, per-rollout validity?

### Takeaway
AlpaSim is a gRPC microservice simulator: runtime hub, renderer (NuRec or OmniDreams), driver, controller, physics, optional CAT-K trafficsim, and an eval module. It has per-rollout scorers, seeds, Slurm and enroot. Seeds in `RolloutSpec` were removed on 2026-03-10 and re-added by an external PR on 2026-08-17; trafficsim ignores the seed, and NVIDIA says NuRec has small run-to-run variation.

Upstream supplies most of the raw ingredients for the plan's G1/G2: per-rollout pass/fail, `dist_to_gt_*` deviation metrics, a 4 m corridor failure rule on the challenge branch, and a Drive-IRT multi-model ranking on identical scenes. It does **not** provide:
- a paired two-checkpoint test with a go/no-go verdict,
- stratified or per-scenario-type reporting,
- seed test–retest variance,
- renderer-fidelity validity records.

### Cited Findings
- Architecture (DESIGN.md):
  - Microservices in Python communicating via gRPC: renderer, physics, controller, runtime (hub that drives the loop and logs), driver (the policy), and an eval module run on the logs.
  - DESIGN.md still lists trafficsim as "(coming soon)". The CHANGELOG shows CAT-K landed 2026-06-29, so DESIGN.md is stale.
  - [DESIGN.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/DESIGN.md)
- `RolloutSpec` (runtime.proto at HEAD) has `scenario_id`, `nr_rollouts`, `session_uuids`, `start_time_offset_us` and `fixed64 random_seed = 5`. Proto comment: "rollout k uses random_seed + k as its session seed"; 0 means random. [runtime.proto](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/grpc/alpasim_grpc/v0/runtime.proto)
- Seed history:
  - CHANGELOG (10.03.26): "`RolloutSpec.random_seed` was replaced by `nr_rollouts`".
  - Re-added by external PR #149 (amtellezfernandez, opened 2026-08-13, merged 2026-08-17). The PR leaves trafficsim's seed unused (it only appears in a log line) and targets reproducible policy behaviour, not bit-exact replay.
  - [CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md); [PR #149](https://github.com/NVlabs/alpasim/pull/149)
- Issue #128 (2026-07-28): an NVIDIA maintainer said ≈ NuRec may have small run-to-run variation, and that the traffic model was not covered by the internal seeding work. He invited the contribution. [issue #128](https://github.com/NVlabs/alpasim/issues/128)
- CAT-K:
  - "SMART/CATK trafficsim integration (29.06.26)", enabled with `trafficsim=catk`.
  - Since the July sync: "Traffic simulation is explicitly disabled by default through `trafficsim=disabled`".
  - Code is in `src/trafficsim/alpasim_trafficsim/catk/`.
  - [CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md)
- Slurm, enroot, MPS:
  - The README points to `src/tools/run-on-slurm`.
  - CHANGELOG (11.08.26): opt-in `wizard.run_method=SLURM_ENROOT`, starting services from squash images.
  - July sync: `wizard.enable_mps=true` (CUDA MPS) and automatic resubmission of timed-out jobs.
  - Code: `src/wizard/alpasim_wizard/deployment/slurm_enroot.py`.
  - [CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md)
- Renderers:
  - Default renderer image is NRE-GA 26.04 (since 11.06.26).
  - OmniDreams through FlashDreams sits behind the same `renderer` endpoint (02.06.26 sync).
  - Camera `height`/`width` are per-camera config fields. Example from `src/wizard/configs/cameras/2cam_2168.yaml`: "This is the original camera resolution, NRE will not do any resizing."
  - Alpamayo drivers "now reject camera images smaller than 320×576 pixels" (11.08.26).
  - [CHANGELOG](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md); [camera config](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/cameras/2cam_2168.yaml)
- Built-in scorers (`src/eval/src/eval/scorers/__init__.py`): Collision, OffRoad, MinDistanceToObstacle, OpenLoopCollision, GroundTruth, MinADE, PlanDeviation, Image and Safety. [scorers](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/__init__.py)
- Metrics and outputs (OPERATIONS.md):
  - `collision_at_fault`, `collision_rear`, `offroad`, `dist_to_gt_trajectory` (MAX over time), `duration_frac_20s`, `avg_dist_between_incidents(_at_fault)`.
  - Aggregates are "mean, std, min, max, quantiles" across rollouts, and videos are organized by violation type.
  - The 05.06.26 CHANGELOG adds `results-summary.json` with per-rollout pass/fail, scene scores, failed-rollout rows and `drive` RPC latency.
  - [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md)
- (d) Validity-type metrics that already exist:
  - `GroundTruthScorer` emits `dist_to_gt_trajectory` and `dist_to_gt_location` (both MAX per rollout), i.e. ego deviation from the log. [ground_truth.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/ground_truth.py)
  - `ImageScorer` emits only `img_is_black` ("More to come...") for one camera. The CHANGELOG (03.02.26) says image-based metrics are not supported in the default in-runtime eval. [image.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/image.py)
  - Error codes include `ROLLOUT_ERROR_CODE_INVALID_SCENE`, `ROLLOUT_ERROR_CODE_INVALID_SCENE_MAP` and rollout retries (`max_rollout_retries`, default 2).
  - On the challenge branch, PR #163 (2026-08-24) logs failure "when leaving the nurec corridor (4m "tunnel") laterally". This is a fixed deviation cap scored as **failure**, not reported separately as invalid. [commit f012862](https://github.com/NVlabs/alpasim/commit/f012862749366a72f356fce73e22091fe5160e41)
- (b) Stratification: code search for "stratif" returns 0 hits on AlpaSim `main`. Aggregation is per metric across all rollouts. [search basis: repo code search, 2026-10-03]
- (a) Paired comparison / (c) test–retest: the challenge `local_evaluation/evaluate.py` (e2e_challenge branch):
  - Fits the "same pinned Drive-IRT algorithm used for the challenge leaderboard". It "averages repeated rollouts for each scene" and outputs a "capability ranking, posterior rank interval, rank spread, and average scene score".
  - It requires identical scored scene IDs across runs and supports `--without-references --run model-a=... --run model-b=...`.
  - The `zoib` fit needs `S + 5N` observations, i.e. "**six subjects minimum**" on 441 scenes. Below that it falls back to an arithmetic average "which produces no posterior rank interval".
  - PAI reference runs are "441 scenes x 3 rollouts" (subjects: alpamayo1, vavam-nonlinear, vavam-linear, alternative_n). PCS anchors are an "arbitrary local scale".
  - [local_evaluation README](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/local_evaluation/README.md)
- Curated split (e2e_challenge, PR #167, Damiano Da Col, external, merged 2026-08-31):
  - `nurec_curated_train` 1761 and `nurec_curated_val` 441, from "2202 of 2523 publicly available scenes".
  - "Road-sharing scenes are grouped and assigned to the same split to avoid leakage."
  - It is for local train/val; the official PAI track uses "an internal set of NuRec-compatible scenes".
  - [PR #167](https://github.com/NVlabs/alpasim/pull/167); [challenge README](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/README.md)
- Challenge contract (e2e_challenge README):
  - "each driver instance must use no more than 16 GiB of VRAM".
  - "target 0.1 seconds or less of model work per `Drive` call", enforced via a per-track throughput budget.
  - 16 replicas across GPUs 4–7, 2 concurrent rollouts each; 6 cameras at 10 Hz; requested 1900×1080 (delivered 1916/1920×1080 JPEGs); nonlinear MPC with optional gain JSON.
  - The reference contract is "10 Hz, 200 simulation steps and 1.7 s of force-GT".
  - [challenge README](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/README.md)
- Challenge operations:
  - The leaderboard was wiped after the 2026-08-31 to 09-06 maintenance ("significant changes ... including the evaluation approach and the tests").
  - Reopening was held pending legal T&C review (2026-09-07). The September submission limit was 3.
  - Branch updates continued: warmup track 2026-09-15 (#183); nuPlan offroad fix 2026-09-14 (#181).
  - [issue #170](https://github.com/NVlabs/alpasim/issues/170); [branch commits](https://github.com/NVlabs/alpasim/commits/e2e_challenge)
- Telemetry:
  - Prometheus on by default since 30.06.26.
  - Each run writes `metrics_plot.png` (RPC durations, rollout/step durations, "Sim seconds per rollout", runtime idle fraction, CPU/GPU utilisation).
  - The Grafana dashboard gained throughput panels in July 2026.
  - [TELEMETRY.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/TELEMETRY.md); [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md)
- Guardrail hook:
  - `SafetyScorer` reads an optional `safety_monitor_safe` field from driver responses and emits `safety_monitor_triggered`. The metric is listed in `src/wizard/configs/base_config.yaml`.
  - No safety-monitor implementation exists in the repo; code search shows only consumers and tests.
  - [safety.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/safety.py)
- Release cadence:
  - CONTRIBUTING says "Approximately once per month, the GitLab repository is synchronized to GitHub".
  - Observed syncs: 2026-03-16 (#48), 06-03 (#96), 07-12 (#116), 08-11 (#145).
  - `main` HEAD is #154, 2026-08-18. No September sync had landed as of 2026-10-03.
  - [runtime.proto history](https://github.com/NVlabs/alpasim/commits/main/src/grpc/alpasim_grpc/v0/runtime.proto); [CONTRIBUTING](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CONTRIBUTING.md)

### Inferences
- **G1's statistical layer is still open, but the plumbing exists.** `results-summary.json` gives per-rollout pass/fail and scene scores on identical scene IDs. A McNemar/clustered-bootstrap layer is a thin analysis add-on, not infrastructure.
- Drive-IRT ranks; it does not test "B is not worse than A". With two subjects it degenerates to averages without uncertainty unless at least four reference subjects are added.
- Seed test–retest is now *possible* (PR #149) but not *reported* anywhere. It is bounded below by NuRec run-to-run variation and by unseeded CAT-K. A test–retest study must report those as noise sources, or run with CAT-K disabled (the default).
- The challenge's 4 m corridor rule shows NVIDIA treats large deviation as **failure**. PhysLoop's G2 rule (report out-of-envelope separately) is a different, defensible convention, but must be explained against this upstream precedent.
- "Render at model resolution" is a camera config change (height/width), bounded by the 320×576 minimum for Alpamayo drivers. It is not a code PR. The official challenge renders ~1080p, so this optimisation cannot apply to challenge numbers.

### Gaps
- Drive-IRT's exact interval type ("95%") is unverified: `evaluate.py` and the drive-irt repo were not read.
- Titles of unread AlpaSim issues relevant to G2: #192 "[BUG] Potential AlpaSim input/integration issue with Alpamayo 1.5 and 2 Super" (2026-09-26) and #193 "Multi-case contact triage after the 4m-window audit" (2026-09-28). Their content was not read.
- I did not check whether `PlanDeviationScorer` or `MinDistanceToObstacleScorer` compute TTC-like quantities.

---

## 3. alpamayo-recipes (github.com/NVlabs/alpamayo-recipes): recipes, GPU split, quantization, pinned versions, distillation

### Takeaway
Recipes at `main` (2026-09-30): SFT for Alpamayo 1, 1.5 and 2 Super (the 2 Super SFT recipe added 2026-09-30); open-loop GRPO; ModelOpt PTQ for Alpamayo 1.5. There is **no distillation recipe**. Several claims check out exactly:
- GRPO local test uses 4 policy GPUs + 1 vLLM rollout GPU.
- NVIDIA's Alpamayo 1.5 run used 512 policy : 128 rollout GPUs.
- Pins are torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, ModelOpt 0.43.0.
- Sizes are BF16 ~22 GB → FP8 ~11 GB → AutoQuant 6.5 BPE ~9 GB.

The quant recipe already quantizes the action expert, measures only open-loop minADE, and publishes no accuracy or latency numbers.

### Cited Findings
- Recipe table:
  - `alpamayo1_sft`, `alpamayo1_5_sft`, `alpamayo1_x_rl` ("open-loop reinforcement learning post-training with Cosmos-RL and GRPO") and `alpamayo1_5_quant` ("FP8 and NVFP4 + FP8 Mixed Precision").
  - The directory `recipes/alpamayo2_sft` also exists (PR #39, 2026-09-30) but is not yet in the table.
  - The "Which Repository" table says this repo is where you "Post-train, quantize or distill", yet no distill recipe exists.
  - [README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/README.md); [commits](https://github.com/NVlabs/alpamayo-recipes/commits/main)
- GRPO recipe (`recipes/alpamayo1_x_rl/README.md`):
  - "The local test config requires at least **5 GPUs**, each with at least 80 GB of VRAM."
  - FAQ table: local test policy "4 GPUs, 1 replica, `dp_shard_size=4`", rollout "1 GPU, 1 replica". Each rollout replica "runs a vLLM engine".
  - Large-scale: "512 policy GPUs and 128 rollout GPUs (640 GPUs total, 80 nodes)", global batch 2560, "what we used to post-train Alpamayo 1.5".
  - Diagnostics: "Rollout too fast (most common)".
  - [README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md)
- The GRPO recipe does not train the expert. FAQ: "The action expert head (flow-matching-based continuous actions) is **not** trained by this RL pipeline. RL post-training for the **action expert pathway** will come in a future release." [README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md)
- Published open-loop recipe numbers:
  - Local test reward −0.28 → −0.21, trajectory L2 1.66 → 1.34, in ~10 min on one 8×H100 node.
  - Prefetch server cuts policy iteration "44 s → 5 s" (one place) versus "44 s → 12 s" (FAQ). The README is internally inconsistent here.
  - [README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md)
- Pins in `recipes/alpamayo1_x_rl/pyproject.toml`: `"torch==2.8.0"`, `"transformers==4.57.1"`, `"vllm==0.11.0"`, cosmos-rl rev `747d1bd`. AlpaGym overrides cosmos-rl to `d2a2c57` ("~88 commits ahead"). [recipe pyproject](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/pyproject.toml); [AlpaGym pyproject](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/pyproject.toml)
- Quant recipe (`recipes/alpamayo1_5_quant/README.md`, added 2026-06-29 via PR #4):
  - Tested with "torch==2.8.0, torchvision==0.23.0, nvidia-modelopt==0.43.0" on a 5090 (CUDA 12) and a B300 (CUDA 13).
  - Table: BF16 ~22 GB 1.00x; FP8 ~11 GB 2.00x; "Autoquant 6.5BPE ~9 GB 2.44x".
  - Formats `fp8`, `nvfp4`, `w4a8_nvfp4_fp8`, `auto`. `--auto_quantize_bits` defaults to 4.8; the README example uses 6.5.
  - `eval.py` "reports per-clip and average minADE metrics" (default 644 clips, 6 samples) plus average evaluation time per clip.
  - **No result values are published.**
  - [README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md)
- The quant recipe covers the expert (`recipes/alpamayo1_5_quant/utils.py` and `quantize.py`):
  - `FP8_CONFIG` enables `"*weight_quantizer"` and `"*input_quantizer"` model-wide and disables only `"*patch_embed*"`.
  - NVFP4 additionally disables `*action_in_proj.encoder.trunk.0*`.
  - AutoQuant `disabled_layers` are `"*lm_head*"`, `"*patch_embed*"`, `"*action_in_proj.encoder.trunk.0*"` and `"*action_out_proj*"`. It scores sensitivity with a teacher-forced flow-matching MSE on the expert's velocity (`v_pred` vs `v_target`).
  - The calibration loop "ensures quantizers in the rollout path (vlm/expert/diffusion-related modules) observe representative activations".
  - [utils.py](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/utils.py); [quantize.py](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/quantize.py)
- Training-efficiency docs (`docs/training_efficiency/`, first public release 2026-09-23, PR #38):
  - Only Overview and Data Loading are written. "05 Measurement" and "06 Methodology" are stubs, the latter announced as "same-window paired runs, bitwise-invariance gates, dose-response ladders".
  - The scope is SFT stages: S1 full SFT; S2 expert-head training with a frozen backbone.
  - [Overview](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/docs/training_efficiency/00_Overview.md)
- Alpamayo 2 Super is described as a teacher "for distillation and quantization into student models that meet in-vehicle latency and safety requirements on DRIVE AGX Thor". [README](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/README.md)

### Inferences
- G6's "including the action expert" is already done as PTQ by NVIDIA. What remains open is closed-loop evaluation of each format, with intervals, plus publication of the numbers. The per-layer AutoQuant scores exist inside the recipe (logged `search_state`) but are not published.
- No distillation procedure has been released ~7 months after NVIDIA first said one was coming (Edge-LLM blog 2026-03-12) and ~4 months after the closed-loop blog (2026-05-31) and AlpaGym ONBOARDING (public from 2026-06-16). G5 stays open. NVIDIA plans a 10B→~2B single-GPU student, so the "NVIDIA ships it" risk for G5 is concrete.

### Gaps
- The Alpamayo 2 Super SFT recipe README was not read (stage list, hardware).
- The quant recipe's actual sensitivity outputs are not public, so I cannot tell which layers AutoQuant 6.5 picks for NVFP4.

---

## 4. NVIDIA statements on closed-loop post-training, teacher→student distillation and Thor deployment

### Takeaway
NVIDIA consistently positions Alpamayo as an offline **teacher** to be distilled into compact in-vehicle models on DRIVE AGX Thor (CES 2026-01-05; Alpamayo 2 Super 2026-05-31). It has said three times that a distillation script or recipe is coming (2026-03-12, 2026-05-31, AlpaGym docs). None had shipped by 2026-10-03.

### Cited Findings
- Closed-loop blog (2026-05-31, Boris Ivanovic and Marco Pavone):
  - ≈ a distillation script to a smaller, single-GPU-compatible checkpoint is planned.
  - GRPO is the default algorithm; the example reward is progress with safety penalties.
  - Recommends ≥40 GB GPUs, tested on 2× RTX 6000 Ada.
  - Mentions an asynchronous distributed RL pipeline.
  - **No** statement about DRIVE AGX Thor, **no** throughput numbers and **no** results.
  - [blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
- HF blog (2026-06-01, Pavone and Ivanovic): describes AlpaGym as high-throughput closed-loop RL with GRPO at release. It lists using pretrained weights as an offline teacher for onboard-ready models as a use case. No numbers and no real-world validation (≈). [HF blog](https://huggingface.co/blog/drmapavone/nvidia-alpamayo-2)
- CES press release (2026-01-05): ≈ Alpamayo models are not meant to run directly in-vehicle. They serve as large teacher models to fine-tune and distill into AV-stack backbones, integrated on Hyperion with DRIVE AGX Thor. [press release](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)
- Alpamayo 2 Super press release (GTC Taipei, 2026-05-31): ≈ designed as a teacher that can be distilled into compact models running on in-vehicle DRIVE AGX Thor compute. It also announces AlpaGym built on AlpaSim and NuRec. [press release](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis)
- TensorRT Edge-LLM blog (2026-03-12, Lin Chai et al.): ≈ an upcoming "Alpamayo 1 workflow" described as a distillation recipe bringing reasoning to the edge. [blog](https://developer.nvidia.com/blog/build-next-gen-physical-ai-with-edge%E2%80%91first-llms-for-autonomous-vehicles-and-robotics/) (fetched via https://developer.nvidia.com/blog/?p=113671)
- Edge-LLM release notes v0.10.0 (2026-08-12), v0.10.1 (2026-09-03) and v0.11.0 (2026-09-29) contain no Alpamayo distillation workflow and no Alpamayo 1.5 support. v0.10.1 only fixes an Alpamayo-R1-10B ONNX export assertion (#166). [releases](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases)

### Inferences
- "Teacher distilled into compact runtime models on DRIVE AGX Thor" (02 §0, 01 C2) is VERIFIED as NVIDIA's positioning. The "planned/forthcoming" distillation is still unshipped after ~7 months (Edge-LLM) and ~4 months (AlpaGym).

### Gaps
- No NVIDIA source gives a distillation timeline. Internal progress is UNVERIFIABLE.

---

## 5. TensorRT-Edge-LLM and Alpamayo on Thor (support, precisions, OOM report, forum statement)

### Takeaway
- **Supported model:** Edge-LLM officially lists only Alpamayo-R1-10B (`alpamayo_r1`), and its docs give no precision for it. Alpamayo 1.5 support was promised for an August release and is absent through v0.11.0 (2026-09-29).
- **"Not available" statement:** NVIDIA's "Alpamayo is not available for AGX Thor currently" (2026-09-07) was posted in the Jetson Thor category. In practice it is partly superseded: a community member ran Alpamayo 1.5 on a Jetson Thor dev kit through PyTorch with 6.5-bit AutoQuant, and Alpamayo-R1 runs on DRIVE AGX Thor through Edge-LLM after NVIDIA support.
- **OOM report:** it was answered by NVIDIA in thread 382501. Thread 382500 got only a community reply.

### Cited Findings
- Supported models (Edge-LLM 0.11.0): one entry, ≈ "Alpamayo (`alpamayo_r1`): text and image/video to reasoning and a trajectory", linking nvidia/Alpamayo-R1-10B. No precisions are listed, and Alpamayo 1.5 and 2 do not appear. [supported models](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html)
- GitHub issue #130 "Alpamayo Support" (opened 2026-07-08, **open**):
  - NVIDIA (linc-nv, 2026-07-30): ≈ Alpamayo 1.5 planned for the August release, with the Python interface already enabled.
  - Users asked for status on 2026-08-12 and 2026-09-15 (≈ it was planned for August but is still not available). No later NVIDIA reply.
  - [issue #130](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130)
- Forum thread 382409 "How to perform Alpamayo1.5 fp8 quantization for Nvidia Thor" (Jetson Thor category, opened 2026-09-04):
  - NVIDIA moderator AastaLLL on 2026-09-07: "Alpamayo is not available for AGX Thor currently." On 2026-09-08 he added ≈ NIM and TensorRT-LLM are not supported on Jetson.
  - Not marked solved; last reply 2026-09-15.
  - [thread](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409)
- Forum thread 382500 "Alpamayo-R1-10B TensorRT engine OOM on DRIVE AGX Thor" (opened 2026-09-06):
  - FP16 engine build failed allocating ~15.17 GB (DriveOS 7.2.5.0).
  - Only reply: community member whitesscott (2026-09-13). He reports ≈ alpamayo1_5 used 14.5 GB on a Thor dev kit with JetPack 7.2.1 using 6.5-bit AutoQuant.
  - No NVIDIA reply in this thread.
  - [thread](https://forums.developer.nvidia.com/t/alpamayo-r1-10b-tensorrt-engine-oom-on-drive-agx-thor/382500)
- Forum thread 382501 "Drive AGX thor with Alpamayo" (DRIVE AGX Thor category, same user, opened 2026-09-06):
  - NVIDIA (SivaRamaKrishnaNV) replied on 09-07, 09-08, 09-09, 09-10 (hugepages / GPU-carveout fix), 09-16 and 09-18.
  - The user reported engines built (09-16), then an empty text-output bug, then on 2026-10-01 ≈ that it now works after NVIDIA's help.
  - [thread p1](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501); [p2](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501?page=2)

### Inferences
- "TensorRT-Edge-LLM runs Alpamayo-R1 on Thor in FP16 only" should read:
  - Edge-LLM officially supports only Alpamayo-R1-10B.
  - Users build FP16 engines on DRIVE AGX Thor.
  - The docs do not state a precision restriction.
- The deployable-precision gap (FP8/NVFP4 for Alpamayo on Thor) is real. It is not documented as a hard FP16-only limit.

### Gaps
- I did not read the Edge-LLM Alpamayo example page (vla.html returned 404 at 0.11.0). Per-precision support for `alpamayo_r1` stays UNVERIFIABLE from primary docs.

---

## 6. NuRec rendering throughput and AlpaSim end-to-end throughput

### Takeaway
- **NuRec:** the "100 FPS at 1080p" six-camera RTX 6000 Pro figure is verified on docs.nvidia.com (page last updated 2026-08-25). The page does not say whether 100 FPS counts per-camera frames or six-camera ticks.
- **AlpaSim:** end-to-end throughput is unpublished. AlpaSim measures it per run (Prometheus, `metrics_plot.png`), and the challenge enforces an unpublished throughput budget.

### Cited Findings
- NuRec "Scaling Neural Reconstruction for Closed-Loop Autonomous Vehicle Simulation" (last updated 2026-08-25):
  - ≈ 3.33 reconstructions per hour and a rendering speed of 100 FPS at 1080p, with six cameras via `car2sim_6cam.yaml` on an RTX6000Pro.
  - ≈ 4–5× reconstruction speed-up on 8 GPUs, non-linear scaling.
  - No AlpaSim end-to-end throughput.
  - [NuRec docs](https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html)
- AlpaSim generates per-run "Sim seconds per rollout" and runtime idle fraction in `metrics_plot.png`, with Prometheus TSDB under `prometheus/`. No benchmark numbers appear in the docs. [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md)
- Challenge throughput: "Submission status reports the observed throughput wall time and the applicable limit". An NVIDIA maintainer said ≈ model time cannot be measured directly, so it is enforced through a throughput limit. The limit value is not published. [challenge README](https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/README.md); [issue #170](https://github.com/NVlabs/alpasim/issues/170)

### Inferences
- 04's "~25 ticks/s per renderer GPU at 4 cameras" assumes 100 FPS = camera-frames. If 100 FPS means six-camera frames per second, the renderer bound is ~6× looser than stated. Measure it before relying on the "renderer sits inside the loop" prediction.

### Gaps
- The FPS definition (aggregate vs per-tick) is UNVERIFIABLE from the page.
- No public AlpaSim scenes/hour or rollouts/hour figure was found anywhere.

---

## 7. Production DRIVE AV stack: Mercedes CLA, S-Class L4, Uber, Hyperion 10, Halos — and what a "Halos-analog guardrail" should mimic

### Takeaway
- **Mercedes:** NVIDIA's own blogs state that DRIVE AV pairs an end-to-end AI stack with a parallel classical stack under Halos, for the Level 2 CLA (2026-01-05) and the L4-ready S-Class (2026-01-29).
- **Uber:** 100,000 vehicles from 2027 (2025-10-28), then 28 cities by 2028 starting in LA/SF in H1 2027 (2026-03-16).
- **Hyperion:** "DRIVE AGX Hyperion 10" is two DRIVE AGX Thor plus 14 cameras, 9 radars, 1 lidar and 12 ultrasonics. NVIDIA renamed DRIVE Hyperion to "NVIDIA Hyperion" in September 2026.
- **Halos:** a full-stack safety *system* (process, platform, tools, inspection lab), not a single runtime monitor. No public NVIDIA source specifies TTC or drivable-area override logic.

### Cited Findings
- Mercedes CLA blog (2026-01-05, Jessica Soares):
  - ≈ DRIVE AV uses an end-to-end AI stack for core driving alongside a parallel classical safety stack, built on Halos, adding redundancy and guardrails.
  - MB.DRIVE ASSIST PRO is described as an enhanced Level 2 system.
  - US production launch expected by end of 2026 (≈).
  - No mention of S-Class, Uber, Hyperion, Thor or Alpamayo.
  - [blog](https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/)
- S-Class blog (2026-01-29, Jessica Soares):
  - ≈ the new S-Class (MB.OS, NVIDIA Hyperion architecture, full-stack DRIVE AV L4 software) is designed to support future robotaxi operations.
  - Its L4-ready architecture combines end-to-end AI with parallel classical driving stacks.
  - Hyperion is developed in accordance with Halos to remove single points of failure.
  - NVIDIA will deliver S-Class rides through Uber.
  - Carries the note that DRIVE Hyperion was renamed NVIDIA Hyperion in September 2026.
  - [blog](https://blogs.nvidia.com/blog/mercedes-benz-l4-s-class-drive-av-platform/)
- Uber, original deal (NVIDIA newsroom, GTC Washington DC, 2025-10-28):
  - ≈ Uber scales its autonomous fleet from 2027, targeting 100,000 vehicles.
  - DRIVE AGX Hyperion 10 = two DRIVE AGX Thor (each >2,000 FP4 TFLOPS), 14 cameras, 9 radars, 1 lidar, 12 ultrasonics.
  - Stellantis to supply at least 5,000 L4 vehicles.
  - The Halos AI Systems Inspection Lab performs independent evaluations.
  - No statement about parallel stacks.
  - [press release](https://nvidianews.nvidia.com/news/nvidia-uber-robotaxi)
- Uber IR update (2026-03-16): ≈ NVIDIA L4 software-driven robotaxis on Uber in 28 cities by 2028, starting in Los Angeles and San Francisco in H1 2027, built on NVIDIA Hyperion and Alpamayo. No fleet number. [Uber IR](https://investor.uber.com/news-events/news/press-release-details/2026/NVIDIA-to-Launch-L4-Software-Driven-Robotaxis-on-Uber-Across-28-Cities-by-2028/default.aspx)
- Rename note: ≈ "the name of NVIDIA DRIVE Hyperion was changed to NVIDIA Hyperion in September 2026", appended to NVIDIA releases. [GTC Taipei release, 2026-05-31](https://nvidianews.nvidia.com/news/nvidia-drive-hyperion-becomes-the-global-platform-for-a-robotaxi-ready-world)
- Halos page (undated in fetch), ≈:
  - A full-stack safety system unifying vehicle architecture, AI models, chips, software, tools and services.
  - Guardrails at design time (DGX), validation time (Omniverse/Cosmos) and deployment time (DRIVE AGX, run-time monitoring and real-time introspection).
  - Halos OS layers: Core = ASIL D DriveOS with hypervisor; SDK; Applications = rule-based safety guardrails incl. an active safety stack; Workflow.
  - Algorithmic safety = a diverse stack combining classical and AI models incl. Alpamayo.
  - ANAB-accredited ISO/IEC 17020 inspection body (2024); TÜV SÜD / TÜV Rheinland assessments; "22,000+ platform safety monitors".
  - [Halos page](https://www.nvidia.com/en-us/ai-trust-center/halos/autonomous-vehicles/)

### Inferences
- A faithful "Halos-analog guardrail" for G4 should mimic only what is public:
  - a rule-based, deployment-time monitor that runs **in parallel** with the E2E policy (stack diversity);
  - it can constrain or override the policy to stay within defined safety bounds;
  - every trigger is logged for introspection.
- The specific mechanism (TTC threshold, drivable-area override) is PhysLoop's own design choice and must be labelled so. It must not be called "Halos".
- AlpaSim's existing `safety_monitor_safe` → `safety_monitor_triggered` path is the natural upstream hook. Nothing requires C++, though C++ may still be a learning goal.

### Gaps
- No primary Mercedes-Benz press release was fetched; the NVIDIA blogs were used instead.
- The Halos page publication date is unknown.
- Whether CLA production vehicles actually shipped with E2E+classical in 2026 is UNVERIFIABLE here.

---

## 8. Contribution realism: can an outsider land upstream PRs in AlpaSim, AlpaGym, alpamayo-recipes?

### Takeaway
- **AlpaSim: realistic.** About 11 external PRs merged, mostly within 0–6 days, including a core runtime change (seeds) that NVIDIA explicitly invited. No CLA or DCO clause.
- **AlpaGym: not realistic today.** Zero external PRs, zero issues, internal MR review, and a public mirror code-frozen since 2026-07-08.
- **alpamayo-recipes: unproven.** Zero external merges. It requires an approved issue before code review and DCO sign-off.

### Cited Findings
- AlpaSim merged PRs (GitHub search, all branches, 2026-10-03): **66** total, 53 by repo collaborators.
- AlpaSim merged PRs by non-collaborator accounts, with create→merge dates:
  - dylantirandaz #66–68 (2026-04-10→04-15; eval/offroad fixes)
  - WCJ-BERT #101 (06-10, same day) and #111 (06-25→08-24)
  - Eashwar93 #112 (06-30→07-06)
  - Dingrui-Wang #144 (08-10→08-12; MPC speed bound)
  - amtellezfernandez #149 (08-13→08-17; seeds)
  - 01Dami23 #167 (08-28→08-31; curated split)
  - hctian713 #161 (08-23→08-24) and #169 (08-31, same day)
  - BorisIvanovic #125 (NVIDIA senior management per README) and jatentaki #98 (affiliation unverified) are also non-collaborator accounts.
  - [PR search](https://github.com/NVlabs/alpasim/pulls?q=is%3Apr+is%3Amerged)
- AlpaSim CONTRIBUTING:
  - "External contributors submit PRs to the GitHub repository; An NVIDIA engineer must approve the PR to trigger CI pipelines".
  - Merged PRs auto-create GitLab MRs; fast-forward merges only.
  - No DCO or CLA clause; code search for "Developer Certificate of Origin" returns 0.
  - [CONTRIBUTING](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CONTRIBUTING.md)
- AlpaSim issues: 94 total. The 25 most recent (2026-08-18 → 2026-10-02) are all closed, mostly within hours to ~5 days, and dominated by challenge logistics. There are no "help wanted" labelled issues. NVIDIA invited the seed feature in #128 (2026-07-28). [issues](https://github.com/NVlabs/alpasim/issues)
- AlpaGym:
  - 8 PRs total, all from NVIDIA staff (chenyx09, karkuspeter, BorisIvanovic). 0 issues via the API.
  - CONTRIBUTING requires DCO sign-off and describes "Two human reviews per MR", with disagreements settled in "an async Slack thread ... Max is the tie breaker".
  - The README routes usage questions to the NVIDIA forum.
  - [CONTRIBUTING](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/CONTRIBUTING.md); [PRs](https://github.com/NVlabs/alpagym/pulls?q=is%3Apr)
- alpamayo-recipes:
  - 27 PRs. External #36 closed **unmerged** (2026-09-12); external #10 open since 2026-06-04; #37 open since 2026-09-16; #13 (Bin-NV, association NONE) open since 2026-06-10.
  - NVIDIA's own #18 has been open since 2026-06-15.
  - CONTRIBUTING: "All enhancement, bugfix, or change requests must begin with the creation of an ... Issue Request", reviewed and approved before code review. "Any contribution which contains commits that are not Signed-Off will not be accepted."
  - [CONTRIBUTING](https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/CONTRIBUTING.md); [PRs](https://github.com/NVlabs/alpamayo-recipes/pulls?q=is%3Apr)

### Inferences
- G7's "≥1 PR merged or reviewed" is realistic **only if aimed at AlpaSim** (e.g., a paired-statistics or stratification extension to `e2e_challenge/local_evaluation`, or eval scorers).
- Targeting AlpaGym, render-at-model-resolution PRs are misdirected (it is AlpaSim camera config), and AlpaGym has no external-merge precedent.

### Gaps
- I did not check whether each external AlpaSim merge targeted `main` or `e2e_challenge`, except #149 (verified on `main`) and the challenge-branch PRs visible in that branch's history (#111, #112, #161, #167, #169).

---

## 9. Claim ledger — every claim in docs 02 (§0, §2 Reuse/New work, §6), 04 (State of the art, Renderer bound, Trainer bound, Tensor contract, Path to the real world), 01 (C4, C9, C10) about these repos and the production stack

### Takeaway
Of 94 rows:
- Most AlpaGym, recipe, challenge and production-stack facts are VERIFIED.
- The material CORRECTIONS:
  - `sde_type` has only one working value (`cps` raises NotImplementedError).
  - The OOM report was answered.
  - "No public RL throughput" should read "no stage-level profile".
  - "Including the action expert" is not new for G6.
  - Episode-length and tensor-contract details (5 SDE steps, mean log-prob, stored SDE chain, `[320, 512]` input) differ from the doc.
  - RolloutSpec seeds have a removal and re-add history.
  - Render-resolution PRs target the wrong repo.
  - "DRIVE Hyperion" was renamed.
- Several items are UNVERIFIABLE (FPS definition, 320×576 resize, Drive-IRT "95%", challenge deadline and email rules) or outside scope.

### Cited Findings

All rows accessed 2026-10-03. Abbreviations:
- AG = AlpaGym @972d160
- AS = AlpaSim main @affc2eab
- E2E = e2e_challenge @2023534
- RC = recipes main @c29dc61
- PIN = recipes @1d99fc5 (PR #18, unmerged)

| # | Claim as written (doc §) | Status | Evidence (≤25 words: quote / config line / path; ≈ = paraphrase of copyrighted prose) | Correct value / note | Primary source |
|---|---|---|---|---|---|
| 1 | "AlpaGym (NVlabs, first commit 2026-06-16)" (02 §0; 04 correction box) | VERIFIED | Commit 92ba5ce, 2026-06-16T16:04:52Z: "Introducing AlpaGym, an E2E closed-loop RL pipeline for AV policies" | Announced 2026-05-31 (press release + blog) before the code went public | https://github.com/NVlabs/alpagym/commit/92ba5cea2234e5407d0977f7b96a7c854f32e039 |
| 2 | AlpaGym "already runs closed-loop GRPO on the action expert" (02 §0; 01 C10) | VERIFIED (nuanced) | bundle.py: "per-sample CFM SDE log-probability"; replay_objective.py `compute_ppo_surrogate` | Expert only (VLM frozen, grads stopped); expert-RL code on unmerged recipes PR #18 | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py |
| 3 | "SDE log-probs (`int_method: sde`…)" (02 §0) | VERIFIED | alpamayo_r1.yaml: `int_method: sde` / `noise_level: 0.2` / `inference_step: 5` | — | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml |
| 4 | "`sde_type ∈ {sde, cps}`" (02 §0) | CORRECTED | PIN flow_matching.py: `elif sde_type == "cps": raise NotImplementedError("CPS is not implemented")` | Only "sde" works; AlpaGym's `DiffusionSamplingConfig` does not expose `sde_type`; absent from all default branches | https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/diffusion/flow_matching.py |
| 5 | "off-policy staleness up to 4 steps" (02 §0; 04 levers table "staleness 4") | VERIFIED | default.yaml: `allowed_outdated_steps: 4` / `on_policy: false` | Cosmos-RL setting | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml |
| 6 | "rollout prefetch" (02 §0) | VERIFIED | default.yaml: `prefetch_rollout: true` — "Requires dp_shard_size == 1." | — | same as #5 |
| 7 | "a `perf/` instrumentation module" (02 §0; 02 §2 G7 Reuse; 01 C10) | VERIFIED | `packages/runtime/src/alpagym_runtime/perf/{instrument,analysis}`; `perf.enabled: false` by default | Opt-in; JSON per worker | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/perf/README.md |
| 8 | "Prometheus merge" (02 §2 G7 Reuse); "Prometheus metrics" (task) | VERIFIED (nuanced) | perf CLI "auto-discovers AlpaSim's Prometheus telemetry (`metrics_worker_*.prom`…)" → "Simulator internals" | Prometheus is AlpaSim's (default since 2026-06-30); AlpaGym parses it | same as #7 |
| 9 | "D4's 'async engine' has a live NVIDIA baseline" (02 §0); "AlpaGym already async, staleness 4" (04 levers) | VERIFIED | Cosmos-RL policy/rollout replicas + `allowed_outdated_steps: 4`; README: "updated weights sync back to the rollout workers" | Default smoke preset: 1 policy + 1 rollout replica | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/README.md |
| 10 | "AlpaGym (closed-loop GRPO, 2 GPUs ≥40 GB)" (04 SotA) | VERIFIED | ONBOARDING: "two … CUDA GPU with at least 40 GB of VRAM (e.g. A6000)"; "The 10B model requires two GPUs" | Tested 2× RTX 6000 Ada | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md |
| 11 | Reward preset "progress 1.0, collision −10, off-road −5, distance-to-GT −0.01" (04 Tensor contract) | VERIFIED (nuanced) | progress_safety.yaml: `progress 1.0`, `collision_any -10.0`, `offroad -5.0`, `distance_to_gt -0.01` | This is the `progress_safety` preset; the **default** is `distance_to_gt` only | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/reward/progress_safety.yaml |
| 12 | "Whether AlpaGym subsamples ticks is undocumented" (04 Trainer bound) | VERIFIED (undocumented) | No doc statement; PPO surrogate over B×T rows padded to `expected_valid_steps` | Code structure suggests no subsampling (inference; packer not read in full) | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/replay_objective.py |
| 13 | "Closed-loop: backpropagating every tick of a 200-tick episode costs ~106 GPU-s" (04 Trainer bound) | CORRECTED (premise) | default.yaml: `expected_valid_steps: 8` (+17 force-GT warmup steps) | AlpaGym default is 8 policy steps per episode; 200 steps is the challenge eval contract | default.yaml as #5; challenge contract in E2E local_evaluation README |
| 14 | "RL that updates only the VLM shifts the KV cache the expert reads… even though AlpaGym now trains the expert" (04) | VERIFIED (clarified) | RC GRPO FAQ: action expert "not trained by this RL pipeline"; PIN `cotrain_vlm: bool = False`, `stop_grad_from_vlm: bool = True` | Drift hazard applies to open-loop recipe; AlpaGym default never updates the VLM | https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/models/expert_model/config.py |
| 15 | AlpaGym "publishes no throughput" (01 C10); "No published end-to-end RL throughput profile" (02 §0) | CORRECTED (partly) | ≈ Classmethod (2026-07-19): 2×L40S, 10 steps ≈32 min (1.5–3.5 min/step), weight sync ≈45 s initial / ≈3 s per step | NVIDIA publishes none; community wall-clock exists; no stage-level profile anywhere | https://dev.classmethod.jp/articles/nvidia-alpagym-on-aws/ |
| 16 | "we profile it with its own perf/ module and send PRs" (01 C10); "AlpaGym PRs (render-at-model-resolution, profiler)" (04 Path) | CORRECTED (target) | AlpaGym: 8 PRs, all NVIDIA; CONTRIBUTING: "Two human reviews per MR"; AlpaSim camera config `height`/`width` | Profiler already exists; render resolution is AlpaSim config; AlpaGym has no external-merge precedent | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/CONTRIBUTING.md |
| 17 | DRIVE AV (Mercedes CLA MB.DRIVE ASSIST PRO) runs E2E stack in parallel with classical safety stack under Halos (02 §0; 01 C9) | VERIFIED | ≈ NVIDIA: DRIVE AV uses an E2E AI stack for core driving plus a parallel classical safety stack built on Halos | CLA is Level 2 ("enhanced level 2"); blog 2026-01-05 | https://blogs.nvidia.com/blog/drive-av-software-mercedes-benz-cla/ |
| 18 | "the L4 S-Class" in the same architecture (02 §0) | VERIFIED | ≈ S-Class L4-ready architecture combines E2E AI with parallel classical driving stacks; Hyperion built per Halos | "L4-ready", for future robotaxi operations; blog 2026-01-29 | https://blogs.nvidia.com/blog/mercedes-benz-l4-s-class-drive-av-platform/ |
| 19 | "the Uber program" in the same architecture (02 §0) | VERIFIED (program) / UNVERIFIABLE (architecture wording) | ≈ 100,000 vehicles from 2027 (2025-10-28); 28 cities by 2028, LA/SF H1 2027 (2026-03-16) | Uber releases don't state parallel stacks; link only via S-Class-on-Uber (2026-01-29) | https://nvidianews.nvidia.com/news/nvidia-uber-robotaxi ; Uber IR 2026-03-16 |
| 20 | DRIVE Hyperion 10 (task) | VERIFIED + SUPERSEDED NAME | ≈ two DRIVE AGX Thor (>2,000 FP4 TFLOPS each), 14 cameras, 9 radars, 1 lidar, 12 ultrasonics | Renamed "NVIDIA Hyperion" in September 2026 | https://nvidianews.nvidia.com/news/nvidia-uber-robotaxi ; rename note in https://nvidianews.nvidia.com/news/nvidia-drive-hyperion-becomes-the-global-platform-for-a-robotaxi-ready-world |
| 21 | Alpamayo "positioned as a teacher distilled into compact runtime models on DRIVE AGX Thor" (02 §0; 01 C2) | VERIFIED | ≈ CES 2026-01-05: teacher models to fine-tune and distill; ≈ A2 Super 2026-05-31: distilled into compact models on DRIVE AGX Thor | — | https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development ; https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis |
| 22 | "a distillation script is 'planned', no public procedure" (02 §0; 01 C2; 02 §0 "No distillation procedure") | VERIFIED (as of 2026-10-03) | AG ONBOARDING: "we are planning to release a distillation script… (e.g. 2B)"; blog 2026-05-31 same (≈) | Also Edge-LLM blog 2026-03-12 (≈ forthcoming distillation recipe); none in RC or Edge-LLM ≤ v0.11.0 | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md |
| 23 | Challenge (NVIDIA AV Research, KE:SAI, OpenDriveLab) "ships a leakage-aware 1,761/441 NuRec split" (02 §0; §2 G1 "e2e_challenge split") | VERIFIED (provenance corrected) | E2E README: `nurec_curated_train` 1761 / `nurec_curated_val` 441; "Road-sharing scenes are grouped… to avoid leakage" | Community PR #167 (merged 2026-08-31); local train/val only; official PAI set is internal. Organizer list not checked | https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/local_evaluation/README.md |
| 24 | "Drive-IRT, an item-response-theory capability score with 95% intervals" (02 §0; §2 G1) | VERIFIED (tool) / UNVERIFIABLE ("95%") | "capability ranking, posterior rank interval, rank spread"; zoib needs "six subjects minimum" on 441 scenes | Below 6 subjects: arithmetic average, no interval | same as #23 |
| 25 | "submissions run in 16 GiB VRAM at 0.1 s model-time per call" (02 §0; §2 G5) | VERIFIED | "no more than 16 GiB of VRAM"; "target 0.1 seconds or less of model work per `Drive` call" | Enforced by an unpublished per-track throughput budget | https://github.com/NVlabs/alpasim/blob/20235345d7ea9eed5d5156099cf219cd35b2e4dc/e2e_challenge/README.md |
| 26 | "Deadline Oct 31 2026 (4-page report, NeurIPS presentation); registration requires an organization email" (02 §0; §6) | UNVERIFIABLE | Not in branch README; HF Space not fetched (outside scope) | Leaderboard reset after 2026-08-31 to 09-06 maintenance; T&C under legal review 2026-09-07 | https://github.com/NVlabs/alpasim/issues/170 |
| 27 | "NVIDIA's PTQ recipe is open-loop only" (02 §0) | VERIFIED | quant README: `eval.py` "reports per-clip and average minADE metrics" (644 clips default) | — | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md |
| 28 | Quant recipe "ModelOpt 0.43: FP8, NVFP4, AutoQuant 6.5 bits; BF16 22 GB→9 GB" (04 SotA) | VERIFIED | "nvidia-modelopt==0.43.0"; BF16 ~22 GB, FP8 ~11 GB, "Autoquant 6.5BPE ~9 GB 2.44x"; formats fp8, nvfp4, w4a8_nvfp4_fp8, auto | AutoQuant default budget is 4.8 bits; 6.5 is the README example | same as #27 |
| 29 | Quant recipe "publishes no accuracy delta or latency" (04 SotA) | VERIFIED | README gives commands and expected log format only; no result values | Tool measures minADE and time per clip | same as #27 |
| 30 | "Alpamayo 1.5 Nano, 10B, public GRPO and NVFP4 recipes" (04 Decision) | VERIFIED (nuanced) | RC: `alpamayo1_x_rl` GRPO (VLM only, open-loop); quant `--quant_format nvfp4` supported | Tested documentation path is FP8 and AutoQuant (FP8+NVFP4) | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/README.md |
| 31 | Open-loop recipe "4 policy + 1 vLLM GPU" (04 prediction 1) | VERIFIED | FAQ: policy "4 GPUs, 1 replica, `dp_shard_size=4`"; rollout "1 GPU, 1 replica" (vLLM engine) | Needs ≥5×80 GB GPUs | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md |
| 32 | "NVIDIA's 512 policy : 128 rollout GPU split" (04 Trainer bound) | VERIFIED (split) / UNVERIFIABLE (doc's 1.9 vs 13 samples/s) | "512 policy GPUs and 128 rollout GPUs (640 GPUs total, 80 nodes)" — used to post-train Alpamayo 1.5 | README says rollout is usually faster than policy ("Rollout too fast (most common)") | same as #31 |
| 33 | Pins "torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, ModelOpt 0.43.0" (CLAUDE.md env facts; task) | VERIFIED | rl pyproject: `"torch==2.8.0"`, `"transformers==4.57.1"`, `"vllm==0.11.0"`; quant README `nvidia-modelopt==0.43.0` | AlpaGym pins cosmos-rl d2a2c57; recipe pins 747d1bd | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/pyproject.toml |
| 34 | "TensorRT-Edge-LLM runs Alpamayo-R1 on Thor in FP16 only" (02 §0) | CORRECTED (partly) / UNVERIFIABLE ("only") | Supported models list only `alpamayo_r1` (Alpamayo-R1-10B); no precision listed | FP16 engine builds reported by users on DRIVE AGX Thor (DriveOS 7.2.5.0); A1.5 promised for August, absent ≤ v0.11.0 | https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/getting_started/supported-models.html ; https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130 |
| 35 | "…with an unanswered OOM report" (02 §0) | CORRECTED | ≈ 382501: NVIDIA replies 09-07 to 09-18; hugepages/carveout fix; engines built 09-16; user reports it working 10-01 | 382500 (cross-post) got only a community reply (09-13) | https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501 ; https://forums.developer.nvidia.com/t/alpamayo-r1-10b-tensorrt-engine-oom-on-drive-agx-thor/382500 |
| 36 | Alpamayo "not available for AGX Thor currently" (NVIDIA forum) (04 SotA) | VERIFIED (quote) / SUPERSEDED in practice | AastaLLL (NVIDIA), 2026-09-07: "Alpamayo is not available for AGX Thor currently." (Jetson Thor category) | Community ran A1.5 on Jetson Thor dev kit (14.5 GB, AutoQuant 6.5, 09-13); A-R1 on DRIVE Thor via Edge-LLM works (10-01) | https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409 |
| 37 | NuRec "renders 100 FPS at 1080p × 6 cams on RTX 6000 Pro" (04 SotA; Renderer bound) | VERIFIED | ≈ 3.33 reconstructions/hour and 100 FPS at 1080P, six cameras (`car2sim_6cam.yaml`), RTX6000Pro | Page last updated 2026-08-25 | https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html |
| 38 | "~25 ticks/s per renderer GPU at 4 cameras, ~3× faster than the policy GPU" (04 Renderer bound) | UNVERIFIABLE | Page does not say whether 100 FPS counts per-camera frames or 6-camera frames | If per-tick, the bound is ~6× looser | same as #37 |
| 39 | "Alpamayo 1.5 resizes inputs to 320×576" (04 Renderer bound); `images [B,4,4,3,320,576]` (Tensor contract) | UNVERIFIABLE (conflict) | AS CHANGELOG 11.08.26: Alpamayo runs "reject camera images smaller than 320×576"; AG config `input_size: [320, 512]` | 4 cams × 4 frames VERIFIED (AG config) | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md ; AG policy yaml as #3 |
| 40 | "rendering at 1080p wastes ~11× the pixels" (04 Renderer bound) | VERIFIED (arithmetic, conditional) | 1920×1080 / (576×320) = 11.25 | Holds only if the model consumes 320×576; challenge renders 1900×1080 request, 6 cams, 10 Hz | E2E README as #25 |
| 41 | "AlpaSim throughput unpublished" (04 SotA; Renderer bound) | VERIFIED | OPERATIONS: per-run `metrics_plot.png` "Sim seconds per rollout"; no benchmark values in docs | Challenge budget value unpublished | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/OPERATIONS.md |
| 42 | "Render at model resolution… prediction ≥4× scenarios per renderer GPU" as an upstream PR target (02 §2 G7; 04 levers/E4) | CORRECTED (mechanism) | `src/wizard/configs/cameras/*.yaml` per-camera `height:`/`width:` | Config change in AlpaSim, floor 320×576 for Alpamayo; prediction itself unmeasured | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/configs/cameras/2cam_2168.yaml |
| 43 | AlpaSim + NuRec "(3DGS, gRPC…)" (04 SotA); C4 "NuRec rebuilds real scenes with 3D Gaussian Splatting and renders novel views" (01 C4) | VERIFIED (gRPC, novel views) / UNVERIFIABLE (3DGS for NuRec) | DESIGN.md: microservices "communicate via gRPC"; NuRec neural reconstruction renderer | MTGS (nuPlan track) uses gsplat kernels (E2E commit 42116f8); NuRec internals not checked | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/docs/DESIGN.md |
| 44 | "AlpaSim wires renderer, traffic, controller and policy over gRPC" (01 C4) | VERIFIED | DESIGN.md: runtime hub, renderer, trafficsim, controller, physics, driver; eval runs on logs | Trafficsim (CAT-K) optional and disabled by default; DESIGN.md stale ("coming soon") | same as #43 |
| 45 | G1 Reuse "AlpaSim scorers" (02 §2) | VERIFIED | Scorers: Collision, OffRoad, MinDistanceToObstacle, OpenLoopCollision, GroundTruth, MinADE, PlanDeviation, Image, Safety | — | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/__init__.py |
| 46 | G1 Reuse "`RolloutSpec` seeds" (02 §2) | VERIFIED (with history) | runtime.proto: `fixed64 random_seed = 5;` — "rollout k uses random_seed + k" | Removed 2026-03-10; re-added 2026-08-17 by external PR #149; trafficsim ignores seed; NuRec small run-to-run variation | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/grpc/alpasim_grpc/v0/runtime.proto ; https://github.com/NVlabs/alpasim/issues/128 |
| 47 | G1 Reuse "Drive-IRT" (02 §2) | VERIFIED | E2E local_evaluation: "fits the same pinned Drive-IRT algorithm used for the challenge leaderboard" | Ranking with posterior rank interval; ≥6 subjects for zoib | same as #23 |
| 48 | G1 Reuse "CAT-K traffic" (02 §2; 01 C5) | VERIFIED (default off) | CHANGELOG: `trafficsim=disabled` by default; "select `trafficsim=catk` to enable CATK" (CAT-K added 2026-06-29) | Seed not consumed by trafficsim (PR #149) | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/CHANGELOG.md |
| 49 | G1 New "Paired regression rule (per-scenario McNemar + log-clustered paired bootstrap)" (02 §2) | NOT FOUND upstream (gap open) | No paired test in AS aggregation (mean/std/quantiles) or E2E evaluate.py (Drive-IRT ranking / average fallback) | Per-rollout pass/fail in `results-summary.json` supplies inputs | OPERATIONS.md; E2E local_evaluation README |
| 50 | G1 New "stratification by scenario type/ODD tag" (02 §2) | NOT FOUND upstream | AS code search "stratif": 0 hits | — | repo code search 2026-10-03 |
| 51 | G1 New "failure taxonomy" (02 §2) | PARTLY exists | Metrics `collision_at_fault`, `collision_rear`, `offroad`, `open_loop_collision`; "Videos… organized by violation type" | No stall/oscillation/red-light labels found | OPERATIONS.md |
| 52 | G1 New "explicit allowed claims list" (02 §2) | NOT FOUND upstream | PCS: "These numbers are an arbitrary local scale" (closest upstream caveat) | Plan's own artifact | E2E local_evaluation README |
| 53 | G1 "Done when… seed test–retest variance" (02 §2) | NOT FOUND upstream | evaluate.py "averages repeated rollouts for each scene" | Seeds now selectable; no test–retest reporting exists | E2E local_evaluation README |
| 54 | G2 Reuse "NuRec" (02 §2) | VERIFIED | CHANGELOG 11.06.26: renderer image `nvcr.io/nvidia/nre/nre-ga:26.04` | — | AS CHANGELOG |
| 55 | G2 Reuse "OmniDreams renderer" (02 §2) | VERIFIED | README: OmniDreams "video-model rendering through FlashDreams"; same `renderer` endpoint (02.06.26) | — | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/README.md |
| 56 | G2 Reuse "Instant NuRec" (02 §2) | UNVERIFIABLE | Not checked (outside scope) | — | — |
| 57 | G2 New "Rendering-fidelity vs ego-deviation curve → deviation cap" (02 §2; 01 C5 validity record) | PARTLY upstream; calibration open | `dist_to_gt_trajectory`/`dist_to_gt_location` (MAX); E2E #163: failure when "leaving the nurec corridor (4m "tunnel") laterally" | Fixed 4 m cap scored as failure, not calibrated to fidelity; no separate reporting | https://github.com/NVlabs/alpasim/commit/f012862749366a72f356fce73e22091fe5160e41 |
| 58 | G2 New "renderer agreement" (02 §2) | NOT FOUND upstream | No NuRec-vs-OmniDreams agreement metric among scorers | Both renderers available behind one endpoint | AS scorers |
| 59 | G2 New "on-rails vs CAT-K agreement" (02 §2) | NOT FOUND upstream | No such metric; trafficsim disabled by default | — | AS scorers/CHANGELOG |
| 60 | G2 New "invalid-rollout rate" (02 §2) | PARTLY upstream | `results-summary.json` failed-rollout rows; `ROLLOUT_ERROR_CODE_INVALID_SCENE(_MAP)`; `img_is_black` scorer | Not a validity-envelope rate; image metrics not in default in-runtime eval | AS CHANGELOG; image.py |
| 61 | G4 Reuse "AlpaSim controller" (02 §2) | VERIFIED | CHANGELOG 26.01.26: linear MPC default, nonlinear selectable; challenge uses nonlinear with optional gain JSON | — | AS CHANGELOG; E2E README |
| 62 | G4 New "Halos-analog rule-based monitor (TTC / drivable-area override, C++)… intervention rate" (02 §2; 01 C9) | NOT FOUND upstream (hook exists) | safety.py: reads `safety_monitor_safe`, emits `safety_monitor_triggered` (in base_config) | No implementation; no public NVIDIA spec of TTC/drivable-area logic (plan's design) | https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/eval/src/eval/scorers/safety.py |
| 63 | G5 Reuse "Alpamayo 2 Super / 1.5 as teachers" (02 §2) | VERIFIED | RC README: A2 Super as "teacher model for distillation and quantization into student models… on DRIVE AGX Thor" | A2 Super SFT recipe added 2026-09-30 | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/README.md |
| 64 | G5 Reuse "AlpaGym" (02 §2) | VERIFIED (fragile) | AG pins `alpamayo1-x-rl` @ `1d99fc5`, head of open PR #18 "Support RL in AlpaGym" | Public AG code frozen since 2026-07-08 sync | https://github.com/NVlabs/alpamayo-recipes/pull/18 |
| 65 | G5 Reuse "OPTED/RoaD ideas" (02 §2) | UNVERIFIABLE | Not checked (outside scope) | — | — |
| 66 | G5 New "Distill to a ≤3B student fitting 16 GiB and ≤0.1 s" (02 §2) | NOT FOUND upstream (gap open; NVIDIA plans it) | AG ONBOARDING: planned script to "a smaller checkpoint (e.g. 2B)… on a single GPU" | Risk: NVIDIA may ship first | AG ONBOARDING |
| 67 | G6 Reuse "ModelOpt recipe" (02 §2) | VERIFIED | `recipes/alpamayo1_5_quant` (PR #4, merged 2026-06-29) | — | RC quant README |
| 68 | G6 Reuse "FlashDrive" (02 §2; 02 §0; 01 C10) | UNVERIFIABLE | Not checked (outside scope) | — | — |
| 69 | G6 New "FP8/NVFP4 including the action expert" (02 §2) | CORRECTED (not new) | utils.py FP8: `"*weight_quantizer"`, `"*input_quantizer"` model-wide, only `"*patch_embed*"` disabled; AutoQuant disables `*action_out_proj*`, `*lm_head*` | NVIDIA PTQ already quantizes expert layers; AutoQuant scores on flow-matching MSE | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/utils.py |
| 70 | G6 New "closed-loop evaluation on G1 with intervals" (02 §2) | NOT FOUND upstream (gap open) | Recipe eval is minADE only; no published values | — | RC quant README |
| 71 | G6 "per-layer sensitivity" (02 §2 Done when) | PARTLY upstream | AutoQuant computes per-layer scores (`search_state` logged) | Not published | utils.py |
| 72 | G7 New "First public end-to-end RL profile" (02 §2) | PARTLY open | Only coarse community wall-clock (Classmethod 2026-07-19) | Stage-level (render/policy/trainer) profile still unpublished | https://dev.classmethod.jp/articles/nvidia-alpagym-on-aws/ |
| 73 | G7 New "targeted upstream PRs (… batching)" (02 §2) | CORRECTED (partly exists) | AG README: "batched GPU inference (`inference/`)"; policy `max_batch_size: 4`; 06-26 sync "Batched deterministic rollouts" | Batching knob exists; scaling study is the gap | AG README; AG policy yaml |
| 74 | G7 "Done when… ≥1 PR merged or reviewed" (02 §2) | Realistic for AlpaSim only | AS: ~11 external merges, mostly 0–6 days; AG: 0 external PRs | — | AS/AG PR searches |
| 75 | "No study correlates AlpaSim or Drive-IRT scores with any held-out or independent outcome" (02 §0; 01 §0, C8) | VERIFIED (absence) | No such study in AS/E2E docs, NVIDIA blogs (HF blog: no real-world validation discussion ≈), web search | — | https://huggingface.co/blog/drmapavone/nvidia-alpamayo-2 |
| 76 | "No go/no-go release rule" (02 §0; 01 §0) | VERIFIED (absence) | AS aggregation = descriptive stats; Drive-IRT = ranking | — | OPERATIONS.md; E2E local_evaluation README |
| 77 | "No evaluation of policy + guardrail" (02 §0) | VERIFIED (absence) | Only consumer hook `safety_monitor_triggered`; no monitor implementation | — | safety.py |
| 78 | §6 risk "NVIDIA publishes its own gate/correlation study" (02 §6) | NOT FOUND (as of 2026-10-03) | None found; RC training-efficiency "06 Methodology" (paired runs, gates) is a stub, training-perf scope | Watch RC docs and AS syncs (next ~monthly sync overdue) | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/docs/training_efficiency/00_Overview.md |
| 79 | §6 risk "Challenge registration blocked (no org email)" (02 §6) | UNVERIFIABLE | Registration rules not checked (HF Space) | — | — |
| 80 | Alpamayo 1.5 Nano "8.2B VLM + 2.3B flow expert, OpenMDW-1.1, 24 GB" (04 SotA) | UNVERIFIABLE (partly consistent) | RC README states 8.2B + 2.3B and "24GB" for **Alpamayo 1 Nano**; 1.5 listed as "10B" | License not checked; AG ONBOARDING: ~21 GB weights | RC README |
| 81 | Alpamayo 2 Super "32B Cosmos 3 Reasoner + 2.3B expert, 1×H100 80 GB, AlpaSim 1.50, minADE6 0.911 m" (04 SotA) | UNVERIFIABLE (partly consistent) | RC README: "34B parameters, built on the Cosmos 3 backbone" | 32+2.3≈34 consistent; other numbers not checked | RC README |
| 82 | "Cosmos-RL (fully async, NCCL/Redis sync; maintenance mode)" (04 SotA) | UNVERIFIABLE | No "maintenance" in cosmos-rl README (code search); AG pins cosmos-rl d2a2c57; NVIDIA PR to bump to 0.4.4 (2026-09-03) | Backend of AlpaGym and recipes | https://github.com/nvidia-cosmos/cosmos-rl |
| 83 | Other 04 SotA items (GR00T, π0.5, Cosmos 3 Edge/Nano, DreamDojo, Newton/Isaac Lab, RoboWorld, OmniDreams 68 FPS, vla-eval, RLinf v0.3, RL-VLA³, verl, AReaL, π_RL, inference levers) | UNVERIFIABLE | Not checked (outside this researcher's scope) | — | — |
| 84 | Tensor contract `vision_tokens [B,2880,4096]` (04) | UNVERIFIABLE | 2,880 = 16 images × 180 tokens assumes 320×576 | At AG's `[320, 512]` the doc's formula gives 2,560 | AG policy yaml |
| 85 | `egomotion [B,16,6]` (04) | PARTLY VERIFIED | AG: `num_historical_waypoints: 16`; history passed as `ego_history_xyz` + `ego_history_rot` | 6-dim encoding not checked | AG policy yaml |
| 86 | `a_t [B,64,2]` (accel, curvature) (04; 01 C1) | PARTLY VERIFIED | AG `num_future_waypoints: 64`; quantize.py: action `(B, n_diffusion_tokens, 2)` | Accel/curvature semantics not checked | AG policy yaml; RC quantize.py |
| 87 | "10 Euler steps δt=0.1" (04; 01 C1 "~10 Euler steps") | CORRECTED (for AlpaGym) | FlowMatching default `num_inference_steps: int = 10`, `int_method="euler"`; AG rollouts `int_method: sde`, `inference_step: 5` | δt = 0.2 in flow time under AG; `step_dt_us: 100000` is waypoint spacing | PIN flow_matching.py; AG policy yaml |
| 88 | `eps_t [10,B,64,2]` stored per step for the policy ratio (04) | CORRECTED | inference_model.py persists "SDE trace fields… (`samples_list` / `timesteps`…)" | Stores the sample chain (steps+1 states), not eps; log-probs recomputed at train time | https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/inference_model.py |
| 89 | `logp_t [10,B]` fp32 (04) | CORRECTED | flow_matching.py: `log_prob = torch.stack(log_prob_list, dim=1).mean(dim=1)`; computed in fp32 (autocast disabled) | One log-prob per sample ([B]), mean over steps and dims | PIN flow_matching.py |
| 90 | `policy_version [B]` (04) | UNVERIFIABLE | Cosmos-RL weight-version internals not checked | — | — |
| 91 | "RLinf issue #1595 (unified serving layer)" (04 Path) | VERIFIED | "[Feature]: unify external model-serving lifecycle and transport" — open, opened 2026-09-22 | — | https://github.com/RLinf/RLinf/issues/1595 |
| 92 | "AlpaSim E2E Closed-Loop Challenge 2026" as a path (04 Path) | VERIFIED (exists) | E2E README: register via HF Space; two tracks (PAI, nuPlan); warmup tracks added 2026-09-15 | Leaderboard reset 2026-09; T&C legal review | E2E README; issue #170 |
| 93 | "Lucid, JLR, Uber, Berkeley DeepDrive are named Alpamayo endorsers" (04 Path) | VERIFIED (wording softer) | ≈ CES 2026-01-05: these companies "are showing interest" in Alpamayo | "Interest", not endorsement | https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development |
| 94 | Nebius hackathon, GTC 2027 poster (Nov 10 2026), arXiv/RSS/CoRL/MLSys, NVIDIA Inception (04 Path) | UNVERIFIABLE | Not checked (outside scope) | — | — |

### Inferences
- The highest-impact corrections for planning:
  - G6's novelty claim is gone; only closed-loop evaluation and publication remain.
  - G7's PR target is wrong repo/mechanism.
  - The AlpaGym cost model (04 Trainer bound, Tensor contract) must be redone against the shipped defaults: 8-step episodes, 5 SDE steps, mean log-prob, `[320, 512]` input.
  - The "unanswered OOM" and "FP16 only" framing overstates NVIDIA's absence on Thor.

### Gaps
- Rows 26, 56, 65, 68, 79–83, 90 and 94 need other researchers or later checks.

---

## 10. Gap table — plan deliverables vs what upstream already ships (as of 2026-10-03)

### Takeaway
- **G1:** the statistical decision layer (paired verdict, stratification, test–retest, allowed claims) is still open, and upstream now provides nearly all inputs. It is worth building as a thin, upstreamable layer on AlpaSim's `local_evaluation`.
- **G2:** the fidelity-calibrated cap and agreement metrics are open; deviation metrics and a fixed 4 m corridor already exist.
- **G4:** a hook exists, the monitor does not.
- **G5:** open; NVIDIA has announced the same thing.
- **G6:** expert quantization is done by NVIDIA; closed-loop evaluation with intervals and published numbers are open.
- **G7:** instrumentation is fully upstream; only a published stage-level profile is open. PRs must go to AlpaSim, not AlpaGym.

### Cited Findings

| Deliverable | Upstream already ships (date) | Still open | Evidence |
|---|---|---|---|
| **G1 Gate v1** | Per-rollout scorers and `results-summary.json` with per-rollout pass/fail and scene scores (2026-06-05). `RolloutSpec.random_seed` (2026-08-17, external PR #149). Curated 1761/441 leakage-aware split (2026-08-31, community). Drive-IRT local multi-model ranking on identical scenes with posterior rank interval; ≥6 subjects for zoib; PAI references at 441×3 rollouts. CAT-K optional (2026-06-29) | Paired two-checkpoint verdict (McNemar / clustered bootstrap) with go/no-go. Stratification by scenario type/ODD (0 code hits). Seed test–retest variance (repeats are averaged; trafficsim unseeded; NuRec small run-to-run variation). Per-stratum Clopper–Pearson bounds. Extended taxonomy (stall/oscillation/red-light). Allowed-claims list | AS runtime.proto; AS OPERATIONS.md; E2E local_evaluation README; AS issue #128; AS PR #149 |
| **G2 Validity envelope** | Per-rollout `dist_to_gt_trajectory` and `dist_to_gt_location` (MAX). Challenge 4 m lateral NuRec-corridor failure (2026-08-24, #163). `img_is_black` only (not in default in-runtime eval). Invalid-scene/map error codes and retries. NuRec and OmniDreams behind one `renderer` endpoint (2026-06-02). Seedable rollouts | Rendering-fidelity vs deviation curve and a calibrated cap. Out-of-envelope rollouts *reported separately* (upstream scores as failure). Renderer agreement (NuRec vs OmniDreams). On-rails vs CAT-K agreement. Envelope-level invalid-rollout rate | AS ground_truth.py; image.py; E2E commit f012862; AS CHANGELOG |
| **G4 Guardrail-in-the-loop** | Eval hook: optional `safety_monitor_safe` in driver responses → `safety_monitor_triggered` (MAX per rollout), listed in base_config. Linear and nonlinear MPC controllers. Public NVIDIA architecture: E2E + parallel classical stack under Halos (CLA 2026-01-05; S-Class 2026-01-29) | Any monitor implementation (TTC / drivable-area override). Override path into controller or trajectory. Policy-alone vs policy+monitor scoring. Intervention-rate reporting. No public NVIDIA spec of Halos runtime-monitor logic, so the mechanism is PhysLoop's own design | AS safety.py; NVIDIA CLA and S-Class blogs; Halos page |
| **G5 Teacher→student distillation** | Teacher positioning (CES 2026-01-05; A2 Super 2026-05-31). A2 Super SFT recipe (2026-09-30). AlpaGym expert RL (2026-06-16; code on unmerged recipes PR #18). Challenge 16 GiB / 0.1 s contract | Any public distillation script or procedure. NVIDIA said "planned/forthcoming" three times (Edge-LLM blog 2026-03-12; closed-loop blog 2026-05-31; AlpaGym ONBOARDING, public from 2026-06-16, target "e.g. 2B" single-GPU). Not shipped in RC (HEAD 2026-09-30) or Edge-LLM ≤ v0.11.0 (2026-09-29). Closed-loop SFT/DAgger with teacher labels; student-vs-teacher paired verdict | AG ONBOARDING; NVIDIA blog 2026-05-31; Edge-LLM blog 2026-03-12; RC commits; Edge-LLM releases |
| **G6 Deployment precision** | ModelOpt 0.43 PTQ for Alpamayo 1.5 (2026-06-29): FP8, NVFP4, W4A8, AutoQuant FP8+NVFP4 (default 4.8 BPE; 6.5 BPE → ~9 GB). Expert layers included except action in/out projection edge cases. AutoQuant sensitivity via flow-matching MSE. Open-loop minADE eval script. Edge-LLM supports Alpamayo-R1-10B only; A1.5 promised for August, not shipped by 2026-09-29 | Closed-loop per-format verdicts with intervals. Published accuracy or latency numbers (none in recipe). Published per-layer sensitivity. Thor-native FP8/NVFP4 execution of Alpamayo (forum: ModelOpt fake-quant gives no speed-up on Jetson Thor; Edge-LLM lacks A1.5) | RC quant utils.py, quantize.py, README; Edge-LLM supported-models; Edge-LLM issue #130; forum 382409 |
| **G7 AlpaGym throughput profile** | AlpaGym `perf/` (opt-in JSON scopes, CPU/GPU monitors) plus auto-merge of AlpaSim Prometheus "Simulator internals". AlpaSim Prometheus default (2026-06-30), `metrics_plot.png`, Grafana throughput panels (2026-07). Batched policy inference (`max_batch_size`) and batched deterministic rollouts (06-26 sync). Coarse community wall-clock (Classmethod 2026-07-19) | Stage-level public profile (render vs policy vs trainer vs sync) at a realistic preset (default is an 8-step smoke config). Batch-scaling curve. Render-resolution experiment (AlpaSim camera config, ≥320×576 floor). Upstream contribution path: AlpaSim yes (~11 external merges, 0–6 days); AlpaGym no precedent (0 external PRs, internal MR review, public code frozen since 2026-07-08) | AG perf/README; AS TELEMETRY.md and CHANGELOG; AG default.yaml; Classmethod; AS/AG PR searches |

### Inferences
- **Worth building:**
  - G1 (thin and upstreamable, high value because nothing upstream gives a paired verdict).
  - G2's calibration/agreement parts.
  - G4's monitor plus intervention-rate (hook exists, so the contribution is focused).
  - G6's closed-loop evaluation.
  - G7 as a *measurement report* (not a framework).
- **Reconsider or re-scope:**
  - G6 "including the action expert" as a novelty claim.
  - G7 PRs aimed at AlpaGym.
  - G5's timing (high risk NVIDIA releases its 10B→~2B distillation first; keep the comparison-against-NVIDIA framing ready).
- **Watch items:**
  - AlpaSim's overdue monthly sync (last 2026-08-11).
  - Recipes PR #18 merge or rewrite (AlpaGym's pin).
  - Edge-LLM A1.5 support.
  - Recipes training-efficiency "Measurement/Methodology" chapters (NVIDIA adopting paired-run methodology, currently for training performance only).

### Gaps
- Not determined here:
  - Drive-IRT interval definition.
  - Challenge rules beyond the branch README (deadline, report format, registration email).
  - Contents of unread AlpaSim issues #192/#193 (possible Alpamayo integration artifacts relevant to G2 validity).
  - Whether NVIDIA's internal AlpaGym or AlpaSim already contains a stage-level profile or a paired-comparison tool. Not public, so UNVERIFIABLE.
