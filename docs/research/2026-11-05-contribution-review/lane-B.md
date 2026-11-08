# Researcher B: action-expert RL and simulation systems

Agent-authored, read-only review for PhysLoop S2/S4 (G9, G7), 2026-11-05. It covers no new training, rental, asset download or licence action, and fills none of Huy's prediction or registration slots. Reward, estimator and objective choices remain Huy's; anything here touching them is a labelled proposal.

**Evidence labels.** *Source-inspected*: I read the code at the stated SHA. *Doc-stated*: NVIDIA prose or a commit/PR message. *Derived*: arithmetic or algebra from inspected facts, with assumptions stated. *Hypothesis*: plausible but unmeasured.

**Read window.** All network reads happened between 2026-11-05 16:29 and 16:55 UTC. Branch heads, PR lists and releases were read at 16:29–16:36; the release endpoints were re-queried at 16:50:50. Pinned AlpaGym, Cosmos-RL, recipes PR 18 and AlpaSim files were fetched into the scratchpad `src/` directory. I reused the 2026-11-01 copies in `evidence/rl-gap-review-20261101-v1/pinned-source/` where they existed: `trainer.py`, `rollout_filter.py`, `grpo.py`, the presets and `pr42_*`.

---

## 1. Upstream delta since 2026-11-01 12:06 UTC

| Endpoint | Result (read 2026-11-05) | Class |
|---|---|---|
| AlpaGym main | Still at `972d160` (2026-07-21). It has **no open PRs**. #6, #7 and #8 (#8 bumped Cosmos-RL to 0.4.4) were closed unmerged on 08-25 and 09-03, before the cutoff. The releases endpoint returns `[]`. | No change |
| AlpaSim main | Still at `affc2ea`, with `CHANGELOG.md` headed "August 2026 public sync updates (11.08.26)". Releases are unchanged: latest v2026.5 (2026-06-04). **New open PRs on main:** #234, "collision category flags reset per actor" (erkara, opened 2026-11-01 20:27 UTC), and #238, commit `uv.lock` and freeze resolution (opened 2026-11-05 00:57). | Open PRs only |
| AlpaSim e2e_challenge | **New merged commit `297b4d0`**: #228 at 2026-11-03 15:02 UTC adds nuPlan leaderboard reference scores. Its file list touches only `e2e_challenge/local_evaluation/{README.md,data/nuplan/*}`, and the message says "No evaluator or simulation behavior changes". The S2 candidate `e121e59` is therefore no longer the branch head, but simulator and evaluator code are unchanged by this commit. `CHANGELOG.md` was not touched (inferred from the file list). **New open PRs:** #235, exclude invalid timesteps from `metrics_results.txt`/`results-summary.json` aggregation (2026-11-03), and #236, wrap the heading difference in the wrong-lane check (2026-11-03, updated 11-04). | One merge; open PRs |
| alpamayo-recipes main | Moved `2e907da` → **`c53aa09`** with four merges: #44, training-efficiency chapter 05 "Measurement" (2026-11-03 17:43); **#42**, distillation including the stage-3 expert-RL recipe (2026-11-05 06:27); #43, speculative-decode draft models (06:31); #45, 5 Hz inference for 2B AM1.5 (06:41). The #42 merged `alpamayo1_x_distillation_expert_rl/README.md` and `draft/train_draft_ade.py` are **byte-identical** to the 2026-11-01 pinned copies (diffed). There are no releases, so #42 is merged, not released. **PR #18** (Gym's dependency) is still open at head `1d99fc5`, now with `mergeable_state: dirty`. | Merged (not released) |
| Cosmos-RL main | Head `e82b56f` (2026-10-22); no commits since 2026-10-25. Release v0.4.7 is dated 2026-10-22, before the cutoff. Relative to Gym's pin `d2a2c57` (2026-05-27) there are 76 commits. `grpo.py` differs **only** by `minimum_trainable_completions = 2`; `compute_advantage` and `unbiased` are unchanged. Main also gained four changes absent at the pin: per-completion admission *before* advantage with minimum group size 2 (#740, #752); opt-in sample- and episode-weighted objectives (#758); GRPO numerical-contract fixes (#760); and accepted-rollout staleness p50/p99/max observability (#697). No commit title matching flow/diffusion/KL/reference changes the GRPO estimator. None of these is on AlpaGym's executable path, because Gym's trainer overrides the loss. | No delta since cutoff |
| alpamayo1.5 main, PR #36 | Main is still `36aeb4c`. #36 was merged 2026-08-29, before the cutoff; #37 (GPU CI) has been open since 08-24. No releases. | No change |
| alpamayo2 main | Still `5977692`; no releases. | No change |
| [NuRec release notes](https://docs.nvidia.com/nurec/release-notes/index.html) | Latest entry is **26.10, "October 2026"** (month only): open-sourced under Apache-2.0, `nre`→`nurec` rename across gRPC, images and package, CUDA IPC, and per-request Harmonizer control. **No entry dated on or after Nov 1.** I did not verify which NuRec image Gym's AlpaSim pin actually pulls. | Doc-stated release, before the cutoff |
| [Harmonizer docs](https://docs.nvidia.com/nurec/nurec/harmonizer.html) | "Last updated on Oct 30, 2026". Diffusion-based, fine-tuned from Cosmos-Predict2 0.6B, with stateless and temporal modes. **Silent on determinism, seeds, latency and whether AlpaSim enables it.** | Doc, before the cutoff |

**What matters for S4.** Nothing changed on the executable RL path (AlpaGym, Cosmos pin, PR #18 head). Six items do matter:

1. #42 merged unchanged, so the gap review's PR 42 citation now points at merged code.
2. PR #18 being `dirty` raises dependency-drift risk if Gym ever re-pins recipes.
3. #234 concerns a scorer pattern that **also exists at Gym's AlpaSim pin**; see §2.10.
4. #235 concerns S2's processed outputs on e2e_challenge.
5. The e2e_challenge head moved, but only in data/README files.
6. #238 records the same unpinned-dependency problem the plan already handles with lockfiles.

---

## 2. Learning-loop facts at the pins

### 2.1 What a "group" is (source-inspected)

A group is one Cosmos `RLPayload`, which corresponds to one scene (`scene_ids[prompt_idx]`, `rollout_backend.py` L130–132). `submit_payload` enqueues `rollouts_per_payload = n_generation` sibling `_RolloutJob`s, each with a fresh `uuid4` session (`streaming_worker.py` L160–171; `rollout_backend.py` L83).

Each sibling is a **separate full AlpaSim episode**. `_run_rollout` issues its own `simulate` call with `n_generation=1` and the job's pinned `session_uuid` (L215–231). AlpaSim then opens one driver session, and AlpaGym's `policy_factory` builds one `AlpamayoPolicy` per session (`driver_server.py` L127–133).

Reward is computed per episode on the rollout worker (L251–253). The advantage is computed per payload in the Cosmos reward path, over the list of completions. The rollout filter's docstring states this happens before trainer dispatch (`rollout_filter.py` L22–25).

### 2.2 Shared forced-GT warm-up

- **Source-inspected.** AlpaGym forces `runtime.simulation_config.skip_driver_during_force_gt=true` (`alpasim_wizard.py` L54). During warm-up, AlpaSim's `PolicyEvent` submits egomotion, route and GT and then returns without calling `drive` (AlpaSim `46b401b` `events/policy.py` L67, L125–131).
- **Source-inspected.** At that pin `events/camera.py` has no force-GT condition, so **every warm-up camera frame is rendered and submitted** to AlpaGym's driver.
- **Source-inspected.** AlpaGym's validation requires `n_sim_steps − force_gt/control_dt = expected_valid_steps` (`config_validation.py` L297–318). The 1.5 presets use 30 steps, 1.6 s / 0.2 s = 8 warm-up steps, and 22 closed-loop drive calls.
- **Derived.** All siblings see the same recorded ego path, log-replay traffic and scene. Their first drive call therefore receives identical observations, **up to renderer, physics-blend and Harmonizer nondeterminism**. AlpaSim #149's own message says renderer sampling is unseeded, and `force_gt_blend.py` exists at the pin.

The first closed-loop row of each sibling is therefore the only true same-state contrast in a group; later rows diverge.

### 2.3 Seeds and whether a recorded group can be re-simulated

**Source-inspected:**

1. AlpaSim at Gym's pin draws each driver session seed as `random.randint(0, 2**32-1)` (`services/driver_service.py` L75). `RolloutSpec` has only `scenario_id`, `nr_rollouts` and `session_uuids` (`runtime.proto` L14–20). The seed **cannot be specified**.
2. AlpaGym logs the received seed ("Started AlpaGym driver session=%s seed=%d", `driver_server.py` L132). AlpaSim also broadcasts the `DriveSessionRequest`, including the seed, to its log (L74–82).
3. With `force_determinism: true` (the AR1.5 preset), the per-step seed is `session_seed + inference_step_counter` (`policy.py` L60–72). Each row gets its own `torch.Generator` (`inference_model.py` L142–153).
4. Three things are not fixed by seeds:
   - the engine batches **whatever is queued, up to `max_batch_size` (4 in the AR1.5 preset)**, so batch composition depends on thread timing (`inference_engine.py` L101–117);
   - `set_deterministic` uses `use_deterministic_algorithms(True, warn_only=True)` (`determinism.py` L30);
   - `retry_on_nonfinite` re-samples with the already-advanced generators (L175–199).
5. AlpaSim main `9a3832b` (#149, 2026-08-17) adds `RolloutSpec.random_seed`, with rollout *k* seeded `random_seed + k`. It postdates Gym's pin and is not in Gym's stubs.

**Conclusion.** At the pins a recorded group **cannot be re-simulated with its own seeds** without a change, even though the seeds are recorded. Protocol §6's "if the path cannot repeat, record that limit" applies as written.

**Derived.** Re-simulating a pair with *fresh* seeds produces a new exchangeable pair. For continuous rewards its order agrees with the recorded order with probability 1/2 whatever the signal, so it says nothing about the recorded order. Only same-seed replay isolates exogenous noise (candidate C2).

### 2.4 Failed and retried episodes relative to advantage computation (source-inspected; corrects nvidia-stack.md)

A failed `simulate` call (`rollout_return.success == False` or any exception) never receives a reward. It is re-enqueued at priority 0 with a **fresh `session_uuid` (and therefore a fresh random seed)** for the same scene (`streaming_worker.py` L233–238, L255–256, L277–305). The group is the **first `n_target` successful episodes** (L265–273). Failed attempts are therefore excluded **before** reward and advantage, and retried scenes are retried within the same group. The per-payload retry budget (default 3) is shared across siblings.

On exhaustion the worker resolves `[]` (L306–319), but **`rollout_generation` then raises `RuntimeError`** when `len(episodes) < n_target` (`rollout_backend.py` L234–241). So at this pin there is no silent partial group and no silent drop at the trainer. Cosmos `_main_loop_impl`/`one_step_generation` has no handler around this call (`rollout_control.py` L1719–1736, L1925–1945). The worker-level consequence is likely a replica exit; that is **unverified at runtime**.

**Hypothesis.** If failure correlates with policy behaviour (for example a failure triggered by off-map driving), groups are survivor-conditioned and failure-inducing actions receive no negative advantage. The plan's attempt ledger measures this; nothing at the pin corrects it.

### 2.5 Weight synchronisation and actual policy lag

- **Source-inspected.** Base `default.yaml` sets `allowed_outdated_steps: 4` and `on_policy: false` (L149–150). Neither AR1.5 preset overrides them.
- **Source-inspected.** The controller tags each prompt with `current_step + pending//rollouts_per_global_batch` (`controller.py` L273–275). It soft-throttles once pending reaches (allowed+1) batches (L284–300).
- **Source-inspected.** The rollout main loop only blocks a prompt whose tag exceeds `current_version + allowed` (`rollout_control.py` L1811–1819).
- **Source-inspected.** The Gym trainer keeps the freshest `train_batch_per_replica` rollouts and only *warns* on stale ones (`rollout_filter.py` L57–73).
- **So the configured lag in both presets is up to 4 trainer steps.** The actual lag is not reported anywhere at the pin. Cosmos main #697 added p50/p99/max staleness; the pin lacks it.

**Mechanism (source-inspected).** P2R runs on the rollout main thread in `consume_command` (`rollout_control.py` L1008–1036, L1763). It receives directly into views of the live parameter tensors that `model_param_map` exposes (L585–600; `rollout_backend.py` L275–286). No fence against AlpaGym's background simulate and inference threads exists in the inspected code.

The per-payload `weight_version` is stamped at report time (`rollout_control.py` L1904). It is not recorded per drive step: the packer writes `weight_version=0` per row (`packer.py` L211, L235).

**Prefetch (source-inspected + derived).** Base default `prefetch_rollout: true` (L179); both AR1.5 presets set `false`. With prefetch on:

- the next batch's episodes start during the current `rollout_generation` and can still be running when P2R lands. This yields mixed-version episodes tagged with one version (E09) — a **hypothesis about runtime occurrence**, with **no fence in the inspected source**;
- `submit_payload` re-dispatches a payload whose prefetched future already resolved, by design ("after resolution the next call dispatches afresh", `streaming_worker.py` L135–138). That is wasted simulation and a second retry budget.

### 2.6 How groups map to updates; optimizer steps

- **Source-inspected.** Cosmos buffers rollouts FIFO (`status.py` L736–757). It triggers a step once `train_batch_per_replica × replicas` rollouts are pending (L1255–1272). Gym validation checks divisibility by `dp_shard × mini_batch` but **not by `n_generation`** (`config_validation.py` L331–371).
- **Derived.** The smoke preset (`train_batch_per_replica: 1`, `n_generation: 2`, `max_num_steps: 1`) trains on **one of the two siblings**, at advantage ≈ ±1. Generally, when `train_batch_per_replica × policy_replicas` is not a multiple of G, sibling groups straddle trainer steps. The later sibling is then off-policy by ≥1 step and the within-update zero-sum property of group-centred advantages is lost.
- **Derived.** Optimizer steps per trainer step = `μ × ceil(rows/mini_batch)`, minus zero-gradient skips, where rows = 22 × rollouts per rank. That is 22 for the smoke (1 rank) and 66 for `clrl_test_run` (6 rollouts over `dp_shard 2` = 3 per rank). Each clrl step reduces one row per rank across 2 ranks.

### 2.7 The trainer recomputes the frozen VLM per replay row (source-inspected)

`ExpertModelCosmos.forward` is patched in Gym `bundle.py` L55–114. It re-tokenizes the raw `image_frames` per call. It then calls `cfm_logprob_sde`, which runs `_vlm_prefill`: the full frozen VLM forward over all 16 images, with `use_cache=True` (recipes PR 18 `cosmos_wrapper.py` forward; `model.py` L60–111, L321–393).

This runs for every row in every minibatch and every μ iteration, including padding rows (`trainer.py` L399–404; padding rows are cloned valid rows, `packer.py` L228–238). It is a deliberate choice: the artifact stays about 25× smaller (`bundle.py` docstring). No gradient flows through the VLM because its parameters are frozen and its inputs do not require gradients.

### 2.8 What AlpaGym `perf` measures (source-inspected, doc-stated)

`perf` is off by default (`default.yaml` L107–120). Its scopes are:

- **Rollout side:** `rollout/engine_init`, `rollout/generate`, `rollout/episode`, `rollout/sim_request_build`, `rollout/sim_step_rpc` (wall time of the whole `simulate` call), `rollout/reward_compute`, `rollout/inference_forward` (GPU wall including queue and launch).
- **Driver side:** `driver/session_start`, `driver/drive`, `driver/policy_step`, `driver/input_build`, `driver/output_build`, `driver/session_close`.
- **Trainer side:** `trainer/step` and `trainer/artifact_load/disk_read`.

It also takes 5-second CPU (psutil) and GPU (NVML utilisation, torch allocation) samples. AlpaSim's Prometheus render, physics and drive RPC sums are folded in, but the README says they **overlap and include queue wait, so they do not partition `sim_step_rpc`**. The trainer's reported `train/iteration_time` is hard-coded to `0.0` (`trainer.py` L244). Recipes #44's wall-clock-from-logs method targets the recipes SFT trainer, not this path.

### 2.9 Wall-clock split between rendering, inference, training and idle

**No inspected doc or telemetry states one.** The Gym onboarding guide gives hardware recommendations only. Any split in §3 is derived.

### 2.10 Other decisive facts

- **Collision flags (source-inspected at the Gym pin `46b401b`, `eval/scorers/collision.py` L45–55).** The front, rear and lateral flags are reset for every intersecting actor, so only the last actor's category survives a timestep. This is the same bug PR #234 fixes on main. **Derived:** `collision_any`, the `progress_safety` term, is unaffected, because each actor sets exactly one category, so `any()` is true whenever an intersection exists. `collision_front`/`collision_lateral` can be lost under simultaneous contacts. That affects the front-or-lateral arm and the offline `collision_at_fault`.
- **Observation geometry.** See C1. AlpaGym squashes 1080p frames to 320×512, while the model card and AlpaSim's driver use 320×576.
- **Route (source-inspected, partial).** `route_xy` is built and persisted. The trainer inputs omit it (`inference_model.py` L342–357), and no `route` reference appears in the four inspected PR 18 model files. Whether the released base class consumes it was not checked.

---

## 3. Learning efficiency per paid GPU-hour (derived; no measurements)

**Assumptions:**

- AR1.5 preset: 4 cameras × 4 context frames, 10 Hz cameras (`4cam_1080.yaml` `frame_interval_us 100_000`), control 0.2 s, warm-up 1.6 s, 22 valid steps, K=5 denoising steps, G candidates.
- Model card (doc-stated, read 2026-11-05): VLM 8.2B, expert 2.3B.
- Vision tokens: 160 per image at 320×512 and 180 at 320×576, assuming a Qwen3-VL-style patch of 16 with 2×2 merge (the project's 10-03 verification derived 180).
- FLOP model: 2·params·tokens for forward and 2× forward for backward. Attention-over-cache and ViT patch FLOPs are ignored. The expert is assumed to process 64 action tokens per denoising step (unverified).

### Per-episode work

| Stage | Units per episode | Signal-bearing share |
|---|---|---|
| Rendering | 6.0 s × 10 Hz × 4 cameras = **240 frames**, of which **64 are warm-up (26.7%)** | Only the last ~4 warm-up frames per camera reach the model at the first drive. About **48 frames (20%)** are rendered, encoded, decoded and overwritten unused. Identical warm-up is re-rendered for each of the G siblings and on every revisit. |
| Rollout inference | 22 drive calls, each one prefill P plus K sequential expert steps | All 22 rows become replay rows. |
| Training | 22 rows (+ padding), each P again plus parallel expert forward/backward ≈ 3X | Padding rows and exactly-zero-advantage rows still run the full forward and backward; the optimizer step is skipped only *after* backward. |

### FLOPs per trained row

- Prefill P ≈ 2 × 8.2e9 × ~2.7e3 ≈ **4.4e13 FLOP**.
- Expert forward over K=5 steps ≈ 2 × 2.3e9 × 64 × 5 ≈ 1.5e12 FLOP.
- Rollout ≈ P + 1.5e12; trainer ≈ P + 4.4e12.
- **Frozen-VLM prefill ≈ 2P / (2P + ~6e12) ≈ 94% of model FLOPs per trained row**, and half of that is the trainer's recomputation.
- FLOPs are not wall time: rollout inference runs at batch ≤4 and is latency-bound.
- Reusing the prefill KV instead of recomputing it would need about 2×36×8×128×2 B ≈ 144 KiB per token (Qwen3-VL-8B geometry assumed), so about **400 MB per row**, 8.8 GB per episode. That is not a narrow change. With μ=1 there is no cross-iteration reuse to gain.

### Learning signal per episode budget

**Group mixing.** Let p be the per-episode collision probability. A group is mixed on C_any with probability `1−(1−p)^G−p^G`, so mixed groups per *episode* = `[1−(1−p)^G−p^G]/G ≈ p − (G−1)p²/2`. At p=0.08 (the hypothetical from `protocol-design.md` §5):

| G | Mixed groups per episode |
|---:|---:|
| 2 | 0.074 |
| 4 | 0.071 |
| 6 | 0.066 |

**The count of collision-informative groups per episode budget is nearly independent of G.** G changes the advantage structure (sign-only at G=2, G1), not how many safety contrasts a budget buys.

**Post-incident rows.** C_any and offroad are episode-MAX aggregates. Rows after the first incident at valid step k cannot change those terms. They still carry the episode advantage, so they add zero-mean variance for those terms while still affecting progress and distance-to-GT. With k uniform on 1..22, about **48% of an incident episode's rows are post-incident**. The plan already tracks this diagnostic, and any change is Huy's credit design.

**Rollout/train overlap.** The disaggregated presets overlap rollout and training up to the 4-step lag. Strict `allowed_outdated_steps=0` for E04/E05 parity serialises them: the rollout side waits a full trainer step per version. With prefetch off, each `rollout_generation` ends with idle simulate slots while stragglers finish: slot utilisation ≈ mean / max episode duration within a batch.

**Ranked levers, all unmeasured:**

1. Duplicate frozen prefill: the largest FLOP term, with no narrow fix.
2. Prefill of padding and tied rows: small if episodes rarely end early and rewards are continuous.
3. Warm-up re-rendering: at most 26.7% of render work. The upstream cache (AlpaSim main, `force_gt_frame_cache`, off by default, needs `skip_driver_during_force_gt`) saves `(1 − 1/(G·E))` of it over E visits. Its wall-time value depends on render share; with Harmonizer enabled per frame, that share could be large (hypothesis).
4. Strict-lag serialisation, which is required anyway for parity runs.

The first measurement should be the `perf` + `.prom` profile on the S4 engineering qualification, not a code change.

---

## 4. Candidates

### C1: Observation-geometry parity between AlpaGym training and the gate (correctness; configuration-only)

**Beneficiary and workload.** The S4 baseline arm and every checkpoint evaluated through the S2 gate (E02/E16).

**Existing implementation and specific problem.**

- **Source-inspected.** AlpaGym's AR1.5 preset sets `input_size: [320, 512]` (`alpamayo_r1.yaml`). `ObservationBuffers._decode_to_chw_uint8` decodes each 1080×1916 render (`4cam_1080.yaml`) and applies `tvf.resize(frame, [320, 512], antialias=True)` (`buffers.py` L237–253). This is **anisotropic**: aspect 1.78 becomes 1.60. The squashed frames are persisted as `model_input` and replayed by the trainer, so rollout and trainer agree with each other.
- **Doc-stated.** The Alpamayo 1.5 model card says inputs are 1080×1920 "downsampled to 320x576".
- **Source-inspected, AlpaSim main `affc2ea`.** The Alpamayo driver sets `MIN_FRAME_HW = (320, 576)` with the comment "What the vision processor resizes 16:9 renders to" (`alpamayo_base.py` L370). `main.py` L843–849 rejects smaller frames because a 320x512 render "passes through untouched and costs vision tokens with no trace in the logs". The CHANGELOG (11.08.26) says "Alpamayo runs now reject camera images smaller than 320×576".
- **Source-inspected.** At e2e_challenge and Gym's pin the driver passes native-resolution frames to the model processor, with no pre-resize found by grep. The gate's teacher therefore sees 320×576.
- The project's 10-03 verification logged this as "UNVERIFIABLE (conflict)" (TL-39/TL-84, token count only). It is now **resolved as a train/gate observation mismatch**.

**Strongest baseline and prior art.**

- The shipped preset itself.
- **Source-inspected.** AlpaSim's own `wizard/configs/base_config.yaml` at Gym's pin `46b401b` defaults its two cameras to `height: 320, width: 512` (L307–315). So 320×512 was NVIDIA's June default.
- The 11.08.26 public sync then reclassified 320×512 as a silent wrong-size input and added the 320×576 floor. That pre-empts the "512 was a deliberate training choice" reading, unless NVIDIA evidence says otherwise.
- The model card.
- Protocol E16.

**Narrow change.** In the registered S4 run, set `policy.model.input_size: [320, 576]`. This is a configuration field; the ring, replay and trainer re-tokenisation follow it automatically. Decide it **before** baseline A. Changing it later invalidates the baseline under the plan's bug-correction rule. Choosing it is Huy's call because it changes baseline A's environment.

**Mechanism and units (derived).**

- Horizontal scale relative to the gate geometry is s = 512/576 = **0.889**; vertical is unchanged.
- A feature at horizontal pixel offset Δu from the centre in the gate view appears at 0.889·Δu. To a model calibrated on 576-wide inputs, lateral bearings and lane offsets look about 11% compressed: tan θ′ ≈ 0.889 tan θ, so 30° reads as about 27.2°.
- Tokens per image: (320/32)(512/32) = 160 vs (320/32)(576/32) = 180; per step 2,560 vs 2,880 (−11.1%). The M-RoPE position grid changes from 10×18 to 10×16.
- Consequences:
  - the AlpaGym zero-update teacher is not the gate's teacher;
  - RL adapts the expert to the features of squashed images, then the gate evaluates it on unsquashed ones;
  - a held-out gain or regression mixes a policy change with a distribution shift.
- Cost of the fix: +12.5% vision tokens per prefill.
- **Residual after the fix (derived).** At `[320, 576]` the geometry matches the gate, but the resampling does not: AlpaGym uses torchvision bilinear with antialias, while the gate path uses the processor's own resize of the native render. E16 will still show a small residual. The test's job is to show that residual lies inside d_seed, not to expect bitwise equality.

**Measurable objective.**

1. Open-loop parity at the first closed-loop step, where all inputs except the resize are identical after GT warm-up. Compare d_size = ADE(τ₅₁₂, τ₅₇₆) over the 64-waypoint trajectory, same row seed, metres, against the same-size seed spread d_seed = ADE(τ_seed a, τ_seed b).
2. Closed-loop: paired zero-update reward components and incident indicators for 512 vs 576 on development scenes. Huy sets the margins.

**Smallest disproof experiment.**

- **CPU (public code).** Run Gym's real `_decode_to_chw_uint8` on a synthetic 1080×1916 JPEG and assert output shape and aspect. CPU decode is supported; it only logs a slowness warning (`buffers.py` L155–160). Apply the processor's smart-resize arithmetic to both paths. The min/max-pixel values must come from the Alpamayo 1.5 processor config; if that file is behind the gated repo, **defer** this sub-check and label the token counts as derived.
- **Smallest GPU run.** After S2 admission and assets, on one GPU at zero-update weights: N development scenes × 2 sizes × 4 seeds, first drive call only, no closed loop. That is 8N forwards (N=50 gives 400 forwards).
- What it keeps and drops: it keeps the model, weights, history and seeds, and drops closed-loop compounding and learning.

**Failure conditions.** C1 fails if d_size lies within the d_seed distribution *and* closed-loop paired outcomes are equivalent within Huy's margins. In that case keep 512 and record it. It also fails if 576 breaks memory or latency on the chosen topology, or if NVIDIA evidence shows 1.5 was trained or RL-tuned at 320×512 (none found).

**Adopting component and fit.** PhysLoop run registration needs no upstream change. AlpaGym takes no PRs; an issue could be filed only with Huy's approval. It fits **S4.1 registration and S4.2 correctness as-is**.

### C2: Same-seed group replay instrument (RL infrastructure; instrumentation for S4.4)

**Beneficiary and workload.** The S4.4 diagnostics "whether that order survives re-simulation" and "noise-ranked groups" (G1, KC-4). Today they cannot be measured, and they are what separates G1's noise half from its magnitude half.

**Existing implementation and specific problem (source-inspected, §2.3).**

- Seeds are random per AlpaSim session and not settable at Gym's pin, though they are logged.
- Policy-side noise is reproducible from the session seed. Batch composition (up to 4), `warn_only` determinism, NaN retries, and renderer/Harmonizer sampling are not controlled.
- Fresh-seed re-simulation is uninformative (derived, §2.3).

**Strongest current baseline and prior art.**

- The plan's fallback is to "record that limit".
- AlpaSim main #149 adds `RolloutSpec.random_seed`, merged 2026-08-17. Gym does not pin it.
- Gym already supplies per-row generators and `session_seed + counter` stepping.

**Narrow change (PhysLoop-local, no upstream PR).**

1. A ~20-line wrapper around Gym's `build_policy_factory` maps `session_uuid → recorded seed` and otherwise passes AlpaSim's seed through. Gym pins `session_uuid` in `RolloutSpec`, so a replay harness controls the UUID.
2. A replay driver re-submits a recorded (scene, UUID, seed) at frozen weights, once at `max_batch_size: 1` and once at the run's batch size.
3. The ledger records seed, UUID, weight version and per-step selected-sample hashes.

If the training AlpaSim is ever re-pinned past `9a3832b`, #149 replaces step 1.

**Mechanism and equations (derived).**

- Model: R_i = a(ξ_i) + ε_i, where ξ_i is the policy noise fixed by the seed, a(·) is the action-caused reward with Var σ²_a, and ε_i is exogenous noise with Var σ²_e, independent of ξ.
- For a G=2 group, Δ = R₁ − R₂. A same-seed replay gives Δ′ = a(ξ₁) − a(ξ₂) + ε₁′ − ε₂′.
- corr(Δ, Δ′) = ρ = σ²_a / (σ²_a + σ²_e).
- Under joint normality, **P(order flip) = arccos(ρ)/π**: 0.045 at ρ=0.99, 0.144 at ρ=0.9, 1/3 at ρ=0.5, 1/2 at ρ=0.
- At G=2 the stock advantage is ≈ sign(Δ), so the flip rate is exactly the fraction of G=2 updates whose direction is set by exogenous noise.
- Estimators: σ̂²_e = ½ Var(R − R′) over same-seed replay pairs (reward units); ρ̂ = 1 − σ̂²_e / σ̂²_member.

**Measurable objective.**

- Bitwise reproducibility rate of trajectories and rewards at batch size 1 and at the run's batch size.
- Order-flip rate q̂ with its binomial interval, and ρ̂.
- With n pairs, SE(q̂) = √(q(1−q)/n), at most 0.05 for n=100. Huy sets any decision threshold.

**Smallest disproof experiment.**

- **CPU (public code, no weights).** Drive Gym's real `AlpamayoPolicy` + `InferenceEngine` with a stub `InferenceModel` that draws its outputs from the supplied generators. Then check:
  - (a) the wrapper's seed reaches the policy;
  - (b) per-step seeds equal `seed + t`;
  - (c) per-row draws are invariant to batch composition at the generator level;
  - (d) the NaN-retry path is deterministic given the same sequence.

  This tests the plumbing only; GPU kernel numerics cannot be tested on CPU. The fixture can run with `use_cameras: []`, as Gym's own CPU tests do, because the seed path does not depend on frames.
- **Weight version.** Recorded groups must come from the same weights the replay runs at. Use either groups from a rollout-only pass at the saved zero-update checkpoint, or step-0 groups with that checkpoint retained. Never replay a group recorded at version v with weights at v+1.
- **Comparison reference.** The original run's batch composition depended on thread timing and cannot be reproduced. Recorded-vs-replay at *any* batch size therefore includes composition noise. That noise is real exogenous noise in the actual run, so it belongs in ρ. Replay-vs-replay at `max_batch_size: 1` bounds the floor that excludes composition effects.
- **Smallest GPU run.** During the S4 engineering qualification at zero-update weights:
  - stage 1: 4 episodes, each replayed twice at batch size 1 (8 episodes). Replay-vs-replay bitwise check; this gives the non-composition floor;
  - stage 2: 16 recorded G=2 groups replayed once each at the run's batch size (32 episodes). Recorded-vs-replay order flip; this is the real-run exogenous noise.

  It keeps the simulator, renderer and policy noise, and drops learning.

**Failure and outcome conditions.**

- Bitwise-equal replays with q̂ ≈ 0 eliminate the exogenous half of G1. The informative result is then "magnitude-blind, not noise-ranked".
- Divergence at batch size 4 but not at 1 makes `max_batch_size` a registration trade-off (throughput vs noise) for Huy.
- If simulator-side randomness that the policy seed cannot reach dominates (traffic, renderer, Harmonizer), report it as a limit and stop.
- If the harness cannot reproduce the inference counter (NaN retries), the CPU known-answer test fails first.

**Adopting component and fit.** A PhysLoop-local harness, since AlpaGym accepts no PRs; the AlpaSim side is already upstream (#149). It is **instrumentation for S4.4 and fits as-is**, priced inside the engineering qualification. It runs no training and makes no scientific change.

### Registration guards (source-inspected facts, not candidates)

Record and assert these in the S4.1 registration alongside G4's fields. The first two are not on G4's list.

| Field | Reason |
|---|---|
| `prefetch_rollout: false` | Base default `true`; no P2R fence; re-dispatch (§2.5) |
| `allowed_outdated_steps` and `on_policy` | Default 4 and false in both presets; 0 for parity cohorts |
| `train_batch_per_replica × policy_replicas ≡ 0 (mod n_generation)` | Otherwise groups straddle updates (§2.6) |
| `force_determinism` together with `max_batch_size` | §2.3 |
| `retry_on_nonfinite` and its counts | §2.3 |
| Training AlpaSim SHA, including whether `force_gt_frame_cache` exists or is enabled | §3 |

---

## 5. Evidence log

| Claim | Source (SHA / version) | Date read | Evidence type | Applicability | Uncertainty | Consequence |
|---|---|---|---|---|---|---|
| AlpaGym main unchanged; no open PRs; no releases | [AlpaGym commits](https://github.com/NVlabs/alpagym/commits/main) `972d160`; PR list; releases API | 2026-11-05 16:29–16:51 | Source-inspected | Executable baseline | None | Pins stand |
| e2e_challenge head moved to `297b4d0` (#228), data/README only | [commit 297b4d0](https://github.com/NVlabs/alpasim/commit/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0) file list | 2026-11-05 16:35 | Source-inspected | S2 candidate pin | "No behavior change" per message; files confirm | S2 pin is no longer head; code unchanged |
| AlpaSim open PRs #234, #235, #236, #238 opened after the cutoff | [PRs](https://github.com/NVlabs/alpasim/pulls) | 2026-11-05 16:31 | Doc-stated (PR bodies) | Scorer and evaluator semantics | Unmerged | Track; do not adopt |
| Recipes main → `c53aa09`; #42/#43/#44/#45 merged; #42 RL README and trainer identical to 11-01 copies | [recipes commits](https://github.com/NVlabs/alpamayo-recipes/commits/main); PR 42 files diffed | 2026-11-05 16:31–16:45 | Source-inspected | Gap review's PR 42 citation | None | Cite as merged, not released |
| PR #18 still open at `1d99fc5`, `dirty` | [PR 18](https://github.com/NVlabs/alpamayo-recipes/pull/18) | 2026-11-05 16:31 | Source-inspected | Gym dependency | Drift if re-pinned | Keep exact pin |
| Cosmos `grpo.py` main vs pin differs only by min group size 2; admission, weighted objectives and staleness metrics are on main only | [grpo.py@e82b56f](https://github.com/nvidia-cosmos/cosmos-rl/blob/e82b56f11b5de32d289f1eee29bd9d480b39e44f/cosmos_rl/dispatcher/algo/grpo.py) vs pin; commit messages #740/#752/#758/#760/#697 | 2026-11-05 16:33–16:47 | Source-inspected (diff); doc-stated (messages) | Estimator | Keyword scan of 76 commits | No estimator delta on Gym path |
| NuRec latest 26.10 "October 2026"; nothing ≥ Nov 1 | [release notes](https://docs.nvidia.com/nurec/release-notes/index.html) | 2026-11-05 16:40 | Doc-stated | Renderer | Month-level date; image used by the Gym pin unknown | No delta |
| Harmonizer page updated Oct 30; silent on determinism | [Harmonizer](https://docs.nvidia.com/nurec/nurec/harmonizer.html) | 2026-11-05 16:40 | Doc-stated | Exogenous noise | Enablement in AlpaSim unknown | Possible exogenous noise source (C2) |
| Group = one payload; n_generation separate `simulate(1)` episodes with fresh UUIDs | [streaming_worker.py@972d160](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/episode_runner/streaming_worker.py#L132) L132–275; rollout_backend L83, L130 | 2026-11-05 16:38 | Source-inspected | S4 | None | Siblings are independent episodes |
| Warm-up skips drive but renders all frames at the pin | [alpasim_wizard.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/alpasim_wizard.py#L54) L54; [policy.py@46b401b](https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/runtime/alpasim_runtime/events/policy.py#L125) L125–131; camera.py@46b401b | 2026-11-05 16:46–16:52 | Source-inspected | S4 cost | Render nondeterminism unmeasured | Identical first input across siblings (modulo noise); 26.7% of frames are warm-up (derived) |
| Session seed random and not settable at Gym's AlpaSim pin; logged by Gym | [driver_service.py@46b401b](https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/runtime/alpasim_runtime/services/driver_service.py#L75) L75; runtime.proto L14–20; Gym driver_server L132 | 2026-11-05 16:44 | Source-inspected | S4.4 re-simulation | None | Same-seed replay needs C2 |
| Fresh-seed re-sim of an exchangeable pair flips with p=1/2; same-seed flip = arccos(ρ)/π | Algebra over §2.3 model | 2026-11-05 | Derived | G1 diagnosis | Normality assumption for the closed form | Only same-seed replay is informative |
| Engine batches up to `max_batch_size`; `warn_only` determinism; NaN retry re-samples | [inference_engine.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/inference/inference_engine.py#L101) L101–117; determinism.py L30; inference_model L175–199 | 2026-11-05 16:41 | Source-inspected | Determinism | Numeric effect unmeasured | Exogenous noise sources (C2) |
| Failed episodes are retried with fresh UUID before reward/advantage; exhaustion raises in `rollout_generation` | streaming_worker L233–319; [rollout_backend.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py#L234) L234–241 | 2026-11-05 16:39 | Source-inspected | Data accounting | Cosmos process-level response unverified | Corrects "silent drop"; survivor conditioning is a hypothesis |
| Lag allowed up to 4 steps in presets; P2R writes live params with no fence; version stamped at report | default.yaml L149–150; [rollout_control.py@d2a2c57](https://github.com/nvidia-cosmos/cosmos-rl/blob/d2a2c57c4bd6496482bc42da19a59b4432705eda/cosmos_rl/rollout/worker/rollout_control.py#L1008) L1008–1036, L585–600, L1904; controller L273–300 | 2026-11-05 16:47–16:53 | Source-inspected | E04/E05/E09 | Runtime occurrence of mixed versions unverified | Registration guards |
| Groups can straddle trainer steps; smoke trains one of two siblings | status.py L736–757, L1255–1272; config_validation L331–371; smoke preset | 2026-11-05 16:43 | Source-inspected + derived | S4.1 geometry | Dispatch order under failures | Divisibility guard |
| Trainer recomputes the frozen VLM prefill per row, per μ, padding included | [bundle.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/bundle.py#L55) L55–114; [model.py@1d99fc5](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/models/expert_model/model.py#L321) L60–111, L321–393 | 2026-11-05 16:42 | Source-inspected | Cost | None | Largest compute term |
| ~94% of model FLOPs per trained row are frozen prefill; ~400 MB KV per row | Card (8.2B/2.3B); token arithmetic | 2026-11-05 | Derived | Cost forecast | 64 expert tokens, Qwen3-VL-8B KV geometry and attention FLOPs are assumptions | Profile before any change |
| Mixed-groups per episode ≈ p − (G−1)p²/2 | Algebra | 2026-11-05 | Derived | Group-size choice (Huy) | p hypothetical | G does not buy more safety contrasts |
| `perf` scopes; off by default; `.prom` sums overlap; `train/iteration_time` = 0.0 | [perf README](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/perf/README.md); trainer.py L244 | 2026-11-05 16:48 | Source-inspected, doc-stated | G7 | None | Enable `perf`; do not use iteration_time |
| No doc states a render/inference/train/idle split | Gym README, ONBOARDING, perf README | 2026-11-05 | Doc-stated (absence) | Forecast | Absence of evidence | Measure in qualification |
| Collision category flags reset per actor at Gym's AlpaSim pin; `collision_any` unaffected | [collision.py@46b401b](https://github.com/NVlabs/alpasim/blob/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/scorers/collision.py#L45) L45–58; PR #234 | 2026-11-05 16:53 | Source-inspected + derived | Front/lateral arm, collision_at_fault | Frequency of simultaneous contacts unknown | Do not build the front/lateral arm on this scorer unpatched |
| AlpaGym squashes to 320×512; model and AlpaSim driver use 320×576 | [buffers.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/policies/alpamayo/buffers.py#L253) L253; alpamayo_r1.yaml; [alpamayo_base.py@affc2ea](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/driver/src/alpasim_driver/models/alpamayo_base.py#L370) L370; main.py L843–849; CHANGELOG 11.08.26; [card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | 2026-11-05 16:44–16:56 | Source-inspected + doc-stated | Train/gate parity (E16) | Behavioural effect unmeasured; processor min/max pixels not read | C1 |
| AlpaSim main has a shared force-GT frame cache (off by default) | [force_gt_frame_cache.py@affc2ea](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/runtime/alpasim_runtime/force_gt_frame_cache.py); base_config.yaml | 2026-11-05 16:54 | Source-inspected | Efficiency lever | Needs a pin change; render share unmeasured | §3 lever, not proposed |
| Model card: VLM 8.2B, expert 2.3B, 4 cams × 4 frames, 320×576 | [HF card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | 2026-11-05 16:45 | Doc-stated | FLOP model, C1 | Card revision not pinned in this read | Inputs to §3 and C1 |
