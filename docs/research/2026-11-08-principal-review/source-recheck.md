# Source re-check for the central question (2026-11-08)

Agent-authored, for the [decision brief](decision-brief.md). It re-checks only claims that could change the S4 experiment, one week after [the November 1 audit](../2026-11-01-rl-direction-review/nvidia-stack.md). Public sources only; nothing was installed, downloaded or accepted.

**Verification levels.**
- **Root:** the coordinating agent re-read the raw source on 2026-11-08.
- **Agent:** a research subagent read it, and root did not re-read it.
- **Summary:** the gated Hugging Face dataset cards, arXiv HTML and NVIDIA blogs were read through a fetch summarizer.

## Revisions

| Surface | 2026-11-08 state | Effect | Level |
|---|---|---|---|
| AlpaGym `main` | `972d160` unchanged | None | Agent |
| alpamayo1.5 `main` | `36aeb4c` unchanged | None | Agent |
| AlpaSim `main` / latest release | `affc2ea` / v2026.5, unchanged | None | Agent |
| AlpaSim `e2e_challenge` | Moved to [`297b4d0`](https://github.com/NVlabs/alpasim/commit/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0) (2026-11-03) | nuPlan reference scores only; no runtime or scorer change. Already in the pilot launch audit | Root (via that audit) |
| recipes PR [#18](https://github.com/NVlabs/alpamayo-recipes/pull/18) (Gym's dependency) | Still open at `1d99fc5`, `mergeable_state: dirty` | None | Agent |
| alpamayo-recipes `main` | Moved to `c53aa09` (#42, #43, #44, #45 merged 11-03 to 11-05) | No file under `recipes/alpamayo1_x_rl/` changed. #42 adds an open-loop "expert RL" (see below) | Agent |
| Checkpoint [`nvidia/Alpamayo-1.5-10B`](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | Revision `7aba8293c09993f2e125c6819df05d7fa3e873ea`, modified 2026-08-07, not gated | Gym ONBOARDING downloads it with no revision, so the S4 manifest must pin this hash | Root (API) |

## Facts that bear on the question

1. **"NVIDIA's RL update" is shipped tooling, not a published result.**
   - **Agent, NOT FOUND:** no NVIDIA source reports a closed-loop effect of AlpaGym expert GRPO on Alpamayo 1.5. The agent searched the AlpaGym README and ONBOARDING, the alpamayo1.5 README, the model card, the [NVIDIA tutorial](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/) (2026-05-31) and the Hugging Face Alpamayo 2 blog (2026-06-01, "GRPO with default reward functions is supported at release"), and ran a web search.
   - **Root:** the [README feature table](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/README.md) marks the release as RL post-trained "for reasoning/action consistency". That is a different stage, described for Alpamayo-R1 in [arXiv 2511.00088](https://arxiv.org/abs/2511.00088) §5.3 (summary): an LRM critic, CoC–action consistency and trajectory-quality rewards, with open-loop results.
   - **Root:** the [model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) reports one closed-loop number, "AlpaSim Score of 1.37 ± 0.10" on 913 scenarios, with no before/after-RL comparison.
2. **The update trains every non-VLM module, not only the expert transformer.**
   - **Root:** the Gym trainer optimizes every parameter with `requires_grad` ([`trainer.py` ~L526](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/trainer.py)).
   - **Root (11-01):** the VLM freeze is in `expert_model.py` L77–79.
   - **Agent:** the resulting trainable set is the 36-layer `expert.*` with its norms, `action_in_proj.*` (Fourier encoder plus a LayerNorm) and `action_out_proj.*`.
   - The 11-01 wording "expert only" stays true if "expert" means the whole non-VLM action path. Record the actual mask at load (E03).
3. **Interface (agent, from the card, `config.json` and pinned source).**
   - The expert attends to the VLM prefill KV cache.
   - It outputs 64 waypoints at 10 Hz (6.4 s), parameterized as unicycle acceleration/curvature controls.
   - Inputs are 4 cameras × 4 frames and 16 history waypoints.
   - AlpaSim's MPC tracks the trajectory and replans every `control_timestep_us`.
4. **The training and evaluation paths differ** (P-032's six factors, re-confirmed).
   - **Gym (agent):** SDE with 5 steps and noise 0.2 (4 steps in the smoke preset), at 320×512.
   - **Release (agent):** Euler ODE with 10 steps; the card states 320×576.
   - **Root:** the controller is linear MPC on Gym's AlpaSim pin and `main`, and nonlinear MPC on `e2e_challenge` ([`base_config.yaml` L11](https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/src/wizard/configs/base_config.yaml)).
5. **The training horizon is a fraction of the evaluation horizon.**
   - **Root, smoke preset:** 30 simulation steps, of which 22 are policy steps, at 200 ms after 1.6 s of forced ground truth, so about 4.4 s of driving.
   - **Agent, base default:** 17 policy steps at 10 Hz.
   - **Root, `e2e_challenge`:** 200 steps at 100 ms with 1.7 s of forced ground truth, so about 18 s of driving if the scene is long enough. Whether the runtime clips to scene length is not verified.
6. **Splits.**
   - **Root:** the 1,761/441 curated split ([local evaluation README](https://github.com/NVlabs/alpasim/blob/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0/e2e_challenge/local_evaluation/README.md)) is an 80/20 split of 2,202 public NuRec scenes.
     - It groups road-sharing scenes into the same split, but "different roads or driving directions at the same location may occur across splits".
     - It came from an outside contributor's PR #167 on `e2e_challenge` and is absent from Gym's pin.
     - It says nothing about overlap with the teacher's training data.
   - **Summary:** PAI-AV `alpamayo_aligned` at v26.11 counts train 276,284 / test 28,317 / alpasim 1,550, and "follows the train/test partition of NVIDIA's Alpamayo models".
     - These counts differ from `04` (276,299 / 28,317 / 1,536), so re-pin them at S2.
     - 1,550 `alpasim` clips are fewer than 2,202 curated scenes, so some evaluation scenes may carry `train` labels (inferred).
     - Whether they do needs the gated parquet.
7. **New, and not the same thing.**
   - **Agent:** recipes #42 adds DRaFT, an open-loop "expert RL" that backpropagates a soft-min ADE through the 10-step sampler into the same non-VLM modules.
   - It reports a small minADE change on 2B distilled students only.
   - It is not closed-loop, not GRPO, not the 10B model and not a Gym dependency. Do not cite it as evidence for the AlpaGym update.

## Not verified

- Runtime trainable masks and counts.
- Whether the released expert itself changed during NVIDIA's reasoning/action-consistency RL stage.
- Which NuRec release the card's 913 scenarios come from.
- Clipping of the 200 evaluation steps.
- Curated-scene overlap with `train`-labelled clips.
- Third-party arXiv 2609.00111 (Qwen-Drive), not read.
