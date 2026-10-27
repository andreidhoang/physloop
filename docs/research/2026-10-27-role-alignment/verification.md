# Verification ledger, 2026-10-27 (plan v4.1)

**Status:** evidence record. It holds the facts that plan v4.1 (`../../02_PHYSLOOP_PLAN.md`, D-015) and the documents revised with it rely on, each checked against a primary source. It adds to the 2026-10-25 ledger (`../2026-10-25-teacher-to-thor/verification.md`) and the 2026-10-03 ledger. Live docs win where they differ.

**Method.**
- **Seven read-only research lanes** ran on 2026-10-27:
  - (L1) upstream changes and race check;
  - (L2) RL framework and reward;
  - (L3) simulation quality;
  - (L4) evaluation statistics;
  - (L5) student, correction and data;
  - (L6) cluster and reliability;
  - (L7) Thor, C++ and rollout performance.

  Their notes are in `notes/L1…L7_*.md`.
- **An independent check.** A separate checker re-verified 33 load-bearing claims against primary sources, never against the notes: code at named commits, PR, issue and release metadata, the Hugging Face API, PyPI, arXiv, and vendor pages. Its full ledger is `notes/V_verification.md`; the rows below summarize it.
- **Arithmetic.** `notes/arith_check.py` (output in `notes/arith_check.out.txt`).
- **Prices.** `notes/vast_2026-10-27/` (`python3 -I stats.py notes/vast_2026-10-27` reproduces the table).
- **The synthesis.** `report.md`.
- **Rules kept.** No licence was accepted, no gated file or LFS object was downloaded, nothing was rented, and nothing was posted. Quotes are under 15 words.
- **Verdicts:** VERIFIED, PARTLY (core right, a detail corrected; the wording below is the corrected one), CONTRADICTED.

**Code read (2026-10-27):**
- AlpaSim `main` @ `affc2eab` (2026-08-18), `e2e_challenge` @ `e121e591` (2026-10-20), and AlpaGym's pin `46b401be`;
- AlpaGym `main` @ `972d160` (2026-07-21);
- cosmos-rl `main` @ `e82b56f` and AlpaGym's pin `d2a2c57`;
- alpamayo-recipes `main` @ `2e907dab`, AlpaGym's pin `1d99fc5`, PR #42 @ `5ade7d0`, PR #44 @ `35ea6cc`, PR #45 @ `fbed0a7`;
- alpamayo1.5 `main` @ `36aeb4c5`;
- drive-irt @ `82ddd5a4`.

---

## 0. The facts that decide v4.1

1. **The repeats in S2 are not seeded** (V24, V25, V26).
   - **`e2e_challenge`:** the Alpamayo driver stores the session seed and never uses it. Its VLM sampling (top-p 0.98, temperature 0.6) and flow noise draw from the process-global RNG, and the branch has no determinism switch.
   - **`main`:** the driver reseeds every tick only with `force_determinism: true`, and every shipped Alpamayo config sets it to false.
   - **Elsewhere:** CAT-K ignores its seed. The maintainer's remark that NuRec "might" vary from run to run is a hedged recollection, not a measurement.
2. **The contamination rule as written excludes every clip** (V1). `splits/alpamayo_aligned.parquet` (PAI-AV v26.10) labels all 306,152 clips: train 276,299 / test 28,317 / alpasim 1,536. This is from the public dataset card; the parquet itself is gated and was not read.
3. **AlpaGym's estimator and defaults differ from the plan's text** (V2a–V2c).
   - **Ratio:** a per-step-row geometric-mean ratio. The per-element log-ratio is clamped to ±5 and averaged over T × 64 × 2 elements, then exponentiated and clipped to [0.8, 1.2].
   - **KL:** the closed-form per-transition Gaussian KL, off by default.
   - **Defaults:** the shipped defaults do not train, because a zero advantage makes the trainer skip the optimizer step.
4. **TacitVLA narrows two novelty claims** (V8). It is an NVIDIA paper (2026-10-13) with unpaired AlpaSim scores on the 441 public scenes for variants on a 2B backbone, and no token-budget sweep.
5. **Speed is available upstream, unused** (V4). NVIDIA's opt-in expert CUDA graph (alpamayo1.5 PR #36) is merged, reported bitwise-identical in a PR comment, and never enabled in AlpaSim's driver, which runs eager at batch 1. That driver installs alpamayo1.5 from git with no pinned revision.
6. **The protocol and the tooling sit on different AlpaSim branches** (V11, V13, V25, V29, and the clones).
   - **`e2e_challenge`:** the 441-scene protocol and the post-#215 evaluator.
   - **`main`:** the reseed, the DCGM counter list, the `SLURM_ENROOT` method, CUDA MPS, the Slurm topology skill and the 8-GPU Alpamayo 1.5 topology.

## A. Simulator and evaluation (AlpaSim)

| ID | Fact | Verdict | Evidence | Source |
|---|---|---|---|---|
| V24 | `e2e_challenge` driver ignores `request.random_seed`; top-p 0.98, T 0.6; no `force_determinism` | VERIFIED | `src/driver/src/alpasim_driver/main.py:162, :256`; `models/alpamayo_base.py:233-234, :318-329` @ `e121e591`; upstream `flow_matching.py:171` (`torch.randn`, no generator) @ alpamayo1.5 `36aeb4c5` | [alpasim](https://github.com/NVlabs/alpasim) |
| V25 | `main` reseeds per tick (`session.seed + inference_count`) only with `force_determinism`; all shipped Alpamayo configs false; the adapters call `configure_deterministic_runtime()` when it is true | VERIFIED | `main.py:713-714`; `alpamayo_base.py:612-615`; `configs/driver/alpamayo1_5.yaml:22` @ `affc2eab` | same |
| V26 | CAT-K logs but ignores its seed (PR #149's commit says so); NuRec run-to-run variation is a hedged recollection | PARTLY | `trafficsim/.../servicer.py:204-213`; commit `9a3832b5`; #128 (2026-07-28) | [#128](https://github.com/NVlabs/alpasim/issues/128) |
| V13 | `e2e_challenge`: no CAT-K (stub command; `8gpu_32rollouts` skips traffic). `main`: CAT-K samples unseeded. Harmonizer: on in the `e2e` dev default, off in the competition preset `ec2.yaml`; `main` passes no flag | VERIFIED | `base_config.yaml:181-184`, `:141`; `ec2.yaml:32-44` @ `e121e591`; `rollout.py:175`, `scene_adapter.py:519` @ `affc2eab` | same |
| V10 | #222 (2026-10-22): the internal test set is from a different, harder distribution; NVIDIA *plans* to enable the Harmonizer for the final reruns; renderer errors re-queue the scene. `8gpu_32rollouts` is the production topology per NVIDIA's `e2e_challenge` wizard README; in #222 the reporter says it | PARTLY (attribution) | #222; `src/wizard/configs/e2e_challenge/README.md:48` | [#222](https://github.com/NVlabs/alpasim/issues/222) |
| V32 | `8gpu_alpamayo1_5_eval` (`main` only): 2 driver replicas on each of GPUs 0–6, renderer and physics on GPU 7. Challenge EC2: 16 driver replicas, 4 per GPU on GPUs 4–7, 2 rollouts each, ≤ 16 GiB each | PARTLY (replica count) | `topology/8gpu_alpamayo1_5_eval.yaml` @ `affc2eab`; `topology/8gpu_32rollouts.yaml`, `e2e_challenge/README.md:173, :181` @ `e121e591` | same |
| V4 | Driver: `max_batch_size: 1`, eager PyTorch, `predict_batch` a sequential loop (overridden only by VaVAM), the expert graph never referenced; `alpamayo1_5` installed from git with no rev, no `uv.lock`. PR #36 merged 2026-08-29: off by default; 160/160 bitwise (PR comment, batch 1–4); −15.53% / −31.53% policy-call latency, one paired AlpaGym sample per model (preliminary) | VERIFIED | `configs/driver/alpamayo1_5.yaml:42` (main), `:37` (e2e); `driver/pyproject.toml:53` / `:44`; PR #36 | [PR #36](https://github.com/NVlabs/alpamayo1.5/pull/36) |
| V5 | External-driver mode (`external_static` / `external_dynamic`) and `planner_delay_us` (`DelayBuffer`), both branches; default 0 µs since 2026-03-24 (the old 200 ms config was removed) | VERIFIED | `base_config.yaml:8`; `runtime/alpasim_runtime/delay_buffer.py:8`; `event_loop.py`; `CHANGELOG.md:350` | [alpasim](https://github.com/NVlabs/alpasim) |
| V11 | Run methods on `main`: SLURM (Pyxis + Enroot), SLURM_ENROOT, docker-compose, none; no Kubernetes anywhere. MPS on Slurm methods only. The `optimize-slurm-topology` skill: 10–15 min of congestion; full-run seconds per rollout stabilizes after 30–45 min. The FlashDreams video renderer needs 96 GB with Alpamayo 1.5. `e2e_challenge` has only SLURM and docker-compose. #72: NRE is a proprietary, intentionally obfuscated NGC container; no short-term fix for Kubernetes without Docker-in-Docker; closed `not_planned` | PARTLY (#72 wording) | `schema.py:70-74, :157-160`; `.agents/skills/optimize-slurm-topology/SKILL.md:45-48`; `docs/VIDEO_MODEL.md:56-62`; e2e `schema.py:71-72` | [#72](https://github.com/NVlabs/alpasim/issues/72) |
| V29 | DCGM counter list (9 fields: GPU util, framebuffer, SM active and occupancy, tensor active, DRAM active) on `main` only, with no XID, power, energy or clock field. RPC histograms: `rpc_duration_seconds{service,method,tag}` (`e2e`), `alpasim_rpc_duration_seconds` plus queue-depth, rollout and step series (`main`) | VERIFIED | `telemetry/resources/dcgm-counters.csv` (main); `telemetry_context.py:91-97` (e2e), `:119-125` (main) | same |
| V31 | Per-rollout `.asl` logs record the driver's camera images (JPEG in the docs' example) and egomotion; replay services exist | VERIFIED | `docs/DATA_PIPELINE.md:8-20, :71`; `logging.proto` (fields 11, 12); `runtime/alpasim_runtime/replay_services/` | same |
| V3 | No jerk or comfort scorer on any branch; `utils_rs` has `Trajectory.jerk()` | VERIFIED | `src/eval/src/eval/scorers/` @ three commits; `src/utils_rs/utils_rs.pyi` | same |
| V7 | External merges: `main` #149 (2026-08-17) and #144 (08-12). On `e2e_challenge`, #111 and #161 (08-24) and #167 and #169 (08-31), all from accounts without collaborator status. Branch heads: `e2e_challenge` 2026-10-20, `main` 2026-08-18. #224 closed 2026-10-23: disclosure "until our next release" | VERIFIED | PR lists and `git ls-remote`, 2026-10-27 | [#224](https://github.com/NVlabs/alpasim/issues/224) |
| V21 | #208 acknowledged 2026-10-15, and the reporter offered a PR (none filed by 10-27); #207 has no reply | VERIFIED | issue threads | [#208](https://github.com/NVlabs/alpasim/issues/208) |
| V27 | `e2e_challenge` ships results for 8 reference subjects (441 scenes × 3 rollouts) as LFS objects: three ≈ 3.0 MB, five ≈ 1.0–1.4 MB. Drive-IRT has one commit (`82ddd5a4`, "Release v0.0.1"), the pinned one | PARTLY (sizes) | `local_evaluation/data/pai/reference_manifest.json`; LFS pointers | [drive-irt](https://github.com/kesai-labs/drive-irt) |
| — | #198/#215: a stationary ground-truth tail produced false corridor exits; fixed 2026-10-20; leaderboard recomputed 2026-10-22 with a frozen v1 view | from L1 (A1, A2); not re-checked independently | #198, #215 | [alpasim](https://github.com/NVlabs/alpasim) |

## B. RL (AlpaGym, cosmos-rl) and training efficiency

| ID | Fact | Verdict | Evidence | Source |
|---|---|---|---|---|
| V2a | Ratio: per-element log-ratio clamped to ±5, mean over T × 64 × 2 (T = `inference_step`: 5 in the policy preset, 4 in the smoke preset), exponentiated, clipped [0.8, 1.2], per step row | VERIFIED | `replay_objective.py:59-65`; `trainer.py:403-411`; recipes `flow_matching.py:538-578` | [alpagym](https://github.com/NVlabs/alpagym) |
| V2b | KL: closed-form per-transition Gaussian KL (Flow-GRPO's expression under the reversed time convention), averaged; off by default (`kl_beta: 0.0`; no reference model is built) | VERIFIED | recipes `flow_matching.py:579-583`; `default.yaml:160`; `trainer.py:614` | same |
| V2c | Defaults `n_generation: 1`, `kl_beta: 0` ⇒ every advantage 0 ⇒ the trainer skips `optimizer.step()`; NVIDIA's presets use 2–6 | VERIFIED (stronger) | `default.yaml:174`; cosmos-rl `grpo.py:38-45`; `trainer.py:542-547` | same |
| V2d | Slurm presets: `slurm_full_node_1_3_4` (8 GPUs, 1/3/4), `slurm_partial_node_1_2_1` (4 GPUs), `slurm_distributed_1_1_1` (3 nodes), two 5-node presets; "not expected to run out of the box" | VERIFIED | `conf/topology/*.yaml`; README `:67-71` | same |
| V2e | Reward terms: any AlpaSim aggregated metric × scale, or distance-to-GT × scale, without code changes | VERIFIED | `rewards/compute.py:32-43`; `config.py:335-342` | same |
| V2f | NVIDIA's two experiment presets: 5 Hz, 22 valid steps (4.4 s) after 1.6 s of forced ground truth, `n_sim_steps` 30; policy input 320 × 512; traffic skipped. Base default: 10 Hz, 8 valid steps | VERIFIED | `alpamayo_1_5_clrl_test_run.yaml:10, :27-29`; `policy/alpamayo_r1.yaml:14`; topology `alpagym_*gpu.yaml:45-46` | same |
| V2g | `main`: last commit 2026-07-21 (docs), last code change 2026-07-08. Public NVIDIA branches have commits to 2026-09-03 (cosmos-rl 0.4.4 bump, rollout-transport hardening, humanoid/VLA integration); README: "early but active development" | PARTLY (not frozen) | `git ls-remote` with commit metadata | same |
| V2h | AlpaGym is built on cosmos-rl (pinned `d2a2c57`); it pins recipes `1d99fc5`, alpasim-grpc `46b401be` and alpamayo `4cda35d` | VERIFIED | README `:8-12`; `pyproject.toml:26, :69-72` | same |
| V30 | AlpaGym's opt-in `perf` module (`perf.enabled`, default false). PR #36's 10B AlpaGym iteration: 75% (eager) and 58% (graphs) of its time outside trainer compute, derived from the PR's 125.8 of 167.7 s and 62.6 of 107.3 s; one paired sample | VERIFIED (derived) | `default.yaml:107-108`; `perf/README.md:3`; PR #36 body | [PR #36](https://github.com/NVlabs/alpamayo1.5/pull/36) |
| V20 | cosmos-rl: "limited maintenance" banner; v0.4.7 on PyPI 2026-10-22; π0.5 GRPO with `flow_sde` log-probs. NeMo-RL: GDPO in v0.6.0 (2026-04-30), PPO in v0.7.0 (2026-07-29); no flow-matching or VLA path found on the default branch | PARTLY (versions) | cosmos-rl README `:2-14`; PyPI; NeMo-RL releases | [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) |
| V14 | recipes PR #44 (open, not draft; 2026-10-21): `05_Measurement.md`. GPU-hours from the mean wall clock, "never the median"; severe stall = arrival spread > 2 s; MFU (6N), HFU (8N), OFU via `nvidia-smi dmon --gpm-metrics`; a 2B stage-1, 100k-step reference run on 16 nodes (128 GPUs) at 1.77 s/step true wall clock | PARTLY ("draft") | PR #44 @ `35ea6cc` `:7, :40-44, :132-153, :185-192` | [PR #44](https://github.com/NVlabs/alpamayo-recipes/pull/44) |
| V33 | PR #38 (merged 2026-09-23) added `docs/training_efficiency`; PR #44 fills its measurement chapter | VERIFIED | commit `ae5bc10c` | same repo |
| V12 | NVIDIA acquired SchedMD, the developer of Slurm (announced 2025-12-15) | VERIFIED | NVIDIA blog | [blog](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/) |

## C. Student, data and papers

| ID | Fact | Verdict | Evidence | Source |
|---|---|---|---|---|
| V1 | `alpamayo_aligned` labels every clip: 276,299 / 28,317 / 1,536 (= 306,152). `alpasim` clips are "reserved for AlpaSim closed-loop evaluation". The 26.10 version needs `physical_ai_av` ≥ 0.2.3 (PyPI 2026-10-23). Dataset revision `39414de6`, lastModified 2026-10-23 | VERIFIED (from the public card) | the dataset card, section "Splits"; HF API; PyPI | [PAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| V15 | PR #42 @ `5ade7d0`: stage 1 needs ≥ 128 H100s; stage 2 (expert SFT, 1,200 steps, VLM frozen) and stage 3 (DRaFT through the 10-step Euler sampler, < 200 steps) each need one 8 × H100 node. Token-level on-policy distillation starts at 75k of 100k steps in the reproduced config (60k of 80k in `baseline.yaml`). `teacher_vector_weight: 0.0`; `batch["target_action"]` overrides the logged target | VERIFIED | stage READMEs; `configs/on_policy_nav_lingoqa.yaml:96-99, :142`; `modeling_distill.py:529-537` | [PR #42](https://github.com/NVlabs/alpamayo-recipes/pull/42) |
| V16 | PR #45: newest frame of 4 cameras; Domino drafts 16 tokens, greedy; 6 candidates × 64 steps, 5 flow steps, fixed noise (`action-noise.bin`, 384 × 2 FP32). It names **no unreleased model**: "AM 1.5 Nano / AM 1 Nano" are NVIDIA's names for the released 10B checkpoints | CONTRADICTED (the "unreleased" part) | `recipes/alpamayo1_x_edgellm/README.md:12-14, :45` @ `fbed0a7`; NVlabs repo descriptions | [PR #45](https://github.com/NVlabs/alpamayo-recipes/pull/45) |
| V19 | Cosmos-Reason2-2B: NVIDIA Open Model License (not OpenMDW); gated-auto | VERIFIED | HF API | [card](https://huggingface.co/nvidia/Cosmos-Reason2-2B) |
| V8 | TacitVLA, arXiv 2610.17540 (2026-10-13; mostly NVIDIA authors): Qwen3-VL-2B backbone, 3.9B with the flow expert. 441-scene public split: no reasoning 46.4 ± 2.2, explicit 52.3 ± 1.4, tacit 52.5 ± 0.3 (mean ± SD over three rollouts per scene). 127 ms vs 269 ms (= 126 + 143) on one H100, bf16. No reasoning-length sweep found (text search). Only a project page is announced | VERIFIED | Table 6, Table 15, §6.2 | [arXiv](https://arxiv.org/abs/2610.17540) |
| V17 | OPTED, arXiv 2609.20756 (2026-09-17), 441 held-out NuRec scenes, LTFv6. Base: scene 26.5, at-fault collisions 10.2% per scene, 0.23 km per infraction. With OPTED: 41.8, 23.4%, 0.43 km. Two epochs took 17 h on one 8 × H100 node. RoaD (arXiv 2512.01993) uses no teacher model: guidance is the logged human trajectory | VERIFIED | OPTED Table I, §IV; RoaD Alg. 1 | [OPTED](https://arxiv.org/abs/2609.20756) |
| V18 | ECO, arXiv 2609.31383 (2026-09-25; Toronto-led, NVIDIA co-authors): post-processing raised AlpaSim scene scores by 123% (VaVAM) and 22% (DiffusionDrive). "Perceives but Doesn't Brake", arXiv 2610.28184 (2026-10-20; Politecnico di Milano–led, Pavone co-author): a stationary in-path obstacle failure in Alpamayo 1.5 | PARTLY (authorship) | abstracts | [ECO](https://arxiv.org/abs/2609.31383) · [2610.28184](https://arxiv.org/abs/2610.28184) |
| V23 | AD-Memo, arXiv 2609.38641 (2026-09-29): builds on a 2B, single-camera-focused Alpamayo 2 VLA without an action expert; also uses an 8B checkpoint. Neither is public ("internal" is an inference) | PARTLY | §4, App. D.2 | [arXiv](https://arxiv.org/abs/2609.38641) |
| V9 | Cosmos3-Nano-Sim-Auto (V1.0, 2026-10-20): 16B trainable parameters (safetensors total ≈ 15.16B), OpenMDW-1.1, not gated, pitched for "closed-loop driving policy evaluation". On one H100: 2.5 fps for 4 views at 480p, 1.4 fps for 7 views. Not referenced in AlpaSim | VERIFIED | model card `:48, :65, :356-379`; HF API | [card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto) |
| — | Alpamayo 2 Edge: not public on 2026-10-27. HF returns 401 for every guessed ID; no NGC model; no NVlabs repo | from L1 (M1–M8) | — | [blog](https://blogs.nvidia.com/blog/open-models-europe-industry/) |

## D. Toolchain and prices

| ID | Fact | Verdict | Evidence | Source |
|---|---|---|---|---|
| V6 | TensorRT 11.4.0 (support matrix updated 2026-10-05; PyPI 2026-10-08) does not support JetPack; NVIDIA refers to a future 11.x JetPack release. TensorRT 11 removes the weak-typing APIs. `IPluginV2DynamicExt` has been deprecated since TensorRT 10.0 (still declared), and `IPluginV3` is recommended. ModelOpt 0.47.0 (2026-09-23) adds an Alpamayo QAD example; 0.48.0rc1 is dated 2026-10-19. Holoscan SDK 5.0.0 (PyPI 2026-10-20, GitHub 2026-10-21). TensorRT-Edge-LLM v0.11.0 (2026-09-29); PR #260 open and blocked. JetPack 7.2.1 (CUDA 13.2.2, TensorRT 10.16.2) is the newest | PARTLY (plugin date) | TensorRT migration guides; PyPI; releases | [TensorRT docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) |
| V28 | TensorRT-Edge-LLM ships media-aware KV reuse (v0.10.0, 2026-08-12), plus end-to-end FP8 ViT attention and a ViT/audio embedding cache (v0.10.1, 2026-09-03). None is stated as validated for Alpamayo | VERIFIED | release notes | [releases](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases) |
| V22 | vast.ai on 2026-10-27, per offer `dph_total / num_gpus`, cheapest / median over offers: RTX PRO 6000 Workstation 1.00 / 1.56 (33 offers); Server 1.27 / 1.34; H100 SXM 1.79 / 2.40; B200 6.49 / 10.25. An 8 × H100 node is $31.92–36.00/h (Lambda $3.99, RunPod $3.99 with its tier unverifiable, Nebius $4.50 per GPU-h; vast $32.00 for 8 GPUs). vast's $19.20 8-GPU offer had a 0.2-day maximum and no NVLink | PARTLY (RunPod tier) | `notes/vast_2026-10-27/`; vendor pages | [Lambda](https://lambda.ai/pricing) · [Nebius](https://nebius.com/prices) |

## E. Corrections this ledger makes to live documents

| Doc | v4-era text | Correction |
|---|---|---|
| `02` S2, `01` C6, `CLAUDE.md` / `AGENTS.md` | "seeds", "seed test–retest", seed-paired verdicts | Unseeded replicates on `e2e_challenge`. The seed mode is a pre-registered choice, and coupling is measured (V24–V26) |
| `02` S2, `nano_alpamayo` A1, the 2026-10-25 ledger §E | Exclude clips in `alpamayo_aligned` | Key on the split label; A1 draws only from `test` clips (V1) |
| `02` S4, `03` §4 S4, `04` §2 | "Clipped trajectory-level ratio … k3 KL" | Geometric-mean ratio; Gaussian KL, off by default (V2a, V2b) |
| `02` S4 | Comfort/jerk arm in AlpaGym | Not configurable without a new AlpaSim scorer (V2e, V3) |
| `02` S4 | (none) | `n_generation ≥ 2`; the default does not train (V2c) |
| `CLAUDE.md`, `AGENTS.md`, `04` §5, 2026-10-25 ledger B3 | No external AlpaSim merge since 2026-08-17 | `main` only; four non-collaborator PRs merged on `e2e_challenge`, 08-24 to 08-31 (V7) |
| `02` S2 timing | Re-read after the 10-31 close | #224: next release (V7) |
| `02` §0, S3, S5(c) | No closed-loop evaluation of any NVIDIA 2B student | TacitVLA's unpaired research variants exist; narrow the novelty to paired student-vs-teacher (V8) |
| `02` §2 G9, `04` §5, `01` C10 | AlpaGym code-frozen since 2026-07-08 | `main` only; public branches to 2026-09-03 (V2g) |
| `04` §7 E4 | "Identical policy outputs" at model resolution | Cannot be bit-identical; paired equivalence instead (L3-10) |
