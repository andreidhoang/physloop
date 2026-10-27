> **Adoption note (2026-10-27, added when this report entered the repository).** This is the synthesis report behind plan v4.1 (D-015). It was written before the independent check in `notes/V_verification.md`; where the two differ, `verification.md` wins. Plan v4.1 adopted this report's C-IDs in substance, with four differences:
> - **Proposal numbers differ; `DECISIONS.md` is authoritative.**
>   - This report's P-012, the Slurm path and the AlpaGym scaling model, is not a proposal in the plan. The S4 scaling model is a [C] item under P-005, and the S2 Slurm path is S2 [O] 1.
>   - This report's P-013 (two-node measurement) is DECISIONS **P-012**.
>   - Its P-014 (batched driver) is **P-015**.
>   - Its P-015 (renderer fidelity) is **P-014**.
>   - Its P-016 (replay vs CAT-K) is **P-013**.
>   - Its P-017 (one C++ driver) is **P-019**.
>   - DECISIONS adds P-016 (a third S4 arm), P-017 (a PR #42 pilot student) and P-018 (few-step distillation).
> - **"AM 1.5 Nano / AM 1 Nano" are not unreleased models** (V16, contradicted). They are NVIDIA's names for the released 10B checkpoints.
> - **AlpaGym is not frozen.** `main` is quiet, but NVIDIA's public branches carry commits to 2026-09-03 (V2g). A 2B Alpamayo policy has been timed inside it (alpamayo1.5 PR #36).
> - **Coverage marks.** Plan v4.1 counts Q5 as Partial with its [C] items, because S4's scaling model runs on AlpaGym's single-node Slurm presets. This report counts Q5 as None until its P-012.

# Fix PhysLoop's instrument, then rent the cluster

The best evidence PhysLoop can produce for JR2010863 under plan v4's binding rules comes first from repairing its own measuring instrument, and only then from new work: **nineteen corrections are needed in the live documents** (several span more than one document location), and four of the errors would have broken a stage's main claim, because S2's "seed test–retest" runs on unseeded replicates, S4's ★ estimator row describes an objective AlpaGym does not implement, S4's comfort/jerk arm cannot be built from config, and S5's A1 leakage rule excludes every clip in PAI-AV. After those corrections, **23 CORE changes** (C-20 to C-42) cost about **$10–39** at 2026-10-27 prices and recover about **$102–451 of near-certain GPU spend**, mostly through NVIDIA's own bitwise-verified expert CUDA graph and through reuse of S2's teacher sweeps in S3; they add about **+45 to +99 h in total (≈ 23–36 ★ hours by my allocation)** to the ~320-hour plan. **Sixteen OPTIONAL changes** wait for the KC-1 re-forecast, and **six GATED proposals, P-012 to P-017**, need Huy's decision plus P-005 and cost **$523–1,611** together. Coverage of the 13 JR2010863 lines moves from **3 Full / 8 Partial / 2 None** to **4 / 7 / 2** with CORE, where Q4 turns Full only if S6–S7 deliver the C++ driver server, and to **4 / 8 / 1** once P-012 is adopted, because Q5 then moves from None to Partial through NVIDIA's own Slurm path. X1 (industry infrastructure) stays None, and every line that says "large-scale" stays Partial, since nothing below collaborator scale can show cluster leadership. Upstream barely moved between 10-25 and 10-27 (no AlpaSim commit since 10-20; Alpamayo 2 Edge still not public), so most "deltas" are facts the 10-25 ledger missed: AlpaSim #224 ties disclosure of the withheld simulation choices to NVIDIA's next release, outside PRs did merge on `e2e_challenge`, AlpaSim already ships Slurm tooling, and TacitVLA, recipes PR #44 and the alpamayo1.5 PR #36 graph all predate 10-25.

**How to read this report.** L1-n … L7-n are the rows of the seven research lanes in `research_notes/PhysLoop optimizations for JR2010863/`; every consolidated change cites all the rows it merges, and Appendix A maps all 77 rows. All prices use one basis (§2) unless a row says otherwise. Hours follow the lanes' convention, in which L1 and L4 count agent hours as "focused hours" while L6 splits Huy from agent; I report totals in that convention and add a ★ (Huy) share, which is my own allocation by the contract's ownership rule wherever a lane gave none, and is labelled as inference. Every prediction, tolerance and margin stays `____`, and the posting is paraphrased, never quoted ([JR2010863](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Machine-Learning-and-Simulation-Engineer---Autonomous-Vehicles_JR2010863)).

**The 13 lines, and how they map to the earlier 11-line scoring.**

| Line | Paraphrase | Earlier report row |
|---|---|---|
| I1 | Build a closed-loop, simulation-based RL framework that trains end-to-end AV models such as Alpamayo R1 | not scored |
| I2 | Design and improve the accuracy and performance of that RL framework and of the simulator, using NuRec, traffic models and the Cosmos world model | not scored |
| D1 | Lead design and development of large-scale RL training frameworks for multimodal AV foundation models | 1 |
| D2 | Design, build and optimize simulation and data-processing pipelines for scalable policy training | 2 |
| D3 | Measure and improve simulation quality; refine the RL reward | 3 |
| D4 | Keep training workflows on large GPU clusters reliable and fast through monitoring and debugging tools | 4 |
| D5 | Work with researchers to bring state-of-the-art architectures into efficient, scalable training pipelines | 5 |
| Q3 | Deep RL proficiency (PPO, GRPO), including hyperparameter tuning and reward design | 6 |
| Q4 | Exceptional C++ and Python for efficient systems and data pipelines | 7 |
| Q5 | Extensive large-GPU-cluster, HPC and scheduler/orchestrator experience (Kubernetes, Slurm) | 8 |
| X1 | Industry experience with RL or LLM training/fine-tuning infrastructure | 9 |
| X2 | Simulation and closed-loop evaluation of end-to-end driving models | 10 |
| X3 | A record of large-scale data pipelines and algorithm optimization | 11 |

The marks reuse the earlier report's definitions: **Full** when a finished stage shows every key element of the line on the object the line names, **Partial** when it shows some elements, or shows them at smaller scale or on a neighbouring object, and **None** when no stage shows the line.

---

## 1. Nineteen corrections fix the live documents; four errors would sink a stage

Corrections come first because they are free and because the plan's own science depends on them. Four errors matter most. **S2's reliability report** would publish a "seed test–retest" for a branch whose Alpamayo driver stores the seed and never uses it, so its repeats are unseeded replicates that cannot be replayed ([e2e driver](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/driver/src/alpasim_driver/main.py)). **S4's ★ estimator row** names a trajectory-level ratio and a k3 KL, while AlpaGym actually clips a geometric-mean chain ratio and uses a closed-form Gaussian KL ([replay_objective.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/replay_objective.py)). **S4's second reward arm** needs a jerk scorer that AlpaSim does not have at AlpaGym's pin ([scorers @46b401be](https://github.com/NVlabs/alpasim/tree/46b401beaec27977cbb86826df8e39cec2d49e49/src/eval/src/eval/scorers)). **S5's A1 subset** "excludes clips in `alpamayo_aligned`", a file that labels all 306,152 clips ([PAI-AV splits](https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/main/splits)).

In four places the lanes disagreed among themselves, and each time the primary or code-level reading wins. On external merges, L3 and L6 repeat "no external merge since 08-17", but L1 read the PR list and found `e2e_challenge` merges after that date. On seeding, L3 called `e2e_challenge` unverified, but L4 read the driver, the base class, both adapters and the upstream package and found the seed is never consumed. On TacitVLA, L2 found only an aggregator (secondary, low confidence), while L1 and L5 read the arXiv paper. On OPTED's efficiency claim, L1 left the setting unverified and L5 read it in the paper as the render-free WOMD study.

Huy should record the corrections, together with the CORE set, as one plan-amendment decision (the next free ID is D-015) rather than as proposals. `CLAUDE.md` and `AGENTS.md` change in the same edit, as the contract requires. `04` §7 is preserved verbatim by its own convention, so its items get premise notes instead of edits, and the new decision supersedes D-013's wrong bullet instead of rewriting it.

| ID | Live doc, section | Current text (paraphrase) | Corrected text | Source | Implemented by |
|---|---|---|---|---|---|
| **C-1** | `02` §2 G1 ("seeds (PR #149)"); `02` S2 Build ("441 × ≥ 3 seeds", "seed test–retest", negative control "bit-identical re-export"); S2 Upstream value; `04` §5 AlpaSim row; `04` §6 ("a seeded replay matches the offline scorer") | AlpaSim provides seeds on both branches. S2 measures seed test–retest. A seeded replay is exact | On `e2e_challenge` (e121e591), the Alpamayo driver stores `DriveSessionRequest.random_seed` and never uses it, so S2's ≥ 3 repeats are **unseeded replicates**. On `main`, per-tick reseeding (`session.seed + inference_count`) happens only with `force_determinism: true`, which defaults to false in every shipped Alpamayo config. NuRec varies slightly run to run, and PR #149 aims at the same behaviour, not bit-exact replay. Rename to "replicate test–retest" unless Huy ports the per-tick reseed (★ decision). The export can be bit-identical, but the rollout-level negative control is statistical, or a trajectory-diff tolerance if seeded | L4 F1.1–F1.9, Δ1–Δ2 ([main `alpamayo_base.py`](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/driver/src/alpasim_driver/models/alpamayo_base.py); [#128](https://github.com/NVlabs/alpasim/issues/128)); L3 Δ12 superseded | C-26, C-28 |
| **C-2** | `02` S2 Why ("withheld until the challenge ends on 10-31, #224") and Timing; `02` §8 AlpaSim-instability risk; `00` Do-not list | Disclosure of the withheld simulation choices comes after 10-31 | #224 closed 10-23. NVIDIA tied disclosure to **its next release**, not to a date. At S2's start, check for a new AlpaSim release or sync (`pushed_at`, `main` commits). If there is none, label S2 results provisional. The "no S2 runs before 10-31" rule stays | L1 A4, D2; L3 F32 ([#224](https://github.com/NVlabs/alpasim/issues/224)) | C-26 |
| **C-3** | `CLAUDE.md` and `AGENTS.md`, "Reuse NVIDIA's open pieces"; `04` §5 AlpaSim row; DECISIONS D-013 | AlpaSim has merged no outside PR since 2026-08-17 | True for **`main` only** (no commit since 08-18). On `e2e_challenge`, #167 (author_association NONE; the curated 1,761/441 split that `02` §2 G1 already cites) merged 2026-08-31. #111, #161 and #169 from non-collaborator handles merged 08-24/08-31; their association is unverified. Edit both contract files together | L1 A8b, D1 ([#167](https://github.com/NVlabs/alpasim/pull/167)) | — |
| **C-4** | `02` S2 Manifest (no field); no live doc mentions the Harmonizer | (absent) | AlpaSim exposes NVIDIA's NuRec Harmonizer (CosmosPredict2 0.6B, temporal) through one flag. Its default state differs by setting:<br>• ON in `e2e_challenge`'s developer base config;<br>• OFF in `main` and in the competition preset;<br>• ON for the challenge's final runs (#222).<br>Its AlpaSim cost is unpublished. The paper's 212 ms (H100, 1024×576) is not confirmed to be per frame. Pin the state before S2's first run and record it | L3 F2–F6; L1 A3 ([e2e base_config](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/base_config.yaml); [#222](https://github.com/NVlabs/alpasim/issues/222); [NuRec Harmonizer](https://docs.nvidia.com/nurec/nurec/harmonizer.html)) | C-26, C-27 |
| **C-5** | `04` §5 AlpaSim row ("optional CAT-K, disabled, seed unused"); `01` C4 (including "What we do: … renderer agreement; and replay-vs-CAT-K agreement" in every validity record) | CAT-K is an optional, unseeded traffic mode | CAT-K exists **only on `main`**. It is unseeded at code level: the seed is logged, and sampling uses the global RNG. **`e2e_challenge` has no CAT-K** (trafficsim is a stub), so S2's 441-scene protocol is replay-only. Any replay-vs-CAT-K comparison must run both arms on one SHA. Drop per-rollout renderer/CAT-K agreement from `01` C4's "what we do": it is deferred in v4 and infeasible on `e2e` | L3 F21–F22 ([e2e trafficsim](https://github.com/NVlabs/alpasim/tree/e121e59141dc391f6c295513d6dadc840a419149/src/trafficsim); [servicer.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/trafficsim/alpasim_trafficsim/grpc/servicer.py)) | P-016 |
| **C-6** | `02` S4 ★; `03` §4 S4 row; `04` §2 AlpaGym row | GRPO with a clipped trajectory-level ratio and a k3 KL | **Ratio.** A PPO clip per step row, on a **geometric-mean (GSPO-style) chain ratio** exp(mean over T×64×2 per-element log-ratios). The log-ratio is clamped to ±5 and the ratio clipped to [0.8, 1.2]. At T = 5 this equals exp(116.7)/exp(−142.8) on the joint ratio, so the joint ratio is effectively unclipped, and gradients scale by 1/640.<br>**KL.** The **closed-form per-transition Gaussian KL** (Flow-GRPO form), averaged and off by default (`kl_beta` 0). k3 belongs to the open-loop token recipe only, so `03`'s k3 known answer stays valid there.<br>**Advantage.** cosmos-rl's group-standardized episode reward, broadcast to every valid step row | L2 1.13–1.17, KA-5, KA-6, D-4, D-5 ([flow_matching.py](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_x_rl/diffusion/flow_matching.py)) | C-31 |
| **C-7** | `04` §2 AlpaGym row ("default preset: 8 policy steps, `distance_to_gt`"); `04` §7 E1 premise note; `03` §4 S4 traps | The default preset is a usable baseline | **The default trains nothing.** `n_generation` is 1, so cosmos-rl's group advantage is exactly 0 (L2's inference from code); real runs need G ≥ 2.<br>**NVIDIA's own 1.5 RL test preset:**<br>• trains at **5 Hz** with 22 valid steps;<br>• uses `4cam_1080` and policy input [320, 512];<br>• uses lr 1e-4 (the default is 1e-6) and G = 6.<br>**The gate evaluates at 10 Hz over 200 steps,** with the 320×576 contract in `04` §2. Record or pin the mismatch | L2 1.9–1.11, 1.15, KA-3 ([default.yaml](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/host/src/alpagym_host/conf/default.yaml); [grpo.py @d2a2c57](https://github.com/nvidia-cosmos/cosmos-rl/blob/d2a2c57c4bd6496482bc42da19a59b4432705eda/cosmos_rl/dispatcher/algo/grpo.py)) | C-31, C-34 |
| **C-8** | `02` S4 Build ("the same, plus comfort/jerk") | A comfort/jerk reward arm is a config change | **Not config-only.** AlpaSim at AlpaGym's pin `46b401be` has no jerk or comfort scorer, and AlpaGym's reward kinds are only `metric` and `distance_to_gt`. Replace arm B with a config-only, one-variable arm: `collision_any` → `collision_front` + `collision_lateral`, which is the challenge's at-fault definition and S4's acceptance metric. Jerk becomes a post-hoc diagnostic scorer | L2 1.20–1.21, D-7 | C-32 |
| **C-9** | `04` §7 headline 3, premise note ("AlpaGym already runs staleness up to 4") | Stale rollouts beyond k are dropped | Premise note only. AlpaGym's trainer keeps the freshest batch and only *warns* about stale rollouts. Dropping happens in cosmos-rl's dispatcher: verified on `main` e82b56f, unverified at the pin `d2a2c57`. Prefetch requires `dp_shard_size == 1`, and NVIDIA's test run switches prefetch off | L2 1.19, D-6, D-16 ([rollout_filter.py](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/cosmos/rollout_filter.py)) | C-31 |
| **C-10** | `02` §0 "What survived"; §1 research question ("measured as a curve rather than a binary"); S3 and S5 Upstream value | No closed-loop evaluation of an NVIDIA 2B student exists, and the CoC question is open as a curve | **What TacitVLA published** (2026-10-13; NVIDIA AV authors, an inference from author overlap): AlpaSim closed-loop results on the same 441-scene split for a Qwen3-VL-2B-backbone reasoning VLA.<br>• Scene score: no-reasoning 46.4 ± 2.2, explicit 52.3 ± 1.4, tacit 52.5 ± 0.3 (SD across 3 rollouts; no paired test).<br>• H100 latency: 127 ms vs 269 ms.<br>**What it is not:** Edge, a student of a released teacher, or a budget sweep.<br>**Rewording:**<br>• S3 becomes the "first **paired, stratified, student-vs-teacher** verdict of an NVIDIA student".<br>• S5(c) becomes the dose–response between TacitVLA's endpoints, on the deployable student, under an on-device budget.<br>• Label n = 0 by architecture | L1 R1, D11; L5 4.1–4.2 ([TacitVLA](https://arxiv.org/abs/2610.17540)); L2 5.9 superseded | C-37 |
| **C-11** | `02` §2 G5, S3 KC-E, §5 collaborator row, S5 Why; `01` Model facts; `04` §5 and §9; `nano_alpamayo/README.md` | PR #42 recommends ≥ 128 H100s; offline distillation trains the student only on the teacher's state distribution | **Scale.** The 128-H100 figure is **stage 1's alone**. Stage 2 (expert SFT, 1,200 steps) and stage 3 (DRaFT, < 200 steps) each run on **one 8×H100 node**.<br>**Stage 1 already includes open-loop, token-level on-policy distillation** (from step 75k of 100k).<br>**The student's action head is trained on logged trajectories,** not on teacher actions (teacher-vector weight 0.0). A DAgger arm with teacher labels therefore changes both the state distribution and the target | L5 1.4–1.8, Δ1–Δ3 ([PR #42](https://github.com/NVlabs/alpamayo-recipes/pull/42) @ 5ade7d0) | C-24, C-43 |
| **C-12** | `02` S5 Why ("RoaD and OPTED report large sample-efficiency gains in AlpaSim") | Both report large sample-efficiency gains in AlpaSim | **OPTED.** Its ≈ 6.8k-vs-6.2M-episode figure is from its **render-free WOMD** study, not from sensor-based AlpaSim runs. In AlpaSim, OPTED raised LTFv6's scene score from 26.5% to 41.8% while LTFv6's **at-fault collisions rose from 10.2% to 23.4%**.<br>**RoaD.** It runs no RL or DAgger comparison, and it uses no teacher | L5 2.1–2.5, Δ9–Δ10 ([OPTED](https://arxiv.org/html/2609.20756); [RoaD](https://arxiv.org/abs/2512.01993)); L1-11 superseded | S5 Why text; C-43 |
| **C-13** | `02` S5 ★ ("exclude `alpamayo_aligned` clips"); `02` S2 contamination report; `nano_alpamayo` A1; `04` §9 PAI-AV row | `alpamayo_aligned` marks the clips aligned with Alpamayo training; exclude them | **The file labels every clip:** train 276,299 / test 28,317 / alpasim 1,536. Excluding the clips "in" it excludes everything. A1 should draw only from `test`-labelled clips: location-grouped, read with `physical_ai_av` ≥ 0.2.3 at PAI-AV revision 39414de6. Keep `alpasim` clips out of A1, because they are the closed-loop pool. The S2 report states overlap per label | L5 5.6, Δ7 ([PAI-AV card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); [devkit](https://pypi.org/project/physical_ai_av/)) | C-38 |
| **C-14** | `04` §7 E4 ("identical policy outputs"); `04` §4 Renderer ("Alpamayo drivers reject images smaller than 320×576") | Render at model resolution: ≥ 4× scenarios per renderer GPU with identical outputs | Premise note:<br>• **Outputs differ.** Rasterizing at 320×576 differs from resampling a 1080p frame, so outputs cannot be bit-identical; acceptance must be a paired equivalence verdict.<br>• **The rejection path is unverified for A1.5.** It exists in `base.py`, but the A1.5 path does not call it.<br>• **No 320×576 camera preset exists.**<br>• **The gain is bounded.** 3DGS cost is not pixel-bound, and NVIDIA's topologies give the renderer 1 of 8 GPUs (A1.5 eval) or 4 of 8 (challenge), so E4's end-to-end gain is Amdahl-limited | L3 F18, Δ15, I-12, I-13; L7 §5 | C-29 |
| **C-15** | `01` FlashDrive bullet; `04` §4 lessons | 717 → 151 ms came partly from "adaptive steps" | FlashDrive's "adaptive steps" is a **fixed velocity-reuse schedule**: 4 of 8 steps reuse the previous velocity. There is no per-sample rule and no fixed-K ablation. Its streaming KV reuse also needs expert-only fine-tuning: minADE6 is 0.96 without it vs 0.77 with it | L5 3.1–3.2, Δ12 ([FlashDrive](https://arxiv.org/abs/2608.12932)) | C-44 |
| **C-16** | `02` S7 Build ("AlpaSim applies each action after the measured on-device delay") | Latency injection is something to build | **AlpaSim already ships both mechanisms on both branches:**<br>• `driver_source=external_static` / `external_dynamic` for an external driver;<br>• `planner_delay_us`, a fixed `DelayBuffer` modelling capture → planner → controller delay. NVIDIA's own Alpamayo configs used 200 ms until 2026-03-24.<br>**Injection is quantized to the control step.** At Δt = 100 ms, delays of 139, 171 and 200 ms all land 2 ticks late, so p50 and p99 inside one 100-ms band are indistinguishable | L7 2.1–2.4b, Δ3 ([config.py](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/config.py); [delay_buffer.py](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/delay_buffer.py)) | C-25 |
| **C-17** | `02` §5 price basis; S3, S4 and S5 Cost lines; `04` §9 prices | 10-25 basis; an 8×H100 node at $15–32/h | **The 10-27 basis is in §2.** The low end of the 8×H100 range cannot be rented for 24–48 h: the $19.20 offer allows 0.2 days and reports `bw_nvlink` 0.<br>**Re-priced stages:**<br>• S2 $136–530;<br>• S3 $102–530, plus Super $61–204;<br>• S4 training $403–1,080;<br>• **S5 node $766–1,728** (was $360–1,536) | L1 Q8, D16, L1-15 ([vast.ai API](https://console.vast.ai/api/v0/bundles/); [Lambda](https://lambda.ai/pricing); [Nebius](https://nebius.com/prices)) | §8 |
| **C-18** | `02` S6 Build ("a camera-batched ViT; streaming reuse of history frames"); `02` §2 G6 ("ModelOpt 0.43 PTQ") | S6 builds these items; upstream offers ModelOpt 0.43 | **TensorRT-Edge-LLM 0.10.x already ships:** end-to-end FP8 ViT attention, DART token pruning, a ViT-embedding cache and media-aware KV reuse.<br>**ModelOpt 0.47.0 (09-23) adds** FP8 vision-encoder recipes for `qwen3_vl` and `examples/alpamayo/qad.py`.<br>Building them in S6 would race NVIDIA, so consume them instead (this strengthens P-008). The recipes pin of 0.43.0 stays unless Huy decides otherwise | L7 1.7–1.10, 3.1–3.2, Q6 ([Edge-LLM releases](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases); [ModelOpt releases](https://github.com/NVIDIA/Model-Optimizer/releases)) | P-008 (existing proposal); C-41 |
| **C-19** | `02` S2 Acceptance ("every retry is logged, because prefill emits all-NaN trajectories") | NaN retries are the retry audit | **Also log:**<br>• renderer-OOM re-queues (NVIDIA re-queues OOM scenes, #222);<br>• rollout retries (`runtime.max_rollout_retries`, default 2).<br>AlpaSim exports no retry counter, so rates are computed post-hoc | L1 A3; L6 Q1 #10, #16 | C-26 |

---

## 2. Upstream barely moved since 10-25; the ledger missed more than NVIDIA shipped

Very little changed in the 48 hours after the plan was adopted. AlpaSim has had **no commit on any branch since 10-20**; its only new activity is two challenge-logistics issues (#225, #226) closed on 10-26 and an unanswered 10-26 follow-up on #219 ([AlpaSim](https://github.com/NVlabs/alpasim)). **Alpamayo 2 Edge was still not public on 10-27**: nine guessed Hugging Face IDs return 401, the NGC catalog has no Alpamayo model resource, NVlabs has no Edge repository, and the 10-21 blog still promises the release "later this month" ([HF API](https://huggingface.co/api/models?author=nvidia&search=alpamayo&full=true); [blog](https://blogs.nvidia.com/blog/open-models-europe-industry/)). AlpaGym (frozen since 07-21), recipes PRs #42, #43 and #45, TensorRT-Edge-LLM v0.11.0 with its PR #260, and JetPack 7.2.1 are unchanged. The only new releases are cluster tooling, NVSentinel v1.29.0 and KAI Scheduler v0.16.14 on 10-26 ([NVSentinel](https://github.com/NVIDIA/NVSentinel)). Prices moved, as the tables below show.

The larger delta is what the 10-25 ledger missed even though it predates 10-25. The plan relies on four AlpaSim facts it did not record: #224's "next release" wording; #222's statements (the Harmonizer is on for final runs, the internal test set is harder, OOM scenes are re-queued, the official topology is `8gpu_32rollouts`, and system-clock policies carry no guarantee); #198's recalculated leaderboard and minimal reproduction; and the `e2e_challenge` merges. Four pieces of tooling change the cost model: AlpaSim's Slurm run methods (`SLURM`, `SLURM_ENROOT`) with CUDA MPS and the `optimize-slurm-topology` skill (synced 07-01, SLURM_ENROOT 08-11); alpamayo1.5 PR #36's opt-in expert CUDA graph (merged 08-29, bitwise identical on 160 of 160 paired evaluations); AlpaSim's `planner_delay_us` and external-driver modes; and recipes PR #44, the training-efficiency "Measurement" chapter (10-21) ([AlpaSim Slurm skill](https://github.com/NVlabs/alpasim/blob/main/.agents/skills/optimize-slurm-topology/SKILL.md); [PR #36](https://github.com/NVlabs/alpamayo1.5/pull/36); [PR #44](https://github.com/NVlabs/alpamayo-recipes/pull/44)). Four papers narrow PhysLoop's novelty: TacitVLA (10-13), OPTED's details (09-17), ECO (09-25), whose trajectory post-processing alone moved AlpaSim scene scores by +22% to +123%, and "Perceives but Doesn't Brake" (10-20) on Alpamayo 1.5's stationary-lead failures ([ECO](https://arxiv.org/abs/2609.31383); [arXiv 2610.28184](https://arxiv.org/abs/2610.28184)). And Cosmos3-Nano-Sim-Auto (card dated 10-20; 16B; OpenMDW-1.1; ungated) is a world model built for closed-loop policy evaluation, with no AlpaSim integration and no published closed-loop agreement study ([card](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto)).

Together these facts change five things in the plan. S2 must pin the Harmonizer and the scorer SHA and treat its repeats as replicates. S4 must be rewritten against AlpaGym's real objective and defaults. S5's novelty claim narrows to closed-loop, state-visited correction of the action head, because PR #42 already distils open loop. Two S6 build items would race NVIDIA's shipped TensorRT-Edge-LLM features. And S7's build effort shrinks, because AlpaSim already has the mechanism.

**The price basis.** I re-ran L1's saved 10-27 vast.ai offers with the lane's own `stats_vast.py` during synthesis, and the table reproduces exactly. Each row carries its date and source; the table after it lists where the lanes disagree.

| Item | 10-27 price (source; date) | 10-25 price (plan) |
|---|---|---|
| RTX PRO 6000 Workstation | **$1.00 cheapest / $1.56 median** per GPU-h; 33 offers, down from 49 (vast.ai on-demand, 08:31 UTC) | $1.00 / $1.60 |
| RTX PRO 6000 Server | $1.27 / $1.34 (vast.ai, 10-27) | $1.24 / $1.47 |
| H100 SXM | **$1.79 / $2.40** (vast.ai, 10-27) | $1.93 / $2.40 |
| 8×H100 node, ≥ 24 h | **$32.00/h** (vast.ai, Germany; the only offer with ≥ 24 h availability verified). List prices: $31.92/h (Lambda, read 10-27, page undated) and $36.00/h (Nebius, read 10-27, page undated). RunPod community $21.52/h only if an 8-GPU pod exists, which is unverified (pricing page "updated 2026-10-08") | "$15–32/h" |
| 8× RTX PRO 6000 VM (root, for Slurm + Enroot) | **$16.00/h ($2.00/GPU), a single verified offer in Czechia.** This is my synthesis-time reading of L1's raw 10-27 file, not a lane finding. Nebius lists $1.80/GPU; its 8-GPU preset is unverified | L6 (10-25 snapshot): $18.78/h ($2.35/GPU) |
| 2-node InfiniBand cluster | Nebius 2 × 8 H100 at **$72/h** (read 10-27; self-service; quotas unverified). RunPod Instant Clusters list H200 SXM at $6.35, unit unstated. Lambda 1-Click requires 16 GPUs for 2 weeks: ≈ $33,116 | — |
| Jetson AGX Thor dev kit | $5,499 (NVIDIA Jetson FAQ, read 10-27) | $5,499 |

| Topic | One lane says | Another says | Resolution |
|---|---|---|---|
| vast.ai on 10-27 | L6: HTTP 403 at 08:29 UTC on three variants, so it used the 10-25 snapshot | L1: offers pulled at 08:31 UTC and saved | L1's table reproduces from the saved JSON, so the 10-27 basis is used. L6's dollar figures are re-priced from their GPU-hours |
| 8× RTX PRO 6000 VM | L6 (10-25): $18.78/h | Synthesis re-analysis (10-27): $16.00/h, one offer | Both are shown. P-012's premium is priced on both |
| 8×H100 low end | Plan and L6 (10-25): $15.46/h (India; CUDA ≤ 12.6, driver 560) | L1 (10-27): $19.20 offer is unusable (0.2-day maximum, NVLink 0); $32.00 is the only ≥ 24-h offer | S5's node is $766–1,728. $516 is possible only on the unverified RunPod pod |
| Multi-node | L1-8: RunPod H200 at $101.6/h for 2 nodes, $152.4/h for 3, unit unstated | L2-6: vast $139–230 (no vast cluster offer exists) or Lambda $287–412 (clusters carry a 2-week minimum) | L6-7's Nebius $72/h is the only priced, self-serve InfiniBand option → P-013 |
| Renderer share of step time | L3: NVIDIA's 4:4 challenge topology ("renderer-stability") implies rendering is not negligible | L7: the 7:1 A1.5 evaluation topology implies a driver share f ≈ 0.875 | Two inferences from two NVIDIA topologies. Neither is a prediction; C-29 measures it |
| Expert-graph saving | L7: −100 to −312 GPU-h counting 26–27 sweeps, which includes TransFuser and S6 | — | I use L7's S2–S5 basis (22–23 Alpamayo sweeps), remove the S3 teacher sweeps that C-21 reuses, and split teacher-path from student sweeps (§8) |

---

## 3. Twenty-three CORE changes: six savings, then the instrument

CORE is what Huy should adopt now into the stage cards. Its priority order follows the brief's first principles: savings that are free or bitwise-safe come first (C-20 to C-25), then the changes that make the plan's own science correct (C-26 to C-28, C-31 to C-34 and C-36 to C-38), and last the cheap role evidence that rides on runs the plan already makes (C-29, C-30, C-35 and C-39 to C-42). The register below keeps numeric order, and §8 totals use the same three-way partition.

None of these changes races NVIDIA, and every ★ core stays Huy's. The hours column gives the lane total, with the ★ share stated where the lane stated it.

| ID · change (lane rows) | Stage-card change: **B** Build · **★** · **A** Acceptance · **P** Prediction · **K** Kill | Cost and hours (10-27 prices) | Gate |
|---|---|---|---|
| **C-20 · NVIDIA's expert CUDA graph behind a bitwise gate** (L7-2) · S2 → S3–S5 · G7 | **B:**<br>• Add a driver flag that calls `enable_diffusion_expert_cuda_graph()` after model placement.<br>• Pin and record the resolved `alpamayo1_5` commit (≥ 24179cf). The driver's dependency is an unpinned git source.<br>• Record graphs on/off, `max_batch_size`, replicas and `force_determinism` (C-26).<br>• Decide before S2's first sweep, so S2's teacher rollouts stay reusable (C-21).<br>**★:** the sweep determinism mode. Deterministic sweeps carry the bitwise claim per run. Default-mode sweeps carry "same kernels ⇒ same output distribution", stated in the manifest.<br>**A:** eager and graph outputs are bitwise identical (max abs diff 0) on every compared tick, with `force_determinism=true` and the same seeds, and S2's negative control still passes. Otherwise the lever is not used.<br>**P:** sweep GPU-h saved `____`.<br>**K:** bitwise failure → eager driver | Test: +1 GPU-h ($1.00–1.56).<br>**Saves 48–154 GPU-h ($48–240) on teacher-path sweeps:** S2's 2–3 A1.5 sweeps, S3's mini-control and ≤ 10 S4 sweeps.<br>A further 27–92 GPU-h ($27–143) if Edge exposes the same API.<br>Per sweep the saving is 11.3–13.6%: Amdahl with f = 0.73–0.875 (inferred) and s = 1.184, from one "preliminary" paired AlpaGym sample.<br>+3–5 h (agent; Huy reviews) | P-005 (S2 tier). Upstream PR only after NVIDIA's post-10-31 sync |
| **C-21 · S3 teacher-arm reuse with a concurrent drift sentinel** (L4-3) · S3 · G5 | **B:**<br>• Reuse S2's 1.5 BF16 and FP8 rollouts as S3's reference arm when every manifest field matches: SHA, lockfile, renderer and Harmonizer state, GPU edition, `n_sim_steps`, seed mode and driver levers.<br>• A concurrent mini-control re-runs the teacher on a pre-registered random road sample.<br>**★:** the mini-control fraction, the equivalence margin and the sentinel rule.<br>**A:** the sentinel result is printed beside the verdict.<br>**P:** paired teacher re-run minus S2 `____`.<br>**K:** sentinel fails → full teacher sweeps | **−54 to −135 GPU-h (−$54 to −$211)** at an *illustrative* 20% mini-control.<br>+1.5 h (★ 0.5) | none |
| **C-22 · S3 estimand decision: more scenes, fewer replicates** (L4-2) · S3 · G5 | **B:** a decision slot before S3's pre-registration.<br>• If the claim targets a road super-population, verdict sweeps run more distinct scenes at one replicate each, drawn from the 2,202-scene pool minus contamination. A pre-registered random subsample at ≥ 2 replicates estimates σ_w.<br>• If the claim is about these 441 scenes, only replicates help.<br>• S4 and S5 stay on the 441.<br>**★:** inference target, scene pool, power branch with design effect 1 + (m − 1)ρ.<br>**A:** unchanged.<br>**P:** S2's within-scene share of paired-difference variance `____`.<br>**K:** none | **Conditional:** −19 to −92 GPU-h (−$19 to −$144) on the 1–2 student sweeps once C-21 is in, or −56 to −184 GPU-h without C-21. Illustrative ρ = 0.6; not additive with C-47.<br>+2 h (★ 1) | none. The contamination report must also cover the 1,761 |
| **C-23 · KC-E capture list, re-scoped fallback, licence** (L1-1, L5-8, L1-14) · S3 · G5 | **B:**<br>• **Release-day capture** on 10-31 and at S3's start: repo ID and revision; licence terms for evaluation, fine-tuning and derivatives; backbone; action expert and flow steps; runtime-CoC switch; camera count; disclosed teacher; any NVIDIA closed-loop number.<br>• **While KC-E is open,** S3 measures NVIDIA's deployment levers on the released teacher as known-order pairs, inside S3's 3–4 sweeps: K = 5 with fixed noise vs K = 10 (PR #45's setting); CoC capped vs uncapped; FP8 vs BF16 from S2.<br>• **The PR #42 stage-1 pilot leaves the KC-E branch.** Pre-write its experiment file with `____` slots only.<br>**★:** the fallback decision.<br>**A:** the capture list is complete.<br>**P/K:** S3's slots and the KC-E order are unchanged | **Counterfactual:** avoids ≈ 629 H100-h (≈ $1,126–1,510) for a 10% pilot that would stop before the recipe's on-policy phase. Full stage 1 is ≈ 6,293 H100-h ($11.3k–15.1k), a lower bound by inference.<br>−10 to −20 h in that branch; +1 h now | KC-E, P-005. The PR #42 student's VLM, Cosmos-Reason2-2B, is under the **NVIDIA Open Model License**, not OpenMDW: read S5's fine-tuning prerequisite and P-004 against it ([card](https://huggingface.co/nvidia/Cosmos-Reason2-2B)) |
| **C-24 · S5(a) learner = PR #42's stage-2 trainer** (L5-1) · S5 · G5 | **B:**<br>• The learner is `alpamayo1_x_distillation_expert_sft/train.py` @ 5ade7d0: VLM frozen; expert and action projections trained.<br>• Teacher actions at visited states enter through `batch["target_action"]`. Stage 3's DRaFT is the second available loss form.<br>• Before round 1: DRaFT `--parity-check` bit-equal, overfit one batch, loss at init.<br>• The first pilot check is whether stage 2's batch builder passes `target_action` through (unverified).<br>**★:** unchanged; NVIDIA's two loss forms are inputs to Huy's loss and aggregation math.<br>**A:** + the parity check and trainer commit are in every round manifest.<br>**P/K:** unchanged | **−4 to −8 h** (+3 h if Edge's architecture differs); $0 | KC-E, KC-L2 |
| **C-25 · S7 on AlpaSim's own external driver and delay buffer** (L7-8) · S7 · G7 | **B:**<br>• Thor serves `EgodriverService` through `driver_source=external_static`; no fork.<br>• Constant-delay arms via `planner_delay_us`, at measured model latency and at glass-to-action (P-009 ii).<br>• A per-step trace-driven delay becomes an AlpaSim PR only if Huy's rule says the constant arms fall short.<br>**★:** lockstep semantics, injection design and TOST margin (unchanged).<br>**A:** unchanged (Thor-in-loop equals cloud within S2's noise floor by TOST; latency-injected score reported with a CI).<br>**P:** score lost at the injected delay `____`.<br>**K:** KC-7 | **−4 to −8 h**, which fund C-42; $0 (S7 stays ≈ $200) | P-002 |
| **C-26 · Full-protocol manifest and the #198 known answer** (L1-3, L1-4, L1-9, L1-12, L3-1, L4-1, L6-1, L7-2) · S2, reused S3–S7 · G1, G2, G7 | **B:** the manifest adds:<br>• renderer image (`nre-ga:26.04`) and flags (`--no-enable-nrend`, Harmonizer); `render_bundling`; camera request (4 × 1080×1900, decoded width);<br>• `nre_max_workers`, `n_concurrent_rollouts`, `replicas_per_container`, driver `max_batch_size`; `trafficsim` mode; topology and policy clock source;<br>• evaluator SHA at or after #215 (merge 1b024f0f); branch, `force_determinism` and seed mode; `alpamayo1_5` commit and graphs flag;<br>• controller and trajectory post-processing identity and version; OOM re-queue and retry counts per policy; the Slurm fields if P-012 is adopted.<br>Also add #198's minimal reproduction as a failing known-answer test: two stationary-tail GT polylines, pre-fix 5.0 m lateral, expected ≈ 0. Keep #207/#208 fields locally and file no competing PR (NVIDIA acknowledged #208 on 10-15).<br>**★:** semantics for corridor exits inside the GT terminal segment; motion-limit diagnostics (ECO moved scene scores +22–123% with plan geometry alone).<br>**A:** a record missing any field is "provisional"; the #198 test passes.<br>**P/K:** none | $0; +3–5 h (mostly agent) | none |
| **C-27 · Harmonizer cost pilot and pinned state** (L3-2, L1-4) · S2 · G7, G1 | **B:**<br>• 5 scenes × 1 replicate, A1.5 BF16, Harmonizer OFF vs ON, renderer `--enable-timing`.<br>• Pin the state for all S2 sweeps and record it in `experiments/` before the first run.<br>**★:** the choice. ON gives parity with the challenge's final runs; OFF gives cost and comparability with local runs.<br>**A:** per-step renderer time ON vs OFF, at p50 and p99.<br>**P:** seconds added per step at 4 × 1080×1900 on RTX PRO 6000 `____`.<br>**K:** above Huy's ceiling `____` s/step → pin OFF and run ON only inside P-015 | 0.5–1.5 GPU-h ($0.50–2.34); +1 h.<br>It prevents a possible **+249–873 GPU-h ($249–1,362 at 10-27 prices; L3 gave $249–1,396 on the 10-25 basis)** if ON for all four sweeps (inference: assumes 212 ms is per frame and scales with pixels) | P-005 (S2 tier); after 10-31 |
| **C-28 · Replicate semantics, CRN audit, noise decomposition** (L4-1, L3-3) · S2 · G1, G3·R1 | **B:**<br>• **Seed mode.** Either port `main`'s per-tick reseed into the pinned `e2e` driver, or label S2's repeats unseeded replicates (`seed_mode` in the manifest).<br>• **CRN audit.** Var(d) under matched-seed vs label-permuted pairing of the same rollouts.<br>• **Negative control.** A same-policy, same-seed re-run, which bounds NuRec's run-to-run variation.<br>• **Noise decomposition:**<br>  (1) render-determinism probe: 441 scenes × 10 poses × 4 cameras rendered twice, per Harmonizer state;<br>  (2) policy probe: identical inputs fed twice;<br>  (3) traffic: replay is deterministic;<br>  (4) first-divergence tick of each replicate pair.<br>**★:** the ρ_w\|scene estimator with a road-cluster CI; the S3–S7 pairing rule; the variance-component model.<br>**A:** ρ_w with CIs for (BF16, FP8) and (TransFuser, 1.5); each component's share of discordance with a CI; replicate κ per deviation bin published as the floor for every later agreement claim.<br>**P:** ρ_w BF16 vs FP8 `____`; ρ_w TransFuser vs 1.5 `____`; throughput change in deterministic mode `____`; renderer share of discordance `____`.<br>**K:** renders not repeatable at matched poses → report the renderer as a noise source before any verdict; port fails → unseeded, no CRN | ≈ $0 inside sweeps (1.2–3.8 GPU-h standalone, ≤ $6); +6–8 h (★ 3–4) | none |
| **C-29 · S2 throughput NVIDIA's way: stabilized metric, critical-path profile, one-variable config levers** (L1-6, L3-5, L6-2, L7-1, L7-3, L3-10 bundling) · S2 · G7, G1 | **B:**<br>• **Metric.** "First measured throughput" becomes AlpaSim's stabilized full-run `seconds_per_rollout`. It is read only after the 30–45 min stabilization that NVIDIA's `optimize-slurm-topology` skill prescribes, and recorded in that skill's Markdown + JSON experiment record.<br>• **Alongside it:** rollouts/GPU-h; a per-service GPU-time split from Prometheus (`e2e`'s 14 unprefixed series, not `main`'s names); `rpc_queue_depth_at_start` min/max; DCGM tensor-active; a driver CUDA-event split (encode / prefill / decode / expert / host residual) with NVTX and nsys. GPU-h comes from the mean s/step.<br>• **Amdahl first.** Bound every lever with S = 1/((1 − f) + f/k) before running it.<br>• **Arms, one variable each:** 1 vs 2 driver replicas per GPU (bitwise per rollout); `render_bundling` NONE vs `BATCH_RENDER_RGB` (expected bit-identical). MPS only under P-012.<br>• **Topology-invariance control:** 100 scenes × 1 replicate at two concurrency settings, compared by the gate's paired TOST.<br>**★:** the stabilization criterion, invariance margin, cost model and bottleneck diagnosis.<br>**A:** every throughput number names preset, SHA, GPU edition and Harmonizer state; `02` §5's 34–85 GPU-h single-stream bound is replaced by the measurement.<br>**P:** renderer share at 4 × 1080p `____`; driver share f `____`; top driver stage `____`; 2-vs-1 replica throughput ratio `____`; steady-state GPU-h per 1,000 rollouts `____`.<br>**K:** a lever that breaks invariance is not used for verdict sweeps | +6.6–14.4 GPU-h ($7–22), plus ≈ 1.3 node-h of candidate start-up ($19–21, likely absorbed into real sweep rollouts); +10–15 h (★ 3–4) | P-005 (inside S2) |
| **C-30 · Validity envelope as a dose–response with NVIDIA's clip-quality score** (L3-4, L1-3) · S2 · G2 | **B:**<br>• Per-policy failure rate and `scene_score` by deviation bin (< 1, 1–2, 2–4, ≥ 4 m), drawn as a curve.<br>• Join `clip_ratings_*.csv` (`quality_score`, perception-based beta; `time_of_day`; `ego_speed`).<br>• Add an endpoint-overshoot class (post-#215 semantics) and use `img_is_black` as a render-failure count.<br>• ≥ 4 m exits stay separate.<br>**★:** strata, semantics, covariate analysis, diagnosis.<br>**A:** curves with road-clustered CIs; rating coverage stated (ratings exist from 26.02).<br>**P:** rank correlation between `quality_score` and validity-exit rate `____`.<br>**K:** none | $0; +2–3 h | KC-L to publish |
| **C-31 · S4 estimator truth: known-answer suite and step-1 parity** (L2-2) · S4 · G9 | **B:** failing tests from KA-1…KA-8, all recomputed by script before use:<br>• group advantage, e.g. [−10, 0, 0, 0] → [−1.732051, 0.577350 × 3];<br>• affine invariance;<br>• G = 1 → A = 0;<br>• mixed-group probability 1 − p^G − (1 − p)^G;<br>• geometric-mean ratio: N = 640 at T = 5; clip ↔ exp(116.6858)/exp(−142.8119); gradient scale 1/640;<br>• closed-form KL coefficients c(t) = 0.5 / 0.65025 / 1.734 / 3.9015 / 10.404 at a = 0.2;<br>• k3 for the token recipe only;<br>• all-safe std amplification: ±1.7167, and one collision → −√5.<br>**Step-1 parity:** with `allowed_outdated_steps=0` and one optimization iteration, the ratio must equal 1 within Huy's tolerance (the same idea as PR #42's bit-equal sampler check). Real runs need G ≥ 2.<br>**★:** estimator math and tolerances.<br>**A:** every KA passes; parity reported.<br>**P:** parity deviation at step 1 `____`.<br>**K:** parity failure blocks training | CPU; ≤ 1 H100-h inside the first run (≤ $2.40); +4–6 h | none |
| **C-32 · S4 reward arms, config-only and one variable** (L2-3) · S4 · G9 | **B:**<br>• Arm A is `progress_safety` as shipped.<br>• Arm B replaces `collision_any` (−10) with `collision_front` + `collision_lateral`. Record or cap the −20 a front-and-lateral episode scores.<br>• Jerk becomes a post-hoc diagnostic scorer beside the planned stall and oscillation scorers.<br>**★:** term choice and scales.<br>**A:** unchanged (the three paired tests on the 441 held-out scenes).<br>**P:** first hacking mode under A `____` and B `____`; rear-collision change A → B `____`.<br>**K:** KC-4 | $0 (still 2 arms × 3 seeds); −6 to +2 h. The −6 h holds only if S4's ~50 h already included building a jerk scorer, which the plan does not state (conditional) | P-005 as before; KC-L2 |
| **C-33 · Zero-scale hacking monitors and the stationary-lead stratum** (L2-4, L1-5) · S4 (S2/S3 strata) · G9 | **B:**<br>• **Log as zero-scale terms, without changing the reward:** `progress_rel`; `dist_traveled_m` vs `gt_dist_traveled_m` (stall); `wrong_lane`; `min_distance_to_lane_boundary_m`; `min_distance_to_obstacle_m` (a close-encounter proxy); `plan_deviation`; `collision_rear`.<br>• **Candidate stratum:** a stationary or slow in-path lead (#224 → alpamayo1.5 #40). Read it together with the stall monitor.<br>**★:** verifier logic; whether the stratum is pre-registered.<br>**A:** every monitor appears in the per-episode logs. Whether zero-scale terms reach the trainer's logs is unverified; the first run checks.<br>**P:** the monitor that moves first `____`.<br>**K:** none (instrumentation; KC-4 unchanged) | $0; +1.5–3 h | none |
| **C-34 · S4 train/eval protocol pinned, or the mismatch recorded** (L2-7) · S4 · G9 | **B:**<br>• Huy chooses between training at 10 Hz (and setting `expected_valid_steps`) and keeping NVIDIA's 5 Hz / 22-step preset with the mismatch recorded in the manifest.<br>• Check whether the gate's 1.5 driver feeds 320×576, while AlpaGym trains at 320×512.<br>**★:** the choice.<br>**A:** the control rate, valid steps, camera preset and input size of every S4 run are manifest fields.<br>**P:** score change from the train/eval mismatch `____`.<br>**K:** none | $0 if recorded. Training at 10 Hz roughly doubles rollout steps (P-005).<br>+1–2 h | none (P-005 for 10 Hz) |
| **C-35 · Training telemetry in NVIDIA's own vocabulary** (L1-10, L2-1 instruments, L5-5 metrics, L6-5 i–ii, L7-10) · S4, S5 · G9, G5, G7 | **B:**<br>• **AlpaGym.** `perf.enabled=true` on every S4 run (perf scopes plus AlpaSim's "Simulator internals"), and a `weight_version` lag histogram.<br>• **PR #44 definitions in every training log:**<br>  – wall-clock s/step from log timestamps, with ~20 warm-up steps dropped per autoresume segment;<br>  – GPU-h from the mean wall clock, never the median;<br>  – severe stall = cross-rank arrival spread > 2 s;<br>  – MFU (6N), HFU (8N) and OFU via `nvidia-smi dmon --gpm-metrics`;<br>  – a CUDA-event split into arrival / forward / backward / host residual.<br>• **Collective forensics.** PyTorch Flight Recorder (buffer 2000, dump on timeout) and an `ncclras` snapshot on any stall.<br>• **Reconciliation.** Compare the preserved E1 / headline 1 with NVIDIA's PR #36 AlpaGym numbers: 75.0% of iteration time outside trainer compute in eager mode, 58.4% with graphs (one pair).<br>**★:** the reconciliation; FLOPs per sample.<br>**A:** every training number states its averaging window.<br>**P:** existing E1 slot.<br>**K:** none (instrumentation) | $0; +4–7 h (Huy +1–2) | none. Build no generic harness: NVIDIA lists its own as coming in PR #44 |
| **C-36 · S1 intersection–union power accounting** (L4-7) · S1 · G1 | **B:**<br>• Power tables add overall GO power = Π per-stratum power for road-disjoint strata. Four strata at 0.80 each give 0.41, and 0.80 overall needs 0.946 per stratum.<br>• S3–S5 allocate extra replicates by minimax SE-to-margin, not by Neyman.<br>**★:** allocation objective.<br>**A:** the intersection–union tables exist at KC-1.<br>**P:** S3 strata returning INSUFFICIENT `____`.<br>**K:** KC-1 unchanged | $0; +1 h ★ | none |
| **C-37 · S5(c) CoC-budget curve as a clean one-variable measurement** (L5-4, L1-2) · S5 · G5 (A6) | **B:**<br>• (1) Measure the student's natural CoC length first, uncapped. PR #42 evaluates with a 256-token cap; the model family runs ≈ 16 tokens.<br>• (2) Place caps inside that range: a cap above the realized length is inert.<br>• (3) Budget forcing on the same checkpoint: stop at n tokens and append the trajectory-start token.<br>• (4) Log the realized length per tick.<br>• (5) Pair caps by common seeds and scenes.<br>• (6) Attach S6 latency per realized length.<br>• Label n = 0 by architecture (C-10).<br>**★:** arm design and caps.<br>**A:** A6 shape `____` plus the realized-length distribution per cap.<br>**P:** unchanged `____`.<br>**K:** KC-5 | $0; +4–5 h | KC-E (Edge's runtime switch) |
| **C-38 · A1 leakage rule by label; chunk-aware manifest** (L5-6) · S5 (S2 report) · G5 (A1) | **B:**<br>• A1 draws only `test`-labelled clips; `alpasim` clips are excluded; gold-644 overlap is reported per label (C-13).<br>• Select by chunk: PAI-AV stores each sensor in 3,146 chunks of ≤ 100 clips, so a scattered 644-clip, 4-camera subset touches up to ≈ 5.0 TB of chunks for ≈ 0.05 TB of clip bytes.<br>• Check the selection for bias.<br>**★:** audit rule, stratification, bias check.<br>**A:** the manifest states label and chunk counts.<br>**P:** none.<br>**K:** none | $0; saves download time and storage; +2–4 h | No PAI-AV download before S2. Licence acceptance is Huy's |
| **C-39 · S5's DAgger loop as a measured, manifest-driven pipeline** (L5-5) · S5 · G5, G7 | **B:**<br>• **Per-round manifest:** student and teacher hashes, AlpaSim SHA and lockfile, scene IDs, seeds, label stride, `n_sim_steps`, trainer commit.<br>• **Observations** come from AlpaSim's per-rollout `.asl` logs, which store the camera JPEGs and egomotion the driver received.<br>• **Teacher labels** are produced offline in replay, keyed by (session, timestamp), with no re-rendering.<br>• **Data hygiene:** dedup keys (scene, seed, tick); road-disjoint splits.<br>• **Measurement:** per-stage throughput with C-35's metrics, and a bitwise-invariance re-run of one round.<br>• **Clarify KC-5's "three ablation rounds"** against DAgger iterations (OPTED ran ≈ 113).<br>**★:** aggregation math, split, leakage audit (unchanged).<br>**A:** every round reproducible from its manifest.<br>**P:** share of round time per stage `____`.<br>**K:** KC-5 unchanged | 1–2 H100-h of micro-benchmarks ($2–5). +8–12 h, mostly agent; most of it applies the contract's existing manifest and bitwise rules, so the true increment is lower | P-005 (S5), KC-L, KC-L2 |
| **C-40 · Upstream contributions, filed after NVIDIA's next sync** (L6-8, L1-9, L7-2) · S1–S7 | **B:**<br>• S1's paired-layer issue (already planned).<br>• If C-52 runs: an AlpaSim PR adding XID, power, energy and clock fields to `dcgm-counters.csv`, with a J-per-rollout rule.<br>• Issues for any autoresume or retry defect found by C-53.<br>• The C-20 driver flag.<br>• A short note on single-node Slurm + Enroot (P-012).<br>**★:** none (Huy's outward actions).<br>**A:** "filed", not "merged".<br>**P:** none.<br>**K:** none | $0; +3–5 h (Huy 1–2) | Huy's outward action |
| **C-41 · TensorRT-11-ready C++** (L7-7) · S6 · G6 | **B:**<br>• Strongly typed networks with explicit ModelOpt Q/DQ; no builder precision flags.<br>• Custom ops as `IPluginV3` / `IPluginCreatorV3One`.<br>• CI on TRT 10.16 (Jetson) and TRT 11.x (x86). TRT 11.4.0 excludes JetPack, and a later 11.x restores it ([migration guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html)).<br>**★:** plugin and kernel design.<br>**A:** the same source builds on both; no speed claim.<br>**P:** none.<br>**K:** none (KC-6 unchanged) | $0; +0–2 h | P-002 for the device part |
| **C-42 · Production-grade C++ driver server on Thor** (L7-9) · S7 (built in S6) · G7 | **B:** a C++ gRPC `EgodriverService` with:<br>• nvJPEG decode into pre-allocated pinned and device buffers;<br>• TensorRT on dedicated streams with a captured fixed-shape step;<br>• CUDA-event per-stage timing plus a host monotonic clock for glass-to-action;<br>• bounded lock-free SPSC queues and the deterministic `inference_seed` contract;<br>• Prometheus histograms.<br>Tests: mock-engine unit tests, golden bitwise replay, compute-sanitizer (memcheck, racecheck, synccheck, initcheck), ASan/UBSan/TSan host builds, clang-tidy.<br>**★:** the latency model.<br>**A:** p50/p99/p99.9 per stage on Jetson AGX Thor, with device, JetPack and TRT named and labelled "real engine"; zero sanitizer findings; golden replay bitwise.<br>**P:** S6's existing slots `____`.<br>**K:** KC-6 and KC-7 unchanged | $0; +6–10 h, funded by C-25 | P-002, or KC-6's RTX PRO 6000 fallback |

**Why these are CORE.** Without the instrument fixes (C-26 to C-28, C-31 to C-34, C-36 to C-38), S2's reliability statistics have the wrong meaning, S4's ★ estimator row and its comfort arm are wrong, S5's A1 rule is empty, and the CoC curve confounds information with truncation shift. C-20 to C-25 cut cost or hours without touching an estimand. The evidence items (C-29, C-30, C-35, C-39 to C-42) produce JR2010863 evidence on D2, D3, D4, D5, X3 and Q4 from runs the plan already makes. They do it in NVIDIA's own measurement vocabulary, so the numbers can be compared directly with the AV Simulation team's own runs. That comparability is the strongest form of "checkable" evidence a solo project can offer.

---

## 4. Sixteen OPTIONAL changes wait on the KC-1 re-forecast

OPTIONAL items add evidence or savings, but they cost hours that P-003 has not yet confirmed exist. The table lists them in priority order, ranked first by role evidence per ★ hour and then by whether the item also improves validity or cuts cost. The top four explain the ranking: C-45 answers the posting's explicit ask for hyperparameter tuning and costs no money if it is carved out of S4's budget; C-43 repairs the mechanism attribution in S5(a); C-46 pairs a design doc with a real measurement as a stand-in for cluster scale; and C-52 answers the posting's explicit ask for monitoring and debugging tools. Every saving in this tier is conditional on S2's measurements, and the inputs behind it are illustrative, not predictions.

| ID · change (lane rows) | Stage-card change (B / ★ / A / P / K) | Δ hours | Δ $ (10-27) | Gate |
|---|---|---|---|---|
| **C-45 · Budgeted tuning, successive halving** (L2-5) · S4 | **B:** ≈ 9 configurations, η = 3 over 3 rungs, on tuning scenes from the 1,761 (never the 441).<br>Knobs: SDE `noise_level` (0.2 shipped); `n_generation`; `optm_lr` (1e-6 vs NVIDIA's 1e-4); `kl_beta` with `reference_reset_interval`; `allowed_outdated_steps` 0 vs 4; `inference_step` 4 or 5.<br>The rung metric is a non-gameable per-group statistic, reported as IQM with intervals.<br>`unbiased_advantage` only if Huy rules the generated-TOML route config-level (untested).<br>**★:** design, rung metric.<br>**A:** IQM with intervals per rung; final claims still come only from the gate on the 441.<br>**P:** largest rung-0 effect `____`; surviving configuration `____`.<br>**K:** none (KC-4 unchanged) | +6–8 | $0 if carved from S4's 225–450 H100-h; $193–259 if additive | P-005 if additive; KC-L2 |
| **C-43 · RoaD decomposition arm** (L5-2) · S5(a) | **B:** the student samples K candidates per tick and executes the one closest to the log, with recovery blending past a threshold and no teacher. This separates on-policy states from the logs → teacher target switch (C-11).<br>**★:** arm, distance rule, threshold.<br>**A:** a paired verdict per arm against the base student.<br>**P:** `____` per arm.<br>**K:** KC-5 unchanged | +4–6 | one collection + one sweep: $34–132 (collection unmeasured) | P-005 beyond S5's ≈ 6 sweeps |
| **C-46 · Capacity and service-balance model** (L3-11, L7-11) · S2 | **B:** a design doc from OPERATIONS.md's capacity formulas and C-29's per-service rates, N_drv·μ_drv ≈ N_rnd·μ_rnd. It predicts throughput for k renderer : m driver GPUs against NVIDIA's 4:4 and 7:1 topologies.<br>Validated by concurrency {1, 2, 4, 8} × NRE workers {1, 2, 4} on 20 scenes, plus one 2-GPU run, with outputs identical across concurrency.<br>**★:** the model.<br>**A:** measured curve against the model, with residuals explained.<br>**P:** saturation concurrency `____`; driver:renderer ratio `____`.<br>**K:** none | +5–7 (★ mostly) | 4–15 GPU-h: $4–23 | P-005 |
| **C-52 · DCGM fields, goodput waterfall, alerts** (L6-3) · S2 | **B:**<br>• Add XID, power, energy, SM clock, temperature and PCIe replay counters to AlpaSim's DCGM list.<br>• Goodput (useful ÷ paid GPU-s), J per rollout, retry rates.<br>• A dashboard joining `alpasim_gpu_workload_info{gpu}` to DCGM.<br>• Alerts: XID ≠ 0, queue starvation, p95 divergence.<br>• A per-sweep loss waterfall.<br>**★:** goodput definition, thresholds.<br>**A:** the waterfall sums to paid GPU-h within `____ (Huy)`.<br>**P:** goodput `____`; largest loss `____`.<br>**K:** none (instrumentation) | +8–11 (Huy 2–3) | $0. PROF fields need host-level DCGM on a VM (unverified) | none |
| **C-53 · Failure-injection drill** (L6-4) · S2 | **B:** requeue mid-sweep; SIGKILL a driver replica; SIGSTOP the renderer; a time-limit requeue. Measure detection and recovery time, lost and duplicated rollouts, and paired score equality against an uninterrupted reference.<br>**★:** equality tolerance; diagnosis of any anomaly.<br>**A:** a per-fault record of detection, recovery, loss and duplication.<br>**P:** lost/duplicated `____`; recovery time `____`.<br>**K:** loss → AlpaSim issue and no auto-requeue | +5–7 (Huy 2–3) | 0.25–0.5 node-h: $4–8 | Requeue legs need P-012 |
| **C-54 · S5 scaling and resiliency ladder** (L6-6) · S5 | **B:**<br>• The data-parallel modes the stage-2 trainer supports, at 1/2/4/8 GPUs, strong and weak scaling, ×3 after warm-up.<br>• nccl-tests busbw as the ceiling; MFU from Huy's FLOPs.<br>• An NVRx v0.7.0 hang and crash drill with Flight Recorder.<br>**★:** FLOPs per sample, efficiency definitions and floor, diagnosis.<br>**A:** scaling curves timed after warm-up with device sync, with CIs; the interconnect is named.<br>**P:** 8-GPU efficiency `____`; MFU `____`; detection latency `____`.<br>**K:** below `____ (Huy)` → train at the cheapest-per-step GPU count | +10–14 (Huy 4–6) | 16–24 H100-h, absorbed in S5's "pilot first" (2–3 node-h ≈ $64–108) | none |
| **C-44 · K ladder anchored on NVIDIA's deployed K = 5** (L5-3) · S5(b) | **B:**<br>• Run {10, 5, 4, 2} closed loop, plus FlashDrive's reuse-4 schedule vs uniform K = 4 at the same NFE. K = 8 and K = 1 run open loop only.<br>• One fixed noise file per scene and seed (PR #45's `action-noise.bin`).<br>• Claim only the ODE-error order as known. CPU known answer (1-D flow): std 0.3405 / 0.3692 / 0.43078 at K = 4 / 5 / 10.<br>**★:** prediction and tolerance.<br>**A:** + the CPU known answer passes; closed-loop order is reported, not assumed.<br>**P:** existing `____`.<br>**K:** unchanged | +3 | +2 sweeps: $68–265 | P-005 (KC-$) |
| **C-47 · Group-sequential paired verdict** (L4-5) · S1 tables, S3–S5 use | **B:**<br>• S1 tables at one-sided α = 0.05, K = 2 and 3: O'Brien–Fleming, Pocock, Lan–DeMets. Example: OBF at K = 3 has n_max 1.027× fixed, and expected n 0.837× fixed at the powered alternative.<br>• Looks come after pre-registered random road batches, with the intersection–union rule; futility → INSUFFICIENT; stage-wise-ordering bounds.<br>**★:** spending function, number of looks, futility rule.<br>**A:** boundary known answers reproduce; a Monte Carlo shows sequential false-GO ≤ α on clustered synthetic truth.<br>**P:** stopping look `____`.<br>**K:** unchanged | +3 S1 (or S2 day 1) and +4 S3–S5 | Conditional: −97 to −255 GPU-h (illustrative; not additive with C-21/C-22) | none |
| **C-49 · Covariate-driven replicate allocation for rare failures** (L4-9) · S4 | **B:** replicates ∝ predicted within-scene SD, predicted from independent data (S2 teacher replicates, the reference panel). ≥ 1 replicate per scene. The double-sampling estimator if allocation depends on outcomes.<br>**★:** allocation rule and estimator.<br>**A:** road-level bootstrap of the weighted estimate.<br>**P:** share of discordant pairs in the top decile `____`.<br>**K:** none | +2 | Conditional: −170 to −425 GPU-h (illustrative); a misprediction costs +14.6% variance, still unbiased | none |
| **C-48 · Anytime-valid e-process on S4 checkpoints** (L4-6) · S4 | **B:** an e-value per checkpoint on a fixed subset, so a futile arm can stop early. S4's acceptance verdict stays fixed-n. Never used for S3's one-shot verdict: the width costs 1.55–1.85× fixed-n.<br>**★:** e-process design, monitoring subset, futility threshold `____`.<br>**A:** a per-checkpoint e-value log.<br>**P:** first stop/certify `____`.<br>**K:** KC-4 unchanged | +3 | Monitoring unpriced until the S4 pilot | none |
| **C-55 · Batched AlpaSim driver for the PR #42 student architecture** (L5-7) · S3 | **B:** batch concurrent `drive` calls. Measure ticks/s, p50/p99 and VRAM against B ∈ {1, 2, 4, 8, 16}.<br>It runs today on Cosmos-Reason2-2B plus a random-init expert, labelled "timing only", and on 1.5 FP8 for correctness. Race check at Edge's release: consume NVIDIA's adapter if one ships.<br>**★:** latency model and prediction.<br>**A:** a batch curve within 16 GiB, labelled "timing only" until real student weights exist.<br>**P:** knee `____`.<br>**K:** none (KC-3 unchanged) | +6–10 | 4–8 GPU-h: $4–12 | P-005; KC-E; licence acceptance is Huy's |
| **C-50 · Reference-panel noise inputs and IRT strata** (L4-10, L4-8) · S1/S2 | **B:**<br>• Read the 8 shipped reference `results-summary.json` objects (Git LFS) for provisional base rates, ICC, design effect and σ_d, all labelled provisional.<br>• Use Drive-IRT (82ddd5a4) only for strata or Horvitz–Thompson inclusion probabilities, never as a deterministic subset.<br>**★:** interpretation; whether IRT strata enter the pre-registered strata.<br>**A:** values labelled provisional and kept internal until P-004.<br>**P:** discrimination vs discordance rank correlation `____`.<br>**K:** none | +5–6 | $0 | Huy's go-ahead for the download; KC-L |
| **C-56 · OOD-separation arm for the CoC curve** (L5-4 optional) · S5 | **B:** one stage-2-shaped expert SFT with CoC truncated at a random n, so every cap is in-distribution. The gap between the two curves estimates the truncation shift.<br>**★:** whether to run; arm design.<br>**A:** both curves reported with realized lengths per cap.<br>**P:** `____`.<br>**K:** none | +6 | Tens of H100-h ($20–150, inference) + sweeps (≈ $34–132 each) | P-005 if beyond the node window |
| **C-57 · Token-level on-policy distillation at visited states** (L5-9) · S5 | **B:** PR #42's `training/on_policy.py` applied to CoC that the student samples at AlpaSim-visited states, scored by teacher log-probs; a separate variable, run after C-24.<br>**★:** divergence choice, arm design.<br>**A:** measured by nano_alpamayo A2 and A3.<br>**P:** `____`.<br>**K:** none | +6–8 | $20–100 (inference) | KC-E (tokenizer match); KC-L2 |
| **C-51 · Control variates, measure first** (L4-4) · S2 | **B:** ρ(d, x) for pre-run covariates and ρ(d_T, d_200) for truncated horizons; adopt only above Huy's threshold `____`. Expected yield is low, because pairing already removes difficulty.<br>**★:** cross-fitted adjustment inside the road bootstrap; adoption threshold.<br>**A:** both ρ with road-cluster CIs.<br>**P:** ρ(d, x_difficulty) `____`; ρ(d_100, d_200) `____`.<br>**K:** none | +3 | $0 | none |
| **C-58 · Decode micro-benchmark** (L5-11) · S5 | **B:** PyAV `SeekVideoReader` vs PyNvVideoCodec (NVDEC) on one PAI-AV chunk; decoded frames per requested frame.<br>**★:** the prediction.<br>**A:** frames/s per CPU core and per GPU.<br>**P:** `____`.<br>**K:** none | +3–4 | ≈ 1 GPU-h: $1–2.40 | After S2; licence |

---

## 5. Six gated proposals carry the scale evidence, Slurm first

Each gated proposal needs Huy's decision and P-005, and several also need a G3 leg un-deferred in DECISIONS or a prior proposal adopted. In priority order, **P-012(a)** comes first because it is the cheapest way to move Q5 off None, at about 2–3 ★ hours. **P-014 and P-016** follow: both buy I2/X3 evidence, P-014 also carries a large conditional saving, and P-016 informs S4's reward-hacking audit. **P-015** carries D3's "improve" half and the Cosmos-based element of I2. **P-012(b)** is the best D1 stand-in but is ★-heavy. **P-013** adds a second node only after the single-node model exists, and **P-017** carries the highest feasibility risk. The texts below use DECISIONS.md's own nested format, so they can be pasted into `docs/DECISIONS.md` unchanged.

- **P-012 · *(proposed 2026-10-27; C-59; L1-7, L2-1, L6-1, L6-5 iii)* Run S2's sweeps and S4's training through NVIDIA's own Slurm paths on one rented node.** Two parts that can be decided separately.
  - **(a) S2.**
    - **Run method.** AlpaSim `wizard.run_method=SLURM_ENROOT` on one root VM with single-node Slurm 26.05.x, Enroot 4.2.1 and Pyxis 0.24.0. NVIDIA now owns Slurm's developer, SchedMD ([NVIDIA](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/)).
    - **Submission.** Sweeps go in as array jobs through AlpaSim's `submit.sh`: `--requeue`, `USR1@300`, autoresume.
    - **One method for the whole stage.** Every S2 sweep uses the same run method.
    - **Manifest additions:** Slurm and Enroot versions, job and array IDs, a `sacct` export, `resume.sh` and `reeval.sh`.
    - **What it unlocks:** the MPS arm of C-29 and the requeue legs of C-53 ([submit.sh](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/tools/run-on-slurm/submit.sh); [slurm_enroot.py](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/src/wizard/alpasim_wizard/deployment/slurm_enroot.py)).
  - **(b) S4.**
    - **Presets.** AlpaGym `deploy=slurm`, using only shipped presets: `local_disaggregated_2gpu` → `slurm_partial_node_1_2_1` (4 GPUs) → `slurm_full_node_1_3_4` (8 GPUs), with `perf.enabled`. AlpaGym's README warns that the presets are not expected to run out of the box ([topologies](https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf/topology)).
    - **★ design doc.** The balance N_gen·g ≈ N_train·t and Amdahl S(N) = 1/((1 − f) + f/N), fitted on held-in configurations and then predicting the held-out ones. It also predicts NVIDIA's 3- and 5-node presets, labelled unvalidated.
    - **Reliability check.** Force one time-limit requeue and check checkpoint continuity.
  - **★:** the balance and Amdahl model, the predictions, and the bottleneck diagnosis.
  - **Acceptance:** every S2 sweep has its `sacct` record and resume/reeval scripts in `evidence/`; the model's prediction error on the held-out single-node configurations is reported, with the averaging window and step count behind every mean.
  - **Predictions (Huy):**
    - steady-state episodes/h at 1_3_4 `____`;
    - share of rollout-worker time in `rollout/sim_step_rpc` `____`;
    - fitted serial fraction f `____`;
    - predicted speedup for `slurm_distributed_1_1_1` `____`;
    - GPU-h per GRPO step, 8-GPU ÷ 2-GPU `____`.
  - **Kill:**
    - Slurm + Enroot cannot start a 1-GPU smoke run within `____ (Huy)` h → docker-compose, with the failure recorded as a negative result.
    - The AlpaGym presets are not running after 8 agent-hours → local backend, plus an issue-ready failure note; label the model "not validated beyond 2–4 GPUs".
  - **Cost (10-27).**
    - **(a) VM premium** on S2's 136–340 GPU-h:
      - $0 if a Slurm-capable VM at ≤ $1.56/GPU exists at S2's start;
      - +$82–109 on Nebius at $1.80/GPU;
      - +$136–150 on the one vast VM at $2.00/GPU;
      - L6's 10-25 basis was +$184–255.
    - **(a) other costs:** bootstrap ≈ $15–21; a < $1 Enroot-in-Docker test first; MPS and run-method invariance $1–10.
    - **(b)** 36–96 GPU-h: $65–192 on an RTX PRO 6000 VM, or $115–230 on H100. About half overlaps the S4 pilot.
    - **Hours:** (a) Huy +2–3, agent +6–8; (b) Huy ≈ +12–17, agent ≈ +10–16.
  - **Gate:** P-005. Never mix run methods inside S2.
  - **Moves:** Q5 None → Partial; D1 gains a checkable prediction of NVIDIA's own presets; D4 gains measured requeue and autoresume.

- **P-013 · *(C-60; L1-8, L2-6, L6-7)* One multi-node measurement on 2 nodes with InfiniBand.**
  - **When:** only after P-012(b) or C-54 shows the single-node system is not bound elsewhere.
  - **Target:**
    - preferred: AlpaGym with a custom 2-node topology YAML (still config-level), testing P-012's model against NVIDIA's render-bottleneck claim;
    - otherwise: the S5 learner's step at 8 vs 16 GPUs.
  - **Method:** nccl-tests first (busbw), then NCCL RAS and Flight Recorder on an injected cross-node hang.
  - **★:** the design, the predictions and the diagnosis.
  - **Acceptance:** the P-012 model's prediction error at 2 nodes, and busbw reported against the 400 Gb/s per-GPU NIC.
  - **Predictions:** inter-node busbw `____`; scaling efficiency `____`.
  - **Kill:** busbw below `____ (Huy)` within 45 minutes, or spend reaching `____ (Huy)`.
  - **Cost:**
    - Nebius 2 × 8 H100 at $72/h × 4–6 h = **$288–432**, plus $10–30 of staging (quotas unverified);
    - up to $610 on RunPod, if its H200 $6.35 is per GPU-hour;
    - AlpaGym's smallest shipped multi-node preset is 3 nodes and 24 GPUs (≈ $216–324 for 2–3 h at Nebius list).
    - Excluded: Lambda 1-Click (2-week minimum) and vast.ai (no H100 cluster offers).
  - **Hours:** Huy +8–12, agent +6–10.
  - **Gate:** P-005 as a new budget line; KC-L.

- **P-014 · *(C-61; L7-4, L3-10 E2)* Promote `04` §7 E2: a GPU-batched `predict_batch` for the Alpamayo AlpaSim adapter.**
  - **Build.** Use a per-sample `torch.Generator` for top-p and flow noise, seeded from each session's `inference_seed`. Today no Alpamayo adapter overrides `predict_batch`, and A1.5 runs at `max_batch_size: 1` ([driver config](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/driver/alpamayo1_5.yaml)).
  - **Acceptance (P-009 iv ladder):**
    - E0a: the B = 1 path is bitwise unchanged;
    - E0b: per-layer comparison against FP32;
    - E0c: action error in metres, with injected shared noise;
    - E1: minADE6;
    - E2: a paired closed-loop verdict, batched vs unbatched on the same checkpoint. This is a **known-null pair** the gate must not flag.
  - **Timing:** decide before S2's first sweep.
  - **★:** the E0c and E1 tolerances, the E2 margin (equivalence or non-inferiority), and the predictions.
  - **Predictions:** speed-up at B = k `____`; E2 verdict `____`. The preserved 22-Sep text stands.
  - **Kill:** E2 not equivalent → stay at B = 1.
  - **Cost:** +1 equivalence sweep ($34–132). Then a conditional −263 to −964 GPU-h on S3–S5 (−$263 to −$1,504), only if Alpamayo's batch speed-up matches non-Alpamayo analogs (s = 2.26–2.84; inference). Not additive with C-20.
  - **Hours:** +10–14.
  - **Gate:** P-005; P-009 adopted.

- **P-015 · *(C-62; L3-6, L3-7, L3-8, L1-4 optional)* Un-defer a minimal G3·R2/R4 scoped to NVIDIA's Cosmos-based Harmonizer, in two steps.**
  - **(a) Open loop.**
    - Render the 4 policy cameras at lateral offsets d ∈ {0, 1, 2, 4} m, at 20 timestamps per scene.
    - Measure detector recall/IoU against projected ClipGT cuboids, normalised at d = 0, plus KID against the clip's recorded front-wide video.
    - Harmonizer OFF on all 441 scenes; ON on a 100-scene subset.
    - Measure the policy-sensitivity ratio (appearance change ÷ seed change) on 100 scenes.
  - **(b) Closed loop,** only if (a) shows a Harmonizer effect beyond the replicate floor at d ≥ 2 m.
    - One arm per policy with the Harmonizer flipped.
    - κ and paired difference per deviation bin, read beside S2's replicate κ.
    - Does the BF16-vs-FP8 verdict survive?
    - Rank agreement over 3 policies is reported as INSUFFICIENT.
  - **★:** metric semantics, κ and its bootstrap, the invariance rule, any cap, and the diagnosis.
  - **Acceptance:** fidelity-vs-deviation curves with road-clustered CIs, with detector and KID versions in the manifest; for (b), κ per bin beside the replicate floor and the verdict under both Harmonizer states.
  - **Predictions:**
    - recall at 2 m and 4 m vs 0 m `____`;
    - the Harmonizer's recall change at 4 m `____`;
    - appearance ÷ seed ratio `____`;
    - label-flip share `____`.
  - **Kill:**
    - detector recall at d = 0 below `____` → KID only;
    - κ below `____` in the < 1 m bin → report the renderer as the dominant noise before any S3 verdict.
  - **Cost:** (a) 6.7–19.6 GPU-h ($7–31); (b) 22.7–56.3 GPU-h ($23–88), up to +$227 if the flipped arm is ON and the cost scales with pixels.
  - **Hours:** +16–22.
  - **Gate:** DECISIONS un-deferral + P-005 + KC-L.
  - **Race note.** NVIDIA ran initial challenge runs OFF and plans final runs ON, so it holds both datasets. Re-check the challenge reports in November.

- **P-016 · *(C-63; L3-9, L2-9)* Un-defer a minimal G3·R3: replay vs CAT-K.**
  - **Design:**
    - one SHA (`main`@affc2eab or the post-10-31 release), with only `trafficsim` toggled;
    - the 441 scene IDs passed explicitly;
    - one evaluator (port `e2e`'s corridor scorer, or re-evaluate the logs);
    - A1.5 BF16: replay × 1, CAT-K × 3 (CAT-K is unseeded).
  - **Timing:** run it before S4's reward ablation. Replay followers cannot yield, so `collision_any` can punish legitimate braking.
  - **Conditional S4 arm.** Only if the flip rate beyond the floor is material by Huy's rule: a reactive-traffic S4 arm through `alpasim.wizard_args.extra_overrides` (feasibility unverified).
  - **Upstream.** A seeding PR only if NVIDIA's next release lacks seeding.
  - **★:** the agreement statistics, the invariance rule, and the diagnosis of every flip.
  - **Acceptance:** κ, flip rate per failure type and paired difference by deviation bin, each beside S2's replicate κ; CAT-K's own repeatability across its 3 repeats; rank claims INSUFFICIENT at n ≤ 3. The single replay arm borrows S2's replay repeatability, and the report says so.
  - **Predictions:**
    - at-fault change, replay → CAT-K `____`;
    - failure-label flip share `____`;
    - reactive vs replay rear-collision and stall rates `____`.
  - **Kill:** `FAILED_PRECONDITION` on more than `____`% of scenes → report coverage and stop.
  - **Cost:**
    - 45.4–112.7 GPU-h ($45–176) for one policy; $91–352 for two;
    - the S4 arm adds +112–225 H100-h ($200–540) only if triggered;
    - CAT-K's own GPU need is unknown.
  - **Hours:** +10–13 (+6–10 for the S4 arm).
  - **Gate:** DECISIONS un-deferral + P-005 + KC-L. The training data and licence of the CAT-K checkpoint are unknown: check before publishing.

- **P-017 · *(C-64; L7-6)* One C++ driver, two targets.**
  - **When:** after P-007 (S6a).
  - **Build.** Serve the student's AlpaSim sweeps from the S6 C++ runtime (TensorRT C++ engines or the Edge-LLM runtime, plus C-42's server) on the rented RTX PRO 6000, through `driver_source=external_static`. x86 SM120 is an Edge-LLM "Developer" platform ([support matrix](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/support-matrix.md)).
  - **Acceptance:** ladder E0a–E2 against the PyTorch driver, with injected reference x_0 (Edge-LLM's pi0.5 parity method). A cloud FP8 verdict never stands in for the Thor-engine verdict.
  - **★:** kernels, the latency model, and the verdicts.
  - **Prediction:** `____`.
  - **Kill:** no TensorRT path for the student on SM120 → drop. PR #45's fork targets DRIVE.
  - **Cost:** +$34–132.
  - **Hours:** +12–20.
  - **Gate:** P-007 adopted + P-005.

---

## 6. Each stage adopts the 2026 NVIDIA stack it can measure

The posting asks for current NVIDIA technology, so the rule applied here is simple. Adopt a 2026 component when it produces a measurement inside an existing stage. Decline it when it would race NVIDIA, re-open deferred scope, or produce no number.

| Stage | Adopted (version, date) | Declined (reason) |
|---|---|---|
| S2 | **AlpaSim:** `e2e_challenge` @ e121e591 with the post-#215 scorer; NRE-GA 26.04.<br>**NuRec Harmonizer** (CosmosPredict2 0.6B), pinned.<br>**Throughput:** Prometheus telemetry with NVIDIA's `optimize-slurm-topology` metric; alpamayo1.5 PR #36 expert CUDA graph.<br>**P-012:** Slurm 26.05.5 (10-13), Enroot 4.2.1, Pyxis 0.24.0, CUDA MPS.<br>**C-52:** DCGM 4.7.0 (10-21) and dcgm-exporter 4.8.4 | **Cosmos3-Nano-Sim-Auto renderer arm:** no AlpaSim adapter; would race NVIDIA's own report; ≈ 9.8 H100-h per 441-scene seed for rendering alone (inference).<br>**OmniDreams/FlashDreams:** single-view only in open AlpaSim.<br>**Kubernetes stack** (KAI v0.16, NVSentinel v1.29, GPU Operator 26.7): AlpaSim ships no Kubernetes path, and NRE runs under Slurm + enroot (#72) ([#72](https://github.com/NVlabs/alpasim/issues/72)).<br>**Lambda 1-Click:** 2-week minimum |
| S3 | **Edge**, captured at KC-E.<br>**PR #45 deployment levers:** K = 5, fixed noise.<br>**PR #42 student architecture** on Cosmos-Reason2-2B (NVIDIA Open Model License) | **PR #42 stage-1 pilot:** undertrained before its on-policy phase; would race the unreleased "AM 1.5 Nano" |
| S4 | **AlpaGym** 972d160 on cosmos-rl d2a2c57, as shipped.<br>**PR #44 measurement vocabulary.**<br>**Collective forensics:** PyTorch Flight Recorder; NCCL RAS (≥ 2.24).<br>**P-012:** AlpaGym Slurm presets | **NeMo-RL 0.7.0:** no flow-policy or AlpaSim path.<br>**RLinf / π_RL:** no AlpaSim connector.<br>**cosmos-rl 0.4.7 VLA trainer directly:** duplicates AlpaGym.<br>**GDPO:** NeMo-RL only.<br>**NVRx inside AlpaGym:** a code change ([NeMo-RL](https://github.com/NVIDIA-NeMo/RL/blob/main/README.md)) |
| S5 | **Learner:** PR #42 stage-2 trainer with DRaFT parity.<br>**Data:** AlpaSim `.asl` logs replayed for labels.<br>**C-54:** NVRx v0.7.0, nccl-tests 2.21.1, NCCL 2.32 RAS | **Cosmos/NeMo Curator as the loop:** raw-video curation, no measurement.<br>**TacitVLA:** no code released.<br>**MeanFlow:** from-scratch retraining |
| S6 | **Device:** Jetson AGX Thor, JetPack 7.2.1 (CUDA 13.2.2, TRT 10.16.2).<br>**TRT 11:** strong typing + `IPluginV3`.<br>**TensorRT-Edge-LLM 0.11.0,** consumed: SM120 developer wheels; pi0.5 parity procedure.<br>**ModelOpt 0.47.0:** QAD and FP8-ViT recipes as references | **Building FP8 or camera-batched ViT, or ViT reuse:** Edge-LLM 0.10.x ships them.<br>**An Alpamayo 1.5 exporter for Edge-LLM:** issue #130 and PR #260 cover it.<br>**Torch-TensorRT 2.14:** drift from the pinned torch |
| S7 | **AlpaSim** `external_static` + `planner_delay_us`.<br>**C++ server** checked with compute-sanitizer | **Holoscan SDK 5.0.0** (10-21): a 6-day-old ground-up rewrite with Thor known issues. Borrow its vocabulary only ([release](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v5.0.0)) |

Hardware follows the same rule. Every sweep runs on **RTX PRO 6000 Blackwell (sm_120)**, with the Workstation or Server edition recorded, and vast.ai reports these offers as PCIe-only (`bw_nvlink` 0). S5 uses **one 8×H100 SXM node with NVLink**, and S6–S7 use the **Jetson AGX Thor** under P-002. B200, B300 and GB200/GB300 appear only as priced options or in NVIDIA's own published numbers; no stage needs them.

---

## 7. Line by line, depth rises faster than marks

Most of the gain is depth inside existing marks, not new marks. Only two marks move: **Q4 goes from Partial to Full**, conditionally, and **Q5 goes from None to Partial**, only through P-012. Elsewhere the evidence becomes more checkable: D3's two broken premises are repaired and its "improve" half becomes real under P-015, X2 gains correct replicate semantics and a drift sentinel, and D1 gains a falsifiable prediction of NVIDIA's own presets. The hours and dollars in the table are those of the changes most specific to each line; changes shared across lines are counted once, in §8.

| Line | Plan v4 evidence today | Optimized evidence | Stage-card changes | Δ hours / $ (10-27) | Gate | Mark |
|---|---|---|---|---|---|---|
| **I1** | **S4:** GRPO on Alpamayo 1.5 in AlpaGym (expert only, one node, 2 arms × 3 seeds, hacking audit).<br>**S5:** a correction loop.<br>**Weakness:** the ★ estimator row is wrong, the comfort arm is not config, and the default trains nothing | **S4** on AlpaGym's real objective, with the KA suite and parity, one-variable arms, zero-scale monitors, a pinned train/eval protocol and NVIDIA's telemetry vocabulary.<br>**S5** learner = NVIDIA's stage-2 trainer, fed by a manifest-driven DAgger pipeline.<br>**OPTIONAL:** budgeted tuning.<br>**GATED:** preset ladder + scaling model; conditional reactive-traffic arm | S4: C-31–C-35; S5: C-24, C-39; optional: C-45 (S4); gated: P-012(b), P-016 | **CORE S4:** +4.5–20 h, ≤ $2.<br>**C-24:** −4 to −8 h.<br>**C-45:** +6–8 h, $0 if carved.<br>**P-012(b):** +22–33 h, $65–230 | P-005, KC-L, KC-L2 | Partial → Partial: AlpaGym is code-frozen, so the framework is operated, measured and predicted, never developed |
| **I2** | S2 reliability, validity bins and one throughput number on NuRec. Renderer/traffic agreement deferred. No Cosmos element. Harmonizer unpinned | **CORE:**<br>• renderer/policy/traffic noise decomposition;<br>• Harmonizer (Cosmos-based) costed and pinned;<br>• dose–response against NVIDIA's own quality score;<br>• stabilized metric + critical-path profile;<br>• expert graph behind a bitwise gate;<br>• AlpaGym telemetry.<br>**GATED:** Harmonizer fidelity + minimal R2; replay vs CAT-K; batching with a known-null verdict | S2: C-20, C-27–C-30; S4: C-35; gated: P-014, P-015, P-016 | **CORE (S2):** +28–42 h, $8–32 added, $8–54 saved in S2.<br>**P-015:** +16–22 h, $30–119.<br>**P-016:** +10–13 h, $45–176.<br>**P-014:** +10–14 h, +$34–132 | P-005; KC-L; DECISIONS un-deferral | Partial → Partial: the Cosmos world model is reached only through the Cosmos-based Harmonizer; Sim-Auto is declined to avoid racing NVIDIA |
| **D1** | AlpaGym on one node, as shipped | **CORE:** PR #44 telemetry; E1 reconciled against NVIDIA's PR #36 numbers; estimator audit.<br>**GATED:** ★ scaling model fitted on 2/4/8 GPUs, predicting NVIDIA's 3- and 5-node presets; one 2-node test | S4: C-31, C-35; gated: P-012(b) (S4), P-013 (S4 or S5) | **C-35:** +4–7 h.<br>**P-012(b):** +22–33 h, $65–230.<br>**P-013:** +14–22 h, $298–640 | P-005 (new line for P-013) | Partial → Partial |
| **D2** | S2 protocol and one throughput number. S5 run-hours unmeasured | **CORE:** stabilized metric, profile, invariance control, replica/bundling arms; expert graph; full manifest; DAgger pipeline over `.asl` logs; chunk-aware A1.<br>**OPTIONAL:** capacity/balance model.<br>**GATED:** Slurm arrays + MPS; batching | S2: C-20, C-26, C-29; S5: C-38, C-39; optional: C-46 (S2); gated: P-012(a), P-014 | **CORE:** +26–41 h; saves 48–154 GPU-h (near-certain).<br>**C-46:** +5–7 h, $4–23 | P-005 | Partial → Partial, the strongest Partial (single node) |
| **D3** | S2 ICC, W, noise floor, negative control, validity bins. S4 arms and hacking audit with new stall/oscillation scorers.<br>Two premises wrong (C-1, C-8); the "improve" half deferred | **CORE:** replicate semantics + CRN + noise decomposition; dose–response with NVIDIA's quality score; Harmonizer pinned on measured cost; at-fault reward arm; zero-scale monitors; stationary-lead stratum.<br>**GATED:** fidelity curve + R2; R3 | S2: C-27, C-28, C-30; S4: C-32, C-33; gated: P-015, P-016 | **CORE:** +4.5–17 h, ≈ $1–8 | KC-L to publish; P-015/P-016 as in §5 | Full → Full, with premises repaired. The "improve" half becomes real only with P-015 |
| **D4** | S2 Prometheus dashboards; NaN-retry logging | **CORE:** PR #44 metrics, Flight Recorder and NCCL RAS in S4; S2 profile + invariance; OOM-aware retry audit.<br>**OPTIONAL:** DCGM XID/energy + goodput waterfall + alerts; fault drill; NVRx ladder.<br>**GATED:** requeue/autoresume; cross-node RAS | S2: C-19, C-29; S4: C-35; optional: C-52, C-53 (S2), C-54 (S5); gated: P-012, P-013 | **CORE:** +14–22 h.<br>**OPTIONAL:** +23–32 h, $4–8 (+$64–108 absorbed) | P-005 | Partial → Partial (single node plus one 2-node pilot) |
| **D5** | S5 puts NVIDIA's student into a correction loop | **CORE:** stage-2 trainer with DRaFT parity and the `target_action` plug; DAgger pipeline; KC-E capture list; upstream issues as the partner channel.<br>**OPTIONAL:** batched student driver inside 16 GiB; token-level OPD.<br>**GATED:** one C++ driver for cloud and Thor | S3: C-23; S5: C-24, C-39; all stages: C-40; optional: C-55 (S3), C-57 (S5); gated: P-017 (S6) | **CORE:** +4–14 h net.<br>**C-55:** +6–10 h, $4–12.<br>**P-017:** +12–20 h, $34–132 | KC-E, KC-L2, P-005; P-007 for P-017 | Partial → Partial |
| **Q3** | S4 ★ GRPO math (misdescribed), 2 arms, ≥ 3 seeds | **CORE:** corrected estimator with KA-1…KA-8 and step-1 parity; one-variable arms; monitors.<br>**OPTIONAL:** successive-halving tuning on separate scenes; e-process monitoring; rare-failure allocation.<br>PPO with a critic stays unexercised: config cannot reach it | S4: C-6, C-7, C-31–C-33; optional (S4): C-45, C-48, C-49 | **CORE:** −0.5 to +11 h.<br>**OPTIONAL:** +11–13 h | P-005, KC-L2 | Full → Full |
| **Q4** | Python everywhere; C++ only in S6–S7 | **CORE:** a C++ `EgodriverService` on Thor with zero hot-path allocations, sanitizer-clean runs, golden bitwise replay and p99.9 histograms; TRT-11 strongly typed engines with `IPluginV3`; the Python gate library with known-answer tests.<br>**GATED:** the same server serves cloud student sweeps | S6: C-41; S7: C-42 (funded by C-25); gated: P-017 | +6–12 h, offset by C-25's −4 to −8 h | P-002, or the KC-6 fallback | Partial → **Full (conditional):** only if S6–S7 deliver C-42's acceptance. Reviewers judge "exceptional" |
| **Q5** | None: at most one 8×H100 node in S5 | **GATED only:**<br>• S2 sweeps as Slurm array jobs on NVIDIA's own `SLURM_ENROOT` path, with requeue, autoresume and MPS on one VM;<br>• S4 on AlpaGym's Slurm presets;<br>• one 2-node InfiniBand run.<br>Kubernetes stays absent by design | gated: P-012 (S2, S4), P-013; optional: C-53 (S2) | **P-012(a):** Huy +2–3, agent +6–8, $16–182.<br>**P-013:** $298–640 | P-005 | None → None (CORE) → **Partial** (P-012) |
| **X1** | None | Stand-ins only, labelled "solo reproduction on rented nodes": NVIDIA's production-shaped paths operated with measured numbers; upstream PRs and issues; PR #44 vocabulary | S4: C-35; all stages: C-40; gated: P-012 | as above | — | None → None |
| **X2** | The S1–S3 gate and verdicts; S5–S7 verdicts | **CORE:** correct replicate semantics with CRN; intersection–union power; teacher reuse with a drift sentinel; estimand decision; dose–response; novelty positioned against TacitVLA.<br>**OPTIONAL:** group-sequential looks.<br>**GATED:** known-null batched-vs-unbatched verdict | S1: C-36; S2: C-28, C-30; S3: C-10, C-21, C-22; optional: C-47; gated: P-014 | **CORE** saves $56–218 in S3 (near-certain) | KC-L | Full → Full |
| **X3** | S6 latency work; no large data pipeline | **CORE:** expert graph with measured sweep savings; rollout profile; DAgger pipeline; chunk-aware A1 (≈ 5 TB of chunks → ≈ 0.05 TB of clip bytes).<br>**OPTIONAL:** K ladder at NVIDIA's K = 5; decode micro-benchmark.<br>**GATED:** batching | S2: C-20, C-29; S5: C-38, C-39; optional (S5): C-44, C-58; gated: P-014 | see D2 | P-005 | Partial → Partial |

**Coverage, before and after.** "Before" is plan v4 as written, scored on all 13 lines; the earlier report scored 11. The carrying artifact is the public deliverable a reviewer would check.

| Line | Before | After CORE | + OPTIONAL | + GATED | Artifact that carries the evidence |
|---|---|---|---|---|---|
| I1 | Partial | Partial | Partial | Partial | S4 RL study with estimator audit; S5 correction report |
| I2 | Partial | Partial | Partial | Partial | S2 G1 v1 report (noise decomposition, Harmonizer, profile); P-015/P-016 reports |
| D1 | Partial | Partial | Partial | Partial | P-012(b) design doc, with predictions for NVIDIA's presets; P-013 run |
| D2 | Partial | Partial | Partial | Partial | S2 throughput record; S5 pipeline manifests |
| D3 | Full | Full | Full | Full | S2 reliability and validity report; S4 reward arms and monitors |
| D4 | Partial | Partial | Partial | Partial | Telemetry and goodput dashboards; fault-drill and NVRx records |
| D5 | Partial | Partial | Partial | Partial | S5 learner integration with parity; student driver |
| Q3 | Full | Full | Full | Full | S4 known-answer suite, arms, tuning study |
| Q4 | Partial | Full (cond.) | Full (cond.) | Full (cond.) | C++ driver server on Thor with sanitizer and replay logs |
| Q5 | None | None | None | Partial | Slurm `sacct` records, requeue logs, preset ladder |
| X1 | None | None | None | None | — |
| X2 | Full | Full | Full | Full | `physloop-gate` verdicts with replicate floor and sentinel |
| X3 | Partial | Partial | Partial | Partial | Graph savings record; pipeline throughput |
| **Total** | **3 / 8 / 2** | **4 / 7 / 2** | **4 / 7 / 2** | **4 / 8 / 1** | |

---

## 8. CORE pays for itself in GPU-hours, not in focused hours

Three rules govern the totals. First, every figure uses the 10-27 price basis. Second, savings sit in three buckets that are never summed together: **near-certain** savings (the expert graph on teacher-path sweeps behind its bitwise gate, S3 teacher reuse if its sentinel passes, and the hour savings in C-24 and C-25); **conditional** savings, which depend on a ★ decision or an S2 measurement (the estimand choice, the graph on Edge sweeps, the group-sequential design, the allocation rule and batching); and **counterfactual** savings, of which the only one is the KC-E fallback re-scope, which saves money only if Edge is not public at S3's start. Third, known overlaps are netted out: C-21, C-22 and C-47 all act on S3's sweeps, P-014 is not additive with C-20, and C-20 already excludes the S3 teacher sweeps that C-21 reuses. The arithmetic is in `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/calc_report/totals.py` and `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/calc_report/hours.py`.

| Tier | Changes | Δ hours: total (★ share; my allocation where a lane gave none) | Added spend | Near-certain savings | Conditional savings | Counterfactual |
|---|---|---|---|---|---|---|
| CORE corrections | C-1 – C-19 | +3–6 (★ 1–2) | $0 | — | — | — |
| CORE savings | C-20 – C-25 | −8.5 to +1.5 (★ 2.5–3) | $1–2 | −$102 to −$451 (102–289 GPU-h) | −$46 to −$287 (C-22; graph on Edge sweeps) | −$1,126 to −$1,510 and −10 to −20 h (C-23) |
| CORE instrument and evidence | C-26 – C-42 | +50.5–91 (★ 19.5–30.5) | $9–38 | — | — | — |
| **CORE total** | 23 changes + 19 corrections | **+45 to +99 (★ 23–36)** | **$10–39** | **net −$63 to −$441** | as above | as above |
| OPTIONAL | C-43 – C-58 | +82–105 (★ 33–42) | $189–825 (+$193–259 if C-45 is additive; C-54's $64–108 absorbed) | — | −$97 to −$398 (C-47); −$170 to −$663 (C-49); illustrative | — |
| GATED | C-59 – C-64 (P-012 – P-017) | +92–135 (★ 38–54) | **$523–1,611** (worst cases add up to +$227 for P-015, +$176 for a second P-016 policy, +$200–540 for the triggered S4 arm) | — | P-014: −$263 to −$1,504 (minus its $34–132 sweep) | — |

| Stage | Plan h | Plan cloud $ as written (10-25) | Re-priced (10-27) | CORE Δh (★) | CORE $: added / saved | OPTIONAL Δh / $ | GATED Δh / $ |
|---|---|---|---|---|---|---|---|
| S1 | 20 | $0 | $0 | +4–7 (★ 2–3) | $0 | +3 / $0 (C-47 tables) | — |
| S2 | 40 | $136–544 + storage | $136–530 | +28–42 (★ 9.5–14) | +$8–32 / −$8 to −$54 | +26–34 / $8–31 | +44–60 / $125–609 (P-012a, P-014, P-015, P-016) |
| S3 | 30 | $102–544 (+ Super $66–204) | $102–530 (+ $61–204) | +4.5 (★ 2) | −$56 to −$218 near-certain; −$23 to −$180 conditional; counterfactual −$1,126 to −$1,510 | +6–10 / $4–12 | — |
| S4 | 50 | $774–2,440 | $743–2,406 | +4.5–20 (★ 4.5–7.5) | +$0–2 / −$39 to −$179 | +11–13 / $0 (or $193–259) | +22–33 / $65–230 (P-012b) |
| S5 | 60 | $564–2,352 | **$970–2,524** | +6–17 (★ 4–6) | +$2–5 / −$23 to −$107 conditional | +32–41 / $177–781 | +14–22 / $298–640 (P-013) |
| S3–S5 | — | — | — | — | — | +4 / −$97 to −$398 conditional (C-47) | — |
| S6 | 60 | Thor $5,499 + $102–408 | Thor $5,499 + $102–398 | +0–2 (★ 0–1) | $0 | — | +12–20 / $34–132 (P-017) |
| S7 | 20 | ≈ $200 | ≈ $200 | −2 to +6 (★ 1–2) | $0 | — | — |
| S8 | 40 | Rig ≈ $5–8k | unchanged | — | — | — | — |
| **All stages (cloud = S2–S7)** | **320** | **$1,878–6,692** | **$2,253–6,792** | **+45 to +99 (★ 23–36)** | **net −$63 to −$441** | **+82–105 / $189–825** | **+92–135 / $523–1,611** |

**What the totals mean.** CORE saves money but costs time. It is cash-negative, because two bitwise-safe or sentinel-guarded reuses pay for its probes several times over, yet it is not hour-neutral: Huy carries roughly 23–36 extra ★ hours, about 7–11% of the 320-hour plan. Most of those hours buy validity rather than new scope. About +12 to +45 h of the total belong to corrections, savings and instrument fixes (C-1–C-28, C-31–C-34, C-36–C-38), net of the hours that C-24, C-25 and C-32 return, while the cheap role-evidence items (C-29, C-30, C-35, C-39–C-42) add +33 to +54 h; if KC-1 shows fewer hours than planned, those evidence items are the first to drop back to OPTIONAL. CORE's low end also assumes C-32's −6 h, which holds only if S4's ~50 h already included building a jerk scorer; if it did not, CORE adds +51 to +99 h instead of +45 to +99 h. Prices also raised the plan's floor: re-pricing moves its cloud low end from $1,878 to $2,253, almost entirely because no 8×H100 node can be rented below $32/h for S5's 24–48 h today. Adopting every tier takes the 10-27 cloud total to ≈ $3.0k–9.5k before conditional savings, so P-005 must be either the ≤ $6k tier with the KC-$ cap rule or the collaborator tier. If the hours have to come from somewhere, the plan's own risk rule already says to cut S8 first when effort runs over; S8 carries no JR2010863 line, which makes its optional 40 h the natural reserve for OPTIONAL and GATED items. That trade is Huy's to make under P-006, and this report does not recommend dropping S8.

---

## 9. Cluster leadership stays out of reach below collaborator scale

Even with every tier adopted, a finished PhysLoop cannot evidence the parts of JR2010863 that are about scale, industry and leadership. No rearrangement of a solo, ~320-hour, ≤ $6k plan changes that.

**Industry experience (X1).** There is none, and there will be none. The honest stand-in is a public record: NVIDIA's own production-shaped paths operated with measured numbers, plus upstream PRs and issues. It must always be labelled as a solo reproduction on rented nodes.

**Leading large-scale RL framework design (D1).** AlpaGym is code-frozen and must be used as shipped. PhysLoop can measure NVIDIA's framework and predict its behaviour under NVIDIA's own presets, but it never designs one. Even P-013's two nodes are not the multi-node scale the posting describes.

**Large GPU clusters (Q5).** The ceiling is one rented node plus one 2-node pilot, so the plan cannot show failure statistics at 100–1,000+ GPUs, fairshare, preemption or multi-tenancy, NVL72 rack-scale NVLink, or any Kubernetes at all. On that last point, AlpaSim ships no Kubernetes path and NRE runs under Slurm with enroot, so a Kubernetes deployment would be code, not evidence.

**Large-scale data pipelines (X3).** S5's loop runs tens to a few hundred DAgger iterations on one node. Nothing in the plan curates fleet-scale data.

**The intro's productizing at scale.** No shipped production code exists.

**Partnering with researchers (D5).** Only upstream issues and PRs can show this, and AlpaSim has merged no outside PR on `main` since 08-17. The two remaining channels are `e2e_challenge`-style merges and filed issues.

**The Cosmos world-model element of I2.** It is declined on purpose. Cosmos3-Nano-Sim-Auto has no AlpaSim adapter, and building one would race NVIDIA's own technical report and its "multi-GPU optimization underway".

**Smaller gaps.** PPO with a critic is unreachable through AlpaGym's config, so Q3's PPO is exercised only as PPO-clip with group advantages. DRIVE AGX Thor numbers, DriveOS and QNX stay out of scope, because Jetson results never transfer 1:1. And "safe" or real-world validation claims stay forbidden by `02` §6.

---

## 10. Every unverified input stays labelled

These items are marked unverified, inferred or low-confidence in the notes, and they stay that way here. Each names what it affects and what resolves it. The illustrative ρ, σ_d and mini-control inputs behind every conditional saving belong to the same class: they are derivation inputs, not predictions.

| Item | Status in the notes | Affects | Resolves it |
|---|---|---|---|
| Edge: repo, licence, architecture, runtime-CoC switch, camera count, teacher, what "10 Hz" assumes. "Built on Cosmos-Reason2-2B" | Not public on 10-27; the backbone is an inference | S3, S5, S6; C-23, C-37, C-55 | KC-E capture list, 10-31 |
| TacitVLA authors' NVIDIA role; TacitVLA and OPTED code | Inference from author overlap; code "not stated" / "coming soon" | C-10, C-12 | Project pages |
| Qwen-Drive's AlpaSim RL gain (0.27 → 0.37) | Unverified snippet | S4's novelty wording | Read §3.3 of arXiv 2609.00111 |
| Driver share f = 0.73–0.875; the L3 vs L7 topology inferences | Inference | C-20, C-29, P-014 | C-29 profile |
| PR #36 latency (−15.5% for 10B, −31.5% for 2B) | One paired sample, "preliminary" | C-20 magnitude | C-29 and the bitwise test |
| Expert graph with `force_determinism` and the worker thread; FP8 fake-quant capture | Unverified / untested | C-20 | Bitwise test before sweeps |
| Batching speed-up s ≈ 2.0–2.6 | Inference from non-Alpamayo analogs | P-014 | E2 ladder |
| Harmonizer 212 ms per frame; the $249–1,362 exposure (10-27 prices) | Unit unconfirmed; inference | C-27 | 5-scene pilot |
| NRE determinism; meaning of `--no-enable-nrend`; NRE newer than 26.04 | Unknown | C-26, C-28 | Render-determinism probe; NuRec docs |
| `quality_score` scale and method; 26.01 coverage; front-wide MP4 "usually" present | Unknown | C-30, P-015 | Inspect at S2 download |
| A1.5 accepts sub-320×576 renders | Unverified | C-14 / E4 | Open-loop screen |
| Road IDs and cluster count for the 441 | Not published | Every cluster-adjusted figure; C-36, C-47 | Derive in S2 |
| Reference `results-summary.json` holds per-rollout records | Low (inference from file size) | C-50 | Read one LFS object |
| AlpaGym's "0.42 baseline" | Units undefined | P-012(b) | Measure; optional issue |
| Zero-scale terms reach trainer logs; new scorer metrics reach `episode.metrics.aggregated` | Unverified | C-33, C-8 | Code read and first run |
| `unbiased_advantage` via the generated TOML; staleness at pin d2a2c57 | Untested / unverified | C-45, C-9 | Dry run; read at pin |
| CAT-K via `extra_overrides`; CAT-K GPU cost; `catk_v120` data and licence | Unverified / unknown | P-016 | Config read; NVlabs/catk licence |
| AlpaGym Slurm presets on one rented VM; AlpaGym on sm_120 | README: not out of the box; untested | P-012 | Bounded bring-up (8-h kill) |
| Enroot inside a vast Docker rental | Likely fails (inference) | P-012 cost | A < $1 test |
| DCGM PROF fields under Enroot; NVRx `ft_launcher` needing Slurm; NCCL version in torch 2.8.0 | Unverified | C-52, C-54, C-35 | Host `dcgmi dmon`; NVRx docs; lockfile |
| RunPod's $6.35 unit, H100 cluster price and 8-GPU pod; Nebius 8-GPU RTX PRO 6000 preset and quotas; Lambda VM/root access; undated price pages | Unverified | P-012, P-013, S5 cost | Console quotes (Huy's sign-in) |
| The $16.00/h vast VM offer | Synthesis re-analysis of one offer | P-012 | Re-sample at S2 start |
| PR #42 stage-1 ≈ 6,293 H100-h; its match to PR #44's internal run | Inference; lower bound | C-23 | — |
| PR #42 stage 2 passes `target_action` | Unverified | C-24 | Pilot's first check |
| `.asl` logs hold every camera at driver resolution; concurrent `drive` calls per replica | Unverified | C-39, C-55, P-014 | Inspect one S2 rollout; runtime config |
| DriveOS 7.2.5 vs 7.2.6 for TRT 11 | Conflicting docs | Claims hygiene only | DRIVE docs |
| ModelOpt `qad.py` targets R1 only; "AM 1.5 Nano" / "AM 1 Nano" | Inference from class name; names only | C-18, C-23 | Read the example; watch HF |
| Sim-Auto's multi-view fps definition; sm_120 kernels; ≈ 9.8 H100-h per seed | Inference; sm_120 is a community claim | The declined renderer arm | Cosmos technical report |
| Author association of `e2e` PRs #111, #161, #169 | Unverified | C-3 wording | GitHub API |

---

## Conclusion

The binding constraint on PhysLoop's JR2010863 case is not a shortage of ideas, since the seven lanes proposed 77 rows; it is the instrument. Plan v4 was written against a ledger that misread four upstream facts (seeding, AlpaGym's objective and defaults, `alpamayo_aligned`, #224) and missed the tooling NVIDIA's own simulation team uses: Slurm run methods, the stabilized throughput metric, PR #44's vocabulary and PR #36's graph. Once the instrument is fixed, PhysLoop gains a kind of evidence a solo project rarely has. Its central claims become **predictions about NVIDIA's own systems, stated in NVIDIA's own units**, which a reviewer on the AV Simulation team can check against internal runs: a replicate noise floor decomposed by source, a throughput profile in `seconds_per_rollout`, a scaling model with written predictions for AlpaGym's 3- and 5-node presets, and training telemetry with GPU-hours from the mean. That is a better answer to the scale lines than renting scale, and, paired with honest None and Partial marks, it is the strongest signal of research judgment the plan can send.

The decision path follows directly. Adopt the corrections and CORE now, as one decision. Treat P-012(a) as the single gated decision that changes a mark, for about two ★ hours. After that, let KC-1's measured weekly hours, not ambition, decide what else enters the plan.

---

## Tóm tắt

- **Kết luận chính.** Bằng chứng mạnh nhất cho JR2010863 đến trước hết từ việc sửa "instrument" của plan v4, không phải từ việc thêm việc mới:
  - cần 19 corrections trong live docs (một số correction trải trên nhiều vị trí tài liệu);
  - 4 trong số đó làm hỏng claim chính của một stage: S2 "seed test–retest" thực ra là unseeded replicates; S4 ★ estimator mô tả sai objective; comfort/jerk arm không phải config-only; A1 rule của `alpamayo_aligned` loại bỏ toàn bộ clip.
- **AlpaGym estimator thật.** Đó là PPO clip trên geometric-mean (GSPO-style) chain ratio, cộng với closed-form Gaussian KL (mặc định tắt), không phải k3. Default `n_generation: 1` cho advantage bằng 0, tức default không train gì. NVIDIA tự train ở 5 Hz, 22 steps, G = 6, lr 1e-4, trong khi gate đánh giá ở 10 Hz, 200 steps.
- **CORE: 23 changes (C-20 – C-42) + 19 corrections.**
  - Chi phí thêm khoảng $10–39 theo giá vast.ai 10-27.
  - Tiết kiệm near-certain khoảng $102–451 GPU spend, nhờ expert CUDA graph (alpamayo1.5 PR #36, bitwise 160/160) và teacher reuse trong S3.
  - Thêm khoảng +45 đến +99 h tổng; trong đó ★ khoảng 23–36 h theo phân bổ của tôi, tức 7–11% plan ~320 h.
- **OPTIONAL: 16 changes (C-43 – C-58).** Chỉ adopt nếu KC-1 re-forecast cho phép. Thứ tự ưu tiên:
  - C-45: HPO bằng successive halving, Q3;
  - C-43: RoaD control, sửa mechanism attribution của S5(a);
  - C-46: capacity model;
  - C-52: DCGM, goodput, alerts.
- **GATED: P-012 – P-017,** tổng khoảng $523–1,611.
  - P-012(a), chạy S2 qua `SLURM_ENROOT` của AlpaSim với array, requeue, autoresume và MPS, là cách rẻ nhất (★ chỉ +2–3 h) để đưa Q5 từ None lên Partial.
  - Thứ tự sau đó: P-014 ≈ P-016 → P-015 → P-012(b) → P-013 → P-017.
- **Coverage 13 dòng:** trước 3 Full / 8 Partial / 2 None → sau CORE 4 / 7 / 2 → sau P-012 4 / 8 / 1.
  - Q4 chỉ Full nếu S6–S7 hoàn thành C++ `EgodriverService` (sanitizer-clean, golden bitwise replay, p99.9).
  - X1 vẫn None; mọi dòng "large-scale" vẫn Partial.
- **Upstream 10-25 → 10-27 gần như đứng yên:** AlpaSim không commit từ 10-20, Edge vẫn chưa public. Phần lớn "delta" là thứ ledger 10-25 bỏ sót: #224 gắn disclosure với "next release"; merge trên `e2e_challenge` (#167, 08-31); Slurm tooling; TacitVLA; recipes PR #44; PR #36.
- **Giá 10-27:**
  - RTX PRO 6000 WS $1.00/$1.56; H100 $1.79/$2.40 (vast.ai).
  - 8×H100 node thuê được ≥ 24 h chỉ còn khoảng $32/h, nên S5 node tăng lên $766–1,728 (trước đây $360–1,536).
  - L6 bị HTTP 403 còn L1 lấy được data. Bảng của L1 tái tạo đúng từ JSON đã lưu.
- **Savings chia 3 nhóm, không cộng chung:**
  - near-certain;
  - conditional: estimand ★, ρ đo ở S2, batching (P-014);
  - counterfactual: KC-E fallback tránh khoảng 629 H100-h ($1.1k–1.5k), nhưng chỉ khi Edge chưa public lúc S3 bắt đầu.
- **Lane disagreements** được giải quyết bằng nguồn primary hoặc code: external merges (L1 thắng L3/L6), seeding trên `e2e` (L4 thắng L3), TacitVLA (L1/L5 thắng L2), OPTED (L5 thắng L1).
- **Không race NVIDIA.** Đã từ chối:
  - Cosmos3-Nano-Sim-Auto renderer arm và OmniDreams arm;
  - tự build FP8 hoặc camera-batched ViT (TensorRT-Edge-LLM 0.10.x đã ship);
  - Alpamayo 1.5 exporter cho Edge-LLM;
  - PR #42 stage-1 pilot.
  Mọi upstream PR đều chờ sync post-10-31.
- **Cách ghi nhận:**
  - corrections + CORE gom thành một D-entry (D-015);
  - `CLAUDE.md` và `AGENTS.md` sửa cùng lúc;
  - `04` §7 chỉ thêm premise note;
  - OPTIONAL là menu cho KC-1;
  - GATED là P-012…P-017 trong DECISIONS.
- **Mọi prediction, tolerance và margin vẫn là `____` của Huy;** ★ cores không đổi chủ. Các input "illustrative" (ρ, σ_d, mini-control 20%) chỉ là derivation input.
- **Plan vẫn không thể chứng minh:**
  - industry infra (X1);
  - lead framework design at scale (D1);
  - cluster lớn hơn 2 node, và Kubernetes (Q5);
  - fleet-scale data (X3);
  - partner với researchers (D5);
  - Cosmos world-model simulation (cố ý từ chối vì race);
  - PPO có critic.

---

## Appendix A. Every lane row and where it went

**L1**
- L1-1 → C-23
- L1-2 → C-10, C-37
- L1-3 → C-26, C-30
- L1-4 → C-26, C-27, P-015
- L1-5 → C-33
- L1-6 → C-29
- L1-7 → P-012
- L1-8 → P-013
- L1-9 → C-26, C-40
- L1-10 → C-35
- L1-11 → C-12
- L1-12 → C-26
- L1-13 → §2 (no change)
- L1-14 → C-23
- L1-15 → C-17, §8
- L1-16 → C-2, C-3
- L1-17 → §2, §6 (declined)

**L2**
- L2-1 → C-35 (instruments), P-012 (sweep and model)
- L2-2 → C-31, C-6, C-7
- L2-3 → C-32, C-8
- L2-4 → C-33
- L2-5 → C-45
- L2-6 → P-013
- L2-7 → C-34
- L2-8 → declined (Appendix B)
- L2-9 → P-016

**L3**
- L3-1 → C-26
- L3-2 → C-27
- L3-3 → C-28
- L3-4 → C-30
- L3-5 → C-29
- L3-6 → P-015
- L3-7 → P-015
- L3-8 → P-015
- L3-9 → P-016
- L3-10 → C-29 (bundling), P-014 (E2), C-14 (E4 premise)
- L3-11 → C-46

**L4**
- L4-1 → C-28, C-1, C-26
- L4-2 → C-22
- L4-3 → C-21
- L4-4 → C-51
- L4-5 → C-47
- L4-6 → C-48
- L4-7 → C-36
- L4-8 → C-50
- L4-9 → C-49
- L4-10 → C-50

**L5**
- L5-1 → C-24
- L5-2 → C-43, C-11
- L5-3 → C-44
- L5-4 → C-37, C-56
- L5-5 → C-39, C-35
- L5-6 → C-38, C-13
- L5-7 → C-55
- L5-8 → C-23
- L5-9 → C-57
- L5-10 → deferred (Appendix B)
- L5-11 → C-58

**L6**
- L6-1 → P-012
- L6-2 → C-29 (the MPS arm through P-012)
- L6-3 → C-52
- L6-4 → C-53
- L6-5 → C-35 (i–ii), P-012 (iii)
- L6-6 → C-54
- L6-7 → P-013
- L6-8 → C-40

**L7**
- L7-1 → C-29
- L7-2 → C-20, C-26
- L7-3 → C-29
- L7-4 → P-014
- L7-5 → declined (Appendix B)
- L7-6 → P-017
- L7-7 → C-41
- L7-8 → C-25, C-16
- L7-9 → C-42
- L7-10 → C-35
- L7-11 → C-46

**GATED IDs:** C-59 = P-012; C-60 = P-013; C-61 = P-014; C-62 = P-015; C-63 = P-016; C-64 = P-017.

## Appendix B. Rows declined or deferred

| Row | Why not now | What would reopen it |
|---|---|---|
| L2-8 · Lagrangian collision penalty arm | Under group-std normalization the multiplier bites only in mixed groups. Checkpoint-boundary restarts reset optimizer state. It adds a third arm ($200–540 at 10-27 H100 prices, +8–12 h) for a mechanism C-32 and C-33 already probe | A gated proposal after S4's pilot shows collision-rate targets matter more than the arm comparison |
| L5-10 · Few-step distillation of the flow expert | A training change, worth paying for only if C-44's ladder shows a cliff above the K that S6's latency allows. Edge may already be few-step (race check) | C-44 shows that cliff, and Edge is not few-step |
| L7-5 · Static KV cache + CUDA-graphed decode in the PyTorch driver | Its marginal value collapses if P-014 batches decode. It changes numerics, so it needs the full ladder | C-29 shows decode as the top driver stage after C-20, and P-014 is rejected |
| L3-10 E4 as a speed lever; L1-17 and L3 candidate (vi) renderer swaps | E4 stays the plan's preserved optional experiment, with C-14's premise note. Sim-Auto and OmniDreams arms race NVIDIA or re-open deferred R2 | NVIDIA ships a multi-view world-model renderer for AlpaSim, and the policy panel reaches ≥ 10 independent policies |
