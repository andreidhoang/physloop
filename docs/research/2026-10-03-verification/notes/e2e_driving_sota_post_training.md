# End-to-end driving models and their post-training (imitation, RL, distillation, world models), 2025 to 3 Oct 2026, plus verification of PhysLoop learning-algorithm and model citations

## KQ1. Do the learning-algorithm and model citations in the PhysLoop docs hold up? (claim ledger)

### Takeaway
Every arXiv ID the docs cite for learning algorithms and models points to the paper the docs claim, and most of the attributed numbers check out. Five framings are wrong, and each one changes the plan:
- **OPTED.** RL trains a privileged *teacher*. The camera student then gets on-policy *supervision*. RL is never applied to the student.
- **nanoAlpamayo RL acceptance row.** The figures (2.12 to 1.92 m; 3.1 to 4.5; 0.62 to 0.85) come from NVIDIA's *internal* Chain-of-Causation (CoC) dataset, scored with a large-reasoning-model (LRM) critic. They are not on the public validation set.
- **Public CoC labels.** These cover only about 1,740 out-of-distribution (OOD) clips, roughly 10 h.
- **No source found.** Nothing supports the "n=12" group size or the "1.9x" RLinf multiple.
- **π*0.6 counter-position.** The paper never mentions simulation, so the "simulators miss long-tail contact failures" rationale has no support there.

### Cited Findings
Conventions:
- Status values are VERIFIED, CORRECTED, NOT FOUND and UNVERIFIABLE.
- Every source was accessed on **2026-10-03**.
- Quotes are exact. Quotes made of table cells are separated with " / ".
- "Preprint" means an arXiv paper with no venue stated.

**Table A: `docs/03_MASTERY_PLAN.md`**

| # | Claim as written (section) | Status | Exact quote or table cell | Correct value if different | Primary source (date) | Accessed |
|---|---|---|---|---|---|---|
| A1 | §1 P1: Alpamayo predicts "(accel, curvature) × 64 waypoints at 10 Hz" | VERIFIED | "each trajectory consists of 64 waypoints with 2 quantized values per waypoint"; "future waypoint sampled at 10 Hz" | — | [Alpamayo-R1 v2 HTML, §3/§5.1](https://arxiv.org/html/2511.00088v2) (v2 7 Jan 2026) | 2026-10-03 |
| A2 | §1 P1: open-loop L2 "ρ = −0.36 vs closed-loop score, Bosch 2026" | VERIFIED (with caveat) | "Traditional open-loop metrics (ADE/FDE) remain uncorrelated with closed-loop driving quality ( ρ=-0.36 , p=0.43 )" | The correlation is L2 vs Bench2Drive Driving Score, small n, and **not significant (p=0.43)**. In the same paper NAVSIM PDMS vs DS has ρ=0.90 (n=8). Authors are at Bosch Corporate Research, Shanghai. Preprint. | [arXiv 2605.00066 HTML](https://arxiv.org/html/2605.00066) (30 Apr 2026) | 2026-10-03 |
| A3 | §1 P2: "180 tokens/image at 320×576 → 2,880 tokens for 4 cams × 4 frames" | VERIFIED by derivation for Qwen3-VL-based Alpamayo; not true for AR1's formula | Alpamayo2-Super config: "patch_size = 16", "spatial_merge_size = 2", "max_pixels = 196608"; AR1 paper: patch features "then 2× bilinearly downsampled" to W/28×H/28 | With patch 16 and merge 2, a 320×576 image (184,320 px, inside the 163,840–196,608 bounds) gives (320/32)×(576/32) = 10×18 = **180** tokens. AR1's Qwen2.5-VL-style tokenizer uses 28-px tokens and so gives roughly 220–231 tokens. The Alpamayo 1.5 config file seen here does not expose its vision patch size. The 4 cams × 4 frames layout is not checked here (Alpamayo-stack lane). | [HF config, nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super/blob/main/config.json); [AR1 v2 §3.2.1](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A4 | §1 P2: "flow-matching expert reading the VLM KV cache" | VERIFIED | "the action-expert takes as input both the KV-cache from the sequence" | Also from AR1: "we apply a stop-gradient to the KV-cache produced by the VLM", so expert gradients never reach the VLM | [AR1 v2 §3.2.2/§5.1](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A5 | §1 P3: "privileged-teacher RL (OPTED)" | CORRECTED | "a privileged teacher is trained using RL on vectorized inputs (HD-map and bounding boxes)" | RL trains only the render-free privileged **teacher**. The camera **student** is post-trained by on-policy teacher supervision in AlpaSim, i.e. closed-loop distillation. It was applied to TransFuser and VaVAM, not to a VLA. Driving score improves ×1.6 and ×9.5, with about 1000× fewer simulator interactions than direct RL. Preprint (Da Col, Igl, Karkus, Chitta, Ivanovic, Pavone, Schindler, Sakaridis). | [arXiv 2609.20756](https://arxiv.org/abs/2609.20756) (17 Sep 2026) | 2026-10-03 |
| A6 | §1 P3: flow matching "(conditional OT path, Euler sampler, SDE variant with log-probs)" | VERIFIED (concept) | Flow-GRPO: "an ODE-to-SDE conversion … enabling statistical sampling for RL exploration" | — | [Flow-GRPO arXiv 2505.05470](https://arxiv.org/abs/2505.05470) (8 May 2025); [Lipman et al. 2412.06264](https://arxiv.org/abs/2412.06264) | 2026-10-03 |
| A7 | §1 P3 / §4 Stage 3: "GRPO (DeepSeekMath 2402.03300)" | VERIFIED | "we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO)" | — | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) (5 Feb 2024) | 2026-10-03 |
| A8 | §1 P4: "PhysicalAI-AV (1,700 h)" | VERIFIED | "a total of 1700 hours of driving recorded from planned data-collection drives in 25 countries" | 306,152 clips of 20 s each. Gated under the "NVIDIA Autonomous Vehicle Dataset License Agreement", which runs for 12 months and bars redistributing the data or derivatives | [HF dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (last modified 12 Sep 2026) | 2026-10-03 |
| A9 | §1 P4 and §4 Stage 5: "auto-label CoC with Alpamayo 2 Super" | VERIFIED | "generate structured CoC auto-labels" | — | [NVIDIA blog, Alpamayo 2 Super](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) (4 Aug 2026) | 2026-10-03 |
| A10 | §1 P4 and §4 Stage 5: "Instant NuRec" | VERIFIED (exists) | "a fully simulatable 3D Gaussian Splatting (3DGS) world in a single forward pass" | — | [arXiv 2607.14203](https://arxiv.org/abs/2607.14203) (15 Jul 2026, NVIDIA) | 2026-10-03 |
| A11 | §1 P4: "targeted synthetic variants (Cosmos Transfer)" | UNVERIFIABLE (here) | — | Not checked in this lane (data and simulation lane) | — | 2026-10-03 |
| A12 | §2: backbone "initialized from a pretrained VLM, as NVIDIA does"; 0.5B–2B | VERIFIED | "The 0.5B model uses a DINOv2 … vision encoder combined with the Qwen2.5-0.5B … language model" | Main AR1 uses the Cosmos-Reason VLM. The 0.5B DINOv2 + Qwen2.5-0.5B design comes from the **scaling ablation**. Table 10 only says the 0.5B "uses a smaller backbone". | [AR1 v2 §6.5.1/§6.4](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A13 | §2: "on a PhysicalAI-AV subset with public CoC labels" | CORRECTED | "human-verified reasoning labels for a curated subset of Out-of-Distribution (OOD) driving scenarios"; "Train / 1450 / 1728 / Released" | Public CoC labels exist only for **1,450 train clips (1,728 labels) and 290 val clips (349 labels)**, with the test split held out for the OOD Reasoning Challenge 2026. The card says the labels are "intended to support Vision-Language-Action (VLA) model evaluation". AR1's CoC SFT set is "700K video segments", which is internal. Training-scale CoC labels have to be **auto-labeled** (e.g. with Alpamayo 2 Super). | [HF dataset card, "Reasoning Labels"](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); [AR1 v2 §6.1](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A14 | §2 table: "minADE6 on 644 PAI samples: 0.913 m" (0.5B) | VERIFIED | "Alpamayo-R1-0.5B / 0.913"; "Open-loop results are evaluated on 644 examples from the PhysicalAI-AV evaluation set" | Table 10 first appears in **v2 (7 Jan 2026)**. The v1 text has no 0.913 value. Only the 10B weights are released. | [AR1 v2 Table 10](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A15 | §2 table: "RL post-training, ADE on CoC set: 2.12 → 1.92 m (−9.4%)" | VERIFIED number, framing CORRECTED | "a 9.4% reduction in most-likely mode ADE (2.12m → 1.92m)"; "Evaluations are conducted on the full CoC dataset introduced in Sec. 6.2" | The metric is **most-likely-mode ADE** (top of 6 rollouts), not minADE6. The data is the **internal** CoC dataset, not the public val set. RL is open-loop GRPO on tokens with rewards r_reason + r_consistency. No seeds or error bars are reported. | [AR1 v2 Table 9, §6.3](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A16 | §2 table: "Reasoning quality / consistency 3.1 → 4.5 / 0.62 → 0.85" | VERIFIED number, framing CORRECTED | "SFT / 2.12m / 3.1 / 0.62 / 6.9"; "SFT + RL (r_reason + r_consistency) / 1.92m / 4.5 / 0.85 / 6.2" (table cells) | The reasoning score is a 0–5 rubric graded by an LRM critic (DeepSeek-R1 / Cosmos-Reason are named). Consistency is a rule-based binary match between meta-actions and the reasoning text. Both are measured on the internal CoC set. A safety-reward variant reaches close-encounter 3.7% at ADE 1.94 m. | [AR1 v2 Table 9, §5.3.2](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A17 | §2 table: "Closed-loop (920 NuRec scenarios): close-encounter 9.0%, off-road 19%, AlpaSim 0.35" | VERIFIED | "9.0 ± 1.0 / 19.0 ± 0.0 / 0.35 ± 0.01"; "Closed-loop metrics are at-fault." | — | [AR1 v2 Table 10](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A18 | §2: "RL through the flow expert … (Flow-SDE, as AlpaGym does)" | VERIFIED | "int_method: sde" / "noise_level: 0.2" / "inference_step: 5" | AlpaGym's first commit is dated 2026-06-16, with the message "Introducing AlpaGym, an E2E closed-loop RL pipeline for AV policies". Apache-2.0. | [AlpaGym policy config](https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml); [commits](https://github.com/NVlabs/alpagym/commits/main) | 2026-10-03 |
| A19 | §2: acceptance = "reproducing NVIDIA's published small-model numbers on the public validation set" | CORRECTED | (see A15–A16) | Only the Table 10 rows (0.913 m; 9.0 / 19.0 / 0.35) are on public data. The Table 9 RL rows are on internal data and an internal LRM critic, so they cannot be reproduced as published targets. | [AR1 v2](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A20 | §4 Stage 2: "128 trajectory tokens (64 × {accel, curvature})" | VERIFIED | "resulting in 128 discrete tokens per trajectory" | — | [AR1 v2 §5.1](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| A21 | §4 Stage 3: "Flow-SDE log-probs" | VERIFIED | π_RL: "formulating a two-layer MDP that employs ODE-to-SDE conversion for efficient RL exploration" | — | [arXiv 2510.25889](https://arxiv.org/abs/2510.25889) | 2026-10-03 |
| A22 | §4 Stage 4: "Distillation; OPTED" | VERIFIED (exists), scope note | "OPTED matches closed-loop performance with approximately three orders of magnitude fewer simulator interactions than direct RL" | OPTED distils a render-free RL teacher into a camera policy. It is **not** a recipe for distilling a large reasoning VLA into a small one. | [arXiv 2609.20756](https://arxiv.org/abs/2609.20756) | 2026-10-03 |
| A23 | §4 Stage 4: "roofline of memory-bound expert (OI 54 vs ridge ≈ 281)" | UNVERIFIABLE (here) | VLA-Perf exists: "How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf" | Belongs to the deployment lane. The number was not re-checked here. | [arXiv 2602.18397](https://arxiv.org/abs/2602.18397) (20 Feb 2026) | 2026-10-03 |
| A24 | §4 Stage 4: "FlashDrive as competitor on the expert" | VERIFIED | "adaptive step caching that concentrates compute where it matters"; "717ms to 151ms (4.7x)" | FlashDrive targets the flow-matching denoising stage among four bottlenecks, applied to Alpamayo 1.5-10B with W4A8. Preprint. | [arXiv 2608.12932](https://arxiv.org/abs/2608.12932) (13 Aug 2026) | 2026-10-03 |
| A25 | §4 Stage 4: "≤ 16 GiB and ≤ 0.1 s per call" (challenge envelope) | see KQ8 | — | Challenge rules are checked in KQ8 | — | 2026-10-03 |
| A26 | §7: "Alpamayo-R1 (arXiv 2511.00088)" | VERIFIED | "Alpamayo-R1 is also referred to as Alpamayo 1." | v1 30 Oct 2025; v2 7 Jan 2026. NVIDIA (Yan Wang + 41 authors). Preprint. | [arXiv 2511.00088](https://arxiv.org/abs/2511.00088) | 2026-10-03 |
| A27 | §7: "Flow Matching Guide and Code (arXiv 2412.06264)" | VERIFIED | Title "Flow Matching Guide and Code"; Lipman, Havasi, Holderrieth, … | — | [arXiv 2412.06264](https://arxiv.org/abs/2412.06264) (9 Dec 2024) | 2026-10-03 |
| A28 | §7: "OPTED (arXiv 2609.20756)" | VERIFIED (exists) | Title "OPTED: On-Policy Fine-Tuning for End-to-End Driving using a Render-Free Teacher" | See A5 | [arXiv 2609.20756](https://arxiv.org/abs/2609.20756) | 2026-10-03 |
| A29 | §7: "RoaD (arXiv 2512.01993)" | VERIFIED | "improves driving score by 41% and reduces collisions by 54%" (AlpaSim) | Title "RoaD: Rollouts as Demonstrations for Closed-Loop Supervised Fine-Tuning of Autonomous Driving Policies". NVIDIA (Garcia-Cobo, Igl, Karkus, Zhang, Watson, Chen, Ivanovic, Pavone). Comment field reads "Preprint". | [arXiv 2512.01993](https://arxiv.org/abs/2512.01993) (1 Dec 2025) | 2026-10-03 |
| A30 | §7: nuPlan misalignment (2306.07962) | VERIFIED | Title "Parting with Misconceptions about Learning-based Vehicle Motion Planning" | CoRL 2023; Dauner, Hallgarten, Geiger, Chitta | [arXiv 2306.07962](https://arxiv.org/abs/2306.07962) | 2026-10-03 |
| A31 | §7: pseudo-simulation (2506.04218) | VERIFIED | Title "Pseudo-Simulation for Autonomous Driving"; comment "CoRL 2025" | — | [arXiv 2506.04218](https://arxiv.org/abs/2506.04218) (4 Jun 2025) | 2026-10-03 |
| A32 | §7: Waymo approval guidelines (2505.09880) | VERIFIED | Title "Determining Absence of Unreasonable Risk: Approval Guidelines for an Automated Driving System Deployment" | Favaro et al. (Waymo) | [arXiv 2505.09880](https://arxiv.org/abs/2505.09880) (15 May 2025) | 2026-10-03 |
| A33 | §7: OmniDreams (2606.03159) | VERIFIED (exists) | Title "NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation" | — | [arXiv 2606.03159](https://arxiv.org/abs/2606.03159) (2 Jun 2026) | 2026-10-03 |

**Table B: `docs/04_CORE_PLAN_TECHNICAL.md` ("State of the art": RL frameworks and flow-expert algorithms; "Learner math"; E5)**

| # | Claim as written | Status | Exact quote or table cell | Correct value if different | Primary source (date) | Accessed |
|---|---|---|---|---|---|---|
| B1 | "RLinf v0.3 (hybrid pipelining 1.9–2.4×; …)" | CORRECTED | "achieves a 1.61×-1.88× training speedup on ManiSkill"; "The hybrid execution mode for embodied RL achieves up to 2.434× throughput" | **1.61–1.88×** comes from the hybrid fine-grained pipeline in RLinf-VLA (RSS 2026). **Up to 2.434×** is the README and v0.1 (17 Dec 2025) claim against "existing frameworks". Neither figure is a v0.3 result, and no 1.9× figure was found. The headline multiple in v0.3 (15 Jul 2026) is DreamZero SFT, "nearly 4×". | [RLinf-VLA arXiv 2510.06710](https://arxiv.org/abs/2510.06710); [RLinf README](https://github.com/RLinf/RLinf); [v0.3 release](https://github.com/RLinf/RLinf/releases/tag/v0.3) | 2026-10-03 |
| B2 | RLinf: "Flow-SDE/Flow-Noise" | VERIFIED | README (2025/10): "RLinf supports reinforcement learning fine-tuning for π₀ and π₀.₅!", linked to π_RL | π_RL defines Flow-Noise and Flow-SDE and is the RLinf team's paper | [RLinf README](https://github.com/RLinf/RLinf); [arXiv 2510.25889](https://arxiv.org/abs/2510.25889) | 2026-10-03 |
| B3 | RLinf: "DSRL" | VERIFIED | "Extended DSRL (Diffusion Steering via Reinforcement Learning) to the Pi0.5 model" | DSRL for π0 arrived in v0.2 (26 Mar 2026) and was extended to π0.5 in v0.3 | [v0.3 release](https://github.com/RLinf/RLinf/releases/tag/v0.3) | 2026-10-03 |
| B4 | RLinf: "RECAP" | VERIFIED | "Added RECAP (offline advantage-based policy optimization) training pipeline support" | v0.3 also adds PR #1015, "add Pi0.6 RECAP pipeline". This is an open re-implementation of PI's method. | [v0.3 release](https://github.com/RLinf/RLinf/releases/tag/v0.3) | 2026-10-03 |
| B5 | RLinf: "GR00T N1.7" | VERIFIED | "Added GR00T-N1.6 / N1.7 model RL fine-tuning support" | — | [v0.3 release](https://github.com/RLinf/RLinf/releases/tag/v0.3) | 2026-10-03 |
| B6 | RLinf: "Isaac Lab" | VERIFIED | "Added IsaacLab π0.5 PPO fine-tuning support" | IsaacLab has been supported since v0.1 | [v0.3 release](https://github.com/RLinf/RLinf/releases/tag/v0.3) | 2026-10-03 |
| B7 | "Cosmos-RL (fully async, NCCL/Redis sync; maintenance mode)" | VERIFIED | "no longer under active development and will receive only limited maintenance updates"; "Fully asynchronous (replicas specialization)"; "controller over Redis streams" | Superseded by Cosmos 3. Commits are still landing (latest 2026-09-24). | [Cosmos-RL README](https://github.com/nvidia-cosmos/cosmos-rl); [docs/payload_transport/nccl.md](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/docs/payload_transport/nccl.md) | 2026-10-03 |
| B8 | "RL-VLA³ (fully async, +85%, no community)" | VERIFIED (+85%, async); UNVERIFIABLE ("no community") | "throughput improvements of up to 85.2% over synchronous baselines" | COLM 2026. Scaling validated "from 8 to 256 GPUs". "No community" is an opinion with no source. | [arXiv 2602.05765](https://arxiv.org/abs/2602.05765) (5 Feb 2026; v3 4 Sep 2026) | 2026-10-03 |
| B9 | "verl rollout-correction math (TIS, MIS, Geo-RS)" | VERIFIED | "decoupled_geo_rs() / Geo-RS / … / Geometric RS (seq_mean_k1)"; "Seq-MIS (Sequence-Level Masked IS)" | Docs page last updated 09/23/2026 | [verl docs](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html); [source md](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md) | 2026-10-03 |
| B10 | Architecture: "rollout correction with verl bounds 0.5/5.0" | VERIFIED (as an option) | "`\"0.5_5.0\"`: IcePop, zero weights outside `[lower, upper]`"; "Default: `2.0`" | 0.5/5.0 is verl's documented **IcePop** two-sided bound, not the default. The default TIS upper clamp is 2.0, and the Seq-MIS example uses "0.5_2.0". | [verl source md](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md) | 2026-10-03 |
| B11 | "AReaL (`max_head_offpolicyness`, 2.77×)" | VERIFIED | "by setting `max_head_offpolicyness=0`" (synchronous mode); "up to 2.77× training speedup compared to synchronous systems with the same number of GPUs" | Measured on math and code reasoning LLM RL, not VLAs. AReaL 2.0 released 2026-07-01. | [arXiv 2505.24298](https://arxiv.org/abs/2505.24298); [AReaL README](https://github.com/inclusionAI/AReaL) | 2026-10-03 |
| B12 | "π_RL Flow-SDE (π0.5 ManiSkill 40.1→90.9%)" | VERIFIED (version-dependent) | v3 Table 4 text: "π0.5 improves from 40.1% to 90.9%"; Flow-SDE row cell "90.9" | The **v2 abstract (27 Nov 2025) says "40.1% to 90.8%"**. The v1 abstract (29 Oct 2025) says "40.0% to 84.8%". Measured on 4,352 pick-and-place variations with 320 parallel envs. Preprint. | [arXiv 2510.25889v3 HTML](https://arxiv.org/html/2510.25889v3); [v2 abs](https://arxiv.org/abs/2510.25889v2); [v1 abs](https://arxiv.org/abs/2510.25889v1) | 2026-10-03 |
| B13 | "ReinFlow" | VERIFIED | "injects learnable noise into a flow policy's deterministic path" | arXiv 2505.22094 (28 May 2025). Journal ref: NeurIPS 2025. Tonghe Zhang, Chao Yu et al. | [arXiv 2505.22094](https://arxiv.org/abs/2505.22094) | 2026-10-03 |
| B14 | "DSRL" (arXiv 2506.15799) | VERIFIED | "adapting the BC policy by running RL over its latent-noise space" | Wagenmaker et al. (Levine). The paper says it "requires only black-box access to the BC policy". | [arXiv 2506.15799](https://arxiv.org/abs/2506.15799) (18 Jun 2025) | 2026-10-03 |
| B15 | "DiffusionNFT" | VERIFIED | "optimizes diffusion models directly on the forward process via flow matching" | arXiv 2509.16117 (19 Sep 2025), **ICLR 2026 Oral**. Zheng, …, Ermon, Zhu, Ming-Yu Liu. Likelihood-free and "up to 25× more efficient than FlowGRPO", but on image generation. No control or driving results in the abstract. | [arXiv 2509.16117](https://arxiv.org/abs/2509.16117) | 2026-10-03 |
| B16 | "RECAP (closed)" | VERIFIED (method published); note | "RL with Experience and Corrections via Advantage-conditioned Policies ( Recap )" | The method is public on arXiv and has an open re-implementation in RLinf v0.3 (B4). Only PI's model is closed. | [arXiv 2511.14759](https://arxiv.org/abs/2511.14759) (18 Nov 2025) | 2026-10-03 |
| B17 | "π*0.6: task throughput 2×, failures halved, Physical Intelligence Nov 2025" | VERIFIED (scope-limited) | "Throughput more than doubles on the diverse laundry folding and espresso tasks"; "the rate of failure reduces by about a factor of two" | This holds for the **hardest tasks** (diverse laundry, espresso), not uniformly | [arXiv 2511.14759 HTML §results](https://arxiv.org/html/2511.14759) | 2026-10-03 |
| B18 | "Counter-position (PI): all RL for π0.6 ran on real robots because simulators miss long-tail contact failures" | Real-robot RL: VERIFIED. Stated reason: NOT FOUND | "improve through real-world deployments via reinforcement learning (RL)" | The paper text contains **zero** occurrences of "simul". The rationale may come from an interview, but no primary source was found. (The PI blog returned HTTP 403.) | [arXiv 2511.14759](https://arxiv.org/html/2511.14759) | 2026-10-03 |
| B19 | Learner math: Flow-SDE as a Gaussian chain, so "per-step ratios r_t = π_θ/π_old exist" | VERIFIED (concept); AlpaGym nuance | Flow-GRPO ODE-to-SDE (A6). AlpaGym: "The trajectory probability ratio is clipped" | AlpaGym clips a **trajectory-level** ratio. Per-step ratios are mathematically available but are not what AlpaGym's config describes. | [AlpaGym default.yaml](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml) | 2026-10-03 |
| B20 | Learner math: "GRPO's clipped objective … group-normalized advantages over n=12 rollouts" | CORRECTED (n=12 NOT FOUND) | AlpaGym: "grpo_ratio_clip_low: 0.2"; "n_generation: 6" (CLRL test run); "batch_size * n_generation = 32" (public run, n_generation: 2); AR1 Eq. 10: "A_i = r_i − r̄" | AlpaGym presets use group sizes **1 (default), 2, 6**, a [0.8, 1.2] clip, and "kl_beta: 0.0". **AR1's own GRPO is different**: an unclipped softmax weight exp(βA_i)/Σexp(βA_j) on a mean-centred (not std-normalised) advantage, plus a KL term. AR1 does not state its group size K. No source uses n=12. | [AlpaGym configs](https://github.com/NVlabs/alpagym/tree/main/packages); [AR1 v2 Eq. 10](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| B21 | Learner math: staleness k, versions v < v_k − k dropped, the rest corrected with sequence-level TIS | VERIFIED (building blocks exist) | AlpaGym: "allowed_outdated_steps: 4" / "on_policy: false"; verl: "Seq-TIS (Sequence-Level Truncated IS)" | — | [AlpaGym default.yaml](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml); [verl](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md) | 2026-10-03 |
| B22 | Tensor contract: "AlpaGym preset: progress 1.0, collision −10, off-road −5, distance-to-GT −0.01" | VERIFIED | "metric_name: collision_any / scale: -10.0"; "metric_name: offroad / scale: -5.0"; "kind: distance_to_gt / scale: -0.01" | This is the `progress_safety` preset. The repo **default** reward is distance_to_gt (−0.01) only. | [progress_safety.yaml](https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/reward/progress_safety.yaml) | 2026-10-03 |
| B23 | Tensor contract: "10 Euler steps δt=0.1" | VERIFIED for AR1, CORRECTED for AlpaGym | AR1: "By default, we use δt=0.1 during inference"; AlpaGym: "inference_step: 5" | The AlpaGym Alpamayo-1.5 RL config samples 5 SDE steps with noise 0.2 | [AR1 v2 §5.1](https://arxiv.org/html/2511.00088v2); [AlpaGym policy config](https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml) | 2026-10-03 |
| B24 | E5 / prediction 5: Flow-SDE gives "close-encounter −15% relative in 200 GPU-h" | UNVERIFIABLE (a prediction, not a citation) | — | Context: RoaD reports −54% collisions in AlpaSim using closed-loop SFT, and OPTED reports that direct RL needs about 1000× more sim interactions than OPTED. Both suggest direct RL in AlpaSim is sample-hungry. | [RoaD](https://arxiv.org/abs/2512.01993); [OPTED](https://arxiv.org/abs/2609.20756) | 2026-10-03 |
| B25 | Models line: "Alpamayo 2 Super (32B Cosmos 3 Reasoner + 2.3B expert, …, AlpaSim 1.50, minADE6 0.911 m)" (outside assigned sections; cross-check only) | AlpaSim and minADE VERIFIED; expert size UNVERIFIABLE | "AlpaSim Score: 1.50 ± 0.13 (closed-loop on 913 scenes)"; "minADE_6: 0.911 m"; "2-billion-parameter diffusion-based Action Expert" | The blog says a 34B total, made of a "32-billion-parameter NVIDIA Cosmos 3 Super Reasoner" plus a **2B** expert. The HF API reports 35,813,934,322 BF16 parameters in total. The expert config shows a qwen3_vl_text stack with hidden size 1536 and 64 layers. The exact expert size (2.3B vs 2B) is not settled here (Alpamayo-stack lane). | [NVIDIA blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) (4 Aug 2026) | 2026-10-03 |

**Table C: `docs/02_PHYSLOOP_PLAN_v2.md` (§2 G5 and supporting §0 lines)**

| # | Claim as written | Status | Exact quote | Correct value if different | Primary source (date) | Accessed |
|---|---|---|---|---|---|---|
| C1 | G5: "Alpamayo 2 Super / 1.5 as teachers" and §0 "teacher distilled into compact runtime models on DRIVE AGX Thor" | VERIFIED | "As a teacher model, it can be distilled into compact models that run on NVIDIA DRIVE AGX Thor" | — | [NVIDIA blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) (4 Aug 2026) | 2026-10-03 |
| C2 | G5: "OPTED/RoaD ideas" for "closed-loop SFT/DAgger with teacher labels" | VERIFIED (relevant) | OPTED: "This teacher then provides supervision to the pre-trained student during closed-loop post-training"; RoaD: "leveraging the policy's own closed-loop rollouts as additional training data" | Both are closed-loop supervised methods tested in AlpaSim, but on small end-to-end policies (TransFuser, VaVAM, and RoaD's policies), not on 10B–34B reasoning VLAs | [OPTED](https://arxiv.org/abs/2609.20756); [RoaD](https://arxiv.org/abs/2512.01993) | 2026-10-03 |
| C3 | G5: "Distill to a ≤ 3B student fitting 16 GiB and ≤ 0.1 s" | see KQ8 (challenge envelope) and KQ4 | — | — | — | 2026-10-03 |
| C4 | §0: "a distillation script is 'planned', no public procedure" | see KQ4 | — | — | — | 2026-10-03 |
| C5 | §0: "AlpaGym (NVlabs, first commit 2026-06-16) … SDE log-probs … off-policy staleness up to 4" | VERIFIED | First commit "2026-06-16T16:04:52Z"; "int_method: sde"; "allowed_outdated_steps: 4" | The repo was created 2026-05-19, but the first commit is 2026-06-16. The `sde_type ∈ {sde, cps}` detail was not checked. | [AlpaGym](https://github.com/NVlabs/alpagym) | 2026-10-03 |

**Table D: `nano_alpamayo/README.md`**

| # | Claim as written | Status | Exact quote | Correct value if different | Primary source | Accessed |
|---|---|---|---|---|---|---|
| D1 | "reproducing NVIDIA's published 0.5B numbers on the public validation set … minADE6 ≤ 1.0 m on the 644-sample val" | VERIFIED (the 0.913 m target exists on public val) | "Alpamayo-R1-0.5B / 0.913" | See A14 | [AR1 v2 Table 10](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| D2 | "≥ 5% ADE gain from RL over 3 seeds" | CORRECTED (basis) | "a 9.4% reduction in most-likely mode ADE" | NVIDIA's −9.4% is open-loop, on internal CoC data, most-likely-mode ADE, single run. It is not a public-val minADE6 figure. | [AR1 v2 Table 9](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| D3 | "closed-loop within 20% relative on the gate" | VERIFIED (reference values exist) | "9.0 ± 1.0 / 19.0 ± 0.0 / 0.35 ± 0.01" | Reference values are on AlpaSim's 920-scenario public set. PhysLoop's gate uses a different scene set (441 val), so "within 20%" needs a re-baselined reference. | [AR1 v2 Table 10](https://arxiv.org/html/2511.00088v2) | 2026-10-03 |
| D4 | "GRPO through the expert (Flow-SDE) with the expert-drift monitor" | VERIFIED (precedent) | "int_method: sde" (AlpaGym) | — | [AlpaGym](https://github.com/NVlabs/alpagym) | 2026-10-03 |

**Table E: citations named in the assignment that the docs use implicitly**

| # | Claim | Status | Exact quote | Correct value if different | Primary source (date) | Accessed |
|---|---|---|---|---|---|---|
| E1 | Flow-GRPO | VERIFIED | "the first method to integrate online policy gradient reinforcement learning (RL) into flow matching models" | arXiv 2505.05470 (8 May 2025). Image generation: GenEval 63% to 95% | [arXiv 2505.05470](https://arxiv.org/abs/2505.05470) | 2026-10-03 |
| E2 | RAD: 3DGS-based RL for end-to-end driving, Horizon Robotics 2025 | VERIFIED | "a collision rate that is 3× lower" (vs IL methods) | arXiv 2502.13144 (18 Feb 2025). HUST + Horizon Robotics. **NeurIPS 2025** per the repo. Code released 28 Sep 2025. The repo now also lists "RAD-2" (a generator–discriminator framework). | [arXiv 2502.13144](https://arxiv.org/abs/2502.13144); [hustvl/RAD](https://github.com/hustvl/RAD) | 2026-10-03 |

Additional verified facts the ledger relies on:
- AR1 RL is **open-loop** and runs on a customised Cosmos-RL with vLLM rollouts. Its trajectory reward is "λ_L2‖x_pred − x_expert‖² + λ_coll·I[collision] + λ_jerk·J" — [AR1 v2 §5.3](https://arxiv.org/html/2511.00088v2)
- AR1 trains on "80,000 hours of driving data collected from multiple ego-vehicles operating in more than 2,500 cities in 25 countries". That data is internal; PhysicalAI-AV is a 1,700-h public subset-like release — [AR1 v2 §6.1](https://arxiv.org/html/2511.00088v2); [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- AR1 0.5B data-scaling ablation, evaluated on internal D_overall: "The 100k model exhibits clear overfitting (1.111m without early stopping; 1.016m with early stopping)". 500k segments reach 0.880 m and 2M reach 0.874 m — [AR1 v2 §6.5.2](https://arxiv.org/html/2511.00088v2)
- AR1's AlpaSim protocol excludes events where "the ego deviates more than 4 m from the original recorded trajectory", and the AlpaSim score is the average km driven between events — [AR1 v2 §6.1](https://arxiv.org/html/2511.00088v2)

### Inferences
- The nanoAlpamayo acceptance table mixes two kinds of target:
  - **Public, reproducible:** the Table 10 rows.
  - **Not reproducible as published:** the Table 9 rows, which use internal data and an internal LRM critic.

  Re-base the RL row on a metric the project can compute itself: public-val minADE6 or most-likely-mode ADE, plus a CoC-consistency score using an *open* judge. Paired seeds are also needed, because NVIDIA reports no seeds for Table 9.
- The "public CoC labels" assumption breaks Stage 3 as written. 2,077 human labels are enough for evaluation but not for CoC SFT. The realistic path is to auto-label 100–300 h with Alpamayo 2 Super, which NVIDIA advertises for exactly this purpose. That makes the project depend on a 34B model at labelling time, which is an extra compute line.
- AR1's own GRPO variant (softmax-weighted, unclipped, mean-centred) differs from AlpaGym's clipped GRPO. "GRPO" in the docs therefore names two different estimators, and E5 must say which one it uses.
- Two RLinf numbers, two π_RL versions and a framework's lifetime claims were merged into single figures. Pin version numbers in future citations.

### Gaps
- Alpamayo 1.5's vision-token count (180/image) and the expert size of 2.3B vs 2B were not verified here. They belong to the Alpamayo-stack lane and need the model config and card.
- The architecture and training data of the Table 10 "Alpamayo-R1-0.5B" row are not specified in the paper beyond "a smaller backbone".
- AR1 does not state its RL group size K. The rollouts-per-prompt actually used by NVIDIA's closed-loop runs (beyond the AlpaGym presets) is unknown.
- The PI blog (pi.website) returned 403. The "simulators miss long-tail contact failures" rationale could not be traced to any primary source.

## KQ2. What does the 2025–2026 driving VLA, planner and world-model landscape look like, and where does Alpamayo-R1 sit?

### Takeaway
Only partly answered in this pass, using sources opened directly. Alpamayo-R1 sets the open reference design for a reasoning VLA. It has a Cosmos-Reason VLM, Chain-of-Causation text, 128 discrete control tokens, and a flow-matching expert that reads the VLM KV-cache. NVIDIA benchmarks it on PhysicalAI-AV and AlpaSim, not on NAVSIM or Bench2Drive. The survey of other VLAs and planners and the leaderboard snapshot were delegated and did not return before hand-off.

### Cited Findings
- AR1 public benchmark (Table 10) reports only "minADE 6 @6.4s" on PhysicalAI-AV plus AlpaSim close-encounter rate, off-road rate and score. 10B: 0.849 / 4.0 / 16.0 / 0.72. 0.5B: 0.913 / 9.0 / 19.0 / 0.35 — [AR1 v2](https://arxiv.org/html/2511.00088v2) (7 Jan 2026, preprint)
- AR1 related work lists the reasoning-VLA peer set: EMMA, ReCogDrive, ReasonPlan, Drive-R1, AdaThinkDrive, AutoDrive-R², and CoT-Drive, which "distills reasoning into lightweight models" — [AR1 v2 §2](https://arxiv.org/html/2511.00088v2)
- Alpamayo 2 Super (4 Aug 2026): 34B reasoning VLA, minADE6 0.911 m, AlpaSim 1.50 ± 0.13 on 913 scenes, LingoQA 79.2, OpenMDW-1.1 licence — [NVIDIA blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/)
- Cross-benchmark ranking agreement:
  - NAVSIM PDMS vs Bench2Drive DS: "Spearman rank correlation: ρ=0.90 ( p=0.002 )" (n=8).
  - Under the CARLA v2 protocol, Hydra-NeXt has the best DS (65.89) while UniAD has the lowest L2 (0.73) but DS 37.72.

  Source: [Bosch, arXiv 2605.00066](https://arxiv.org/html/2605.00066) (30 Apr 2026, preprint)
- World-model simulators from NVIDIA: OmniDreams, "a foundation generative world model mid- and post-trained from the Cosmos diffusion" model, for closed-loop AV simulation — [arXiv 2606.03159](https://arxiv.org/abs/2606.03159) (2 Jun 2026)

### Inferences
- The open frontier now has two benchmark ecosystems: NAVSIM/Bench2Drive (academic) and PhysicalAI-AV/AlpaSim (NVIDIA). A result in one does not transfer automatically to the other. The PDMS–DS agreement (ρ=0.90) comes from only 8 methods.

### Gaps
- EMMA, AutoVLA, ORION, OpenDriveVLA, DriveVLA-W0, ReCogDrive, SimLingo and Impromptu VLA have no per-paper verification here. The same goes for DiffusionDrive, GoalFlow and Hydra-MDP, for GAIA-2/3, Cosmos-Drive-Dreams and the Waymo world model, and for the NAVSIM v2 EPDMS and Bench2Drive leaders. The delegated survey did not return before hand-off, so all of these are UNVERIFIABLE in this file.

## KQ3. How is closed-loop post-training done in 2026, and which failure modes are reported?

### Takeaway
Four verified patterns, each run in a neural-reconstruction simulator:
- **RAD:** direct RL with an imitation regularizer in 3DGS.
- **RoaD:** closed-loop SFT on the policy's own guided rollouts.
- **OPTED:** RL for a privileged teacher, then on-policy distillation into the camera policy.
- **AlpaGym:** closed-loop GRPO through an SDE-sampled flow expert.

The supervised and distillation routes report far better sample efficiency than direct RL.

### Cited Findings
- RAD (HUST + Horizon Robotics, NeurIPS 2025): "we incorporate IL into RL training as a regularization term"; "a 3x lower collision rate" — [arXiv 2502.13144](https://arxiv.org/abs/2502.13144) (18 Feb 2025)
- RoaD (NVIDIA): "robust closed-loop adaptation with orders of magnitude less data than reinforcement learning". AlpaSim driving score +41%, collisions −54% — [arXiv 2512.01993](https://arxiv.org/abs/2512.01993) (1 Dec 2025, preprint)
- OPTED (NVIDIA + ETH authors): "approximately three orders of magnitude fewer simulator interactions than direct RL post-training, while staying closer to the human prior" — [arXiv 2609.20756](https://arxiv.org/abs/2609.20756) (17 Sep 2026, preprint)
- AlpaGym (NVIDIA AV research): closed-loop GRPO on Alpamayo 1.5 configured as follows (see B19–B23):
  - expert sampling: "int_method: sde", noise 0.2, 5 steps;
  - "allowed_outdated_steps: 4";
  - ratio clip 0.2;
  - KL off by default.

  Source: [AlpaGym configs](https://github.com/NVlabs/alpagym) (first commit 16 Jun 2026)
- Failure mode, numerical: "The AR1.5 VLM prefill intermittently emits an all-NaN trajectory", worked around with retry and force_determinism — [AlpaGym policy config comment](https://github.com/NVlabs/alpagym/blob/main/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs/policy/alpamayo_r1.yaml)
- Failure mode, simulator validity: AR1's AlpaSim protocol drops events where the ego "deviates more than 4 m" in order "To mitigate rendering artifacts" — [AR1 v2 §6.1](https://arxiv.org/html/2511.00088v2)
- Failure mode, reward proxy: a reasoning-only reward raises the reasoning score but worsens ADE (2.19 m) and consistency (0.53). The paper calls the result "ungrounded or overconfident reasoning" — [AR1 v2 Table 9](https://arxiv.org/html/2511.00088v2)
- Failure mode, train–inference mismatch: verl's rollout-correction module was motivated by "RL training collapse" caused by the training-inference mismatch — [verl rollout_corr.md](https://github.com/volcengine/verl/blob/main/docs/algo/rollout_corr.md)
- Failure mode, flow and diffusion RL: discretised-reverse-process GRPO inherits "solver restrictions, forward-reverse inconsistency" — [DiffusionNFT arXiv 2509.16117](https://arxiv.org/abs/2509.16117) (ICLR 2026 Oral)
- Simulator limits: reconstruction-based simulators "struggle to generalize to highly dynamic or novel scenes" ([OmniDreams](https://arxiv.org/abs/2606.03159)), and NuRec-style methods "typically require per-scene tuning" ([Instant NuRec](https://arxiv.org/abs/2607.14203))

### Inferences
- For a solo engineer, closed-loop SFT or distillation (RoaD/OPTED style) is the cheaper first lever in AlpaSim. Direct flow-expert RL (E5) should be budgeted as the sample-hungry option.

### Gaps
- RL inside learned world models for driving, DAgger variants beyond RoaD and OPTED, and quantified reward-hacking cases: the delegated survey did not return, so these are UNVERIFIABLE here.

## KQ4. How is teacher→student distillation done for driving VLAs?

### Takeaway
NVIDIA publicly positions Alpamayo 2 Super as a teacher for compact in-car models, but no public distillation procedure was verified in this pass. The closest published closed-loop distillation recipe is OPTED, which uses a privileged non-VLA teacher. AR1 itself distils frontier-LLM knowledge into CoC labels rather than into a smaller policy.

### Cited Findings
- "As a teacher model, it can be distilled into compact models that run on NVIDIA DRIVE AGX Thor inside the vehicle" — [Alpamayo 2 Super blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) (4 Aug 2026)
- AR1 CoC auto-labeling with GPT-5: "This approach distills world knowledge from large models into structured CoC annotations" — [AR1 v2 §4.3.2](https://arxiv.org/html/2511.00088v2)
- OPTED: an RL teacher on vectorized inputs supervises the camera student on-policy, with driving score ×1.6 for TransFuser and ×9.5 for VaVAM — [arXiv 2609.20756](https://arxiv.org/abs/2609.20756)
- AR1's bibliography cites "Distilling multi-modal large language models for autonomous driving" and "DSDrive: Distilling large language model for lightweight end-to-end" driving — [AR1 v2 references](https://arxiv.org/html/2511.00088v2)

### Inferences
- A G5 student recipe can be assembled as follows. Use teacher CoC and trajectory labels to SFT the student, as AR1 does with GPT-5 labels. Then run on-policy correction in AlpaSim, as OPTED/RoaD do. Treat this assembly as untested on reasoning VLAs.

### Gaps
- Published student sizes and teacher–student quality gaps for driving VLAs, and whether NVlabs repos contain a distillation script or roadmap item: UNVERIFIABLE here (the delegated survey did not return).

## KQ5. What do production teams publicly disclose, and what is the convergent architecture?

### Takeaway
Only NVIDIA's disclosure was verified in this pass: a large reasoning VLA used as teacher, distilled to DRIVE AGX Thor. The Tesla, Waymo, Wayve, XPeng, Li Auto, NIO, Momenta and Mobileye findings were delegated and did not return.

### Cited Findings
- NVIDIA: Alpamayo 2 Super is a teacher "distilled into compact models that run on NVIDIA DRIVE AGX Thor" — [NVIDIA blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/) (4 Aug 2026)
- AR1 reports on-vehicle "real-time performance (99 ms latency)" on an "RTX 6000 Pro Blackwell platform" — [AR1 v2 abstract and §6.8](https://arxiv.org/html/2511.00088v2)

### Inferences
- None beyond the NVIDIA teacher→student pattern can be supported from this pass.

### Gaps
- All other company disclosures: UNVERIFIABLE here.

## KQ6. What has NVIDIA's AV research group published in 2025–2026, and which open problems do they name?

### Takeaway
NVIDIA's verified 2025–2026 AV output forms one loop:
- models: Alpamayo-R1, then 1.5, then 2 Super;
- data: PhysicalAI-AV with OOD CoC labels;
- simulators: AlpaSim/NuRec, Instant NuRec, OmniDreams;
- post-training: RoaD, OPTED, AlpaGym.

The open problems they name are closed-loop vs open-loop gaps, the cost of sensor-sim post-training, reasoning–action grounding, and the generalisation limits of reconstruction simulators.

### Cited Findings
- AR1 (30 Oct 2025; v2 7 Jan 2026): "strong open-loop results do not necessarily translate into reliable closed-loop driving performance" — [arXiv 2511.00088v2](https://arxiv.org/html/2511.00088v2)
- RoaD (1 Dec 2025): open-loop behaviour cloning suffers "from covariate shift when deployed in closed loop" — [arXiv 2512.01993](https://arxiv.org/abs/2512.01993)
- OPTED (17 Sep 2026): closed-loop post-training "requires costly simulation for sensor-based policies" — [arXiv 2609.20756](https://arxiv.org/abs/2609.20756)
- OmniDreams (2 Jun 2026): reconstruction simulators "are fundamentally constrained by their initial captured data" — [arXiv 2606.03159](https://arxiv.org/abs/2606.03159)
- Instant NuRec (15 Jul 2026): existing NuRec-style methods "remain relatively slow and typically require per-scene tuning" — [arXiv 2607.14203](https://arxiv.org/abs/2607.14203)
- Release dates:
  - Alpamayo-R1-10B HF repo created 2025-11-22; Alpamayo-1.5-10B created 2026-03-03; Alpamayo2-Super created 2026-05-27. All carry "license:openmdw-1.1" ([HF API](https://huggingface.co/nvidia/Alpamayo-1.5-10B)).
  - The PhysicalAI-AV OOD reasoning test split is "Reserved for OOD Benchmark Challenge" ([HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)).

### Inferences
- NVIDIA has already published closed-loop SFT (RoaD), privileged distillation (OPTED) and closed-loop RL infrastructure (AlpaGym). PhysLoop's differentiation is still measurement: gate validity and the paired release rule. It is not a new post-training algorithm.

### Gaps
- A complete Pavone-group publication list (e.g. Cosmos-Drive-Dreams, GTRS, CAT-K, Cosmos-Reason1) was not verified here.

## KQ7. Is nanoAlpamayo (0.5B–2B VLM backbone) feasible for a solo engineer on rented compute?

### Takeaway
The architecture is feasible: NVIDIA's own 0.5B variant exists and RL infrastructure is open. The data plan is the binding risk:
- 100–300 h is 0.1–0.4% of AR1's 80,000 h.
- AR1's 0.5B overfits at its smallest data-scaling point.
- Public CoC labels cover only about 1,740 clips.
- The dataset licence restricts redistribution.

### Cited Findings
- Data: 1,700 h, "306,152 clips that are each 20 seconds long". Gated. Licence: "You may not create derivative works of the Dataset, sell, rent, sublicense, transfer, distribute". The agreement "expires twelve (12) months after" download — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- Public CoC labels: train 1,450 clips (1,728 labels), val 290 clips (349 labels), test held out — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- NVIDIA's training scale: "80,000 hours of driving data"; CoC set "700K video segments" — [AR1 v2 §6.1](https://arxiv.org/html/2511.00088v2)
- 0.5B data scaling on the internal test set: 100k segments overfit (1.016 m with early stopping); 500k gives 0.880 m; 2M gives 0.874 m — [AR1 v2 §6.5.2](https://arxiv.org/html/2511.00088v2)
- Reference 0.5B design: DINOv2 + Qwen2.5-0.5B. 7B gives an "11%" minADE6 reduction vs 0.5B (reduced training budget) — [AR1 v2 §6.5.1](https://arxiv.org/html/2511.00088v2)
- Community redistributions of Alpamayo weights (z-lab/Alpamayo-1.5-10B, z-lab/Alpamayo-R1-10B) are listed among models trained on the dataset — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

### Inferences
- 100–300 h corresponds to 18,000–54,000 twenty-second clips (100 × 3600 / 20 = 18,000). Each clip yields several 2 s history + 6.4 s future samples.
- AR1's segment length for its "100k segments" point is not stated. If a segment is about 8 s, 100k segments is roughly 220 h. At that scale AR1's 0.5B overfit to 1.016 m on its internal set, which suggests minADE6 ≤ 1.0 m on 100–300 h is at risk. This is a hypothesis, because the test sets differ.
- Budget the Alpamayo 2 Super auto-labelling run for CoC SFT data explicitly.

### Gaps
- Small-backbone licences (Qwen2.5-VL-3B, Qwen3-VL-2B, InternVL3-1B/2B, SmolVLM2, Cosmos-Reason2-2B), published GPU-hour numbers for SFT and GRPO on 0.5–3B VLAs, current H100 prices, and open small-scale Alpamayo reproductions were all delegated and did not return. They are UNVERIFIABLE here, so no dollar estimate is given in this file.

## KQ8. Which open problems are most valued now and tractable for one engineer?

### Takeaway
Evidence verified in this pass supports three problems:
- **Closed-loop evaluation reliability.** Open-loop metrics are uncorrelated with closed-loop scores, and simulators exclude large deviations because of rendering artifacts.
- **Reasoning–action faithfulness.** Consistency degrades without an explicit reward.
- **Train–inference mismatch in RL**, including for flow experts.

All three need measurement skill more than fleet data, which fits a solo engineer.

### Cited Findings
- Open-loop vs closed-loop: "(ADE/FDE) remain uncorrelated with closed-loop driving quality ( ρ=-0.36 , p=0.43 )" — [arXiv 2605.00066](https://arxiv.org/html/2605.00066)
- Faithfulness: with r_reason only, consistency drops 0.62 → 0.53 while reasoning grading rises 3.1 → 4.5 — [AR1 v2 Table 9](https://arxiv.org/html/2511.00088v2)
- Simulator validity: deviations over 4 m are excluded "To mitigate rendering artifacts" — [AR1 v2 §6.1](https://arxiv.org/html/2511.00088v2)
- Deployment cost: VLA inference is "a cascade of four" bottlenecks. Alpamayo 1.5 goes from 717 to 151 ms with W4A8 — [FlashDrive arXiv 2608.12932](https://arxiv.org/abs/2608.12932)
- Community challenges exist on NVIDIA data: the "PhysicalAI-AV-OOD-Reasoning-Challenge-2026" and "AV-Causal-Scenario-Retrieval-Challenge" spaces use the dataset — [HF card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

### Inferences
- Ranking by tractability for one engineer, from this pass's evidence only:
  1. Gate validity (sim-metric reliability).
  2. Reasoning–action consistency with an open judge.
  3. Flow-expert RL stability diagnostics.
  4. Expert quantization.

### Gaps
- Workshop CFPs, the AlpaSim challenge rules (16 GiB / 0.1 s, deadline, registration) and survey-level open-problem lists were delegated and did not return. They are UNVERIFIABLE here.
