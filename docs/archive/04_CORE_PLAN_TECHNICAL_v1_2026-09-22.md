> **ARCHIVED — historical, superseded on 2026-10-03 (DECISIONS D-010, D-011). Never act on this file; the live version is listed in `DOC_AUTHORITY.md`.**

# 04 — PhysLoop core plan: technical reference (22 September 2026, corrected 23 September)

Mirror of the Claude Doc "PhysLoop: Post-Training Engine for Physical AI — Core Plan".
**Status:** active reference for the workload derivation, architecture, experiment method
and correctness tests. Its sequencing and two deliverables (RLinf-fork async engine;
"first RL through the expert") are **superseded by `02_PHYSLOOP_PLAN_v2.md`** — see the
correction below and `DECISIONS.md` D-003/D-004.

> **Correction, Sep 23 2026.** The claim that NVIDIA never trains the flow-matching action
> expert holds only for the open-loop recipe. AlpaGym (first commit Jun 16 2026) already
> runs closed-loop GRPO on the expert with SDE log-probs and bounded staleness, so the
> RL-through-the-expert and async-engine deliverables are largely taken. The workload
> analysis and experiment method below remain valid.

## Decision (as written Sep 22)

We build **PhysLoop**: an open post-training engine for embodied foundation models whose
job is to make the rollout → evaluation → deployment loop run ≥3× faster per GPU at matched
policy quality. Beachhead: autonomous driving on NVIDIA's open Alpamayo stack (Alpamayo
1.5 Nano, 10B, public GRPO and NVFP4 recipes).

**The bet.** Physical-AI post-training in 2026 is generation-bound and memory-bound, not
training-bound; whoever lifts rollout and evaluation throughput sets the field's iteration
speed, because NVIDIA chose to turn robotics' data problem into a compute problem.

**Falsifiable predictions (written before any run)**
1. Open-loop: the stock recipe (4 policy + 1 vLLM GPU) is training-bound on vision-token
   backprop, so the rollout GPU idles ≥30%. Closed-loop: the stock 2-GPU AlpaGym loop leaves
   the trainer idle ≥40%, because per-tick policy inference cannot batch across time and the
   renderer sits inside the loop.
2. Batching the action expert across ≥16 parallel environments cuts per-sample action
   latency ≥4× before it flattens (expert at 54 FLOP/byte, ~5× below the Blackwell BF16
   ridge of 281).
3. Async partial rollouts with ≤2 policy-versions of staleness retain ≥95% of synchronous
   GRPO's final closed-loop score.
4. NVFP4 on the VLM plus FP8 on the expert degrade minADE and closed-loop collision rate
   by ≤1% relative to BF16.
5. RL through the flow expert lowers AlpaSim close-encounter rate ≥15% relative within
   200 GPU-h, and the token-only recipe measurably shifts the expert's conditioning (KL
   between pre- and post-RL VLM inputs to the expert grows with RL steps). *(Now an
   ablation inside G5; AlpaGym already implements the expert path.)*

**Out of scope.** Training a new foundation model, building a robot or a simulator,
competing with vLLM/SGLang.

## The production problem

Every embodied post-training loop measured in 2026 stalls in the same place: trainers wait
on rollouts, rollouts wait on policy inference, evaluation waits on world-model inference.

| Stage | Measured bottleneck | Source |
|---|---|---|
| Rollout, real robots | 1M one-minute trajectories = 700 robot-days | Chelsea Finn, YC podcast, Jul 2026 |
| Rollout, LLM RL (proxy) | Trainer idle 30% at 10.5% MFU with 60% of rollouts discarded (Qwen3-235B); idle 74% on GLM-5; sandbox stragglers of 7,500 s | SemiAnalysis, Jun 16 2026 |
| Policy inference, π0 | 52.6 ms on Jetson Thor, 3.2 ms on B100; expert at 54 FLOP/byte is memory-bound; 10→50 denoising steps = 2.15× latency | VLA-Perf, Feb 20 2026 (arXiv 2602.18397) |
| Policy inference, π0.5 on edge | 458 ms naive on Thor, 7.6 Hz only with async foresight | Jetson-PI, Aug 2026 (arXiv 2607.12659) |
| World-action models | 3–4× slower inference than VLAs | NVIDIA WAM blog, Jun 15 2026 |
| Evaluation | Neural-sim eval reaches r = 0.989 vs real, but "slow inference limits large-scale throughput" | RoboWorld, Jul 2026 (arXiv 2607.01060) |

RL post-training already moves deployed metrics (π*0.6: task throughput 2×, failures
halved, Physical Intelligence Nov 2025). Counter-position (PI): all RL for π0.6 ran on real
robots because simulators miss long-tail contact failures. The engine must be the
invariant across that debate.

## State of the art, verified Sep 22 2026

**Models (open weights):** Alpamayo 1.5 Nano (8.2B VLM + 2.3B flow expert, OpenMDW-1.1,
24 GB); Alpamayo 2 Super (32B Cosmos 3 Reasoner + 2.3B expert, 1×H100 80 GB, AlpaSim
1.50, minADE6 0.911 m); GR00T N1.7-3B (TensorRT 27.9 ms on H100, 80.4 ms on Thor); π0.5
(openpi); Cosmos 3 Edge/Nano (4B/16B); DreamDojo (10.81 FPS after 4-step distillation).
Not available: π0.6/π0.7, Gemini Robotics On-Device 2, GR00T N2, Helix, Skild Brain.

**Environments:** AlpaSim + NuRec (3DGS, gRPC; NuRec renders 100 FPS at 1080p × 6 cams on
RTX 6000 Pro; AlpaSim throughput unpublished); AlpaGym (closed-loop GRPO, 2 GPUs ≥40 GB);
Newton 1.6 via Isaac Lab 3.0-EA Kit-less (Isaac Lab 0.9–1.6M state FPS on 8 GPUs; Kit RTX
rendering needs RT cores, so not H100); RoboWorld (15.3 FPS on H100, r = 0.989);
OmniDreams (68 FPS single-view on GB300); vla-eval (2,000 LIBERO episodes in 18 min on
1×H100).

**RL frameworks:** RLinf v0.3 (hybrid pipelining 1.9–2.4×; Flow-SDE/Flow-Noise, DSRL,
RECAP; GR00T N1.7; Isaac Lab); Cosmos-RL (fully async, NCCL/Redis sync; maintenance
mode); RL-VLA³ (fully async, +85%, no community); verl rollout-correction math (TIS, MIS,
Geo-RS); AReaL (`max_head_offpolicyness`, 2.77×). Algorithms for flow experts: π_RL
Flow-SDE (π0.5 ManiSkill 40.1→90.9%), ReinFlow, DSRL, DiffusionNFT, RECAP (closed).

**Inference levers, measured:** cross-env batching (π0.5 44→100 samples/s at B=32; expert
time share 57%→13.5%; PhyAI arXiv 2608.03682); CUDA graphs + fusion (106.5→27.3 ms,
realtime-vla; GR00T eager→TensorRT 3.08× on H100); one-step distillation (SnapFlow
274→83 ms); streaming chunk decode (FlashVLA up to 20×); W4A4 (FoldQuantVLA 1.25–1.52×;
HoloQ-VLA π0.5 98.0% vs 97.1%); world-model step optimization (DreamZero 5.7 s→150 ms, 38×).
The Alpamayo quantization recipe (ModelOpt 0.43: FP8, NVFP4, AutoQuant 6.5 bits; BF16
22 GB→9 GB) publishes no accuracy delta or latency; Alpamayo is "not available for AGX
Thor currently" (NVIDIA forum).

## First-principles workload analysis

**Descent ladder.** L5: one GRPO epoch over ~1,000 AlpaSim scenarios takes hours, GPUs idle
waiting on each other. L4: renderer ↔ policy ↔ learner pipeline; batch across scenarios;
async weight sync with bounded staleness. L3: GRPO with n=12 samples per prompt; 10-step
Euler flow matching over 64 waypoints; 40–200 autoregressive reasoning tokens per tick.
L2: AR decode and the expert are memory-bound; prefill of 2,880 vision tokens is
compute-bound; FP8 halves bytes, NVFP4 quarters them. L1 (verified specs): B200
2.25 PFLOPS BF16 dense, 8 TB/s; RTX PRO 6000 504 TFLOPS, 1.79 TB/s; H100 989 TFLOPS,
3.35 TB/s; Jetson Thor 258 TFLOPS, 273 GB/s.

**Ridge points (dense):** B200 281 BF16 / 562 FP8; H100 295 / 591; H200 206 / 412;
RTX PRO 6000 281 / 562; Jetson Thor 947 / 1,894 (everything memory-bound; weight bytes
set latency).

**Cost model for one policy tick.** For a weight-streaming operator with P parameters,
b bytes per parameter and batch B:
`T(B) = max(2·P·B / F_peak, b·P / BW)`, crossover `B* = b·F_peak / (2·BW)`.
Below B* the operator is memory-bound and per-sample cost falls linearly with batch;
above it, flat. Prefill has B × tokens rows, so it is compute-bound already at B=1.

**Alpamayo 1.5 Nano, one tick, derived (RTX PRO 6000, BF16, B=1).** Vision: 4 cameras ×
4 frames at 320×576, patch 16, merge 2 → 180 tokens per image, 2,880 vision tokens.
Prefill: 2 × 8.2e9 × ~3,000 tokens ≈ 4.9e13 FLOP → ≈98 ms at peak, ~150 ms at 65%
attainment. Reasoning decode: 16.4 GB of weights per token at 1.79 TB/s → 9.2 ms/token;
40 tokens → 366 ms. Expert: 2.3B × 10 Euler steps → 46 GB of weight traffic → 26 ms.
Total ≈ 500 ms per tick (~2 Hz).

The paper reports 99 ms end-to-end on the same GPU (Table 14: vision 3.43 + prefill 16.54
+ reasoning 70 ms/40 tokens + trajectory 8.75 ms/5 steps). The table does not state model
size or precision, and 1.75 ms per decoded token is 5× faster than the BF16 bandwidth
bound for an 8.2B VLM. Inference: the 99 ms figure is a smaller or quantized on-vehicle
configuration with far fewer vision tokens. **Open question to settle by measurement.**

**Batching across scenarios (B=32).** Decode stays memory-bound (32 < 281), per-sample
decode drops 32× to 11 ms. Expert crosses B*=5 and drops to ~0.1 ms/sample. Prefill
unchanged at ~98–150 ms/sample. Per-tick cost per sample ≈ 110–160 ms → one GPU serves
~6–9 policy ticks/s ≈ 0.03–0.045 20-second episodes per second.

**Renderer bound.** NuRec: 100 FPS at 1080p across six cameras on an RTX 6000 Pro
→ ~25 ticks/s per renderer GPU at 4 cameras, ~3× faster than the policy GPU above.
Alpamayo 1.5 resizes inputs to 320×576, so rendering at 1080p wastes ~11× the pixels the
policy sees. Prediction: rendering at model resolution raises scenarios per renderer GPU
≥4× (unmeasured; AlpaSim throughput is unpublished).

**Trainer bound.** A GRPO update on 10.5B parameters over a ~3,300-token tick costs
≈ 6 × 10.5e9 × 3,300 ≈ 2.1e14 FLOP, ~0.53 s per tick on an H100 at 40% MFU. Open-loop:
trainer ~1.9 samples/s/GPU vs vLLM rollout ~13 samples/s/GPU → training-bound, which
explains NVIDIA's 512 policy : 128 rollout GPU split. Closed-loop: backpropagating every
tick of a 200-tick episode costs ~106 GPU-seconds, 2–3× more than generating it. Whether
AlpaGym subsamples ticks is undocumented; instrument it first.

**Balance equation:** `N_gen · g ≈ N_train · t` (g, t = episodes/s per rollout / trainer GPU).

**Levers with predicted multiples**

| Lever | Mechanism | Predicted gain | Evidence class |
|---|---|---|---|
| Cross-scenario dynamic batching | memory-bound decode and expert amortize weight reads | 2.3–2.8× per-GPU rollout throughput, plateau at B≈16–32 | Reported for π0.5/GR00T (PhyAI); Alpamayo unmeasured |
| History-frame KV reuse across ticks | only 4 of 16 images are new each tick | 2.5–3.5× on the prefill term at B≥16, ≤1% minADE change | Mechanism from VLA-Cache; AV unmeasured |
| Render at model resolution | 11× fewer pixels per frame | ≥4× scenarios per renderer GPU | Unmeasured |
| CUDA graphs + fused Euler loop | removes launch gaps | 3–4× at B=1, 1.3–2× at rollout batch | realtime-vla; GR00T TensorRT 3.08× |
| FP8 prefill, NVFP4 decode weights | prefill compute-bound (2× peak), decode byte-bound (4× fewer bytes) | 1.4–1.8× prefill, ~2× decode, ≤1% minADE | HoloQ-VLA; Alpamayo recipe (no numbers) |
| Async learner, bounded staleness | removes generate-then-train barrier | 1.5–1.9× wall-clock | RL-VLA³ +85%; RLinf 1.9–2.4× — *AlpaGym already async, staleness 4* |
| Tick subsampling / frozen lower layers | cuts the 2.1e14 FLOP per tick training term | 2–4× on the trainer term; quality effect unknown | Unmeasured |

**Correctness hazard the numbers expose.** RL that updates only the VLM shifts the KV
cache the expert reads without adapting the expert. Measure the drift (E5) even though
AlpaGym now trains the expert.

## System architecture (as designed Sep 22; engine parts superseded)

Components: environment backends (AlpaSim + NuRec; Newton/Isaac Lab; neural world model)
behind one `Step(obs_batch) → action_chunk_batch` interface; async scheduler with
version-tagged steps; policy service (batched VLM prefill + AR decode + flow expert, CUDA
graphs); staleness-indexed trajectory store (NCCL P2P + Redis control plane, Cosmos-RL
pattern); learner (Flow-SDE GRPO through the expert + token GRPO on CoC + value head +
rollout correction with verl bounds 0.5/5.0); instrumented NCCL weight sync; eval service
(closed-loop + neural-sim tier). Each environment step carries the policy version that
produced it; the learner corrects or discards by version, never by wall-clock.

**Tensor contract (Alpamayo 1.5, batch of B scenarios):** `images [B,4,4,3,320,576]`
uint8→bf16; `vision_tokens [B,2880,4096]`; `egomotion [B,16,6]`; `coc_tokens [B,≤200]`;
`vlm_kv [layers,B,kv_heads,L,d]` bf16/fp8; `a_t [B,64,2]` (accel, curvature), 10 Euler
steps δt=0.1; `eps_t [10,B,64,2]` stored per step for the policy ratio; `logp_t [10,B]`
fp32; `trajectory [B,64,3+9]`; `reward [B]` (AlpaGym preset: progress 1.0, collision −10,
off-road −5, distance-to-GT −0.01); `policy_version [B]`.

**Learner math.** Flow-SDE turns the deterministic Euler path into a Gaussian chain
`a_{t−δ} ~ N(μ_θ(a_t, KV), σ_t² I)`, so per-step ratios `r_t = π_θ/π_old` exist and GRPO's
clipped objective applies with group-normalized advantages over n=12 rollouts. Staleness
at step level: trajectories from version `v < v_k − k` are dropped; the rest use
sequence-level truncated importance sampling.

## Experiments and ablations (predictions written Sep 22)

| # | Experiment | Variable | Prediction | Kill signal |
|---|---|---|---|---|
| E1 | Profile stock loops | open-loop recipe vs 2-GPU AlpaGym | open-loop: rollout GPU idle ≥30%; AlpaGym: trainer idle ≥40%, policy prefill largest stage | both balanced within 15% → throughput thesis weaker |
| E2 | Policy batch curve | B ∈ {1,4,8,16,32,64} | per-sample expert latency ≥4× lower by B=16; E2E 2.3–2.8×, plateau by B=32 | <1.5× at B=32 → prefill dominates; move KV reuse first |
| E3 | History-frame KV reuse | reuse ViT features/static-token KV for 12 of 16 images | 2.5–3.5× on prefill at B≥16; minADE6 change ≤1% | minADE6 worse >2% → reuse only ViT features |
| E4 | Render at model resolution | 1920×1080 vs 576×320 | ≥4× scenarios per renderer GPU; identical policy outputs | <2× → 3DGS cost not pixel-dominated |
| E5 | RL objective through the deployed path | token GRPO vs Flow-SDE vs DSRL | Flow-SDE: close-encounter −15% relative in 200 GPU-h; token-only improves minADE6 but closed-loop less; drift KL grows | Flow-SDE unstable → DSRL primary |
| E6 | Staleness sweep | k ∈ {0,1,2,4,8} | k≤2 keeps ≥95% of k=0 score; k=8 loses >10% | k=1 loses >5% → step-level MIS first |
| E7 | Precision | BF16 vs FP8 vs NVFP4 on VLM and expert | ≤1% minADE6/close-encounter for FP8 and NVFP4 on the VLM; NVFP4 on the expert costs >1% | >3% on NVFP4-VLM → keep FP8 |
| E8 | Horizontal transfer (dropped in v2) | GR00T N1.7 on Newton | ≥2× vs RLinf collocated | — |

Three seeds each, avg@3; git hash and config on every plot; negative results kept with mechanism.

## Correctness and verification

| Test | Catches | Pass condition |
|---|---|---|
| Sync-equivalence | scheduler/staleness bugs | async at k=0 reproduces the synchronous reward curve within 3-seed noise |
| Log-prob parity | rollout vs learner numerics | KL(train‖infer) < 1e-3 nats on tokens; per-step Gaussian log-density parity within 1e-4 on Flow-SDE |
| Determinism | non-reproducible environments | seeded AlpaSim replay matches offline scorer |
| Overfit one scenario | reward/mask/advantage bugs | reward reaches ceiling in <50 steps |
| Gradient dependency | batch leakage | ∂loss_i/∂input_j = 0 for i≠j |
| Reward hacking | comfort/progress gaming, CoC length | length and comfort stats logged; reward up while close-encounter flat → flag |
| Kernel adversarial harness | agent- or hand-written kernels | NaN/Inf/empty/odd shapes; tolerance vs eager (bf16 1e-2 rel, fp32 1e-5) |
| Expert drift monitor | train/deploy mismatch | KL between pre/post-RL expert inputs logged per update |

Eval lock: AlpaSim score and close-encounter rate on a fixed scenario subset, 3 seeds
(model-card noise floor: AlpaSim 1.37 ± 0.10 for Alpamayo 1.5 on 913 scenarios); minADE6 on
PAI validation clips; 30-scenario vibe set watched at every checkpoint.

**Success estimates (Sep 22):** ≥3× throughput vs stock at matched reward 0.65; Flow-SDE
beats token-only closed-loop 0.5; async k≤2 keeps ≥95% 0.7; NVFP4 within 1% 0.6; neural
evaluator Spearman ≥0.9 vs AlpaSim 0.55; field keeps moving to RL post-training loops
through 2028 0.75; a GTC/MLSys-grade artifact in 6 months 0.5.

## Path to the real world (as of Sep 22; see 02 for the current list)
RLinf issue #1595 (unified serving layer) · AlpaGym PRs (render-at-model-resolution,
profiler) · Nebius × NVIDIA hackathon (dropped, D-002) · GTC 2027 poster (Nov 10 2026) ·
AlpaSim E2E Closed-Loop Challenge 2026 · arXiv preprint Jan 2027 → RSS/CoRL 2027 ·
MLSys 2028 · NVIDIA Inception · one partner with a vehicle (Lucid, JLR, Uber, Berkeley
DeepDrive are named Alpamayo endorsers).

## Sources (all opened Sep 22 2026)
Alpamayo-R1 https://arxiv.org/html/2511.00088v2 · Alpamayo-1.5 card
https://huggingface.co/nvidia/Alpamayo-1.5-10B · Alpamayo2-Super card
https://huggingface.co/nvidia/Alpamayo2-Super · alpamayo-recipes
https://github.com/NVlabs/alpamayo-recipes · AlpaSim https://github.com/NVlabs/alpasim ·
AlpaGym https://github.com/NVlabs/alpagym · NuRec docs
https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html ·
Cosmos-RL https://github.com/nvidia-cosmos/cosmos-rl · Newton
https://github.com/newton-physics/newton · Isaac Lab 3.0-EA
https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA · Cosmos 3 tech report
https://arxiv.org/html/2606.02800v4 · DreamDojo https://arxiv.org/html/2602.06949 ·
DreamZero https://arxiv.org/html/2602.15922 · OmniDreams https://arxiv.org/html/2606.03159v2 ·
RoboWorld https://arxiv.org/html/2607.01060v3 · RLinf https://github.com/RLinf/RLinf ·
RL-VLA³ https://arxiv.org/abs/2602.05765 · verl rollout correction
https://verl.readthedocs.io/en/latest/algo/rollout_corr.html · π_RL
https://arxiv.org/abs/2510.25889 · DSRL https://arxiv.org/abs/2506.15799 · SemiAnalysis RL
systems https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching · VLA-Perf
https://arxiv.org/html/2602.18397v1 · PhyAI https://arxiv.org/html/2608.03682v3 ·
realtime-vla https://arxiv.org/html/2510.26742 · FlashVLA https://arxiv.org/html/2608.27384 ·
SnapFlow https://arxiv.org/html/2604.05656 · HoloQ-VLA https://arxiv.org/html/2605.28803v3 ·
FoldQuantVLA https://arxiv.org/html/2609.24433 · GR00T deployment
https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md · Specs:
https://www.nvidia.com/en-us/data-center/hgx/ · https://www.nvidia.com/en-us/data-center/h100/ ·
https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/
