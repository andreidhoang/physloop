# 04 — Technical reference v2 (3 October 2026)

**Status:** active reference. It covers verified model facts, the tensor contract, hardware, the cost model, upstream tooling, correctness tests and the preserved pre-registered predictions. It contains **no plan or sequencing**; that lives in `02_PHYSLOOP_PLAN.md`. Every number was verified on 2026-10-03 (`research/2026-10-03-verification/`). v1, the 22 September "post-training engine" core plan, is archived; its predictions are reproduced verbatim in §7.

## 1. Model facts (cite the card/config for released checkpoints, the paper for the research model)

| Item | Alpamayo-R1 paper (arXiv 2511.00088 v2; "Alpamayo 1") | Released Alpamayo 1.5 (`nvidia/Alpamayo-1.5-10B`) | Alpamayo 2 Super (`nvidia/Alpamayo2-Super`) |
|---|---|---|---|
| VLM | Cosmos-Reason (paper's ablations use Qwen2.5-VL family) | Cosmos-Reason2-8B, Qwen3-VL architecture; card "Backbone 8.2B" | 32B backbone, per the card field |
| Action expert | 2B (paper) | 2.3B (card); 2.279e9 parameters in safetensors | 2.3B (card field) vs 2B (card prose); 35.81B parameters in safetensors; reported as 32B / 34B / "3×10B" in different sources |
| Total size | — | 11.08B parameters; 22.16 GB in BF16 | Peaks at 72,115 MiB on one H100 80 GB at batch 1 |
| Cameras / images | 2 front cameras, 448×280, 160 tokens per image | 4 cameras × 4 frames, 320×576; 180 tokens per image (derived: patch 16, 2×2 merge) → 2,880 | — |
| Flow steps | Default δt = 0.1 (10 steps); latency table and ablation use 5 | Code default `num_inference_steps = 10` (Euler) | Card measures 10 |
| Reasoning length | Table 14 times 40 tokens | ≈ 16 tokens per tick (FlashDrive) | — |
| Scores | 0.5B: minADE6 0.913 m (644 public PAI-AV samples, Table 10); closed loop 9.0% / 19.0% / 0.35 (920 NuRec scenes, > 4 m excluded); 10B: 0.849 m, 4.0% / 16.0% / 0.72 | minADE6 0.916 m (1,434 samples); AlpaSim 0.81 ± 0.01 (910) → **1.37 ± 0.10 (913) on 2026-08-07, weights unchanged** | minADE6 0.911 m (1,434); AlpaSim 1.50 ± 0.13 (913) |
| License | — | Weights OpenMDW-1.1 (since 2026-08-04) | OpenMDW-1.1 |

Card scores use a different protocol from the paper and are **not comparable** with it. The released Alpamayo 1 is not RL post-trained.

## 2. Tensor contract (Alpamayo 1.5, batch B, verified against card and config)

| Tensor | Shape / dtype | Note |
|---|---|---|
| `images` | `[B, 4, 4, 3, 320, 576]` uint8 → bf16 | 4 cameras × 4 frames over 0.4 s at 10 Hz |
| `vision_tokens` | `[B, 2880, 4096]` | Derived. Risk if frames are packed as video (temporal patch 2): the count would halve |
| `egomotion` | `[B, 16, 3+9]` → 48 history tokens | **Corrected:** v1's `[B,16,6]` appears nowhere |
| VLM KV cache | 36 layers × 8 KV heads × 128, bf16 | **147,456 bytes per token**; a 3,000-token prompt holds 0.442 GB |
| `coc_tokens` | ≈ 16 per tick measured | v1's "≤ 200" is unverified |
| `a_t` | `[B, 64, 2]` (accel, curvature) | `accel_bounds [-9.8, 9.8]`, `curvature_bounds [-0.33, 0.33]`, dt 0.1 |
| Training trajectory tokens | 128 = 64 × 2 | `num_bins 3000`; used in training only, while inference decodes through the expert |
| `trajectory` | `[B, 64, 3+9]` | Position (x, y, z) + rotation matrix |
| Decoder | AR1 Eq. 5 | Heading integrated exactly, position by the trapezoid rule; forward Euler would add 0.60 m of error on a 6.4 s arc |
| AlpaGym RL (as shipped) | Expert only, VLM frozen, 5 SDE steps, trajectory-level clipped ratio | `sde_type="cps"` raises NotImplementedError. Default preset: 8 policy steps, `distance_to_gt` reward |

## 3. Hardware (official datasheets; dense; derived values marked)

| Device | Peak (dense) | Memory BW | Memory | Ridge (BF16, FLOP/B) |
|---|---|---|---|---|
| H100 SXM | 989 TFLOPS BF16 | 3.35 TB/s | 80 GB | 295 |
| B200 | 2.25 PFLOPS BF16 | 8 TB/s | — | 281 |
| RTX PRO 6000 Blackwell, Workstation | ≈ 500 TFLOPS BF16 (derived; NVIDIA prints no "504") | 1.792 TB/s | 96 GB | ≈ 279 |
| RTX PRO 6000 Blackwell, Server | — | 1.597 TB/s | 96 GB | — |
| Jetson AGX Thor (T5000) | 2,070 FP4 TFLOPS sparse; ≈ 258 BF16 (derived from "517 dense FP8") | 273 GB/s | 128 GB | ≈ 947 |
| DRIVE AGX Thor dev kit | 1,000 INT8 TOPS / 2,000 FP4 | 273 GB/s | 64 GB (GPU memory set by carveout) | — |

DRIVE AGX Thor is designed for ASIL-D and runs DriveOS 7. TensorRT 11.3 lists FP8 and FP4 on Thor (SM 11.0), but users report that ModelOpt checkpoints run as fake quantization on Thor. **Label Thor precision claims as emulated unless a real engine was built.** TensorRT-Edge-LLM supports only Alpamayo-R1-10B, with FP16 export; Alpamayo 1.5 support is still absent at v0.11.0.

## 4. Cost model (corrected)

**Weight-streaming operator.** For P parameters at b bytes each and batch B:

T(B) = max(2PB / F_peak, bP / BW)

The crossover is B* = b·F_peak / (2·BW). Below B* the operator is memory-bound; above it, the floor is compute. KV-cache reads grow with B and must be added to the memory term.

**One tick, Alpamayo 1.5, BF16, B = 1, RTX PRO 6000 WS** (bound vs FlashDrive measurement):

| Stage | Bound | Measured (unoptimized → system-optimized) |
|---|---|---|
| Encode + prefill | ≈ 5.48e13 FLOP → 110 ms at peak, 169 ms at 65% | 252 → 229 ms (prefill runs at ~47% of peak) |
| Decode | 15.17 GB weights + 0.44 GB KV → 8.7 ms/token × ~16 tokens = 139 ms | 17.7 → 11.2 ms/token; 171–272 ms in total |
| Expert (8 steps) | 5.0 GB per step → 2.79 ms/step → 22.3 ms | 192.9 → **113.9 ms (~20% of bound; cause undiagnosed publicly)** |
| Total | ≈ 271 ms (8 steps, 16 tokens) | **716.9 → 513.3 ms**; 151 ms with all of FlashDrive's optimizations |

Lessons:
- **v1 was right by accident.** Its 490 ms total holds, but its split was wrong in all three stages.
- **Quantization is a small lever here.** FlashDrive's 717 → 151 ms comes mostly from streaming KV reuse, speculative reasoning, adaptive steps and CUDA graphs. W4A8 removes only 24.6 ms; end to end it is worth 1.16× once the other optimizations are applied.
- **AR1's 99 ms is not the released model.** It implies 1.75 ms/token, 5× below the BF16 bandwidth bound, so it does not describe the released 10B model.

**Batched, B = 32:**
- **Decode** costs 16.4 ms per step, including KV reads, which is 0.51 ms per sample per token.
- **The expert is compute-bound** at 22.4 ms per step, so about **7.0 ms per sample over 10 steps**. v1's "0.1 ms" was off by about 70×.
- **Rollout throughput** is about **4 ticks/s per GPU**, not v1's 6–9.

**Thor decode floor** (273 GB/s): Alpamayo 1.5 in BF16 = 15.17 GB/token → **55.6 ms/token**. A 3B student needs 22 ms in BF16 and **11.0 ms** in FP8. A 0.1 s budget therefore allows only a few reasoning tokens (inference).

**AlpaSim sweep cost.** FlashDrive measured 463 ms (optimized) to 1,150 ms (baseline) per rollout step, covering inference, trajectory optimization and rendering. One sweep is 441 scenes × 3 seeds × 200 steps, so it takes **34–85 GPU-hours single-stream**. That is an upper bound, and S2's measurement replaces it.

**Renderer.** NuRec's "100 FPS at 1080p on an RTX 6000 Pro" does not say whether it counts aggregate or per-camera frames, so it means 25 or 150 ticks/s at 4 cameras. 1080p has 11.25× the pixels of 320×576, and Alpamayo drivers reject images smaller than 320×576.

**Balance equation (RL):** N_gen · g ≈ N_train · t.

## 5. Upstream tooling facts (2026-10-03)

| Repo | Facts that constrain the plan |
|---|---|
| **AlpaSim** (`NVlabs/alpasim`) | gRPC microservices: runtime hub, renderer (default NRE-GA 26.04), driver, controller (linear MPC; nonlinear MPC on the challenge), physics, optional CAT-K traffic (disabled by default; its seed is unused). Seeds: `RolloutSpec.random_seed` on `main`, re-added by external PR #149 on 2026-08-17; `DriveSessionRequest.random_seed` on `e2e_challenge`. Ships `results-summary.json`, Prometheus telemetry (default since 2026-06-30), the 1,761/441 split (PR #167) and Drive-IRT. **Lacks** a paired verdict, stratification, test–retest, agreement tables and a monitor implementation (the `safety_monitor_safe` hook exists). About 11 external merges, usually within 0–6 days, with no CLA. Syncs from NVIDIA's internal GitLab roughly monthly |
| **AlpaGym** (`NVlabs/alpagym`) | First public commit 2026-06-16. Code-frozen mirror since 2026-07-08. Expert RL runs on unmerged recipes PR #18. Zero external PRs; DCO plus two internal reviewers. Opt-in `perf/` module parses AlpaSim telemetry. Needs 2 GPUs with ≥ 40 GB each |
| **alpamayo-recipes** | ModelOpt 0.43 PTQ: `fp8` model-wide except `patch_embed`; `nvfp4` (also skips `action_in_proj` trunk.0); `w4a8_nvfp4_fp8`; AutoQuant (default 4.8 bits; the 6.5-bit example is about 9 GB). Open-loop minADE only; no accuracy or latency numbers published. Alpamayo 2 Super SFT recipe (2026-09-30). Pins: torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, ModelOpt 0.43.0. **No distillation recipe** |
| **Challenge** (`e2e_challenge`) | Score = 0 on an at-fault collision (front or lateral; rear is never at fault), off-road, or `left_corridor_laterally` ≥ 4 m; otherwise min(progress / 0.8, 1). Close-encounter has no scorer; stall, oscillation and red-light have no scorer anywhere. 16 GiB hard limit, 0.1 s target, images ≤ 40 GiB, no outbound network |
| **Drive-IRT** (`kesai-labs/drive-irt`) | Zero-and-one-inflated Beta IRT with per-item discrimination, fitted by variational inference. Its 95% intervals are variational and exclude seed and scene variance. Needs ≥ 6 subjects (otherwise falls back to a plain average). No license; no external validation |

## 6. Correctness tests (carried over from v1, still valid)

| Test | Catches | Pass condition |
|---|---|---|
| Determinism and negative control | Non-reproducible environment; false blocks | A seeded replay matches the offline scorer; a bit-identical re-export is never flagged |
| Known-answer statistics | Wrong tails, one- vs two-sided confusion | The S1 known-answer suite (`02` §3) |
| Log-prob parity | Rollout vs learner numerics | KL(train ‖ infer) < 1e-3 nats on tokens; per-step Gaussian log-density parity within 1e-4 on Flow-SDE |
| Sync-equivalence | Scheduler and staleness bugs | Async at k = 0 reproduces the synchronous reward curve within 3-seed noise |
| Overfit one scenario / one batch | Reward, mask and advantage bugs | Reward reaches its ceiling in < 50 steps; loss at init matches theory |
| Gradient dependency | Batch leakage | ∂loss_i / ∂input_j = 0 for i ≠ j |
| Reward hacking | Comfort or progress gaming; CoC length | Length and comfort stats logged; reward rising while close-encounter stays flat → flag |
| Expert drift monitor | Train/deploy mismatch | KL between pre- and post-RL expert inputs logged per update |
| Retry audit | Biased scores from NaN-trajectory retries | Every retry logged; the retry rate is reported per policy |

## 7. Pre-registered predictions of 22 September 2026, preserved verbatim

These predictions were written before any run. They are kept unchanged. A premise note records where the verification changed an assumption. Reconcile each one against measurement when its experiment runs (`02` stage in brackets).

**Headline predictions.** Text as written:
1. "Open-loop: the stock recipe (4 policy + 1 vLLM GPU) is training-bound on vision-token backprop, so the rollout GPU idles ≥30%. Closed-loop: the stock 2-GPU AlpaGym loop leaves the trainer idle ≥40%, because per-tick policy inference cannot batch across time and the renderer sits inside the loop." *[S8, optional]*
2. "Batching the action expert across ≥16 parallel environments cuts per-sample action latency ≥4× before it flattens (expert at 54 FLOP/byte, ~5× below the Blackwell BF16 ridge of 281)." *[S8]* Premise note: 54 FLOP/byte is π0's modeled intensity. Alpamayo's expert crosses into compute-bound at B* ≈ 4.4 samples.
3. "Async partial rollouts with ≤2 policy-versions of staleness retain ≥95% of synchronous GRPO's final closed-loop score." *[not scheduled]* Premise note: AlpaGym already runs staleness up to 4.
4. "NVFP4 on the VLM plus FP8 on the expert degrade minADE and closed-loop collision rate by ≤1% relative to BF16." *[S4]*
5. "RL through the flow expert lowers AlpaSim close-encounter rate ≥15% relative within 200 GPU-h, and the token-only recipe measurably shifts the expert's conditioning (KL between pre- and post-RL VLM inputs to the expert grows with RL steps)." *[S7]* Premise note: the challenge has no close-encounter scorer, and AlpaGym trains only the expert.

**Experiment table.** Text as written:

| # | Experiment | Variable | Prediction | Kill signal | Stage / premise note |
|---|---|---|---|---|---|
| E1 | Profile stock loops | open-loop recipe vs 2-GPU AlpaGym | open-loop: rollout GPU idle ≥30%; AlpaGym: trainer idle ≥40%, policy prefill largest stage | both balanced within 15% → throughput thesis weaker | S8 (optional). AlpaGym's default is an 8-policy-step preset, not 200 ticks |
| E2 | Policy batch curve | B ∈ {1,4,8,16,32,64} | per-sample expert latency ≥4× lower by B=16; E2E 2.3–2.8×, plateau by B=32 | <1.5× at B=32 → prefill dominates; move KV reuse first | S8 |
| E3 | History-frame KV reuse | reuse ViT features/static-token KV for 12 of 16 images | 2.5–3.5× on prefill at B≥16; minADE6 change ≤1% | minADE6 worse >2% → reuse only ViT features | Not scheduled; FlashDrive measured streaming KV reuse |
| E4 | Render at model resolution | 1920×1080 vs 576×320 | ≥4× scenarios per renderer GPU; identical policy outputs | <2× → 3DGS cost not pixel-dominated | S8. An AlpaSim camera-config change, not a PR |
| E5 | RL objective through the deployed path | token GRPO vs Flow-SDE vs DSRL | Flow-SDE: close-encounter −15% relative in 200 GPU-h; token-only improves minADE6 but closed-loop less; drift KL grows | Flow-SDE unstable → DSRL primary | S7 (last, as an ablation) |
| E6 | Staleness sweep | k ∈ {0,1,2,4,8} | k≤2 keeps ≥95% of k=0 score; k=8 loses >10% | k=1 loses >5% → step-level MIS first | Not scheduled |
| E7 | Precision | BF16 vs FP8 vs NVFP4 on VLM and expert | ≤1% minADE6/close-encounter for FP8 and NVFP4 on the VLM; NVFP4 on the expert costs >1% | >3% on NVFP4-VLM → keep FP8 | S4. FlashDrive: W4A8 is 1.16× end to end |
| E8 | Horizontal transfer | GR00T N1.7 on Newton | ≥2× vs RLinf collocated | — | Dropped (v2) |

**Success estimates of 22 Sep**, as written: "≥3× throughput vs stock at matched reward 0.65; Flow-SDE beats token-only closed-loop 0.5; async k≤2 keeps ≥95% 0.7; NVFP4 within 1% 0.6; neural evaluator Spearman ≥0.9 vs AlpaSim 0.55; field keeps moving to RL post-training loops through 2028 0.75; a GTC/MLSys-grade artifact in 6 months 0.5."

## 8. Sources

- **Models:** arXiv 2511.00088 v2; HF `nvidia/Alpamayo-1.5-10B` (card, config, safetensors); HF `nvidia/Alpamayo2-Super`.
- **NVIDIA repos:** `NVlabs/alpamayo1.5` (`diffusion/flow_matching.py`); `NVlabs/alpasim` (main and `e2e_challenge`); `NVlabs/alpagym`; `NVlabs/alpamayo-recipes`; `NVIDIA/TensorRT-Edge-LLM` (issue #130).
- **Papers:** FlashDrive (arXiv 2608.12932, Tables 1 and 3); VLA-Perf (arXiv 2602.18397, modeled).
- **NVIDIA docs:** NVIDIA datasheets (H100, B200, RTX PRO 6000 Workstation and Server, Jetson Thor, DRIVE AGX); TensorRT support matrix; NuRec scaling docs.
- **Full ledger:** `research/2026-10-03-verification/report.md` §10, parts F–H.
