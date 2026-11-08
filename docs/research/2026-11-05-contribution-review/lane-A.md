# Researcher A — hardware, kernels and AV inference lane (PhysLoop S2/S4)

Agent-authored independent review, 2026-11-05. Read-only: public GitHub source, public HF config JSON (no weights), NVIDIA docs. No GPU code, no downloads of weights/datasets/gated assets, no license acceptance, no rental, no posting. Device/HIL/Thor work (S6/S7) stays deferred; nothing here reactivates it. This is not Huy's mastery evidence and fills no scientific prediction slot.

Evidence labels used throughout: **[S]** source-inspected at the stated SHA; **[D]** doc-stated by the named source; **[Δ]** derived here (arithmetic in `scratchpad/derive/derive.py`, output `derive_out.txt`); **[H]** hypothesis, not established. Agreement with other lanes would not upgrade any label.

Local copies of every inspected file are under `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/0fe84b24-06c3-443d-87b9-6d50134535e0/scratchpad/src/` (per repo/SHA) and `.../scratchpad/hf/` (configs). NVIDIA RTX PRO Blackwell architecture whitepaper v1.1 PDF copy: `.../scratchpad/nvpdf/rtxpro_bw_arch_v1_1.pdf`, SHA-256 `1ca55d8f2e87610d1cb9b2c9bda2b2221b1a16e76c1b2e758135c29758b4b676`.

---

## 0. Revision check (reads at ≈ 2026-11-05 16:29 UTC)

| Reference | Pinned in brief | Head observed now | Moved since 2026-11-01? | What moved |
|---|---|---|---|---|
| AlpaGym main | 972d160e… | 972d160e… (2026-07-21) | No | — |
| recipes PR #18 head (Gym's expert dependency) | 1d99fc50… | 1d99fc50…, still open, `mergeable_state: dirty`, updated 2026-06-30 | No | — |
| recipes main | 2e907dab… | **c53aa09f503eae54a2397e4d57c22cf299c111b7** | **Yes** | f600b20f (#44 Measurement chapter, 11-03); 3be3b729 (#42 distillation, 11-05); 4d61fe84 (#43 speculative-decode draft models, 11-05); c53aa09f (#45 5 Hz 2B inference, 11-05). `recipes/alpamayo1_5_quant/README.md` is byte-identical at 2e907dab and c53aa09f [S]. |
| AlpaSim e2e_challenge | e121e591… | **297b4d00cfdd2c4f636b9d9bcee7a85c699174c0** (2026-11-03) | **Yes** | #228 "Add nuPlan leaderboard reference scores", external SII author; inspected file list is `e2e_challenge/local_evaluation/README.md` + reference `results-summary.json` data; commit text says no evaluator/simulation change [S/D]. Driver/runtime files used below are unchanged by it. |
| AlpaSim main | affc2eab… | affc2eab… (2026-08-18) | No | — |
| alpamayo1.5 main | 36aeb4c5… | 36aeb4c5… (2026-09-09) | No | — |

Consequence: PRs #42/#45 that the docs describe as "open" are now merged on recipes main; #44 is merged. None of them changes Gym's dependency (PR #18 head) or the AlpaSim driver.

---

## 1. Observation → action path, as dispatched

Two different code paths serve the same released weights. **(E)** = AlpaSim `e2e_challenge` Alpamayo 1.5 driver (the S2 instrument; also the documented evaluation target for Gym-trained checkpoints, per `convert_alpagym_checkpoint_to_inference.py` L4–9 [S]). **(G)** = AlpaGym rollout (`AlpamayoR1InferenceModel` → recipes PR #18 `ExpertModelRL`, class lineage `alpamayo_r1.models.alpamayo_r1.AlpamayoR1`, not `alpamayo1_5.Alpamayo1_5` [S: pr18 `expert_model.py` L24–31]). Line locators are at the pinned SHAs (E: alpasim e121e591 + alpamayo1.5 36aeb4c5; G: alpagym 972d160e + recipes 1d99fc50).

| Quantity | (E) AlpaSim e2e driver | (G) AlpaGym rollout | Label |
|---|---|---|---|
| Entry point | `AlpamayoBaseModel.predict` → `Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout` (`alpamayo_base.py` L416–491, L318–329) | `sample_trajectories_from_data(with_vlm_rollout=False)` → `_sample_trajectories_from_data_without_vlm_rollout` (`inference_model.py` L104–170; pr18 `model.py` L395–561) | [S] |
| Cameras / order | 4: cross_left(0), front_wide(1), cross_right(2), front_tele(6), sorted by index (`alpamayo1_5.yaml` L32–36; `alpamayo_base.py` L33–41, L385–388) | same 4, same sort (`alpamayo_r1.yaml` L9–13; `tokenize_online.py` L94) | [S] |
| Rendered frame | 1080 high, 1900 requested, decoded 1916 or 1920 wide, 10 Hz, 30 ms shutter (`alpamayo_configs.yaml` L10–38; Gym's AlpaSim pin `4cam_1080.yaml` "NRE will resize to 1916") | same renderer config (`+cameras=4cam_1080`, smoke/test-run yaml L30) | [S] |
| Decode / resize | PIL (CPU) decode at full resolution (`main.py` L784); Qwen fast processor `smart_resize`, factor 32, min/max pixels 163,840/196,608 (`helper.py` L23–25, L190–199) → **320×576** every call | nvJPEG on GPU + `tvf.resize(…, [320, 512], antialias=True)` at ingest (`buffers.py` L237–253; `input_size: [320, 512]` `alpamayo_r1.yaml` L14); 320×512 = min_pixels exactly, so the processor keeps **320×512** | [S] + [Δ] smart_resize(1080,1916)=(320,576) |
| Patches / vision tokens | 720 patches → 2×2 merge → **180/image, 2,880 total** | 640 → **160/image, 2,560 total** (−11.1%) | [Δ] |
| Image geometry | x-scale/y-scale = (576/1916)/(320/1080) = 1.015 | (512/1916)/(320/1080) = **0.902** (≈10% horizontal squeeze) | [Δ] |
| Frames per camera | 4, nearest to t0−{300,200,100,0} ms, anchored at newest frame (`alpamayo_base.py` L335–374) | ring of 4; every 10 Hz frame between ticks is appended (`driver_server.py` L144; `buffers.py` L19–103) → t0−300…t0 at 100 ms | [S]; G spacing [S→H, confirm from recorded `ModelInput.relative_timestamps`] |
| Ego history | 16 poses at 0.1 s, interpolated, rig frame at newest camera time (`alpamayo_base.py` L76–152, L206–211) → 48 history tokens (`helper.py` L106) | last 16 buffered poses (pose reporting 100 ms), rig frame at newest pose (`policy.py` L164–200); `tokens_per_history_traj` 48 | [S] |
| Prompt | system + per-camera "Front camera: frame k " + images + history + "output the chain-of-thought reasoning of the driving process, then output the future trajectory." + assistant `<|cot_start|>` (`helper.py` L38–142) | same camera/frame labels (`conversation.py` L155–196; constants) + "output the future trajectory." + assistant `<|traj_future_start|>` (`conversation.py` L18–22; `tokenize_online.py` L76–78; `inference_model.py` L123–126) | [S] |
| Prefix length | ≈3,090 tokens before CoC (+ sampled CoC) | ≈2,730 tokens | [Δ] estimate ±3%; confirm by logging `prompt_cache.get_seq_length()` |
| Autoregressive decode | HF `generate`, do_sample, T 0.6, top-p 0.98, discrete-trajectory logits masked, stop one token after `<|traj_future_start|>`, `max_new_tokens` = config `tokens_per_future_traj` = **128** (driver passes none) (`alpamayo1_5.py` L288–321; HF config.json) | **none** (Gym comment: skipping it removes "~2x latency") | [S] |
| Backbone | Qwen3-VL-8B architecture (Cosmos-Reason2-8B base, `vlm_name_or_path`): ViT 27 layers × 1152, MLP 4304, 16 heads, patch 16, temporal patch 2, 2×2 merge, deepstack taps 8/16/24 = **0.576 B**; LLM 36 layers × 4096, 32 Q / 8 KV heads × 128, MLP 12,288, vocab 155,697, untied head = **8.22 B** (6.95 B body + 0.64 B embed + 0.64 B head) | identical weights | [D] configs; [Δ] counts |
| Expert | 36-layer Qwen3 decoder copy of the text config with hidden **2048**, 16 Q heads × 128, **8 KV heads × 128** (inherited), MLP **8,256**, no embedding = **2.279 B**; action_in_proj 1.35 M (Fourier 20/dim + timestep → MLP 60→512→512→2048, LayerNorm); action_out_proj Linear(2048→2) | identical weights | [D]+[Δ]; whole model sums to **11,078,526,194 = HF safetensors count exactly** |
| Expert conditioning | **Shared-KV joint attention, not cross-attention**: expert layer *l* runs with `past_key_values=prompt_cache`, so its 64 query tokens attend to [VLM layer-*l* prefix K/V ‖ their own K/V]; cache cropped after each step; non-causal (`is_causal=False`); RoPE ids continue after the prefix (`alpamayo1_5.py` L101–113, L188–242, L357–387) | same mechanism; mask all-zeros (pr18 `model.py` L417–501) | [S] |
| Action representation | x ∈ ℝ^{64×2}: normalized (accel, curvature) per 0.1 s; unicycle integration to xyz + rotation (accel ±9.8 m/s², curvature ±0.33 1/m, dt 0.1, mean/std normalization; HF `action_space_cfg`); **horizon 64 × 0.1 s = 6.4 s**; rig frame at t0 (x fwd, y left), metres | same | [S]/[D] |
| Denoising | Euler **ODE**, `num_inference_steps` default **10**, x0 ~ N(0, I) from the process-global RNG (`flow_matching.py` L32–50, L171–191) | Flow-SDE, `noise_level` 0.2, **5** steps (4 in smoke), one seeded `torch.Generator` per row (`alpamayo_r1.yaml` L34–38; `inference_model.py` L128–153); per-step std in normalized units **[0.200, 0.179, 0.110, 0.073, 0.045]** | [S]; std [Δ] from pr18 `flow_matching.py` L590–658 |
| Samples / selection | 1 sample; index [0,0,0] (`alpamayo1_5_model.py` L55; `alpamayo_base.py` L44–59) | 1 sample; `IdentitySelector` (`selectors.py` L46–67; `closest_to_previous` exists, not default) | [S] |
| Policy cadence | `control_timestep_us` 100 ms → **10 Hz**; `assert_zero_decision_delay: true`; driver skipped during 1.7 s force-GT (`base_config.yaml` L275–293; `alpamayo_configs.yaml` L8) | 200 ms → **5 Hz**; 1.6 s force-GT; 30 sim steps → 22 policy steps (smoke/test-run yaml L27–30) | [S] |
| Controller cadence / type | **Nonlinear** MPC by default (`base_config.yaml` L11 `controller: nonlinear`), `dt_mpc` 0.1 s, horizon 20 steps = 2.0 s (`configs/controller/default.yaml` L1–3; `mpc_controller.py` L19–20) → one MPC solve per 10 Hz policy tick | Gym's AlpaSim pin 46b401be defaults to **linear** MPC (`base_config.yaml` L11 `controller: default`), same `dt_mpc` 0.1 s and horizon 20 → **two** MPC solves per 5 Hz policy tick tracking the same reference; Gym `default.yaml` `extra_overrides: ""` does not change it | [S] |
| Rectification | none configured for Alpamayo 1.5 (`alpamayo1_5.yaml`/`alpamayo_configs.yaml` have no `rectification:` key; driver rectifies only when configured, `main.py` L233–327) | none (`buffers.py` decode + resize only) | [S] — both feed unrectified f-theta frames |
| Batching | `max_batch_size: 1`; `predict_batch` = sequential loop (`base.py` L242–253) | `InferenceEngine` opportunistically drains 1–**4** queued rows from different sessions (`inference_engine.py` L99–157) | [S] |

**Sampled → selected → commanded → executed.** *Sampled*: each candidate x^(k) after the last ODE/SDE step, integrated to `pred_xyz/pred_rot`. *Selected*: candidate [0,0] (identical to sampled when K=1). *Commanded*: (E) the selected xy+yaw converted to local-frame poses at t_now + k·100 ms with the current pose prepended (`main.py` ≈L938–1017), transformed noisy→true frame (`events/policy.py` L157–161), pushed through `DelayBuffer`, handed to the controller as a reference; trajectory optimizer disabled for A1.5 (`alpamayo1_5.yaml` L48–50). (G) the same idea via `build_policy_output_trajectory`. *Executed*: the vehicle-model propagation under MPC/physics between policy ticks (`events/controller.py` L49–170; controller `system.py` L96–200, re-anchoring the reference at the vehicle's pose at the reference's first timestamp, L332–359). Scorers and Gym's `distance_to_gt` see the executed path, not the plan. [S]

**Does compute time ever affect simulated outcome?** No, in all three inspected AlpaSim trees (e2e_challenge, main, Gym's pin 46b401be): `PolicyEvent.run` awaits `driver.drive` and simulated time advances only through scheduled events (`events/policy.py` L59–70, L135–141; `event_loop.py` scheduling ≈L464–520) [S]. The only latency in dynamics is `planner_delay_us`, a configured simulated-time constant (default 0 in `config.py` ≈L312 and `base_config.yaml` L288) consumed by `DelayBuffer.at(step_start_us)` in the controller event (`events/controller.py` L123–127; Gym pin L110–114, same buffer code) [S]. From `item_at` semantics the effective delay is **⌈d/Δt⌉·Δt** for d > 0: 150 ms → 200 ms at 10 Hz, 50 ms → 200 ms at 5 Hz [Δ]. Therefore **any speed-up in this lane buys only wall-clock/GPU-hours (or changes numerics), never different driving outcomes**, unless a delay model is explicitly configured (S7, deferred). One caveat [H]: in (G), opportunistic batch composition depends on timing, and bf16 GEMMs are not guaranteed batch-invariant, so a faster or slower engine can change which bitwise trajectory a seeded rollout produces (not its distribution).

---

## 2. Runtime precision and kernel dispatch — source-inferred

| Component | (E) AlpaSim driver | (G) Gym rollout | Gym trainer replay |
|---|---|---|---|
| Weight dtype | bf16 (`from_pretrained(dtype=bf16)`, `alpamayo1_5_model.py` L76–80) plus `torch.autocast(bf16)` (`alpamayo_base.py` L488–491) | bf16 (`bundle.py` L157–162); `torch.inference_mode` | FSDP-wrapped `ExpertModelCosmos`; param/compute dtype set by Cosmos config, **not inspected** [H] |
| Attention impl | `sdpa` forced by driver (overrides HF config's `flash_attention_2`); expert forced to sdpa anyway (`alpamayo1_5.py` L106–109) | `sdpa` | `sdpa` (ExpertModel declares `_supports_flash_attn_2=False`, pr18 `expert_model.py` L41–43) |
| Vision attention | transformers v4.57.1 non-FA2 path **splits per image and calls SDPA once per image per layer** (16 × 27 = 432 calls/policy call) (`modeling_qwen3_vl.py` L222–243) | same | same |
| Expert attention | explicit additive mask tensor → PyTorch's flash backend is ineligible; memory-efficient or math backend expected [H] | same | parallel replay of all T steps in one forward with a block-diagonal mask (pr18 `model.py` L210–315) |
| torch.compile / CUDA graph | neither; PR #36 graph is opt-in and never enabled by the driver [S] | neither; PR #36 lives in alpamayo1.5, not in the PR #18 lineage [S] | n/a (PR #36 falls back to eager whenever grad is enabled, `diffusion_expert_cuda_graph.py` L165–181) |
| Determinism switches | none; session seed ignored on e2e_challenge (per 04 §10) | `set_deterministic()`: cudnn deterministic, TF32 allowed, `use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG` set **at runtime** (`determinism.py` L22–35) — best-effort only | — |

Confirming capture (not run): Nsight Systems `--trace=cuda,nvtx,osrt,cublas` with NVTX ranges around ViT, LLM prefill, each `generate` step and each expert step, after warm-up; the kernel names in the CUDA trace identify the SDPA backend (flash vs memory-efficient vs math) and GEMM families; Nsight Compute on the top-5 kernels for achieved tensor throughput and DRAM bytes. Check `torch.backends.cuda.*_sdp_enabled()` and log `attn_implementation` at load.

---

## 3. FLOPs, bytes and the roofline (all [Δ], bf16, 2 FLOP/MAC, B = 1)

Hardware [D]: RTX PRO 6000 Blackwell **Workstation** — NVIDIA RTX PRO Blackwell architecture whitepaper v1.1, Appendix A Table 4 (pp. 46–47): peak BF16 tensor **503.8 dense** / 1,007.6 sparse TFLOPS (FP32 accumulate), FP8 **1,007.6 dense** / 2,015.2 sparse, **1,792 GB/s**, boost 2,617 MHz, 128 MB L2, PCIe Gen 5, 600 W; footnotes: rates at boost clock; slash value uses sparsity. The product page gives 1,792 GB/s and "4000 TOPS" FP4 with sparsity. Ridge: **281 FLOP/B BF16, 562 FLOP/B FP8**. This corrects `04` §3's "NVIDIA prints no 504": the whitepaper prints 503.8.

| Stage | (G) Gym rollout | (E) AlpaSim driver | Regime |
|---|---|---|---|
| ViT, per image | 6.3e11 FLOP (640 patches) | 7.2e11 (720 patches) | compute-bound |
| ViT, 16 images | 1.01e13 → 20 ms at peak | 1.15e13 → 23 ms | compute |
| LLM prefill | L ≈ 2,730: 4.01e13 → 80 ms at peak; 13.9 GB weights → ~2,900 FLOP/B | L ≈ 3,090: 4.57e13 → 91 ms | compute (≫ ridge) |
| CoC decode, per token | — | weights 15.6 GB/token (+KV) → **8.7 ms/token** bound | memory |
| Expert, per step, B = 1 | 3.4e11 FLOP, 4.96 GB (4.56 GB weights + 0.40 GB prefix KV) → 69 FLOP/B → **2.8 ms** bound | same per step | memory; crosses to compute at B ≈ 6–8 |
| Expert, all steps | 5 × 2.8 = 13.8 ms | 10 × 2.8 = 28 ms | memory |
| Policy call, bound | ≈ 103 ms compute + ≈ 14 ms expert ≈ **117 ms** | ≈ 114 + 8.7·N_CoC + 28 ms → **≈ 281 ms at N_CoC = 16** (N ≤ 128) | — |
| Prefill share of FLOPs | 96.7% | 94.2% (excluding decode) | — |

Which part dominates. (G): ViT + LLM prefill dominate both FLOPs and bound time (~88%); the expert's share in practice is inflated by launch overhead (≈ 36 layers × ~15 kernels × steps), consistent with PR #36's −15.5% policy-call latency at 10B inside AlpaGym [D]. (E): CoC decode is memory-bound and grows with N_CoC; above ≈ 13 generated tokens its bound exceeds prefill-at-peak; the 10 eager expert steps add launch-bound time. `04` §4's third-party breakdown (encode+prefill 252 ms, decode 171–272 ms, expert(8) 193 ms) has the same ordering [D, third party]. Confirm with: nsys per-phase NVTX + CUDA-event timing after warm-up; per-tick CoC length histogram; ncu achieved TFLOP/s on prefill GEMMs and DRAM GB/s on expert GEMMs; record actual clocks/power, since peaks assume boost clock.

Trainer replay per row (G) [Δ]: frozen prefill 5.0e13 FLOP (no grad) + expert parallel forward over 5×64 tokens 1.75e12 + backward ≈ 3.5e12. **The frozen VLM is 90.6% of replay FLOPs per row.** With `mini_batch: 1` every row also pays one AdamW step over 2.28 B parameters (≈ 77 GB of traffic with fp32 master + moments, ≈ 43 ms at peak bandwidth; the dtype assumption is [H]). Prefix KV is 147,456 B/token → **0.403 GB per row**.

---

## 4. Prior art at current upstream

- **alpamayo1.5 PR #36** (merged 2026-08-29; follow-up #38 fixes capture on a non-current device, 2026-09-09) [S/D]. It adds opt-in bounded CUDA-graph replay of inference-time expert forwards. It is lazy, keyed by exact tensor and prompt-KV shape signatures, and holds at most `max_graphs` (default 4) graphs with **no eviction**. Any later new signature runs eager (`diffusion_expert_cuda_graph.py` L153–260). It is eager whenever grad is enabled, in training mode or causal mode. The PR body reports, for one preliminary paired sample in **NVIDIA's AlpaGym integration (not public Gym main)**: 10B policy-call latency −15.53%, synchronized trainer-step throughput +56.32%, iteration 167.7 → 107.3 s while trainer compute stayed 41.9 → 44.7 s. That pipeline was therefore rollout-supply-bound before and after. A PR comment reports 160/160 bitwise-identical eager-vs-graph evaluations at batch 1–4 [D].
  - Derived limit [Δ from S]: in (E), the KV length = prompt + variable CoC length, so only the first 4 distinct lengths would ever be graphed and the rest fall back to eager. In (G), the prefix length is fixed, so batch sizes 1–4 fill the 4 slots. The AlpaSim driver does not enable it, and the Gym PR #18 path does not contain it.
- **Quantization recipe** (`recipes/alpamayo1_5_quant/README.md`, unchanged at c53aa09f) [D/S]:
  - Method: ModelOpt 0.43 PTQ formats `fp8`, `nvfp4`, `w4a8_nvfp4_fp8` and `auto` (AutoQuant mixing NVFP4 and FP8, default 4.8 effective bits; a 6.5-bit example is ≈ 9 GB), with `max` or `smoothquant` calibration on 100 clips by default.
  - Output: `mtq.compress` stores real low-bit weights unless `--fake_quant`, which keeps Q/DQ nodes for "downstream SDKs like TensorRT".
  - Measurement: eager HF/PyTorch `eval.py` reports open-loop minADE (644 clips, 6 samples, `max_generation_length` 256, seed 42) and synchronized mean ms per clip.
  - Tested on an RTX 5090 with CUDA 12 and a B300 with CUDA 13. No accuracy or latency numbers are published.
  - Open question [H]: whether compressed weights hit FP8/FP4 tensor-core GEMMs in eager mode or are dequantized.
- **Alpamayo NIM** [D]:
  - **1.0.0** (release notes dated 2026-08-27) serves Alpamayo 1.5 with bf16/fp8/w4a16 profiles.
    - Validated on H100 80 GB and NVL and H200 for all three; RTX 4090 for fp8 and w4a16; RTX 3090/3090 Ti for w4a16. RTX PRO 6000 is not listed and goes down the "compatible-unvalidated" path.
    - K default 1, with a maximum of 15 on H100 and 20 on H200; each K is warmed for CUDA graphs.
    - Input: 16 images (4 cameras × 4 frames, t0 ≈ t−0.3 s) and 16 ego poses. Defaults: T 0.6, top-p 0.98, `max_tokens` 256. Output: `reasoning_text` with no documented skip flag, and 64 points at 0.1 s.
    - **No latency or throughput numbers**, and no performance page (404).
  - **2.0.0 (latest)** serves Alpamayo 2 Super only: H100/H200, ≥ 80 GB, bf16/fp8 plus DFlash2 profiles, vLLM-Omni with prefix caching, no numbers.
- **recipes PR #44** (merged 2026-11-03, `docs/training_efficiency/05_Measurement.md`) [D]:
  - Wall clock comes from log timestamps; GPU-hours = mean wall clock × steps × GPUs, never the median; drop the first ~20 steps.
  - MFU uses 6N per component against the dense peak, HFU uses 8N, and OFU = DCGM tensor activity × clock ratio. CUDA-event decomposition splits a step into arrival / forward / backward / host residual, and event intervals include launch gaps.
  - Relevance: a frozen VLM forward inside the trainer is 2N/token of work that counts as neither useful training compute nor gradient work, so report it separately. OFU uses the same DCGM tensor-active signal the Vast host check validated.
- Recipes #43 (speculative-decode draft models) and #45 (5 Hz 2B inference on DRIVE Thor) are device/student work, deferred to S6/S7.

---

## 5. Reuse and caching — only what the source supports

| Axis | What is actually reusable | Already done? | Ceiling / caveat |
|---|---|---|---|
| (a) Across denoising steps | Prefix K/V for all 36 layers, computed once and cropped back after each step | **Yes**, both paths [S] | Expert tokens' own K/V and all activations depend on x_t and t, so nothing else is reusable. Remaining waste is launch overhead (PR #36's target). |
| (b) Across a GRPO group sharing forced-GT | Within one call, K > 1 samples share one prefill via `batch_repeat_interleave` in (G) [S] (CFG path in (E) also says "vision encoder runs only once"). Across the G rollouts of a group, only the **first post-handoff call** has identical inputs, and only if rendering is bitwise identical across sessions (unestablished, AlpaSim #128). | First part yes; cross-session no | ≤ 1/22 ≈ 4.5% of a rollout's prefill [Δ]. Not worth building. |
| (c) Consecutive calls, sliding history | **ViT per-image outputs**: vision attention is confined to each image by `cu_seqlens` and position embeddings are per-image (`modeling_qwen3_vl.py` L222–243, L716–735), so an image's features (incl. deepstack taps) depend only on its pixels — exact in math, bitwise not guaranteed across batch shapes [H]. **LLM K/V: not reusable exactly**: M-RoPE positions of every image depend on all preceding tokens (`get_rope_index`, L916ff), causal context changes, and "frame k" labels shift. | No | ViT reuse covers 8/16 images at 5 Hz (G) or 12/16 at 10 Hz (E), ≈ 10–14% of non-decode FLOPs [Δ]. Prior art: TensorRT-Edge-LLM embedding cache (per `04` §10). LLM K/V reuse needs FlashDrive-style approximation, which changes numerics (`04` §7 E3: +2.7% minADE6) and would break rollout/replay parity unless replay used the same approximation. |
| (d) Trainer replay, VLM frozen | The prefix K/V of each row is a deterministic function of frozen weights and stored inputs | **No**: the patched `ExpertModelCosmos.forward` reruns `tokenize_for_generation` → `cfm_logprob_sde` → `_vlm_prefill` (ViT + 36-layer prefill) for **every minibatch of every optimization iteration** (`bundle.py` L55–114; pr18 `model.py` L60–111, L321–389; `trainer.py` L301–375). The rollout already computed the same prefix once. | See Candidate 2. Also: rollout gets the NaN retry, but the trainer recomputation does not; a non-finite gradient **raises** `FloatingPointError` (`trainer.py` ≈L508–533). |

---

## 6. Candidates

### Candidate 1 (qualifies): the training inference path and the evaluation inference path are not the same function

- **Beneficiary / workload.** S4 (G9): every claim that a Gym-trained expert "transfers" or "fails to transfer" when evaluated by the S2 instrument.
- **Existing implementation and the specific problem.** NVIDIA's documented workflow trains the expert in (G) and evaluates the converted checkpoint in the AlpaSim `alpamayo1_5` driver (E) [S: converter docstring]. Section 1 shows these differ in five ways at once:
  1. image geometry 320×512 vs 320×576: 2,560 vs 2,880 vision tokens, and a horizontal scale ratio 512/576 = 0.889;
  2. prompt text and **no CoC vs sampled CoC** (≤ 128 tokens, T 0.6) in the prefix the expert attends to;
  3. RoPE start position of the 64 expert tokens (follows prefix length);
  4. 5-step SDE (η 0.2, seeded) vs 10-step Euler ODE (unseeded);
  5. 5 Hz vs 10 Hz replanning.

  The release `Alpamayo1_5` class has **no prefill-only (no-CoC) sampler and no SDE option** (`alpamayo1_5.py` L244, L440 are the only samplers) [S]. The stock evaluator therefore cannot run the trained configuration. Gym cannot train under the evaluation configuration either: CoT replay raises `NotImplementedError` (`bundle.py` L76–82).
  - `rl-foundations.md` already notes the 5-vs-10 Hz issue; `protocol-design.md` allows intentional differences as explicit fields. Neither enumerates factors 1–4 [S: repo grep]. Camera set, camera/frame labels and ego-history length match.
- **Strongest current baseline.** None of PR #36, the quant recipe or NIM addresses this. NIM 1.0.0 always returns `reasoning_text` with no documented skip [D]. Flow-GRPO-style "train SDE, deploy ODE" is a literature precedent for factor 4 only [H: precedent, not shown for this model].
- **Narrow change (instrumentation, not a new method).**
  - (i) Record all five as explicit S4.1 fields.
  - (ii) Add an opt-in "trained-path" mode to the AlpaSim Alpamayo 1.5 driver, implemented as a driver model variant: `prompt_mode: traj_future` (prefill + expert, no `generate`), configurable integrator/steps/noise with per-row seeds, and an optional explicit pre-resize. A Gym-side `input_size: [320, 576]` is the config-only alternative for geometry.
  - Use the mode only as a diagnostic arm unless Huy registers it.
- **Causal mechanism.** RL changes θ to raise E_{c∼p_train, x∼SDE₅(v_θ(·|c))}[R], while S4.6 measures E_{c∼p_eval, x∼ODE₁₀(v_θ(·|c))}[R]. The update transfers only if Δθ's effect on v generalizes across the conditioning shift c_train → c_eval.
  - That shift has five parts: N_v changes by −320 tokens; image column positions scale by 0.889 for the same scene; CoC K/V (≈ 16–128 tokens) is present only at evaluation; RoPE offsets shift; and the plan is executed for 200 ms vs 100 ms between replans.
  - SDE per-step std in normalized action units is [0.200, 0.179, 0.110, 0.073, 0.045] vs 0 for the ODE [Δ].
  - An observed transfer failure is therefore confounded with this path shift. That is a candidate mechanism alongside those listed in S4.5 (reward/eval mismatch, horizon, credit, …).
- **Measurable objective and boundary.** Boundary: policy inference → simulator outcome.
  - Open-loop: for each factor f, cross-configuration divergence D_f = E‖τ_A − τ_B‖ (ADE over 64 waypoints, m) versus the within-configuration sampling spread D_within on identical recorded observations.
  - Closed-loop: the paired S2 score difference between stock and trained-path arms, against same-policy re-run noise.
- **Smallest disproof experiment** (after S2 admission; same asset gates).
  - Base checkpoint only. Open-loop 2³ factorial (geometry × prompt/CoC × integrator) on ≈ 500 recorded development ticks with K = 8 samples each ≈ 32k calls ≈ **2–4.5 GPU-h** [Δ].
  - If every D_f ≤ D_within (threshold is Huy's), the candidate is refuted for the base policy. Repeat once on the first RL checkpoint, because sensitivity may grow with training.
  - Preserves: identical frames, weights and scorer. Omits: closed-loop compounding and the cadence factor, which needs the closed-loop arm. Prediction slot left for Huy.
- **Transfer limits.** Semantic and device-independent, but specific to these pins and Alpamayo 1.5. It says nothing about Alpamayo 2/Super, NIM internals or Thor.
- **Where the benefit disappears or reverses.**
  - All D_f within sampling spread → nothing to fix.
  - The registered S2 gate must stay the stock challenge configuration → the trained-path arm is diagnostic only and cannot replace the verdict.
  - Evaluating a no-CoC configuration is less deployment-relevant, since NIM and the challenge run CoC.
  - Aligning training geometry to 320×576 costs ≈ +12% prefill compute [Δ] and departs from NVIDIA's shipped Gym preset. Whether 320×512 was intentional for 1.5 is unknown [H].
- **Adopting component / fit.**
  - AlpaSim driver (accepts outside PRs; #228 by an external author merged on e2e_challenge 2026-11-03). Re-check NVIDIA's latest release (#224) and #208 first.
  - Gym config or overlay is PhysLoop-local, since AlpaGym accepts no PRs.
  - Fits **S4 as instrumentation** (S4.1/S4.2/S4.6). Making it a registered evaluation arm is Huy's scientific decision. No new stage.

### Candidate 2 (conditional; does **not** qualify under the shipped configuration): frozen-prefix recomputation in the Gym replay trainer

- **Beneficiary / workload.** The Gym trainer GPU during S4, only if Huy registers `grpo_optimization_iterations` μ ≥ 2, or the trainer becomes the critical path (colocated topology, or rollouts sped up).
- **Existing implementation and the specific problem.** See §5(d). The VLM is frozen (`cotrain_vlm` false; pr18 `expert_model.py` L77–79), yet its full prefix (≈ 90.6% of replay FLOPs per row [Δ]) is recomputed every iteration.
- **Strongest baseline.** None public. PR #36 is rollout-only and goes eager under autograd.
- **Narrow change.** Per training step, compute each row's prefix K/V once (no-grad, same kernels). Keep it in pinned host memory (0.403 GB/row; 132 rows = 53 GB for `clrl_test_run` 6 × 22) and copy it per minibatch for iterations 2..μ. Acceptance: new_logprob bitwise equal to recompute.
- **Mechanism.**
  - Per-row cost with recompute is μ(P+E+O); with caching it is P + μ(E+O) + (μ−1)X.
  - Assumed values: P ≈ 166 ms (5.0e13 FLOP at 60% of 503.8 TFLOP/s), E ≈ 21 ms, O ≈ 54 ms (AdamW traffic at 80% of BW), X = 0.403 GB / 25–50 GB/s ≈ 8–16 ms.
  - Trainer speed-up: **1.00× at μ = 1**, 1.45–1.49× at μ = 2, 1.88–1.97× at μ = 4 [Δ; all efficiencies are assumptions].
- **Measurable objective / boundary.** Trainer seconds per replayed row, and iteration wall clock using PR #44's mean-wall-clock definition. Boundary: RL infrastructure only. No outcome change, per §1.
- **Smallest disproof.** One recorded training step at μ = 2, with and without cache: require bitwise new_logprob equality, then compare time per row. Use Gym `perf` scopes to check that the trainer is on the critical path. Drop the idea if the speed-up is < 1.2× or the trainer idles.
- **Where it disappears or reverses.**
  - μ = 1 (every shipped preset) → exactly zero.
  - Rollout-bound pipeline (PR #36's 10B data: trainer ~45 s vs ~63 s other [D]) → savings become idle time.
  - Host RAM: 53 GB vs 97 GiB guest RAM on the tested Vast VM [D, `vast-host-validation`] → spill risk.
- **Transfer.** Datacenter trainer only.
- **Adoption / fit.** PhysLoop-local Gym overlay (no PRs accepted), or NVIDIA's PR #18 (open, internal). It fits S4 infrastructure **only if** μ ≥ 2 is registered. Otherwise do not build it.

### Considered and rejected (with the decisive reason)

- **Enabling or porting the CUDA graph, or quantizing:** excluded by the brief; both are prior art (§4); quantization changes numerics (P-009 ladder needed).
- **ViT cross-tick cache:** exact but ≤ 10–14% ceiling; prior art exists.
- **LLM K/V reuse across ticks:** not exact.
- **Group-shared first prefill:** ≤ 4.5%, and needs deterministic rendering.
- **CPU preprocessing in (E)** (16 full-HD resizes per call on CPU, each frame resized 4×): plausible S2 throughput item but unmeasured [H]. Measure with nsys `osrt`/CPU sampling before proposing.
- **Intermittent non-finite VLM prefill:** NVIDIA documents it as scene-dependent and only reduced by determinism (`alpagym_host/config.py` L111–116). There is no root cause, so no mechanism to propose. See §7 instead.

---

## 7. Notes for S4.2 correctness (engineering, no scientific slot filled)

1. **Rollout vs replay numerics are structurally different even at identical weights.**
   - Rollout: sequential per-step expert forwards (Q = 64, cache append/crop) at an opportunistic batch of 1–4 rows.
   - Replay: one parallel forward over T·64 queries with a block-diagonal mask at `mini_batch` rows.
   - [Δ] For a per-element mean shift δ, the mean-log-density difference has std ≈ (δ/s)/√N and bias −(δ/s)²/2, with N = 640 and s from the SDE stds above. Plausible bf16 δ gives ~1e-4–1e-3 [H magnitude]: harmless next to the 0.8–1.2 clip, but not "exactly 1".
   - For E04, pin `max_batch_size: 1` in the parity cohort, or record batch composition, and derive the tolerance from the measured δ.
2. **`set_deterministic` is best-effort** (`warn_only=True`; the cuBLAS workspace env var is set after process start) [S]. It does not make batch composition deterministic.
3. **Non-finite prefill.** Rollout retries up to 3× (`inference_model.py` L172–199); the trainer recomputation has no retry and raises on a non-finite gradient.
   - Count rollout retries, and record per-row prefix finiteness at replay.
   - A diagnosis capture would use per-module `isfinite` forward hooks behind a flag, plus `compute-sanitizer --tool initcheck/racecheck` on a fixed reproducer.
   - Hypotheses only [H]: weight-sync/forward overlap, nondeterministic kernels, bf16 overflow.
4. **Docs/04 corrections.**
   - BF16 dense peak is printed (503.8 TFLOPS, whitepaper v1.1 Table 4).
   - recipes #42/#44/#45 are merged as of 2026-11-03/05.
   - The (E) driver's CoC budget is `tokens_per_future_traj` = 128, not 64 (the HF config overrides the class default).
   - Gym's vision input is 320×512 / 2,560 tokens, not the 320×576 / 2,880 in `04` §2.

---

## Evidence log

| Claim | Source URL (SHA/version) | Date read | Evidence type | Applicability | Uncertainty | Consequence |
|---|---|---|---|---|---|---|
| e2e_challenge head moved to 297b4d00; data/README only | https://github.com/NVlabs/alpasim/commit/297b4d00cfdd2c4f636b9d9bcee7a85c699174c0 | 2026-11-05 | source-inspected (file list) + doc-stated (msg) | S2 pin | first page of file list only | Re-pin S2 manifest; driver unchanged |
| recipes main moved to c53aa09; #44 (11-03), #42/#43/#45 (11-05) merged | https://github.com/NVlabs/alpamayo-recipes/commits/c53aa09f503eae54a2397e4d57c22cf299c111b7 | 2026-11-05 | source-inspected | docs freshness | — | Update 04/nvidia-stack "open" statuses |
| PR #18 head unchanged, open, dirty | https://github.com/NVlabs/alpamayo-recipes/pull/18 (1d99fc50) | 2026-11-05 | source-inspected | Gym dependency | — | Pin still valid |
| Driver forces sdpa, bf16, K=1, T 0.6, top-p 0.98 | https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/driver/src/alpasim_driver/models/alpamayo1_5_model.py#L19-L80 | 2026-11-05 | source-inspected | S2 eval path | runtime may differ if config overridden | Manifest fields |
| Driver CoC via generate, max_new_tokens = tokens_per_future_traj (=128) | https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/alpamayo1_5.py#L288-L321 ; https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/main/config.json (HF sha 7aba8293) | 2026-11-05 | source-inspected + doc (config) | (E) | driver installs alpamayo1.5 unpinned | Decode cost ∝ N_CoC ≤ 128 |
| Expert = 36-layer Qwen3 decoder, hidden 2048, MLP 8256, 8 KV heads; joint attention over VLM per-layer KV | alpamayo1_5.py L101–113, L357–387 (36aeb4c5); HF config.json | 2026-11-05 | source-inspected + derived | both paths | — | Conditioning = prefix KV, no cross-attn |
| Param total 11,078,526,194 reproduced exactly from configs | HF API nvidia/Alpamayo-1.5-10B; Qwen3-VL-8B config.json; derive.py | 2026-11-05 | derived (known answer) | architecture reading | config of Cosmos-Reason2-8B assumed = Qwen3-VL-8B text/vision | Validates FLOP model inputs |
| (E) images → 320×576, 180 tok/img; (G) 320×512, 160 tok/img | helper.py L23–25, L190–199 (36aeb4c5); alpagym buffers.py L237–253, alpamayo_r1.yaml L14 (972d160e) | 2026-11-05 | source-inspected + derived (smart_resize) | train vs eval | processor resample filter not verified; confirm via logged image_grid_thw | Candidate 1 factor 1 |
| Gym skips CoC; prompt "output the future trajectory." ends at traj_future_start | alpagym inference_model.py L123–126; tokenize_online.py L76–78; conversation.py L18–22 (972d160e) | 2026-11-05 | source-inspected | training path | — | Candidate 1 factor 2 |
| Gym SDE η 0.2, 5 steps (4 smoke), seeded; (E) Euler ODE 10 steps | alpamayo_r1.yaml L34–38; smoke yaml L14–19 (972d160e); alpamayo1.5 flow_matching.py L32–50, L171–191 | 2026-11-05 | source-inspected | both | — | Candidate 1 factor 4 |
| SDE per-step std [0.200,0.179,0.110,0.073,0.045] | pr18 flow_matching.py L590–658 (1d99fc50) | 2026-11-05 | derived | Gym | formula transcription | Parity tolerance math |
| 5 Hz Gym vs 10 Hz e2e; force-GT 1.6 vs 1.7 s | smoke/test-run yaml L27–30 (972d160e); base_config.yaml L275–293 (e121e591) | 2026-11-05 | source-inspected | both | — | Factor 5 (already noted in rl-foundations) |
| Compute time never enters dynamics; delay = simulated DelayBuffer, ⌈d/Δt⌉Δt | events/policy.py L59–70, L135–141; events/controller.py L123–127; delay_buffer.py L35–49 (e121e591; same buffer at 46b401be) | 2026-11-05 | source-inspected + derived | all AlpaSim paths | — | Speed-ups only buy wall clock |
| Controller re-anchors reference at its first timestamp | controller system.py L332–359 (e121e591) | 2026-11-05 | source-inspected | executed path | — | Delayed plans are frame-consistent |
| Vision attention per image (cu_seqlens, SDPA split loop); M-RoPE positions depend on preceding tokens | https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py L222–243, L727–735, L916ff | 2026-11-05 | source-inspected | both (pinned transformers 4.57.1) | actual installed version per lockfile unverified | ViT exact reuse yes; LLM KV reuse no |
| Trainer recomputes VLM prefill every minibatch × iteration | alpagym bundle.py L55–114; trainer.py L301–375 (972d160e); pr18 model.py L60–111, L321–389 | 2026-11-05 | source-inspected | S4 trainer | — | Candidate 2 |
| Replay prefill = 90.6% of per-row trainer FLOPs; KV 0.403 GB/row | derive.py | 2026-11-05 | derived | S4 trainer | prefix length ±3%; FSDP dtype unverified | Candidate 2 sizing |
| Trainer raises on non-finite grad; rollout-only NaN retry | trainer.py ≈L508–533; inference_model.py L172–199; alpagym_host/config.py L111–116 (972d160e) | 2026-11-05 | source-inspected | S4 reliability | root cause unknown | S4.2 instrumentation |
| set_deterministic is warn-only, env var set at runtime | alpagym policies/determinism.py L22–35 (972d160e) | 2026-11-05 | source-inspected | Gym | effect on cuBLAS handle timing [H] | Do not claim determinism from flag |
| Opportunistic 1–4 row batching across sessions | alpagym inference/inference_engine.py L99–157 | 2026-11-05 | source-inspected | Gym rollout | batch-invariance of kernels [H] | Pin B=1 for parity cohort |
| PR #36: opt-in graphs, signature incl. KV shape, max_graphs 4 no eviction, eager under grad | https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/diffusion_expert_cuda_graph.py L153–260 | 2026-11-05 | source-inspected + derived | rollout only | — | CoC path gets limited graph coverage |
| PR #36 10B AlpaGym: −15.5% call, +56% trainer-step, trainer 41.9→44.7 s of 167.7→107.3 s; 160/160 bitwise | https://github.com/NVlabs/alpamayo1.5/pull/36 (body + comment) | 2026-11-05 | doc-stated | NVIDIA-internal Gym integration | n=1 paired sample, preliminary | Trainer not critical path there |
| RTX PRO 6000 WS: BF16 503.8 dense / FP8 1,007.6 dense TFLOPS; 1,792 GB/s; 128 MB L2; PCIe Gen5 | https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf Table 4 pp.46–47 (sha256 1ca55d8f…); https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ | 2026-11-05 | doc-stated | roofline | boost-clock peaks; Server edition differs | Ridge 281 / 562 FLOP/B; corrects 04 §3 |
| Quant recipe: ModelOpt PTQ fp8/nvfp4/w4a8/auto; mtq.compress; eager eval minADE + ms/clip; no numbers | https://github.com/NVlabs/alpamayo-recipes/blob/c53aa09f503eae54a2397e4d57c22cf299c111b7/recipes/alpamayo1_5_quant/README.md ; quantize.py L255–340; eval.py L90–105 | 2026-11-05 | doc-stated + source-inspected | prior art | real low-bit GEMM execution [H] | Not a candidate |
| NIM 1.0.0: Alpamayo 1.5, bf16/fp8/w4a16, H100/H200/RTX 4090/3090; CUDA graphs per K; reasoning_text returned; no latency numbers | https://docs.nvidia.com/nim/alpamayo/1.0.0/overview ; …/1.0.0/support-matrix ; …/1.0.0/api-reference ; …/1.0.0/release-notes (dated 2026-08-27) | 2026-11-05 | doc-stated | prior art | RTX PRO 6000 unvalidated | No public latency baseline |
| NIM 2.0.0: Alpamayo 2 Super only, H100/H200, vLLM-Omni prefix caching | https://docs.nvidia.com/nim/alpamayo/overview ; …/support-matrix | 2026-11-05 | doc-stated | out of scope | — | Not applicable to 1.5 study |
| PR #44 measurement definitions (mean wall clock, MFU/HFU/OFU, CUDA-event split) | https://github.com/NVlabs/alpamayo-recipes/blob/f600b20fbdbe8246e4d25d53fa36e31bb25f6550/docs/training_efficiency/05_Measurement.md | 2026-11-05 | doc-stated | S4 G7 telemetry | — | Use for trainer accounting |
| Gym→AlpaSim alpamayo1_5 driver is the documented evaluation route | alpagym scripts/convert_alpagym_checkpoint_to_inference.py L4–9 (972d160e) | 2026-11-05 | source-inspected (docstring) | S4.6 | — | Candidate 1 is on the intended path |
| Release Alpamayo1_5 has only CoC samplers (no prefill-only/SDE) | alpamayo1_5.py L244, L440 (36aeb4c5) | 2026-11-05 | source-inspected | (E) | — | Parity mode needs driver code |
| Policy-call bounds: (G) ≈117 ms; (E) ≈281 ms at 16 CoC tokens; expert step 2.8 ms memory-bound at B=1 | derive.py | 2026-11-05 | derived | RTX PRO 6000 WS, bf16 | peak-based; eager launch overhead excluded | Prefill dominates (G); decode+prefill (E) |
