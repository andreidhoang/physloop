# L7 — Thor serving, HIL and rollout performance for JR2010863 (I2 "performance" half, Q4, X3, D2)

**As of:** 2026-10-27. **Scope:** lane L7 only (S6–S7 facts; S6's serving techniques applied to AlpaSim's policy-inference step in S2–S5; exceptional C++ in S6–S7; re-check of P-007…P-010).
**Method:** read-only. GitHub (MCP read tools and raw files), Hugging Face API, NVIDIA docs and news pages, arXiv HTML. Nothing was signed into, downloaded from gated sources, posted, rented or bought. The source code read was public AlpaSim / alpamayo1.5 / TensorRT-Edge-LLM files, treated as data. Arithmetic: `scratchpad/calc/l7_arith.py` and `scratchpad/calc/delay_ka.py` (reproducible with `python3 -I <file>`).
**Baseline ledgers** (not re-researched; deltas reported in §3): `docs/research/2026-10-25-teacher-to-thor/verification.md` §B, §C; `docs/research/2026-10-03-verification/notes/deployment_hardware_compute.md`; `docs/04_CORE_PLAN_TECHNICAL.md` §3, §4, §9; `docs/02_PHYSLOOP_PLAN.md` S6, S7; `docs/DECISIONS.md` P-007…P-010.
**Conventions:** P = primary source, S = secondary. Confidence: High / Medium / Low. "Inference" marks my reasoning, not a source statement. Prediction and tolerance slots are written `____`; no value is proposed for any of them. Price basis is the plan's: vast.ai on-demand, 2026-10-25, RTX PRO 6000 Workstation $1.00 cheapest / $1.60 median, H100 SXM $1.93 / $2.40. A saving is negative.
**Structure note:** the caller's six-section structure is used. The researcher template's discipline (Takeaway / Cited Findings / Inferences / Gaps, inline URL on every fact) is kept inside §2, per key question.

---

## 1. Findings table

**Headline (inference from §2).** The AlpaSim driver that every S2–S5 sweep runs is the stock **eager PyTorch/HF path at batch 1**:
- `max_batch_size: 1`;
- no Alpamayo adapter overrides `predict_batch`;
- NVIDIA's own expert CUDA graph is not enabled.

NVIDIA's own A1.5 evaluation topology gives the driver 7 of 8 GPUs. So the policy-inference step is where S2–S5's ~884–2,295 RTX PRO 6000 GPU-hours go. That is exactly where S6's serving skills apply.

The safest lever is already upstream, opt-in and bitwise-verified by NVIDIA: the alpamayo1.5 PR #36 expert CUDA graph. The largest lever, GPU batching across rollouts, changes numerics and RNG streams. It therefore needs the P-009 ladder plus a paired verdict. That verdict doubles as a known-null validity test of the gate.

S7 shrinks, because AlpaSim already ships an external-driver mode and a `planner_delay_us` delay buffer. The freed hours go to the C++ driver server, which is the Q4 evidence.

Sweep inventory used below (plan v4 §3, RTX PRO 6000):

| Stage | Sweeps |
|---|---|
| S2 | 4 |
| S3 | 3–4 |
| S4 | ≤ 10 evaluation sweeps |
| S5 | ~6 |
| S6 | ~3 |
| **Total** | **26–27 sweeps**, ×34–85 GPU-h = **884–2,295 GPU-h = $884–3,672** |

The 22–23 sweeps after S2 total 748–1,955 GPU-h.

| ID | JR2010863 lines | Stage · deliverable | Kind | What changes in the stage card | GPU-h and $ delta (price basis: vast.ai, 2026-10-25) | Hours delta | Gate | Owner | Sources |
|---|---|---|---|---|---|---|---|---|---|
| **L7-1** Rollout performance profile | I2-perf, D2, D4, X3, X2 | S2 · G7 (+G1 manifest) | (a) | **Build:** expand "first measured throughput" into a per-service GPU-time split (driver / renderer / physics / controller / queue wait), using AlpaSim Prometheus plus `nsys` with NVTX. Add a per-call CUDA-event split of the driver: encode / prefill / decode / expert / host residual. GPU-h is computed from the **mean** s/step. **★:** Huy diagnoses the profile. **Acceptance:** every sweep's GPU-h is derived from the measured profile, with the manifest. **Prediction:** driver share *f* of GPU-time per step: `____`; top driver stage: `____`. **Kill:** none new. **Cost:** a profiling subset | +2–4 GPU-h ⇒ +$2–6 | +4–6 | P-005 (already required for S2) | Agent: harness, plots. ★ Huy: diagnosis, predictions | S02, S05, S10, S11, S19, S27b |
| **L7-2** NVIDIA's expert CUDA graph in the AlpaSim driver, behind a bitwise gate | I2-perf, X3, D2, Q4 (Python) | S2 → S3–S6 · G7 | (a) | **Build:** a driver flag that calls `enable_diffusion_expert_cuda_graph()` after model placement. Pin and record the resolved `alpamayo1_5` commit (≥ `24179cf`, PR #36). Record graphs on/off, `max_batch_size`, replicas and `force_determinism` in the manifest. **Acceptance:** eager vs graph driver outputs bitwise identical (max abs diff = 0) on every compared tick, with `force_determinism=true` and the same inference seeds; the S2 negative control still passes; otherwise the lever is not used. **Sweep mode (★ protocol choice).** If the sweeps also run with `force_determinism=true`, the bitwise claim transfers per run; that mode's throughput cost is unmeasured, and L7-1 measures it. If they run at AlpaSim's default `false`, the claim is "same kernels ⇒ same output distribution": established under deterministic mode, assumed for the non-deterministic one, and the manifest must say so. **Prediction:** sweep GPU-h saved: `____`. **Upstream:** AlpaSim PR exposing the flag, only after re-checking NVIDIA's post-10-31 sync | **S2–S5 (22–23 Alpamayo-driver sweeps; S2's TransFuser sweep excluded): −85 to −266 GPU-h ⇒ −$85 to −$425** at $1.00–1.60. Adding S6's ~3 sweeps (26–27 in all): −100 to −312 GPU-h (−$100 to −$499). That is −11.3% to −13.6% per A1.5 sweep. Bitwise test ≈ +1 GPU-h | +3–5 | none beyond P-005 | Agent: flag, test. Huy: review | S01, S04, S06, S16, S17, S18 |
| **L7-3** Driver replicas per GPU (config only) | I2-perf, D2, Q5 | S2 · G7 | (a) | **Build:** a throughput arm of 1 vs 2 (vs 3) driver replicas per RTX PRO 6000. NVIDIA's A1.5 evaluation topology runs 2 per GPU; the challenge preset runs 4 at 16 GiB. **Acceptance:** per-rollout outputs are unchanged, checked bitwise on a subset as in L7-2. **Prediction:** throughput ratio, 2 vs 1 replica: `____` | +1–2 GPU-h ⇒ +$1–3. The saving is unmeasured (no public A1.5 number) | +1–2 | none | Agent. ★ Huy: prediction | S05, ledger 10-03 Q2 (challenge preset) |
| **L7-4** GPU-batched `predict_batch` for the Alpamayo adapter (promote `04` §7 E2) | I2-perf, X3, D2, D3 | S2 (equivalence) → S3–S6 · G7, G3 (known-null pair) | (c) | **Build:** stacked forward with padded prompts and masks. Use a per-sample `torch.Generator` for top-p sampling and flow noise, seeded from each session's `inference_seed`, so per-session streams survive batching. **Acceptance:** P-009(iv) ladder: E0a B=1 path bitwise-unchanged; E0b per-layer vs FP32 at B=k; E0c action error in metres, B=1 vs B=k with injected shared noise; E1 minADE6; **E2 paired closed-loop verdict, batched vs unbatched driver on the same checkpoint**. This is a known-null pair that the gate must not flag. **★:** Huy owns the E0c/E1 tolerances, the E2 margin (equivalence or NI) and the predictions. **Prediction:** E2's preserved 22-Sep prediction stands; new slots: per-sample driver speed-up at B = k: `____`; E2 verdict: `____`. **Kill:** E2 not equivalent within Huy's margin → stay at B = 1 | +1 equivalence sweep: +34–85 GPU-h (+$34–136). Then on S3–S5 (19–20 sweeps): **−263 to −964 GPU-h (−$263 to −$1,542)**; on S3–S6 (22–23 sweeps): −304 to −1,108 GPU-h (−$304 to −$1,773). Both hold **only if** Alpamayo's batch speed-up matches published non-Alpamayo analogs (s = 2.26–2.84). Not additive with L7-2 | +10–14 | P-005 (extra sweep within tier) | Agent: adapter, harness. ★ Huy: tolerances, margin, predictions | S02, S03, S04, S19, ledger 10-03 Q3 (PhyAI) |
| **L7-5** Static KV cache + CUDA-graphed VLM decode in the PyTorch driver (conditional) | I2-perf, X3 | S3–S5 · G7 | (c), only if L7-1 shows decode is the top driver stage after L7-2 **and** L7-4 is not adopted | **Build:** HF static cache plus graph capture of the decode step. **Acceptance:** the same ladder and paired verdict as L7-4. **Prediction:** `____` | +34–85 GPU-h (equivalence sweep), then −70 to −220 GPU-h incremental over L7-2 (FlashDrive's system-opt s = 1.397 vs 1.184) | +8–12 | P-005 | Agent. ★ Huy | S18, S19 |
| **L7-6** "One C++ driver, two targets": S6a's runtime serves AlpaSim sweeps | I2-perf, Q4, X3, D5 | S6a (P-007) → S5/S6 sweeps · G6, G7 | (c) | **Build:** the S6 runtime (TensorRT C++ engines or the TensorRT-Edge-LLM C++ runtime, plus the C++ `EgodriverService` of L7-9), built first on the rented RTX PRO 6000 (Edge-LLM supports x86 SM120 at "Developer" level, with TRT 10/11 wheels). It plugs into AlpaSim via `driver_source=external_static` for the student sweeps. **Acceptance:** ladder E0a–E2 against the PyTorch driver, injecting the reference initial noise as Edge-LLM's own pi0.5 parity procedure does, plus a paired closed-loop verdict. A cloud FP8 verdict never stands in for the Thor-engine verdict. **Prediction:** `____` | +34–85 GPU-h (equivalence sweep) plus S6a's few GPU-h. The saving is unmeasured | +12–20 now; part recovered in S6b (inference) | **P-007 adopted** + P-005 | Agent: plumbing. ★ Huy: kernels, latency model, verdicts | S21, S22, S24, S13, S12 |
| **L7-7** TRT-11-ready C++ (strong typing, `IPluginV3`) | Q4 | S6 · G6 | (a) | **Build:** engines as strongly typed networks with explicit Q/DQ from ModelOpt (no builder precision flags). Custom ops, including any fused flow-step op, as `IPluginV3` / `IPluginCreatorV3One`. CI builds against TRT 10.16 (Jetson) and TRT 11.x (x86). **Acceptance:** the same source builds on both. No speed claim | 0 | +0–2 | P-002 for the device part only | ★ Huy: plugin and kernel design. Agent: build matrix | S35, S20, S21 |
| **L7-8** Use AlpaSim's own external-driver mode and `planner_delay_us` | X2, D3, I2 | S7 · G7 | (a) | **Build:** Thor serves `EgodriverService` through `driver_source=external_static` (no AlpaSim fork). Constant-delay arms use `planner_delay_us` at the measured model latency and at glass-to-action (P-009 ii). Add per-step trace-driven delay as an AlpaSim PR **only if** Huy's rule says the constant arms are insufficient. **Known answer (CPU):** at Δt = 100 ms, delays of 139, 171 and 200 ms give the same applied-plan sequence (2 ticks late). **★ (unchanged):** lockstep semantics, injection design, TOST margin. **Prediction:** score lost at the injected delay: `____` | 0 (S7 stays ~$200) | **−4 to −8** (mechanism exists) | P-002 | ★ Huy. Agent: configs | S07, S08, S09, S10, S12, S13 |
| **L7-9** Production-grade C++ driver server on Thor | Q4, D2, X2 | S7 (built during S6) · G7 | (a) | **Build:** a C++ gRPC `EgodriverService` with: nvJPEG decode into pre-allocated pinned and device buffers; TensorRT execution on dedicated streams with a captured fixed-shape step; CUDA-event per-stage timing plus a host monotonic clock for glass-to-action; bounded lock-free SPSC queues; the deterministic `inference_seed` contract; Prometheus histograms. Tests: mock-engine unit tests, a golden bitwise replay, `compute-sanitizer` (memcheck / racecheck / synccheck / initcheck), ASan / UBSan / TSan host builds, clang-tidy. **Acceptance:** p50/p99/p99.9 per stage on Jetson AGX Thor (device, JetPack and TRT named; "real engine" label); zero sanitizer findings; golden replay bitwise | 0 | +6–10 (funded by L7-8's −4 to −8) | P-002 | Agent: plumbing, tests. ★ Huy: latency model | S12, S20, S26, S41 |
| **L7-10** Measure the RL framework's own performance in S4 (AlpaGym telemetry, no internals touched) | I2-perf, D1, D4, X1 | S4 · G9 | (a) | **Build:** record the public AlpaGym's opt-in `perf/` telemetry (trainer idle share, rollout supply rate) in the S4 manifest for every arm. Report it next to NVIDIA's published 10B number (75% rollout-wait eager; 58% with graphs; one paired sample). **Prediction:** the preserved 22-Sep headline prediction 1 / E1 is the slot. No new value | 0 (same runs) | +2–3 | P-005 (S4 already) | Agent: parsing. ★ Huy: reconciliation | S16, ledger 10-25 B1, `04` §5 |
| **L7-11** Service-balance model of a rollout fleet (stand-in for cluster scale) | D2, D4, Q5, X3 | S2 · G7 | (b) | **Build:** a design doc extending `04` §4's balance equation to AlpaSim's services: N_drv·μ_drv ≈ N_rnd·μ_rnd, using L7-1's per-service rates. Validate on one 2-GPU AlpaSim run (`topology/2gpu.yaml`) against NVIDIA's 7:1 A1.5 evaluation topology. **Prediction:** driver:renderer GPU ratio at balance: `____` | +2–8 GPU-h ⇒ +$2–13 | +3–4 | P-005 | ★ Huy: model. Agent: run | S05, S11 |

**Kind rule applied.**
- Rows marked (a) each add ≤ ~10 h inside an existing stage deliverable. They net to +8 to +24 h after L7-8's saving.
- Whether any of them exceeds its stage's effort estimate is Huy's call at KC-1. Any row he rules over budget becomes (c).
- L7-4, L7-5 and L7-6 are already (c), because each adds a sweep or more hours than an existing deliverable absorbs.

**Totals.**
- **Recommended core set** (L7-1, L7-2, L7-3, L7-7, L7-8, L7-9, L7-10): **+8 to +24 h** net on ~320 h. All of it stays within the existing stages and gates.
  - **Across S2–S5, which the caller asked for:** **−79 to −263 GPU-h net** (L7-2's −85 to −266, minus L7-1/L7-3's +3 to +6 of profiling), i.e. **about −$79 to −$421** (low end at $1.00, high end at $1.60).
  - **Including S6's sweeps:** −94 to −309 GPU-h (about −$94 to −$494).
  - Price basis: vast.ai RTX PRO 6000 Workstation $1.00–1.60, 2026-10-25.
- **Gated options:**
  - L7-4: +10–14 h. On S3–S5 it nets −178 to −930 GPU-h after its equivalence sweep (published-analog basis);
  - L7-6: +12–20 h, P-007;
  - L7-5: conditional.
- The (b) row L7-11 adds +3–4 h.

---

## 2. Facts ledger

Source IDs used in §1 are listed at the end of this section. All accesses were on **2026-10-27** unless a row says otherwise.

### Q1 — Thor software and support (as of 2026-10-27)

#### Takeaway
Nothing on the Jetson side moved since the 10-25 ledger:
- JetPack 7.2.1 is still the newest for Jetson AGX Thor;
- TensorRT-Edge-LLM is still v0.11.0 and exports only Alpamayo-R1-10B in FP16;
- PR #260 is still open, unreviewed and awaiting CI vetting;
- recipes PR #45 is unchanged;
- Edge has no public weights, and NVIDIA states no camera, reasoning or precision configuration for its "10Hz" claim.

**New this check:** TensorRT 11.4.0 formally excludes JetPack, but says a later 11.x will restore it. The 11.x API changes (strong typing, `IPluginV3`) are what S6's C++ should target now.

#### Cited findings

| # | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| 1.1 | The JetPack archive's newest Thor entries are 7.2.1 (L4T 39.2.1), 7.2 (L4T 39.2), 7.1 (L4T 38.4) and 7.0; nothing newer is listed | [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) (S32) | P | High |
| 1.2 | JetPack 7.2.1 ships CUDA 13.2.2 and TensorRT 10.16.2 (ledger fact, 10-25; not re-fetched on 10-27) | [JetPack 7.2.1](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1) | P | High (10-25) |
| 1.3 | TensorRT "Latest" is **11.4.0**; the docs were last updated 2026-10-05 | [TRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) (S34) | P | High |
| 1.4 | The TRT 11.4.0 Jetson migration page says JetPack is "not supported in TensorRT 11.4.0", that Jetson must stay on a TRT 10.x release supported by its JetPack, and that the migration steps "apply only after JetPack support returns in a later TensorRT 11.x release" (no date) | [TRT 10→11 Jetson migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html) (S35) | P | High |
| 1.5 | The same page says: TRT 11.x requires "CUDA 13.2 Update 2 or later"; TRT 10.x engines run on the 11.x runtime; all precision-enabling builder flags are removed (strong typing); `IPluginV2DynamicExt` and `IPluginCreator` are deprecated, with migration to `IPluginV3` + `IPluginCreatorV3One` | S35 | P | High. A search snippet of another 11.x page said "Update 1"; the direct fetch of the latest page says Update 2 |
| 1.6 | DRIVE side: the newest DriveOS found is **7.2.5 EA**. Its announcement cites CUDA 13.3. A TensorRT 11.0.1 release-notes PDF is labelled for DriveOS 7.2.5 and dated 2026-07-02. The 11.4.0 DriveOS migration page lists DRIVE Thor on DriveOS **7.2.6+** for 11.x, while other pages say 7.2.5+. No public 7.2.6 announcement was found | [DriveOS 7.2.5 announcement](https://forums.developer.nvidia.com/t/announcement-from-nvidia-nvidia-driveos-7-2-5-now-available/379371); [DriveOS 7.2.5 TRT notes](https://developer.nvidia.com/docs/drive/drive-os/7.2.5/public/NVIDIA-TensorRT-Release-Notes.pdf); [TRT DriveOS migration](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/api/migration/tensorrt-10x-to-11x-driveos.html) (S36–S38) | P, read via a search summary only | Medium. **Conflict** 7.2.5 vs 7.2.6 |
| 1.7 | TensorRT-Edge-LLM's newest release is **v0.11.0 (2026-09-29)**. It added: x86-64 and AArch64 wheels; IGX Thor (SM110 iGPU + SM120 dGPU); experimental pi0.5 FP16 policy inference; opt-in in-flight batching for the experimental server; "Thor-optimized NVFP4-A16 dense and routed-MoE kernels" | [Edge-LLM releases](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases) (S20) | P | High |
| 1.8 | Edge-LLM v0.10.1 (2026-09-03) added: end-to-end FP8 ViT attention and DART visual-token pruning; a media artifact cache retaining ViT embeddings; "Hopper support for development"; TensorRT-RTX as a backend; QNX cross-compilation; mock-engine unit tests. It also fixed #166 (the Alpamayo-R1-10B export assertion) | S20 | P | High |
| 1.9 | Edge-LLM v0.10.0 (2026-08-12) added: Cosmos3-Edge reasoning and experimental policy/action generation; an experimental ONNX-less direct engine builder; media-aware KV reuse for VLM requests; Qwen3-Omni full-loop CUDA graph | S20 | P | High |
| 1.10 | Edge-LLM's supported models include, under "Action and Multimodal Reasoning": Alpamayo (`alpamayo_r1`, nvidia/Alpamayo-R1-10B); Cosmos3-Edge; pi0.5. It also lists Cosmos-Reason2-2B/8B as VLMs. The Alpamayo example states "Only FP16 is supported for Alpamayo export", with engines built on the Thor device. **No Alpamayo 1.5 and no Edge** | [supported-models.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/supported-models.md); [alpamayo.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/alpamayo.md) (S22, S23) | P | High |
| 1.11 | Edge-LLM support matrix: Jetson Thor, JetPack 7.2, CUDA 13.2, platform TRT, built on device (**Official**). DRIVE Thor, DriveOS 7.2, CUDA 13.3 (**Official**). **x86-64 Linux GPU (Developer)**. Wheels for x86-64 SM86/SM100/**SM120** with TRT 10, and SM80/86/100/**120** with TRT 11 | [support-matrix.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/support-matrix.md) (S21) | P | High. The DRIVE Thor wheel row says "platform TensorRT 10", which **conflicts** with 1.6 (DRIVE on TRT 11) |
| 1.12 | External PR #260, "feat: add Alpamayo fused gate/up plugin for SM110", is **open**: created 10-21, last updated 10-21, `mergeable_state: blocked`. Its only comment is the copy-pr-bot notice that it needs vetting before CI runs. The body says it carries "the reviewed plugin implementation" with dynamic-shape, serialization, CUDA graph and input-contract tests | [PR #260](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260) (S25) | P | High |
| 1.13 | alpamayo-recipes PR #45 ("5Hz inference code for 2B AM1.5 (and AM1)") is open, last updated 2026-10-21; #42 and #43 are open. New: PR #44 (docs, training-efficiency "Measurement" chapter) is open, updated 10-22 | [recipes PRs](https://github.com/NVlabs/alpamayo-recipes/pulls) (S27) | P | High |
| 1.14 | Hugging Face lists 3 nvidia Alpamayo models: Super (05-27), 1.5 (03-03), R1 (2025-11-22). `nvidia/Alpamayo-2-Edge`, `Alpamayo2-Edge`, `Alpamayo-2-Edge-2B` and `Alpamayo2-Edge-2B` return 401 | [HF API](https://huggingface.co/api/models?author=nvidia&search=alpamayo) (S28) | P | High |
| 1.15 | The press release (2026-10-21) says Edge is "a 2-billion-parameter reasoning model built on NVIDIA Cosmos", "optimized for the NVIDIA DRIVE AGX Thor™ platform", "designed to run at 10Hz". It states **no** camera count, runtime reasoning, precision or engine | [press release](https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion) (S29) | P | High |
| 1.16 | The 10-21 blog says Edge is "optimized for the NVIDIA DRIVE AGX Thor platform" and available "later this month". It gives no size, no 10 Hz figure, no configuration and no licence. The Alpamayo page says "2B VLA for in-car reasoning" with no configuration | [blog](https://blogs.nvidia.com/blog/open-models-europe-industry/); [Alpamayo page](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/) (S30, S31) | P | High |
| 1.17 | `nvidia/Cosmos3-Edge` has 3.86B parameters (BF16 safetensors), OpenMDW-1.1, created 2026-07-01. So "Cosmos3-Edge" is **not** the 2B Edge as-is | [HF API Cosmos3-Edge](https://huggingface.co/api/models/nvidia/Cosmos3-Edge) (S44) | P | High (size). What Edge is built on is unknown |
| 1.18 | Jetson FAQ: Jetson AGX Thor Developer Kit "$5499"; T5000 "$4999 (1KU+)"; T4000 "$2999 (1KU+)". No date or note | [Jetson FAQ](https://developer.nvidia.com/embedded/faq) (S33) | P | High |
| 1.19 | Stock: no dated US availability found. An RS UK listing shows units "ready to ship" but is undated | [RS UK](https://uk.rs-online.com/web/p/processor-development-tools/0606864) (search summary only) | S | Low |
| 1.20 | Model Optimizer: **0.47.0 (2026-09-23)** stable; **0.48.0rc1 (2026-10-17)** pre-release. The plan pins 0.43.0 | [ModelOpt releases](https://github.com/NVIDIA/Model-Optimizer/releases) (S39) | P | High |
| 1.21 | FlashRT has no Alpamayo PR after #229: PRs #235–#242 (10-22 → 10-27) cover GR00T, X-VLA, pi0.5 on AMD, and "Flint NVFP4 W4A16 … on SM120" | [FlashRT PRs](https://github.com/flashrt-project/FlashRT/pulls) (S43) | P | High |

#### Inferences
- **No official Alpamayo product path on Jetson AGX Thor exists for 1.5 or Edge.** The only NVIDIA-documented Alpamayo engine path on any Thor is Edge-LLM's R1-10B FP16 example. (The 10-25 ledger B6 already records the Sep-07 "not available for AGX Thor" quote, about the NIM.)
- **Edge's 10 Hz configuration is unknowable today.** Given PR #45's 171 ms FP8 for NVIDIA's 2B students on DRIVE, a 10 Hz Edge plausibly runs with reasoning off or shortened, fewer cameras or frames, or a faster runtime. That is an inference; resolve it at KC-E.
- **S6 code should target TensorRT 11's API now.** Jetson's next TRT major is announced but undated, and DRIVE is already on 11.x.

#### Gaps
- Edge's 10 Hz camera / reasoning / precision configuration: unstated in every NVIDIA source checked (S29–S31).
- Whether a DriveOS 7.2.6 exists publicly: DRIVE downloads need program membership.
- A dated US dev-kit stock or lead time.
- JetPack 7.2.1's CUDA/TRT versions were not re-fetched on 10-27 (they rest on 10-25).

### Q2 — Holoscan, HIL and how to build S7's lockstep and latency injection

#### Takeaway
AlpaSim already contains the two S7 mechanisms:
- an **external driver** endpoint (`driver_source=external_static` / `external_dynamic`);
- a constant **planner delay** (`planner_delay_us`, a `DelayBuffer`), which NVIDIA's own Alpamayo configs set to 200 ms until 2026-03-24.

No NVIDIA HIL reference design with AlpaSim or DRIVE Sim was found. Holoscan SDK 5.0.0 (2026-10-21) is a ground-up rewrite with the right real-time primitives. It is too new, and too sensor-centred, to be S7's base.

#### Cited findings

| # | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| 2.1 | `SimulationConfig.planner_delay_us: int = 0`. The comment says it "models time delays from image capture to planner output to controller". `event_loop.py` builds `DelayBuffer(self.unbound.planner_delay_us)` | [config.py](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/config.py); [event_loop.py](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/event_loop.py) (S07) | P | High |
| 2.2 | `DelayBuffer.item_at(t)` pops queued plans while the *next* plan's age is ≥ `delay_us`, then returns the oldest remaining plan. It is a fixed delay for all items | [delay_buffer.py](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/delay_buffer.py) (S08) | P | High |
| 2.3 | CHANGELOG, "Alpamayo 1.5 driver support (24.03.26)": `planner_delay_us` now defaults to 0 everywhere, and the legacy `alpamayo_runtime_configs` file "which set 200ms delay" was removed | [CHANGELOG](https://github.com/NVlabs/alpasim/blob/main/CHANGELOG.md) (S09) | P | High |
| 2.4 | External drivers: `driver_source=external_static` sets a fixed driver address at startup. `external_dynamic` supplies drivers per request through `SimulationRequest.available_drivers` and `n_concurrent_per_driver`. In both, Wizard neither launches nor owns the driver process | [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/main/docs/OPERATIONS.md); [driver_source configs](https://github.com/NVlabs/alpasim/tree/main/src/wizard/configs/driver_source) (S10, S13) | P | High |
| 2.4b | The same mechanisms exist on the **`e2e_challenge` branch** (head 2026-10-20), where S2's scoring protocol lives: `planner_delay_us` in `config.py`, `delay_buffer.py`, `external_static.yaml` and `external_dynamic.yaml` (HTTP 200). There, too, the A1.5 driver config has `max_batch_size: 1` | [e2e_challenge tree](https://github.com/NVlabs/alpasim/tree/e2e_challenge) | P | High |
| 2.5 | Driver gRPC: `EgodriverService` has `start_session`, `close_session`, `submit_image_observation` (JPEG `image_bytes` per camera, with frame start/end µs), `submit_egomotion_observation`, `submit_route`, `submit_recording_ground_truth`, `drive(time_now_us, time_query_us)` → trajectory. `DriveResponse.debug_info.unstructured_debug_info` takes arbitrary bytes | [egodriver.proto](https://github.com/NVlabs/alpasim/blob/main/src/grpc/alpasim_grpc/v0/egodriver.proto) (S12) | P | High |
| 2.6 | `assert_zero_decision_delay` verifies that camera frames and pose updates complete exactly at decision time. DESIGN.md lists real-time execution as a non-goal, i.e. sim time is decoupled from wall time | [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/main/docs/OPERATIONS.md); [DESIGN.md](https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md) (S10) | P | High |
| 2.7 | Seeds: PR #149 added `RolloutSpec.random_seed`. Per-inference seeds are `session.seed + session.inference_count`. The PR states the goal is re-running a rollout with the same policy behaviour, "not bit-exact replay" (NuRec caveat) | [PR #149](https://github.com/NVlabs/alpasim/pull/149); [driver main.py](https://github.com/NVlabs/alpasim/blob/main/src/driver/src/alpasim_driver/main.py) (S15, S02) | P | High |
| 2.8 | Holoscan SDK **v5.0.0 (2026-10-21)** is "a ground-up rewrite". Its new primitives: temporal contracts with `max_start_delay`; deterministic clock identities; CUDA Graph regions; CUDA streams per execution partition; CUDA Green Context manual placement (developer preview); bounded drop-oldest queues; HoloMQ/iceoryx2 zero-copy GPU IPC; external channel clients. It requires CUDA 13, has no source or binary compatibility with 4.x, and lists Thor known issues (SIPL capture; a display crash above 144 Hz) | [Holoscan v5.0.0](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v5.0.0) (S41) | P | High |
| 2.9 | Holoscan Sensor Bridge: 2.7.0 (2026-07-17) adds unified Linux/MCU emulators. 2.6.0 (2026-06-04) adds JetPack 7.2 support for AGX Thor (SIPL) and CoE offload on AGX Thor | [HSB releases](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/releases) (S42) | P | High |
| 2.10 | No NVIDIA reference design for HIL with AlpaSim or DRIVE Sim was found. The search returned only DRIVE dev-kit material: SKU10 is the bench kit | [DRIVE FAQ](https://developer.nvidia.com/drive/faq) (search summary) | S | Medium (absence of evidence) |

#### Inferences
- **The lockstep is native.** AlpaSim's runtime awaits `drive()` in sim time, so a Thor endpoint behind `external_static` is lockstep by construction, and network latency cannot enter the score. On-device time must be logged by the driver itself, for example in `unstructured_debug_info`.
- **Constant-delay injection is a config value.** It quantizes to control steps (known answer below), so p50 and p99 latencies that fall in the same (100, 200] ms band are indistinguishable at 10 Hz.
- **Known answer** (re-typed `DelayBuffer` logic; `scratchpad/calc/delay_ka.py`). Plans are added at t = 0, 100, …, 500 ms and queried at the same instants:

  | Delay | Applied plans |
  |---|---|
  | 0 ms | plan 0, 1, 2, 3, 4, 5 |
  | 100 ms | plan 0, 0, 1, 2, 3, 4 (1 tick late) |
  | 139, 171 or 200 ms | plan 0, 0, 0, 1, 2, 3 (2 ticks late, matching `03` §4 S7's 2-tick known answer) |

  This holds if `item_at` is called only at control-step times (see Gaps).
- **Glass-to-action is AlpaSim's own concept.** P-009(ii)'s glass-to-action is what `planner_delay_us` models ("image capture to … controller"). S7 can inject both model-only latency and glass-to-action latency as two arms.

#### Gaps
- `item_at`'s call frequency: it is passed from `event_loop.py`, and whether controller or physics sub-steps query it was not traced. That decides whether sub-tick delays matter.
- Whether NuRec rendering is deterministic run-to-run. PR #149 implies not bit-exact; this sets the floor for S7's TOST equivalence.

### Q3 — TensorRT / ModelOpt / CUDA features that matter in 2026

#### Takeaway
The 2026 toolchain moved toward the four things S6 needs:
- strongly typed TRT 11 engines with `IPluginV3`;
- ModelOpt QAD for Alpamayo;
- FP8 ViT recipes;
- Edge-LLM CUDA-graph flow loops, with an explicit engine-vs-PyTorch parity procedure.

None of them adds Alpamayo 1.5 or Edge to a Thor engine path.

#### Cited findings

| # | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| 3.1 | ModelOpt 0.47.0 adds `examples/alpamayo/qad.py`. It runs quantization-aware distillation of the quantized Alpamayo VLM against the FP16 VLM with `QADTrainer`, supports FSDP2, and re-exports an `AlpamayoR1` checkpoint | [ModelOpt 0.47.0](https://github.com/NVIDIA/Model-Optimizer/releases) (S39) | P | High. That the example targets R1 rather than 1.5 is an inference from the class name |
| 3.2 | ModelOpt 0.47.0 also adds: opt-in FP8 Vision Encoder recipes for `qwen3_vl` (Cosmos-Reason2 is a Qwen3-VL architecture per `04` §1); `nvfp4_act_headroom` calibration; ONNX Autotune that keeps Q/DQ only when it meets a TensorRT speed-up threshold (1.02× default) | S39 | P | High |
| 3.3 | Edge-LLM pi0.5 runs as four TensorRT components under an experimental C++ runtime. `--steps` sets the denoise count and `--cudaGraph` captures the loop. A `cond/` component hoists the AdaRMS modulation out of the per-step graph | [pi05.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/pi05.md) (S24) | P | High |
| 3.4 | Edge-LLM's pi0.5 correctness check runs both sides on the same pixels, token IDs and initial noise `x_0` taken from the reference, because "a shared `--seed` does not give a shared trajectory". It reports cosine similarity and max absolute error against thresholds | S24 | P | High |
| 3.5 | Torch-TensorRT's newest release is v2.14.0 (2026-09-10) | [Torch-TensorRT releases](https://github.com/pytorch/TensorRT/releases) (S40) | P | High |
| 3.6 | FlashDrive's "system optimizations" are per-stage CUDA graphs, fused QKV and fused gate/up projections, and "max-autotune" compilation. They give 1.40× (716.9 → 513.3 ms) on RTX PRO 6000 (edition unstated). The paper names no inference framework and gives no Jetson JetPack/TRT/CUDA versions | [FlashDrive HTML](https://arxiv.org/html/2608.12932) (S19) | P | High |
| 3.7 | Per-precision support by SM in the TRT 11.4.0 matrix is behind an interactive filter and was not readable. The 10-25 ledger C3 (TRT 10.16 lists FP8/FP4 on sm_110) stands | S34 | P | Medium |

#### Inferences
- **sm_120 and sm_110 share the FP8/NVFP4 formats but not the kernels or tactics.** An FP8 verdict measured on RTX PRO 6000 is evidence about the format and that engine, not about a Thor engine (consistent with plan §6).
- **Torch-TensorRT 2.14 implies a newer torch than the pinned 2.8.0** (Torch-TensorRT versions track PyTorch). Using it in the driver means environment drift (inference).

#### Gaps
- No readable sm_110 / sm_120 precision table for TRT 11.4.
- No NVIDIA statement on quantizing Alpamayo's flow expert in TensorRT, beyond Edge-LLM's FP16-only Alpamayo export.

### Q4 — Rollout acceleration: where the wall-clock goes, which levers exist, which checks keep the gate valid

#### Takeaway
No public source gives a component split of an AlpaSim rollout step. Two primary facts bracket the driver's share at **f ≈ 0.73–0.875** of GPU time:
- FlashDrive's step totals combined with its inference table;
- NVIDIA's own 7-of-8-GPU driver provisioning.

The stock A1.5 driver is eager, batch 1 and ungraphed. NVIDIA's bitwise-verified expert graph cuts its policy-call latency by 15.5% (10B) and 31.5% (2B) in AlpaGym. By Amdahl that is 11–14% of every A1.5 sweep. GPU batching could save ~40–57% (published non-Alpamayo analogs) but needs the ladder and a paired verdict.

#### Cited findings

| # | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| 4.1 | AlpaSim per-step rollout latency is 1,150 ms (Alpamayo 1.5) → 463 ms (+FlashDrive, 1 sampled trajectory), on 100 random clips. It covers model inference, trajectory optimization and rendering. **No component split; GPU not stated** | [FlashDrive Table 3](https://arxiv.org/html/2608.12932) (S19) | P | High |
| 4.2 | FlashDrive Table 1 (RTX PRO 6000, 1 trajectory), totals in ms: 716.9 baseline; 513.3 +system opts; 355.8 +streaming; 401.3 +speculative reasoning; 450.4 +adaptive steps; 176.0 all; 151.4 +quantization. Action stage: 192.9 → 113.9 (system opts) → 47.5 (all) | S19 | P | High |
| 4.3 | AlpaSim's A1.5 driver config: `max_batch_size: 1` ("A1.5 is memory intensive, start with batch size 1"); `num_trajectory_samples: 1`; `image_decode_device: cuda`; `force_determinism: false`. Four cameras rendered at 1900×1080 at 10 Hz | [alpamayo1_5.yaml](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/driver/alpamayo1_5.yaml); [alpamayo_configs.yaml](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/driver/alpamayo_configs.yaml) (S04) | P | High |
| 4.4 | The driver servicer's worker thread queues `drive` jobs up to `max_batch_size` and calls `predict_batch`. The base `predict_batch` "calls predict sequentially", and models must override it for GPU batching. **No Alpamayo adapter overrides it**, on `main` (head 2026-08-18) or on `e2e_challenge` (head 2026-10-20) | [main.py](https://github.com/NVlabs/alpasim/blob/main/src/driver/src/alpasim_driver/main.py); [base.py](https://github.com/NVlabs/alpasim/blob/main/src/driver/src/alpasim_driver/models/base.py) (S02, S03); commit lists (S14) | P | High |
| 4.5 | The Alpamayo adapter runs `sample_trajectories_from_data_with_vlm_rollout` under `torch.no_grad()` and `torch.autocast(bfloat16)`, with `MAX_GENERATION_LENGTH = 256`, top-p 0.98 and temperature 0.6. `force_determinism` sets `torch.use_deterministic_algorithms(True)`, cuDNN deterministic and `CUBLAS_WORKSPACE_CONFIG=":4096:8"` (TF32 stays allowed) | [alpamayo_base.py](https://github.com/NVlabs/alpasim/blob/main/src/driver/src/alpasim_driver/models/alpamayo_base.py) (S01) | P | High |
| 4.6 | In alpamayo1.5, the VLM rollout uses HF `self.vlm.generate(...)`. The expert loop runs inside an optional `_diffusion_expert_cuda_graph.sampling()` context | [alpamayo1_5.py](https://github.com/NVlabs/alpamayo1.5/blob/main/src/alpamayo1_5/models/alpamayo1_5.py) (S18) | P | High |
| 4.7 | **alpamayo1.5 PR #36** (merged 2026-08-29): opt-in `enable_diffusion_expert_cuda_graph(max_batch_size, max_graphs)`; "off by default"; the VLM rollout is unchanged. Paired AlpaGym workloads, one paired sample per model, "preliminary":<br>• 10B: policy-call latency −15.53%, completed-rollout throughput +73.08%, synchronized trainer-step throughput +56.32%;<br>• 2B: −31.53%, +53.43%, +0.59% (training-bound);<br>• 10B iteration 167.700 → 107.279 s; trainer compute 41.931 vs 44.679 s | [PR #36](https://github.com/NVlabs/alpamayo1.5/pull/36) (S16) | P | High (as reported; n = 1 pair) |
| 4.8 | PR #36 parity: a NVIDIA reviewer asked for parity evidence. The author reported 160 paired eager-vs-graph expert evaluations, real 2B and 10B experts, batch 1–4: all bitwise identical (max_abs_difference = 0) | [PR #36 comments](https://github.com/NVlabs/alpamayo1.5/pull/36) (S16) | P | High |
| 4.9 | Follow-up #38 (2026-09-09): "Fix CUDA graph capture on non-current device" | [alpamayo1.5 commits](https://github.com/NVlabs/alpamayo1.5/commits/main) (S17) | P | High |
| 4.10 | The AlpaSim driver neither calls `enable_diffusion_expert_cuda_graph` nor contains "cuda_graph", on main or e2e_challenge. The driver's `alpamayo1_5` dependency is an **unpinned git source** | [driver pyproject.toml](https://github.com/NVlabs/alpasim/blob/main/src/driver/pyproject.toml) (S06), S01 | P | High |
| 4.11 | NVIDIA's `8gpu_alpamayo1_5_eval` topology: driver at 2 replicas per container on GPUs 0–6; renderer and physics on GPU 7; renderer runs 14 concurrent rollouts, each driver replica 1. `1gpu` topology: renderer, driver, physics and trafficsim on GPU 0, 4 concurrent rollouts | [8gpu_alpamayo1_5_eval.yaml](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/topology/8gpu_alpamayo1_5_eval.yaml); [1gpu.yaml](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/topology/1gpu.yaml) (S05) | P | High |
| 4.12 | AlpaSim telemetry is on by default. `metrics_plot.png` shows RPC duration, blocking and queue depth, rollout and step duration, CPU and GPU utilization and memory, "Async worker idle percentage" and "Sim seconds per rollout". The Slurm exporter publishes `alpasim_gpu_workload_info` for joining with DCGM | [TELEMETRY.md](https://github.com/NVlabs/alpasim/blob/main/docs/TELEMETRY.md); [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/main/docs/OPERATIONS.md) (S10, S11) | P | High |
| 4.13 | recipes PR #44 (open, docs): measurement chapter items include wall clock as complete s/step from log timestamps; "GPU-hour: use the mean wall clock, never the median"; MFU/HFU/OFU; and CUDA-event decomposition into arrival / forward / backward / host residual | [PR #44](https://github.com/NVlabs/alpamayo-recipes/pull/44) (S27b) | P | Medium (unmerged draft) |
| 4.14 | Batching analogs (not Alpamayo), from the 10-03 ledger: PhyAI π0.5 2.26× (Hopper, B = 32); GR00T 2.84× (RTX 5090); π0.5 on Thor plateaus at ~2.2× by B = 16 | [PhyAI](https://arxiv.org/html/2608.03682v3), via ledger 10-03 Q1 | P via ledger | High (10-03) |

#### Inferences (labelled; the arithmetic is in `l7_arith.py`)
- **Sweep cost from per-step time.** 441 × 3 × 200 = 264,600 steps. At 463 / 1,150 ms per step that is **34.0 / 84.5 GPU-h** (reproduces the plan).
- **Driver share *f*.**
  - From FlashDrive: if FlashDrive's step used its 151.4–176.0 ms inference row, the non-inference remainder is 287–312 ms. The baseline inference would then be 838–863 ms, so f ≈ **0.73–0.75**.
  - From NVIDIA's provisioning: 7 of 8 GPUs for the driver suggests f ≈ **0.875**, if the topology is balanced.
  - Bracket: **f ∈ [0.73, 0.875]**.
- **Amdahl:** sweep saving = 1 − ((1 − f) + f/s). Results, as % of each sweep:

  | Lever | s | Saving per sweep |
  |---|---|---|
  | Expert graph, 10B | 1.184 | 11.3–13.6% |
  | Expert graph, 2B | 1.460 | 23.0–27.6% |
  | FlashDrive system optimizations | 1.397 | 20.7–24.9% |
  | Batching analogs | 2.26 / 2.84 | 40.7–48.8% / 47.3–56.7% |

- **Batching bound for Alpamayo.** At B = 1 the call is ~717 ms (encode 87 + prefill 165 + decode 272 + action 193). Batching amortizes only the memory-bound stages:
  - decode falls to 0.51 ms per sample per token at B = 32 (`04` §4);
  - the expert falls to its ≈7–11 ms compute floor per sample;
  - prefill and encode (~229–252 ms) stay roughly per-sample.

  Per-sample time ≈ 252 + ~15 + ~9 ≈ 276 ms ⇒ s ≈ 2.6 from the unoptimized B = 1 baseline. From the system-optimized one, 513 / 253 ⇒ ≈ 2.0. That is consistent with the analogs.
- **Numerics classes.** The binding rule assigns each lever a check:

  | Lever | Numerics class | Required check |
  |---|---|---|
  | CUDA graphs over the same kernels (PR #36); extra replicas per GPU (same B = 1 kernels) | **Bitwise** | Bitwise test |
  | Batching (kernel choice and reduction order vary with B; padding; RNG streams); `torch.compile` fusion or max-autotune; static padded KV; TRT engines; rendering at model resolution | **Numerics-changing** | Ladder + paired verdict |
  | FP8 / NVFP4 | **A different policy** (separate checkpoint row; never a speed-up for BF16 sweeps) | Its own paired verdict |

- **Seeding under batching.** AlpaSim's seeding calls a global `torch.manual_seed(inference_seed)` per `predict` (when `force_determinism`). A batched forward breaks per-session streams unless each sample gets its own generator. Paired verdicts pair by scene, so seeds stay replicates; but "seed test–retest" changes meaning if RNG streams change.
- **S4 training rollouts cannot take these levers.** Public AlpaGym has been code-frozen since 2026-07-08, before PR #36 (08-29). It cannot call the graph API without an internals change, which the binding rules forbid.

#### Gaps
- No measured A1.5 per-service GPU-time split, and no measured A1.5 batch curve: L7-1 and L7-4 produce them.
- No A1.5 measurement of replicas per GPU (L7-3).
- Which FlashDrive configuration produced 463 ms.
- Which RTX PRO 6000 edition FlashDrive used.
- Whether vast.ai containers expose GPU performance counters (needed by `ncu`) and allow clock locking. Not checked; a common cloud restriction.

### Q5 — "Exceptional C++" for S6–S7, and which NVIDIA C++ codebases to interoperate with

#### Takeaway
NVIDIA's edge-inference code is C++:
- **TensorRT-Edge-LLM:** "High-performance, light-weight C++ LLM and VLM Inference Software for Physical AI". Google-style guidelines, clang-format, DCO; it merges outside contributions.
- **Holoscan SDK 5.0.**

AlpaSim's driver interface is a gRPC proto, so S7's natural C++ artifact is an `EgodriverService` server around a TensorRT (or Edge-LLM) runtime. The latency-injection mechanism itself belongs in AlpaSim's Python runtime, where it already exists.

#### Cited findings

| # | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| 5.1 | The Edge-LLM repo description reads "High-performance, light-weight C++ LLM and VLM Inference Software for Physical AI". The repo has `cpp/`, `kernelSrcs/`, `unittests/`, `.clang-format` and `CODING_GUIDELINES.md` | [Edge-LLM repo](https://github.com/NVIDIA/TensorRT-Edge-LLM) (S20, S26) | P | High |
| 5.2 | Edge-LLM coding rules (derived from the Google C++ Style Guide): smart pointers (`unique_ptr` for single ownership); exceptions "must not be thrown across library boundaries"; `//!` doc comments; clang-format ≥ 14 | [CODING_GUIDELINES.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CODING_GUIDELINES.md) (S26) | P | High |
| 5.3 | Edge-LLM requires DCO sign-off on every commit. Release 0.10.0 credits an outside contributor's PR #98 | [CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CONTRIBUTING.md); S20 | P | High |
| 5.4 | Edge-LLM runtime features relevant to S6: CUDA-graph loops (pi0.5 `--cudaGraph`; Qwen3-Omni full-loop graph); paged KV; C++ chat-template rendering (0.11.0); C++ audio preprocessing (0.9.0); mock-engine unit tests (0.10.1); QNX cross-compilation (0.10.1) | S20, S24 | P | High |
| 5.5 | Holoscan 5.0 C++ primitives: `Input<T>::selected_sample()` zero-copy borrow; bounded pools with loan/commit; `RegionConstraint::cuda_graph()`; `Contract::max_start_delay`; drop-oldest bounded queues; deterministic clock identities; CMake ≥ 4.0 | S41 | P | High |
| 5.6 | AlpaSim services are Python, with Rust utilities (`src/utils_rs`) and gRPC protos (`src/grpc/alpasim_grpc/v0/*.proto`) | [AlpaSim tree](https://github.com/NVlabs/alpasim) | P | High |

#### Inferences: what production-grade means in S6–S7, concretely
**Which pieces are C++:**
- (1) The TensorRT engine runner and flow-step loop: a fixed-shape enqueue captured in a CUDA graph, static KV buffers, and the K-step expert loop as a captured graph or one `IPluginV3` op.
- (2) The gRPC `EgodriverService` server on Thor.
- (3) The P-009 bandwidth-contention generator (CUDA).
- (4) The timing harness: NVTX ranges, CUDA events, `CLOCK_MONOTONIC` host stamps.

The latency **injection** stays in AlpaSim (Python `DelayBuffer`). Only the latency **measurement** is C++.

**Properties, each paired with the check that makes it evidence:**

| Property | Check |
|---|---|
| Determinism | Seeded on-device sampling from `inference_seed`; golden-output **bitwise replay** test |
| Zero-copy | nvJPEG decode straight into pre-allocated device buffers; pinned-host staging. Allocation count on the hot path = 0, checked with a counting allocator hook |
| Streams and events | One compute stream per stage, with event-based dependencies and no `cudaDeviceSynchronize` on the hot path. Checked with an nsys timeline |
| Bounded latency | Lock-free SPSC ring buffers between gRPC threads and the inference thread, with bounded depth and a drop policy. p50/p99/p99.9 histograms exported to Prometheus |
| Pinned clocks | `nvpmodel` / `jetson_clocks` recorded in the manifest (P-009 iii) |
| Memory safety | `compute-sanitizer` memcheck / racecheck / synccheck / initcheck; ASan / UBSan / TSan host builds |
| Static analysis | clang-tidy; Edge-LLM's clang-format |
| Testability | GoogleTest with a mock engine (Edge-LLM precedent); gRPC contract tests against `egodriver.proto` |
| Portability | Strongly typed engines, `IPluginV3`, CI on TRT 10.16 and TRT 11 (L7-7) |

**Interoperate with or contribute to:**
- **Edge-LLM** (DCO): file issues or PRs for what breaks, never a parallel Alpamayo 1.5 exporter;
- **AlpaSim:** the driver flag (L7-2) and, if needed, a trace-driven `DelayBuffer`;
- **not AlpaGym**, which takes no outside PRs.

#### Gaps
- Whether PR #45's forked Edge-LLM builds for x86 SM120: needed for L7-6 on the student.
- Edge's runtime format at release.

### Q6 — P-007…P-010 re-checked against today's facts (for JR2010863)

#### Takeaway
- **P-007 becomes stronger and should be amended.** The S6a expert diagnosis runs on the same GPU class as every S2–S5 sweep. NVIDIA's PR #36 already removed CPU launch overhead on the PyTorch path. FlashDrive's 113.9 ms row already includes CUDA graphs, so the residual 5× gap is GPU-side.
- **P-008 becomes stronger.** NVIDIA now ships FP8 ViT, ViT-embedding caching and media-aware KV reuse.
- **P-009 becomes much stronger.** Its ladder becomes the gate's protocol for every driver change, with NVIDIA's own parity precedent; and its glass-to-action matches `planner_delay_us`.
- **P-010 becomes stronger only as P-007's H2 discriminator.** It is weak as stand-alone role evidence.

#### Cited findings and reasoning

| Proposal | Fact (source) | Effect for JR2010863 |
|---|---|---|
| **P-007 (S6a on RTX PRO 6000)** | FlashDrive's system-optimized expert (113.9 ms for 8 steps) **already includes per-stage CUDA graphs and QKV / gate-up fusion** (S19). The bound is 5.00 GB / 1.792 TB/s = 2.79 ms per step ⇒ 22.3 ms ⇒ 19.6% (arithmetic). The Server edition gives 25.0 ms (S19; `04` §4). PR #36 graphs cut 10B policy-call latency by 15.5% (S16). Edge-LLM supports x86 SM120 (S21) | **Stronger.** Split H1 into H1a (CPU launch overhead: addressed by graphs; NVIDIA already did it for PyTorch) and H1b (GPU-side per-kernel ramp and tail, roughly 36 layers × ~10 kernels × 8 steps ≈ 2,880 kernels per tick: inference). H1a cannot explain FlashDrive's residual, so E-K0's `ncu` per-kernel achieved-bandwidth table should discriminate H1b, H2 and H3. Run E-K0 with graphs **on** (`enable_diffusion_expert_cuda_graph`) as well as eager. Any fix found is a rollout-cost lever for S2–S5 before it is a Thor lever (I2-perf, X3) |
| **P-008 (narrow S6)** | Edge-LLM 0.10.1: end-to-end FP8 ViT attention, DART pruning, a media artifact cache retaining ViT embeddings. 0.10.0: media-aware KV reuse (S20). ModelOpt 0.47: FP8 Vision Encoder recipes for `qwen3_vl` (S39) | **Stronger.** Don't build a camera-batched or FP8 ViT, or ViT-embedding reuse, from scratch: that races NVIDIA. Consume them if the runtime supports the model. Moving streaming KV reuse to S5 stays correct (it is a model change; FlashDrive needs expert fine-tuning) |
| **P-009 (contention, glass-to-action, pinned clocks, ladder)** | `planner_delay_us` is defined as capture → planner output → controller (S07). PR #36's reviewer demanded parity, and the author answered with bitwise 160/160 (S16). Edge-LLM's parity procedure injects the reference `x_0` (S24) | **Stronger.** (ii) glass-to-action becomes S7's injected delay. (iv) the ladder becomes the gate's instrument for L7-2…L7-6 in S2–S5: E0a = bitwise for numerics-preserving levers; E0b/E0c use injected reference noise, since a shared seed is not a shared trajectory across implementations. (iii) Add a recorded-clocks fallback for rented GPUs, where locking may be refused (unverified) |
| **P-010 (GEMV starvation exercise)** | The residual expert gap is post-graphs (above). The expert's GEMMs at B = 1 are M = 64 skinny GEMMs, whose CTA count is far below 188 SMs unless split-K is used (inference from shapes in `04` §2) | **Stronger only as the H2 discriminator** feeding P-007, with a known answer on CPU (Little's law: 1.792 TB/s × 500–800 ns ⇒ 0.90–1.43 MB in flight ⇒ 56–90 KB per CTA for 16 CTAs). As stand-alone role evidence it remains weak (an exercise, not a measurement of the loop). Cost ≈ 1 GPU-h ($1.00–1.60), 2–4 h |

#### Gaps
- No public per-kernel profile of Alpamayo's expert exists; E-K0 is its first.
- Whether `ncu` counters are available on the rented host (Q4 gap).

### Source list (IDs used in §1)

**AlpaSim** (main @ `affc2ea`, 2026-08-18, unless noted):

| ID | Source |
|---|---|
| S01 | [`alpamayo_base.py`](https://github.com/NVlabs/alpasim/blob/main/src/driver/src/alpasim_driver/models/alpamayo_base.py) |
| S02 | [`main.py` (driver)](https://github.com/NVlabs/alpasim/blob/main/src/driver/src/alpasim_driver/main.py) |
| S03 | [`base.py` (`predict_batch`)](https://github.com/NVlabs/alpasim/blob/main/src/driver/src/alpasim_driver/models/base.py) |
| S04 | [driver `alpamayo1_5.yaml`](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/driver/alpamayo1_5.yaml) and [`alpamayo_configs.yaml`](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/driver/alpamayo_configs.yaml) |
| S05 | [`8gpu_alpamayo1_5_eval.yaml`](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/topology/8gpu_alpamayo1_5_eval.yaml) and [`1gpu.yaml`](https://github.com/NVlabs/alpasim/blob/main/src/wizard/configs/topology/1gpu.yaml) |
| S06 | [driver `pyproject.toml`](https://github.com/NVlabs/alpasim/blob/main/src/driver/pyproject.toml) |
| S07 | [runtime `config.py`](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/config.py) and [`event_loop.py`](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/event_loop.py) |
| S08 | [`delay_buffer.py`](https://github.com/NVlabs/alpasim/blob/main/src/runtime/alpasim_runtime/delay_buffer.py) |
| S09 | [CHANGELOG](https://github.com/NVlabs/alpasim/blob/main/CHANGELOG.md) |
| S10 | [OPERATIONS.md](https://github.com/NVlabs/alpasim/blob/main/docs/OPERATIONS.md), [DESIGN.md](https://github.com/NVlabs/alpasim/blob/main/docs/DESIGN.md) |
| S11 | [TELEMETRY.md](https://github.com/NVlabs/alpasim/blob/main/docs/TELEMETRY.md) |
| S12 | [`egodriver.proto`](https://github.com/NVlabs/alpasim/blob/main/src/grpc/alpasim_grpc/v0/egodriver.proto) |
| S13 | [`driver_source` configs](https://github.com/NVlabs/alpasim/tree/main/src/wizard/configs/driver_source) |
| S14 | Commit lists: [main](https://github.com/NVlabs/alpasim/commits/main) (head 2026-08-18) and [e2e_challenge](https://github.com/NVlabs/alpasim/commits/e2e_challenge) (head 2026-10-20) |
| S15 | [PR #149](https://github.com/NVlabs/alpasim/pull/149) |

**alpamayo1.5:**

| ID | Source |
|---|---|
| S16 | [PR #36 and its comments](https://github.com/NVlabs/alpamayo1.5/pull/36) |
| S17 | [commits (#38)](https://github.com/NVlabs/alpamayo1.5/commits/main) |
| S18 | [`alpamayo1_5.py`](https://github.com/NVlabs/alpamayo1.5/blob/main/src/alpamayo1_5/models/alpamayo1_5.py) |

**Papers and NVIDIA software:**

| ID | Source |
|---|---|
| S19 | [FlashDrive HTML](https://arxiv.org/html/2608.12932) |
| S20 | [TensorRT-Edge-LLM releases](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases) |
| S21 | [Edge-LLM support-matrix.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/support-matrix.md) |
| S22 | [Edge-LLM supported-models.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/supported-models.md) |
| S23 | [Edge-LLM alpamayo.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/alpamayo.md) |
| S24 | [Edge-LLM pi05.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/pi05.md) |
| S25 | [Edge-LLM PR #260](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260) |
| S26 | Edge-LLM [CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CONTRIBUTING.md) and [CODING_GUIDELINES.md](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CODING_GUIDELINES.md) |
| S27 | [recipes PR #45](https://github.com/NVlabs/alpamayo-recipes/pull/45) |
| S27b | [recipes PR #44](https://github.com/NVlabs/alpamayo-recipes/pull/44) |
| S28 | [HF API, Alpamayo models](https://huggingface.co/api/models?author=nvidia&search=alpamayo) |

**NVIDIA announcements, platform docs and other projects:**

| ID | Source |
|---|---|
| S29 | [Press release, 2026-10-21](https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion) |
| S30 | [Blog, 2026-10-21](https://blogs.nvidia.com/blog/open-models-europe-industry/) |
| S31 | [Alpamayo page](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/) |
| S32 | [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) |
| S33 | [Jetson FAQ](https://developer.nvidia.com/embedded/faq) |
| S34 | [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) |
| S35 | [TRT 10→11 Jetson migration](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html) |
| S36–S38 | DriveOS migration, the DriveOS 7.2.5 announcement and the TRT notes PDF (Q1 row 1.6) |
| S39 | [ModelOpt releases](https://github.com/NVIDIA/Model-Optimizer/releases) |
| S40 | [Torch-TensorRT releases](https://github.com/pytorch/TensorRT/releases) |
| S41 | [Holoscan SDK v5.0.0](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v5.0.0) |
| S42 | [HSB releases](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/releases) |
| S43 | [FlashRT PRs](https://github.com/flashrt-project/FlashRT/pulls) |
| S44 | [HF API, Cosmos3-Edge](https://huggingface.co/api/models/nvidia/Cosmos3-Edge) |

---

## 3. Deltas against the earlier ledger (dated 2026-10-27)

| # | Earlier ledger said (date) | Delta found 2026-10-27 | Source |
|---|---|---|---|
| Δ1 | Nothing about a public expert CUDA graph. `01` C10 and P-007 treat the expert's ~20% efficiency as undiagnosed | **New:** alpamayo1.5 PR #36 (merged 2026-08-29) plus #38 (09-09) add an **opt-in, bitwise-verified** (160/160) expert CUDA graph. NVIDIA reports −15.5% (10B) and −31.5% (2B) policy-call latency in paired AlpaGym workloads (n = 1 pair each, "preliminary") | S16, S17 |
| Δ2 | AlpaSim is "gRPC microservices … Prometheus telemetry" (`04` §5) | **New:** the A1.5 driver runs at `max_batch_size: 1`. GPU batching is not implemented for any Alpamayo adapter, and the expert graph is not enabled (main 08-18; e2e_challenge 10-20). The `alpamayo1_5` dependency is an unpinned git source, so whether a future lock resolves past PR #36 depends on install date. This extends issue #207 | S01–S06 |
| Δ3 | S7 plans to build latency injection (`02` S7) | **New:** AlpaSim already has `planner_delay_us` (a fixed-delay `DelayBuffer`) and external-driver modes. NVIDIA's Alpamayo configs used a 200 ms planner delay until 2026-03-24 | S07–S10, S13 |
| Δ4 | AlpaSim throughput was known only via FlashDrive's 1,150 → 463 ms (10-03 Q3) | **Refined:** FlashDrive gives no component split, no AlpaSim GPU, and no statement of which inference row produced 463 ms. NVIDIA's own A1.5 evaluation topology provisions 7/8 GPUs for the driver. Inferred driver share f ≈ 0.73–0.875 | S19, S05 |
| Δ5 | "TensorRT 11.x (current 11.4.0) does not support Jetson Thor" (10-25 C3) | **Consistent, with two additions:** NVIDIA says a later 11.x will restore JetPack (no date). The 11.x API requires strong typing (precision builder flags removed), deprecates `IPluginV2DynamicExt` in favour of `IPluginV3`, needs CUDA 13.2 Update 2+, and runs TRT 10.x engines on its runtime | S35 |
| Δ6 | "DRIVE AGX Thor on DriveOS 7.2.6+" for TRT 11.x (10-25 C3) | **Conflict:** the DriveOS 7.2.5 EA TRT notes PDF is for TRT 11.0.1 (dated 2026-07-02); the docs disagree on 7.2.5+ vs 7.2.6+; no public 7.2.6 announcement was found. Edge-LLM's wheel matrix lists DRIVE Thor with "platform TensorRT 10". Irrelevant to Jetson claims | S36–S38, S21 |
| Δ7 | Edge-LLM v0.11.0; #260 open; Alpamayo R1 FP16 only (10-25 B5) | **Unchanged.** #260 has no review, only a CI-vetting bot notice. **New context:** x86 SM120 is a supported Developer platform with TRT 10/11 wheels; pi0.5's parity procedure; FP8 ViT, ViT cache and media-aware KV reuse shipped in 0.10.x | S20–S25 |
| Δ8 | ModelOpt 0.43 pinned (`04` §5) | **New:** 0.47.0 (09-23) stable and 0.48.0rc1 (10-17). 0.47 adds `examples/alpamayo/qad.py` (QAD), FP8 ViT recipes for `qwen3_vl`, `nvfp4_act_headroom`, and speed-thresholded ONNX Q/DQ | S39 |
| Δ9 | Edge download promised "later this month"; HF 401 (10-25) | **Unchanged on 10-27:** 401 on all guessed names; HF lists 3 Alpamayo models. **New:** no NVIDIA source states the 10 Hz configuration | S28–S31 |
| Δ10 | Holoscan only as camera I/O (C7) | **New:** Holoscan SDK 5.0.0 (10-21) is a ground-up, CUDA-13-only rewrite with real-time primitives and Thor known issues. HSB 2.7.0 adds emulators | S41, S42 |
| Δ11 | JetPack 7.2.1; $5,499 (10-25) | **Unchanged:** archive's newest is 7.2.1 (L4T 39.2.1); FAQ $5,499; T5000 $4,999 / T4000 $2,999 (1KU+) | S32, S33 |
| Δ12 | recipes PRs #42, #43, #45 open (10-25) | **Unchanged.** **New:** open docs PR #44 (measurement methodology: GPU-hours from the mean wall clock; CUDA-event decomposition) | S27, S27b |
| Δ13 | FlashRT #229 is the best Jetson Alpamayo result (10-25 B4) | **Unchanged:** no Alpamayo PR among #235–#242 (to 10-27) | S43 |
| Δ14 | — | **New:** Torch-TensorRT v2.14.0 (09-10) | S40 |

---

## 4. First-principles rationale per finding

### Cross-stage technique matrix (objective 2)
How S6's serving techniques apply to the policy step inside AlpaSim rollouts in S2–S5.

| S6 technique | Where in S2–S5 | Numerics class → check | Speed-up input (published or derived; not a prediction) | Role evidence |
|---|---|---|---|---|
| CUDA graphs (expert) | Every Alpamayo-driver sweep (L7-2) | Bitwise → bitwise test (NVIDIA precedent 160/160) | 10B: −15.5% call latency ⇒ 11–14% per A1.5 sweep at f = 0.73–0.875. The 2B figure (−31.5% ⇒ 23–28%) is NVIDIA's **internal** 2B through the alpamayo1.5 graph API. It applies to S3/S5 student sweeps **only if Edge ships on a code path exposing the same expert-graph API** (unknown until release); it is not an Edge measurement | I2-perf, X3: same pattern as NVIDIA's PR #36 (paired workload + parity) |
| Concurrency without batching (replicas) | Every sweep (L7-3) | Bitwise per rollout | NVIDIA runs 2 per GPU (A1.5 evaluation) and 4 per GPU (challenge); magnitude unmeasured | D2, Q5 |
| Batching across environments | Post-S2 sweeps (L7-4) | Numerics- and RNG-changing → ladder + known-null paired verdict | s ≈ 2.0–2.6 (Amdahl bound from FlashDrive stages); analogs 2.26–2.84 ⇒ 41–57% per sweep | I2-perf, X3, D3 (the verdict also tests the gate) |
| Static KV + decode graphs | Conditional (L7-5) | Numerics-changing → ladder + verdict | FlashDrive system-opt s = 1.397 total; incremental over L7-2 is 9–11 pp | X3 |
| Real TRT engines (C++ runtime) | Student sweeps via S6a (L7-6) | Numerics- and RNG-changing → ladder (inject reference `x_0`) + verdict | Unmeasured; Edge-LLM supports SM120 for development | Q4, D5, I2-perf |
| FP8 / NVFP4 | **Not** a lever for BF16 sweeps; only the FP8 policy rows, with a real engine where one exists | Different policy → its own paired verdict; FP8 on sm_120 ≠ FP8 on Thor | FlashDrive W4A8 is only 1.16× after the other optimizations | G6 on device (S6) |

### L7-1 — rollout profile (S2)
- **Mechanism.** Amdahl: a lever's sweep saving is 1 − ((1 − f) + f/s). Without *f*, every speed-up claim about the loop is a guess. FlashDrive gives step totals but no split. NVIDIA's provisioning gives an indirect *f*.
- **What S2 adds.** S2 already ships "first measured throughput" and Prometheus dashboards. Adding NVTX ranges and a CUDA-event split inside `predict()` (encode / prefill / decode / expert / host residual), plus per-service GPU time, turns a throughput number into a decision input for L7-2 to L7-6.
- **Why this beats the alternatives.** nsys alone has no service attribution. Telemetry alone has no driver stages. NVIDIA's measurement draft (PR #44) prescribes the same decomposition and the mean-not-median GPU-hour.
- **Cost.** +4–6 h; +2–4 GPU-h.
- **Role evidence.** A public "where the GPU-hours go in a closed-loop VLA rollout" report: I2-perf, D4 (monitoring and debugging tools), X3.

### L7-2 — expert CUDA graph (S2 onward)
- **Mechanism.** At B = 1 the expert issues roughly 2,880 small kernels per tick (36 layers × ~10 kernels × 8 steps; inference). CPU launch latency is serialized with GPU work. Graph replay launches the whole loop with one call over the same kernels, so the numerics are identical (NVIDIA's 160/160 bitwise).
- **Arithmetic.**
  - s = 1/(1 − 0.1553) = 1.184 for 10B. With f = 0.73–0.875, the saving is 11.3–13.6% per A1.5 sweep: 3.9–11.6 GPU-h per sweep.
  - **Over S2–S5:** 22–23 Alpamayo-driver sweeps (S2's TransFuser sweep excluded) save −85 to −266 GPU-h (−$85 to −$425). Including S6: −100 to −312 GPU-h (−$100 to −$499).
  - For student sweeps, s = 1.460 (NVIDIA's internal 2B, PR #36) would give 23.0–27.6%. That needs two conditions: Edge exposes the same alpamayo1.5 expert-graph API (unknown until release), and *f* holds (it falls as the driver gets faster; unmeasured).
  - Whether S2's FP8 (ModelOpt fake-quant) row captures cleanly in a graph is untested.
- **Determinism mode.** The bitwise test runs with `force_determinism=true`, while AlpaSim's default is `false`. If the sweeps run deterministic, bitwise identity transfers per run. If not, the claim is "same kernels ⇒ same output distribution" and the manifest says so. The mode is Huy's protocol choice, and L7-1 measures its throughput cost.
- **Why it beats the alternatives.** Zero validity risk. No new code beyond a flag. It does not race NVIDIA: it consumes NVIDIA's code and exposes it where NVIDIA has not.
- **Risks.** Compatibility with `force_determinism` and the driver's worker thread is unverified (#38 fixed capture on a non-current device). Graph memory is bounded by `max_graphs`.
- **Manifest.** Must record the `alpamayo1_5` commit; the dependency is unpinned (S06).
- **Upstream.** Re-read AlpaSim after the 10-31 challenge close before opening a PR. NVIDIA's internal sync may already wire it.

### L7-3 — replicas per GPU (S2)
- **Mechanism.** The eager B = 1 path has host gaps between kernels: Python, per-token HF `generate`, and a `.cpu()` sync per call. A second process's kernels fill them. Per-rollout kernels are unchanged, so outputs are bitwise.
- **Memory.** 22.16 GB BF16 weights plus KV and activations per replica ⇒ 2–3 replicas per 96 GB card (inference).
- **Prior.** NVIDIA's own topology choice (2 per GPU) and the challenge preset (4 per GPU) are strong priors that it helps. Its size is unmeasured for A1.5.
- **Cost.** About 1–2 h.

### L7-4 — GPU batching with per-sample RNG (gated)
- **Mechanism.** Decode and the expert are memory-bound at B = 1: weight reads amortize across B. Decode falls 8.7 → 0.51 ms per sample per token at B = 32, including KV reads. The expert crosses B* ≈ 4.4 into its compute floor of ≈ 7 ms per sample. Prefill is already compute-bound and does not amortize.
- **Bound.** Per-sample time → ≈ 253–276 ms ⇒ s ≈ 2.0–2.6, consistent with the PhyAI and GR00T analogs.
- **Validity.** Batching changes kernel selection and reduction order, and padding and masks. AlpaSim seeds per call with a global `torch.manual_seed`. So the check is the ladder plus a **paired verdict of batched vs unbatched on the same checkpoint**. That is a known-null pair: if the gate flags it, either the batching or the gate is wrong. This makes the performance change also a D3 "measure simulation quality" artifact.
- **Cost.** +10–14 h; +1 sweep ($34–136); potentially −263 to −964 GPU-h (−$263 to −$1,542) on S3–S5, or −304 to −1,108 GPU-h (−$304 to −$1,773) on S3–S6 (published-analog basis).
- **Gate.** Kind (c), because it adds hours and a sweep beyond S2's budget.

### L7-5 — static KV + decode graphs (conditional)
- **Mechanism.** At B = 1, HF `generate` pays per-token host overhead: FlashDrive's decode was 271.7 ms unoptimized vs 170.6 ms system-optimized.
- **Interaction.** Once L7-4 batches, decode per sample is about 8–20 ms, so L7-5's marginal value collapses. Keep it conditional on L7-1's profile, with L7-4 rejected.

### L7-6 — one C++ driver, two targets (gated on P-007)
- **Mechanism.** Write the inference runtime once (TensorRT C++ / Edge-LLM), run it on SM120 in the cloud for S5/S6 student sweeps, then on SM110 for S6b and S7. The C++ `EgodriverService` makes it a drop-in AlpaSim driver.
- **Validity.** The numerics and RNG change, so use Edge-LLM's own parity method: inject the reference `x_0`, run the ladder, then a paired verdict. A cloud-FP8 verdict does not substitute for the Thor-engine verdict.
- **Feasibility risk.** No public TRT path exists for 1.5 or Edge. PR #45's fork targets DRIVE.
- **Cost.** +12–20 h now.

### L7-7 — TRT-11-ready C++
- **Mechanism.** TRT 11 removes precision builder flags and deprecates the V2 plugin interfaces. DRIVE is already on 11.x and JetPack's 11.x is announced.
- **Why now.** Writing strongly typed networks with explicit Q/DQ and `IPluginV3` now costs ~0–2 h and avoids a port. It is also checkable C++ craftsmanship (Q4).

### L7-8 — S7 uses AlpaSim's own mechanisms
- **Mechanism.** Lockstep is native: the sim awaits `drive()`. A fixed delay is native too (`DelayBuffer`). The known answer shows the quantization: at Δt = 100 ms, 139 ms equals 200 ms (2 ticks).
- **Design inputs for Huy (★).** Injection can run at control-step quantization (cheap); or sub-tick resolution can come from a finer control step or controller-side application, as an AlpaSim PR.
- **Hours.** −4 to −8 h.
- **Note.** That NVIDIA once ran Alpamayo at a 200 ms planner delay shows the latency-cost question is native to the simulator, which is X2 evidence.

### L7-9 — production-grade C++ server
- **Mechanism.** Every property in §2 Q5 maps to a test whose output is public: bitwise replay; zero hot-path allocations; sanitizer-clean runs; p99.9 histograms.
- **Why this is Q4 evidence.** "Exceptional C++" becomes checkable instead of claimed.
- **Funding.** The hours come from L7-8.

### L7-10 — AlpaGym performance telemetry (S4)
- **Mechanism.** I2 names "the RL framework's" performance. NVIDIA's 10B eager AlpaGym iteration spent (167.700 − 41.931) / 167.700 = 75.0% outside trainer compute, falling to 58.4% with graphs.
- **What PhysLoop measures.** The public, frozen AlpaGym's share, through its opt-in `perf/` module, without touching internals.
- **Prediction slot.** The preserved 22-Sep headline 1 / E1 ("trainer idle ≥ 40%") is the slot. The reconciliation is Huy's.

### L7-11 — service-balance model (design doc + small measurement)
- **Mechanism.** Throughput is maximized when the driver and renderer capacities match: N_drv·μ_drv ≈ N_rnd·μ_rnd. This extends `04` §4's N_gen·g ≈ N_train·t.
- **Measurement.** L7-1's per-service rates plus one 2-GPU AlpaSim run test the model against NVIDIA's 7:1 choice. It stands in for cluster-scale work the plan cannot afford (D4, Q5).

### CPU-checkable known-answer candidates (for `03` §4; values are arithmetic, not predictions)
- **KA-1.** 264,600 steps × 0.463 / 1.150 s ⇒ 34.0 / 84.5 GPU-h.
- **KA-2.** Amdahl: f = 0.73, s = 1.184 ⇒ 11.3%; f = 0.875, s = 2.26 ⇒ 48.8%.
- **KA-3.** Expert bound: 5.00 GB per step ⇒ 2.79 ms (RTX PRO 6000 WS), 3.13 ms (Server), 18.32 ms (Jetson AGX Thor). FlashDrive measured 14.24 ms per step ⇒ 19.6% of the WS bound.
- **KA-4.** Expert FLOPs at B = 1: 3.50e11 per step ⇒ 0.70 ms at 500 TFLOPS, so not compute-bound.
- **KA-5.** `DelayBuffer` at Δt = 100 ms: delays 100 / 139 / 200 ms ⇒ 1 / 2 / 2 ticks late.
- **KA-6.** AlpaGym 10B rollout-wait share: 0.750 eager, 0.584 with graphs.
- **KA-7.** Little's law for P-010: 0.90–1.43 MB in flight ⇒ 56–90 KB per CTA at 16 CTAs.

---

## 5. Rejected options

| Option | Why rejected |
|---|---|
| Swap the AlpaSim driver's VLM to vLLM or SGLang for prefill and decode | The expert attends over the VLM's prompt KV (`04` §2), so exposing a paged KV to an external expert is custom integration. NVIDIA does it only inside the Alpamayo 2 Super NIM on vLLM-Omni, one request at a time (10-25 B8). It also changes numerics and RNG streams (ladder + verdict) for gains that native batching (L7-4) delivers with less risk |
| Use FP8 to speed up the BF16 sweeps, or pull a real FP8 A1.5 engine into S2 | FP8 is a different policy (its own S2 row). No real-engine path exists for 1.5: Edge-LLM is R1 FP16 only. The ModelOpt PyTorch checkpoint is fake-quant, slower not faster (10-25 C3); label that row "emulated" and budget it as at least the BF16 cost (inference) |
| Render at model resolution (`04` §7 E4) as a speed lever | The renderer gets 1 of 8 GPUs in NVIDIA's topology, so its share is small. It changes the pixels the policy sees, i.e. the instrument (validity risk, ladder + verdict). Keep it optional, as the plan does |
| Enable expert graphs or batching inside AlpaGym for S4 training rollouts | Modifies AlpaGym internals (binding rule). Public AlpaGym, code-frozen 07-08, predates PR #36 |
| Build S7 on Holoscan SDK 5.0 | Six days old, a ground-up rewrite with Beta APIs and listed Thor known issues. S7 has no sensor I/O; AlpaSim's interface is gRPC. Borrow its design vocabulary (bounded queues, temporal contracts, clock identities); keep Holoscan for S8's HSB path |
| Contribute Alpamayo 1.5 or Edge export to TensorRT-Edge-LLM | Races NVIDIA: issue #130 has promised 1.5, and external PR #260 already carries an SM110 Alpamayo plugin. Consume, and report what breaks |
| CUDA Green Contexts for P-009's contention arm | Green Contexts partition SMs, not DRAM bandwidth, so they do not model memory contention. Developer preview in Holoscan 5.0. A synthetic copy kernel is the right stand-in |
| Torch-TensorRT compile of the PyTorch driver | v2.14 implies a newer torch than the pinned 2.8.0 (environment drift); it changes numerics; Edge-LLM is NVIDIA's edge path anyway |
| Write the latency-injection shim in C++, or fork AlpaSim's runtime | The mechanism exists in AlpaSim's Python runtime (`planner_delay_us`). A per-step extension is a small upstream PR. C++ belongs on the measurement side |
| HSB emulator, or sensor-injection HIL, in S7 | Adds hardware, scope and Thor SIPL risk. That is S8's territory (P-006). S7's lockstep through gRPC is sufficient for the score question |
| Build an FP8 or camera-batched ViT, or ViT-feature reuse, in S6 | NVIDIA ships these: Edge-LLM 0.10.x FP8 ViT attention, DART, ViT embedding cache, media-aware KV reuse; ModelOpt 0.47 FP8 ViT recipes (P-008 stronger) |
| Quote cloud (SM120) FP8 results as Thor results | Forbidden by plan §6; different kernels, tactics and device |

---

## 6. Open questions, and what would resolve them

1. **Driver share *f* and per-stage split under PhysLoop's 1-GPU topology.** Resolved by L7-1's profile in S2 (nsys + CUDA events + AlpaSim telemetry).
2. **Does `enable_diffusion_expert_cuda_graph` coexist with `force_determinism` and the driver's worker thread, and stay bitwise end to end?** Resolved by L7-2's bitwise test on a scene subset, before any sweep.
3. **Will NVIDIA's post-challenge AlpaSim sync (after 10-31) wire graphs or batching into the driver?** Re-read AlpaSim at S2's start (the plan already requires this). Do not open the PR if it is landing.
4. **Edge:** weights, licence, AlpaSim adapter, runtime (PyTorch, or Edge-LLM / TensorRT), and what "10Hz" assumes (cameras, reasoning on or off, precision). Resolved by the KC-E re-check on 10-31, the Edge model card, and NVIDIA's AV forum.
5. **Does PR #45's forked Edge-LLM build for x86 SM120?** It decides L7-6's feasibility for the student. Read the branch's build files; a CPU-only check.
6. **When does a TensorRT 11.x JetPack release ship?** Watch the TRT release notes. L7-7 makes S6 indifferent to it.
7. **DriveOS 7.2.5 vs 7.2.6 for TRT 11.x.** Conflicting docs; only matters for DRIVE claims, which PhysLoop does not make. Resolved via DRIVE program docs.
8. **Do rented vast.ai hosts expose GPU performance counters (`ncu`, ERR_NVGPUCTRPERM) and allow clock locking?** Test on the first S6a rental. If not, pick a VM-type instance and record clocks instead (P-009 iii).
9. **Is `DelayBuffer.item_at` queried only at control steps?** Read `event_loop.py`'s call site and write a CPU test. This decides whether sub-tick latencies are resolvable in S7 (★ design input).
10. **Does ModelOpt 0.47's `examples/alpamayo/qad.py` cover 1.5 or the 2B student, or only R1?** Read the example's README. It matters for S6's NVFP4 QAD path.
11. **Does the public AlpaGym `perf/` module report trainer idle and rollout supply directly?** Read `alpagym/perf/`; needed for L7-10.
12. **PR #260's provenance ("reviewed plugin implementation") and its merge.** Watch the PR. If it merges, S6 consumes it instead of writing a gate/up fusion.
13. **Dev-kit US stock and lead time on purchase day** (P-002). NVIDIA marketplace and authorized distributors.
14. **Is NuRec rendering deterministic run-to-run?** PR #149 says "not bit-exact". This sets S7's TOST noise floor. Resolved by a repeat-render hash test in S2.
