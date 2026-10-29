# Track A: state of the art for driving-VLA research and on-device deployment (verified to 2026-10-28)

**Scope.** This track covers what NVIDIA and others ship for driving-VLA research and on-device serving, as deltas since the 10-27 ledgers plus the nine named questions. It is evidence only. Nothing is scored and no candidate is picked.

**Method.** All checks ran read-only on 2026-10-28; the Hugging Face probes ran at 16:18 UTC. Sources used:
- GitHub MCP read tools and the unauthenticated REST API;
- raw files at named commits;
- the Hugging Face API, the NGC catalog API and the arXiv API;
- PyPI JSON, NVIDIA docs and blog pages, and company newsrooms.

Nothing was signed into, accepted, downloaded from gated sources, posted or rented.

**Labels.**
- **reported:** a source states it.
- **inferred:** my derivation, with the arithmetic in `capstone_scratch/A/arith_A.py`; run it with `python3 -I arith_A.py`, output in `arith_A.out.txt`.
- **unverified:** what I tried is stated.

Every "checked" date is 2026-10-28 unless a row says otherwise.

**Raw evidence and copies:** `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/capstone_scratch/A/`. It holds:
- `edgellm_1028/`: Edge-LLM docs fetched 10-28. The Alpamayo, pi0.5, supported-models and support-matrix pages are byte-identical to the 10-27 cache.
- `pr45/`: README, `run.py`, `core.py`, `prepare.py` and `pyproject.toml` at `fbed0a7`.
- `arxiv/`: FlashDrive and FlashRT-PAI HTML-to-text, plus the RTC abstracts.
- `flashrt/`: FlashRT `docs/alpamayo15.md`.
- `hf/`: the HF JSON and the Qwen3-VL-2B config.
- `nv/`: the JetPack, TensorRT and ModelOpt pages.
- `tesla/`: the Q3 2026 deck as text.

**Suspicious content.** None observed. No fetched page addressed an AI or asked for an action.

---

## 1. Takeaways

1. **Almost nothing on the inference path moved between 10-27 and 10-28.** The one change is recipes PR #44, a training-efficiency document. Still true:
   - TensorRT-Edge-LLM is at v0.11.0, with no `main` commit since 10-20.
   - PRs #42, #45 and #260 are unchanged.
   - JetPack 7.2.1 (TensorRT 10.16.2) is the newest JetPack, and TensorRT 11.4.0 the newest TensorRT.
   - AlpaSim has had no push since 10-20, and alpamayo1.5 no commit since 09-09.
   - Alpamayo 2 Edge is still not public: 13 guessed HF IDs return 401 and NGC lists 0 Alpamayo models.
   - [Q1.1, Q1.2, Q2.6, Q2.7, Q3.1, Q6.1, Q6.2, Q7.3, Q9.1]
2. **The stock Edge-LLM runtime rejects running its action runner together with context (KV) reuse, and speculative requests that carry images bypass reuse.** So on the Alpamayo path, upstream does not provide FlashDrive-style streaming reuse as shipped. Only the separate ViT-embedding cache is independent of this, and the docs do not say whether the action runner uses it. This contradicts the S6 card's plan to "consume" media-aware KV reuse. [Q1.5, Q1.6, §3 C3]
3. **PR #45's "171 ms FP8" covers one narrow configuration** [Q2.1–Q2.5]:
   - **Device and software:** DRIVE AGX Thor at 1,575 MHz on DRIVE OS 7.0.5 with TensorRT 11.2. DRIVE OS 7.2.6 with TensorRT 11.3 builds but was not re-timed.
   - **Shapes:** B = 1, 6 candidates × 64 steps, 5 flow steps with fixed noise.
   - **Language:** greedy Domino speculation (8.30 accepted tokens per round, at most 128 new tokens) and a 3,006-row full prefill on every request.
   - **Vision:** only the newest frame is encoded.
   - **Timer:** it starts at token handoff, so prompt building and the three older ticks are excluded.
   - **Runtime:** Python, with a C++ runtime "planned". The forked Edge-LLM is exactly PR #260's head commit.
   - **Model:** the README names it the "Alpamayo-R1 2B distill", and the code loads it through `alpamayo_r1`.
4. **The strongest public Jetson AGX Thor numbers for any Alpamayo model are still FlashRT's, for the 10B teacher.**
   - 811 ms with official weights (10 Euler steps, a 128-token budget, 1 sample).
   - 345 ms in streaming mode with fine-tuned weights.
   - Its published stage times put the FP8 expert at about 66–79% of the 273 GB/s bound and W4A8 decode at about 75% (inferred). The teacher is already close to its bandwidth bound on Thor.
   - [Q4.2–Q4.6]
5. **No public latency was found for any 2B Alpamayo student on Jetson AGX Thor, and no 2B Alpamayo weights are public.**
   - NVIDIA's own Jetson AGX Thor numbers for the student backbone's architecture (Qwen3-VL-2B, Edge-LLM v0.10.0): 28.0 ms for an NVFP4 prefill of 2,048 tokens and 7.35 ms per decoded token.
   - So 16 chain-of-causation (CoC) tokens without speculation would take about 118 ms (inferred), making reasoning the largest term of the student's tick on Jetson, with Qwen3-VL-2B as the proxy. PR #45's Domino speculation (on DRIVE) is the incumbent counter-lever.
   - [Q1.10, Q7.3, §4.1]
6. **FlashDrive is still v1 (08-13), and its code and checkpoints are public** (`z-lab/flashdrive`, which cites CoRL 2026).
   - **Per-technique savings over its 513.3 ms system-optimized baseline (RTX PRO 6000):** streaming 157.5 ms, speculative reasoning 112.0 ms, velocity-reuse steps 62.9 ms. CUDA graphs and fusion save 203.6 ms against the raw baseline, and W4A8 24.6 ms.
   - **Jetson Thor:** 3,770 → 944 ms.
   - **Closed loop:** 100 AlpaSim clips, unpaired, no CI.
   - [Q5.1–Q5.6]
7. **The nearest scoop risks are dated.** [Q1.3, Q1.4, Q2.1, Q7.1]
   - NVIDIA promised Edge "later this month", i.e. by 10-31.
   - Edge-LLM's Alpamayo 1.5 support is overdue against a July 30 statement ("planned on August release") by `linc-nv`, the assignee of Edge-LLM's roadmap issue #32. That this is NVIDIA staff is inferred.
   - It is 29 days since v0.11.0, already longer than each of the three completed gaps before it (20, 22 and 26 days).
   - PR #260's merge would move PR #45 onto upstream Edge-LLM, and #45's C++ runtime is "planned".
8. **Industry primary sources publish architecture, not deadlines.** [Q8.1–Q8.9]
   - **Waymo:** a fast sensor-fusion path, a slower driving VLM, and a separate onboard trajectory validator.
   - **XPeng:** VLA 2.0 removes the language step.
   - **Li Auto, Wayve, comma.ai:** compute disclosed, latency not.
   - **Robotics stacks** do publish Jetson AGX Thor latencies: GR00T N1.7 at 39.9 ms (4 steps, 1 camera), π0.5 at about 49 ms.
   - **Physical Intelligence** publishes how to act under more than 300 ms of inference delay (RTC), counting the delay in controller steps.
   - **Tesla:** no technical primary source from 2025–26 was retrievable.
9. **ModelOpt's `examples/alpamayo` covers Alpamayo 1 only.** It defaults to fake quantization, and its QAD trains only the VLM (prompt-only KL, expert frozen). The 0.48 and 0.49 changelog sections list no Alpamayo item. [Q7.7]
10. **Per-row RNG streams for batched flow sampling already exist upstream as an open PR** (alpamayo1.5 #26, by NVIDIA's Karkus, opened 06-29). The per-sample-generator half of the batching lever (L7-4) is therefore not open work. [Q3.3]

---

## 2. Findings table

Columns: ID | claim | source date | checked | link | label. "Thor" always names the device as the source does.

### Q1: TensorRT-Edge-LLM

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q1.1 | The latest release is **v0.11.0 (2026-09-29)**; none after it. Its notes do not mention Alpamayo. It adds: IGX Thor, DFlash2, experimental pi0.5 FP16 policy inference (LIBERO, DROID and ALOHA contracts), in-flight batching for the server, and "Thor-optimized NVFP4-A16 dense and routed-MoE kernels". Earlier releases: v0.10.1 (09-03) added end-to-end FP8 ViT attention, DART pruning, the media artifact cache and SM110 dense NVFP4-A16 GEMM/GEMV. v0.10.0 (08-12) added media-aware KV reuse and Cosmos3-Edge policy/action generation | 2026-09-29 | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/releases | reported |
| Q1.2 | `main`'s last commits are `5cb4d258` (#258) and `1231d641` (#259), both CI, on 2026-10-20; nothing since. PRs updated since 10-27: #261 and #262 (CI). No commit or PR since 10-27 touches Alpamayo, VLA, action experts or Thor | 2026-10-20 | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/commits/main ; https://github.com/NVIDIA/TensorRT-Edge-LLM/pulls | reported |
| Q1.3 | **Issue #130** is open, labelled "New Model" and "Roadmap", last updated 09-15. On 07-30 the account `linc-nv` wrote "Alpamayo 1.5 is planned on August release." and said the Python interface should already work. Users asked again on 08-12 and 09-15 with no reply. The releases on 08-12, 09-03 and 09-29 shipped without 1.5 | 2026-07-08 → 09-15 | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130 | reported. That `linc-nv` is NVIDIA staff is inferred from the handle (`author_association` NONE) and from being the assignee of roadmap issue #32 (Q1.14) |
| Q1.4 | **PR #260** is open, `mergeable_state` blocked: 1 commit, 5 files, +1,373 lines, last updated 2026-10-21T17:39Z. Head: `jin-yc10:alpamayo-fused-gate-up` @ `2f932f37`. The body says it carries the reviewed plugin with dynamic-shape, serialization, CUDA graph and input-contract tests | 2026-10-21 | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260 | reported |
| Q1.5 | **KV-reuse support matrix:**<br>• "Action runner": **Rejected**. The doc says the runtime cannot initialize an action runner and generalized context reuse together.<br>• Speculative requests that contain image or audio input: **Bypassed** (they run cold).<br>• Vanilla image input: supported in the C++ runtime. The cache identity is the decoded image bytes plus placeholder positions; changing content or order invalidates the prefix.<br>• Tensor parallel: not supported | doc @ `main` | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/kv-cache-reuse.md | reported |
| Q1.6 | **The encoder embedding cache is separate from KV reuse.**<br>• Default budget 256 MiB, LRU, keyed on a hash of decoded image bytes; identical media skips the ViT.<br>• `--encoderCacheBudgetBytes` sets the budget; the Python API does not expose it.<br>• Hit and miss counters are not in the profile JSON.<br>• **Whether the action runner (Alpamayo) path uses this cache is not stated** | doc @ `main` | 2026-10-28 | same as Q1.5 | reported; the action-runner interaction is **unverified** (docs silent; code not read) |
| Q1.7 | **Alpamayo path:**<br>• `alpamayo_r1` (Alpamayo-R1-10B) only. The doc states "Only FP16 is supported for Alpamayo export"; engines are built on the Thor device.<br>• The example builds both the LLM and action engines with `--maxBatchSize 6`. The example request uses temperature 1.0, top-p 1.0, top-k 50 and `max_generate_length` 128.<br>• The Alpamayo page documents no CUDA-graph, speculative-decoding or streaming flags.<br>• The file is byte-identical to the 10-27 cache | doc @ `main` | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/alpamayo.md | reported. CUDA graphs and speculative decoding on the Alpamayo path: **unverified** (not documented) |
| Q1.8 | **pi0.5 runtime** (experimental, FP16, `-DBUILD_EXPERIMENTAL_MODELS=ON`):<br>• Four TensorRT components: visual, prefix, action and `cond`. `cond/` precomputes the AdaRMS schedule and re-runs only when the step count or batch changes.<br>• `--steps` sets the denoise count, `--cudaGraph` captures the loop, and `--batch B` replicates the request.<br>• Parity check: the reference's `x_0` is injected, because a shared `--seed` does not give a shared trajectory. It reports cosine similarity and max absolute error; task success is not claimed | doc @ `main` | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/pi05.md | reported |
| Q1.9 | **The benchmarks page has v0.10.0 and older sections only (no v0.11.0).** v0.10.0 ran on JetPack 7.2, CUDA 13.2 and TensorRT 10.16. It has no Alpamayo, pi0.5 or Cosmos3-Edge rows. Power mode is not stated; a note says production results may vary with power mode | doc @ `main` | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md | reported |
| Q1.10 | **Qwen3-VL-2B-Instruct on Jetson AGX Thor (v0.10.0).**<br>• `llm_bench`, NVFP4 LLM / FP16 ViT, B = 1: 2,048-token prefill in **28.0 ms** (73,082 tok/s); decode **136.0 tok/s** at a past KV length of 2,048.<br>• INT4 AWQ: 145.3 ms and 135.3 tok/s.<br>• NVFP4, B = 8: 317.7 ms prefill, 564.0 tok/s aggregate decode.<br>• Runtime (`llm_inference`, COCO VLM set, B = 1, 10 warm-up runs), NVFP4: prefill of 292 tokens 12.7 ms; ViT 265 tokens 11.4 ms; decode 151.9 tok/s.<br>• All are real TensorRT engines | doc @ `main` | 2026-10-28 | same as Q1.9 | reported |
| Q1.11 | **DART visual-token pruning** is training-free and needs no engine change. On Qwen3-VL-4B (FP16, single image, 1,995-token prefill):<br>• ratio 0.25: prefill 1.32× faster on Jetson Thor; described as about lossless on MME, MMBench, POPE, GQA and VQAv2;<br>• ratio 0.5: 1.95×, with −1 to −2 points on sensitive benchmarks | doc @ `main` | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/visual-token-pruning.md | reported |
| Q1.12 | **FP8 ViT:** the quantization doc exports with `--visual_quantization fp8` and a multimodal calibration set. ModelOpt 0.47 adds FP8 vision recipes for `qwen3_vl`; 0.48 keeps the vision-block Linears in high precision. **NVFP4-A16 on Thor:** kernels shipped in v0.10.1 and v0.11.0, but no latency is published for them, because the benchmark tables predate v0.11.0 | 2026-09-03 / 09-29 | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/quantization.md ; https://github.com/NVIDIA/Model-Optimizer/blob/main/CHANGELOG.rst | reported; their effect on Thor latency is **unverified** |
| Q1.13 | **Speculative decoding** in Edge-LLM: EAGLE3, MTP, DFlash, DFlash2 (v0.11.0), JetSpec and DSpark. Context reuse is supported for attention-only EAGLE3, DFlash, JetSpec and DSpark on text. Speculative requests that carry images bypass reuse (Q1.5) | releases and doc | 2026-10-28 | Q1.1 and Q1.5 links | reported |
| Q1.14 | **Roadmap issue #32** ("Developer Roadmap … (H1 2026)"), open, assigned to `linc-nv`, last updated 08-11:<br>• Alpamayo 1 is struck through as done in 0.7.1;<br>• "Cosmos 3", "Native Video Processing" and "Agentic Workflow Integration" are still listed;<br>• no Alpamayo 1.5 or Edge item;<br>• the stated cadence is a monthly release | 2026-02-05 → 08-11 | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/32 | reported |
| Q1.15 | **Supported models:** Alpamayo (`alpamayo_r1`, R1-10B), Cosmos3-Edge and pi0.5 (with its own exporter and runtime). Cosmos-Reason2-2B and 8B are listed as VLMs | doc @ `main` | 2026-10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/supported-models.md | reported |

### Q2: alpamayo-recipes PRs #42, #44 and #45

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q2.1 | **PR #45 state:** open, blocked; created and last updated 2026-10-21T17:58Z; 1 commit, 20 files, +3,093 lines. The body claims "~5Hz on Thor" without naming the device. It uses a forked Edge-LLM until the Edge-LLM PR lands, and will switch once that merges. The authors expect the code to work for AM 1.5 Nano / AM 1 Nano but say they have not timed it | 2026-10-21 | 2026-10-28 | https://github.com/NVlabs/alpamayo-recipes/pull/45 | reported |
| Q2.2 | **Settings behind 171.2 ms.**<br>• **Device:** DRIVE AGX Thor, SM 11.0 at 1,575 MHz. Timed on **DRIVE OS 7.0.5 + TensorRT 11.2**. Per the README, DRIVE OS 7.2.6 + TensorRT 11.3 builds and runs the same steps but was not re-timed.<br>• **Engines:** six. B = 1 request.<br>• **Vision:** 4 cameras × 4 frames at 576×320; only the newest frame per camera is encoded, and older frames are reused from earlier requests of the clip.<br>• **Prompt:** 16-point ego history; 3,006 prompt rows with full-attention prefill on every request.<br>• **Language:** Domino drafts 16 tokens, the verifier checks 17 rows, decoding is greedy.<br>• **Action:** 6 candidates × 64 steps, 5 flow steps, fixed noise (`action-noise.bin`, 384 × 2 FP32).<br>• **KV:** one paged FP16 pool per layer, 40 pages × 128 rows.<br>• **Sample:** 260 requests, 20 gold clips × 13 sliding windows. Latency is taken over 240; each clip's opening window is excluded.<br>• **Results:** FP8 median / p90 **171.2 / 173.2 ms**; FP16 220.6 / 225.1. Accepted tokens per round 8.30 (FP8) / 8.44 (FP16). About 10 ms is host work feeding the paged pool.<br>• The README's introduction rounds the FP8 figure to about 175 ms | README @ `fbed0a7` (10-21) | 2026-10-28 | https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/recipes/alpamayo1_x_edgellm/README.md | reported |
| Q2.3 | **Precision per engine (FP8 build):**<br>• visual: FP8 GEMMs with FP8-rounded Q/K;<br>• prefill: FP16 attention, FP8 FFN via `AlpamayoFusedGateUpPlugin`;<br>• `verify_q17`: FP16 attention, FP8 FFN and LM head;<br>• `domino`: FP8 projections and FFN;<br>• `action`: FP8 FFN, FP16 attention.<br>Scales are amax/448 with per-tensor ModelOpt max calibration. These are real TensorRT engines, not emulation | same | 2026-10-28 | same | reported |
| Q2.4 | **Timer scope and gate.**<br>• `request_ms` runs from the token handoff to the trajectory and includes the newest-tick ViT. Prompt building, the older vision ticks and the metric are outside it. Timing is `torch.cuda.synchronize()` plus `perf_counter`.<br>• `MAX_NEW_TOKENS` is 128.<br>• An optional **paired open-loop gate** against a reference run has guideline limits: mean ΔminADE6 0.02, p90 0.10, endpoint 0.05 m. It is "guidance, not a hard gate".<br>• **Not published:** a per-stage breakdown, a profiler trace, the power mode, and a CoC-length summary (per-row tokens are recorded) | `run.py` @ `fbed0a7` | 2026-10-28 | https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/recipes/alpamayo1_x_edgellm/run.py | reported |
| Q2.5 | **Fork and model identity.**<br>• `.gitmodules` points to `jin-yc10/TensorRT-Edge-LLM`, branch `alpamayo-fused-gate-up`, pinned at **`2f932f37`**, which is exactly PR #260's head.<br>• The runtime installs `alpamayo_r1` from `NVlabs/alpamayo` @ `4cda35d` and loads the checkpoint as `AlpamayoR1` (`INFERENCE_MODEL_TYPE = "alpamayo_r1"`; source type `alpamayo_reasoning_vla_expert_distill`).<br>• The README calls the model the "Alpamayo-R1 2B distill".<br>• torch 2.10.0 (cu130). Weights are not distributed; a bundle `MANIFEST.json` hash is given | `fbed0a7` | 2026-10-28 | https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/.gitmodules | reported (fork, pin, model identity). Inferred: the timed checkpoint is probably the AM1/R1-lineage student. Whether it is the AM1- or AM1.5-distilled student is **unverified**: the README never says |
| Q2.6 | **PR #42** is open, head `5ade7d0`, last updated 2026-10-20T23:31Z: unchanged, no weights. **PR #43** (drafters) is open, updated 10-20. Recipes `main` has had no merge since #41 (`2e907dab`, 10-16) | 2026-10-20 | 2026-10-28 | https://github.com/NVlabs/alpamayo-recipes/pull/42 | reported |
| Q2.7 | **PR #44 moved on 10-28.** Its head went from `35ea6cc` to `a0887d4` (9 commits, 06:41–09:57Z).<br>• **Item 6:** regression bands from a reference run of a 2B stage-1 model on 16 nodes at global batch 512: iteration time mean 1.689 s, median 1.531 s, p95 2.170 s; dense MFU 16.3%; OFU 14.5%. Bands use the median, because rare recompiles skew the mean.<br>• **Item 7:** NaN/inf gradient forensics.<br>• **Items 8–10** (benchmark harness, per-component loss logging, profiler) are still "To be added"<br>Training-side only | 2026-10-28 | 2026-10-28 | https://github.com/NVlabs/alpamayo-recipes/pull/44 | reported |

### Q3: alpamayo1.5 PR #36 follow-ups and the AlpaSim driver

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q3.1 | **No follow-up since 10-27.** `main`'s last commits are `24179cfa` (#36, 08-29) and `36aeb4c5` (#38, CUDA graph capture on a non-current device, 09-09). No PR since 09-09; open PRs are #37 (CI) and #26. The repo description now calls the 10B model "Alpamayo 1.5 Nano" | 2026-09-09 | 2026-10-28 | https://github.com/NVlabs/alpamayo1.5/commits/main | reported |
| Q3.2 | **AlpaSim's driver config still does not enable the expert graph,** and alpamayo1.5 is still installed unpinned. Both AlpaSim branch heads are the SHAs read in code on 10-27 (`main` `affc2eab`, `e2e_challenge` `e121e591`) | — | 2026-10-28 | https://github.com/NVlabs/alpasim | inferred (identical SHAs to the 10-27 code read) |
| Q3.3 | **alpamayo1.5 PR #26** (karkuspeter, open since 06-29, blocked): `sample` and `_euler` accept one `torch.Generator` per batch row, so each row's initial noise is deterministic. It adds no `diffusers` dependency | 2026-06-29 | 2026-10-28 | https://github.com/NVlabs/alpamayo1.5/pull/26 | reported |

### Q4: FlashRT PR #229 and FlashRT-PAI

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q4.1 | **Repo `flashrt-project/FlashRT`.** PR #229 was merged 2026-10-20T20:01Z by LiangSu8899: 35 commits, +14,803 lines. All Alpamayo sources sit behind `FLASHRT_ENABLE_ALPAMAYO15` | 2026-10-20 | 2026-10-28 | https://github.com/flashrt-project/FlashRT/pull/229 | reported |
| Q4.2 | **Protocol.**<br>• **Inputs:** 9 PAI-AV clips (one with 120 windows, eight with 60) from t0 = 1.7 s at 0.1 s steps; 4 cameras × 4 frames; a 128-token reasoning budget (the official example uses 256); one trajectory sample.<br>• **Seeds 0–2 (seeded mode):** full requests reseed each window with seed·1000 + k, and a given seed gives identical tokens and trajectories across processes. 573 steady windows per seed, 1,719 per arm.<br>• **Latency:** "model time per window", from tokenized inputs on the host to the trajectory on the host, synchronized. The runner's `request_ms` adds preprocessing and tokenization on top.<br>• **Device:** Jetson AGX Thor, **MAXN with locked clocks (GPC 1,575 MHz, EMC 4,266 MHz)**, on NGC PyTorch 26.05 (torch 2.12.0a0, CUDA 13.2). The JetPack version is not named | doc @ `main` | 2026-10-28 | https://github.com/flashrt-project/FlashRT/blob/main/docs/alpamayo15.md | reported |
| Q4.3 | **Full requests** (official weights, 10 Euler steps), mean / median / p90 and peak memory:<br>• official BF16 (SDPA): 3,322 / 3,168 / 3,755 ms, 21.6 GB;<br>• FlashRT `bf16`: 2,079 / 1,935 / 2,475 ms;<br>• `balanced` (W4A8 decode with ParoQuant weights `z-lab/Alpamayo-1.5-10B-PARO`, FP8 expert, FP8 prefill): 897 / 844 / 1,026 ms, 35.0 GB;<br>• `fast` (adds NVFP4 prefill and an INT4 LM head): **811 / 759 / 924 ms**, 32.0 GB.<br>**Stage split** (CUDA events, one clip, 117 windows, seed 0), `balanced`: vision 121, LM prefill 246, decode 22.1 per token, expert 126 ms. `fast`: prefill 184, decode 20.8 per token.<br>Reasoning averages 16 tokens on that clip and 19 over the 9 clips. A torch.profiler per-phase kernel table is published. These are FlashRT's own CUDA, CUTLASS and FA4 kernels, not TensorRT | same | 2026-10-28 | Q4.1 and Q4.2 links | reported |
| Q4.4 | **Streaming** (the FlashDrive procedure with z-lab fine-tuned weights; 8 steps, velocity reused on steps 3–6; one sample; the opening window of a stream returns no trajectory), mean / median:<br>• FlashDrive (`torch.compile` max-autotune): 958 / 856 ms;<br>• with the FlashRT backend: 458 / 398 ms;<br>• adding INT8 decode, INT8 prefill and an FP8 expert: **345 / 295 ms**.<br>The backend keeps an INT8 LM copy of about 7 GB | same | 2026-10-28 | same | reported |
| Q4.5 | **Quality:** single-sample open-loop ADE against the logged future, paired per (seed, clip, window), with 95% bootstrap CIs over the 9 clips (width about ±0.05 m). Every paired CI includes 0 (`fast` vs official: −0.001 [−0.082, +0.089] m). The `bf16` engine reproduces the official token stream on 2 of 1,719 windows, which the guide attributes to BF16 near-ties. The guide states this is not closed-loop evidence | same | 2026-10-28 | same | reported |
| Q4.6 | **Efficiency against Jetson AGX Thor's 273 GB/s:**<br>• `balanced` FP8 expert: 12.6 ms per step against an 8.35 ms weights-only bound, i.e. **66%** (79% counting a BF16 KV read of about 0.44 GB);<br>• `bf16` expert: 78–86%;<br>• W4A8 decode: about 75%;<br>• `bf16` decode: about 92% | — | 2026-10-28 | `arith_A.py` | inferred |
| Q4.7 | **No FlashRT Alpamayo PR after #229.** PRs #234–#243 (10-21 → 10-28) cover Qwen3-4B on Thor, X-VLA, GR00T N1.5, AMD π0.5, LaWAM and others; 9 merged in that window | 2026-10-28 | 2026-10-28 | https://github.com/flashrt-project/FlashRT/pulls | reported |
| Q4.8 | **FlashRT-PAI** (arXiv 2610.24349 v1, 2026-10-17; Su et al.; its code is the FlashRT repo).<br>• Thor runs at MAXN on JetPack 7.2 and covers π0.5, GR00T N1.7, LingBot-VA and Cosmos3-Edge-Policy. **The paper does not mention Alpamayo.**<br>• 1.3–3.7× faster than official compiled paths at 16-bit; 3.38–12.66× faster than official eager with reduced precision.<br>• LIBERO-10, 4-bit π0.5: 94.04% on Thor vs 92.4% for the official FP16 reference.<br>• Latency comes from recorded inputs offline; closed-loop success is reported for π0.5 and LingBot-VA in LIBERO only | 2026-10-17 | 2026-10-28 | https://arxiv.org/abs/2610.24349 | reported |

### Q5: FlashDrive (arXiv 2608.12932)

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q5.1 | Only **v1 (2026-08-13)** exists. Authors: Zekai Li, Yihao Liang, Hongfei Zhang, Jian Chen, Yesheng Liang, Zhijian Liu | 2026-08-13 | 2026-10-28 | https://arxiv.org/abs/2608.12932 | reported |
| Q5.2 | **Code and pre-trained checkpoints** are at `github.com/z-lab/flashdrive`, as the paper states. The repo's last commit, `26265174` (2026-09-15), updates the arXiv link and a CoRL 2026 citation | 2026-09-15 | 2026-10-28 | https://github.com/z-lab/flashdrive | reported |
| Q5.3 | **Table 1** (RTX PRO 6000, edition unstated; 1 trajectory; Alpamayo 1.5-10B):<br>• baseline 716.9 ms (encode 87.0, prefill 165.3, decode 271.7, action 192.9);<br>• with system optimizations (CUDA graphs, fused QKV and gate/up, max-autotune) 513.3;<br>• **each technique alone on top of the system optimizations:** streaming 355.8, speculative reasoning (DFlash diffusion drafter) 401.3, adaptive steps (velocity reused on 4 of 8 steps) 450.4;<br>• all three: 176.0;<br>• with W4A8: **151.4** (encode 12.0, prefill 47.2, decode 45.3, action 46.9).<br>Open loop over 100 clips and 12k windows: minADE6@6.4 s 0.767 → 0.844 m; minADE1 1.705 → 1.573 m | 2026-08-13 | 2026-10-28 | https://arxiv.org/html/2608.12932v1 | reported |
| Q5.4 | **Savings against the 513.3 ms system-optimized baseline:**<br>• streaming −157.5 ms (30.7%);<br>• speculative reasoning −112.0 ms (21.8%);<br>• adaptive steps −62.9 ms (12.3%);<br>• the three sum to 332.4 ms against 337.3 ms combined, so they are nearly additive.<br>System optimizations save −203.6 ms (28.4%) against the raw baseline; W4A8 saves −24.6 ms | — | 2026-10-28 | `arith_A.py` | inferred |
| Q5.5 | **Table 2:**<br>• "Jetson Thor", 1 sample: 3,770.3 → 943.6 ms (4.0×); 6 samples: 14,596.5 → 1,522.6 ms (9.6×);<br>• RTX 5090, 1 sample: 878.1 → 183.7 ms;<br>• RTX PRO 6000, 6 samples: 2,609.8 → 245.8 ms.<br>**Not stated:** JetPack, TensorRT, CUDA or power mode; no TensorRT is named. The device is given as "Jetson Thor" without a module or kit | 2026-08-13 | 2026-10-28 | same | reported |
| Q5.6 | **Table 3, closed loop in AlpaSim on 100 randomly sampled clips:**<br>• collision 0.19 → 0.15;<br>• off-road 0.41 → 0.32;<br>• wrong lane 0.45 → 0.51;<br>• plan deviation 0.24 → 0.16;<br>• progress 0.85 → 0.85;<br>• per-step rollout time 1,150 → 463 ms.<br>**Not stated:** replicates, pairing, CIs, the AlpaSim commit, or the rollout GPU | 2026-08-13 | 2026-10-28 | same | reported |

### Q6: TensorRT and JetPack

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q6.1 | **JetPack 7.2.1** (Jetson Linux 39.2.1, CUDA 13.2.2, TensorRT 10.16.2) is the only current JetPack. The archive lists 7.2.1, 7.2, 7.1 and 7.0 for Thor. No new JetPack since 10-27 | page as of 10-28 | 2026-10-28 | https://developer.nvidia.com/embedded/jetpack/downloads ; https://developer.nvidia.com/embedded/jetpack-archive | reported |
| Q6.2 | **TensorRT 11.4.0 is the newest.** Docs were last updated 2026-10-05; PyPI 11.4.0.106 was uploaded 2026-10-08, with no later upload. JetPack is not supported in 11.4.0, and Jetson stays on 10.x until "a future TensorRT 11.x JetPack release". The support-matrix filter lists DRIVE Thor on DriveOS 7.2.6+ | 2026-10-05 / 10-08 | 2026-10-28 | https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html ; https://pypi.org/project/tensorrt/ | reported |
| Q6.3 | **No JetPack with TensorRT 11 is released or dated** | — | 2026-10-28 | Q6.1–Q6.2 links | absence in the sources searched (searched: JetPack downloads and archive, TensorRT Jetson migration page, support matrix, PyPI) |
| Q6.4 | **Versions behind the competing numbers:**<br>• PR #45: DRIVE OS 7.0.5 + TensorRT 11.2;<br>• Edge-LLM benchmarks: JetPack 7.2 + TensorRT 10.16;<br>• GR00T tutorial: JetPack 7.2 + TensorRT 10.15.1.29;<br>• FlashRT: CUDA 13.2 on NGC PyTorch 26.05, no TensorRT.<br>None names JetPack 7.2.1 | various | 2026-10-28 | R2, E8, I7, F2 | reported |

### Q7: Alpamayo 2 Edge, 2 Super and ModelOpt

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q7.1 | **Announcement** (press release, GTC Berlin, 2026-10-21): a 2-billion-parameter reasoning model built on NVIDIA Cosmos, optimized for DRIVE AGX Thor and "designed to run at 10Hz" for in-vehicle scene understanding and motion planning. **Not stated:** licence, cameras, precision, CoC length, flow steps, runtime configuration, teacher. The blog (Kari Briski, 10-21) promises GitHub and Hugging Face availability "later this month", with no size or Hz figure, and shows no update note on 10-28 | 2026-10-21 | 2026-10-28 | https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion ; https://blogs.nvidia.com/blog/open-models-europe-industry/ | reported |
| Q7.2 | **No GTC Berlin session on Edge was found.** A web search of the session catalog returned only navigation pages | — | 2026-10-28 | https://www.nvidia.com/gtc/session-catalog/sessions/ | **unverified** (the catalog is JS-rendered) |
| Q7.3 | **Not public at 2026-10-28 16:18 UTC.**<br>• **HF:** the author=nvidia search returns 3 Alpamayo models (R1-10B, 1.5-10B, 2-Super; last modified 08-04 / 08-07 / 08-07). **13 guessed IDs return 401**, among them `Alpamayo2-Edge`, `Alpamayo-2-Edge-2B`, `Alpamayo2-Edge-FP8`, `Alpamayo-2-Nano`, `Alpamayo-1.5-2B`, `Alpamayo-R1-2B` and `Alpamayo-1.5-Nano`. The newest nvidia model was created 10-27 (`Metatron-v1`).<br>• **NGC:** 0 Alpamayo models; containers only (the 1.5-10B NIM and the 2-Super NIM, the latter created 10-19 and modified 10-20).<br>• **GitHub:** no Edge repo in NVlabs (the newest repo, 10-22, is unrelated); `alpamayo2`'s last commit is 10-07 | 2026-10-28 | 2026-10-28 | https://huggingface.co/api/models?author=nvidia&search=alpamayo ; NGC search API ; https://github.com/NVlabs | reported (a 401 means private or nonexistent) |
| Q7.4 | **No licence is published for Edge** | — | 2026-10-28 | Q7.1 and Q7.3 links | **unverified** (nothing to read) |
| Q7.5 | **The same press release announces:**<br>• the **Alpamayo Agentic Factory**, an open blueprint expected in early 2027, in which agents connect data preparation, post-training and simulation evaluation;<br>• NIMs for Alpamayo 1.5 and the NuRec Harmonizer.<br>Both are in the 10-25 lane-A notes | 2026-10-21 | 2026-10-28 | Q7.1 link | reported |
| Q7.6 | **Alpamayo 2 Super:** the HF card was last modified 2026-08-07 (unchanged), and a NIM container was published 10-19/10-20. There is no new weight. Its auto-labeler role per NVIDIA's blog is from the 10-25 ledger A2 and was not re-fetched | 2026-08-07 / 10-20 | 2026-10-28 | https://huggingface.co/nvidia/Alpamayo2-Super | reported |
| Q7.7 | **ModelOpt `examples/alpamayo` (`main`):**<br>• titled "Quantizing Alpamayo 1"; tested at `NVlabs/alpamayo` @ `4cda35d` with `--ckpt nvidia/Alpamayo-R1-10B`;<br>• FP8, NVFP4 and AutoQuantize. The default saves **fake-quantized** weights; `--real-quant` packs NVFP4. The vision tower is kept in high precision;<br>• QAD (0.47.0) distils the VLM only: a prompt-only KL up to `<\|cot_start\|>`, the expert frozen, no rollout sampling.<br>• **Versions:** PyPI's newest stable is 0.47.0 (09-23), with 0.48.0rc1 (10-19). The CHANGELOG lists 0.48.0 as "2026-10-xx" and 0.49.0 as "2026-11-xx", with no Alpamayo entry.<br>There is no Alpamayo 1.5 or Edge example | 2026-09-23 / 10-19 | 2026-10-28 | https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/alpamayo ; https://pypi.org/project/nvidia-modelopt/ | reported |

### Q8: Others' shipped practice (primary sources, 2025–2026)

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q8.1 | **Waymo** (AI Team blog).<br>• The foundation model uses a "Think Fast and Think Slow" design. A Sensor Fusion Encoder handles rapid reactions; a Driving VLM, trained using Gemini, handles complex semantic reasoning. Both feed a World Decoder.<br>• Teachers too large for real-time onboard use are distilled into onboard students, and the onboard architecture mirrors the foundation model.<br>• A **separate onboard validation layer verifies** the generative model's trajectories.<br>• Simulator and Critic teachers and students exist; critic students flag problem scenarios in logs.<br>• **No latency, deadline, frequency or compute numbers.** Which component runs onboard is not stated | 2025-12-09 | 2026-10-28 | https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving | reported |
| Q8.2 | **XPeng** (press release, AI Day).<br>• VLA 2.0 follows a "Vision-Implicit Token-Action" path that removes the language-translation step.<br>• Vehicle-side parameter count in the billions; a 72B cloud base model.<br>• It runs on the Ultra variant at 2,250 TOPS on XPeng Turing chips, with XPeng's own compiler. Training data: about 100M clips.<br>• Rollout: pioneer users by end of December 2025, Ultra models in Q1 2026. VW is the launch customer.<br>• **No latency or frequency** | 2025-11-05 | 2026-10-28 | https://www.xpeng.com/news/019a56f54fe99a2a0a8d8a0282e402b7 | reported. The "language is a bottleneck" wording itself remains secondary (Electrek, 10-25 ledger D3) |
| Q8.3 | **Li Auto** (Li i8 launch, SEC Exhibit 99.1).<br>• The VLA driver model combines spatial, linguistic and behavioural intelligence, and is described as reasoning and communicating.<br>• Li AD Max runs on an NVIDIA Thor-U at 700 TOPS. Rollout starts with i8 deliveries on 2025-08-20.<br>• **No latency, frequency or parameter count.** MindVLA's internals (GTC 2025) appear only in secondary sources | 2025-07-29 | 2026-10-28 | https://www.sec.gov/Archives/edgar/data/1791706/000110465925071734/tm2521968d1_ex99-1.htm | reported (release); MindVLA specifics **unverified** |
| Q8.4 | **comma.ai.**<br>• "chestnut": a 1B-parameter openpilot model on an external GPU (the kit uses an RX 9060 8 GB) next to the comma four. It has 30× the parameters and 100× the FLOPs of the latest on-device model. Earlier on-device models used about 10 W.<br>• The fleet logs about 1M driving minutes a day, and 15% of drives are ingested for training.<br>• It is slated for openpilot 0.11.2, so unshipped on 08-12. **No latency, frame rate or shadow-mode statement.**<br>• openpilot 0.9.9 (2025-06-19) learns each car's lateral delay online, initially in passive shadow (10-27 practitioner notes; not re-fetched) | 2026-08-12 | 2026-10-28 | https://blog.comma.ai/chestnut/ ; https://blog.comma.ai/099release/ | reported (chestnut); 0.9.9 from the 10-27 notes |
| Q8.5 | **Wayve** (Gen 3).<br>• Built on DRIVE AGX Thor (up to 2,000 FP4 TFLOPS) to run "advanced generative AI models in real-time".<br>• Language-based interaction is a goal (it links LINGO-2), but the post does not say language runs onboard.<br>• **No latency, frequency or compression statement.** GAIA-3 (2025-12-02) is an evaluation world model (10-25 ledger D4) | 2025-09-19 | 2026-10-28 | https://wayve.ai/thinking/wayve-gen-3/ | reported |
| Q8.6 | **NVIDIA GR00T N1.7 on the Jetson AGX Thor dev kit** (Jetson AI Lab).<br>• Setup: 4 denoising steps, 1 camera, B = 1, JetPack 7.2, TensorRT 10.15.1.29, MAXN with `jetson_clocks`. Values are medians of 20 runs after 5 warm-ups.<br>• End to end: PyTorch eager about 126 ms (8.0 Hz); `torch.compile` about 105; TensorRT bf16 about 81; TensorRT FP8 about 44; TensorRT mixed NVFP4 **about 40 ms** (39.9 median, 25.1 Hz).<br>• Split at mixed NVFP4: backbone 13.6, action head 17.2 ms (a TensorRT CUDA graph) | undated (repo pinned at a 2026-07-08 commit) | 2026-10-28 | https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/ | reported |
| Q8.7 | **openpi π0.5 on Jetson AGX Thor** (Jetson AI Lab).<br>• Setup: two 224×224 images, JetPack 7.2, MAXN.<br>• PyTorch BF16 about 132 ms; TensorRT FP8 about 54; TensorRT FP8 + NVFP4 **about 49 ms** (48.84 ± 0.16). Cosine similarity 0.9946 against the reference.<br>• Denoising steps and the TensorRT version are not stated | undated (pin 2026-06-16) | 2026-10-28 | https://www.jetson-ai-lab.com/tutorials/openpi_on_thor | reported |
| Q8.8 | **Physical Intelligence.**<br>• RTC freezes the actions due during the inference delay and inpaints the rest. The delay is counted in controller steps, works with any flow or diffusion VLA, and was shown with π0.5 under more than 300 ms of delay.<br>• Training-time RTC removes the inference-time overhead, shown on π0.6 | 2025-06-09 (v2 2025-12-05); 2025-12-05 | 2026-10-28 | https://arxiv.org/abs/2506.07339 ; https://arxiv.org/abs/2512.05964 | reported |
| Q8.9 | **Tesla.** The Q3 2026 Update deck (PDF created 2026-10-22, +07:00) has business metrics only: FSD attach rate, Cybercab deployment in Austin, training-compute build-out. It says nothing on shadow mode, triggers, the data engine, onboard reasoning or latency. tesla.com's CVPR 2026 event page returns 403. The ICCV 2025 WDFM-AD page lists Elluswamy as a keynote speaker with no abstract | 2026-10-22 | 2026-10-28 | https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q3-2026-Update.pdf ; https://wdfm-ad.github.io/iccv25/ | reported (deck); talk content **unverified** (secondary only) |

### Q9: AlpaSim

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| Q9.1 | **Nothing new.** `pushed_at` is 2026-10-20T13:35:17Z. `main` head `affc2eab` (08-18); `e2e_challenge` head `e121e591` (10-20). The latest release is v2026.5 (2026-06-04). No new commit, release or tag since 10-27 | 2026-10-20 | 2026-10-28 | https://github.com/NVlabs/alpasim ; https://github.com/NVlabs/alpasim/releases | reported |
| Q9.2 | **Activity since 10-27:**<br>• PR #228 (open, 10-28, external): "Add nuPlan leaderboard reference scores";<br>• issue #227 (closed 10-28): a challenge warm-up failure;<br>• #74 closed 10-28.<br>None touches external-driver mode or `planner_delay_us` | 2026-10-28 | 2026-10-28 | https://github.com/NVlabs/alpasim/pull/228 ; https://github.com/NVlabs/alpasim/issues/227 | reported |
| Q9.3 | **External-driver mode and `planner_delay_us` are unchanged** (same SHAs as the 10-27 code read) | — | 2026-10-28 | Q9.1 link | inferred |

### Other

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| X.1 | **arXiv API, submissions since 2026-10-20:**<br>• `all:Alpamayo`: only 2610.28184 (already in the ledgers);<br>• `all:AlpaSim`: 2610.28696 (PCLA-Bridge, cross-domain deployment of CARLA agents; not read);<br>• no hits for VLA + driving + latency, "Jetson Thor" + driving, or "DRIVE Thor".<br>TacitVLA (2610.17540) is still v1 | 2026-10-20 | 2026-10-28 | https://export.arxiv.org/api/query | reported (indexing may lag) |
| X.2 | **Qwen3-VL-2B-Instruct text config:** hidden 2,048, 28 layers, 16 heads, 8 KV heads, head_dim 128, intermediate 6,144, vocab 151,936, tied embeddings. That gives 1.409e9 non-embedding plus 0.311e9 embedding parameters. **Decode bound** on 273 GB/s at a past KV of 2,048: **5.80 ms per token** with an FP16 LM head (NVFP4 body including scales), or 4.09 ms with an NVFP4 head. Edge-LLM's 7.35 ms is therefore 79% or 56% of the bound; the LM-head precision in the v0.10.0 row is not stated | config @ `main` | 2026-10-28 | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json | reported (config); inferred (bound) |

---

## 3. Deltas against the 10-27 ledgers

### 3.1 What changed since 10-27
- **PR #44 only.** Its head moved `35ea6cc` → `a0887d4` on 10-28, adding items 6 (regression bands) and 7 (NaN/inf forensics). Items 8–10 (benchmark harness, per-component loss logging, profiler) are still placeholders, so the race note "do not build a generic harness" stands. [Q2.7]
- **The rest was re-confirmed unchanged on 10-28:**
  - Edge-LLM v0.11.0, with no `main` commit since 10-20;
  - PRs #260, #42, #43 and #45;
  - alpamayo1.5 (last commit 09-09);
  - both AlpaSim branches and its releases;
  - JetPack 7.2.1 and TensorRT 11.4.0;
  - ModelOpt 0.47.0 / 0.48.0rc1;
  - the HF Alpamayo list, with Edge still absent;
  - NGC and the NVlabs repos;
  - TacitVLA and FlashDrive, both still v1. [§2]

### 3.2 What the ledgers had wrong or imprecise

| # | Ledger or plan text | Correction (2026-10-28) | IDs |
|---|---|---|---|
| C1 | "NVIDIA's 2B students run at 171 ms (FP8) on DRIVE AGX Thor" (`02` §1 and S6 Why; `01` C2 and C11; `04` §9); "DRIVE OS 7.0.5 / 7.2.6, TensorRT 11.2 / 11.3" (L5 1.13) | One configuration was timed: DRIVE OS 7.0.5 with TensorRT 11.2; 7.2.6 with 11.3 builds but was not re-timed. The timer excludes prompt building and the three older vision ticks. The README labels the model "Alpamayo-R1 2B distill", and the runtime pins `NVlabs/alpamayo` @ `4cda35d` and loads it as `AlpamayoR1`. Which teacher's student was timed is not stated. One discriminator (inferred): PR #42's AM1.5 recipe pins alpamayo1.5 @ `36aeb4c5` (L5 1.6), whereas PR #45 pins the R1 package. The README's introduction rounds the figure to about 175 ms | Q2.2, Q2.4, Q2.5 |
| C2 | "External PR #260 adds an SM110 Alpamayo plugin" (`04` §3; 10-25 B5) | Its head commit `2f932f37` is exactly the submodule pin of NVIDIA's own PR #45. #45's body says it will switch to upstream once that PR merges. So #260 is the dependency of NVIDIA's Thor recipe, not a side contribution | Q1.4, Q2.5 |
| C3 | S6 Build: consume TensorRT-Edge-LLM's media-aware KV reuse (v0.10.0), FP8 ViT attention and the ViT-embedding cache rather than rebuild them (P-008 note; V28) | Upstream's own support matrix **rejects** an action runner together with generalized context reuse, and **bypasses** reuse for speculative requests that carry images. As shipped, media-aware KV reuse cannot be used on the Alpamayo (action-runner) path. The encoder-embedding cache is separate; whether the action runner uses it is undocumented | Q1.5, Q1.6 |
| C4 | `01` C10 and S6: "the expert still runs at about 20% of its bandwidth bound" | That figure comes from RTX PRO 6000 (FlashDrive-derived). On **Jetson AGX Thor**, FlashRT's published stage times put the teacher's FP8 expert at about 66–79% of the bound and the BF16 expert at 78–86% (inferred). The "expert-efficiency diagnosis" is largely answered for the 10B teacher on Thor; it remains open only for the 2B student's small expert (about 0.116B parameters, L5 1.3i) | Q4.3, Q4.6 |
| C5 | L7 3.1: the ModelOpt QAD example "targets R1 rather than 1.5", by inference from the class name | Now **reported**: the README is titled "Quantizing Alpamayo 1" and uses `nvidia/Alpamayo-R1-10B`. The default export is fake-quantized, so any number from it is emulated unless `--real-quant` and a real engine are used | Q7.7 |
| C6 | 10-25 B4 and `04` §9: FlashRT "Jetson AGX Thor (MAXN) … 811–897 ms" | Confirmed, with more detail: locked clocks (GPC 1,575 / EMC 4,266 MHz) on NGC PyTorch 26.05; the JetPack version is unnamed. Latency is model time and excludes preprocessing and tokenization. The stage split is from one clip (117 windows, seed 0), not from the 1,719-window headline. The engine uses FlashRT's own kernels, not TensorRT | Q4.2, Q4.3 |
| C7 | Ledgers recorded issue #130 only as "open" | NVIDIA's `linc-nv` said Alpamayo 1.5 was planned for the August release, and two follow-ups went unanswered. Support is now overdue by three releases | Q1.3 |
| C8 | 10-25 D3 cites Electrek for XPeng's "language is a bottleneck" | XPeng's own release (2025-11-05) states that the language-translation step was removed. The "bottleneck" wording itself remains secondary | Q8.2 |

### 3.3 What is new: it existed before 10-28 but is absent from the ledgers
- The Edge-LLM KV-reuse matrix and encoder-embedding cache [Q1.5, Q1.6].
- NVIDIA's Jetson AGX Thor numbers for Qwen3-VL-2B (the student backbone's architecture) [Q1.10].
- DART pruning's Thor speed-ups [Q1.11].
- Edge-LLM roadmap issue #32 [Q1.14].
- PR #45 detail [Q2.4]: the paired open-loop gate thresholds, `MAX_NEW_TOKENS` = 128, the Python runtime, and the planned C++ runtime.
- alpamayo1.5 PR #26 [Q3.3].
- The FlashRT-PAI paper [Q4.8], and FlashRT's full protocol and profiler tables [Q4.2–Q4.5].
- FlashDrive's public code (`z-lab/flashdrive`, CoRL 2026 citation) and its per-technique savings [Q5.2, Q5.4].
- Thor latencies for GR00T N1.7 and π0.5 [Q8.6, Q8.7].
- PI's RTC [Q8.8].
- comma.ai's chestnut post [Q8.4].
- Primary sources for XPeng and Li Auto [Q8.2, Q8.3].
- The Tesla Q3 2026 deck, which has no technical content [Q8.9].
- AlpaSim #228 [Q9.2].

---

## 4. Raw opportunity and kill signals (unscored)

### 4.1 Gaps nobody has filled
Each is stated as "no public X found on 2026-10-28". Unless a line says otherwise, the search covered:
- the Edge-LLM docs, benchmarks, releases and PRs;
- recipes PRs #42–#45;
- FlashRT PRs #229–#243, its docs and FlashRT-PAI;
- FlashDrive and `z-lab/flashdrive`;
- the HF, NGC and NVlabs listings;
- arXiv since 10-20.

NVIDIA forums were not re-searched.

1. **No public latency for any 2B Alpamayo student on Jetson AGX Thor.** The only 2B number is PR #45's, on DRIVE AGX Thor. [Q2.2, Q4.7, Q5.5]
2. **No public 2B Alpamayo weights.** 13 HF IDs return 401, NGC has 0 models, PR #45's README says its bundle's weights are not distributed, and PR #42 has no weights. So a public "strongest baseline that does not change behaviour" for a 2B student on Jetson is undefined until Edge or a PR #42-trained student exists. [Q7.3, Q2.5, Q2.6]
3. **No public paired closed-loop verdict between a Thor-served (or quantized) Alpamayo and its BF16 reference.** [Q5.6, Q4.5, Q2.4]
   - FlashDrive's AlpaSim check covers 100 clips, unpaired, with no CI, on a workstation GPU.
   - FlashRT and PR #45 are open-loop only.
4. **No public runtime that combines Edge-LLM's action runner with KV or media reuse.** [Q1.5, Q2.2, Q4.4, Q5.3]
   - PR #45 re-prefills 3,006 rows every request and caches only ViT outputs.
   - FlashDrive's streaming reuse needs expert fine-tuning.
   - FlashRT's streaming mode is FlashDrive's procedure with fine-tuned weights.
5. **No public CoC-budget sweep at a fixed on-device latency.** TacitVLA has only the two endpoints, FlashRT fixes a 128-token budget, and PR #45 is greedy with a 128-token cap. [Q4.2, Q2.4; L1-2]
6. **No public glass-to-action (camera to trajectory) latency for any Alpamayo model on either Thor.** PR #45's timer starts at token handoff, FlashRT reports model time with host CPU video decode outside it, and FlashDrive does not define its scope. [Q2.4, Q4.2]
7. **No public sustained (30-min), thermal or energy-per-step figure for Alpamayo on either Thor.** FlashRT locks clocks at MAXN, and the others do not state a power mode. [Q4.2, Q2.4, Q5.5]
8. **No public C++ runtime for PR #45's engines** ("planned"). [Q2.2]
9. **No public on-device reasoning-VLA latency from Waymo, XPeng, Li Auto, Wayve, comma.ai or Tesla** (searched: the sources in Q8.1–Q8.9). Only robotics flow-matching policies publish Thor latencies: GR00T N1.7 at about 40 ms, π0.5 at about 49 ms. [Q8.1–Q8.9]
10. **No public table of precision formats (or token-identity results) for an Alpamayo engine on Jetson AGX Thor using TensorRT.** Edge-LLM supports Alpamayo export in FP16 only; FlashRT has precision tiers on its own kernels. [Q1.7, Q4.3]

### 4.2 Scoop risk: what NVIDIA and others are about to ship

| Item | Signal | Date | IDs |
|---|---|---|---|
| Alpamayo 2 Edge weights | "later this month" | by 2026-10-31 (3 days) | Q7.1, Q7.3 |
| Edge-LLM release | Cadence is monthly. 29 days since v0.11.0, already longer than each of the three completed gaps (20, 22, 26 days). `linc-nv` stated an August target for Alpamayo 1.5 | imminent by cadence (inferred) | Q1.1, Q1.3, Q1.14 |
| PR #260 merge | Moves PR #45 onto upstream Edge-LLM (#45's body) | undated | Q1.4, Q2.1 |
| PR #45 C++ runtime | "A C++ runtime for these engines is planned." | undated | Q2.2 |
| PRs #42 and #43 merge | Distillation recipe and spec-decode drafters | undated | Q2.6 |
| PR #44 items 8–10 | Benchmark harness, per-component loss logging, profiler | undated | Q2.7 |
| ModelOpt 0.48 / 0.49 | CHANGELOG headings "2026-10-xx" and "2026-11-xx"; no Alpamayo item yet | Oct / Nov 2026 | Q7.7 |
| AlpaSim next release | Discloses the "simulation choices" withheld under #224 | undated | Q9.1; L1 A4 |
| Alpamayo Agentic Factory | Agents connecting data preparation, post-training and simulation evaluation | early 2027 | Q7.5 |
| FlashRT | Alpamayo 1.5 already on Thor; about 9 merged PRs a week across models. It could add a 2B student once weights exist (inferred) | ongoing | Q4.7 |
| JetPack with TensorRT 11.x | Promised, undated | undated | Q6.2 |

### 4.3 Baselines a contribution must beat

**(a) Speed on Jetson AGX Thor.** Never compare these rows with (b).

| Row | System and model | Shapes | Precision | Real or emulated | Versions and power | Latency | Trace | Omitted by the source |
|---|---|---|---|---|---|---|---|---|
| J1 | Official Alpamayo 1.5-10B (FlashRT's baseline arm) | B = 1, S = 1, K = 10, budget n = 128 (mean 19), 4 cams × 4 frames | BF16, SDPA | real | MAXN, locked clocks; NGC PyTorch 26.05 / CUDA 13.2 | mean 3,322 / median 3,168 ms | per-phase only for FlashRT arms | T_p, JetPack |
| J2 | FlashRT `fast` (official weights) | as J1 | NVFP4 prefill, INT4 LM head, W4A8 decode, FP8 expert | real (FlashRT kernels) | as J1 | 811 / 759 / p90 924 ms | CUDA events plus torch.profiler (one clip) | T_p, JetPack |
| J3 | FlashRT `balanced` | as J1 | W4A8 decode, FP8 expert, FP8 prefill | real | as J1 | 897 / 844 ms; vision 121, prefill 246, decode 22.1 per token, expert 126 | yes | T_p, JetPack |
| J4 | FlashRT streaming (FlashDrive procedure) | B = 1, S = 1, K = 8 with steps 3–6 reused, budget 128 | INT8 decode and prefill, FP8 expert | real | as J1 | 345 / 295 ms | yes | T_p, JetPack. Changes behaviour: fine-tuned weights |
| J5 | FlashDrive (paper) | S = 1 (or 6), K = 8 with 4 reused, n about 16 | W4A8 plus all techniques | real kernels (described) | none stated | 943.6 ms (1 sample), 1,522.6 ms (6 samples); baseline 3,770.3 / 14,596.5 | none | JetPack, TensorRT, power, T_p, module or kit |
| J6 | Edge-LLM, Qwen3-VL-2B (student-backbone architecture, not an Alpamayo student) | B = 1; prefill 2,048 tokens; decode at past KV 2,048 | NVFP4 LLM, FP16 ViT | real TensorRT | JetPack 7.2, TensorRT 10.16, Edge-LLM 0.10.0 | prefill 28.0 ms; decode 7.35 ms per token | profile JSON, not a trace | power mode |
| J7 | GR00T N1.7 (robotics) | B = 1, 4 steps, 1 camera | mixed NVFP4 | real TensorRT | JetPack 7.2, TensorRT 10.15.1.29, MAXN | 39.9 ms median | per-stage | — |
| J8 | openpi π0.5 (robotics) | 2 × 224² images | FP8 + NVFP4 | real TensorRT | JetPack 7.2, MAXN | about 49 ms | total vs model only | steps, TensorRT version |

**(b) Speed on DRIVE AGX Thor.**

| Row | System | Shapes | Precision | Versions | Latency | Omitted |
|---|---|---|---|---|---|---|
| D1 | PR #45, "Alpamayo-R1 2B distill" (Python runtime plus TensorRT engines) | B = 1; S = 6 × 64; K = 5 with fixed noise; greedy Domino (8.30 accepted per round), at most 128 new tokens; T_p = 3,006 rows; 4 × 4 frames at 576×320, newest frame encoded | FP8 (FP16 attention) / FP16 | DRIVE OS 7.0.5, TensorRT 11.2, 1,575 MHz | 171.2 / p90 173.2 ms (FP8); 220.6 (FP16) | power mode, per-stage split, trace, CoC length, glass-to-action |

**(c) Workstation and datacenter, for context only:** FlashDrive on RTX PRO 6000 (edition unstated), 716.9 → 151.4 ms [Q5.3]. TacitVLA on one H100 (bf16), 127 vs 269 ms (10-27 ledger V8).

**(d) Behaviour-preservation references already public:**
- PR #36: the expert CUDA graph, 160/160 bitwise at batch 1–4 (10-27 ledger V4).
- FlashRT: `bf16` is the numerics reference; per-preset token-identity counts; paired ADE CIs [Q4.5].
- Edge-LLM pi0.5: parity with the reference's `x_0` injected, judged by cosine and max absolute error [Q1.8].
- PR #45: an open-loop paired ΔminADE6 gate (0.02 / 0.10 / 0.05 m, guidance only) [Q2.4].
- alpamayo1.5 PR #26: per-row generators for batched determinism [Q3.3].

**(e) Quality protocols any comparison faces:**

| Source | Loop | Setting | Pairing and intervals |
|---|---|---|---|
| FlashDrive | closed | AlpaSim, 100 clips; replicates and commit unstated | unpaired, no CI |
| FlashRT | open | 9 clips × 3 seeds | paired, bootstrap CI over clips |
| PR #45 | open | minADE6 on 240–260 requests | optional paired gate, no CI |
| TacitVLA | closed | AlpaSim 441 public scenes × 3 rollouts (seed mode not stated) | ± SD, unpaired (10-27 ledger V8) |
| FlashRT-PAI | closed | LIBERO success, π0.5 and LingBot-VA only | — |

### 4.4 Kill and steering signals, grouped by the three paths (raw; not weighed)

**Path 1 (research paper: quality vs latency, CoC, flow steps, precision)**
- No paired closed-loop per-format verdict exists [4.1 #3].
- The CoC curve between TacitVLA's endpoints is unmapped [4.1 #5].
- PI counts the inference delay in controller steps (RTC). That matches AlpaSim's quantized `planner_delay_us` behaviour (10-27 L7 known answer: 139–200 ms all land 2 ticks late at 10 Hz). Inferred link.
- NVIDIA authors are active on this line (TacitVLA, "Perceives but Doesn't Brake"; see the 10-27 ledgers).
- If Edge ships without a runtime-CoC switch, the CoC arm may not exist on Edge (unknown until release) [Q7.1].

**Path 2 (2B student at 10 Hz on Jetson AGX Thor; a win over the strongest behaviour-preserving baseline)**
- No Jetson number and no public 2B weights exist [4.1 #1–2].
- Decode on Jetson AGX Thor for the 2B backbone's architecture is 7.35 ms per token. Without speculative decoding, 16 CoC tokens would take about 118 ms and be the largest term of the student's tick (inferred, Qwen3-VL-2B proxy). PR #45's Domino on DRIVE is the incumbent counter-lever.
- PR #45 re-prefills 3,006 rows each request, and Edge-LLM cannot pair the action runner with KV reuse [C3].
- The C++ runtime for #45 is "planned" (scoop).
- FlashRT shows that someone outside NVIDIA already engineers Alpamayo close to Thor's bandwidth bound [C4].
- Edge-LLM's Alpamayo support is FP16-only, and its Alpamayo 1.5 support is overdue (scoop).

**Path 3 (shadow mode in HCMC, open-loop)**
- Primary 2025–26 sources are thin on shadow mode and triggers:
  - Tesla's retrievable primary sources contain none;
  - comma.ai's passive shadow is used for online delay learning in 0.9.9 (10-27 notes);
  - comma.ai ingests 15% of fleet drives;
  - Waymo's critic models flag problem scenarios offboard.
  [Q8.4, Q8.1, Q8.9]
- FlashRT notes that host CPU YUV→RGB conversion differs by up to 2 levels between aarch64 and x86. Sensor-path differences reach the model inputs [Q4.2 doc §Limitations].

---

## 5. Gaps: what I could not verify, and what I tried
- **Edge's configuration, licence and teacher.** Nothing is published; all channels were checked [Q7.1–Q7.4].
- **Whether Edge-LLM's action runner uses the encoder-embedding cache.** The docs are silent and runtime code was not read [Q1.6].
- **CUDA graphs and speculative decoding on Edge-LLM's Alpamayo path.** Undocumented; pi0.5's `--cudaGraph` was not assumed to apply [Q1.7].
- **Which teacher's student produced PR #45's 171.2 ms.** The README and code point to the R1 lineage but never say [Q2.5].
- **PR #45's power mode, per-stage split, CoC length distribution, and the DRIVE OS 7.2.6 + TensorRT 11.3 timing.** Not published [Q2.4].
- **FlashRT's JetPack/L4T version.** Only CUDA 13.2 / NGC PyTorch 26.05 is stated [Q4.2].
- **FlashDrive's Thor setup** (JetPack, TensorRT, power, module) **and its AlpaSim replicates and commit.** Not in the paper [Q5.5, Q5.6].
- **The effect of Edge-LLM's NVFP4-A16 Thor kernels.** There is no v0.11.0 benchmark section [Q1.9, Q1.12].
- **Edge-LLM benchmark coverage.** Only the Qwen3-VL-2B rows of the v0.10.0 Jetson AGX Thor tables were transcribed. The v0.9.0 and v0.8.0 sections, and other models' rows, were not read [Q1.10].
- **LM-head precision in Edge-LLM's Qwen3-VL-2B NVFP4 row,** which moves the inferred efficiency between 56% and 79% [X.2].
- **A GTC Berlin Edge session.** Only navigation pages came back [Q7.2].
- **Tesla primary technical content from 2025–26.** The event page returns 403, the ICCV page lists the speaker only, X posts are not fetchable, and the Q3 2026 deck has no technical content [Q8.9].
- **Li Auto MindVLA internals.** Secondary sources only; the SEC exhibit has no latency [Q8.3].
- **Wayve inference numbers.** None found; job postings only (secondary) [Q8.5].
- **PCLA-Bridge (2610.28696).** Title only, not read [X.1].
- **Whether the next DriveOS is 7.2.6 or later.** Not re-checked; the 10-27 conflict stands.

---

## 6. Source list
Accessed 2026-10-28 unless noted. "P" marks a primary source.

**TensorRT-Edge-LLM**
- E1. Releases: https://github.com/NVIDIA/TensorRT-Edge-LLM/releases (P)
- E2. `main` commits: https://github.com/NVIDIA/TensorRT-Edge-LLM/commits/main (P)
- E3. Issue #130: https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130 (P)
- E4. PR #260: https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260 (P)
- E5. KV-cache reuse doc: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/kv-cache-reuse.md (P)
- E6. Alpamayo example: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/alpamayo.md (P)
- E7. pi0.5 example: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/pi05.md (P)
- E8. Performance benchmarks: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/performance/performance-benchmarks.md (P)
- E9. Visual-token pruning: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/visual-token-pruning.md (P)
- E10. Quantization: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/features/quantization.md (P)
- E11. Roadmap issue #32: https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/32 (P)
- E12. Pull requests: https://github.com/NVIDIA/TensorRT-Edge-LLM/pulls (P)
- E13. Supported models: https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/supported-models.md (P)

**alpamayo-recipes**
- R1. PR #45: https://github.com/NVlabs/alpamayo-recipes/pull/45 (P)
- R2. PR #45 README @ `fbed0a7`: https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/recipes/alpamayo1_x_edgellm/README.md (P)
- R3. PR #45 `run.py` @ `fbed0a7`: https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/recipes/alpamayo1_x_edgellm/run.py ; also `core.py`, `pyproject.toml` (P)
- R4. `.gitmodules` @ `fbed0a7`: https://github.com/NVlabs/alpamayo-recipes/blob/fbed0a70d5452bc50527e1c11ae6855fccbea58c/.gitmodules (P)
- R5. PR #42: https://github.com/NVlabs/alpamayo-recipes/pull/42 (P)
- R6. PR #44: https://github.com/NVlabs/alpamayo-recipes/pull/44 (P)
- R7. Commits: https://github.com/NVlabs/alpamayo-recipes/commits/main (P)
- R8. PR #43: https://github.com/NVlabs/alpamayo-recipes/pull/43 (P)

**alpamayo1.5 and AlpaSim**
- A1. alpamayo1.5 commits: https://github.com/NVlabs/alpamayo1.5/commits/main (P)
- A2. alpamayo1.5 PR #26: https://github.com/NVlabs/alpamayo1.5/pull/26 (P)
- A3. alpamayo1.5 PR #36: https://github.com/NVlabs/alpamayo1.5/pull/36 (P; 10-27 ledger)
- S1. AlpaSim repo and branches: https://github.com/NVlabs/alpasim (P)
- S2. AlpaSim releases: https://github.com/NVlabs/alpasim/releases (P)
- S3. AlpaSim #228 and #227: https://github.com/NVlabs/alpasim/pull/228 ; https://github.com/NVlabs/alpasim/issues/227 (P)

**FlashRT and FlashDrive**
- F1. FlashRT PR #229: https://github.com/flashrt-project/FlashRT/pull/229 (P)
- F2. FlashRT Alpamayo guide: https://github.com/flashrt-project/FlashRT/blob/main/docs/alpamayo15.md (P)
- F3. FlashRT pull requests: https://github.com/flashrt-project/FlashRT/pulls (P)
- F4. FlashRT-PAI, arXiv 2610.24349: https://arxiv.org/abs/2610.24349 (P)
- D1. FlashDrive, arXiv 2608.12932: https://arxiv.org/abs/2608.12932 and https://arxiv.org/html/2608.12932v1 (P)
- D2. FlashDrive code: https://github.com/z-lab/flashdrive (P)

**TensorRT, JetPack, PyPI**
- T1. JetPack downloads: https://developer.nvidia.com/embedded/jetpack/downloads (P)
- T2. JetPack archive: https://developer.nvidia.com/embedded/jetpack-archive (P)
- T3. TensorRT 10→11 Jetson migration: https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html (P)
- T4. TensorRT support matrix: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html (P)
- T5. PyPI `tensorrt`: https://pypi.org/project/tensorrt/ (P)
- T6. PyPI `nvidia-modelopt`: https://pypi.org/project/nvidia-modelopt/ (P)

**NVIDIA announcements and model listings**
- N1. Press release, 2026-10-21: https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion (P)
- N2. NVIDIA blog, 2026-10-21: https://blogs.nvidia.com/blog/open-models-europe-industry/ (P)
- N3. HF Alpamayo search: https://huggingface.co/api/models?author=nvidia&search=alpamayo (P)
- N4. HF newest nvidia models: https://huggingface.co/api/models?author=nvidia&sort=createdAt&direction=-1&limit=40 (P)
- N5. NGC catalog search: https://api.ngc.nvidia.com/v2/search/catalog/resources/CONTAINER (P)
- N6. NVlabs repos: https://github.com/NVlabs (P)
- N7. ModelOpt `examples/alpamayo`: https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/alpamayo (P)
- N8. ModelOpt CHANGELOG: https://github.com/NVIDIA/Model-Optimizer/blob/main/CHANGELOG.rst (P)
- N9. GTC session catalog: https://www.nvidia.com/gtc/session-catalog/sessions/ (P; JS-rendered)
- N10. alpamayo2 repo: https://github.com/NVlabs/alpamayo2 (P)
- N11. Qwen3-VL-2B config: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json (P)

**Industry**
- I1. Waymo AI Team foundation-model post, 2025-12-09: https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving (P)
- I2. XPeng AI Day press release, 2025-11-05: https://www.xpeng.com/news/019a56f54fe99a2a0a8d8a0282e402b7 (P)
- I3. Li Auto i8 launch, SEC Exhibit 99.1, 2025-07-29: https://www.sec.gov/Archives/edgar/data/1791706/000110465925071734/tm2521968d1_ex99-1.htm (P)
- I4. comma.ai, "Introducing chestnut", 2026-08-12: https://blog.comma.ai/chestnut/ (P)
- I5. openpilot 0.9.9 notes, 2025-06-19: https://blog.comma.ai/099release/ (P; from 10-27 notes)
- I6. Wayve Gen 3, 2025-09-19: https://wayve.ai/thinking/wayve-gen-3/ (P)
- I7. GR00T N1.7 on Jetson Thor tutorial: https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/ (P, NVIDIA)
- I8. openpi π0.5 on Jetson Thor tutorial: https://www.jetson-ai-lab.com/tutorials/openpi_on_thor (P, NVIDIA)
- I9. RTC, arXiv 2506.07339: https://arxiv.org/abs/2506.07339 (P)
- I10. Training-time RTC, arXiv 2512.05964: https://arxiv.org/abs/2512.05964 (P)
- I11. Tesla Q3 2026 Update: https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q3-2026-Update.pdf (P)
- I12. Tesla CVPR 2026 event page: https://www.tesla.com/event/tesla-x-cvpr-2026 (P; 403)
- I13. ICCV 2025 WDFM-AD workshop: https://wdfm-ad.github.io/iccv25/ (P)
- I14. Wayve GAIA-3: https://wayve.ai/press/wayve-launches-gaia3/ (P; 10-25 ledger, not re-fetched)

**Other**
- X1. arXiv API: https://export.arxiv.org/api/query (P)
- X2. TacitVLA, arXiv 2610.17540: https://arxiv.org/abs/2610.17540 (P)

**Repo ledgers used as the baseline (not re-researched):**
- `docs/research/2026-10-27-role-alignment/verification.md`
- `docs/research/2026-10-27-role-alignment/notes/` L1, L5 and L7
- `docs/research/2026-10-25-teacher-to-thor/verification.md`
- `docs/04_CORE_PLAN_TECHNICAL.md`
- `docs/02_PHYSLOOP_PLAN.md` §1 and S6
- `docs/01_SYSTEM_DESIGN.md` C1, C10 and C11
