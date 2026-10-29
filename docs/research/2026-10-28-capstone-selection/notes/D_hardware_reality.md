# Track D — Hardware reality for inference-led AV work (as of 2026-10-28)

**Scope.** Four parts:
- Jetson AGX Thor (sm_110) against the rented proxy GPUs;
- what is rentable today;
- the path (3) shadow-mode rig;
- Vietnamese legal and privacy constraints.

Evidence only. Nothing here scores or picks a capstone path.

**Method.** Read-only throughout:
- NVIDIA docs, datasheets and forums; GitHub (read tools and raw files); vendor store pages; Vietnamese official texts;
- the vast.ai public `bundles` endpoint, with no account and no key.

Nothing was signed into, carted, bought, registered, accepted or downloaded from a gated source.

**Labels:**
- **reported**: a source states it;
- **inferred**: arithmetic or reasoning shown, or pointed to a script line;
- **unverified**: what was tried is stated.

"Checked" is the date I read the source. All checks are 2026-10-28 unless noted. Rows carried from the 10-25 or 10-27 ledgers say so and were not re-fetched.

**Scratch artefacts.** These are not in the repo, which is read-only for this lane. All live under `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/capstone_scratch/D/`:
- `vast_2026-10-28/od_*.json` plus `fetched_at_utc.txt` (2026-10-28 16:15:50 UTC);
- `scripts/stats.py`, a byte-identical copy of the repo's 10-27 `stats.py`;
- `scripts/vast_extra.py`: GPU-weighted medians, host power caps, VM offers, CUDA level;
- `scripts/d_arith.py`, with its output in `scripts/d_arith.out.txt`: every inferred number below;
- `scripts/tables.py` and `scripts/ptxgrep.py`: parse the saved NVIDIA HTML in `dl_cc/` and `dl_ptx/`;
- `dl_src/`: raw upstream source files that were grepped.

Reproduce from a separate directory: `cd <scripts> && python3 -I stats.py <vast dir>`, then `python3 -I vast_extra.py <vast dir>` and `python3 -I d_arith.py`.

---

## 1. Takeaways

1. **The RTX PRO 6000 is not an instruction-set proxy for Thor; B200/B300 are.** In PTX ISA 9.4:
   - `tcgen05.*` (5th-gen tensor core with Tensor Memory) is supported on sm_100a and sm_110a and their families, not on sm_120.
   - Warp-level FP4/FP6 `mma` and the `.kind`/`.block_scale` qualifiers require sm_120a/sm_120f.
   - `wgmma` requires sm_90a.

   So the three proxies each use a different MMA family: H100 uses `wgmma`, RTX PRO 6000 uses block-scaled `mma.sync`, and B200 and Thor use `tcgen05`. Triton's matmul pass, vLLM's CMake and FlashAttention-4's separate sm100 and sm120 kernel files encode the same split. A kernel tuned on the RTX PRO 6000 or on H100 never issues `tcgen05`, Thor's tensor-core path. Plain FP8 `mma.sync` (sm_89+) does run on Thor. [DC2, DL1, DL4, DL5]
2. **Per SM, Thor is datacenter-Blackwell class, about 8× an RTX PRO 6000 SM. In aggregate it is tiny.**
   - Dense FP8 tensor throughput per SM per clock: Thor ≈ 16.4k FLOP, RTX PRO 6000 WS ≈ 2.0k (8.1×; inferred).
   - Thor has 20 SMs, 273 GB/s and a user-reported 32 MiB L2.
   - Its ridge points are 947 (BF16), 1,894 (FP8) and 3,791 (FP4) FLOP/B. The RTX PRO 6000 WS has 279 / 558 / 1,116; B200 has 281 / 562 / 1,125.
   - Thor is therefore far more often memory-bound at the same arithmetic intensity. [DT1, DT2, DT9, DP1, DP4; d_arith]
3. **No green-context or SM-limiting setup on a proxy reproduces Thor. MIG comes closest, and the rentable MIG is on the wrong ISA.**
   - Green contexts partition SMs only, with an 8-SM minimum and 8-SM alignment on CC ≥ 9.0, so 20 SMs is not even expressible.
   - MIG also partitions L2 and DRAM bandwidth. B200 1g.23gb gives 1/7 of the SMs, 1/8 of the L2 and 1/8 of the memory: about 643 FP8 TFLOPS, ~1 TB/s and ~15.8 MB, on the tcgen05 ISA (inferred).
   - RTX PRO 6000 MIG 24 GB slices are rentable on RunPod ($0.50 community / $0.69 secure). They land near Thor on the memory side (~399 GB/s, ~32 MB L2) but run sm_120's ISA. [DC1, DC6, DC7, DR3; d_arith]
4. **Thor's measured efficiency is high on streaming work.**
   - CUTLASS dense FP4 GEMM reached ~875–878 TFLOP/s against the 1,035 peak (84.5–84.8%; forum user, CUDA 13.0, "130 W").
   - TensorRT-Edge-LLM decodes Qwen3-8B NVFP4 at 47.3 tok/s at batch 1 (JetPack 7.2, CUDA 13.2, TRT 10.16). That is 76–94% of the 273 GB/s bound computed from the model's config.
   - cuBLAS FP16 (~130 TFLOPS, ~50%) and a copy kernel (~200 GB/s, ~73%) are user reports with no versions stated. [DT10, DT11, DT15, DQ1; d_arith]
5. **Power mode and DVFS change the device you are measuring.**
   - The default mode is 120 W: GPU 1,386 MHz, 88% of the MAXN clock.
   - MAXN (1,575 MHz) is labelled experimental and throttles at the TDP budget.
   - The 90 W and 70 W modes cut the GPU from 10 to 6 TPCs (12 SMs) and from 4 to 3 FBPs.
   - DVFS is on by default. `jetson_clocks` locked the GPU at 1,574 MHz in a staff reply.
   - A dev-kit user hit over-current throttling with Qwen3-VL-8B on 5 × 4 images in both 120 W and MAXN (JetPack 7.1). [DT3, DT13, DT14]
6. **Library support for sm_110 is real but younger than for B200; it is mostly B200 code rebuilt.** Each project, as of 10-28:
   - **CUTLASS:** its README lists "DRIVE Thor, 11.0, CUDA 13.0"; Thor runs SM100 kernels built for 110a.
   - **FlashAttention 4:** the CuTe-DSL `flash_fwd_sm100.py` asserts the SM 10.x or 11.x family, so Thor takes the B200 path. Separate `flash_fwd_sm120.py` and `flash_bwd_sm120.py` serve the RTX PRO 6000.
   - **FlashInfer:** lists "SM 11.0, Jetson Thor" and gained experimental sm110 TMEM decode kernels between 10-21 and 10-25.
   - **vLLM:** builds its SM100-family kernels for 11.0f.
   - **PyTorch:** its CUDA 13 aarch64 wheel arch list includes 11.0.
   - **SGLang:** its README lists Jetson Orin, not Thor.
   - **TensorRT-Edge-LLM:** still v0.11.0; PR #260 is still open. [DL1–DL8]
7. **What is rentable today (2026-10-28):**
   - vast.ai on-demand, cheapest / per-offer median $/GPU-h:
     - RTX PRO 6000 WS $1.00 / $1.47 (35 offers);
     - Server $1.24 / $1.38 (41);
     - Max-Q $0.83 / $1.60 (30);
     - H100 SXM $1.93 / $3.14 (16; median up from $2.40 on 10-27);
     - H200 $4.34 / $6.83;
     - B200 $7.50 / $8.75;
     - B300 $8.50 / $9.94.
   - No hourly Jetson or DRIVE AGX Thor access was found.
   - The dev kit is still $5,499 MSRP, and SabrePC lists it at $6,268.86, out of stock. [DR1, DR3–DR7, DT5]
8. **Rented "Workstation" cards are often not running at 600 W.** 15 of 35 vast.ai RTX PRO 6000 WS offers report `gpu_max_power` of 350–550 W, and 4 of 41 Server offers report 350–450 W. Editions and power limits must go in the manifest before roofline claims. [DR1; vast_extra]
9. **Vietnam, cross-border transfer (engineering reading, not legal advice).** The signed Decree 356/2025/NĐ-CP:
   - is in force from 2026-01-01, and Decree 13/2023 has ceased to have effect (Art. 42);
   - requires "cơ quan, tổ chức, cá nhân" (individuals included) who transfer personal data across borders to prepare an impact-assessment dossier and file it with the Ministry of Public Security within 60 days (Art. 18). Form 01b is the individual's version;
   - defines a transfer in Art. 17(1)(c) to include processing data collected in Vietnam on a platform abroad;
   - lists in Art. 17(3) no research exemption among its extra exemptions.

   Uploading identifiable HCMC footage to foreign GPU hosts therefore falls on the face of the text. Anonymizing on the device before upload is the design lever; that is an inference. [DV1]
10. **Vehicle and permit (engineering reading, not legal advice).**
    - A smart-vehicle permit appears to require driverless control after 2026-07-01 (10-25 ledger).
    - Press reports on Circular 30/2026/TT-BXD (effective 2026-07-01) say cameras and sensors inside the cabin are not a modification. A roof rack on a passenger car was a simple modification (cải tạo đơn giản) under Circular 47/2024. Official texts were not read.
    - Rig parts: Leopard's Thor page shows LI-IMX728-9295 (120° image only, no prices). The HSB deserializer kit is $388 for one camera, with its power supply on backorder. The ZED-X20P is $299.95, in stock, with no IMU. [DV4, DV5, DG1–DG3]

---

## 2. Findings table

Labels: **R** = reported, **I** = inferred (arithmetic in `d_arith.out.txt` or shown), **U** = unverified (attempt stated).

### 2.1 Q1 — Jetson AGX Thor (T5000 module and dev kit), NVIDIA primary docs

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F1.1 | NVIDIA's Jetson Thor Module datasheet (doc 4767587, Jan 2026) gives the T5000 as: a 2,560-core Blackwell GPU with 5th-gen Tensor Cores; MIG with 10 TPCs; GPU max 1.57 GHz; 14-core Neoverse-V3AE at 2.6 GHz (1 MB L2 per core, 16 MB shared L3); 128 GB 256-bit LPDDR5X at 273 GB/s; 2,070 TFLOPS FP4 sparse; 40–130 W; up to 20 cameras via HSB; 4× 25 GbE; 4× CAN. It prints **no** SM count, **no** tensor-core count and **no** L2 size. The T4000: 1,536 cores, 6 TPCs, 1.53 GHz, 64 GB, 273 GB/s, 1,200 TFLOPS FP4 sparse, 40–70 W | Jan 2026 (PDF metadata 2026-01-22) | 10-28 | [DT1](https://dam-cdn.nvd.orangelogic.com/AssetLink/x20v2h2y15uh0ne54rpb73xh4x7s0m1s.pdf) | R |
| F1.2 | NVIDIA Thor blog cells (per the 10-03 ledger): 2,070 TFLOPS sparse FP4; 1,035 TFLOPS = dense FP4 = sparse FP8 = sparse INT8; 517 TFLOPS = dense FP8 = sparse FP16. So dense FP4 / FP8 = 1,035 / 517. BF16/FP16 dense ≈ 258.5 is derived (517 / 2); NVIDIA prints no dense BF16 | 2025 (blog); ledgers 10-03 Q2 and 10-25 C2 | 10-25 (not re-fetched) | [DT2](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) | R (2,070, 1,035, 517); I (258.5) |
| F1.3 | SM count 20; L2 32 MiB; 228 KiB max shared memory per SM. All come from a user's CUDA Driver API query on the dev kit. NVIDIA staff confirmed 2,560 CUDA cores (Jan 5 and 7, 2026), said future Jetson docs will not give tensor-core counts (Jan 9), and declined to say more (Jan 19). Users derived 80 tensor cores (4 per SM); the old "96" is unresolved | 2026-01-03 → 01-19 | 10-28 | [DT9](https://forums.developer.nvidia.com/t/clarification-on-cuda-core-and-tensor-core-counts-for-jetson-agx-thor/356355) | R (user runtime); U as an NVIDIA spec |
| F1.4 | Cross-check of 20 SMs: 2,560 cores ÷ 128 FP32 lanes per SM = 20. The 10 TPCs × 2 SMs per TPC (the TRM excerpt quoted in DT9) also give 20. NVIDIA staff's CUDA-core FP32 figures, 8.064 TFLOPS (MAXN) and 7.096 TFLOPS (120 W), equal 2,560 × 2 × 1.575 GHz and × 1.386 GHz exactly | 2025-11-03 (staff) | 10-28 | [DT11](https://forums.developer.nvidia.com/t/performance-benchmarking-on-jetson-thor/349494) | I (consistent) |
| F1.5 | Compute capability 11.0 = **sm_110**. "Targets sm_{101,101f,101a} are renamed to targets sm_{110,110f,110a} from PTX ISA version 9.0." CUTLASS 4.2.0 (2025-09-15) says the same, keyed to CUDA 13.0: below 13.0 Thor is SM101; from 13.0 it is SM110 and SM101 is no longer valid. FlashAttention's `setup.py` maps 110 to 101 below CUDA 13.0 | PTX ISA 9.4 (page undated); CUTLASS 2025-09-15 | 10-28 | [DC2](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html); [DL3](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md); [DL1](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py) | R |
| F1.6 | The CUDA 13.4 Update 2 release notes contain no rename statement. They record cuBLAS 13.0 adding sm_110 support for arm64-sbsa Linux, and mention Thor GPUs in the cuBLAS 13.0U1 (TF32 GEMMs) and 13.1 (FP16/FP32 GEMMs) notes | page current at 13.4U2 | 10-28 | [DC3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) | R |
| F1.7 | Per-SM limits for CC 11.0, from the CUDA Programming Guide v13.4.2, Tables 30–33 (last updated 2026-09-10): <br>• 1,536 resident threads (48 warps, 24 blocks); <br>• 64K registers per SM and 255 per thread; <br>• 228 KB shared memory per SM and 227 KB per block; <br>• 256 KB unified L1/shared; <br>• FP32:FP64 = 64:1; DPX "Multiple Instr." (not native); clusters, DSMEM and TMA "Yes"; <br>• tensor input types TF32, BF16, FP16, FP8, FP6, FP4, INT8, but **no FP64** tensor. <br>For comparison: CC 10.0 and 10.3 have 2,048 threads/SM and 228 KB; CC 12.x has 1,536 threads, **100 KB** shared memory per SM (99 per block) and 128 KB unified; CC 9.0 has 2,048 threads and 228 KB | 2026-09-10 | 10-28 | [DC1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) | R |
| F1.8 | Family targets (Table 28): `compute_110f` is compatible with 11.0 only, and the family currently has one member. `compute_100f` covers 10.0, 10.3 and 10.7, **not 11.0**. `compute_120f` covers 12.0 and 12.1. So sm_100a/100f cubins built for B200 do not load on Thor; rebuilding for 110a/110f is required | 2026-09-10 | 10-28 | DC1 | R; I (the cubin consequence) |
| F1.9 | sm_110 Blackwell features, from the PTX ISA 9.4 Target ISA notes: <br>• `tcgen05.alloc`, `tcgen05.ld` and `tcgen05.mma` are supported on sm_100a and sm_110a (formerly sm_101a), and on the sm_100f/sm_110f families; <br>• `.kind::i8` is on sm_100a and sm_110a only (sm_103 is not listed); <br>• the block-scale `.block16`/`.block32` qualifiers are allowed on the sm_100f and sm_110f families; <br>• the `.scale_vec::NX` form requires sm_100a; <br>• mxf4 K = 96 is sm_103a/sm_107a only; <br>• for mxf8f6f4/mxf4/mxf4nvf4 the 1-CTA dense M = 128 (f16 allows M = 64). <br>Other ISA: `cp.async.bulk.tensor` (TMA) needs sm_90+, with `.multicast::cluster` advised for sm_110a/f and `.tile::scatter4` / `.im2col::w::128` on sm_110a/f; `mapa` and `barrier.cluster` need sm_90+; `wgmma.mma_async` **requires sm_90a**; warp-level `mma` FP8 needs sm_89+, while warp-level FP6/FP4 and `.kind`/`.block_scale`/`.scale_vec_size` require **sm_120a/sm_120f** | PTX ISA 9.4 | 10-28 | DC2 (parsed locally: `scripts/ptx_targets.txt`) | R |
| F1.10 | Thor therefore has: tcgen05 plus TMEM, TMA (with multicast), clusters and DSMEM, block-scaled tcgen05 MMA (MXFP8/MXFP4/NVFP4 kinds), INT8 tcgen05, and warp-level FP8 `mma`. It lacks: `wgmma`, warp-level block-scaled FP4 `mma` (an sm_120 feature), FP64 tensor, and native DPX | — | 10-28 | DC1, DC2 | I (from R) |
| F1.11 | NVIDIA staff on the forum said Thor "can natively support FP8 and FP4" (Aug 27, 2025), that NVFP4 was still being confirmed (Aug 29), that NVFP4 model support in vLLM and SGLang would follow within months with no date given (Sep 2), and later that Thor runs NVFP4 and has MXFP4 instruction support (Dec 2025) | 2025-08-27 → 12-18 | 10-28 | [DT12](https://forums.developer.nvidia.com/t/how-the-2070-tflops-of-jetson-agx-thor-t5000-is-calculated/342806); DT10 | R |
| F1.12 | **Power modes** (Jetson Linux r39.2.1, Thor power page), T5000 by mode: <br>• **MAXN (ID 0):** no budget; 14 CPU cores at 2,601 MHz; 10 TPC; 4 FBP; GPU 1,575 MHz; <br>• **120 W (ID 1, default):** 14 CPU cores at 2,601; 10 TPC; 4 FBP; GPU 1,386; <br>• **90 W (ID 2):** 12 CPU cores at 2,601; **6 TPC, 3 FBP**; GPU 1,530; <br>• **70 W (ID 3):** 12 cores at 1,998; 6 TPC, 3 FBP; GPU 1,530. <br>EMC is 4,266 MHz in every mode. T4000: MAXN or 70 W (default), 6 TPC, 1,530 MHz. MAXN is described as an experimental, unconstrained mode for building custom profiles; hardware throttles when module power exceeds the TDP budget, and sustained heavy loads in MAXN are advised against. Dev-kit limits: 130 W TDP; average VDD_GPU 100 W and instantaneous 120 W (50% throttle); VIN average 168 W (75% throttle); the bundled adapter is 140 W | page undated (r39.2.1 = JetPack 7.2.1) | 10-28 | [DT3](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html) | R |
| F1.13 | Peak per mode, assuming tensor peak scales with SMs × clock. MAXN: FP4/FP8/BF16 1,035 / 517 / 258.5. **120 W: 911 / 455 / 227.5** (×0.880). 90 W or 70 W: 603 / 301 / 150.7 (×0.583). If GPU-reachable bandwidth scaled with FBPs (3 of 4), it would be ~205 GB/s — unverified, since no NVIDIA source states the bandwidth per FBP | — | 10-28 | DT2, DT3; d_arith | I |
| F1.14 | **Unified memory** (CUDA for Tegra note v13.4): <br>• device, host and unified memory all sit on the same physical SoC DRAM; <br>• Thor adds two-way Sysmem Full Coherency, where earlier Tegra had one-way I/O coherency; <br>• pageable memory is directly GPU-accessible and cached in the iGPU L2 when `pageableMemoryAccess` = 1 (Thor onward, on L4T); <br>• `cudaMallocManaged` currently maps GPU-uncached and I/O-coherent, i.e. like pinned memory; <br>• registered or pageable memory "can outperform" pinned or unified memory. <br>So the 273 GB/s is shared by the CPU, GPU, ISP, codecs and NVMe DMA; a camera pipeline's traffic subtracts directly from the model's bandwidth | page undated (CUDA 13.4) | 10-28 | [DT7](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html) | R; I (the contention reading, P-009) |
| F1.15 | The newest JetPack for Thor is still 7.2.1 (L4T 39.2.1), then 7.2 (L4T 39.2) and 7.1 (L4T 38.4). JetPack 7.2.1 = CUDA 13.2.2 and TRT 10.16.2 (10-25 ledger) | page undated | 10-28 | [DT6](https://developer.nvidia.com/embedded/jetpack-archive) | R |

### 2.2 Q2 — Thor measured vs peak

| ID | Claim (baseline · shapes · precision · device/versions/mode · engine · trace) | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F2.1 | **CUTLASS dense FP4 GEMM** reached ~878 TFLOP/s with `cutlass_profiler` exhaustive search: 4096 × 8192 × 8192, FP4 inputs, 256 × 256 × 256 tile, 4 × 1 × 1 cluster, 5 stages, 0.626 ms. A second user got ~875 TFLOP/s. Baseline: NVIDIA's 1,035 dense ⇒ **84.5–84.8%**. Built with `110a`, CUDA 13.0, "130 W" (dev kit; mode name not given), on a real device; profiler output, no trace. Kernels were the `cutlass3x_sm100_bstensorop_*` family | 2025-11-25, 12-08, 12-12 | 10-28 | [DT10](https://forums.developer.nvidia.com/t/verifying-claimed-tops-performance-on-jetson-thor-cutlass-kernel-for-sm110-does-not-run-sm80-gives-very-low-performance-6-9-tflop-s/352063) | R (user); I (%) |
| F2.2 | In the same thread, an Orin-era sparse INT8 kernel built for sm_110 did not run; forcing sm_80 ran at ~6.9 TFLOP/s on 16384³. A `tcgen05` kernel compiled with nvcc 13.0.8 for sm_110 was rejected, and staff advised CUDA 13.1. No FP8 or sparse result was posted | 2025-11-20 → 12-24 | 10-28 | DT10 | R |
| F2.3 | **cuBLAS FP16 GEMM ~130 TFLOPS** against the ~258.5 dense FP16 ⇒ ~50%. **Custom copy kernel ~200 GB/s** against 273 ⇒ ~73%. Sizes, JetPack, CUDA, mode and clocks were not stated, and there is no trace | 2025-10-29 / 11-04 | 10-28 | DT11 | R (user); I (%) |
| F2.4 | **TensorRT-Edge-LLM benchmark page** (published under the 0.11.0 docs, labelled as the v0.10.0 report): Jetson AGX Thor dev kit, JetPack 7.2, CUDA 13.2, TRT 10.16, power mode **not stated**. Decode is measured at 2,048 past-KV with 3 warm-up and 10 measured iterations; the runtime uses 10 warm-ups. Batch-1 decode in tok/s: <br>• Qwen3-8B: NVFP4 47.3, INT4-AWQ 49.0; <br>• Qwen3-4B: 79.8 / 83.3; <br>• Qwen3-1.7B: 153.6 / 154.1; <br>• Qwen3-0.6B: 306.4 / 319.7; <br>• Qwen3-VL-8B: NVFP4 47.1. <br>Speculative decoding on Qwen3-8B NVFP4: EAGLE3 153.7 and DFlash 90.0. Prefill (batch 1, prompt-dependent): Qwen3-8B NVFP4 27.7 ms, INT4 52.2 ms. These are real engines. **No Alpamayo, Cosmos-Reason2 or pi0.5 rows** | page live 10-28 | 10-28 | [DT15](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html) | R |
| F2.5 | **Decode against the bandwidth bound.** Qwen3-8B has 36 layers, hidden 4,096, intermediate 12,288, 8 KV heads × 128, vocab 151,936 and an untied lm_head, giving 6.946 B non-embedding linear parameters plus 622 M in the lm_head. NVFP4 is taken as 4.5 bits per parameter: e2m1 values plus one 8-bit scale per 16 elements, the `.kind::mxf4nvf4` format with `.ue4m3` scales and `.block16` in the PTX ISA (DC2). This is a format assumption: the engine's exact per-tensor layout was not read. Per token that is 4.41 GB (lm_head NVFP4, FP8 KV) to 5.45 GB (lm_head FP16, FP16 KV) ⇒ bound **50.1–61.9 tok/s**. Measured 47.3 = **76–94% of bound**, below it, so not a bug signal | — | 10-28 | [DQ1](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json); DT15; d_arith | I |
| F2.6 | **NVIDIA's Jetson benchmarks page** (vLLM; JetPack 7.0, CUDA 13.0, TRT 10.13; ISL/OSL 2048/128; precision and power mode **not stated**), concurrency 1 / 8 in tok/s: Llama 3.1 8B 41.3 / 150.8; Qwen3 30B-A3B 61 / 226.4; Qwen2.5-VL 7B 45 / 252; Llama 3.3 70B 4.7 / 12.6. These cannot be turned into bound fractions without the precision | page undated | 10-28 | [DT16](https://developer.nvidia.com/embedded/jetson-benchmarks) | R |
| F2.7 | **Throttling and DVFS:** <br>• (a) Over-current throttling in all modes, including 120 W and MAXN, was reported with Qwen3-VL-8B serving 5 concurrent requests × 4 images at 1280 × 720 (dev kit, JetPack 7.1, vLLM container 25.12.post1, USB-C power). Staff called OC a module-protection design and suggested a custom power mode built from `tegrastats`; there was no resolution. <br>• (b) `deviceQuery` showed 1,049 MHz in MAXN or 120 W even with `jetson_clocks`; staff said DVFS is on by default and `jetson_clocks` locks the maximum (GR3D_FREQ 1,574 MHz). <br>• (c) A 70 W `nvpmodel.conf` from the Power Estimator behaved like MAXN at ~110 W in `jtop` (snippet only, thread not opened). <br>No 30-minute sustained-clock trace was found | 2026-01-14 → 01-19; 2025-09-25 → 10-08 | 10-28 | [DT14](https://forums.developer.nvidia.com/t/system-throttled-due-to-over-current-error-when-deploying-qwen3-vl-8b-with-vllm-on-jetson-thor-for-multi-image-inference/357532); [DT13](https://forums.developer.nvidia.com/t/jetson-thor-gpu-stuck-at-1-05ghz/346044); [DT17](https://forums.developer.nvidia.com/t/nvpmodel-conf-for-70w-with-jetson-thor/348826) | R (a, b); U (c, snippet) |
| F2.8 | Alpamayo on Jetson AGX Thor (10-25 ledger B4, FlashRT PR #229, MAXN, official weights, 10 flow steps, full request): official BF16 3,322 ms; FlashRT bf16 2,079 ms; "balanced" 897 ms; "fast" 811 ms. Unchanged; no newer Alpamayo-on-Thor measurement was found | 2026-10-20 | 10-25 (ledger) | [PR #229](https://github.com/flashrt-project/FlashRT/pull/229) | R (ledger) |

### 2.3 Q3 — The proxies

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F3.1 | **RTX PRO 6000 Blackwell Workstation datasheet** (5349469, Jun 2026): 24,064 CUDA cores; 5th-gen Tensor Cores; 4,000 AI TOPS (footnote: FP4 with sparsity, peak at boost clock); FP32 126 TFLOPS; 96 GB GDDR7 ECC on 512-bit; **1,792 GB/s**; 600 W total board power; MIG up to 4 × 24 GB. **No SM count and no L2 printed** | Jun 2026 | 10-28 | [DP1](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf) | R |
| F3.2 | Derived for the WS: SMs = 24,064 / 128 = **188**; boost clock = 126 TFLOPS / (2 × 24,064) = **2.618 GHz**; dense FP4 / FP8 / BF16 = 2,000 / 1,000 / 500 (4:2:1 from 4,000 sparse FP4). Secondary spec databases list 188 SMs and **128 MB L2** (TechPowerUp, via search only) | — | 10-28 | DP1; [DP7 locator](https://www.techpowerup.com/gpu-specs/rtx-pro-6000.c4272) | I (SMs, clock, dense); U (L2: secondary only) |
| F3.3 | **Max-Q** (NVIDIA product page): FP32 110 TFLOPS; 3,511 AI TOPS (FP4 sparse); 1,792 GB/s; **300 W**; dual-slot active. Derived: 2.286 GHz boost; dense FP4 / FP8 / BF16 1,755.5 / 877.8 / 438.9 | page undated | 10-28 | [DP2](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/) | R; I |
| F3.4 | **Server edition:** 1,597 GB/s; FP4 4 PF, FP8 2 PF and BF16 1 PF, read as sparse-equivalent (10-03 ledger) | 10-03 | 10-03 (ledger) | [DP3](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | R (ledger) |
| F3.5 | **H100 SXM:** FP8 dense 1,979, BF16 989.5, 3.35 TB/s (10-03 ledger). Hopper tuning guide: **50 MB L2**, 228 KB shared per SM, 256 KB unified, clusters up to 8 portable and 16 opt-in (updated 2026-10-12). SM count 132 was **not** found on the fetched NVIDIA pages | 2026-10-12 | 10-28 | [DC5](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html); [DP5](https://www.nvidia.com/en-us/data-center/h100/) | R; U (132 SMs) |
| F3.6 | **HGX B200 (8 GPUs):** FP4 144 PF sparse / 72 PF dense; FP8/FP6 72 PF and BF16 36 PF sparse, with dense = ½; INT8 72 POPS; FP64 296 TF; 1.4 TB. **HGX B300:** FP4 144 / **108** PF; FP8 72; BF16 36; **INT8 3 POPS**; FP64 10 TF; 2.1 TB. Per GPU: B200 dense FP4 / FP8 / BF16 = 9 / 4.5 / 2.25 PF; B300 = **13.5** / 4.5 / 2.25 PF. Per-GPU bandwidth is not on the HGX page (B200 8 TB/s from DGX B200, 10-03 ledger) | page live 10-28 | 10-28 | [DP4](https://www.nvidia.com/en-us/data-center/hgx/) | R; I (÷ 8) |
| F3.7 | Blackwell tuning guide v13.4 (updated 2026-10-12): **L2 126 MB (stated for GB200)**; 228 KB shared per SM (CC 10.0); 256 KB unified; non-portable clusters up to 16. No SM counts. B200 = 148 SMs comes only from secondary sources (Chips and Cheese) | 2026-10-12 | 10-28 | [DC4](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html); [DP7 locator](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut) | R (L2); U (148 SMs) |
| F3.8 | **Conflict:** NVIDIA's Blackwell Ultra blog snippet gives 160 SMs "in the full GPU implementation" and 15 PF dense NVFP4, while the HGX B300 page implies 13.5 PF dense per GPU. Probably full die vs shipping SKU, but this is unverified (full blog not opened) | 2025 (blog) | 10-28 | [DP6](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era) | U |
| F3.9 | **MMA families** (PTX ISA): H100 = `wgmma` (sm_90a) + `mma.sync`; B200 = `tcgen05` (sm_100a/f) + `mma.sync`; B300 = `tcgen05` via the sm_100f/103f family, without `.kind::i8`; RTX PRO 6000 = `mma.sync` including warp-level block-scaled FP4/FP6 (sm_120a/f), **no `tcgen05`, no `wgmma`**; Thor = `tcgen05` (sm_110a/f) + `mma.sync` FP8, no `wgmma` | PTX ISA 9.4 | 10-28 | DC2 | R |
| F3.10 | **Triton** `getMMAVersionSafe`: capability 90–99 prefers MMA v3 (`wgmma`); 100–119 prefers **v5 (`tcgen05`)**, with the comment excluding "consumer Blackwell (sm120)" and INT8 gated per target; 120–129 gets **v2 (`mma.sync`) only**. Its arch string is `sm_{cap}a` for cap ≥ 90, so Thor compiles as `sm_110a` | main, fetched 10-28 | 10-28 | [DL4](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp); [compiler.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py) | R |
| F3.11 | **Answer to "does a kernel tuned on the RTX PRO 6000 or H100 use Thor's instructions?"** No. An sm_120 kernel issues `mma.sync` and cannot issue `tcgen05`; an H100 kernel issues `wgmma`, which Thor does not have (FA3, Hopper-only, is a `wgmma` kernel). B200 kernels share Thor's ISA family but must be recompiled (sm_100a/100f do not cover 11.0). **The architecturally closest proxy is B200 (sm_100)**: same tcgen05/TMEM, 228 KB shared and 256 KB unified per SM. Thor differs from it in 1,536 vs 2,048 threads/SM, no FP64 tensor, 20 vs ~148 SMs, 273 GB/s vs 8 TB/s and 32 MiB vs 126 MB L2 | — | 10-28 | DC1, DC2, DL4, DL5 | I (from R) |

### 2.4 Q4 — Why a proxy fix may not transfer, and whether a proxy can emulate Thor

| ID | Mechanism (numbers) | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F4.1 | **SM count and starvation.** A skinny GEMM at M = 64 with 128-wide N tiles: N = 2,048 gives 16 CTAs = 8.5% of 188 SMs but 80% of Thor's 20; N = 12,288 gives 96 CTAs = 51% of 188, i.e. 4.8 waves on Thor. Split-K or stream-K tuned to fill 188 SMs over-splits on Thor (extra reduction passes), so a "starvation fix" found on the proxy can be a regression on Thor | — | 10-28 | d_arith; shapes from `04` §2 | I |
| F4.2 | **Ridge points** (dense FLOP/B), BF16 / FP8 / FP4: <br>• Thor MAXN 947 / 1,894 / 3,791 (120 W: 833 / 1,667 / 3,337); <br>• RTX PRO 6000 WS 279 / 558 / 1,116; Server 313 / 626 / 1,252; Max-Q 245 / 490 / 980; <br>• H100 295 / 591 / n/a; B200 281 / 562 / 1,125; B300 281 / 562 / 1,688 (B300 bandwidth secondary). <br>An operator at 300–900 FLOP/B is compute-bound on every proxy but memory-bound on Thor | — | 10-28 | d_arith (inputs F1.2, F3.1–F3.6) | I |
| F4.3 | **L2 against the working set.** KV for ~3k tokens in BF16 = 0.442 GB: 13.2× Thor's 32 MiB, 8.8× H100's 50 MB, 3.5× B200's 126 MB. Expert weights 5.0 GB per step are 39–149× every L2. Weights between ~34 MB and ~126 MB (for example a compact or FP4 student expert) can stay L2-resident across K flow steps on B200 or RTX PRO 6000 but are re-read from DRAM K times on Thor, so the proxy under-states Thor's per-step time by up to ~K× in that band | — | 10-28 | d_arith; L2 from DT9, DC4, DC5; DP7 (unverified) | I |
| F4.4 | **Unified-memory contention.** The CPU, iGPU, ISP and codecs share one 273 GB/s DRAM (DT7). A discrete proxy's GDDR7/HBM is private to the GPU, and camera DMA or CPU traffic land on PCIe/host DRAM instead. No proxy setting reproduces this. P-009's synthetic contention arm stays the only stand-in | — | 10-28 | DT7 | I |
| F4.5 | **Launch overhead on the Arm CPU.** Unmeasured on Neoverse-V3AE. Illustration only, with assumed per-launch cost: 2,880 launches per tick (L7) × 2 / 4 / 6 µs = 5.8 / 11.5 / 17.3 ms of a 100 ms tick. NVIDIA's expert CUDA graph (alpamayo1.5 PR #36) was measured on x86 hosts only | — | 10-28 | d_arith; L7 §2 Q4 | I (assumption flagged) |
| F4.6 | **Clocks and power modes.** The default 120 W mode runs the GPU at 88% of MAXN; 90 W and 70 W remove 4 of 10 TPCs; DVFS is on unless `jetson_clocks` is run (F1.12, F2.7). On rented proxies, 15 of 35 WS offers report 350–550 W caps (F6.2). A proxy at an unknown power cap against a Thor at an unrecorded mode is two uncontrolled variables | — | 10-28 | DT3, DT13, DR1 | R; I |
| F4.7 | **Green contexts:** for CC ≥ 9.0 (9.0, 10.x, 11.0, 12.x) the minimum SM partition is 8 and SM co-scheduling alignment is 8 (Table 30). Counts are multiples of the alignment unless backfill is set (driver docs, search summary). FlashInfer's wrapper rounds 10 up to 16. So a 20-SM partition is not expressible on a proxy; 16 or 24 are. Green contexts partition SMs, not L2 or DRAM bandwidth (10-27 L7 rejected-options row) | 2026-09-10 | 10-28 | DC1; [DC8](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/structCUdevSmResource.html); [FlashInfer green_ctx](https://docs.flashinfer.ai/generated/flashinfer.green_ctx.split_device_green_ctx_by_sm_count.html) | R (table); I (20 not expressible); U (driver-doc wording via search summary) |
| F4.8 | **MIG** does partition memory. A memory slice is about 1/8 of the GPU's memory "including the corresponding memory controllers and cache", and each instance has dedicated memory that limits capacity and bandwidth. Smallest profiles: <br>• B200 1g.23gb: 1/8 memory, 1/7 SMs, 1/8 L2, up to 7 instances; <br>• H100 1g.10gb: 1/8 / 1/7 / 1/8; <br>• RTX PRO 6000 1g.24gb: 1/4 / 1/4 / 1/4, up to 4 instances | page undated | 10-28 | [DC6](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html); [DC7](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/concepts.html) | R |
| F4.9 | **Emulation table** (FP8 dense TFLOPS / GB/s / L2 MB / ridge / MMA): <br>• **Thor:** 517 / 273 / 33.6 / 1,894 / tcgen05; <br>• RTX PRO 6000 WS green context 24 SMs: 128 / 1,792 / 128 / 71 / mma.sync; <br>• RTX PRO 6000 Server MIG 1g.24gb: 250 / 399 / 32 / 626 / mma.sync; <br>• B200 green context 24 SMs: 730 / 8,000 / 126 / 91 / tcgen05 (148 SMs unverified); <br>• **B200 MIG 1g.23gb: 643 / 1,000 / 15.8 / 643 / tcgen05**; <br>• H100 MIG 1g.10gb: 283 / 419 / 6.2 / 675 / wgmma. <br>None matches Thor on ISA, compute, bandwidth and L2 at once. B200 1g.23gb matches ISA and compute (1.24×) and is 3.7× on bandwidth and 0.47× on L2; the rentable RTX PRO 6000 MIG matches bandwidth (1.46×) and L2 but not ISA | — | 10-28 | d_arith from F4.8, F3.x, F1.x | I |
| F4.10 | **What cannot be emulated:** <br>• unified-memory contention (F4.4); <br>• Thor's 1,536-thread/SM occupancy ceiling on a B200 (2,048); <br>• the Neoverse-V3AE host and its launch path; <br>• the Thor power-management and over-current behaviour; <br>• JetPack's TensorRT 10.16 tactics (TRT 11.x on x86); <br>• LPDDR5X latency (Little's-law bytes in flight at an assumed 500–800 ns: Thor 6.8–10.9 KB per SM; RTX PRO 6000 4.8–7.6 KB per SM). <br>MIG needs host administration: not possible in a vast.ai container (inference; VM offers are rare, F6.3); RunPod sells RTX PRO 6000 MIG slices only | — | 10-28 | DT7, DC1, DR1, DR3; d_arith | I |
| F4.11 | **MPS:** the NVIDIA MPS docs advertise "memory and SM partitioning capabilities" (MPS v3 interface, MPS v3 memory-partitioning page). How `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` rounds SMs, and whether bandwidth is partitioned, were not read (overview page only) | page undated | 10-28 | [DC9](https://docs.nvidia.com/deploy/mps/index.html) | U |

### 2.5 Q5 — sm_110 support in open kernels and engines

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F5.1 | **FlashAttention.** <br>• `setup.py` defaults `FLASH_ATTN_CUDA_ARCHS` to 80, 90, 100, 110 and 120, and builds `compute_110f` → `sm_110` from CUDA 13.0, so FA2 builds for Thor. <br>• **FA3** is the Hopper build (`hopper/` directory). <br>• **FA4** (CuTe DSL) is described as optimized for Hopper and Blackwell (H100, B200). Its `flash_fwd_sm100.py` asserts the arch is in the sm_100f or sm_110f family. <br>• `flash_attn/cute` also contains separate `flash_fwd_sm120.py` and `flash_bwd_sm120.py` (directory listing, 10-28). So Thor runs FA4's SM100 (B200) kernel, and the RTX PRO 6000 runs a different SM120 kernel. <br>• No Thor benchmark of FA4 was found | main (directory listing and raw files 10-28) | 10-28 | [DL1 setup.py](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py); [flash_attn/cute](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute); [README](https://github.com/Dao-AILab/flash-attention) | R; I (FA3 is wgmma-based: inferred from Hopper-only) |
| F5.2 | **FlashInfer.** <br>• The README table lists Blackwell "SM 11.0, Jetson Thor" alongside SM 10.0/10.3 (B200/B300) and SM 12.0/12.1 (RTX 50, DGX Spark). <br>• Experimental `flashinfer/experimental/sm110_xqa` and `sm110_gqa_decode` packages exist: built for `sm_110a` with CUDA ≥ 13.0, TMEM kernel variants, a benchmark script. <br>• Collaborator PRs #6944, #6945, #6981, #6982 (10-21 → 10-22), #7116 (10-23) and #7293 (10-25) added them. #6295, which adds the remaining SM100-family modules to the SM110 ahead-of-time provider, closed 10-28 | 2026-10-21 → 10-28 | 10-28 | [DL2 README](https://github.com/flashinfer-ai/flashinfer); [PR #6295](https://github.com/flashinfer-ai/flashinfer/pull/6295); [PR #6945](https://github.com/flashinfer-ai/flashinfer/pull/6945) | R |
| F5.3 | **CUTLASS.** <br>• The README support table lists "NVIDIA DRIVE Thor, 11.0, CUDA 13.0" (not Jetson by name), and B300 10.3 / CUDA 13.0. <br>• Changelog: 4.2.0 (2025-09-15) has the SM101→SM110 rename; 4.6.1 (2026-07-13) fixed a Thor build failure with the CUDA 12.9 wheel; the latest is **4.8.1 (2026-10-15)**. <br>• On Thor, users and staff build with `-DCUTLASS_NVCC_ARCHS=110a` and run SM100 kernels (F2.1). <br>• No dedicated `sm110a` example directory was confirmed (code search returned nothing) | 2026-10-15 | 10-28 | [DL3](https://github.com/NVIDIA/cutlass/blob/main/README.md) | R; U (no sm110-specific example) |
| F5.4 | **Triton:** tcgen05 (MMA v5) is chosen for capability 100–119, which includes 110 (F3.10). There is no `Thor` or `sm_110` string in the NVIDIA backend (code search, 10-28), so support is generic by capability. Untested on Thor here | main, 10-28 | 10-28 | DL4 | R; U (on-device behaviour) |
| F5.5 | **vLLM (main CMake):** for CUDA ≥ 13.0, `CUDA_SUPPORTED_ARCHS` includes 11.0. The SM100-family CUTLASS kernel lists are `10.0f;10.7f;11.0f` (FP4 SM100, scaled-mm, MLA), while sm_120 uses separate `12.0a;12.1a` and SM12x FP4 paths. NVIDIA's own Thor vLLM numbers are in F2.6, and the 10-27 ledger notes an MoE-performance gap thread | main, fetched 10-28 | 10-28 | [DL5](https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt) | R |
| F5.6 | **PyTorch:** `.ci/wheel/linux/build_env_setup.py` documents TORCH_CUDA_ARCH_LIST `8.0;9.0;10.0;11.0;12.0+PTX` for CUDA 13.x aarch64 wheels, so official CUDA-13 aarch64 builds target sm_110. Which released torch version started shipping this was not determined, and the plan pins torch 2.8.0 | main @ 9768e2e (code-search index) | 10-28 | [DL6](https://github.com/pytorch/pytorch/blob/main/.ci/wheel/linux/build_env_setup.py) | R; U (release mapping) |
| F5.7 | **SGLang:** the README's NVIDIA row lists B200/B300/GB200/GB300, H100/H200, A100, RTX 30/40/50, RTX PRO 6000, DGX Spark and **Jetson Orin, not Thor** | main, 10-28 | 10-28 | [DL7](https://github.com/sgl-project/sglang) | R |
| F5.8 | **TensorRT-Edge-LLM:** the latest release is still **v0.11.0 (2026-09-29)**; v0.11.0 notes list "Thor-optimized NVFP4-A16" kernels (10-27 L7). External PR #260, an Alpamayo fused gate/up plugin for SM110, is still **open**, last updated 10-21 | 2026-09-29; 10-21 | 10-28 | [DL8](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases); [PR #260](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260) | R |
| F5.9 | **TensorRT 10.16 support matrix:** FP8 and FP4 are listed as supported on Jetson AGX Thor (sm_110) (10-25 ledger C3). TRT 11.4 has no JetPack support (10-27 L7) | 10-25 | 10-25 (ledger) | [DL9](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) | R (ledger) |
| F5.10 | **Under-used sm_110 features.** tcgen05 2-CTA (`cta_group::2`) MMAs, TMA multicast across clusters, DSMEM, TMEM-resident accumulators across Euler steps, and INT8 tcgen05 are present in the ISA (F1.9). The only Thor-specific public kernels found are FlashInfer's experimental decode/XQA, PR #260's gate/up plugin and Edge-LLM's NVFP4-A16 kernels; everything else is B200 code rebuilt. No public flow-matching-expert kernel for sm_110 was found | — | 10-28 | F5.1–F5.8 | I |

### 2.6 Q6 — What is rentable today

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F6.1 | **vast.ai on-demand** (public endpoint HTTP 200; filters as on 10-27: `gpu_name`, `rentable=true`, `rented=false`, `type=on-demand`; `dph_total / num_gpus`; cheapest / per-offer median [GPU-weighted median]; offers, GPUs): <br>• RTX PRO 6000 WS: $1.00 / $1.47 [$1.56]; 35, 62; <br>• RTX PRO 6000 S: $1.24 / $1.38 [$1.33]; 41, 127; <br>• Max-Q: $0.83 / $1.60 [$1.65]; 30, 51; <br>• H100 SXM: $1.93 / **$3.14** [$3.67]; 16, 38; <br>• H100 NVL: $2.61 / $2.72; <br>• H100 PCIe: $2.97 / $2.97; <br>• H200: $4.34 / $6.83; 7, 29; <br>• H200 NVL: $3.32 / $4.01; <br>• B200: $7.50 / $8.75; 8, 33; <br>• B300: $8.50 / $9.94 [$11.25]; 7, 22; <br>• GH200: no offers. <br>vast reports compute_cap 1200 (RTX PRO 6000), 900 (H100/H200), 1000 (B200), 1030 (B300) | 2026-10-28 16:15:50 UTC | 10-28 | [DR1 endpoint](https://console.vast.ai/api/v0/bundles/) (raw JSON in scratch `vast_2026-10-28/`) | R |
| F6.2 | **vast.ai host power caps** (`gpu_max_power`): WS 20 of 35 offers at 600 W, 15 at 350–550 W; Server 27 of 41 at 600 W, 10 at 500 W, 4 at 350–450 W; Max-Q 26 of 30 at 300 W. Whether the field is the current or the default limit is unverified | 10-28 | 10-28 | DR1; `vast_extra.py` | R (field); U (semantics) |
| F6.3 | **vast.ai platform facts:** <br>• VM-enabled offers: WS 1 (from $1.31), Server 3 (from $1.24), Max-Q 2 (from $1.14); none for H100/H200/B200/B300. <br>• `cuda_max_good` ≥ 13.2 (JetPack 7.2.1-era CUDA): WS 25 of 35, Server 29 of 41, Max-Q 24 of 30, H100 SXM 7 of 16, B200 7 of 8. <br>• 8-GPU offers: RTX PRO 6000 Server $10.67/h (7 offers), 8 × H100 SXM $29.33/h (1), 8 × B200 $70.00/h (2) | 10-28 | 10-28 | DR1 | R |
| F6.4 | **RunPod** (page "Updated October 8, 2026"), on-demand secure / community: <br>• RTX PRO 6000 $2.49 / $1.69 (edition not named); <br>• **RTX PRO 6000 MIG 48 GB $1.29 / $1.00; MIG 24 GB $0.69 / $0.50** (Server Edition partitions); <br>• H100 SXM $3.99 / $2.69; H100 PCIe $2.89 / $1.99; H100 NVL $3.19 / $2.59; <br>• H200 $5.29 / $3.59; B200 $7.99 / $5.98; B300 $8.99 / $6.94. <br>No spot prices listed | 2026-10-08 | 10-28 | [DR3](https://www.runpod.io/pricing) | R |
| F6.5 | **Lambda** (undated): B200 SXM6 $6.69 (8×) to $6.99 (1×); H100 SXM $3.99 to $4.29; H100 PCIe $3.29; GH200 $2.29. **No RTX PRO 6000, H200 or B300 listed.** 1-Click Clusters: H100 from $6.16 and B200 from $9.86 per GPU-h at 16 GPUs, 2 weeks to 1 year | undated | 10-28 | [DR4](https://lambda.ai/pricing) | R |
| F6.6 | **Nebius** (undated), on-demand (preemptible from): RTX PRO 6000 $1.80 ($0.79); H100 $4.50 ($0.79); H200 $5.40 ($0.79); B200 $8.50 ($0.99); B300 $9.50 ($0.99) | undated | 10-28 | [DR5](https://nebius.com/prices) | R |
| F6.7 | **Remote Thor.** No hourly rental of Jetson AGX Thor or DRIVE AGX Thor was found: no cloud Jetson lab, no LaunchPad Thor lab (LaunchPad gives up-to-2-week lab access, but no Thor listing surfaced) and no device farm. The only remote-style access found was GTC 2026's in-person workshop, which assigned SSH access to dev kits | 10-28 | 10-28 | [DR7 GTC26 setup](https://www.jetson-ai-lab.com/tutorials/gtc26/setup); [LaunchPad](https://nvidia.com/en-au/data-center/launchpad) | U (search-negative; 2 searches) |
| F6.8 | **Dev-kit price and stock.** <br>• NVIDIA Jetson FAQ: dev kit **$5,499** (MSRP); T5000 $4,999 and T4000 $2,999 (1KU+); "No guarantee of availability" for dev kits. <br>• **SabrePC: $6,268.86, Out of Stock** (unchanged from 10-25). <br>• RS UK and Micro Center returned 403; Arrow timed out; Fusion WW forbids crawling; Seeed URL 404. <br>• Search snippets only (dates unknown): XSNet AU AUD 6,261.75 with 5 in stock; Core Electronics AU quotes a 30–45 day lead time. <br>• No NVIDIA-store listing was found | FAQ undated; 10-28 | 10-28 | [DT5](https://developer.nvidia.com/embedded/faq); [DR6](https://www.sabrepc.com/945-14070-0080-000-NVIDIA-S104023743) | R (FAQ, SabrePC); U (others) |

### 2.7 Q7 — The path (3) rig, re-checked

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F7.1 | Leopard's GMSL2-cameras-on-Jetson-Thor page lists LI-GMSL2-DESER-HOLOSCAN, LI-VENUS-ISX031-GMSL2, **LI-IMX728-9295** (image filename contains `120H`; no 30°/60° variant shown) and LI-AR0234CS-STEREO-GMSL2-30. It shows **no prices**, only a contact link. It says nothing about PTP or cameras per deserializer | page undated (© 2026) | 10-28 | [DG1](https://leopardimaging.com/gmsl2-cameras-on-thor/) | R |
| F7.2 | So the 10-25 ledger's "9295 at $519 (120°) / $539 (30°)" could **not** be re-verified from the Thor page; a 9295 30° tele variant was not found. Prices that do exist are for the IMX728-**GM2B** series: $459 (120H) and $539 (030H, search snippet) | 10-25; 10-28 | 10-28 | DG1; [GM2B 120H](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/adi-gmsl2-cameras/li-imx728-gm2b/li-imx728-gm2b-120h-sf-aa/) | U |
| F7.3 | **LI-GMSL2-DESER-HOLOSCAN: $388.00.** It "Supports 1x Leopard Imaging FAKRA camera"; ADI GMSL2 MAX9296; for Jetson Thor, AGX Orin or IGX Orin with the Lattice HSB. Kit stock: adapter board 20 in stock (backorder allowed); MAX9296 board 262; cable 425; **power supply on backorder**. Thor, HSB and camera are not included. Four cameras therefore need 4 kits = $1,552, assuming one HSB accepts four (unverified) | page live 10-28 | 10-28 | [DG2](https://leopardimaging.com/product/accessories/adapters-carrier-boards/for-holoscan/li-gmsl2-deser-holoscan/) | R; I ($1,552) |
| F7.4 | **Lattice HSB board** (LF-SNSR-ETH-EVN, CertusPro-NX sensor-to-Ethernet bridge): search snippets show Mouser VN $357.75 with 7 in stock (and a passed "expected 21/07/2026" date), Mouser UK £290.85, and Farnell FR €331.86 excluding VAT. Direct Mouser VN fetches timed out twice. The 10-25 figure ($372.50, 0 in stock) was not re-verified | snippets, dates unknown | 10-28 | [Mouser VN (timeout)](https://www.mouser.vn/ProductDetail/Lattice/LF-SNSR-ETH-EVN); [Farnell FR](https://fr.farnell.com/lattice-semiconductor/lf-snsr-eth-evn/kit-pont-capteur-vers-ethernet/dp/4640362) | U |
| F7.5 | **Time sync.** HSB IP implements PTP per IEEE 1588-2019 as a receiver-only (slave) clock, latches host time from SYNC messages and supports frequency adjustment. NVIDIA reports an end-to-end standard deviation under 20 ns in a 10G test (HSB 2.5.0 docs). The Thor path uses camera-over-Ethernet rather than ConnectX (HSB 2.0 overview reprint) | HSB 2.0–2.6 docs | 10-28 (search summary) | [DG4](https://docs.nvidia.com/holoscan/sensor-bridge/latest/ptp.html) | R (search summary; page not opened) |
| F7.6 | **RTK GNSS:** SparkFun ZED-X20P breakout **$299.95, "In stock"** (volume $254.96–284.95). It supports RTK, PPP-RTK and PPP with RTCM and native L-band. **No IMU**, so a separate IMU is needed. No international-shipping statement on the page | page live 10-28 | 10-28 | [DG3](https://www.sparkfun.com/sparkfun-allband-gnss-rtk-breakout-zed-x20p-qwiic.html) | R |
| F7.7 | **Read-only CAN.** The Jetson Linux r39.2.1 Thor-series feature table lists listen-only mode for bus monitoring. The CAN page says no pinmux changes are needed on Thor and the dev kit has built-in transceivers (both via search summary). Listen-only is a controller software mode: the transceiver can still transmit if the mode is changed, so "no actuation path in hardware" needs a receive-only transceiver or a severed TX line | r39.2.1 | 10-28 (search summary) | [DT8](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SO/JetsonThorSeries.html); [CAN page](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/ControllerAreaNetworkCan.html) | R (summary); I (hardware read-only) |
| F7.8 | **Import or shipping into Vietnam:** no primary source states it for any rig part. Leopard links a shipping-information page (not read); Mouser has a Vietnam storefront (mouser.vn), but its fetch timed out | — | 10-28 | DG2; Mouser VN | U |

### 2.8 Q8 — Vietnamese legal and privacy constraints (engineering reading, not legal advice)

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F8.1 | **Law 91/2025/QH15 (PDPL)** has been in force since **2026-01-01**: the Cần Thơ legal-dissemination page quotes the effective date of 01/01/2026 and says it was adopted at the 15th National Assembly's 9th session. Secondary sources give the adoption date as 2025-06-26 | 2026-01-13 (page) | 10-28 | [DV2](https://pbgdpl.cantho.gov.vn/luat-bao-ve-du-lieu-ca-nhan-nam-2025-bao-dam-quyen-rieng-tu-trong-ky-nguyen-so) | R (official provincial page) |
| F8.2 | **Decree 356/2025/NĐ-CP** is the PDPL implementing decree: Government, Hà Nội, 31/12/2025, signed for the Prime Minister by a Deputy PM; e-signed copy stamped by the Government Portal on 05.01.2026. Its legal bases are Law 91/2025 and the **Data Law 60/2024/QH15**. **Art. 42(1):** effective 01/01/2026. **Art. 42(2):** Decree 13/2023/NĐ-CP ceases to have effect from that date | 2025-12-31 | 10-28 | [DV1](https://hcc.nghean.gov.vn/laws/detail/Nghi-dinh-so-356-2025-ND-CP-cua-Chinh-phu-Quy-dinh-chi-tiet-mot-so-dieu-va-bien-phap-thi-hanh-Luat-Bao-ve-du-lieu-ca-nhan-2058/) (signed scan; pages 1, 17–20, 35–37, 39–42 read) | R |
| F8.3 | **Cross-border transfer, Decree 356 Art. 18.** <br>• **(1)** Agencies, organizations and **individuals** transferring personal data across borders under Law Art. 20(1) must prepare a cross-border transfer impact-assessment dossier. <br>• **(2)** Its contents: a Form 09 report; a copy of the contract or transfer document binding sender and recipient; data-protection policies and procedures. <br>• **(3)** The report covers the parties' contacts, purpose, data types, flow diagram, consent, retention and deletion, post-transfer security, the receiving system's diagram, onward transfer, a self-assessment and a risk assessment. <br>• **(4)** It is filed online, in person or by post with the Ministry of Public Security's specialized authority, with Form 01a (organization) or **01b (individual: name, ID number)**, within **60 days** of starting the transfer. <br>• **(5)–(6)** The authority answers in 15 days and may require completion within 30 days, with administrative penalties otherwise. <br>• **(7)** Dossiers are updated per Art. 20 | 2025-12-31 | 10-28 | DV1 (pp. 18–20, 40) | R |
| F8.4 | **What counts as a transfer, Decree 356 Art. 17(1)** (the cross-border transfer article, p. 17): <br>• (a) storing personal data collected in Vietnam on servers outside Vietnam or on a foreign provider's cloud; <br>• (b) transferring it to a recipient abroad; <br>• (c) **processing data collected in Vietnam on a platform outside Vietnam for further processing**, which covers sending recordings to rented GPU hosts abroad. <br>**Art. 17(2):** the authority may order a transfer stopped where it affects national defence or security. <br>**Art. 17(3)**, exemptions beyond Law Art. 20(6)(a–c), which was not read: journalism and media; transfers already made public by law; emergencies to protect life, health or property; cross-border HR management; contracts and logistics, payments, hotels, visas and scholarships. **No research or statistics exemption appears in this list** | 2025-12-31 | 10-28 | DV1 (pp. 17–18) | R; U (Law Art. 20(6) content) |
| F8.5 | **Processing impact assessment, Decree 356 Art. 19:** controllers and processors prepare and keep a processing impact-assessment dossier from the start of processing (Form 10). **Art. 41:** small and start-up enterprises may opt out of PDPL Arts. 21, 22 and 33(2) for 5 years, and household or micro businesses are exempt from them, except when processing sensitive data or ≥ 100k data subjects. **Art. 39:** the specialized authority sits under the Ministry of Public Security | 2025-12-31 | 10-28 | DV1 (pp. 20, 35–36) | R |
| F8.6 | **Interplay with the Data Law.** Decree 356 Art. 42(3) amends Decree 165/2025/NĐ-CP (Data Law, 30/06/2025) Art. 16(2): when core or important data is personal data, its cross-border transfer uses the PDPL impact-assessment dossier instead of the Data Law's risk assessment | 2025-12-31 | 10-28 | DV1 (p. 36) | R |
| F8.7 | **Data localization.** Decree 53/2022/NĐ-CP (Cybersecurity Law 2018) applies its local-storage duty to enterprises providing telecom or internet services that collect user data (10-03/10-25 framing; search summary 10-28). A new **Cybersecurity Law 116/2025/QH15** (passed 10/12/2025) reportedly took effect on **2026-07-01**, and the Ministry of Public Security published a draft implementing decree on 2026-02-13. A "Decree 333/2026" is mentioned in one comparison; its status is unverified. The official texts were not read (vbpl.vn renders by script) | 2025-12-10 / 2026-07-01 (reported) | 10-28 | [DV3 locator](https://luatvietnam.vn/bai-viet-lien-quan/luat-an-ninh-mang-2025-422396.html); [vbpl.moj.gov.vn attachment](https://vbpl.moj.gov.vn/TW/Lists/vbpq/Attachments/187039/116.2025.QH15.docx) (not opened) | U |
| F8.8 | **Smart-vehicle permit.** Law 36/2024 Art. 35(3) requires a permit for smart vehicles (phương tiện giao thông thông minh); Law 118/2025 narrowed the definition from 2026-07-01 to vehicles controlled without a driver (10-25 ledger, official consolidated VBHN 55). Decree 151/2024 Art. 25 issues the permit (Traffic Police; owners, including individuals, may apply). Decree 236/2026/NĐ-CP amends Decree 151 on traffic databases and priority vehicles; **no change to Art. 25 surfaced** (search summary) | 10-25; 10-28 | 10-28 | [VBHN 55](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf) (10-25); DV6 search | R (10-25 ledger); U (Art. 25 unchanged) |
| F8.9 | **Vehicle modification** (press reports only): Circular 30/2026/TT-BXD (Ministry of Construction, effective 2026-07-01) clarifies the term cải tạo xe cơ giới (motor-vehicle modification). Added accessories using manufacturer-supplied genuine parts are not a modification, and cameras, screens, speakers and sensors installed **inside the passenger compartment** are not a modification. A roof rack on a passenger car was classed as a simple modification (cải tạo đơn giản) under Circular 47/2024/TT-BGTVT. Official texts not read | 2026 (press) | 10-28 | [DV4 VOV](https://vov.vn/o-to-xe-may/cac-phu-kien-duoc-lap-them-cho-o-to-khong-bi-phat-va-truot-dang-kiem-post1318906.vov) (search summary) | U |
| F8.10 | **Recording in public places:** PDPL Art. 32 (effective 2026-01-01) allows recording in public places without consent in listed cases but requires notice that recording is happening (10-25 ledger, read from a provincial government page). Not re-checked | 10-25 | 10-25 (ledger) | 10-25 lane D note | R (ledger) |
| F8.11 | **Engineering reading (inference).** <br>• A passive camera rig is not a smart vehicle under the post-2026-07-01 definition on its plain wording (10-25). <br>• Cabin-internal cameras avoid the modification question; external cross-view cameras on a rack likely do not. <br>• Uploading identifiable footage (faces, plates) to foreign GPU hosts is a cross-border transfer by an individual: an Art. 18 dossier within 60 days plus an Art. 19 processing dossier. <br>• Anonymizing on the device before upload, so that data leaving Vietnam is no longer personal data, is the lever that would avoid it; whether blurred data still counts as personal data is a question for counsel. <br>Engineering reading, not legal advice | — | 10-28 | F8.2–F8.10 | I |

### 2.9 Suspicious content
- `fusionww.com` (part 945-14070-0080-000) told automated readers that crawling is not permitted and pointed them to an API and an email address. I did not use the API or contact anyone, and quote nothing further. No page in this pass addressed an AI or asked me to act. (The 10-25 pass found injected text on `hethongphapluat.com`, which this pass did not fetch.)

---

## 3. Device table

Source IDs are in §7. "d" = derived or inferred (`d_arith.out.txt`). Dense TFLOPS. Ridge = dense peak ÷ DRAM bandwidth, in FLOP/B.

| Row | Jetson AGX Thor (T5000) | RTX PRO 6000 WS | RTX PRO 6000 Server | H100 SXM | B200 |
|---|---|---|---|---|---|
| CC | 11.0 = sm_110 (sm_101 before CUDA 13.0) [DC2, DL3] | 12.0 [DR1 vast `compute_cap` 1200; DL2] | 12.0 [DR1] | 9.0 [DL3, DR1] | 10.0 [DL3, DC4] |
| SMs | 20 [DT9 user runtime; d: 2,560/128; 10 TPC DT1] | 188 [d: 24,064/128, DP1; DP7 secondary] | 188 (same die) [d, DP3; U] | 132 [U: not on DC5/DP5] | 148 [U: DP7 secondary] |
| Bandwidth | 273 GB/s LPDDR5X, 256-bit, **shared with CPU/ISP** [DT1, DT7] | 1,792 GB/s GDDR7 512-bit [DP1] | 1,597 GB/s [DP3] | 3.35 TB/s [DP5] | 8 TB/s [10-03 ledger, DGX B200] |
| L2 | 32 MiB [DT9 user runtime; not in DT1] | 128 MB [U: DP7 secondary] | 128 MB [U: DP7] | 50 MB [DC5] | 126 MB [DC4, stated for GB200] |
| Smem/SM (unified) | 228 KB (256 KB) [DC1] | 100 KB (128 KB) [DC1] | 100 KB (128 KB) [DC1] | 228 KB (256 KB) [DC1, DC5] | 228 KB (256 KB) [DC1, DC4] |
| Max threads/SM | 1,536 [DC1] | 1,536 [DC1] | 1,536 [DC1] | 2,048 [DC1] | 2,048 [DC1] |
| Dense FP8 | 517 [DT2]; 455 at 120 W [d, DT3] | 1,000 [d from DP1] | 1,000 [DP3] | 1,979 [DP5] | 4,500 [d ÷ 8, DP4] |
| Dense FP4 | 1,035 [DT2]; 911 at 120 W [d] | 2,000 [d from DP1] | 2,000 [DP3] | none (no FP4 TC in CC 9.0) [DC1] | 9,000 [d ÷ 8, DP4] |
| Dense BF16 | 258.5 [d from DT2]; 227.5 at 120 W [d] | 500 [d] | 500 [DP3, d] | 989.5 [DP5] | 2,250 [d ÷ 8, DP4] |
| Ridge BF16 / FP8 / FP4 | 947 / 1,894 / 3,791 (120 W: 833 / 1,667 / 3,337) [d] | 279 / 558 / 1,116 [d] | 313 / 626 / 1,252 [d] | 295 / 591 / — [d] | 281 / 562 / 1,125 [d] |
| MMA family | tcgen05 + TMEM (sm_110a/f), block-scaled tcgen05 kinds, INT8 tcgen05; `mma.sync` FP8; **no wgmma**; no warp-level FP4 [DC2] | `mma.sync` incl. warp-level block-scaled FP4/FP6 (sm_120a/f); **no tcgen05, no wgmma** [DC2, DL4] | same as WS [DC2] | `wgmma` (sm_90a) + `mma.sync`; no FP4 [DC2] | tcgen05 + TMEM (sm_100a/f) + `mma.sync`; no wgmma [DC2] |
| Price/h (10-28) | **Not rentable** [DR7]; buy $5,499 MSRP [DT5]; SabrePC $6,268.86 OOS [DR6]; d: $2.75/h if amortized over 2,000 h (assumption) | vast $1.00 / $1.47 (GPU-wtd $1.56) [DR1]; RunPod $1.69 / $2.49 (edition unnamed) [DR3]; Nebius $1.80 [DR5] | vast $1.24 / $1.38 [DR1]; RunPod MIG 24 GB $0.50 / $0.69 [DR3] | vast $1.93 / $3.14 [DR1]; RunPod $2.69 / $3.99 [DR3]; Lambda $3.99–4.29 [DR4]; Nebius $4.50 [DR5] | vast $7.50 / $8.75 [DR1]; RunPod $5.98 / $7.99 [DR3]; Lambda $6.69–6.99 [DR4]; Nebius $8.50 [DR5] |

Extra columns, for Q3 completeness:

| Row | RTX PRO 6000 Max-Q | B300 |
|---|---|---|
| CC | 12.0 [DR1] | 10.3 [DL3, DR1] |
| SMs | 188 [d from cores; DP2 prints none] | U (160 "full GPU" per DP6 snippet) |
| Bandwidth | 1,792 GB/s [DP2] | U (~8 TB/s secondary; not on DP4) |
| L2 | U | U |
| Smem/SM | 100 KB [DC1] | 228 KB [DC1] |
| Dense FP8 / FP4 / BF16 | 877.8 / 1,755.5 / 438.9 [d from DP2] | 4,500 / 13,500 / 2,250 [d ÷ 8, DP4]; DP6 says 15,000 FP4 (conflict) |
| Ridge BF16 / FP8 / FP4 | 245 / 490 / 980 [d] | 281 / 562 / 1,688 [d, BW U] |
| MMA family | as RTX PRO 6000 [DC2] | tcgen05 (sm_100f/103f), **no `.kind::i8`**; INT8 3 POPS sparse per 8 GPUs [DC2, DP4] |
| Price/h | vast $0.83 / $1.60 [DR1] | vast $8.50 / $9.94 [DR1]; RunPod $6.94 / $8.99 [DR3]; Nebius $9.50 [DR5] |

Per-SM-per-clock dense tensor throughput [d]: Thor 16,413 FLOP (FP8) and 32,857 (FP4) per SM per clock at 1.575 GHz, against RTX PRO 6000 WS 2,032 and 4,063 at 2.618 GHz, i.e. **8.1×**. B200 needs its SM count and clock, neither printed by NVIDIA, so it is not computed.

H200 prices (Q6): vast $4.34 / $6.83 [DR1]; RunPod $3.59 / $5.29 [DR3]; Nebius $5.40 [DR5]; not on Lambda [DR4].

---

## 4. Deltas vs the 10-27 ledgers

| # | 10-27 ledger said | 10-28 finding | Sources |
|---|---|---|---|
| Δ1 | L6: vast.ai `bundles` returned 403 on 10-27 08:29 UTC; L1 pulled it at 08:30–08:31 | The public endpoint returned **HTTP 200** on 10-28 16:15 UTC with the same filters (no auth). A fresh snapshot is saved in the scratchpad, not in the repo | DR1 |
| Δ2 | vast 10-27 cheapest / median: <br>• WS $1.00 / $1.56 (33); Server $1.27 / $1.34 (30); Max-Q $0.80 / $1.44 (41); <br>• H100 SXM $1.79 / $2.40 (17); H200 $6.43 / $6.43 (2); <br>• B200 $6.49 / $10.25 (12); B300 $8.98 / $11.25 (8) | 10-28: <br>• WS $1.00 / **$1.47** (35); Server **$1.24** / $1.38 (41); Max-Q $0.83 / $1.60 (30); <br>• H100 SXM **$1.93 / $3.14** (16); H200 $4.34 / $6.83 (7); <br>• B200 $7.50 / **$8.75** (8); B300 $8.50 / $9.94 (7). <br>H100 SXM's median rose by 31% in one day; the plan's H100 basis ($2.40) is stale | DR1, DR2 |
| Δ3 | "RTX PRO 6000 Workstation" is treated as the 600 W edition (`04` §3 ridge 279) | **New:** 15 of 35 WS offers report 350–550 W `gpu_max_power`. The edition label does not fix the power limit | DR1 |
| Δ4 | `04` §3 gives Thor ridge ≈ 947 (BF16) only; "188 SMs" has no source in the repo | **New:** ISA-level facts. Thor = tcgen05 family with B200; RTX PRO 6000 = `mma.sync` only; H100 = `wgmma`; the rename is at PTX ISA 9.0 / CUDA 13.0. Per-SM Thor tensor rate is 8.1× an RTX PRO 6000 SM. 188 SMs is now derived from the datasheet (24,064/128). Thor's 20 SMs and 32 MiB L2 are user-runtime only | DC1, DC2, DP1, DT9 |
| Δ5 | Thor power: 40–130 W; P-009(iii) "pinned power mode and clocks" | **New:** official mode table. 120 W is the default (GPU 1,386 MHz); MAXN is "experimental"; 90/70 W drop to 6 TPC and 3 FBP; DVFS is on by default; OC throttling reported with multi-image VLM serving | DT3, DT13, DT14 |
| Δ6 | Thor measured-vs-peak: none in the 10-27 ledger beyond Alpamayo FlashRT | **New:** CUTLASS dense FP4 ~875–878 TFLOP/s (≈85%); cuBLAS FP16 ~130 TFLOPS (~50%); copy ~200 GB/s (~73%); Edge-LLM Qwen3-8B NVFP4 47.3 tok/s = 76–94% of the bandwidth bound | DT10, DT11, DT15 |
| Δ7 | NVIDIA datasheet "96 Tensor Cores" (older brief) | **New:** the Jan 2026 module datasheet drops tensor-core counts; staff said future docs will not give them | DT1, DT9 |
| Δ8 | Library support: Edge-LLM only (10-27 L7) | **New:** <br>• FA2 builds sm_110; FA4's CuTe SM100 forward allows the 11.x family, and separate sm120 forward and backward kernels exist for the RTX PRO 6000; <br>• FlashInfer lists Thor and added experimental sm110 TMEM decode kernels (10-21 → 10-25) plus an SM110 AOT provider (PR #6295, closed 10-28); <br>• vLLM CMake targets 11.0f in its SM100-family kernels; <br>• PyTorch CUDA-13 aarch64 arch list includes 11.0; <br>• CUTLASS 4.8.1 (10-15) lists DRIVE Thor 11.0; <br>• SGLang README lists no Thor | DL1–DL7 |
| Δ9 | Edge-LLM v0.11.0; PR #260 open; JetPack 7.2.1 newest; dev kit $5,499 | **Unchanged** on 10-28 (#260 last updated 10-21) | DL8, DT6, DT5 |
| Δ10 | Rented-GPU emulation of Thor not discussed (L7 rejected green contexts for contention only) | **New:** green contexts cannot form a 20-SM partition (8-SM alignment); MIG partitions L2 and bandwidth; RunPod rents RTX PRO 6000 MIG 24 GB at $0.50/$0.69; no B200 MIG rental found | DC1, DC6, DC7, DR3 |
| Δ11 | Rig (10-25 C6/C7): 9295 at $519/$539; deserializer $388; HSB $372.50 with 0 in stock; ZED-X20P $299.95 | Deserializer $388 is confirmed (1 camera; PSU backordered); ZED-X20P $299.95 in stock (no IMU). **The 9295 prices and a 9295 30° variant could not be re-verified** (Thor page shows prices for neither). HSB board stock is only at search-snippet level (Mouser VN 7 in stock) | DG1–DG3, DG5 |
| Δ12 | Legal (10-25): PDPL Art. 32 public recording; Decree 13/2023 status unknown; permit narrowing 2026-07-01 | **Engineering reading, not legal advice.** **New:** the signed Decree 356/2025 text. Decree 13/2023 ceased on 2026-01-01; Art. 17(1)(c) counts processing on a platform abroad as a transfer; Art. 18 cross-border dossier, covering individuals, within 60 days; Art. 17(3) exemptions without research; Art. 41 SME relief; Art. 42(3) Data-Law interplay. **New:** Cybersecurity Law 116/2025 reportedly in force 2026-07-01 (unverified); press on Circular 30/2026/TT-BXD vehicle modification (unverified) | DV1, DV3, DV4 |

---

## 5. Raw opportunity and kill signals (unscored)

**Path 1 — research paper**
- Opportunity:
  - Proxy-to-Thor transfer is documented nowhere for VLA serving. Each mechanism is concrete and measurable: ISA family, ridge, L2 residency across flow steps, split-K over-splitting, power mode (F3.11, F4.1–F4.10).
  - Edge-LLM's decode at 76–94% of the bound gives a calibrated baseline for the bandwidth side (F2.5).
  - RTX PRO 6000 MIG 1g.24gb at $0.50–0.69/h is a cheap memory-side slice (F6.4, F4.9).
- Kill:
  - Without a device, every Thor number is a model (no hourly Thor, F6.7), so a paper about Thor transfer needs P-002 anyway.
  - B200 MIG, the closest ISA and compute slice, was not found for rent; vast containers cannot enable MIG (F4.10).

**Path 2 — Jetson AGX Thor engineering that does not change behaviour**
- Opportunity:
  - B200-family tcgen05 code rebuilds for 110a: CUTLASS reaches ≈85% of FP4 peak on Thor (F2.1). FA4, FlashInfer and vLLM already admit 11.x (F5.1–F5.5).
  - No public sm_110 flow-matching-expert kernel exists (F5.10).
  - PR #260 is the only Alpamayo SM110 plugin, and it is still unmerged (F5.8).
  - B200 is the right code-development proxy (same ISA family, F3.11), at $5.98–8.75/h.
- Kill:
  - A $5,499 purchase is required (F6.8).
  - Default 120 W runs at 88% clock; MAXN is "experimental" with TDP throttling; OC throttling was seen under multi-image VLM load (F1.12, F2.7). 30-minute sustained 10 Hz is untested by anyone public.
  - Toolchain churn: nvcc 13.0.8 rejected a tcgen05 sm_110 kernel; staff said use 13.1 (F2.2). JetPack 7.2.1 ships CUDA 13.2.2.
  - Kernels tuned on the RTX PRO 6000 do not issue Thor's MMA instructions (F3.9–F3.11).
  - The plan's pinned torch 2.8.0 predates the CUDA-13 aarch64 wheels (F5.6).

**Path 3 — open-loop shadow mode in HCMC (never validation; no actuation path)**

The legal items below are an engineering reading, not legal advice.
- Opportunity:
  - The dev kit has CAN listen-only plus transceivers (F7.7) and HSB PTP with under 20 ns standard deviation in NVIDIA's 10G test (F7.5).
  - On-device anonymization before upload fits Thor's role and is the lever against the cross-border dossier (F8.11).
  - Decree 356 provides an individual's filing form (01b), so the path is procedurally defined (F8.3).
- Kill:
  - Identifiable footage processed on foreign GPU hosts is a transfer under Art. 17(1)(c). It triggers an Art. 18 dossier within 60 days plus an Art. 19 processing dossier, and there is no research exemption in Art. 17(3) (F8.3–F8.5).
  - An external roof rack is likely a modification (press, F8.9).
  - 9295 camera prices and the 30° variant are unverified; the deserializer PSU is backordered; HSB stock is unverified; one deserializer kit is needed per camera (F7.1–F7.4).
  - Import into Vietnam is undocumented (F7.8).
  - Cybersecurity Law 116/2025 obligations are unread (F8.7).

---

## 6. Gaps

1. **No NVIDIA-printed SM count or L2 for Thor** (datasheet omits them; 20 SMs and 32 MiB are from a user's runtime query). Same for B200 SMs (148, secondary), RTX PRO 6000 L2 (128 MB, secondary) and H100 SXM SMs (132, NVIDIA pages fetched did not print it).
2. **Thor DRAM bandwidth in 90/70 W modes** (3 of 4 FBPs) is unstated. ~205 GB/s is an unverified proportional guess.
3. **No measured Thor FP8 GEMM, no official STREAM/bandwidthTest, no 30-minute sustained-clock trace.** cuBLAS FP16 and copy numbers lack versions and modes. The Edge-LLM table omits the power mode.
4. **Kernel-launch overhead on Neoverse-V3AE** is unmeasured. The 5.8–17.3 ms per tick illustration is an assumption.
5. **Rented-host controls untested:** clock locking (`nvidia-smi -lgc`), `ncu` counters, MPS SM-percentage semantics, MIG on vast (container) and any B200 MIG rental.
6. **FA4, FlashInfer sm110 and Triton tcgen05 on a real Thor:** support is in code, but no published Thor benchmark was found. The TensorRT 10.16 support matrix was not re-fetched (10-25).
7. **Remote Thor access:** search-negative only, not proven absent.
8. **Dev-kit stock:** no NVIDIA-store listing found. RS UK and Micro Center (403), Arrow (timeout) and Fusion (crawling forbidden) were unreadable.
9. **Rig:** LI-IMX728-9295 per-FOV prices and a 30° tele variant; how many deserializer kits one Lattice HSB accepts; HSB board live stock; an IMU choice; Vietnam import or shipping terms.
10. **Legal, official texts not read:** <br>• Law 91/2025 Art. 20(6)(a–c) exemptions, a personal/household or research exemption, and the public-recording article (10-25 ledger only); <br>• Law 116/2025 data-storage articles, and whether Decree 53/2022 survives 2026-07-01; <br>• Circular 30/2026/TT-BXD and Circular 47/2024 texts; <br>• whether blurred or anonymized footage leaves the PDPL's scope; <br>• surveying-and-mapping or Data-Law important-data status of georeferenced street imagery.

---

## 7. Source list

All checked 2026-10-28 unless noted. P = primary; S = secondary, used only as a locator.

**Thor (NVIDIA)**

| ID | Source | Date | P/S |
|---|---|---|---|
| DT1 | [Jetson Thor Module datasheet, doc 4767587](https://dam-cdn.nvd.orangelogic.com/AssetLink/x20v2h2y15uh0ne54rpb73xh4x7s0m1s.pdf) | Jan 2026 | P |
| DT2 | [Introducing NVIDIA Jetson Thor (blog)](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) | 2025; via ledgers 10-03 Q2 and 10-25 C2 | P |
| DT3 | [Jetson Linux r39.2.1, Jetson Thor power and performance](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html) | undated | P |
| DT4 | [Jetson Thor product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) | undated | P |
| DT5 | [Jetson FAQ (prices)](https://developer.nvidia.com/embedded/faq) | undated | P |
| DT6 | [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) | undated | P |
| DT7 | [CUDA for Tegra application note v13.4](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html) | undated | P |
| DT8 | [Jetson Linux r39.2.1, Jetson Thor Series](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SO/JetsonThorSeries.html) and [CAN page](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/HR/ControllerAreaNetworkCan.html) | search summary | P |
| DT9 | [Forum 356355: core and tensor-core counts](https://forums.developer.nvidia.com/t/clarification-on-cuda-core-and-tensor-core-counts-for-jetson-agx-thor/356355) | 2026-01 | P (forum) |
| DT10 | [Forum 352063: CUTLASS on sm_110](https://forums.developer.nvidia.com/t/verifying-claimed-tops-performance-on-jetson-thor-cutlass-kernel-for-sm110-does-not-run-sm80-gives-very-low-performance-6-9-tflop-s/352063) | 2025-11 → 12 | P (forum) |
| DT11 | [Forum 349494: benchmarking on Thor](https://forums.developer.nvidia.com/t/performance-benchmarking-on-jetson-thor/349494) | 2025-10 → 11 | P (forum) |
| DT12 | [Forum 342806: the 2070 TFLOPS](https://forums.developer.nvidia.com/t/how-the-2070-tflops-of-jetson-agx-thor-t5000-is-calculated/342806) | 2025-08 → 09 | P (forum) |
| DT13 | [Forum 346044: GPU at 1.05 GHz](https://forums.developer.nvidia.com/t/jetson-thor-gpu-stuck-at-1-05ghz/346044) | 2025-09 → 10 | P (forum) |
| DT14 | [Forum 357532: over-current with Qwen3-VL-8B](https://forums.developer.nvidia.com/t/system-throttled-due-to-over-current-error-when-deploying-qwen3-vl-8b-with-vllm-on-jetson-thor-for-multi-image-inference/357532) | 2026-01 | P (forum) |
| DT15 | [TensorRT-Edge-LLM performance benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html) | live | P |
| DT16 | [Jetson benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks) | undated | P |
| DT17 | [Forum 348826: 70 W nvpmodel](https://forums.developer.nvidia.com/t/nvpmodel-conf-for-70w-with-jetson-thor/348826) | snippet only | P (forum) |

**CUDA and ISA (NVIDIA)**

| ID | Source | Date |
|---|---|---|
| DC1 | [CUDA Programming Guide v13.4.2 §5.1, compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) | 2026-09-10 |
| DC2 | [PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | — |
| DC3 | [CUDA 13.4U2 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) | — |
| DC4 | [Blackwell tuning guide v13.4](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | 2026-10-12 |
| DC5 | [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | 2026-10-12 |
| DC6 | [MIG supported profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html) | — |
| DC7 | [MIG concepts](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/concepts.html) | — |
| DC8 | [CUdevSmResource, CUDA 13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/structCUdevSmResource.html) | search summary |
| DC9 | [MPS docs](https://docs.nvidia.com/deploy/mps/index.html) | — |

**Proxies**

| ID | Source | Date | P/S |
|---|---|---|---|
| DP1 | [RTX PRO 6000 Workstation datasheet, 5349469](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf) | Jun 2026 | P |
| DP2 | [RTX PRO 6000 Max-Q product page](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/) | — | P |
| DP3 | [RTX PRO 6000 Server Edition page](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | 10-03 ledger | P |
| DP4 | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) | — | P |
| DP5 | [H100 page](https://www.nvidia.com/en-us/data-center/h100/) | 10-03 ledger | P |
| DP6 | [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era) | snippet only | P |
| DP7 | [TechPowerUp RTX PRO 6000](https://www.techpowerup.com/gpu-specs/rtx-pro-6000.c4272); [Chips and Cheese B200](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut) | — | S, locators |
| DQ1 | [Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) | — | P |

**Libraries** (raw files fetched 10-28 into scratch `dl_src/`)

| ID | Source | Date |
|---|---|---|
| DL1 | FlashAttention [setup.py](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py), [flash_fwd_sm100.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py), [README](https://github.com/Dao-AILab/flash-attention) | — |
| DL2 | FlashInfer [README](https://github.com/flashinfer-ai/flashinfer) and PRs [#6295](https://github.com/flashinfer-ai/flashinfer/pull/6295), [#6944](https://github.com/flashinfer-ai/flashinfer/pull/6944), [#6945](https://github.com/flashinfer-ai/flashinfer/pull/6945), [#6981](https://github.com/flashinfer-ai/flashinfer/pull/6981), [#6982](https://github.com/flashinfer-ai/flashinfer/pull/6982), [#7116](https://github.com/flashinfer-ai/flashinfer/pull/7116), [#7293](https://github.com/flashinfer-ai/flashinfer/pull/7293) | — |
| DL3 | CUTLASS [README](https://github.com/NVIDIA/cutlass/blob/main/README.md), [CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md) | 4.8.1, 2026-10-15 |
| DL4 | Triton [AccelerateMatmul.cpp](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp), [compiler.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py) | — |
| DL5 | vLLM [CMakeLists.txt](https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt) | — |
| DL6 | PyTorch [build_env_setup.py](https://github.com/pytorch/pytorch/blob/main/.ci/wheel/linux/build_env_setup.py) | — |
| DL7 | SGLang [README](https://github.com/sgl-project/sglang) | — |
| DL8 | TensorRT-Edge-LLM [releases](https://github.com/NVIDIA/TensorRT-Edge-LLM/releases), [PR #260](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260) | — |
| DL9 | [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) | 10-25 ledger |

**Prices**

| ID | Source | Date | P/S |
|---|---|---|---|
| DR1 | [vast.ai `bundles`](https://console.vast.ai/api/v0/bundles/), snapshot in scratch `vast_2026-10-28/` | 2026-10-28 16:15:50 UTC | P |
| DR2 | Repo `docs/research/2026-10-27-role-alignment/notes/vast_2026-10-27/` | 2026-10-27 08:30 UTC | P |
| DR3 | [RunPod pricing](https://www.runpod.io/pricing) | updated 2026-10-08 | P |
| DR4 | [Lambda pricing](https://lambda.ai/pricing) | undated | P |
| DR5 | [Nebius prices](https://nebius.com/prices) | undated | P |
| DR6 | [SabrePC 945-14070-0080-000](https://www.sabrepc.com/945-14070-0080-000-NVIDIA-S104023743) | live | P |
| DR7 | Searches for Thor rental: [GTC26 workshop setup](https://www.jetson-ai-lab.com/tutorials/gtc26/setup), [LaunchPad](https://nvidia.com/en-au/data-center/launchpad) | — | P; negative result |

**Rig**

| ID | Source | Date | P/S |
|---|---|---|---|
| DG1 | [Leopard, GMSL2 cameras on Thor](https://leopardimaging.com/gmsl2-cameras-on-thor/) | — | P |
| DG2 | [Leopard LI-GMSL2-DESER-HOLOSCAN](https://leopardimaging.com/product/accessories/adapters-carrier-boards/for-holoscan/li-gmsl2-deser-holoscan/) | — | P |
| DG3 | [SparkFun ZED-X20P](https://www.sparkfun.com/sparkfun-allband-gnss-rtk-breakout-zed-x20p-qwiic.html) | — | P |
| DG4 | [HSB PTP docs](https://docs.nvidia.com/holoscan/sensor-bridge/latest/ptp.html) | search summary | P |
| DG5 | Lattice LF-SNSR-ETH-EVN: [Mouser VN](https://www.mouser.vn/ProductDetail/Lattice/LF-SNSR-ETH-EVN) (timeout), [Farnell FR](https://fr.farnell.com/lattice-semiconductor/lf-snsr-eth-evn/kit-pont-capteur-vers-ethernet/dp/4640362) | snippets | — |
| DG6 | [Leopard IMX728-GM2B 120H](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/adi-gmsl2-cameras/li-imx728-gm2b/li-imx728-gm2b-120h-sf-aa/) | 10-25 ledger | P |

**Vietnam legal** (engineering reading, not legal advice)

| ID | Source | Date | P/S |
|---|---|---|---|
| DV1 | Decree 356/2025/NĐ-CP, Government-signed scan, via the [Nghệ An provincial portal](https://hcc.nghean.gov.vn/laws/detail/Nghi-dinh-so-356-2025-ND-CP-cua-Chinh-phu-Quy-dinh-chi-tiet-mot-so-dieu-va-bien-phap-thi-hanh-Luat-Bao-ve-du-lieu-ca-nhan-2058/) | 2025-12-31 | P |
| DV2 | [Cần Thơ legal-dissemination page on Law 91/2025](https://pbgdpl.cantho.gov.vn/luat-bao-ve-du-lieu-ca-nhan-nam-2025-bao-dam-quyen-rieng-tu-trong-ky-nguyen-so) | 2026-01-13 | P |
| DV3 | Law 116/2025/QH15: [luatvietnam locator](https://luatvietnam.vn/bai-viet-lien-quan/luat-an-ninh-mang-2025-422396.html); official [.docx on vbpl.moj.gov.vn](https://vbpl.moj.gov.vn/TW/Lists/vbpq/Attachments/187039/116.2025.QH15.docx), not opened | — | S, locator |
| DV4 | [VOV on accessories and Circular 30/2026](https://vov.vn/o-to-xe-may/cac-phu-kien-duoc-lap-them-cho-o-to-khong-bi-phat-va-truot-dang-kiem-post1318906.vov) | search summary | S, press |
| DV5 | 10-25 lane D note (repo) and the official [VBHN 55/VBHN-VPQH](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf) | 10-25 | P |
| DV6 | Searches on Decree 236/2026 amending Decree 151/2024 | search summary | S |
