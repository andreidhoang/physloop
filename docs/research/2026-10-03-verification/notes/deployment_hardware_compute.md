# VLA deployment efficiency, edge/datacenter hardware specs and rented-compute economics for driving VLAs on DRIVE/Jetson Thor (as of 2026-10-03)

## Q1. Do the cited VLA-efficiency sources exist under the claimed IDs, and do they report the attributed numbers?

### Takeaway
All nine claimed arXiv IDs resolve to the named papers, and the GR00T README numbers exist. Most attributed numbers are reported, but several are framed wrongly in the docs:
- **VLA-Perf** numbers come from an analytical roofline model, not from measurements.
- **FlashDrive's** 717 → 151 ms comes mostly from streaming KV reuse, speculative decoding, adaptive steps and CUDA graphs. W4A8 removes only 24.6 ms.
- **FlashVLA's** "20×" applies to action decoding only. End-to-end it is ≤2.43×.
- **FoldQuantVLA's** 1.25–1.52× is the desktop result. Orin gets 1.20–1.33×.
- **Jetson-PI's** 7.6 Hz needs system optimizations in addition to the async scheduling.

### Cited Findings

#### Conventions used in every ledger in this file
- Status ∈ {VERIFIED, CORRECTED, NOT FOUND, UNVERIFIABLE}. CORRECTED means the number or source exists but the doc's value, attribution or framing is wrong.
- The evidence column gives verbatim **data cells, labels and numbers**, which are facts. Prose is **paraphrased** to respect quoting limits; the single verbatim prose quote in this file is marked with quotation marks.
- Access date for every row is **2026-10-03** unless stated.
- Paper metadata comes from the arXiv API (export.arxiv.org). Numbers come from the arXiv HTML full texts.

#### Ledger: papers and sources

| Claim as written (doc § ) | Status | Evidence (data cell / paraphrase, ≤25 words) | Correct value if different | Primary source | Access |
|---|---|---|---|---|---|
| FlashDrive is arXiv 2608.12932, CoRL 2026 (02 §0; 01 C10) | VERIFIED (venue only via project page) | arXiv title "FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving"; authors Zekai Li, Yihao Liang, Hongfei Zhang, Jian Chen, Yesheng Liang, Zhijian Liu; submitted 2026-08-13 | arXiv comment lists only "15 pages; 8 figures"; the "CoRL 2026" label appears on the z-lab project page | https://arxiv.org/abs/2608.12932 ; https://z-lab.ai/projects/flashdrive | 2026-10-03 |
| Alpamayo 1.5: 717 → 151 ms with W4A8 + speculative decoding (02 §0) | CORRECTED (numbers right, attribution incomplete) | Table 1 (RTX PRO 6000, 1 trajectory) totals: Alpamayo 1.5 716.9 ms; "+ All above" 176.0 ms; "+ Quantization" 151.4 ms | 4.7× = CUDA graphs + fusion (1.40×) + streaming KV reuse + DFlash speculative reasoning + adaptive-step flow matching. W4A8 removes only 24.6 ms (176.0 → 151.4) | https://arxiv.org/html/2608.12932 | 2026-10-03 |
| FlashDrive keeps the action expert in BF16 (02 §0; 01 C10) | VERIFIED | §3.4 paraphrase: VLM weights are 4-bit (ParoQuant) with 8-bit activations (Marlin W4A8); the action expert stays BF16; footprint ~18.3 GB | — | https://arxiv.org/html/2608.12932 | 2026-10-03 |
| FlashDrive closed-loop evaluation in simulation (02 §0) | VERIFIED | Table 3: 100 random AlpaSim clips; collision 0.19 → 0.15, off-road 0.41 → 0.32, wrong-lane 0.45 → 0.51 (regresses) | Also reports AlpaSim per-step rollout latency 1,150 → 463 ms | https://arxiv.org/html/2608.12932 | 2026-10-03 |
| FlashDrive hardware (key question) | VERIFIED | Table 2, 1 sample: Jetson Thor 3770.3 → 943.6 ms (4.0×); RTX PRO 6000 716.9 → 151.4; 6 samples on Jetson Thor 14596.5 → 1522.6 ms | Primary GPU is a single RTX PRO 6000 (edition unstated); the edge device is **Jetson** Thor, not DRIVE Thor | https://arxiv.org/html/2608.12932 | 2026-10-03 |
| VLA-Perf is arXiv 2602.18397, Feb 20 2026 (04 production table) | VERIFIED | Title "How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf"; Jiang, Clemons, Sankaralingam, Kozyrakis; 2026-02-20 | — | https://arxiv.org/abs/2602.18397 | 2026-10-03 |
| π0 52.6 ms on Jetson Thor vs 3.2 ms on B100, listed as "Measured bottleneck" (04 production table) | CORRECTED (values exist; not measurements) | Table 3 E2E: Jetson Thor 52.57 ms (19.0 Hz); B100 3.18 ms (314.4 Hz). The paper calls VLA-Perf an analytical, roofline-based model that estimates upper bounds | Modeled optimum, not measured. The authors' own validation shows real systems reach 73.3–82.6% of roofline. Assumes Thor 400 TFLOP/s BF16 and 270 GB/s | https://arxiv.org/html/2602.18397v1 | 2026-10-03 |
| π0 action expert ~54 FLOP/byte, memory-bound (04; 01 C10) | VERIFIED | Table 4 header "Action (OI=54.0)", memory-bound on every listed GPU; vision OI 321.4, VLM OI 542.8 | This is π0's expert at chunk 50, not Alpamayo's | https://arxiv.org/html/2602.18397v1 | 2026-10-03 |
| 10 → 50 denoising steps = 2.15× latency (04) | VERIFIED (modeled, B100) | §4.5 paraphrase: going from 10 to 50 steps raises action latency 5× and overall VLA latency 2.15× (chunk 50, B100) | Roofline-model result | https://arxiv.org/html/2602.18397v1 | 2026-10-03 |
| PhyAI is arXiv 2608.03682 (04 levers) | VERIFIED | "PhyAI: Real-Time Physical AI at the Edge, Scalable Rollouts in the Cloud"; 26 authors; v1 2026-08-04, v3 2026-08-14 | — | https://arxiv.org/abs/2608.03682 | 2026-10-03 |
| π0.5 44 → 100 samples/s at B=32 via cross-env batching (04 levers) | VERIFIED (framing qualified) | B=1: 22.59 ms, 44.26 samples/s; B=32: 100.02 samples/s with a 319.94 ms synchronized batch; BF16, CUDA Graphs, 3 views, 10 Euler steps | Static batch sweep on an unnamed "Hopper-series GPU". On Thor: 7.43 → 16.25 samples/s at B=16, then 15.89 at B=32 | https://arxiv.org/html/2608.03682v3 | 2026-10-03 |
| Expert time share 57% → 13.5% (04 levers) | VERIFIED | At B=1 the expert is 8.8% of FLOPs but 57.2% of latency; at B=32 its share is 13.5% (Hopper-series GPU) | — | https://arxiv.org/html/2608.03682v3 | 2026-10-03 |
| Jetson-PI is arXiv 2607.12659, Aug 2026 (04 production table) | CORRECTED (date) | "Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference"; v1 2026-07-14, v5 2026-09-05; arXiv comment "CoRL 2026" | First posted July 2026, not August | https://arxiv.org/abs/2607.12659 | 2026-10-03 |
| π0.5 458 ms naive on Thor, 7.6 Hz only with async foresight (04 production table) | VERIFIED (qualified) | Table 4, Jetson Thor: Naive PI05 457.9 ms (2.18 Hz); final row 309.5 ms total, 131.8 ms reaction time, 7.59 Hz | 7.59 Hz needs scheduling plus graph reuse plus GPU buffering/unroll, not async alone. The abstract's headline is Orin (8.66×) | https://arxiv.org/html/2607.12659 | 2026-10-03 |
| realtime-vla is arXiv 2510.26742 (04 levers) | VERIFIED | "Running VLAs at Real-time Speed", Ma et al. (Dexmal), 2025-10-30 | — | https://arxiv.org/abs/2510.26742 | 2026-10-03 |
| CUDA graphs + fusion 106.5 → 27.3 ms (04 levers) | VERIFIED (baseline qualified) | Table 1, π0 on RTX 4090, 2 views: naive torch 106.5 ms; openpi/jax 53.7 ms; ours 27.3 ms | 3.9× vs naive PyTorch, but only 1.97× vs the openpi JAX path | https://arxiv.org/html/2510.26742 | 2026-10-03 |
| SnapFlow is arXiv 2604.05656, 274 → 83 ms (04 levers) | VERIFIED | π0.5 (3B), 10-step → 1-step; E2E 274 ms → 83 ms; LIBERO 98.75% vs 97.75%; measured on a single A800-80G | — | https://arxiv.org/abs/2604.05656 | 2026-10-03 |
| FlashVLA is arXiv 2608.27384, "up to 20×" (04 levers) | CORRECTED | Authors Zekai Li, Jiaming Tang, Zhijian Liu (2026-08-27). Per-step action-decoding latency is up to 20× lower (19.9× measured) | End-to-end per action ≤2.43× (LIBERO, 53.8 → 22.1 ms/step). Per invocation 45.8 → 26.7 ms on RTX 4090 and 37.0 → 20.3 ms on RTX 5090 | https://arxiv.org/html/2608.27384 | 2026-10-03 |
| HoloQ-VLA is arXiv 2605.28803, π0.5 98.0% vs 97.1% (04 levers) | VERIFIED | LIBERO W4A4: Pi-0.5 98.0% vs FP16 97.1%; GR00T-N1.5 87.8% vs 87.0%; static memory −74.2% | Reports **no latency or speedup** (relevant to the FP8/NVFP4 lever row) | https://arxiv.org/abs/2605.28803 | 2026-10-03 |
| FoldQuantVLA is arXiv 2609.24433, W4A4 1.25–1.52× (04 levers) | CORRECTED | Abstract cells: 1.20 to 1.33× over FP TensorRT on Orin; 1.25 to 1.52× on desktop | 1.25–1.52× is the desktop number (RTX 4070 Ti SUPER, TensorRT 10.15). Orin gives 1.20–1.33×. No Thor result | https://arxiv.org/abs/2609.24433 | 2026-10-03 |
| GR00T N1.7-3B TensorRT 27.9 ms on H100, 80.4 ms on Thor; eager → TRT 3.08× (04 models and levers) | VERIFIED (qualified) | README (4 denoising steps, 1 camera): H100 eager 85.8 → TRT 27.9 ms, 3.08x; AGX Thor 112.8 → 80.4 ms, **1.40x**; RTX Pro 6000 Blackwell 78.4 → 27.9 ms, 2.81x | 3.08× holds only on H100; Thor gets 1.40×. TRT engines are BF16-only with a static batch of 1. README last changed 2026-08-20 | https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md | 2026-10-03 |
| Alpamayo-R1 Table 14: 99 ms on the same GPU (vision 3.43 + prefill 16.54 + reasoning 70 ms/40 tok + trajectory 8.75 ms/5 steps) (04 one-tick) | VERIFIED | Table 14 "RTX 6000 Pro Blackwell": 3.43ms; 16.54ms; 70ms (40 tokens); 8.75ms (5 steps); 99ms | The table states neither model size nor precision, as the doc says | https://arxiv.org/html/2511.00088 | 2026-10-03 |
| SemiAnalysis Jun 16 2026: trainer idle 30% at 10.5% MFU, 60% of rollouts discarded (Qwen3-235B); idle 74% on GLM-5; 7,500 s stragglers (04 production table) | VERIFIED | Paraphrase: trainer waits 30%, 10.5% MFU; 60% of dispatched rollouts discarded; GLM-5 section: trainer waits 74%; tail latency up to 7500s (Qwen3-235B run) | Published 2026-06-16 | https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching | 2026-10-03 |
| World-action models are 3–4× slower than VLAs (NVIDIA WAM blog, Jun 15 2026) (04 production table) | VERIFIED | Paraphrase: citing Fast-WAM, WAM modes take 590–800 ms per action chunk vs ~190 ms for Pi-0.5, a 3–4x slowdown; dated Jun 15, 2026 | — | https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/ | 2026-10-03 |
| RoboWorld r = 0.989; slow inference limits large-scale throughput (04 production table) | VERIFIED | Pearson r=0.989, Spearman ρ=0.970 over 8 policies; the abstract names slow inference as a throughput limit; Step Forcing 15.3 FPS | Replicating RoboArena costs ~100 H100 GPU-hours | https://arxiv.org/abs/2607.01060 | 2026-10-03 |
| Chelsea Finn, YC podcast, Jul 2026: 1M one-minute trajectories = 700 robot-days (04 production table) | UNVERIFIABLE | No transcript found. Search found a Finn YC talk dated Aug 13 2026, without this figure | Arithmetic holds: 10^6 min = 16,667 h = 694 robot-days | (search only; no primary transcript) | 2026-10-03 |
| DreamZero 5.7 s → 150 ms, 38× (04 levers) | VERIFIED (qualified) | Table 1 paraphrase: system opts ~9× on H100 and ~16× on GB200; adding DreamZero-Flash gives 38× on GB200, 5.7 s → 150 ms | The 38× is GB200-only and uses CFG parallelism over 2 GPUs | https://arxiv.org/abs/2602.15922 | 2026-10-03 |

#### Supporting details worth carrying into the report
- **FlashDrive per-stage baseline (RTX PRO 6000, BF16/FP16 baseline, 1 trajectory)** ([arXiv HTML](https://arxiv.org/html/2608.12932)):

  | Configuration | Encode | Prefill | Decode | Action | Total |
  |---|---|---|---|---|---|
  | Unoptimized | 87.0 ms | 165.3 ms | 271.7 ms at 56.4 tok/s | 192.9 ms | 716.9 ms |
  | CUDA graphs + fusion | 40.5 ms | 188.3 ms | 170.6 ms at 89.5 tok/s | 113.9 ms | 513.3 ms |

  - Reasoning chains average ~16 tokens.
  - The evaluated flow solver uses 8 steps.
  - FP16 needs ~31.6 GB for 6 trajectory samples.
- **FlashDrive streaming (history-frame KV reuse) is measured on Alpamayo 1.5.**
  - Encode is 3.4× faster and prefill 3.2× faster.
  - It needs expert-only fine-tuning.
  - minADE6 goes 0.770 → 0.791 vs the system-optimized baseline (+2.7%) ([arXiv HTML](https://arxiv.org/html/2608.12932)).
- **PhyAI on GR00T N1.7 (RTX 5090)** ([arXiv HTML](https://arxiv.org/html/2608.03682v3)):
  - Throughput goes 48.8 → 138.8 samples/s from B=1 to B=32 (2.84×).
  - B=8 already reaches 92.2% of the B=32 throughput.
  - FlashRT's FP8 π0 runs in 49.5 ms on Thor.

### Inferences
- The docs' "measured" production-table row for π0 on Thor is a modeled upper bound. Measured Thor numbers in the same family are slower:
  - Jetson-PI: π0.5 at 457.9 ms naive.
  - PhyAI: π0.5 at 7.43 samples/s at B=1 (≈135 ms).
  - Use measured numbers when stating Thor feasibility.
- **Thor feasibility for the teacher.** FlashDrive's Jetson Thor result (943.6 ms per tick fully optimized, 1 sample) is the most relevant public datapoint for "teacher on Thor". The optimized 10B teacher is ~9× over a 100 ms budget on Jetson Thor.
- **Batching levers.** The 2.3–2.8× batching gain cited from PhyAI holds on datacenter/desktop GPUs. On Thor, π0.5 plateaus at ~2.2× by B=16.

### Gaps
- The PhyAI "Hopper-series GPU" model is not named.
- FlashDrive does not state the RTX PRO 6000 edition (Workstation or Server). This changes memory bandwidth (1,792 vs 1,597 GB/s).
- No primary source found for the Chelsea Finn figure.

## Q2. Hardware datasheet values (dense), ridge points, Jetson vs DRIVE AGX Thor, and the challenge limit

### Takeaway
The H100, H200 and B200 values in the docs match NVIDIA's pages once halved from the sparse figures. The Thor and RTX PRO 6000 values need qualifiers:
- **Jetson AGX Thor:** the 258 TFLOPS BF16 figure is derived from NVIDIA's "517 TFLOPs (Dense FP8 | Sparse FP16)"; NVIDIA prints no dense BF16 number.
- **RTX PRO 6000:** NVIDIA prints no "504 TFLOPS" figure (≈500 dense is derived). The 1.79 TB/s bandwidth is the Workstation Edition; the Server Edition is 1.597 TB/s.
- **DRIVE AGX Thor** is a different product from Jetson AGX Thor: 1,000 INT8 TOPS / 2,000 FP4, 64 GB, 273 GB/s, DriveOS 7, ASIL-D design.

The challenge's 16 GiB / 0.1 s is a per-replica packing rule on datacenter GPUs and has nothing to do with Thor.

### Cited Findings

#### Ledger: hardware specifications

| Claim as written (doc § ) | Status | Evidence (verbatim spec cells) | Correct value if different | Primary source | Access |
|---|---|---|---|---|---|
| H100 989 TFLOPS BF16 dense (04 L1) | VERIFIED | H100 SXM: "BFLOAT16 Tensor Core* 1,979 teraFLOPS"; "* With sparsity" | Dense 989.5 TFLOPS. FP8 3,958 sparse, 1,979 dense | https://www.nvidia.com/en-us/data-center/h100/ | 2026-10-03 |
| H100 3.35 TB/s (04 L1) | VERIFIED | "GPU Memory 80GB"; "GPU Memory Bandwidth 3.35TB/s" (SXM) | H100 NVL: 94 GB, 3.9 TB/s | https://www.nvidia.com/en-us/data-center/h100/ | 2026-10-03 |
| H200 ridge 206 / 412 (04 ridge points) | VERIFIED | H200 SXM: "BFLOAT16 Tensor Core² 1,979 TFLOPS", "FP8 … 3,958", "² With sparsity", "141GB", "4.8TB/s" | 989.5 / 4.8 = 206.1; 1,979 / 4.8 = 412.3 | https://www.nvidia.com/en-us/data-center/h200/ | 2026-10-03 |
| B200 2.25 PFLOPS BF16 dense (04 L1) | VERIFIED | HGX B200 (8 GPUs): "FP16/BF16 Tensor Core 36 PFLOPS"; footnote: dense is ½ the sparse spec shown | 36 / 2 / 8 = 2.25 PFLOPS per GPU. FP8 dense 4.5 PFLOPS per GPU; FP4 dense 9 | https://www.nvidia.com/en-us/data-center/hgx/ | 2026-10-03 |
| B200 8 TB/s (04 L1) | VERIFIED | DGX B200: "1,440 GB total, 64 TB/s HBM3e bandwidth" (8 GPUs) | 8 TB/s per GPU | https://www.nvidia.com/en-us/data-center/dgx-b200/ | 2026-10-03 |
| RTX PRO 6000 504 TFLOPS BF16 (04 L1) | CORRECTED (not printed by NVIDIA; ≈ derived) | Server Ed.: "FP4 Tensor Core 4 PFLOPS"; "FP8 Tensor Core 2 PFLOPS"; "FP16 \| BF16 Tensor Core 1 PFLOP". WS datasheet: "4000 AI TOPS", footnoted as sparse FP4 | ≈500 TFLOPS dense, derived as 1 PFLOP ÷ 2 (sparse-equivalent, consistent with the FP4 sparse footnote). 503.8 appears only on third-party spec databases | https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/ ; https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf | 2026-10-03 |
| RTX PRO 6000 1.79 TB/s, 96 GB (04 L1) | VERIFIED for Workstation; CORRECTED for Server | WS datasheet: "96 GB GDDR7 with ECC", "Memory Bandwidth 1792 GB/s", 600 W. Server Ed. page: "Memory Bandwidth 1597 GB/s" | Server Edition: 1.597 TB/s. Clouds list both: vast.ai "RTX PRO 6000 WS" and "S"; RunPod "RTX PRO 6000" and "WK" | WS datasheet (above); SE page (above) | 2026-10-03 |
| Jetson Thor 258 TFLOPS (BF16 dense) (04 L1) | VERIFIED (derived) | Jetson T5000: "2070 TFLOPS (Sparse FP4)"; "1035 TFLOPS (Dense FP4 \| Sparse FP8 \| Sparse INT8)"; "517 TFLOPs (Dense FP8 \| Sparse FP16)" | Dense FP16 = 517 / 2 ≈ 258.5 TFLOPS. BF16 is assumed to run at the FP16 rate; NVIDIA prints no BF16 figure | https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/ | 2026-10-03 |
| Jetson AGX Thor 2070 FP4 TFLOPS sparse, 273 GB/s, 128 GB (key question; 01 C2) | VERIFIED | "2070 TFLOPS (FP4—Sparse)"; "128 GB 256-bit LPDDR5X 273 GB/s"; 2560-core Blackwell GPU; 14-core Neoverse-V3AE; 130 W | Jetson T4000: 1200 TFLOPS (FP4—Sparse), 64 GB, 273 GB/s | https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/ | 2026-10-03 |
| "Thor bandwidth is 273 GB/s" (01 C2) | VERIFIED (both Thors) | DRIVE AGX Thor DevKit: "System RAM 64 GB LPDDR5X"; "Memory bandwidth (256-bit LPDDR5) Up to 273 GB/s" | — | https://developer.nvidia.com/drive/agx | 2026-10-03 |
| DRIVE AGX Thor figure, precision and how it differs (key question) | VERIFIED (sparsity unstated) | DevKit SoC: "up to 1000 INT8 TOPS \| 2000 FP4 FLOPs"; DriveOS 7, TensorRT 10, CUDA 13; designed for ISO 26262 ASIL-D | 64 GB, not 128 GB. Runs DriveOS with DriveWorks, not JetPack. Automotive I/O (16× GMSL2 + 2× GMSL3). Dense BF16 not published | https://developer.nvidia.com/drive/agx ; https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ | 2026-10-03 |
| Ridge points B200 281/562; H100 295/591; RTX PRO 6000 281/562; Jetson Thor 947/1,894 (04) | VERIFIED (with scope notes) | Recomputed below from the cells above | The 947/1,894 pair is **Jetson AGX Thor (T5000)**. DRIVE AGX Thor ≈916/1,832 only if its 2,000 FP4 is sparse (assumption). RTX PRO 6000 **Server** ≈313/626 | (computed) | 2026-10-03 |
| "Blackwell BF16 ridge ≈ 281" (04 prediction 2; 01 C10) | VERIFIED for B200 and RTX PRO 6000 WS | 2,250 / 8 = 281.2; 500 / 1.792 = 279 | VLA-Perf's own B100 balance OI is 218.8; its Thor balance OI is 1,481.5 (it assumes 400 TFLOP/s) | https://arxiv.org/html/2602.18397v1 | 2026-10-03 |
| "Submissions run in 16 GiB VRAM at 0.1 s model-time per call" (02 §0; 01 C2) | CORRECTED (nuance) | README paraphrase: aim for ≤0.1 s model work per Drive call, enforced by a per-track throughput budget; each driver instance ≤16 GiB VRAM | A throughput budget, not a hard per-call limit. EC2 preset runs 16 replicas on GPUs 4–7 with 2 concurrent rollouts each; GPU model unstated. Image ≤40 GiB; no outbound network | https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md | 2026-10-03 |
| Registration requires an organization email; deadline Oct 31 2026 (02 §0) | VERIFIED | Challenge page paraphrase: the team captain's verified HF primary email must be an organization address; registration closes Oct 18, deadline Oct 31, 2026 | New: registration closes **Oct 18, 2026** | https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026 | 2026-10-03 |

#### Ridge-point recomputation (dense; FLOP/byte = peak FLOP/s ÷ memory bytes/s)

| Device | BF16 dense | FP8 dense | Bandwidth | BF16 ridge | FP8 ridge |
|---|---|---|---|---|---|
| H100 SXM | 989.5 TFLOPS | 1,979 | 3.35 TB/s | 989.5e12 / 3.35e12 = **295.4** | **590.7** |
| H200 SXM | 989.5 | 1,979 | 4.8 TB/s | **206.1** | **412.3** |
| B200 (per GPU) | 2,250 | 4,500 | 8.0 TB/s | **281.2** | **562.5** |
| RTX PRO 6000 WS | ≈500 (derived) | ≈1,000 | 1.792 TB/s | **≈279** (281 with 503.8) | **≈558** (562) |
| RTX PRO 6000 Server Ed. | ≈500 | ≈1,000 | 1.597 TB/s | **≈313** | **≈626** |
| Jetson AGX Thor (T5000) | 258.5 (derived) | 517 | 273 GB/s | **947** | **1,894** |
| DRIVE AGX Thor (assumes 2,000 FP4 is sparse) | ≈250 | ≈500 | 273 GB/s | **≈916** | **≈1,832** |

### Inferences
- **Challenge limit vs Thor.** The 16 GiB / 0.1 s limit is unrelated to Thor:
  - It is a VRAM cap per replica on shared datacenter GPUs. The preset packs 4 replicas per GPU, which implies ≥64 GiB-class cards, though the model is unstated.
  - Thor's binding constraint is 273 GB/s of bandwidth, plus DriveOS GPU carveouts (see Q4), rather than capacity.
- **Bandwidth gap.** A model that meets 0.1 s on an RTX PRO 6000 (1.6–1.8 TB/s) or an H100 (3.35 TB/s) will be about 6–12× slower in bandwidth-bound stages on Thor (273 GB/s).
- **What fits in 16 GiB (17.18 GB).** Alpamayo 1.5 in BF16 has 22.16 GB = 20.6 GiB of weights and does not fit. FP8 (~11 GB ≈ 10.3 GiB) fits with ~5.7 GiB headroom.

### Gaps
- NVIDIA publishes no dense BF16 TFLOPS for Jetson Thor, DRIVE Thor or RTX PRO 6000. All dense BF16 values above are derived from sparse or FP4 figures under the standard Blackwell 4:2:1 FP4:FP8:BF16 ratio.
- NVIDIA does not say whether DRIVE Thor's "2,000 FP4" figure is sparse.
- The GPU model used by the challenge's EC2 evaluation is not stated.

## Q3. Recomputation of the docs' first-principles workload numbers from verified specs

### Takeaway
The docs' ~490 ms per-tick total for Alpamayo 1.5 at BF16, B=1 on an RTX PRO 6000 happens to match FlashDrive's measured system-optimized 513 ms. The stage split is wrong, though:
- **Encode + prefill** is underestimated (doc 98–150 ms, no ViT; measured 229–252 ms).
- **Decode** is overestimated (doc assumes 40 tokens; Alpamayo 1.5 produces ~16).
- **Expert** is underestimated 4–7× (doc 26 ms; measured 114–193 ms).

The batched (B=32) claims contain two clear errors:
- **Expert:** the claimed "~0.1 ms/sample" has a compute floor of ≈7 ms/sample, about 70× higher.
- **Decode:** per-sample KV-cache reads roughly double the per-sample cost relative to the doc.

The memory arithmetic in 01 C2 is correct.

### Cited Findings

#### Verified inputs
- **Parameter split from the safetensors headers** ([HF files](https://huggingface.co/nvidia/Alpamayo-1.5-10B/tree/main)). Total 11,078,526,194 parameters, BF16, 22,157,052,388 bytes:

  | Component | Parameters |
  |---|---|
  | LM transformer layers | 6.946e9 |
  | embed_tokens | 0.638e9 |
  | lm_head (untied) | 0.638e9 |
  | ViT | 0.576e9 |
  | Action expert | 2.279e9 |

  The card's rounded figures are "Backbone: 8.2B", "Action Expert: 2.3B" ([model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B)).
- **VLM attention shapes:** 36 layers; q_proj [4096, 4096]; k_proj and v_proj [1024, 4096], i.e. 8 KV heads × 128.
- **KV-cache size:** 36 × 2 × 1024 × 2 bytes = **147,456 bytes per token**. A 3,000-token prompt therefore holds 0.442 GB of KV.
- **Expert shapes:** 36 layers, hidden 2048, k/v [1024, 2048] matching the VLM's KV dimension. So each expert step attends over the VLM KV cache. Config: `expert_non_causal_attention: true`, per-waypoint action projection, 64 waypoints ([config.json](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/main/config.json)).
- **Inputs:** 4 cameras × 4 frames, downsampled to 320×576 ([model card](https://huggingface.co/nvidia/Alpamayo-1.5-10B)).
- **ViT tokens per image:** patch_embed [1152, 3, 2, 16, 16] (patch 16) and a 4,608-wide merger (2×2 merge) give 320·576/16² = 720 patches → **180 tokens per image** → 2,880 for 16 images. The docs' 180/2,880 is consistent, assuming each frame is a separate image item; TensorRT-Edge-LLM documents an ordered list of image items ([TRT-Edge-LLM Alpamayo doc](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/alpamayo.md)).
- **Stock flow steps:** `num_inference_steps: int = 10` (Euler) ([flow_matching.py](https://github.com/NVlabs/alpamayo1.5/blob/main/src/alpamayo1_5/diffusion/flow_matching.py)). The docs' "10 Euler steps" is the code default. FlashDrive evaluated an 8-step solver; the AR1 paper's Table 14 used 5.
- **Specs:** from Q2.
- **Assumptions (labeled):**
  - ~3,000 prompt tokens.
  - 64 expert tokens per sample.
  - 65% compute attainment, where stated.

#### Step 1. Reproduce the docs' own arithmetic (RTX PRO 6000 at 504 TFLOPS, 1.79 TB/s)
- **Prefill:** 2 × 8.2e9 × 3,000 = 4.92e13 FLOP. 4.92e13 ÷ 5.04e14 FLOP/s = 97.6 ms at peak; ÷ (0.65 × 5.04e14) = 150.2 ms. The arithmetic is correct.
- **Decode:** 16.4e9 B ÷ 1.79e12 B/s = 9.16 ms/token; × 40 tokens = 366.5 ms. Correct.
- **Expert:** 2.3e9 × 2 B × 10 steps = 46 GB; ÷ 1.79e12 B/s = 25.7 ms. Correct.
- **Total:** 97.6 + 366.5 + 25.7 = **489.8 ms** (~2 Hz). Correct.

#### Step 2. Refined model with verified parameters, RTX PRO 6000 WS, BF16, B=1

**Prefill + encode FLOPs**

| Term | Calculation | FLOP |
|---|---|---|
| LM linear | 2 × 6.946e9 × 3,000 | 4.17e13 |
| LM causal attention | 2 × 3,000² × 4,096 × 36 | 2.65e12 |
| ViT | 2 × 0.41e9 block params × 11,520 patches + per-image attention | ≈1.05e13 |
| **Total** | | **≈5.48e13** |

5.48e13 ÷ 5.0e14 = **110 ms at peak**, or 169 ms at 65%.
- Measured (FlashDrive): encode + prefill = 87.0 + 165.3 = **252 ms** unoptimized and 40.5 + 188.3 = **229 ms** system-optimized.
- Prefill alone runs at ~47% of peak (4.4e13 FLOP ÷ 0.188 s = 234 TFLOP/s).

**Decode**
- Bytes per token: (6.946e9 + 0.638e9) × 2 B = 15.17 GB of weights + 0.44 GB of KV = 15.6 GB.
- At 1.792 TB/s: **8.7 ms/token** (WS). At 1.597 TB/s: **9.8 ms/token** (SE).
- Measured: 11.2 ms/token system-optimized (89.5 tok/s, ≈78% of the WS bound); 17.7 ms/token unoptimized (56.4 tok/s).
- Reasoning length is **~16 tokens** (FlashDrive), not 40. Decode bound = 16 × 8.7 = 139 ms; measured 171–272 ms.

**Expert (per Euler step)**
- Bytes: 2.279e9 × 2 B = 4.56 GB of weights + 0.44 GB of KV = 5.0 GB → 2.79 ms/step.
- 10 steps → **27.9 ms**; 8 steps → 22.3 ms.
- Measured for 8 steps: **192.9 ms** unoptimized, **113.9 ms** system-optimized. That is ~20% of bandwidth even after CUDA graphs (22.3 ÷ 113.9), i.e. 4–7× slower than the doc's 26 ms.

**Totals**
- Bound: ≈486 ms (doc assumptions of 10 steps and 40 tokens) or ≈271 ms (8 steps, 16 tokens).
- Measured: 716.9 ms unoptimized and 513.3 ms system-optimized.
- Verdict: the doc's 490 ms total is right by accident. Its per-stage split is wrong in all three stages.

**Alpamayo-R1's 99 ms** works out to 70 ms ÷ 40 = 1.75 ms/token, which is 5.0× below the refined BF16 bandwidth bound (8.7 ms/token). The doc's "5× faster" flag stands. AR1 also used 5 flow steps, versus 10 in Alpamayo 1.5's code default.

#### Step 3. Crossover B* = b·F_peak / (2·BW), with one row per sample and weights only
- **Decode, BF16 (b = 2):** B* = F/BW.
  - RTX PRO 6000 WS: **279**. H100: **295**. B200: **281**. Jetson Thor: **947**.
  - The doc's "decode stays memory-bound at B=32 (32 < 281)" is VERIFIED.
- **FP8 (b = 1 at the FP8 peak):** B* is the same number, because both terms halve.
- **Expert with 64 tokens per sample:** B* (in samples) = 279 ÷ 64 = **4.4** (Thor: 14.8). The doc's "B*=5" is VERIFIED as a number.

#### Step 4. Batched claims at B=32 (RTX PRO 6000 WS, BF16)

**Decode**
- Per batched step: 15.17 GB of weights + 32 × 0.442 GB of KV = 29.3 GB → 16.4 ms per step.
- Per sample per token: 0.51 ms. For 40 tokens: **20.5 ms/sample**; for 16 tokens: 8.2 ms.
- The doc says 11 ms per sample for 40 tokens. **CORRECTED:** the doc omits KV reads, which scale with B.

**Expert**
- Per step per sample: 2 × 2.279e9 × 64 = 2.92e11 FLOP of linear layers, plus attention 64 × 3,064 × 36 × 2,048 × 4 = 5.8e10, for a total of 3.5e11 FLOP.
- At B=32: 1.12e13 FLOP per step ÷ 5e14 FLOP/s = 22.4 ms per step, which is compute-bound since the memory time is 10.4 ms.
- Per sample over 10 steps: **7.0 ms at peak** (10.8 ms at 65%).
- The doc says "~0.1 ms/sample". **CORRECTED:** about 70× too low. Above B* the floor is compute, not amortized weight reads.

**Prefill**
- The doc says "unchanged at ~98–150 ms/sample". **CORRECTED:** 110 ms at peak, 169 ms at 65%. Measured attainment of ~48% implies ~230 ms.
- Streaming KV reuse (FlashDrive) cuts this about 3×.

**Per-tick per sample**
- Peak-bound: 110 + 8–20 + 6–7 ≈ **124–137 ms**, i.e. ≤7–8 ticks/s per GPU.
- At measured prefill attainment: ≈245–260 ms → **~4 ticks/s per GPU → ~0.02 episodes/s** (200-tick episodes).
- The doc says 110–160 ms → 6–9 ticks/s → 0.03–0.045 episodes/s, which is optimistic by up to ~2×.
- Batching-gain check: measured π0.5 batching gain is only 2.26× (PhyAI).

#### Step 5. Renderer and trainer bounds
- **Renderer (NuRec):** NuRec states 100 FPS at 1080P on an RTX6000Pro, with six cameras and 3.33 reconstructions per hour ([NuRec doc](https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html)).
  - If the 100 FPS counts single-camera frames: 100 ÷ 4 = **25 ticks/s** (the doc's reading).
  - If it counts six-camera frames: 100 × 6 ÷ 4 = **150 ticks/s**.
  - UNVERIFIABLE which reading is meant.
- **Pixel ratio:** 1920×1080 ÷ 576×320 = **11.25×**. The doc's "~11×" is VERIFIED.
- **AlpaSim throughput:** the doc says it is unpublished. **CORRECTED:** FlashDrive reports AlpaSim per-step rollout latency of 1,150 → 463 ms, covering inference, trajectory optimization and rendering ([arXiv HTML](https://arxiv.org/html/2608.12932)).
- **Trainer, doc arithmetic:** 6 × 10.5e9 × 3,300 = 2.079e14 FLOP. ÷ (0.40 × 989.5e12 FLOP/s) = **0.525 s/tick**. The arithmetic is VERIFIED.
- **Trainer, refined:**
  - LM: 6 × 6.946e9 × 3,300 = 1.375e14.
  - ViT, if trained: 6 × 0.576e9 × 11,520 = 3.98e13. If frozen, forward only: 1.33e13.
  - Expert: 6 × 2.279e9 × 64 tokens × 10 steps = 8.75e12.
  - Total: **1.6–1.9e14 FLOP → 0.40–0.47 s/tick**.
- **Trainer samples/s:** "~1.9 samples/s/GPU" = 1/0.525 is VERIFIED arithmetic.
- **vLLM rollout:** "~13 samples/s/GPU" is UNVERIFIABLE; no derivation or source is given.
- **GPU split:** "NVIDIA's 512 policy : 128 rollout GPU split" is VERIFIED. The recipe's cluster example uses 512 policy GPUs and 128 rollout GPUs (640 GPUs, 80 nodes). The local test needs ≥5 GPUs of ≥80 GB each, i.e. 4 policy (dp_shard 4) + 1 rollout ([RL recipe](https://github.com/NVlabs/alpamayo-recipes/blob/main/recipes/alpamayo1_x_rl/README.md)).
- **Closed-loop backprop:** 200 × 0.525 = **105 GPU-s**. The doc's ~106 is VERIFIED.
- **"2–3× more than generating it"** is inconsistent with the doc's own batched generation estimate: 200 ticks × 0.11–0.16 s = 22–32 GPU-s, so backprop would be 3.3–4.8× the generation cost.

#### Step 6. Memory arithmetic (01 C2)

| Item | Weights | Fits 16 GiB (17.18 GB)? |
|---|---|---|
| Alpamayo 1.5 BF16 | 22.16 GB = 20.6 GiB | **No** |
| Alpamayo 1.5 FP8 (all parameters) | 11.08 GB = 10.3 GiB | Yes. The recipe table lists FP8 ~11 GB and AutoQuant 6.5 BPE ~9 GB |
| 3B student BF16 | 6.0 GB = 5.59 GiB | Yes |
| KV cache, B=32 × 3,000 tokens | 14.2 GB | Fits the 96 GB card |

Sources: [safetensors index](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/main/model.safetensors.index.json); [quant recipe](https://github.com/NVlabs/alpamayo-recipes/blob/main/recipes/alpamayo1_5_quant/README.md).

#### Step 7. Thor bandwidth floors (273 GB/s, weights only, per decoded token)

| Model / precision | Bytes per token | Floor per token |
|---|---|---|
| Alpamayo 1.5 BF16 | 15.17 GB | **55.6 ms** |
| 3B student BF16 | 6 GB | 22.0 ms |
| 3B student FP8 | 3 GB | 11.0 ms |
| 3B student NVFP4 (~4.5 bits) | 1.69 GB | ~6.2 ms |

- Full Alpamayo 1.5 BF16 bound on Jetson Thor: prefill 212 ms + decode 16 × 57 ms + expert 8 × 18.3 ms ≈ **1.27 s**.
- Measured on Jetson Thor (FlashDrive): **3.77 s** baseline and **0.94 s** fully optimized.

#### Ledger: workload claims

| Claim as written (doc § ) | Status | Evidence | Correct value | Source | Access |
|---|---|---|---|---|---|
| Vision: 4 cams × 4 frames at 320×576, patch 16, merge 2 → 180 tokens/image, 2,880 total (04 one-tick) | VERIFIED | Card: 4 cameras, 4 frames per camera, downsampled to 320x576. Weights: patch_embed [1152,3,2,16,16] | — | HF card and safetensors | 2026-10-03 |
| Prefill ≈ 4.9e13 FLOP, ≈98 ms at peak, ~150 ms at 65% (04) | CORRECTED | FlashDrive Table 1: Encode 87.0, Prefill 165.3 (unoptimized); 40.5 and 188.3 (system-optimized) | ≈5.5e13 FLOP including ViT and attention; 110–169 ms modeled; 229–252 ms measured | arXiv 2608.12932 | 2026-10-03 |
| Decode 16.4 GB/token → 9.2 ms; 40 tokens → 366 ms (04) | CORRECTED | Table 1 decode: 271.7 ms at 56.4 tok/s; 170.6 ms at 89.5 tok/s; ~16-token reasoning | 15.6 GB/token → 8.7 ms (WS) or 9.8 ms (SE); 171–272 ms measured | arXiv 2608.12932; safetensors | 2026-10-03 |
| Expert 2.3B × 10 steps → 46 GB → 26 ms (04) | CORRECTED | Table 1 action: 192.9 ms and 113.9 ms (8-step solver, 1 trajectory) | Bound 27.9 ms (10 steps, including KV); measured 4–7× higher | arXiv 2608.12932; flow_matching.py | 2026-10-03 |
| Total ≈ 500 ms per tick (~2 Hz) (04) | VERIFIED (total), CORRECTED (split) | Table 1 totals 716.9 ms and 513.3 ms | Matches the system-optimized measurement by coincidence | arXiv 2608.12932 | 2026-10-03 |
| B=32 decode per sample drops 32× to 11 ms (04) | CORRECTED | Computed (Step 4) | ≈20.5 ms (40 tokens) or 8.2 ms (16 tokens), because of KV reads | computed | 2026-10-03 |
| Expert crosses B*=5, ~0.1 ms/sample (04) | CORRECTED | Computed (Step 4) | B*≈4.4 is right; cost ≈7.0 ms/sample at peak | computed | 2026-10-03 |
| Per-tick 110–160 ms → 6–9 ticks/s → 0.03–0.045 episodes/s (04) | CORRECTED | Computed (Step 4) | 124–137 ms peak-bound; ~250 ms at measured attainment → ~4 ticks/s → ~0.02 episodes/s | computed | 2026-10-03 |
| NuRec 100 FPS at 1080p × 6 cams on RTX 6000 Pro → ~25 ticks/s (04) | VERIFIED (source); UNVERIFIABLE (25 ticks/s) | NuRec: 6-camera clip, RTX6000Pro, 100 FPS at 1080P, 3.33 reconstructions per hour | 25 or 150 ticks/s depending on FPS definition | NuRec doc | 2026-10-03 |
| AlpaSim throughput unpublished (04) | CORRECTED | FlashDrive: AlpaSim per-step rollout latency 1150 ms → 463 ms | Published by a third party (FlashDrive) | arXiv 2608.12932 | 2026-10-03 |
| Trainer 2.1e14 FLOP/tick, 0.53 s on H100 at 40% MFU (04) | VERIFIED (arithmetic); refined | Computed (Step 5) | 1.6–1.9e14 → 0.40–0.47 s | computed | 2026-10-03 |
| vLLM rollout ~13 samples/s/GPU (04) | UNVERIFIABLE | No derivation in the docs | — | — | 2026-10-03 |
| 512 policy : 128 rollout GPU split; stock 4 policy + 1 vLLM GPU (04) | VERIFIED | Recipe: 512 policy GPUs + 128 rollout GPUs (80 nodes); local test needs ≥5 GPUs of ≥80 GB | — | RL recipe README | 2026-10-03 |
| Alpamayo 1.5 BF16 ≈ 22 GB; FP8 ≈ 11 GB; 3B student BF16 ≈ 6 GB (01 C2) | VERIFIED | index total_size 22157052388; recipe table "BF16 ~22 GB", "FP8 ~11 GB", "Autoquant 6.5BPE ~9 GB" | 20.6 GiB / ~10.3 GiB / 5.6 GiB | HF index; quant recipe | 2026-10-03 |
| Lever: history-frame KV reuse 2.5–3.5× on prefill, ≤1% minADE; "AV unmeasured" (04 levers) | CORRECTED | FlashDrive streaming: encode 3.4×, prefill 3.2×; minADE6 0.770 → 0.791 | Measured on Alpamayo 1.5; minADE6 +2.7%, which misses the ≤1% target | arXiv 2608.12932 | 2026-10-03 |
| Lever: CUDA graphs + fused Euler loop 3–4× at B=1 (04 levers) | CORRECTED | FlashDrive system optimizations 716.9 → 513.3 ms (1.40×); GR00T on Thor 1.40× | For Alpamayo 1.5 the measured gain is 1.40×. 3.08× is GR00T on H100; 3.9× is vs naive PyTorch | arXiv 2608.12932; GR00T README | 2026-10-03 |
| Lever: FP8 prefill / NVFP4 decode 1.4–1.8× prefill, ~2× decode, evidence HoloQ-VLA (04 levers) | CORRECTED | FlashDrive W4A8 after other optimizations: prefill 58.9 → 47.2 ms; decode 57.5 → 45.3 ms; total 176.0 → 151.4 ms | Measured 1.25× prefill, 1.27× decode, 1.16× end-to-end. HoloQ reports no latency | arXiv 2608.12932; arXiv 2605.28803 | 2026-10-03 |
| Lever: cross-scenario batching 2.3–2.8×, plateau at B≈16–32 (04 levers) | VERIFIED (approx.) | PhyAI: π0.5 2.26× (Hopper, B=32); GR00T 2.84× (RTX 5090); Thor π0.5 2.19× peak at B=16 | Alpamayo itself is unmeasured | arXiv 2608.03682 | 2026-10-03 |

### Inferences
- **Reasoning tokens drive B=1 latency.** For Alpamayo-class models, the dominant B=1 cost is (i) the reasoning-token count and (ii) an expert that runs far below its bandwidth bound, not weight bytes alone. FlashDrive's ablation shows the large wins come from algorithmic changes (streaming, speculative drafts, step caching) rather than precision.
- **Thor for a ≤3B student.** Thor's bandwidth floor makes BF16 reasoning at 22 ms/token incompatible with a 0.1 s budget for more than ~3–4 tokens. A ≤3B Thor student likely needs FP8/NVFP4 weights plus few or no reasoning tokens, or a non-autoregressive drafter. This is an inference from the floors above, not a measured result.
- **Plan-level effect.** The corrected batched numbers halve the docs' expected rollout throughput per GPU. Any budget or schedule derived from 6–9 ticks/s/GPU should use ~4 until measured.

### Gaps
- No public measurement of Alpamayo 1.5 at B>1 was found.
- The actual prompt token count per tick is not measured (3,000 is assumed).
- The cause of the expert's low bandwidth utilization (KV reads, small kernels, or attention over ~3k keys across 36 layers) is not diagnosed in public sources.
- The RTX PRO 6000 edition FlashDrive used is unknown.
- **Lever rows not verified here** (outside deployment/hardware scope, or marked unmeasured by the doc itself):
  - "Async learner, bounded staleness 1.5–1.9× (RL-VLA³ +85%; RLinf 1.9–2.4×)" is an RL-framework claim.
  - "Render at model resolution ≥4×" and "Tick subsampling / frozen lower layers 2–4×" are marked unmeasured in the doc, so there is nothing to check. The 11.25× pixel ratio is verified above.
  - 01 C10's "AlpaGym … publishes no throughput" was not checked.

## Q4. Deployment toolchain status on Thor (TensorRT, TensorRT-Edge-LLM, ModelOpt, CUDA graphs, DriveOS) as of October 2026

### Takeaway
**What TensorRT itself supports.** TensorRT 11.3 lists BF16, FP8 and FP4 as supported on both Jetson AGX Thor and DRIVE AGX Thor (SM 11.0).

**What the VLA tooling actually ships.**
- TensorRT-Edge-LLM exports Alpamayo-R1 in **FP16 only**.
- Alpamayo 1.5 support, promised for August, is still missing as of mid-September.
- The GR00T TensorRT pipeline is BF16-only.

**Corrections to the docs.**
- The DRIVE Thor OOM thread **was answered by NVIDIA and reported working on 2026-10-01**.
- The "not available for AGX Thor" quote refers to the Alpamayo 1.5 NIM container.

**Safety certification.** Certified-inference specifics publicly documented (QNX-only safety runtime, kSAFETY engines) are DriveOS 6 documents. DriveOS 7 specifics remain a gap.

### Cited Findings

| Claim as written (doc § ) / question | Status | Evidence | Correct value / note | Source | Access |
|---|---|---|---|---|---|
| TensorRT-Edge-LLM runs Alpamayo-R1 on Thor in FP16 only (02 §0) | VERIFIED | Alpamayo example note: only FP16 is supported for Alpamayo export; engines are built on the Thor device | Supported checkpoint is nvidia/Alpamayo-R1-10B, not 1.5 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/alpamayo.md | 2026-10-03 |
| "…with an unanswered OOM report" (02 §0) | CORRECTED | Topic 382501 (DRIVE AGX Thor, DriveOS 7.2.5 EA): build failed allocating 15,167,621,888 bytes. NVIDIA moderator suggested a hugepages/carveout change; user reports working on 2026-10-01 | Only the duplicate topic 382500 lacks a staff reply. The main thread was answered and resolved | https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501 ; https://forums.developer.nvidia.com/t/alpamayo-r1-10b-tensorrt-engine-oom-on-drive-agx-thor/382500 | 2026-10-03 |
| Alpamayo "not available for AGX Thor currently" (04 SoTA) | VERIFIED (context) | NVIDIA moderator, 2026-09-07: "Alpamayo is not available for AGX Thor currently." The same post points to TensorRT-Edge-LLM | Context is the Alpamayo 1.5 NIM container. NVIDIA also states NIM and TensorRT-LLM are unsupported on Jetson. A user runs 1.5 natively on a Thor dev kit (~14.5 GB) | https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409 ; https://forums.developer.nvidia.com/t/build-alpamayo1-5-native-on-thor/382647 | 2026-10-03 |
| Alpamayo 1.5 in TensorRT-Edge-LLM | NOT FOUND (pending) | Issue #130: a 2026-07-30 reply said August release; 2026-09-15 comment says still unavailable | Quantized Alpamayo export requested in #134 (open) | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/134 | 2026-10-03 |
| Precisions on Thor (FP8, NVFP4?) | VERIFIED | TensorRT 11.3 support-matrix data: Jetson AGX Thor and DRIVE AGX Thor, compute capability 11.0, list FP8, FP4 and BF16 as Supported; DLA N/A | DRIVE Thor needs DriveOS 7.2.6+ for TensorRT 11.x; no DLA support in TRT 11.3 | https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html | 2026-10-03 |
| TensorRT-Edge-LLM platform and precision matrix | VERIFIED | Platforms: Jetson Thor (JetPack 7.0–7.2), NVIDIA DRIVE Thor (DriveOS 7.2, CUDA 13.3). Precision notes: ModelOpt FP8/MXFP8/FP4/NVFP4/INT4-AWQ/INT8-SQ checkpoints | FP4/NVFP4 need Blackwell-class hardware; Orin cannot run FP8 or FP4 engines. Release 0.11.0 is dated 2026-09 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/support-matrix.md ; .../supported-models.md | 2026-10-03 |
| VLA support and CUDA graphs for iterative denoising | VERIFIED | Edge-LLM 0.11.0 adds experimental pi0.5: FP16, four TensorRT components, `--steps` (denoise count) and `--cudaGraph` flags | CUDA-graph denoising loops are also used in PhyAI, Jetson-PI (graph reuse, flow unrolling), FlashDrive (per-stage CUDA graphs) and realtime-vla | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/examples/vla/pi05.md | 2026-10-03 |
| GR00T TensorRT precision | VERIFIED | build_trt_pipeline `--precision` default bf16, bf16 only; static `--batch-size` 1 | — | GR00T deployment README | 2026-10-03 |
| ModelOpt quantization of flow-matching / diffusion experts | VERIFIED (recipe level); user-reported runtime gap | Recipe: FP8, NVFP4+FP8 AutoQuant, nvidia-modelopt==0.43.0, tested on 5090 and B300; eval reports open-loop minADE only. ModelOpt README shows diffusers PTQ/TensorRT work | Forum users (not NVIDIA) report quantized checkpoints running fake-quant with no real-quant GEMM on Thor, i.e. no speedup, as of 2026-09-15 | quant recipe README; https://github.com/NVIDIA/Model-Optimizer ; forum 382409 | 2026-10-03 |
| "NVIDIA's PTQ recipe is open-loop only" (02 §0) | VERIFIED | Recipe eval.py reports per-clip and average minADE; no closed-loop step | No accuracy or latency numbers published | quant recipe README | 2026-10-03 |
| DriveOS 7 stack for E2E/VLA on DRIVE AGX Thor | VERIFIED | DevKit blog paraphrase: DriveOS 7 includes TensorRT 10, CUDA 13 and a C++ DriveOS LLM SDK optimized for FP16/FP8/INT4/NVFP4. Linux profiles enable ASIL-D/ISO 21434 compliance | Listed models: Llama 3.x, Qwen2/2.5, Qwen2-VL. No Alpamayo or VLA deployment procedure is published | https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ | 2026-10-03 |
| Safety-certified inference constraints | VERIFIED (DriveOS 6 docs) | TensorRT 8.6.13 for DRIVE OS 6.0.10: safety runtime only on QNX safety, only kSAFETY engines; proxy runtime is uncertified | DriveOS 7 equivalents not found | https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-tensorrt/release-notes/index.html | 2026-10-03 |
| Halos and certification status | VERIFIED | Halos Core is built on ISO 26262 ASIL D certified DriveOS, with a hypervisor separating safety functions from AI and safety-certified CUDA/TensorRT support | Also: DriveOS 6.0 certified ASIL D conformant; Thor-X SoC assessed ASIL D conformant | https://www.nvidia.com/en-us/ai-trust-center/halos/autonomous-vehicles/ | 2026-10-03 |
| Deterministic memory / carveouts on DriveOS (constraint) | VERIFIED (forum) | User reports cudaMemGetInfo showing 6.0 GiB total GPU memory on a 58 GB DriveOS 7.2.5 EA board. NVIDIA moderator points to hugepages and carveout customization docs | GPU-visible memory on DRIVE Thor is configured by carveout, not the full 64 GB by default | forum 382501 | 2026-10-03 |

### Inferences
- **FP8/NVFP4 expert claims for Thor.** Any FP8/NVFP4 claim on Thor for the Alpamayo expert would be new ground for G6. The general TensorRT stack supports these formats, but the only NVIDIA VLA path (Edge-LLM) is FP16 for Alpamayo, and the ModelOpt recipe runs fake-quant on Thor per user reports.
- **Reporting Thor numbers.** Claims of "FP8 for Thor" in the docs should be reported as an emulated or datacenter-GPU measurement unless a real Thor engine is built.
- **C++ guardrail (G4) vs certification.** A G4 written in C++ matches the DriveOS deployment idiom (C++ runtimes, QNX safety). Certification (kSAFETY engines, ASIL) is outside what a solo project can claim, consistent with the docs' "must not" list.

### Gaps
- No public DriveOS 7 safety-runtime layer list or kSAFETY constraints for Thor.
- No NVIDIA-published Alpamayo latency on DRIVE or Jetson Thor.
- No NVIDIA statement on quantizing the flow-matching expert specifically.
- The ModelOpt issue #1895 cited in the forum could not be retrieved (GitHub API rate limit).

## Q5. Rented-compute prices (Sep–Oct 2026) vs the docs' ranges and budget

### Takeaway
The docs' price bands are stale or too narrow. H100 on-demand now spans $1.60 (thin vast.ai supply) to $4.50:
- Nebius raised H100 to $4.50 effective 2026-10-01.
- Lambda charges $3.99–4.29.
- RunPod charges $2.69 (community) / $3.49 (secure).

Other corrections:
- The B200 floor of $3.5 is not found on any on-demand listing; the lowest is $5.98.
- RTX PRO 6000 at $1.4–2.2 is about right.
- $150/month storage matches only RunPod network storage above 1 TB.
- The Jetson AGX Thor dev kit is **$5,499 (official MSRP)**, up from its $3,499 launch price.

The 12-week budget recomputed from the rows below spans roughly **$3.7k to $15k** depending on provider mix:

| Provider mix | Total (×1.3 contingency) |
|---|---|
| All vast.ai medians, bid-priced distillation | ≈$3.7k |
| vast.ai on-demand throughout | ≈$6.0k |
| RunPod secure for small runs + vast.ai 8×H100 on-demand for distillation | ≈$7.8k |
| Lambda/Nebius on-demand | ≈$13–15k |

The doc's $7.5–8.5k holds only for a specific mixed choice, with distillation never on Lambda/Nebius on-demand. Distillation alone roughly doubles at those rates ($6.4–7.2k vs $3.4k).

### Cited Findings

#### Price snapshot (USD per GPU-hour unless noted)

| GPU | vast.ai (snapshot 2026-10-03 05:04 UTC, unrented offers) | RunPod (page "Updated September 27, 2026"; API 2026-10-03) | Lambda (2026-10-03) | Nebius (2026-10-03) |
|---|---|---|---|---|
| H100 SXM, on-demand | n=5: min 1.60, **median 1.69**, max 3.07 | community **2.69** / secure **3.49** | 8×: **3.99**; 4×: 4.09; 2×: 4.19; 1×: 4.29 | **4.50** effective 2026-10-01 (was 3.85) |
| H100, interruptible / spot | n=3 min bids: 0.60–2.67, median 1.00 | API spot fields equal on-demand; no distinct spot price exposed | none listed | preemptible **from 0.79** (dynamic; capped 1¢ below on-demand) |
| 8×H100 node | 1 offer: 1.68 per GPU = **$13.4/h**; min bid 0.60 per GPU | clusters: H100 SXM "Contact sales" | **$31.92/h** (8 × 3.99) | **$36.00/h** (8 × 4.50) |
| RTX PRO 6000 | WS n=35 median **1.40** (0.92–2.34); Server n=23 median **1.47** (0.87–1.67) | community 1.69 / secure **2.09**; "WK" (workstation) secure 2.19 | not offered | **1.80** (unchanged) |
| RTX PRO 6000, interruptible | WS median 0.93; Server median 0.81 | — | — | preemptible from 0.79 |
| B200 | n=5: min 6.25, median **7.51**, max 9.39; min bid 5.94 | community **5.98** / secure **6.79** | 8×: 6.69 … 1×: 6.99 | **8.50** effective 2026-10-01 (was 7.15); preemptible from 0.99 |
| H200 | n=3 median 5.00 | community 3.59 / secure 4.59 | — | 5.40 effective 2026-10-01 (was 4.50) |
| RTX 5090 | n=64 median **0.54** (min 0.21); bid median 0.40 | community 0.69 / secure 0.99 | — | — |
| Storage | per offer (not captured) | network std <1 TB $0.07/GB-mo, **>1 TB $0.05/GB-mo**; high-performance $0.14; volume $0.10 running / $0.20 idle | not shown on pricing page | shared FS $0.08/GiB-mo; object std $0.0147/GiB-mo; block $0.053–0.118/GiB-mo |

Sources:
- vast.ai: public search endpoint https://cloud.vast.ai/api/v0/bundles/ (also https://vast.ai/pricing). `dph_total` includes a small disk charge.
- RunPod: https://www.runpod.io/pricing ; GraphQL gpuTypes at https://api.runpod.io/graphql
- Lambda: https://lambda.ai/pricing
- Nebius: https://nebius.com/prices

#### Ledger: price claims

| Claim as written (doc § ) | Status | Evidence | Correct value | Source | Access |
|---|---|---|---|---|---|
| H100 ≈ $2.2–3.4/h (02 §4) | CORRECTED | Snapshot above | $1.60–4.50 on-demand. Brackets neither Lambda ($3.99–4.29) nor Nebius ($4.50) nor the vast.ai median ($1.69) | as above | 2026-10-03 |
| H100 spot ≈ $0.9–2.2 (02 §4) | CORRECTED (floor) | vast.ai min bids 0.60–2.67; Nebius preemptible from $0.79 | Floor ≈$0.60–0.79; availability thin (n=3 on vast.ai) | vast.ai; Nebius | 2026-10-03 |
| RTX PRO 6000 ≈ $1.5–2.2/h (02 §4) | VERIFIED (approx.) | Medians 1.40–1.47 (vast.ai); 1.69–2.19 (RunPod); 1.80 (Nebius) | Marketplace median is slightly below $1.5 | as above | 2026-10-03 |
| B200 ≈ $3.5–8/h (02 §4) | CORRECTED | Lowest on-demand: RunPod community 5.98; Nebius 8.50 from Oct 1 | $5.98–9.39 on-demand. Below $5.98 only as preemptible/interruptible | as above | 2026-10-03 |
| Storage $150/month for NuRec ~2.9 TB + PAI subsets (02 §4) | VERIFIED (one provider) | RunPod network storage >1 TB: $0.05/GB/mo → 2,900 × 0.05 = $145 | Nebius shared FS ≈$216/mo (2,701 GiB × 0.08); Nebius object ≈$40/mo; RunPod high-performance ≈$406/mo | RunPod; Nebius | 2026-10-03 |
| Thor dev kit $5,499 (02 §4) | VERIFIED | NVIDIA Jetson FAQ MSRP: "Jetson AGX Thor Developer Kit $5499"; T5000 $4999 (1KU+); T4000 $2999 (1KU+) | Launch price was $3,499 (Aug 2025 blog). The increase was visible ~2026-07-21 per CNX Software | https://developer.nvidia.com/embedded/faq ; https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/ | 2026-10-03 |

#### Budget recomputation (02 §4) with verified prices

| Item | Doc | Implied rate | Recomputed at verified prices |
|---|---|---|---|
| Gate runs, ~450 GPU-h | $900–1,300 | $2.0–2.9/GPU-h | Assuming a 50/50 split, which the doc does not state: vast.ai 225 × 1.45 + 225 × 1.69 = **$707**; RunPod secure 225 × 2.09 + 225 × 3.49 = **$1,256**; Nebius 225 × 1.80 + 225 × 4.50 = **$1,418** |
| Policy zoo, ~300 H100-h | $700–1,000 | $2.33–3.33/GPU-h | vast.ai $507; RunPod secure $1,047; Lambda 1× $1,287; Nebius $1,350 |
| Distillation, 8×H100 spot ~200 h = 1,600 GPU-h | $3,400 | $2.125/GPU-h | vast.ai bid floor 0.60 → **$960** (one 8× offer). Nebius preemptible floor 0.79 → **$1,264** if capacity holds. On-demand: vast.ai 8× 1.68 → $2,688; RunPod 3.49 → $5,584; Lambda 3.99 → **$6,384**; Nebius 4.50 → **$7,200** |
| Precision + profiling, ~150 H100-h | $350–500 | $2.33–3.33/GPU-h | $254 (vast.ai median) to $675 (Nebius) |
| Contingency and total | 30%; ≈ $7,500–8,500 | — | Doc arithmetic checks: (5,800–6,650 incl. $450 of storage) × 1.3 = $7.5–8.6k |

At Lambda/Nebius on-demand rates throughout, the total rises to ≈$13–15k:
- Lambda H100 + Nebius RTX PRO 6000: (1,370 + 1,287 + 6,384 + 644 + 450 storage) × 1.3 ≈ $13.2k.
- All Nebius: (1,418 + 1,350 + 7,200 + 675 + 648 storage) × 1.3 ≈ $14.7k.

### Inferences
- **Supply and price direction.** vast.ai H100 SXM supply was thin at snapshot time (5 rentable single-GPU offers, 1 eight-GPU offer). Budgets that assume spot H100 should carry a fallback rate. Datacenter-cloud prices rose on 2026-10-01 at Nebius (H100 +17%, H200 +20%, B200 +19%); the direction is up.
- **Cost-efficient GPU choice.** For the gate (rendering plus inference), RTX PRO 6000 at ~$1.4–1.8/h is the most cost-efficient verified option. It is also the GPU of record for FlashDrive, GR00T and NuRec numbers, which simplifies comparability.

### Gaps
- vast.ai and Lambda storage prices were not captured. Lambda's pricing page shows no storage line.
- RunPod spot/interruptible discounts: none visible in the API fields.
- All marketplace prices are point-in-time.

## Q6. Free or discounted compute a solo independent researcher can obtain in 2026

### Takeaway
None of the major GPU programs is open to an unaffiliated individual without a company or an academic appointment:
- NVIDIA Inception requires an incorporated company.
- The NVIDIA Academic Grant Program requires full-time faculty.
- Nebius research credits require an academic or non-profit affiliation.
- Nebius startup credits require ≥$5M of VC funding.

Google TRC accepts anyone but provides TPUs, which do not run the CUDA/TensorRT/NuRec stack. Hugging Face ZeroGPU and community grants cover Spaces demos only.

Realistic options are (a) incorporating to join Inception and use its partner credits, (b) partnering with an academic collaborator, or (c) paying marketplace or spot prices.

### Cited Findings

| Program | Eligibility (paraphrase of primary text) | What it gives | Fit for PhysLoop (AlpaSim/NuRec/TensorRT) | Source | Access |
|---|---|---|---|---|---|
| NVIDIA Inception | Must employ ≥1 developer, keep a working website, be officially incorporated and be under 10 years old. Unincorporated applicants cannot apply. No fees or deadlines | Partner cloud credits, preferred hardware/software pricing, training | Only after incorporating | https://www.nvidia.com/en-us/startups/ | 2026-10-03 |
| NVIDIA Academic Grant Program | Applicant must be full-time faculty at an accredited institution that awards PhDs; competitive; proposal template | Up to 30,000 H100 80 GB hours (AI training track), or up to eight RTX PRO 6000 GPUs (simulation track) | Not eligible as an independent; possible via a faculty co-PI | https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/ | 2026-10-03 |
| Google TPU Research Cloud | Anyone can express interest; rolling invitations; must publish TRC-supported work and give feedback | Free Cloud TPU quota for a limited period; other GCP services billed | TPUs only; unsuited to CUDA/TensorRT/NuRec/Thor work | https://sites.research.google/trc/faq/ | 2026-10-03 |
| Nebius Research Credits | Postgraduate or PhD students, postdocs, faculty, or researchers at accredited academic or non-profit research organizations; one applicant per project; one application per project | GPU credits, Token Factory tokens, Tavily credits (amount set per proposal). 2026–27 cycle reviewed through summer 2026 | Not eligible as an unaffiliated individual | https://nebius.com/nebius-research-grants | 2026-10-03 |
| Nebius for Startups | Credits only via partner VCs; ≥$5M raised from an approved partner; incorporated <5 years; company website and email | Credits and discounts | Not eligible | https://nebius.com/startups | 2026-10-03 |
| Lambda research grants | "Apply for a grant" links to a Typeform; eligibility not published on the page | Unstated | UNVERIFIABLE | https://lambda.ai/research | 2026-10-03 |
| Hugging Face community GPU grants / ZeroGPU | Community GPU grants are requested per Space (application in Space settings). ZeroGPU is free to all users with a daily quota (PRO: 8× quota) | Dynamically allocated RTX Pro 6000 Blackwell GPUs for Spaces | Demos only; not batch simulation or training | https://huggingface.co/docs/hub/spaces-gpus ; https://huggingface.co/docs/hub/spaces-zerogpu | 2026-10-03 |

### Inferences
- **Fastest path to subsidized NVIDIA compute.** Incorporate a minimal company, then apply to Inception. Inception's credits come through partner clouds and amounts are not published.
- **Larger grants.** The larger grants (Academic Grant's 30,000 H100-hours; Nebius research credits) require an academic co-PI.
- **Two blockers, one fix.** The AlpaSim challenge also requires an organization email (Q2). An academic or company affiliation therefore unlocks both the challenge registration and the grants.
- **No-affiliation baseline.** Without an affiliation, the realistic floor is marketplace spot: H100 at ~$0.60–1.00/h on vast.ai bids, RTX PRO 6000 at ~$0.8–0.9/h, and Nebius preemptible from $0.79/h, accepting interruption risk.

### Gaps
- Inception credit amounts and the partner list are not stated on the public page.
- Lambda grant eligibility is unpublished.
- Whether NVIDIA Academic Grant awards can include an independent co-investigator was not found.
