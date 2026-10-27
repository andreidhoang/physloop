# Verification ledger, 2026-10-25

**Status:** evidence record. It checks the claims of `decision_input_2026-10-25.md` against primary sources and reports what changed since the 2026-10-03 verification (`../2026-10-03-verification/`). Live docs win where they differ.

**Method.** Four read-only research lanes ran on 2026-10-25:
- (A) models, announcements and licenses;
- (B) open tooling and Thor serving;
- (C) hardware, toolchain and prices;
- (D) landscape, legal, challenge and roles.

No license was accepted, no dataset was downloaded, and nothing was posted. Quotes are under 15 words.

**Raw evidence in `notes/`:**
- lane A's and lane D's full ledgers (`lane_A_models_licenses.md`, `lane_D_landscape_legal.md`);
- the raw vast.ai offer JSON with its stats script (`vast_2026-10-25/`; `python3 -I stats.py .` reproduces the price table);
- the arithmetic check.

Lanes B and C reported inline; their findings are in §B and §C below. PR #42 was re-fetched directly: training code only, no weights. Verdicts: VERIFIED, PARTLY, CONTRADICTED, UNVERIFIED. The arithmetic is in `review.md` §3 and `notes/arith_check.py`.

---

## 0. The five facts that decide the plan

1. **NVIDIA's distillation recipe is public, but not merged.**
   - alpamayo-recipes PR #42 (opened 2026-10-20) distills Alpamayo 1 / 1.5 into a ~2B student. The student is `nvidia/Cosmos-Reason2-2B` plus a compact action expert, followed by expert SFT and DRaFT RL on the expert.
   - It recommends at least 128 H100s and ≥ 100 TB of storage.
   - It reports open-loop numbers only: full-PAI-val minADE6 0.8875 m for the R1.5-2B student vs 0.8298 m for the teacher (+6.95%).
   - The decision doc's "the distillation script is not released" is out of date, and its planned student is the same student.
2. **Alpamayo 2 Edge is announced, not released.**
   - Announced 2026-10-21 at GTC Berlin: 2B, built on Cosmos, "optimized for the NVIDIA DRIVE AGX Thor platform", "designed to run at 10Hz".
   - The 10-21 blog says it "will be available for download … later this month".
   - On 2026-10-25 every guessed Hugging Face name returns 401, and no NVlabs repo exists. **Re-check on 2026-10-31.**
3. **NVIDIA's own 2B students miss 10 Hz on DRIVE AGX Thor.**
   - PR #45 (open, 10-21) runs them through a forked TensorRT-Edge-LLM: median 171.2 ms in FP8 (~5.8 Hz) and 220.6 ms in FP16.
   - These figures and the DRIVE OS versions come from the README on the PR's branch, read by lane A. The PR description itself says only "~5Hz on Thor", without naming the device. Neither the PR nor the branch links weights.
   - This independently confirms `review.md` §3.1: the reasoning student sits above 100 ms.
4. **AlpaGym cannot train reasoning.**
   - AlpaGym pins alpamayo-recipes `1d99fc50`, the head of the still-unmerged PR #18. That commit hard-sets `cotrain_vlm = False` and freezes LM and vision parameters.
   - The default rollout generates no reasoning text. Shipped rewards are `distance_to_gt`, `progress_safety` and `metrics` only.
   - A reasoning or reasoning–action-consistency reward exists only in the **open-loop** `alpamayo1_x_rl` recipe. Its reasoning-reward sign was inverted until the 2026-10-06 fix (#40, issue #29), so earlier RL results from that recipe are suspect.
5. **The gate's niche is still open.**
   - Neither AlpaSim (`main` or `e2e_challenge`) nor any NVIDIA repo provides a paired two-checkpoint verdict, stratification, seed test–retest or a monitor implementation.
   - NVIDIA's distillation (#42) and Thor (#45) PRs measure open-loop error and latency only.
   - *Did distillation make the deployable model worse in closed loop, and where?* is unanswered for NVIDIA's own students.

## A. Models, announcements, licenses

| ID | Claim (decision doc) | Verdict | Evidence | Source |
|---|---|---|---|---|
| A1 | Alpamayo 2 Edge announced 10-21; no weights or card | VERIFIED | 2B, built on Cosmos, DRIVE AGX Thor, 10 Hz. No license is stated. HF 401 on all Edge names; HF API lists 3 nvidia Alpamayo models, newest created 05-27 | [Berlin release](https://nvidianews.nvidia.com/news/european-mobility-leaders-build-safe-vehicles-and-robotaxi-fleets-on-nvidia-hyperion); [blog 10-21](https://blogs.nvidia.com/blog/open-models-europe-industry/); [HF API](https://huggingface.co/api/models?author=nvidia&search=alpamayo) |
| A1d | Alpamayo Stack = VLA + Halos classical safety stack | VERIFIED (paraphrase) | Layers: reasoning model, policy evaluator and safety monitors, Halos OS, Halos SAFE | [Alpamayo page](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/) |
| A2 | Production students are distilled; script not released | PARTLY / out of date | NVIDIA says the teacher "can be distilled into compact models that run on NVIDIA DRIVE AGX Thor". No source says Edge itself is distilled from a released teacher. PR #42 is public (open) | [A2S blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/); [PR #42](https://github.com/NVlabs/alpamayo-recipes/pull/42) |
| A3 | Weights OpenMDW-1.1 but cards say non-commercial | PARTLY | The R1 and 1.5 cards carry both the "non-commercial use" line and OpenMDW-1.1; their YAML field and LICENSE file are OpenMDW-1.1. The Super card has no non-commercial line. A NVIDIA blog says distilled models can be deployed commercially without further permission. Model-weight terms are separate from the PAI-AV/NuRec **data** license | [1.5 card](https://huggingface.co/nvidia/Alpamayo-1.5-10B); [R1 card](https://huggingface.co/nvidia/Alpamayo-R1-10B) |
| A4 | Card AlpaSim score 1.37 vs 0.81 | VERIFIED, unchanged since 08-07 | 1.37 ± 0.10 on 913 scenarios; the card defines neither the score nor "±". The AR1 paper defines its AlpaSim score as average km driven between events (> 4 m excluded). The challenge `scene_score` is a different metric in [0, 1]. Equating the card number with the paper metric is an inference | [1.5 card](https://huggingface.co/nvidia/Alpamayo-1.5-10B); [AR1 v2](https://arxiv.org/abs/2511.00088) |
| A5 | "Production RL for 1.5 used 640 GPUs" | VERIFIED (scope) | "512 policy GPUs and 128 rollout GPUs" for post-training 1.5; this is **open-loop** GRPO (global batch 2,560) | [recipes RL README](https://github.com/NVlabs/alpamayo-recipes/blob/main/recipes/alpamayo1_x_rl/README.md) |
| A6 | AR1 0.5B GRPO: grade 3.1 → 4.4, consistency 0.62 → 0.83, close-encounter 6.9% → 3.7% | VERIFIED | Table 9, full rewards; that row is not the best on grade (4.5) or consistency (0.85). CoC volume 700K / 3M / ~3.7M VERIFIED; ~10% human-verified VERIFIED; GPT-5 auto-labels VERIFIED; Qwen3-VL is a teacher for backbone adaptation, not CoC. 160 tokens/image is the 448×280 setting | [AR1 v2](https://arxiv.org/abs/2511.00088) |
| A7 | AlpaJudge exists | VERIFIED (new) | HF repo created 09-02; OpenMDW-1.1 weights, Apache-2.0 code. RB mode is a DeBERTa-v2 Lingo-Judge (~0.74 GB) that scores a candidate against a reference. RF mode is a Qwen3-VL-2B that outputs five rubric probabilities (~4.9 GB). Inference only; referenced nowhere in AlpaGym or the recipes | [HF](https://huggingface.co/nvidia/alpajudge); [GitHub](https://github.com/NVlabs/alpajudge) |
| A8 | PAI-AV: 1,700 h, 306K clips, 25 countries, 133 TB | VERIFIED | **Vietnam is not included**: the US plus 24 European countries. NuRec "1500+ scenes", ≈ 2.9 TB total (≈ 1.9 GB per scene, not 1.5). License §1/§3/§4.6/§8 unchanged in substance. **v26.10 (10-23)** adds `splits/alpamayo_aligned.parquet`, a split aligned with Alpamayo training data, which matters for contamination | [PAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles); [NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) |
| A9 | Blog reward: progress +1, collision −10, off-road −5, −0.01 × dist | VERIFIED | The collision term is `collision_any`, not at-fault | [closed-loop blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/) |
| A10 | Uber L4 LA/SF H1 2027; Bolt/Lucid 25K+; Hyperion 10 = 14 cameras, 9 radars, 1 lidar | VERIFIED / VERIFIED / PARTLY | NVIDIA's own pages conflict on lidar count (1 in Oct 2025, 3 in a 2026-09 blog) and on single vs dual Thor | [Uber](https://investor.uber.com/news-events/news/press-release-details/2026/NVIDIA-to-Launch-L4-Software-Driven-Robotaxis-on-Uber-Across-28-Cities-by-2028/default.aspx); [2026-09 blog](https://blogs.nvidia.com/blog/robotaxi-leaders-full-stack-open-platform/) |

## B. Open tooling and Thor serving

| ID | Claim | Verdict | Evidence | Source |
|---|---|---|---|---|
| B1 | AlpaGym: 13 commits, 1.5 only, GRPO on Cosmos-RL, ≥ 2 × 40 GB | VERIFIED | Last code change 2026-07-08, last commit 07-21 (docs). 0 external PRs, 0 issues ever. PR #18 still open (last updated 06-30). **VLM frozen** (see §0.4) | [alpagym](https://github.com/NVlabs/alpagym); [PR #18](https://github.com/NVlabs/alpamayo-recipes/pull/18) |
| B2 | Recipes: SFT ZeRO-2 8×H100; open-loop GRPO ≥ 5 × 80 GB; ModelOpt quant | VERIFIED / PARTLY | GRPO VERIFIED. SFT: Alpamayo 1 validated on 8 × H100 with ZeRO-2; 1.5 ships `zero2.json` without naming a GPU; Alpamayo 2 SFT uses ZeRO-3 on 4 × 8 H100. New open PRs: #42 distillation, #43 speculative-decoding drafts, #45 DRIVE Thor inference | [recipes](https://github.com/NVlabs/alpamayo-recipes) |
| B3 | AlpaSim: gRPC, metrics `collision_at_fault`, `offroad`, `dist_to_gt`; TransFuser baseline | PARTLY / VERIFIED | `collision_at_fault` is a derived event (front or lateral); scorers emit `dist_to_gt_trajectory` and `dist_to_gt_location` (no bare `dist_to_gt`). TransFuser driver at `plugins/transfuser_driver/`. **Deltas:** `main` not synced since 2026-08-11. No external merge since #149 (08-17), so the "0–6 days" baseline is out of date. `e2e_challenge` corridor scorer fixed mid-challenge (#215, 10-20). Issue #207: one SHA resolved ~25 packages differently because `uv.lock` is gitignored. Issue #208: `n_sim_steps` silently clipped and not recorded. Issue #224: NVIDIA says some simulation choices affecting inference are withheld until after the challenge | [alpasim](https://github.com/NVlabs/alpasim) |
| B4 | FlashRT: 345 ms (best Jetson Thor result); official BF16 3.3 s | VERIFIED with corrections | PR #229, merged 10-20, Alpamayo 1.5, Jetson AGX Thor (MAXN). Full request, official weights, 10 flow steps: official BF16 **3,322 ms**, FlashRT bf16 2,079, "balanced" **897**, "fast" **811** ms. **345 ms** mean (295 median) is per window under the FlashDrive *streaming* procedure, with z-lab fine-tuned PARO/DFlash weights and 8 steps. It is not comparable with the official-weights numbers. Reasoning averaged 16–19 tokens | [FlashRT PR #229](https://github.com/flashrt-project/FlashRT/pull/229) |
| B5 | TensorRT-Edge-LLM supports 1.5? | No (baseline holds) | Issue #130 open; v0.11.0 (09-29) does not mention Alpamayo; the exporter recognises only `alpamayo_r1`, FP16. External PR #260 (10-21) adds an SM110 Alpamayo plugin | [issue #130](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130) |
| B6 | Forum: "Alpamayo not available for AGX Thor" (09-07/08); "empty CoC on DRIVE Thor" | VERIFIED / PARTLY | Staff, 09-07: "Alpamayo is not available for AGX Thor currently". Thread 382501 covers **Alpamayo 1** on DRIVE AGX Thor: an OOM at a 6 GiB carveout, then empty `output_text` on 09-16. It was fixed by 10-06 (carveout raised 6 → 32 GB, missing tokenizer files, misaligned input). **There is no open empty-CoC bug to fix upstream.** NVFP4 455 s vs FP16 424 s is a per-clip time from a community post | [382409](https://forums.developer.nvidia.com/t/how-to-perform-alpamayo1-5-fp8-quantization-for-nvidia-thor/382409); [382501](https://forums.developer.nvidia.com/t/drive-agx-thor-with-alpamayo/382501) |
| B8 | NIM on vLLM-Omni, one request at a time | PARTLY | True for the Alpamayo 2 Super NIM (released 10-20). The 1.5 NIM uses PyTriton + TensorRT-LLM. Neither runs on Jetson | [NIM 2.0.0](https://docs.nvidia.com/nim/alpamayo/2.0.0/overview.md) |
| B9 | CoC typically 11–20 tokens | VERIFIED (nuance) | Alpamayo 1 on DGX Spark; "between 11 and 20" tokens | [arXiv 2605.08975](https://arxiv.org/abs/2605.08975) |
| B10 | Curvature diverges near v ≈ 0 (issue #113) | VERIFIED | max\|κ\| ≈ 3,000 reported; closed 08-05: curvature zeroed below a speed threshold and clipped | [issue #113](https://github.com/NVlabs/alpamayo/issues/113) |

## C. Hardware, toolchain, prices

| ID | Claim | Verdict | Evidence | Source |
|---|---|---|---|---|
| C1 | Jetson AGX Thor dev kit $5,499, up from $3,499 | VERIFIED (MSRP) | NVIDIA FAQ lists $5,499. Newegg returned 403. SabrePC lists $6,268.86, out of stock | [Jetson FAQ](https://developer.nvidia.com/embedded/faq); [CNX](https://www.cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101/) |
| C2 | 128 GB at 273 GB/s; 517 dense FP8 TFLOPS; 40–130 W; 20 cameras via HSB | VERIFIED | **517 dense FP8 is an official NVIDIA figure** (blog Table 1), not a derivation | [Jetson Thor blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/) |
| C3 | JetPack 7.2.1, CUDA 13.2, TensorRT 10.16 | VERIFIED | CUDA 13.2.2, TensorRT 10.16.2. **TensorRT 11.x (current 11.4.0) does not support Jetson Thor**; it covers DRIVE AGX Thor on DriveOS 7.2.6+. The TensorRT 10.16 matrix lists FP8 and FP4 as *supported* on Jetson AGX Thor (sm_110). **Correction to `04` §3:** "TensorRT 11.3 lists FP8/FP4 on Thor" is true of DRIVE AGX Thor only. The "fake quantization" reports concern ModelOpt PyTorch checkpoints, not TensorRT engines | [JetPack 7.2.1](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1); [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) |
| C4 | DRIVE AGX Thor: 16 GMSL2 + 2 GMSL3, 4 CAN, 350 W, DriveOS 7.x EA, distributors only | VERIFIED / PARTLY | DriveOS 7.2.x is early access (7.2.5, 08-06); the kit went GA on 7.0.3. Price and lead time UNVERIFIED | [DRIVE AGX](https://developer.nvidia.com/drive/agx) |
| C5 | Prices | VERIFIED except B300 | See below | vast.ai API, 2026-10-25 07:19 UTC; [Lambda](https://lambda.ai/pricing); [RunPod](https://www.runpod.io/pricing) |
| C6 | Rig: IMX728 GMSL2 ~$459; ZED-X20P ~$300; comma four $999 | PARTLY | $459 VERIFIED, but Leopard's Thor-listed series is IMX728-**9295** ($519 for 120°, $539 for 30°). ZED-X20P $299.95 VERIFIED. comma four $999 VERIFIED | [Leopard](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/adi-gmsl2-cameras/li-imx728-gm2b/li-imx728-gm2b-120h-sf-aa/); [SparkFun](https://www.sparkfun.com/sparkfun-allband-gnss-rtk-breakout-zed-x20p-qwiic.html); [comma](https://blog.comma.ai/comma-four/) |
| C7 | Dev kit takes GMSL cameras | **CONTRADICTED** | The dev kit has no native GMSL. Its camera inputs are "HSB camera via QSFP slot" and USB. GMSL needs a Holoscan Sensor Bridge board (Lattice, $372.50, 0 in stock) plus a deserializer (LI-GMSL2-DESER-HOLOSCAN, $388, "supports 1x" camera). Cameras per HSB are UNVERIFIED | [Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/); [Leopard deser](https://leopardimaging.com/) |

**Live on-demand prices, $/GPU-hour (vast.ai, 2026-10-25; "median" per offer):**

| GPU | Offers | Cheapest | Median | Interruptible floor (median of min bids) |
|---|---|---|---|---|
| RTX PRO 6000 Workstation | 49 | 1.00 | 1.60 | 0.93 |
| RTX PRO 6000 Server | 47 | 1.24 | 1.47 | 0.80 |
| H100 SXM | 17 | 1.93 | 2.40 (verified hosts 3.19) | 2.40 |
| B200 | 9 | 7.50 | 9.38 | 7.50 |
| B300 | 5 | **9.01** (doc said 5.90: CONTRADICTED) | 9.01 | 8.50 |

Other providers:
- **Lambda H100 SXM:** $3.99 (8×) to $4.29 (1×).
- **RunPod H100:** $2.69 community / $3.99 secure.
- **RunPod RTX PRO 6000:** $1.69 / $2.49.
- **RunPod B300:** $6.94 / $8.99.

The vast.ai H100 median rose from $1.69 (10-03) to $2.40.

## D. Landscape, legal, challenge, roles

P = primary source, S = secondary.

| ID | Claim | Verdict | Evidence | Source |
|---|---|---|---|---|
| D1 | Waymo ~400k paid rides/week, 1M by end-2026; 6th gen 13 cameras / 4 lidar / 6 radar under $20k; teacher → student, also simulator and critic | PARTLY / VERIFIED / PARTLY / VERIFIED | P: "more than 400,000 rides" per week (Feb 2026; "paid" not stated). A later report gives 500k paid rides/week (TechCrunch, 03-27). The 1M target is VERIFIED. "Under $20k" is Electrek's estimate, not Waymo's. Teacher models for Driver, Simulator and Critic are distilled into students: VERIFIED | [Waymo Feb 2026](https://waymo.com/blog/2026/02/waymo-raises-usd16-billion-investment-round/); [6th gen](https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/); [foundation model](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving) |
| D2 | Tesla robotaxi in 6 cities on FSD v15 Unsupervised; consumers on v14.3 | PARTLY (secondary only) | Tesla Oracle lists 6 cities on "early versions of FSD v15". Electrek says the Bay Area service is not unsupervised. Tesla's Q3 2026 update gives no city count or version | [Tesla Oracle](https://www.teslaoracle.com/2026/10/23/tesla-announces-robotaxi-launch-in-four-more-cities-releases-safety-data-via-new-website-no-kittens-were-hurt/); [Tesla Q3 2026](https://assets-ir.tesla.com/tesla-contents/IR/TSLA-Q3-2026-Update.pdf) |
| D3 | XPeng VLA 2.0: language is a bottleneck; VW licenses it from 2027; ~$500M/yr | VERIFIED with nuances | Language as an intermediate output is "a redundancy or a bottleneck". XPeng says VW is the "first customer" with deliveries planned for 2027, but never says "license". $500M/yr is Electrek's extrapolation from RMB 300M/month | [Electrek](https://electrek.co/2026/06/05/xpeng-vla-interview-cvpr-language-poison-tesla-fsd/) |
| D4 | GAIA-3 is a 15B evaluation world model; Nissan ProPILOT Japan FY2027 | VERIFIED | The doc cites the GAIA-3 page for Nissan; the Nissan page is the source | [GAIA-3](https://wayve.ai/press/wayve-launches-gaia3/); [Nissan](https://global.nissannews.com/en/releases/next-gen-propilot-technology) |
| D5 | DriveVLA-W0 86.1 EPDMS, autoregressive beats flow at 70M frames; DriveZero 95.3 PDMS "above human"; Poutine 7.99 RFS; RAD ~3× lower collisions | VERIFIED (DriveZero with caveats) | DriveVLA-W0: flow wins at NAVSIM scale (87.2 vs 85.3); the order inverts at 70M frames. DriveZero is +0.5 over a Human row that uses ground-truth inputs, and needs 237K extra scenes. ReconDreamer-RL reports 5×, not 3× | [W0](https://arxiv.org/abs/2510.12796); [DriveZero](https://arxiv.org/abs/2609.06055); [Poutine](https://arxiv.org/abs/2506.11234); [RAD](https://arxiv.org/abs/2502.13144) |
| D6 | Challenge: registration closed 10-18; PAI-AV entries due 10-31; 16 GiB; 10 Hz | VERIFIED | 16 GiB is a hard rule. 0.1 s is called a target, but a per-track throughput budget enforces it. The nuPlan track is due 11-16 | [challenge README](https://github.com/NVlabs/alpasim/blob/e2e_challenge/e2e_challenge/README.md) |
| D7 | Vietnam's 2024 Road Traffic Law Art. 34 requires a permit for "smart vehicles" | PARTLY (wrong article; definition changed) | See the note below | [consolidated text VBHN 55](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/3/55-vbhn-vpqh.pdf) |
| D7b | HCMC SHTP and QTSC sandboxes are enterprise-only; Hanoi Phenikaa-X 24 months | PARTLY / VERIFIED | SHTP: "Eligible candidates are enterprises". QTSC's page is silent on eligibility. Phenikaa-X: 24 months, two fixed areas at Hòa Lạc, max 40 km/h, safety supervisor in the seat | [SHTP](https://shtp.hochiminhcity.gov.vn/en/establish-a-sandbox-framework-to-support-trials-in-unmanned-aerial-vehicles-and-autonomous-vehicles.htm); [QTSC](https://qtsc.com.vn/en/news/ho-chi-minh-city-allows-controlled-testing-of-self-driving-vehicles-at-qtsc) |
| D8 | California: installing AV technology makes you a manufacturer; $5M; permit | VERIFIED | §227.02 and §227.04, unchanged in the 2026 text (effective 2026-04-28). They cover SAE levels 3–5 only | [DMV](https://www.dmv.ca.gov/portal/file/adopted-regulatory-text-pdf) |
| D9 | NVIDIA postings 893391627291 and 893392738069 | VERIFIED (IDs, live) / PARTLY (phrases) | 893391627291 = **JR2005596** (Senior Research Scientist – AV); 893392738069 = **JR2010863** (Senior ML and Simulation Engineer – AV), the same postings as `05`. Both are still posted. JR2010863 lists "closed-loop evaluation" as nice-to-have, not required | [JR2005596](https://jobs.nvidia.com/careers/job/893391627291); [JR2010863](https://jobs.nvidia.com/careers/job/893392738069) |
| D10 | Think Autonomous course modules; AlpaSim TransFuser baseline | PARTLY / VERIFIED | The Tesla Patent Study Vault covers trigger classifiers. TransFuser is a lecture, not a named project. AlpaSim ships "Latent TransFuser v6 (LTFv6) … (provisional)" | [course](https://thinkautonomous.ai/end-to-end-revolution); [alpasim](https://github.com/NVlabs/alpasim) |

**D7, what the law says (fact-checking only, not legal advice).**
- **Which article.** In Law 36/2024/QH15 on Road Traffic Order and Safety (effective 2025-01-01), Art. 34(4) defines "phương tiện giao thông thông minh" (smart vehicle). **Art. 35(3)** is the rule that such a vehicle be licensed to operate by a competent authority.
- **The definition changed.** Law 118/2025/QH15 narrowed it from 2026-07-01: from vehicles with partial or full automation of control, to vehicles whose control needs no direct driver input.
- **Who can apply.** Decree 151/2024/NĐ-CP Art. 25 (read from mirrors) has the Traffic Police Department issue permits to owners, including individuals.
- **Dashcams.** No rule was found that requires or prohibits dashcams in private cars.
- **Recording in public.** The Personal Data Protection Law 91/2025/QH15, Art. 32 (effective 2026-01-01), permits recording in public places in listed cases and requires notifying the people recorded.
- **What stays uncertain:**
  - whether a passive, non-actuating camera rig falls under either definition (on the plain wording it seems not to, but that is for counsel);
  - HCMC Resolution 20/2024 on eligibility;
  - whether Decree 151 Art. 25 was revised after Law 118/2025.
- Treat all of this as an engineering reading. Confirm local rules before mounting anything on a car.

**Injection note.** One search result (hethongphapluat.com) contained text addressed to the assistant. It was ignored.

## E. Corrections this ledger makes to live documents

| Doc | Text | Correction |
|---|---|---|
| `04` §3 | "TensorRT 11.3 lists FP8 and FP4 on Thor (SM 11.0)" | True for **DRIVE** AGX Thor only. Jetson AGX Thor runs TensorRT 10.16.2 (JetPack 7.2.1), whose matrix lists FP8/FP4 as supported |
| `04` §5 | AlpaSim merges external PRs "usually within 0–6 days" | No external merge since 2026-08-17; `main` not synced since 2026-08-11 |
| `04` §5, `02` §2 | "No distillation recipe" | PR #42 is public and unmerged (10-20) |
| `02` §3 S2 manifest | "both AlpaSim SHAs" pin the environment | A SHA does not pin the environment (issue #207). The manifest also needs the resolved lockfile and `n_sim_steps`, requested and actual (#208) |
| `nano_alpamayo/README.md` | Data on a PAI-AV subset with location-grouped splits | Exclude the new `alpamayo_aligned` split from any evaluation of Alpamayo-derived students (teacher contamination) |
