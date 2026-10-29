# Track B — Demand signals for inference-centred AV work (checked 2026-10-28)

**Scope.** Open work and regressions in NVIDIA's open AV stack, sm_110 (Thor) gaps in kernel libraries, NVIDIA forum threads, live AV inference postings, 2026 breakage, and outside-PR policy. Deltas are against the 10-27 ledgers (`docs/05_ROLE_MAPPING.md` §1–§3, `L1_upstream_delta.md`, `L7_thor_performance.md` §3, `2026-10-03-verification/notes/role_skill_demand.md`). Those facts are not repeated unless they changed.

**Method (all on 2026-10-28, roughly 15:00–16:30 UTC).** Read-only:
- GitHub REST API (unauthenticated) and the GitHub MCP read tools;
- the GitHub search API for sm_110/Thor terms;
- NVIDIA forums through the Discourse JSON endpoints (`search.json`, `/t/<id>.json`, category listings);
- the Workday CXS JSON search and detail endpoints, Greenhouse (Waymo), Ashby (Wayve) and Lever (Zoox) public board APIs;
- the Hugging Face API;
- one NVIDIA blog via WebFetch.

Nothing was signed into, applied to, posted, downloaded from a gated source, or accepted. Raw JSON is in `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/capstone_scratch/B/` (`gh/`, `forum/`, `jobs/`, `q5/`, `q6/`, `hf/`; `sm110_filtered.txt` holds the filtered kernel-library hit list).

**Labels.** *reported*: a source states it. *inferred*: my derivation, arithmetic shown. *unverified*: I could not confirm it, and I say what I tried. "Checked" is 2026-10-28 for every row. Device names follow the source. "Thor (variant not stated)" means the source did not say Jetson or DRIVE.

---

## 1. Takeaways

1. **NVIDIA ships no Alpamayo 1.5 or 2B-student engine path for Jetson AGX Thor as of 10-28.**
   - TensorRT-Edge-LLM said on 07-30 that Alpamayo 1.5 was planned for its August release. It is missing from all three releases since then and the follow-ups got no reply (E01, E18).
   - NVIDIA forum staff said Alpamayo is not available for AGX Thor (F01).
   - An NVIDIA collaborator called 20–30 s per 10B result in PyTorch on Jetson Thor expected (A01).
   - Edge still returns 401 on Hugging Face (R07).
   - **Scoop risk:** FlashRT's maintainer has publicly planned a reference for the official Alpamayo 1.5 that uses only compute and runtime optimizations and leaves the model's behaviour unchanged, with Jetson Thor as a focus (R03). That is path 2's exact framing, for the 10B.
2. **NVFP4 on sm_110 is fragile, not a free speed-up.**
   - Self-quantized Cosmos-Reason2-2B (the PR #42 student VLM) crashes the TensorRT 10.16.2 builder on Jetson AGX Thor. The outside workaround runs 9–10 % slower than the ONNX NVFP4 control (E04).
   - ModelOpt's real-quant GEMM lookup fails and falls back to fake quantization, which forum users report as slow on Thor (R04, F01).
   - TensorRT 11's NVFP4 tactics on SM 11.0 take FP16/BF16 I/O only, per an accepted community answer (F10).
   - vLLM NVFP4 models hit illegal memory accesses on Jetson AGX Thor (K10, K12).
3. **sm_110 kernel demand is real, but crowded and partly automated.**
   - FlashInfer's CAKE pipeline (a harness plus Claude/Codex agents, run by one collaborator) already ships SM110 XQA/GQA decode routes and invites kernel requests. 50 of the 60 most recent merges are that collaborator's (K03–K07).
   - vLLM has duplicate, unmerged SM110 fixes from as long as 49 days ago (K11, K13).
   - CUTLASS merged a two-line SM110 gate fix after 31 days. An earlier PR describing the same two-line change, with a 1.74× DRIVE AGX Thor FMHA measurement, carries the `inactive-90d` label and is still open (K17, K18).
   - No maintainer asks for an Alpamayo-shaped hand-written kernel.
4. **Engines are adding the flow-matching action expert in the datacenter, not on Thor.**
   - vLLM merged Alpamayo 2 Super's reasoning model with the trajectory expert not loaded (R01).
   - vLLM-Omni draft #9133 adds the two-stage reasoning→expert pipeline (BF16, eager, H200). It says the work is needed for NVIDIA's AV data factory (R02).
   - The Alpamayo flow-matching export question in NVlabs/alpamayo #112 has had no answer since 07-23 (A06).
5. **TensorRT-Edge-LLM does not merge outside PRs on GitHub.**
   - All 17 merges in its history are NVIDIA release or CI PRs. Outside fixes are absorbed into releases and the PRs closed (E21, E22).
   - Since 08-01, 33 PRs came from non-NVIDIA handles: 29 open, 0 merged. 22 of the 60 new issues have no comment (E24).
   - #260 has had no human review in 7 days (E16).
   - NVIDIA is wiring an external-PR CI path (copy-pr-bot, GitHub Actions; E23). It has produced no merge yet.
6. **Posting demand. JR2014515 is newly identified and is the closest NVIDIA line-match to path 2** (J03):
   - review models at operator level, using kernel traces;
   - judge whether VLMs, flow-matching decoders and multi-camera tokenizers compile, fit in memory and meet latency on SoCs;
   - MLPerf Edge methodology;
   - TensorRT, Torch-TRT and MLIR-TRT on Jetson and DRIVE.

   It asks for an MS or equivalent and 12+ years, including 8+ in DL optimization and 5+ in embedded work.

   Elsewhere:
   - Wayve's Inference Optimisation role names Orin/Thor and prints no years line (J10).
   - Waymo's Eval Efficiency role asks for statistically efficient evaluation, which suits path 1 or the gate (J08).
   - Zoox's Senior AI Inference Engineer posting from 10-03 is gone (J14).
7. **NVIDIA has claimed or promised:**
   - the Super RL recipe, called imminent on 09-30 and still unreleased (A09);
   - the SM110 experimental attention trackers in FlashInfer (K01, K02);
   - Edge-LLM Alpamayo 1.5 (E01);
   - the 2B student's distillation, 5 Hz inference and speculative-decoding drafts (recipes #42/#45/#43, ledger).
8. **Forums.**
   - Alpamayo-R1-10B on DRIVE AGX Thor through Edge-LLM 0.9.0 needed a 32 GB GPU carveout to build. The fix went through private messages, and no latency was posted (F03).
   - Throttling while running Alpamayo 1.5 BF16 on Jetson AGX Thor at 120 W is expected, per staff (F05).
   - A third-party dashboard (ZETIC, posted 10-28) publishes VLA latency and energy on Jetson AGX Thor: π0.5 at 132 ms with torch.compile. It has no TensorRT yet and no Alpamayo (F08).

---

## 2. Findings table

### Q1 — Open work in the target repos (since 2026-08-01, newest at top; topic-filtered)

**NVIDIA/TensorRT-Edge-LLM.** Totals since 08-01: 60 issues created (53 open, 22 with 0 comments, 0 assigned) and 42 PRs (33 from non-NVIDIA handles: 29 open, 4 closed, 0 merged). Source: E24, counted from the API list. NVIDIA handles were identified by their release and CI PRs (inferred).

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| E14 | #263 "Hybrid + MTP context reuse rejects prompts shorter than one cache page (HTTP 500)". Open, label `bug`, 0 comments, 0 reactions, unassigned, no maintainer reply, no PR | 2026-10-26 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/263 | reported |
| E16 | PR #260 "Alpamayo fused gate/up plugin for SM110" (fork `jin-yc10`). Open. The only comment is copy-pr-bot saying the PR needs vetting before CI runs on NVIDIA runners. No human review in 7 days | 2026-10-21 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260 | reported (7 days: inferred, 10-21→10-28) |
| E15 | #252: an outside user offers "significant performance improvements" for Orin (sm87) and RTX 4060 (sm89). Open, 0 comments, no maintainer reply | 2026-10-18 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/252 | reported |
| E08 | #250: GDN Tree-MTP greedy decoding diverges at token 14 on Thor SM110 (variant not stated), and a partial workaround still fails. Open, 0 comments | 2026-10-16 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/250 | reported |
| E05 | #232: the Cosmos3-Edge reasoner on Jetson AGX Thor describes the wrong scene compared with BF16. NVIDIA (nvluxiaoz, COLLABORATOR) said on 10-09 it will address this. An outsider (0z5a) traced it on 10-16 to the 0.11.0 exporter and runtime both converting the patch layout, so the conversion happens twice, and opened fix PR #251 (open, no review). Related: #233, where `tensorrt-edgellm-serve` cannot start Cosmos3-Edge on Thor (0 comments) | 2026-10-07 → 10-16 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/232 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/251 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/233 | reported |
| E09 | #228: a 0.11.0 regression where the speculative-base engine for Qwen3.8-27B fails to build at maxInputLen 196608 because Myelin autotune requests too much memory. Open, `bug`, 2 comments. Related: #222 and #223, long-context and MTP build failures, both open | 2026-09-30 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/228 | reported |
| E04 | #216: `tensorrt-edgellm-build --dense nvfp4-qdq` segfaults in a libnvinfer compiler thread on **Jetson AGX Thor** (CC 11.0) for self-quantized **Cosmos-Reason2-2B** (NVFP4 backbone, NVFP4 lm_head, FP8 visual; ModelOpt 0.45.0, Edge-LLM 0.10.1). 10 comments. NVIDIA last replied on 09-20, asking for the quantization command. On 09-28 an outsider reduced the crash to a pure TensorRT graph (no weights, no Edge-LLM) on TRT 10.16.2.10 / CUDA 13.2.78, and opened draft PR #225, a rank-2 NVFP4 GEMM lowering workaround (open, 0 comments). **Speed tuple** (follow-up, 09-29):<br>• baseline: the ONNX NVFP4 path;<br>• workloads: 12 text prompts (439 generated tokens) and 1 image (64 tokens, encoder cache off);<br>• precision: NVFP4 plus FP8 visual;<br>• device: Jetson AGX Thor, 120 W, TRT 10.16.2.10, CUDA 13.2.78, driver 595.78, Edge-LLM `e8b2952`;<br>• real engine, 4 alternating repetitions after 5 warm-ups;<br>• **missing:** no profiler trace; JetPack not stated.<br>Results: native workaround 2,829.88 ms vs ONNX 2,562.46 ms for text; 401.46 vs 367.28 ms for the image. The workaround is 9–10 % slower and "no speedup is claimed"; outputs match exactly | 2026-09-17 → 09-29 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/216 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/225 | reported |
| E04a | From E04's ONNX control: 2,562.46 ms ÷ 439 tokens = **5.84 ms per generated token** (prefill included) and 367.28 ÷ 64 = 5.74 ms per token (vision encoder included). At that rate a 100 ms (10 Hz) tick could hold about 100 ÷ 5.84 ≈ 17 generated tokens if nothing else ran. This is a 2B Qwen3-VL-class VLM in NVFP4 at 120 W, not an Alpamayo student and not an action expert | — | 10-28 | (derived from E04) | inferred |
| E10 | #217: direct export and build fail for Qwen3-VL fine-grained 2-D block-FP8 checkpoints (open, `bug`, 2 comments). An outsider's PR #220 adds blockwise FP8 GEMM for Qwen3-VL (open, 3 comments, no merge) | 2026-09-17 / 09-23 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/217 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/220 | reported |
| E11 | PR #215, an outsider's native continuous batching and chunked prefill for the LLM runtime. Open, 0 comments, no review. PR #199, request batching for the experimental server, is also open | 2026-09-15 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/215 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/199 | reported |
| E12 | Sampling and determinism: #211 says `llm_inference` ignores temperature, top_p and top_k from its input JSON (open, 2 comments). PR #212 advances the Philox offset per sampled token (open, no review). These bear on any seeded-sampling contract | 2026-09-12 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/211 ; https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/212 | reported |
| E26 | #192: a Qwen3.8-27B NVFP4 base-engine build fails on Jetson because Myelin's `removeEmptyProducersFromSubgraph` is not implemented for PLUGIN_V3. Open, 4 comments | 2026-08-24 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/192 | reported |
| E13 | #186: support multimodal calibration when quantizing only the LLM backbone. Open, 1 comment, 2 reactions. An outsider volunteered on 09-30; no maintainer reply | 2026-08-20 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/186 | reported |
| E03 | #166: exporting Alpamayo-R1-10B fails with an AssertionError. Open, `bug`. NVIDIA said on 08-13 it will look into it. The L7 ledger records it as fixed in v0.10.1 release notes, but the issue is still open | 2026-08-06 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/166 | reported |
| E07 | #164: garbage or repetitive output on Jetson AGX Thor (SM110) across all quantization formats using CuTe DSL FMHA. NVIDIA said on 08-13 it should be fixed in 0.10.0. Still open | 2026-08-05 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/164 | reported |
| E06 | #151: a regression on Jetson AGX Thor, layer-0 V-tensor corruption in v0.9.1 when v0.4.0 works. An outsider root-caused it to TensorRT Myelin's horizontal FC fusion miscompiling at maxBatchSize = 1 on SM110 (TRT 10.13.3.9). NVIDIA said on 08-01 the fix goes in the next release. Still open, 6 comments. Closed sm_110 items from the same period: #156 (missing prebuilt XQA cubins for sm_110, closed 08-20) and #160 (NVFP4 MoE plugin shape rejected on SM110, closed 09-03) | 2026-07-28 → 08-15 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/151 ; …/issues/156 ; …/issues/160 | reported |
| E02 | #134 "Support for Quantized Alpamayo Export". Open, 5 comments. NVIDIA answered only a side question about DRIVE AGX Orin; no quantized-Alpamayo answer | 2026-07-21 → 08-24 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/134 | reported |
| E01 | #130 "Alpamayo Support" (labels `New Model`, `Roadmap`; open; 3 comments; unassigned). NVIDIA (linc-nv) said on 07-30 that Alpamayo 1.5 "is planned on August release". User follow-ups on 08-12 and 09-15 got no reply. No Alpamayo 1.5 in v0.10.0 (08-12), v0.10.1 (09-03) or v0.11.0 (09-29). No release and no non-CI commit since 09-29 (E18, E19). The promise is 90 days old (07-30 → 10-28) | 2026-07-30 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130 | reported (slip: inferred) |
| E17 | #32, the H1 2026 developer roadmap (`Roadmap`; 15 reactions; last updated 08-11). It lists Alpamayo 1 as done (0.7.1) and Cosmos 3 and native video as items, with **no Alpamayo 1.5 line**. No H2 2026 roadmap issue appears in the issue list since 08-01 | 2026-02-05 / 08-11 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/32 | reported |
| E25 | Older but relevant: #47, prefill hidden state and KV diverge from PyTorch before the attention-plugin output for the Alpamayo/Qwen3-VL text path. Open, last updated 07-14 | 2026-03-12 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/47 | reported (title and metadata only) |

**NVlabs/alpamayo1.5, NVlabs/alpamayo, NVlabs/alpamayo-recipes, NVlabs/alpamayo2.** The exact recipes org is **NVlabs**: https://github.com/NVlabs/alpamayo-recipes.

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| A02 | alpamayo1.5 #40 (stationary-lead collision in native AlpaSim) and #39 (the same in CARLA): both open, 0 comments, 0 reactions, unassigned. Already in L1 via AlpaSim #224 | 09-30 / 09-26 | 10-28 | https://github.com/NVlabs/alpamayo1.5/issues/40 ; …/issues/39 | reported |
| A03 | alpamayo1.5 PR #37, "ci: add GPU test workflow" by a collaborator: open since 08-24, no activity. The repo has no CI (A05) | 2026-08-24 | 10-28 | https://github.com/NVlabs/alpamayo1.5/pull/37 | reported |
| A01 | alpamayo1.5 #22, installing Alpamayo 1.5 on Jetson Thor with JetPack 7.2 (closed 10-06; 4 comments). NVIDIA (super-anova) said on 06-22 that it was validated only on CUDA 12. A user reported 20–30 s per result for the community R1-10B on Jetson Thor (JetPack 7.2, CUDA 13.2, flash-attn built from source). NVIDIA replied on 07-09: "roughly in line with what we would expect from Jetson Thor". **Speed tuple:** no baseline, shapes, precision, power mode or trace; user-reported | 2026-06-22 → 07-09 | 10-28 | https://github.com/NVlabs/alpamayo1.5/issues/22 | reported |
| A04 | alpamayo1.5 #29, reproducing the paper's reasoning metrics: open, 6 comments, 1 reaction, unassigned | 2026-07-07 → 09-16 | 10-28 | https://github.com/NVlabs/alpamayo1.5/issues/29 | reported |
| A06 | alpamayo (R1):<br>• #112 asks about exporting the flow-matching part of Alpamayo-R1-10B to TensorRT Edge-LLM. Open, **0 comments since 07-23**.<br>• #121 asks for clip IDs and the protocol behind the 937-sample, minADE₆ = 1.22 m open-loop result. Open, 0 comments, 10-20.<br>• PR #122 (Intel XPU support) closed unmerged on 10-22.<br>• #110 (RL training, `enhancement`) and #106 (LoRA/PEFT) are open | 07-23 → 10-22 | 10-28 | https://github.com/NVlabs/alpamayo/issues/112 ; …/issues/121 ; …/pull/122 ; …/issues/110 | reported |
| A07 | recipes #46 (**new 10-28**): `train_hf.py` fails before training on single-run launches with a Hydra `InterpolationToMissingValueError` on `hydra.job.num`. Open, `bug`, assigned to yesfandiari, 0 comments | 2026-10-28 | 10-28 | https://github.com/NVlabs/alpamayo-recipes/issues/46 | reported |
| A08 | recipes PR #44 (measurement chapter) was **updated 10-28**. Its author adopted a reviewer's order: per-rank telemetry → wall clock → CUDA-event decomposition → GPU-hour → MFU/HFU/OFU. Still open | 2026-10-22 → 10-28 | 10-28 | https://github.com/NVlabs/alpamayo-recipes/pull/44 | reported |
| A09 | recipes #33, release of Alpamayo 2 Super training code from SFT to RL (open, 5 comments, assigned yesfandiari):<br>• 08-14, NVIDIA (super-anova): the SFT and **open-loop RL** recipes for Super will be open-sourced in this repo;<br>• 08-19: 3–4 weeks, tentatively;<br>• 09-30, BorisIvanovic: SFT merged, "RL coming very shortly too".<br>No RL PR is open or merged on 10-28 (repo PR list) | 08-14 → 09-30 | 10-28 | https://github.com/NVlabs/alpamayo-recipes/issues/33 | reported |
| A10 | recipes #20, a distillation-recipe request (open, 4 comments). NVIDIA said on 06-26 it was exploring it; PR #42 (distillation, NVIDIA) is now open. PRs #42, #43 and #45 are unchanged since 10-20/10-21 (ledger) | 06-26 → 09-17 | 10-28 | https://github.com/NVlabs/alpamayo-recipes/issues/20 | reported |
| A11 | recipes #35, `zip_cache` causes a CPU-memory OOM (open, `bug`, 4 comments, assigned). #29, an outsider's report that the reasoning-reward sign was inverted in GRPO, was fixed by NVIDIA PR #40 on 10-06: outside bug reports do get acted on | 09-11 / 08-03 → 10-06 | 10-28 | https://github.com/NVlabs/alpamayo-recipes/issues/35 ; …/issues/29 ; …/pull/40 | reported |
| A13 | alpamayo2 (Super): all assigned to yesfandiari.<br>• #19, a bug report of implausible within-frame trajectories and poor frame-to-frame consistency (title in Chinese). Open, 1 comment, 10-20.<br>• #9: under a total visual blackout Super narrates a clear path ahead. Open, 3 comments.<br>• #13 and #14 (VQA and auto-label bugs): open | 08-15 → 10-20 | 10-28 | https://github.com/NVlabs/alpamayo2/issues/19 ; …/issues/9 | reported |

**NVlabs/alpasim.** L1 covers the repo in depth up to 10-27. Only deltas are listed here.

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| A14 | #227 (opened and closed 10-28): a challenge submission failed with CLEAN_SLATE_FAILED after all 32 rollouts were scored. mwatson-nvidia: the simulation ran; the failure came while waiting for multiple replicas' exit events; teardown uses `docker rm -f` (SIGKILL); pinned memory is a possible contributor but not established | 2026-10-28 | 10-28 | https://github.com/NVlabs/alpasim/issues/227 | reported |
| A15 | PR #228 (10-28), "Add nuPlan leaderboard reference scores", by WCJ-BERT (association NONE). Open, 1 comment | 2026-10-28 | 10-28 | https://github.com/NVlabs/alpasim/pull/228 | reported |
| A16 | #74 closed 10-28 (assigned mwatson-nvidia). An NVIDIA collaborator said on 04-27 that the public AlpaSim integration has no wizard option for Alpamayo 1.5 navigation text, and that the tutorial's CFG-navigation command is misleading. Run with `use_classifier_free_guidance_nav=false` until the plumbing exists | 04-27 → 10-28 | 10-28 | https://github.com/NVlabs/alpasim/issues/74 | reported |
| A17 | #17 (sensorsim on Blackwell; open since 01-09; 11 comments). NVIDIA pointed to PR #63 on 03-31, which updates the NuRec image. On 07-02 a user reported it works on Blackwell with NRE GA 26.02.00. The issue is left open. Not a blocker for RTX PRO 6000 on that report | 01-09 → 07-02 | 10-28 | https://github.com/NVlabs/alpasim/issues/17 | reported |
| A18 | No AlpaSim commit since 10-20 (`pushed_at` is 2026-10-20T13:35:17Z); 18 open issues and PRs | 2026-10-20 | 10-28 | https://api.github.com/repos/NVlabs/alpasim | reported |
| A19 | Many NVlabs issues were closed on the same dates (alpamayo1.5 on 10-06; alpamayo on 10-07 and 10-12), including the Thor install thread #22, whose last staff answer was 07-09. That looks like a bulk triage sweep | 10-06 / 10-12 | 10-28 | https://github.com/NVlabs/alpamayo1.5/issues?q=is%3Aissue+is%3Aclosed | inferred |

### Q2 — sm_110 / Thor gaps in kernel libraries (GitHub search, items since 2026-05-01)

**Method (K24).** For each repository I ran two GitHub search-API queries: `repo:<r> sm_110 OR sm110 OR sm_110a OR sm110a` and `repo:<r> "Jetson Thor" OR "AGX Thor" OR "Thor"`. I kept issues whose text matches sm_110 or Thor and PRs whose title does. Raw totals: FlashInfer 184/65, FlashAttention 62/25, CUTLASS 14/25, Triton 18/16, PyTorch 46/76, vLLM 113/122, SGLang 43/47. Filtered list: `capstone_scratch/B/sm110_filtered.txt`.

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| K01 | FlashInfer #5051, tracking the graduation of the **experimental SM110 GQA decode** (owner yyihuang, COLLABORATOR; labels `needs-triage`, `experimental`; 0 comments). Scope: FP16 decode for Jetson AGX Thor at one fixed shape (32 query heads, 8 KV heads, D = 128, one query token), dense stacked KV. Target: v0.6.20, "within four weeks" | 2026-09-09 | 10-28 | https://github.com/flashinfer-ai/flashinfer/issues/5051 | reported |
| K02 | FlashInfer #5292, tracking **experimental SM110 tcgen05 attention** (owner yyihuang; 0 comments; not updated since 09-17). Scope: D128 FP16 decode with fused split reduction, plus D512 FP16/FP8 tree attention. Readiness-review target 2026-10-15; the outcome is not recorded on the issue | 2026-09-17 | 10-28 | https://github.com/flashinfer-ai/flashinfer/issues/5292 | reported (outcome: unverified) |
| K03 | FlashInfer #4254, the long-term **CAKE-generated kernel** tracker (26 reactions; updated 10-28). It tracks kernels "generated from CAKE harness + claude/codex agent". It invites anyone to request a custom kernel and offers to build it | 2026-07-29 → 10-28 | 10-28 | https://github.com/flashinfer-ai/flashinfer/issues/4254 | reported |
| K04 | FlashInfer #6594, the CAKE attention tracker, lists an "SM110 XQA and GQA Decode" group (#5932–#5934, #6061, #7293) plus SM110 GQA PRs | 2026-10-16 → 10-28 | 10-28 | https://github.com/flashinfer-ai/flashinfer/issues/6594 | reported |
| K05 | Merged CAKE SM110 PRs by yyihuang (merge status checked with an `is:merged SM110` search; 86 hits):<br>• #5052 (experimental SM110 GQA decode kernel, 09-09), #5302 (prepared SM110 GQA decode, 09-17), #5293 (experimental SM110 XQA, tcgen05 and register-MMA, 09-27);<br>• #5597, #5658, #5674 (experimental SM110 XQA routes, 09-27 → 09-29);<br>• #5872 (regenerate Thor XQA, 10-03);<br>• #5929 (SM110 GQA decode refactor, 10-05);<br>• #6944, #6945, #6981, #6982 ("faster SM110 GQA decode/XQA", 10-21/22);<br>• #7116, #7293 ("deslop(sm110)", 10-23/25) | 09-27 → 10-25 | 10-28 | https://github.com/flashinfer-ai/flashinfer/pull/6982 (and the others listed) | reported |
| K06 | Outside SM110 PRs to FlashInfer:<br>• merged: #5695 (CUTLASS fused MoE in the SM110 providers; 2 days) and #6295 (SM100-family modules in the SM110 provider; 19 days);<br>• open: #5882 (SM110 W4A16 CuTe DSL MoE, 10-01, 1 comment), #4670 (SM110 MoE cluster shapes, 08-21), and #3734 (Thor CuTe-DSL attention fixes by jimmyzho, CONTRIBUTOR; open since 06-26; 17 comments) | 06-26 → 10-28 | 10-28 | https://github.com/flashinfer-ai/flashinfer/pull/5695 ; …/pull/6295 ; …/pull/5882 ; …/pull/3734 | reported (durations: inferred from dates) |
| K07 | FlashInfer merge volume: the search `is:merged merged:>=2026-10-20` returns **726** PRs, about 83 a day over 8.7 days (inferred). Of the 60 most recently created merged PRs returned (all merged 10-28), **50 are by yyihuang**, nearly all `cake_*` perf PRs | 10-20 → 10-28 | 10-28 | https://github.com/flashinfer-ai/flashinfer/pulls?q=is%3Apr+is%3Amerged+merged%3A%3E%3D2026-10-20 | reported (count) / inferred (share) |
| K09 | vLLM #63267: on Jetson Thor (sm_110), CUDA builds without the FA2/FA3 extensions cannot start. `vllm_flash_attn` raises at import time and FA is still auto-selected. NVIDIA's `ghcr.io/nvidia-ai-iot/vllm:r38.2` image ships no FA extension, and sm_110 has no FA2/FA3 kernels. Open, 3 comments, unlabelled | 2026-10-23 | 10-28 | https://github.com/vllm-project/vllm/issues/63267 | reported |
| K10 | vLLM #62149 (open, `quantization`): Qwen3.6-35B-A3B NVFP4 on **Jetson AGX Thor** (MAXN, vLLM 0.31.0, FlashInfer 0.7.0.post1, Triton 3.7.1, CUDA 13.0.2) hits an illegal memory access in the GDN prefill warm-up on 3 of 4 cold-cache boots and 0 of 3 warm ones. PR #62275, disabling the CUTLASS FP4 MoE backend on sm_110, is open | 10-17 / 10-18 | 10-28 | https://github.com/vllm-project/vllm/issues/62149 ; https://github.com/vllm-project/vllm/pull/62275 | reported |
| K11 | vLLM `cooperative_topk` fails on SM110 with a cluster-launch misconfiguration: issue #56083 (0 comments) and **two competing open PRs**, #56084 (needs-rebase) and #56102, all 09-09, plus a re-report #58954 on 09-28. Nothing merged after 49 days | 09-09 → 09-28 | 10-28 | https://github.com/vllm-project/vllm/issues/56083 ; …/pull/56084 ; …/pull/56102 ; …/issues/58954 | reported (49 days: inferred) |
| K12 | vLLM #55406: Nemotron NVFP4 dies with an illegal memory access in CUDA-graph replay on SM110 (Jetson). Open, 3 comments. PR #55366, adding SM110 to FA4 auto-selection, is open | 09-04 | 10-28 | https://github.com/vllm-project/vllm/issues/55406 ; …/pull/55366 | reported |
| K13 | vLLM: two open PRs add tuned Mamba `selective_state_update` configs for Thor, #44590 (06-05) and #53337 (08-22), neither merged. PR #60843 (host-pinned KV cache on integrated GPUs; `verified`, `nvidia`, 5 comments) is active on 10-28. #45260 is a MEMBER's CUDA-architecture coverage audit (`keep-open`) | 06-05 → 10-28 | 10-28 | https://github.com/vllm-project/vllm/pull/53337 ; …/pull/44590 ; …/pull/60843 ; …/issues/45260 | reported |
| K16 | PyTorch #201193 (open; `triaged`, `module: jetson`, `oncall: pt2`, inductor): Inductor's Blackwell GEMM templates never apply on Jetson Thor (sm_110). The datacenter-Blackwell check covers 100–109, and CUDA 13 renumbered Thor from sm_101 to sm_110. The reporter says the analysis was AI-assisted. A different contributor's PR #202001 (10-25, open) treats Jetson Thor as datacenter Blackwell. Also:<br>• #200432, flexgemm sm110 → sm100 path (COLLABORATOR, 9 comments, active 10-28);<br>• #192209, MapAllocator OOM on unified memory (open);<br>• #192386, skip varlen FA4 tests on Thor (merged 08) | 08-05 → 10-28 | 10-28 | https://github.com/pytorch/pytorch/issues/201193 ; …/pull/202001 ; …/pull/200432 ; …/issues/192209 | reported |
| K17 | CUTLASS PR #3637 (roma5087, CONTRIBUTOR) enables `CUDA_CTA_RECONFIG_ACTIVATED` for SM110a (Jetson AGX Thor). It adds two lines to `include/cutlass/arch/reg_reconfig.h` and closes #3576. Created 09-14, merged 10-15 (31 days) | 09-14 → 10-15 | 10-28 | https://github.com/NVIDIA/cutlass/pull/3637 ; https://github.com/NVIDIA/cutlass/issues/3576 | reported |
| K18 | CUTLASS PR #3308 (pzhao-eng; labels `inactive-30d`, `inactive-90d`; still open) makes the same two-clause `reg_reconfig.h` change three months earlier. **Speed tuple:**<br>• device: **DRIVE AGX Thor** (`sm_110a`);<br>• kernel: example 77 Blackwell FMHA forward, FP16, B = 1, H = 16, D = 64, S = 2,984, tile 256×128×64;<br>• baseline: the same kernel without `setmaxnreg`.<br>Result 1.33 ms → 762 µs (**1.74×**). Spill traffic 529 MB → 0, SM throughput 42 % → 65 %, with a profiler-style screenshot. **Missing:** profiler tool not named; no DriveOS, CUDA or power mode. A **DRIVE** number, not a Jetson one | 2026-06-08 | 10-28 | https://github.com/NVIDIA/cutlass/pull/3308 | reported (same change as K17: inferred from the diff) |
| K19 | Other CUTLASS items:<br>• #3360 (enable SM110 for Blackwell FMHA and warp-group register reconfiguration; open; inactive-30d);<br>• #3323 (CuTeDSL MHA fix for Thor; open; inactive-90d);<br>• #3425 (Jetson Thor example exits with code -6; closed 10-08 after inactivity);<br>• #3283 (enable tcgen05 block-scaled ops on Thor SM110; merged 06-16) | 05-28 → 10-08 | 10-28 | https://github.com/NVIDIA/cutlass/pull/3360 ; …/pull/3323 ; …/issues/3425 ; …/pull/3283 | reported |
| K21 | Triton:<br>• #11923, "Fix SM110 dot opcode checks" (CONTRIBUTOR; open since 09-23; 0 comments);<br>• #10462 (release/3.7.x), separate `ptxas-thor` for sm_110 (closed 06-09);<br>• #11021, "Perf/thor blackwell 2cta" (outsider; closed 07-23) | 06-03 → 09-23 | 10-28 | https://github.com/triton-lang/triton/pull/11923 ; …/pull/10462 ; …/pull/11021 | reported |
| K22 | SGLang:<br>• #41765 (merged 10-06) fixes #41764, where Thor SM110 auto-backend selection crashed FP8 and ModelOpt NVFP4 MoE startup;<br>• open: #45520 (NVFP4 MoE runner on SM110/SM120), #43694 (FA4 on SM100/SM110 for dflash), #37641 (sm_110a gencodes for aarch64 + CUDA 13);<br>• #36131 (NVIDIA collaborator nvpohanh): the unified-cache default flip **regressed long-prefix decode throughput on Spark and Thor** (closed 09-12) | 08-24 → 10-23 | 10-28 | https://github.com/sgl-project/sglang/pull/41765 ; …/pull/45520 ; …/pull/43694 ; …/issues/36131 | reported |
| K23 | FlashAttention PR #2945 (0z5a; open; **0 comments in 27 days**) cuts FA4 inference dispatch overhead and tunes SM110 prefill. **Speed tuple:**<br>• model: Qwen3-0.6B, BF16, all 28 layers through FA4;<br>• baseline: unpatched FA4;<br>• 128 in + 32 out: 766.272 → 717.495 ms (1.068×; 95 % block bootstrap 1.046–1.090);<br>• 512 + 32: 758.452 → 699.223 ms (1.085×);<br>• logits bitwise equal; real kernels.<br>**Missing:** Thor variant not stated, and no JetPack/CUDA version, power mode or profiler trace. Also: PR #2610 (pack_gqa backward on SM100/SM110; open), PR #2590 (sm_110 in Blackwell-family gating; merged 05-30) | 2026-10-01 | 10-28 | https://github.com/Dao-AILab/flash-attention/pull/2945 ; …/pull/2610 ; …/pull/2590 | reported |
| K25 | One outside handle, 0z5a, recurs across Thor work: Edge-LLM #216, #225, #229 and #251; CUTLASS #3694 (closed); FA #2945. That is an active independent contributor in the same niche | 09-28 → 10-16 | 10-28 | (links in E04, E05, K23) | inferred |

### Q3 — NVIDIA Developer Forums (Jetson AGX Thor, DRIVE AGX Thor, Alpamayo categories; since 2026-06)

**Method (F16).** I ran `search.json` queries with `after:2026-06-01` for "alpamayo", "alpamayo thor", "TensorRT-Edge-LLM", "tensorrt edge llm thor", "nvfp4 thor", "VLA thor latency", "cosmos reason thor", "alpamayo drive thor" and "edge-llm alpamayo". I also read the DRIVE AGX Thor General listing: 100 topics since 06-01, filtered by title keywords. Staff flags (`staff`, `moderator`) come from the topic JSON.

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| F01 | Jetson Thor, "How to perform Alpamayo1.5 fp8 quantization for Nvidia Thor" (15 posts, 456 views). The user found no Thor NIM and no working quantized Alpamayo 1.5 on Thor. Staff (AastaLLL) on 09-07: "Alpamayo is not available for AGX Thor currently"; on 09-08: NIM and TensorRT-LLM are not supported on Jetson. The user's quantized model ran very slowly with fake quantization, which they linked to ModelOpt #1895. Their 09-15 question on how to get a real speed-up got no staff reply; the topic auto-closed on 10-14 | 09-04 → 10-14 | 10-28 | https://forums.developer.nvidia.com/t/382409 | reported (the 09-07 quote is already in the 10-25 ledger) |
| F02 | Jetson Thor, "Build alpamayo1_5 native on Thor": a community recipe (PyTorch 2.9.0 from source, SDPA instead of flash-attn). Staff pointed to TensorRT Edge-LLM (accepted) | 2026-09-08 | 10-28 | https://forums.developer.nvidia.com/t/382647 | reported |
| F03 | DRIVE AGX Thor, "Drive AGX thor with Alpamayo" (28 posts, 614 views, solved):<br>• setup: Alpamayo-R1-10B through Edge-LLM 0.9.0 on DriveOS 7.2.5 (CUDA 13.3, TensorRT 11.0.1);<br>• the engine build failed asking for ~15.17 GB, whatever the batch, input or KV settings;<br>• fix: GPU carveout raised to 32 GB, plus hugepages;<br>• tokenizer and preprocessor files were missing, the input JSON did not match the user guide, and the output had a trajectory but no text;<br>• resolved through private messages; staff (SivaRamaKrishnaNV) posted accepted verification steps on 10-08.<br>**No latency was reported** | 09-06 → 10-08 | 10-28 | https://forums.developer.nvidia.com/t/382501 ; https://forums.developer.nvidia.com/t/382500 | reported |
| F05 | Jetson Thor, "System throttled due to Overcurrent". While testing Alpamayo 1.5-10B (a locally built BF16 Docker image) on a Jetson AGX Thor dev kit at **120 W** (JetPack 7.2 GA, L4T 39.2.0, driver 595.78), the board throttled. Staff (accepted): expected, because power capping protects the device when the GPU runs at full speed | 10-14 → 10-15 | 10-28 | https://forums.developer.nvidia.com/t/385881 | reported |
| F06 | Alpamayo Models, "Testing Alpamayo-1.5B in Alpasim: model instability". Code-only post, 113 views, **no reply** | 2026-09-30 | 10-28 | https://forums.developer.nvidia.com/t/384744 | reported |
| F07 | Alpamayo Recipes: the Super SFT recipe uses all seven cameras with four frames each, while the model card's trajectory profile uses cameras [0,1,2,3,5,6]. Which is intended? **No reply** | 2026-10-19 | 10-28 | https://forums.developer.nvidia.com/t/386270 | reported |
| F08 | Jetson Projects (**10-28**): ZETIC launched a public dashboard (ixian.zetic.ai) of VLA latency, energy and memory on Jetson boards. **Speed tuple:**<br>• device: Jetson AGX Thor dev kit 128 GB, JetPack 7.2.1 / L4T R39.2.1, CUDA 13.2.2, PyTorch 2.14.1+cu130, MAXN, GPU clocks pinned at 1,575 MHz;<br>• BF16, batch 1, PyTorch eager vs torch.compile ("reduce-overhead"); latency is p50 per policy call;<br>• power from tegrastats over ~60 s;<br>• π0.5 254 → 132 ms; SmolVLA 197 → 50 ms; GR00T N1.7 109 ms (eager); π0.5 on AGX Orin 949 → 394 ms.<br>TensorRT is "not measured yet". No Alpamayo, no driving, and no trace disclosed | 2026-10-28 | 10-28 | https://forums.developer.nvidia.com/t/387191 | reported |
| F09 | NVIDIA tech blog (09-16): TensorRT Edge-LLM finished the MLPerf Edge Agentic benchmark 6.4× faster on Jetson AGX Thor. **Speed tuple:**<br>• baseline: the llama.cpp reference run (Qwen3.6-27B, Q4_K_M; 2 h 37 min);<br>• Edge-LLM: Qwen3.6-27B with NVFP4 weights and activations, FP8 KV, tree MTP (24 min 36 s);<br>• Jetson AGX Thor dev kit 128 GB, MAXN, SingleStream, concurrency 1; MLPerf Inference v6.1 submission;<br>• **missing:** JetPack and TensorRT versions, llama.cpp version, trace.<br>Whole-benchmark wall time. No Alpamayo, VLA or flow matching | 2026-09-16 | 10-28 | https://developer.nvidia.com/blog/tensorrt-edge-llm-completes-the-mlperf-edge-agentic-benchmark-6-4x-faster-on-jetson-agx-thor/ ; https://forums.developer.nvidia.com/t/383470 | reported |
| F10 | Jetson Thor, "FP4 Quantization fails on Jetson Thor with TensorRT 11.0.0.114". ModelOpt `NVFP4_DEFAULT_CFG` (dynamic activation quantization) fails in Torch-TensorRT; FP8 and FP4 weight-only work. Accepted community answer: TRT 11's NVFP4 GEMM tactics on SM 11.0 offer FP16/BF16 I/O only, so cast the model. Staff asked only for a desktop comparison | 07-23 → 07-24 | 10-28 | https://forums.developer.nvidia.com/t/377887 | reported (community answer) |
| F11 | Jetson Thor, "Cutlass FP4/FP8/FP16 Performance Test" (820 views). Staff: the spec TFLOPS figure is theoretical, NVIDIA has no CUTLASS benchmark score for Thor, and the user should look at LLM perf data instead; staff supplied CUTLASS profiler commands | 06-02 → 06-15 | 10-28 | https://forums.developer.nvidia.com/t/372014 | reported |
| F12 | Jetson Thor: the Edge-LLM high-level Python API reported "No math backend found" on JetPack 7.2, fixed through the Python environment. A community member opened build PRs | 06-30 → 07-02 | 10-28 | https://forums.developer.nvidia.com/t/375013 | reported |
| F13 | DRIVE AGX Thor: TensorRT 10.10.10 on DriveOS 7.0.3 can enable CUDA stream capture implicitly, mostly during engine build and tactic profiling. Staff confirmed (accepted) | 2026-07-07 | 10-28 | https://forums.developer.nvidia.com/t/375838 | reported |
| F14 | DRIVE AGX Thor General: the newest DriveOS announced is 7.2.5 (08-06), and topics through 10-22 still reference 7.2.5. No DriveOS 7.2.6 topic title appears among the 100 topics since 06-01 | 08-06 → 10-22 | 10-28 | https://forums.developer.nvidia.com/t/379371 ; https://forums.developer.nvidia.com/c/autonomous-vehicles/drive-agx-thor/drive-agx-thor-general/742 | reported / inferred (absence) |
| F15 | Jetson Thor: a community member runs Qwen3.8-Flash-Next at 46.7 tok/s on a 128 GB Jetson AGX Thor (NVFP4 routed experts plus blockwise FP8 for 300 smaller tensors). Staff pointed to NVIDIA's NVFP4 tutorial. No power mode or trace in the excerpt read | 09-14 → 10-01 | 10-28 | https://forums.developer.nvidia.com/t/383269 | reported |

### Q4 — Job postings (live-checked 2026-10-28 through each employer's public board API; paraphrased; nothing applied to)

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| J01 | **NVIDIA JR2010863**, Senior ML and Simulation Engineer – AV (Santa Clara). Live (`canApply` true), start date 2026-08-29, L5/L6. Text consistent with the lines mapped in `05` §3 (not diffed word for word).<br>**Duties:** RL training frameworks for multimodal AV models; simulation and data pipelines; measuring and improving simulation quality and the RL reward; GPU-cluster reliability tooling; bringing new architectures into pipelines.<br>**Required:** PPO/GRPO; C++/Python; clusters and HPC with Kubernetes/SLURM.<br>**Stand-out:** simulation and closed-loop evaluation of end-to-end models; data pipelines and algorithm optimization.<br>**No on-device latency line.** BS or equivalent; 12+ years. The text says applications are accepted "at least until" 09-02, yet it is still open | 2026-08-29 | 10-28 | https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Machine-Learning-and-Simulation-Engineer---Autonomous-Vehicles_JR2010863 | reported |
| J02 | **NVIDIA JR2025038**, Senior Integration Engineer, End-to-End Model – AV (Santa Clara). The team is not named formally; it develops NVIDIA's end-to-end driving application. Live, start 2026-09-18, L3/L4. Requirements:<br>(1) integrate learned driving models with vehicle interfaces, sensors, localization and mapping;<br>(2) define model I/O, timing, state and runtime-interface contracts;<br>(3) explain why model behaviour differs between development and target platforms;<br>(4) tune inference for latency, throughput, memory, determinism and power;<br>(5) build driving-quality and regression metrics at scale;<br>(6) in-vehicle testing and driving-data analysis;<br>(7) C++, Python, CUDA and TensorRT on Linux and QNX/RTOS.<br>Stand-out: quantization, compilation, profiling, hardware-aware design. PhD + 1 year, MS + 3 or BS + 5 (or equivalent) | 2026-09-18 | 10-28 | https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Integration-Engineer--End-to-End-Model---Autonomous-Vehicles_JR2025038 | reported |
| J03 | **NVIDIA JR2014515** (**newly identified**): Senior Systems Software Engineer – Deep Learning Solutions (Santa Clara). Live, start 2026-10-14, 224,000–356,500 USD (no level printed). The team is not named beyond the title; it works with NVIDIA's compiler, runtime and hardware groups and with automotive OEMs and robotics partners. Requirements:<br>(1) optimize DL inference for AV and robotics on edge devices, reviewing models at operator and kernel level;<br>(2) find bottlenecks from kernel traces;<br>(3) judge new architectures (vision encoders, multimodal VLMs, hybrid SSM-transformers, **diffusion/flow-matching decoders**, **multi-camera tokenizers**) for compile feasibility, memory footprint and latency on target SoCs;<br>(4) own MLPerf Edge and partner benchmarking: methodology, reproducibility, priorities;<br>(5) build TensorRT and compiler-stack solutions on Jetson, DRIVE and GPU + ARM; Torch-TRT, MLIR-TRT; proofs of readiness;<br>(6) CUDA, GPU architecture, memory hierarchy; embedded OS internals (QNX/Linux) and C/C++.<br>Stand-out: ML compilers (TVM, MLIR, XLA, Triton) or runtime contributions; production AV perception or planning stacks; VLM + action-expert and end-to-end driving models; MLPerf contributions; ISO 26262/SOTIF familiarity.<br>MS or equivalent; 12+ years in industry, including 8+ in DL optimization, inference or compilation; 5+ years of embedded/edge production inference | 2026-10-14 | 10-28 | https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Systems-Software-Engineer---Deep-Learning-Solutions_JR2014515 | reported |
| J04 | **NVIDIA JR2016777**, Senior Software Performance Engineer – AV (Santa Clara). The team is not named formally; it builds NVIDIA's end-to-end driving applications. Live and **reposted**: start date now 2026-10-16 (the 10-03 ledger had 08-29), "Posted 12 Days Ago", L5/L6. Requirements:<br>(1) latency and throughput of NVIDIA's L2/L3/L4 driving stacks;<br>(2) acceleration strategies across heterogeneous engines;<br>(3) C++ product code with CUDA/ARM NEON under MISRA-style standards;<br>(4) performance analysis, optimization and benchmarking;<br>(5) parallel debugging; systems and architecture knowledge.<br>Stand-out: RTOS and scheduling, DL performance on GPUs. MS or PhD (or equivalent); 12+ years | 2026-10-16 | 10-28 | https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Performance-Engineer---AV-Platform_JR2016777 | reported |
| J05 | **Waymo 7845732**, Senior ML Engineer, Runtime and Serving (ML Optimization team; reports to the Senior Manager of Runtime and Optimization; Mountain View; $213k–$263k; originally published 04-29, updated 10-20). Requirements:<br>(1) build the next onboard ML inference engine for Waymo's foundation model;<br>(2) one runtime and serving system for onboard (real-time latency and memory) and offboard (throughput and concurrency);<br>(3) move workloads to a JAX-native runtime, modifying compilers and runtimes (OpenXLA/PjRT, TensorRT);<br>(4) profiling and benchmarking tools for system bottlenecks;<br>(5) 5+ years of ML-systems software engineering, 5+ years of production C++, 3+ years of Python with DL frameworks; accelerator optimization.<br>Preferred: modifying inference engines (TensorRT, ONNX Runtime, TVM), LLM serving (KV/prefix caching, continuous batching), custom kernels (CUDA/CUDA Tile, Triton, Pallas), zero-copy tensor buffers for multi-model pipelines. BS or MS; PhD preferred | 04-29 / 10-20 | 10-28 | https://careers.withwaymo.com/jobs?gh_jid=7845732 | reported |
| J06 | **Waymo 7374300**, Onboard Software Performance Engineer (C++) (Mountain View; $175k–$215k; updated 10-26). Requirements:<br>(1) per-submodule latency targets and compute budgets on the car;<br>(2) compute use on the car **and in simulation**: CPU/GPU/TPU placement; cutting CPU, latency and RAM for cloud simulation at scale;<br>(3) collision-avoidance reaction latency;<br>(4) hardware/software co-design;<br>(5) performance evaluation, debugging and change management.<br>BS (or equivalent), 3 years in software, 3 years of C++, 3 years of profile-guided optimization. Preferred: SIMD/CUDA, ML performance, evaluation techniques | 2025-11-10 / 10-26 | 10-28 | https://careers.withwaymo.com/jobs?gh_jid=7374300 | reported |
| J07 | **Waymo 8013843**, Staff SWE/TL, Onboard Model Consolidation (AMI team in SIML; $251k–$310k). Requirements:<br>(1) an onboard framework for model integration and inference, both onboard and in simulation;<br>(2) **unify data generation between training-data preparation and inference**, a data-engine line;<br>(3) simulation tooling for testing and evaluating consolidated models;<br>(4) high-performance C++ architecture for ML inference;<br>(5) technical leadership.<br>8+ years; BS or MS (MS/PhD preferred). Preferred: edge or onboard inference on accelerators | 06-17 / 10-20 | 10-28 | https://careers.withwaymo.com/jobs?gh_jid=8013843 | reported |
| J08 | **Waymo 8248394**, Senior SWE, Eval Efficiency & Optimization (team not named in the posting; MV, SF, NYC, Pittsburgh; $213k–$263k; originally published 10-02). Requirements:<br>(1) statistical techniques that give evaluation pipelines better signal for fewer resources;<br>(2) quantitative models with rigorous proof of correctness and statistical efficiency;<br>(3) automation for efficient and scalable evaluation infrastructure;<br>(4) data-processing pipelines; C++/Python/SQL;<br>(5) adapting research to applied problems.<br>BS in a quantitative field; 3–5 years | 2026-10-02 | 10-28 | https://careers.withwaymo.com/jobs?gh_jid=8248394 | reported |
| J09 | **Waymo 8027424**, Staff MLE, Generative Model Performance & Efficiency (the posting places it in Waymo's Simulator team, which builds simulated environments for testing and training the Waymo Driver; MV/NYC; $251k–$310k). Requirements:<br>(1) find bottlenecks in memory bandwidth, compute and communication;<br>(2) FP8/INT4 quantization, pruning, distillation, efficient attention;<br>(3) accelerator-specific optimization through XLA;<br>(4) low-latency, high-throughput serving for generative models;<br>(5) profiling with xprof, Perfetto or Nsight.<br>MS/PhD plus 5+ years; JAX/Flax | 06-29 / 10-20 | 10-28 | https://careers.withwaymo.com/jobs?gh_jid=8027424 | reported |
| J10 | **Wayve 26b5f3a6**, Senior ML Performance Engineer, Inference Optimisation (AI Platform / ML Execution; London; published 10-02; no pay printed). Requirements:<br>(1) profile inference across model graphs, compiler and runtime behaviour, kernel execution and memory movement;<br>(2) implement compiler, runtime or kernel optimizations: fusion, scheduling, quantisation-aware performance, custom kernels;<br>(3) benchmarking and regression tests across models, devices and releases;<br>(4) edge targets **NVIDIA Orin/Thor** and Qualcomm;<br>(5) hands-on with TensorRT, CUDA, QNN, Triton or OpenCL, under latency, memory, bandwidth, power and thermal limits.<br>Nice to have: embedded benchmarking, Python/C++. **No years or degree line** | 2026-10-02 | 10-28 | https://jobs.ashbyhq.com/wayve/26b5f3a6-43ca-4168-a1b1-3926d5ffb6bd | reported |
| J11 | **Wayve 8c1987cf**, Staff ML Performance Engineer, Performance Architecture (Sunnyvale; $370,040–$394,900 plus equity; published 10-02). Requirements:<br>(1) profile training and cloud inference with system and kernel profilers;<br>(2) raise MFU, throughput and utilization (parallelism, compilation, mixed precision, caching);<br>(3) reusable cross-target optimizations (kernels, data loaders, Triton);<br>(4) benchmarking tools that catch regressions.<br>10+ years; BS/MS or equivalent. Nice to have: inference serving, Nsight Systems, GPU kernels | 2026-10-02 | 10-28 | https://jobs.ashbyhq.com/wayve/8c1987cf-2f02-4753-a93e-9d17c36b5a53 | reported |
| J12 | **Zoox 2ed8a665**, Senior/Staff SWE, ML Inference Optimization (Software Systems / ML Platform; Foster City; $254k–$350k; Lever `createdAt` 2023-10-28). The 10-03 ledger listed this ID as "ML Performance Optimization", so it appears retitled. Requirements:<br>(1) lead training or inference optimization so VLM, VLA and foundation models deploy on the robotaxi;<br>(2) GPU inference with TensorRT;<br>(3) Nsight or PyTorch Profiler;<br>(4) model compression;<br>(5) Python and C++.<br>Bonus: 10+ years, including 4+ on large-scale training or inference platforms | 2023-10-28 | 10-28 | https://jobs.lever.co/zoox/2ed8a665-cee8-4d70-bcb1-e96b6214b371 | reported (retitle: inferred) |
| J13 | **Zoox bc11276c**, Software Engineer, ML Inference Optimization (Foster City; $177k–$270k; created 2026-05-28). Same duties as J12 at a junior level: 4+ years, including 2+ on training or inference platforms; TensorRT; profilers; Python or C++ | 2026-05-28 | 10-28 | https://jobs.lever.co/zoox/bc11276c-8db7-426e-9d00-d41c2097723a | reported |
| J14 | **Zoox "Senior AI Inference Engineer"** (c88c8b02, in the 10-03 ledger) is **absent** from the live Lever feed on 10-28 (228 postings) | 10-03 → 10-28 | 10-28 | https://api.lever.co/v0/postings/zoox?mode=json | reported |

### Q5 — What 2026 releases broke or outran engines and kernels

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| R01 | vLLM PR #62957, "[Model] Add Alpamayo 2 Super (reasoning model)", **merged 10-22**:<br>• vLLM serves Super's Qwen3-VL reasoning model (VQA, grounding, CoC with meta-actions, auto-labels);<br>• **the trajectory expert in the same checkpoint is not loaded**;<br>• the full policy is left to vLLM-Omni #9133;<br>• validated on 1× H200 NVL | 10-21 → 10-22 | 10-28 | https://github.com/vllm-project/vllm/pull/62957 | reported |
| R02 | vLLM-Omni PR #9133, "[Model] Add Alpamayo 2 Super" (**draft**, open, `high priority`, assigned to princepride, 11 comments, updated 10-26).<br>**Design:** two stages on one GPU. The Qwen3-VL reasoning stage hands its KV cache (~0.8 GB per request) through NIXL to a flow-matching expert stage that runs 10 Euler steps for a 64-waypoint trajectory at 10 Hz.<br>**Scope:** BF16, TP = 1, eager; no CUDA graphs, no FP8.<br>**Stated need:** NVIDIA's AV data factory.<br>**Speed tuple:**<br>• baseline: the NVlabs/alpamayo2 reference (transformers 4.57.1);<br>• cameras [0,1,2,6], K = 6 trajectories per scene, 10 events;<br>• 1× H200 NVL, vLLM 0.31.0: median 2.20 s per scene vs 2.01 s for the reference; 0.76 s per request one at a time;<br>• H200 SXM with a slower host CPU: 1.39 s per request;<br>• expert parity within 3.6 cm.<br>**Missing:** no trace. Datacenter GPU, not Thor | 10-19 → 10-26 | 10-28 | https://github.com/vllm-project/vllm-omni/pull/9133 | reported |
| R03 | FlashRT #223 asks whether Alpamayo 1.5 will be supported on AGX Thor (closed 10-20). The FlashRT maintainer (LiangSu8899, MEMBER) said on 10-12 that 1.5 is on their list, with **two reference versions**:<br>(1) the official Alpamayo 1.5 with pure compute and runtime optimizations, **no behaviour change**;<br>(2) a FlashDrive-based version, "especially on Jetson Thor".<br>#229 (an early baseline port) merged 10-20 (ledger). No newer FlashRT Alpamayo PR through #243 (10-28). FlashRT merges outside PRs in 1–3 days; for example #234, Qwen3-4B on Jetson Thor, 10-21 → 10-24 | 10-12 → 10-28 | 10-28 | https://github.com/flashrt-project/FlashRT/issues/223 ; https://github.com/flashrt-project/FlashRT/pulls | reported |
| R04 | ModelOpt's real-quant GEMM:<br>• #1895 ("RealQuantLinear: No real-quant GEMM found", 16 comments) closed 09-03;<br>• the same warning came back in #2593 (open, 09-29);<br>• open fix PRs: #2550 and #2331 (NVFP4 availability check) and #2673 (re-register the FP8 per-tensor real-quant GEMM).<br>F01 ties this warning to slow fake-quant Alpamayo 1.5 on Jetson AGX Thor. **Fake quantization is not a speed-up** | 07-02 → 10-26 | 10-28 | https://github.com/NVIDIA/Model-Optimizer/issues/1895 ; …/issues/2593 ; …/pull/2550 ; …/pull/2673 | reported |
| R05 | SGLang PR #21059, "Add a new VLA model Alpamayo-R1", closed 08-27 | 03-21 → 08-27 | 10-28 | https://github.com/sgl-project/sglang/pull/21059 | unverified (merge status not checked) |
| R06 | vLLM #42100, "[Feature]: Adding VLA models support into vLLM", closed 09-28 with 11 comments | 05-08 → 09-28 | 10-28 | https://github.com/vllm-project/vllm/issues/42100 | unverified (resolution not read) |
| R07 | Hugging Face on 10-28: still only three nvidia Alpamayo models (R1-10B, 1.5-10B, 2-Super). `nvidia/Alpamayo2-Edge`, `Alpamayo-2-Edge` and `Alpamayo2-Edge-2B` return **401**. The newest nvidia model is `nvidia/Metatron-v1` (10-27) | 2026-10-28 | 10-28 | https://huggingface.co/api/models?author=nvidia&search=alpamayo ; https://huggingface.co/api/models?author=nvidia&sort=createdAt&direction=-1&limit=15 | reported |
| R08 | 2026 breakage collected above:<br>• Cosmos-Reason2-2B (Qwen3-VL) NVFP4 build crash on Jetson AGX Thor (E04);<br>• Qwen3-VL block-FP8 export (E10);<br>• the 0.11.0 Cosmos3-Edge patch-layout regression (E05);<br>• SM110 Myelin miscompile at batch 1 (E06);<br>• FA2/FA3 missing on sm_110 in vLLM (K09);<br>• CUDA 13's sm_101 → sm_110 renumbering leaving Inductor's Blackwell gates off (K16);<br>• NVFP4 MoE illegal memory accesses on Thor (K10, K12);<br>• SGLang Thor auto-backend crash, fixed (K22);<br>• Edge-LLM has no Alpamayo 1.5 (E01);<br>• flow-matching export unanswered (A06);<br>• the Alpamayo 1.5 navigation text is not plumbed into public AlpaSim (A16) | 07 → 10-26 | 10-28 | (see rows) | reported (grouping: inferred) |
| R09 | No NVIDIA-published latency for any Alpamayo model on **Jetson AGX Thor** was found on 2026-10-28. Searched:<br>• Edge-LLM releases and issues since 08-01;<br>• alpamayo1.5, alpamayo, recipes and alpamayo2 issues;<br>• the NVIDIA forum queries in F16 plus the DRIVE AGX Thor listing;<br>• the NVIDIA MLPerf Edge blog.<br>The only engine numbers on Jetson AGX Thor are FlashRT's for the 10B (ledger) | — | 10-28 | (search set above) | reported (absence) |

### Q6 — Do the target repos accept outside PRs?

| ID | Claim | Source date | Checked | Link | Label |
|---|---|---|---|---|---|
| C01 | **TensorRT-Edge-LLM CONTRIBUTING:** clang-format; concise PRs; **DCO sign-off required**, so unsigned commits are not accepted. The file is byte-identical to the 10-27 copy | — | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CONTRIBUTING.md | reported |
| C02 | **TensorRT-Edge-LLM merge record:** all 17 merged PRs in its history are release dumps or CI PRs from NVIDIA handles. Outside fixes are absorbed into releases instead:<br>• #195 was "being introduced in release/0.10.1";<br>• #124 was "already fixed" by 0.10.0;<br>• #148 was superseded by the CuTe DSL FMHA migration.<br>Since 08-01, 33 PRs from non-NVIDIA handles: 29 open, 4 closed, 0 merged | 2025-12-24 → 10-20 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/pulls?q=is%3Apr+is%3Amerged ; …/pull/195 ; …/pull/124 ; …/pull/148 | reported (handle classification: inferred) |
| C03 | **Outside-PR CI path in progress** at TensorRT-Edge-LLM:<br>• #227 (merged 10-06) enables copy-pr-bot for ready PRs;<br>• #231 tests copy-pr-bot and CPU-runner access (open);<br>• #261 ports pre-commit checks to GitHub Actions (open);<br>• #262 waits for trusted internal-pipeline status (open; draft auto-sync disabled).<br>This could open a merge path, but none has produced a merge | 09-29 → 10-27 | 10-28 | https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/227 ; …/pull/231 ; …/pull/261 ; …/pull/262 | reported / inferred |
| C04 | **alpamayo1.5 CONTRIBUTING:** DCO sign-off is required. With no CI/CD in place, a PR is accepted only after manual testing by the developer and/or an Alpamayo researcher. 9 merged PRs ever, by collaborators (super-anova, YurongYou) and contributors (brandonpelfrey, the author of the NVIDIA-reported CUDA-graph PR #36; elliotdub; BorisIvanovic) | — | 10-28 | https://github.com/NVlabs/alpamayo1.5/blob/main/CONTRIBUTING.md ; https://github.com/NVlabs/alpamayo1.5/pulls?q=is%3Amerged | reported (affiliations of CONTRIBUTOR accounts: unverified) |
| C05 | **alpamayo-recipes CONTRIBUTING:** DCO required; basic CPU-only CI runs on PRs. The README carries a "contributions welcome" badge. 21 merged PRs ever, all by collaborators or contributors (annz96, zewenli98 and BorisIvanovic are CONTRIBUTOR accounts). Outsider PRs: #37 (code of conduct; open since 09-16; no response) and #36 (closed unmerged 09-12). Outsider bug reports do get fixed (A11) | — | 10-28 | https://github.com/NVlabs/alpamayo-recipes/blob/main/CONTRIBUTING.md ; …/pull/37 ; …/pull/36 | reported (affiliations: unverified) |
| C06 | **FlashInfer CONTRIBUTING:** no DCO or CLA text found. Rules:<br>• use the PR template;<br>• **performance PRs must report before/after numbers with GPU and problem sizes**;<br>• AI-assisted PRs are accepted, but may be rejected if the author cannot explain the rationale;<br>• an **Experimental Track** needs a named owner, a tracking issue and a graduation plan (default four weeks), and is JIT-only with narrower review;<br>• `!claim` self-assigns an unassigned issue;<br>• public GitHub Actions CI plus NVIDIA's internal GitLab CI | — | 10-28 | https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md | reported |
| C07 | **CUTLASS:** no `CONTRIBUTING.md` at the repo root (404; the root has `CONTRIBUTORS.md`, `EULA.txt`, `LICENSE.txt`). 39 PRs merged since 08-01, many from CONTRIBUTOR accounts. Merge latency varies widely:<br>• #3725: 3 days (10-09 → 10-12);<br>• #3637: 31 days;<br>• #3048: 166 days (02-21 → 08-06);<br>• #2851: 255 days (2025-12-06 → 2026-08-18).<br>#3308 (K18) was effectively superseded by #3637 | 08-01 → 10-20 | 10-28 | https://github.com/NVIDIA/cutlass/pulls?q=is%3Apr+is%3Amerged+merged%3A%3E%3D2026-08-01 ; https://github.com/NVIDIA/cutlass | reported (durations: inferred) |
| C08 | **AlpaSim (ledger plus delta):** it accepts outside PRs; none has merged on `main` since 08-17. New outside PR #228 (10-28, A15) | 10-28 | 10-28 | https://github.com/NVlabs/alpasim/pull/228 | reported |

---

## 3. Deltas vs the 10-27 ledgers

| # | Earlier ledger said | Delta found 2026-10-28 | IDs |
|---|---|---|---|
| D1 | `05` §1: JR2016777 start date 08-29, "not re-checked" | Reposted: start date 2026-10-16, "Posted 12 Days Ago"; same text, L5/L6 | J04 |
| D2 | `05` §1 has no JR2014515 | **JR2014515 = Senior Systems Software Engineer – Deep Learning Solutions** (Santa Clara; start 10-14; 224,000–356,500 USD; MS or equivalent; 12+ years, 8+ in DL optimization, 5+ embedded). It is the closest NVIDIA line-match to Thor serving | J03 |
| D3 | 10-03 ledger: Zoox Senior AI Inference Engineer (c88c8b02) | Gone from the live feed. Zoox's current inference roles are 2ed8a665 (apparently retitled) and bc11276c | J12–J14 |
| D4 | L1/L7: Edge-LLM v0.11.0 latest; PR #260 open; Alpamayo 1.5 unsupported | Unchanged. New: #263 (10-26). **#130's August promise (07-30) is documented and unmet; follow-ups unanswered.** The roadmap #32 has no Alpamayo 1.5 line; no H2 roadmap. #260 still has only the vetting bot. Non-CI commits: none since 09-29 | E01, E14, E16–E19 |
| D5 | (absent) Edge-LLM sm_110 and NVFP4 defects | #216/#225 (Cosmos-Reason2-2B NVFP4 TRT crash on Jetson AGX Thor; workaround 9–10 % slower than ONNX); #232/#251 (0.11.0 Cosmos3-Edge regression); #151 (SM110 Myelin miscompile); #164; #250 | E04–E08 |
| D6 | (absent) Edge-LLM merge mechanics | 17 of 17 merges are NVIDIA release or CI PRs; outside fixes absorbed into releases; 0 of 33 outside PRs merged since 08-01; copy-pr-bot and GitHub Actions path being wired | C02, C03 |
| D7 | L1: AlpaSim no commits since 10-20 | Still true. New: #227 (10-28; SIGKILL teardown), PR #228 (10-28; outsider), #74 closed 10-28 (Alpamayo 1.5 navigation-text CFG not plumbed in public AlpaSim, so run with CFG navigation off) | A14–A18 |
| D8 | L1: recipes #42/#43/#45 open; #44 opened 10-21 | #44 reordered 10-28; new bug #46 (10-28). The Super RL recipe was promised "very shortly" on 09-30 and has no PR on 10-28 | A07–A09 |
| D9 | (absent) | vLLM merged Super's reasoning-only path (10-22). vLLM-Omni #9133 (draft, high priority) serves the full two-stage policy in BF16 eager on H200 and says NVIDIA's AV data factory needs it | R01, R02 |
| D10 | Ledgers record FlashRT #229's numbers | **New:** the FlashRT maintainer's public plan (10-12) for an official-1.5, behaviour-preserving runtime version and a FlashDrive version on Jetson Thor. No new Alpamayo PR through 10-28 | R03 |
| D11 | (absent) Kernel-library sm_110 state | FlashInfer CAKE SM110 decode routes (K03–K05) and 726 merges since 10-20 (K07); vLLM SM110 duplicates and unmerged fixes (K09–K13); PyTorch Inductor Thor gap with a PR offered (K16); CUTLASS SM110 gate merged (K17), #3308 superseded (K18); Triton and SGLang items (K21, K22) | K01–K25 |
| D12 | (absent) Forum threads after 10-25 | 10-14 throttling under Alpamayo 1.5 BF16 at 120 W (expected, per staff); 10-19 Super camera-profile question (no reply); 10-28 ZETIC VLA dashboard on Jetson AGX Thor | F05, F07, F08 |
| D13 | L1 T5 / L7 1.6: newest DriveOS unverified (7.2.5 vs 7.2.6) | Forum evidence only: 7.2.5 is the newest announced and still referenced on 10-22; no 7.2.6 topic among 100 since 06-01. Still unresolved | F14 |
| D14 | L1 M2: Edge 401 (10-27) | Still 401 on 10-28; HF Alpamayo list unchanged | R07 |
| D15 | (absent) NVlabs triage pattern | Bulk closes on 10-06, 10-07 and 10-12 across alpamayo and alpamayo1.5, including the Thor install thread | A19 |

---

## 4. Raw opportunity and kill signals (unscored)

### 4a. Wanted work that nobody owns (no assignee, no NVIDIA PR, no staff answer)
- **Exporting a flow-matching action expert to an edge engine.** alpamayo #112 has had no reply since 07-23. vLLM-Omni has a datacenter two-stage path only (A06, R02).
- **Quantized Alpamayo export to Edge-LLM** (#134) and the **Alpamayo 1.5 port** (#130). Both are unassigned and slipping. #130 also sits in 4b, because NVIDIA promised it (E01, E02).
- **Cosmos-Reason2-2B NVFP4 on Jetson AGX Thor** (#216). The root cause is in the closed TensorRT compiler. An outsider owns the workaround (#225), which is slower than ONNX. NVIDIA has been silent since 09-20 (E04).
- **Edge-LLM runtime features with outside PRs but no review:**
  - continuous batching and chunked prefill (#215);
  - seeded sampling, the Philox offset (#211/#212);
  - multimodal calibration (#186; volunteer, no reply);
  - Cosmos3-Edge patch layout (#251).

  (E05, E11–E13)
- **vLLM on Jetson AGX Thor:**
  - startup without FA2/FA3 (#63267);
  - `cooperative_topk`, with duplicate PRs unmerged for 49 days;
  - Thor Mamba configs, two PRs unmerged since June and August.

  This queue is crowded and slow to merge (K09, K11, K13).
- **Unanswered model-behaviour reports:**
  - stationary-lead collisions in Alpamayo 1.5 (a1.5 #40/#39; NVIDIA closed the related AlpaSim #224 as not actionable);
  - Alpamayo 1.5 instability in AlpaSim (forum 384744);
  - Super's visual-blackout narration (alpamayo2 #9);
  - Super's 7- vs 6-camera profile (forum 386270).

  (A02, A13, F06, F07)
- **Measurement methodology on Thor.** NVIDIA staff say there is no official CUTLASS TFLOPS for Thor (F11). No public, versioned latency for any Alpamayo model on Jetson AGX Thor was found (R09).
- **Stale but resolved:** AlpaSim #17 (Blackwell sensorsim) works with NRE GA 26.02+ on one user's report. Not an opening (A17).

### 4b. Work NVIDIA or adjacent maintainers have claimed or promised (scoop risk)
- **Edge-LLM Alpamayo 1.5** was planned for August. Unmet, but NVIDIA-owned once it lands (E01).
- **The 2B-student path on DRIVE:** recipes PR #42 (distillation), #45 (5 Hz inference for the 2B students through a forked Edge-LLM) and #43 (speculative-decoding drafts). NVIDIA-owned and open (A10, ledger).
- **Super open-loop RL recipe:** called imminent on 09-30; no PR by 10-28 (A09).
- **FlashInfer SM110 attention:** experimental GQA decode (#5051) and tcgen05 attention (#5292), named owner, plus continuous CAKE SM110 XQA/GQA output and an open invitation to request kernels (K01–K05). A hand-written sm_110 decode or attention kernel would be judged against automated output.
- **FlashRT:** official Alpamayo 1.5, behaviour-preserving runtime on Jetson Thor, plus a FlashDrive variant. The maintainer merges Jetson Thor ports in days. **Direct overlap with path 2's "measured win over the strongest baseline that does not change behaviour"** (R03).
- **vLLM-Omni Super full-policy serving** (#9133). High priority, framed as NVIDIA's data-factory need. Datacenter only (R02).
- **PyTorch Inductor Thor gating:** PR #202001 already offered (K16).
- **Third-party measurement artifacts:**
  - ZETIC's VLA-on-Jetson dashboard, with TensorRT "soon" (F08);
  - NVIDIA's own MLPerf Edge results on Jetson AGX Thor, owned by the JR2014515 team (F09, J03).

### 4c. Which posting lines each artifact kind would answer (mapping only, not a score)
- **Path 2, engineering (2B student on Jetson AGX Thor at 10 Hz with a behaviour-preserving win):**
  - **JR2014515:** operator-level review; kernel-trace diagnosis; flow-matching decoder and multi-camera tokenizer feasibility, memory and latency on SoCs; benchmark methodology and reproducibility; TensorRT, Torch-TRT and MLIR-TRT on Jetson. Stand-outs: VLM + action-expert familiarity; Triton/MLIR.
  - **JR2025038:** timing and state contracts; latency, throughput, memory, determinism and power; dev-vs-target discrepancy; C++/CUDA/TensorRT.
  - **JR2016777:** latency and throughput; C++/CUDA; benchmarking.
  - **Wayve Inference Optimisation:** profiling graph → kernel → memory; fusion, scheduling, quantisation-aware performance; regression benchmarks; Orin/Thor.
  - **Zoox ML Inference Optimization:** TensorRT; Nsight; compression.
  - **Waymo Runtime & Serving:** onboard latency and memory; modifying TensorRT-class runtimes; custom kernels; zero-copy multi-model pipelines. Waymo leans JAX/TPU.
  - **Waymo Onboard Software Performance:** latency budgets; profile-guided optimization.

  (J02–J06, J10, J12, J13)
- **Path 1, research paper (closed-loop quality vs latency, CoC budget, flow steps or precision):**
  - **JR2010863 stand-out:** simulation and closed-loop evaluation of end-to-end models; measuring simulation quality (`05` §3 X2, D3).
  - **Waymo Eval Efficiency & Optimization:** statistically efficient evaluation with rigorous proof.
  - **Waymo Generative Model Performance & Efficiency:** FP8/INT4 quantization, distillation, profiling.
  - **JR2014515:** latency on target SoCs for new architectures, partly.
  - **JR2025038:** driving-quality regression metrics.
  - **JR2014651 and JR2005596:** per the ledger.

  (J01, J03, J08, J09)
- **Path 3, shadow mode (open-loop only; it answers data-engine lines, not closed-loop evaluation lines):**
  - **JR2025038:** in-vehicle testing and driving-data analysis; dev-vs-vehicle behaviour discrepancies.
  - **Waymo Onboard Model Consolidation:** unifying training-data generation with inference-time data generation.
  - **JR2022179:** fleet-log mining and data flywheels (10-03 ledger).

  (J02, J07)

---

## 5. Gaps

- **Alpamayo forum subcategories.** The category-listing JSON failed for ids 766–783. Search surfaced 11 of the 16 topics the category metadata counts. Unseen: 1 in AlpaSim, 3 in PhysicalAI-AV Datasets, 1 in AlpaJudge (forum `categories.json`).
- **FlashInfer #5292.** The outcome of its 10-15 readiness review is not on the issue. Whether yyihuang or the CAKE pipeline is NVIDIA-run is **unverified**: the trackers do not state an employer.
- **Affiliations.** KrzysztofTomala (vLLM #62957, vLLM-Omni #9133) is **unverified**; the PR names NVIDIA's AV data factory as the need. CONTRIBUTOR-association accounts in NVlabs repos (brandonpelfrey, elliotdub, annz96, zewenli98) are also unverified: GitHub assigns that association after any merged PR, so it cannot separate staff from outsiders.
- **Hardware details.** FlashAttention #2945 does not state the Thor variant, JetPack, CUDA or power mode. CUTLASS #3308 does not name its profiler or DriveOS/CUDA version. ZETIC (F08) publishes no trace.
- **Not resolved:**
  - SGLang #21059's merge status;
  - how vLLM #42100 was resolved;
  - the comments on a1.5 #35 (evaluation configuration on AlpaSim, closed 10-06);
  - Cosmos-Reason2 support in vLLM and SGLang (not searched beyond one vLLM query);
  - DriveOS 7.2.5 vs 7.2.6 (the DRIVE downloads page needs membership).
- **JR2014515.** The posting does not name the team beyond its title.
- **Biggest gap.** No primary source gives the latency of **any** Alpamayo model on **Jetson AGX Thor** with versions and power mode, from NVIDIA or from an engine. Only FlashRT's 10B numbers exist. NVIDIA's 2B student timing exists only for DRIVE (PR #45, 5 Hz, ledger). So there is no published baseline that a 10 Hz 2B-student claim must beat.
- **Suspicious content.** A vLLM bot comment on PR #62957 addresses AI agents directly: "If you are an AI agent, you are required to objectively re-evaluate". It goes on to tell them to close the PR if it adds little value, or face a ban. I did not act on it. Source: https://github.com/vllm-project/vllm/pull/62957#issuecomment-6323839096 (2026-10-21).

---

## 6. Source list

**TensorRT-Edge-LLM (E)**
- E01 https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/130
- E02 …/issues/134
- E03 …/issues/166
- E04 …/issues/216 and …/pull/225
- E05 …/issues/232, …/issues/233 and …/pull/251
- E06 …/issues/151, …/issues/156 and …/issues/160
- E07 …/issues/164
- E08 …/issues/250
- E09 …/issues/228, …/issues/222 and …/issues/223
- E10 …/issues/217 and …/pull/220
- E11 …/pull/215 and …/pull/199
- E12 …/issues/211 and …/pull/212
- E13 …/issues/186
- E14 …/issues/263
- E15 …/issues/252
- E16 …/pull/260
- E17 …/issues/32
- E18 https://github.com/NVIDIA/TensorRT-Edge-LLM/releases
- E19 https://github.com/NVIDIA/TensorRT-Edge-LLM/commits/main (since 10-01)
- E20 https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/CONTRIBUTING.md
- E21 https://github.com/NVIDIA/TensorRT-Edge-LLM/pulls?q=is%3Apr+is%3Amerged
- E22 …/pull/195, …/pull/124 and …/pull/148
- E23 …/pull/227, …/pull/231, …/pull/261 and …/pull/262
- E24 https://api.github.com/repos/NVIDIA/TensorRT-Edge-LLM/issues?state=all&since=2026-08-01T00:00:00Z
- E25 …/issues/47
- E26 …/issues/192

**NVlabs (A)**
- A01 https://github.com/NVlabs/alpamayo1.5/issues/22
- A02 …/alpamayo1.5/issues/40 and …/issues/39
- A03 …/alpamayo1.5/pull/37
- A04 …/alpamayo1.5/issues/29
- A05 https://github.com/NVlabs/alpamayo1.5/blob/main/CONTRIBUTING.md
- A06 https://github.com/NVlabs/alpamayo/issues/112, …/issues/121, …/pull/122 and …/issues/110
- A07 https://github.com/NVlabs/alpamayo-recipes/issues/46
- A08 …/alpamayo-recipes/pull/44
- A09 …/alpamayo-recipes/issues/33
- A10 …/alpamayo-recipes/issues/20 (PRs #42, #43, #45)
- A11 …/alpamayo-recipes/issues/35, …/issues/29 and …/pull/40
- A12 https://github.com/NVlabs/alpamayo-recipes/blob/main/CONTRIBUTING.md
- A13 https://github.com/NVlabs/alpamayo2/issues/19 and …/issues/9
- A14 https://github.com/NVlabs/alpasim/issues/227
- A15 …/alpasim/pull/228
- A16 …/alpasim/issues/74
- A17 …/alpasim/issues/17
- A18 https://api.github.com/repos/NVlabs/alpasim
- A19 https://github.com/NVlabs/alpamayo1.5/issues?q=is%3Aissue+is%3Aclosed

**Kernel libraries (K)**
- K01 https://github.com/flashinfer-ai/flashinfer/issues/5051
- K02 …/issues/5292
- K03 …/issues/4254
- K04 …/issues/6594
- K05 …/pull/5597, /5658, /5674, /5872, /5929, /6944, /6945, /6981, /6982, /7116 and /7293
- K06 …/pull/5695, /6295, /5882, /4670 and /3734
- K07 https://github.com/flashinfer-ai/flashinfer/pulls?q=is%3Apr+is%3Amerged+merged%3A%3E%3D2026-10-20
- K08 https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md
- K09 https://github.com/vllm-project/vllm/issues/63267
- K10 …/issues/62149 and …/pull/62275
- K11 …/issues/56083, …/pull/56084, …/pull/56102 and …/issues/58954
- K12 …/issues/55406 and …/pull/55366
- K13 …/pull/53337, …/pull/44590, …/pull/60843 and …/issues/45260
- K15 …/issues/59725; https://github.com/sgl-project/sglang/issues/42176
- K16 https://github.com/pytorch/pytorch/issues/201193, …/pull/202001, …/pull/200432, …/issues/192209 and …/pull/192386
- K17 https://github.com/NVIDIA/cutlass/pull/3637 and …/issues/3576
- K18 https://github.com/NVIDIA/cutlass/pull/3308
- K19 …/pull/3360, …/pull/3323, …/issues/3425 and …/pull/3283
- K20 https://github.com/NVIDIA/cutlass (root listing) and …/pulls?q=is%3Apr+is%3Amerged+merged%3A%3E%3D2026-08-01
- K21 https://github.com/triton-lang/triton/pull/11923, …/pull/10462 and …/pull/11021
- K22 https://github.com/sgl-project/sglang/pull/41765, …/issues/41764, …/pull/45520, …/pull/43694, …/pull/37641 and …/issues/36131
- K23 https://github.com/Dao-AILab/flash-attention/pull/2945, …/pull/2610 and …/pull/2590
- K24 GitHub search API, https://api.github.com/search/issues (queries in Q2)

**Engines, releases, models (R)**
- R01 https://github.com/vllm-project/vllm/pull/62957
- R02 https://github.com/vllm-project/vllm-omni/pull/9133
- R03 https://github.com/flashrt-project/FlashRT/issues/223 and https://github.com/flashrt-project/FlashRT/pulls
- R04 https://github.com/NVIDIA/Model-Optimizer/issues/1895, …/issues/2593, …/pull/2550, …/pull/2331 and …/pull/2673
- R05 https://github.com/sgl-project/sglang/pull/21059
- R06 https://github.com/vllm-project/vllm/issues/42100
- R07 https://huggingface.co/api/models?author=nvidia&search=alpamayo and https://huggingface.co/api/models?author=nvidia&sort=createdAt&direction=-1&limit=15
- R08 grouping of the rows above
- R09 absence search (listed in that row)

**NVIDIA forums and blog (F)**
- F01 https://forums.developer.nvidia.com/t/382409
- F02 …/t/382647
- F03 …/t/382501 and …/t/382500
- F05 …/t/385881
- F06 …/t/384744
- F07 …/t/386270
- F08 …/t/387191
- F09 https://developer.nvidia.com/blog/tensorrt-edge-llm-completes-the-mlperf-edge-agentic-benchmark-6-4x-faster-on-jetson-agx-thor/ and …/t/383470
- F10 …/t/377887
- F11 …/t/372014
- F12 …/t/375013
- F13 …/t/375838
- F14 …/t/379371 and https://forums.developer.nvidia.com/c/autonomous-vehicles/drive-agx-thor/drive-agx-thor-general/742
- F15 …/t/383269
- F16 https://forums.developer.nvidia.com/search.json (queries in Q3)

**Job postings (J), via each employer's public board API**
- J01 NVIDIA Workday JR2010863
- J02 JR2025038
- J03 JR2014515
- J04 JR2016777 (URLs in the J rows; detail JSON under `https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/job/...`)
- J05–J09 Waymo: https://careers.withwaymo.com/jobs?gh_jid=7845732, 7374300, 8013843, 8248394 and 8027424 (API `https://boards-api.greenhouse.io/v1/boards/waymo/jobs/<id>`)
- J10–J11 Wayve: https://jobs.ashbyhq.com/wayve/26b5f3a6-43ca-4168-a1b1-3926d5ffb6bd and https://jobs.ashbyhq.com/wayve/8c1987cf-2f02-4753-a93e-9d17c36b5a53
- J12–J14 Zoox: https://jobs.lever.co/zoox/2ed8a665-cee8-4d70-bcb1-e96b6214b371, https://jobs.lever.co/zoox/bc11276c-8db7-426e-9d00-d41c2097723a and the feed https://api.lever.co/v0/postings/zoox?mode=json

**Contribution policy (C)**
- C01 = E20
- C02 = E21/E22
- C03 = E23
- C04 = A05
- C05 = A12
- C06 = K08
- C07 = K20
- C08 = A15
