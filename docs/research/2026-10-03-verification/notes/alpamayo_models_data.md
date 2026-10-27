# Alpamayo model family and open driving data: claim-by-claim verification ledger (state of the world 2026-10-03)

Docs checked (read-only): `docs/01_SYSTEM_DESIGN.md` (§2 map, §3 C1–C2), `docs/02_PHYSLOOP_PLAN_v2.md` (§0, plus §3–§4 and §7 where they name Alpamayo assets), `docs/03_MASTERY_PLAN.md` (§1 P1/P2/P4, §2 nanoAlpamayo acceptance table, §4 Stages 0/2), `docs/04_CORE_PLAN_TECHNICAL.md` ("State of the art", "First-principles workload analysis", "Tensor contract", "Eval lock", "Sources").

**Status legend.** VERIFIED: the primary source states the same value. CORRECTED: the primary source states a different value, or the claim's context or attribution is wrong (the correct value is given). NOT FOUND: I looked where the claim should be and it is not there. UNVERIFIABLE: no accessible primary source. Rows 10.8–10.10 also use UNVERIFIABLE, marked "out of scope": claims belonging to sibling researchers that were not checked here. INFO: facts the task asked for that are not doc claims (no status). Many VERIFIED rows carry a plan-relevant caveat in the note column, e.g. 1.4, 1.8, 1.12, 4.7, 4.12, 6.5. Read the notes, not just the status.

**Evidence convention.** Numeric table cells and metadata field values are reproduced exactly, shown in `code`. Prose passages are given as close paraphrases with an exact location (section, table, or revision-pinned URL). A quotation limit applies, so the ledger uses one verbatim quote, in row 6.1. Every row was accessed on **2026-10-03**. Hugging Face (HF) card URLs are pinned to the commit SHA that was read. Third-party or secondary sources are labelled as such.

---

## 1. Alpamayo-R1 paper (arXiv 2511.00088): identity, versions, architecture, action representation, flow steps, vision tokenization

### Takeaway
The paper exists as cited: NVIDIA; Yan Wang plus 41 co-authors; v1 30 Oct 2025, v2 7 Jan 2026. v2 says AR1 is also called "Alpamayo 1". All action-side claims check out: 64 waypoints over 6.4 s at 10 Hz; unicycle (acceleration, curvature) controls; 128 discrete training tokens; a flow-matching expert that reads the VLM KV cache; default δt = 0.1, so 10 Euler steps. Three corrections:
- **The paper is inconsistent on step count.** Its own latency table and decoding ablation use 5 steps.
- **The paper puts the expert at 2B; the model cards say 2.3B.**
- **The vision-token figures are not from the paper.** The paper uses 448×280 images, 160 tokens per image, and two front cameras. "320×576, 180 tokens/image, 2,880 tokens" is a correct derivation for the *released* Alpamayo 1/1.5 checkpoints, but NVIDIA never states 180 or 2,880.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence (exact cell/field, or paraphrase + location) | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 1.1 | "Alpamayo-R1 … arXiv 2511.00088" (01 C1 source; 02 §7; 03 §7) | VERIFIED | abs page: authors `NVIDIA: Yan Wang and 41 other authors`; history `[v1] Thu, 30 Oct 2025`, `[v2] Wed, 7 Jan 2026`; cs.RO | 42 authors, all NVIDIA; corresponding submitter Yan Wang | https://arxiv.org/abs/2511.00088 | 2026-10-03 |
| 1.2 | 04 Sources cites `arxiv.org/html/2511.00088v2` | VERIFIED | v2 is still the latest version. v2 front matter (paraphrase): since NVIDIA Alpamayo launched at CES 2026, AR1 is also called Alpamayo 1 | v1 has no §6.4 "Public Benchmark Evaluation" (Table 10), and its latency table is **Table 13**, not 14 | https://arxiv.org/html/2511.00088v2 ; https://arxiv.org/html/2511.00088v1 | 2026-10-03 |
| 1.3 | C1: Alpamayo 1.5 has "an ~8.2B VLM (Cosmos-Reason2)" | VERIFIED (card) | A1.5 card fields: `Backbone: 8.2B parameters`; `base_model: nvidia/Cosmos-Reason2-8B`. Config: `vlm_name_or_path: nvidia/Cosmos-Reason2-8B`, `vlm_backend: qwenvl3` | The paper (§3.1) names only "Cosmos-Reason", with no size; its LingoQA table names Cosmos-Reason-7B. The Cosmos-Reason2-8B checkpoint itself has `8,767,123,696` params, base `Qwen/Qwen3-VL-8B-Instruct` | https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md ; …/config.json (same SHA) ; https://huggingface.co/api/models/nvidia/Cosmos-Reason2-8B | 2026-10-03 |
| 1.4 | C1: "a ~2.3B flow-matching action expert" (cites paper and HF card) | VERIFIED (cards); paper says otherwise | A1 and A1.5 cards: `Action Expert: 2.3B parameters`. AR1 v2 §6.4 (paraphrase): AR1-10B pairs Cosmos-Reason with a **2B**-parameter diffusion trajectory decoder. Config `expert_cfg`: `hidden_size 2048, intermediate_size 8256, num_attention_heads 16, head_dim 128` | Paper: 2B; cards: 2.3B. Cite the card for the release | A1.5 card (above); https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/dd4a24cacefc9a6477a6dfc7354de2443401409d/README.md ; https://arxiv.org/html/2511.00088v2 §6.4 | 2026-10-03 |
| 1.5 | C1: the expert "reads the VLM KV cache" | VERIFIED | §5.1 (paraphrase): at each diffusion timestep the expert takes the VLM KV cache of [images, egomotion, reasoning] plus the embedded noisy control. KV is stop-gradiented during training. The expert has the VLM's head count and head dim, with smaller hidden and MLP sizes | — | https://arxiv.org/html/2511.00088v2 §5.1 | 2026-10-03 |
| 1.6 | C1 / P1: "64 waypoints for 6.4 s", "at 10 Hz" | VERIFIED | Eq. 2: τ = {(x, y, θ_yaw)}, i = 1…64, sampled at 10 Hz. Cards: `6.4-second future trajectory (64 waypoints at 10Hz)` | — | arXiv v2 §3; A1.5 card | 2026-10-03 |
| 1.7 | C1 / P1: "(accel, curvature)", unicycle dynamics | VERIFIED | Eq. 3 a = {(aⁱ, κⁱ)}; Eq. 5 unicycle with Euler discretization, ΔT = 0.1 s. Config: `UnicycleAccelCurvatureActionSpace`, `accel_bounds [-9.8, 9.8]`, `curvature_bounds [-0.33, 0.33]`, `dt 0.1`, `n_waypoints 64` | Ground-truth controls come from Tikhonov-regularized least squares (§3.2.2) | arXiv v2 §3.2.2; A1.5 config.json | 2026-10-03 |
| 1.8 | C1: "after ~10 Euler steps", "a ← a + 0.1·v_θ(a, t, KV)" | VERIFIED (paper default) | §5.1 Eq. 8: a_{t+δt} = a_t + δt·v_Θ; default δt = 0.1 at inference; OT path a_t = t·a + (1−t)·ε (t = 0 is noise) | **The paper contradicts itself:** §6.6 sets δt = 0.2 ("5 steps") for the decoding ablation, and Table 14 times 5 steps. A2S card and README measure 10 diffusion steps. **A1.5's default is not stated** in the sources I checked; FlashDrive (third-party) describes an 8-step setting | arXiv v2 §5.1, §6.6 | 2026-10-03 |
| 1.9 | 04 L3 "10-step Euler flow matching over 64 waypoints"; tensor contract "10 Euler steps δt=0.1" | VERIFIED (paper default) | Same as 1.8 | Same caveat as 1.8 | arXiv v2 §5.1 | 2026-10-03 |
| 1.10 | C1 tensor `a_t ∈ [B, 64, 2]` | VERIFIED | Matches Eq. 3 (64 × (a, κ)) | — | arXiv v2 Eq. 3 | 2026-10-03 |
| 1.11 | 03 Stage 2: "128 trajectory tokens (64 × {accel, curvature})" | VERIFIED | §5.1 (paraphrase): 64 waypoints × 2 quantized values = 128 discrete tokens. Config: `tokens_per_future_traj 128`, `num_bins 3000`, `traj_vocab_size 4000` | Discrete tokens are used in training only; inference decodes through the flow-matching expert (§5.1) | arXiv v2 §5.1; A1.5 config.json | 2026-10-03 |
| 1.12 | 03 P2 / 04: "180 tokens/image at 320×576 → 2,880 tokens for 4 cams × 4 frames"; "patch 16, merge 2" | VERIFIED (derived, release only); not from paper | **Paper:** §3.2.1 (paraphrase) uses patch-14 features, 2× downsampled; 448×280 gives `160 tokens per image`; Table 13 baseline `160 (1.0×)`. §6.1: experiments use **two** front cameras (120° wide, 30° tele). **Release:** A1/A1.5 cards specify 4 cameras (front-wide, front-tele, cross-left, cross-right), 4 frames each over 0.4 s at 10 Hz, 1080×1920 downsampled to `320x576`. Configs: `min_pixels 163840`, `max_pixels 196608`; 320×576 = 184,320 falls inside. Qwen3-VL-8B config: `patch_size 16`, `spatial_merge_size 2`, so 184,320 / 32² = 180, and × 16 images = 2,880 | NVIDIA states neither 180 nor 2,880. Residual risk: `temporal_patch_size 2`. If A1.5 packed frames as video, tokens would halve. Not checked in A1.5 code | arXiv v2 §3.2.1, Table 13, §6.1; A1.5 card + config; https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json | 2026-10-03 |
| 1.13 | 04 tensor `vision_tokens [B,2880,4096]` | VERIFIED (derived) | Qwen3-VL-8B-Instruct config: text `hidden_size 4096`, vision `out_hidden_size 4096`, 36 layers, 8 KV heads, head_dim 128 | Cosmos-Reason2-8B's own config is gated and was not read; it declares Qwen3-VL-8B-Instruct as its base | Qwen config (above) | 2026-10-03 |
| 1.14 | 04 tensor `images [B,4,4,3,320,576]` | VERIFIED | Cards: 4 cameras × 4 frames; processor downsamples to 320×576 | — | A1.5 card | 2026-10-03 |
| 1.15 | 04 tensor `egomotion [B,16,6]` | CORRECTED | Cards: egomotion history = `16 waypoints at 10Hz`, each a 3D translation plus a 9D (3×3) rotation. Config: `tokens_per_history_traj 48` | Input is 16 × (3 + 9) floats, tokenized into 48 history tokens. A 6-D format appears nowhere | A1.5 card + config | 2026-10-03 |
| 1.16 | 04 tensor `trajectory [B,64,3+9]` | VERIFIED | Cards: output position `(x, y, z)` plus rotation matrix `R_rot` at 64 waypoints | — | A1.5 card | 2026-10-03 |
| 1.17 | 04: `coc_tokens [B,≤200]`; "40–200 autoregressive reasoning tokens per tick" | UNVERIFIABLE (upper bound) | Table 14 uses `70ms (40 tokens)`. FlashDrive (third-party) reports A1.5 reasoning sequences average about 16 tokens | No NVIDIA source states ≤200 | arXiv v2 Table 14; https://arxiv.org/html/2608.12932v1 §3.2 | 2026-10-03 |
| 1.18 | 03 P1: why (accel, curvature) × 64 at 10 Hz | VERIFIED (rationale exists) | §3.2.2 (paraphrase): raw x, y training was sensitive to sensor noise and hurt convergence; the unicycle control representation gave better closed-loop performance | — | arXiv v2 §3.2.2 | 2026-10-03 |
| 1.19 | 04: "rendering at 1080p wastes ~11× the pixels the policy sees" | VERIFIED (arithmetic) | 1920×1080 / (576×320) = 2,073,600 / 184,320 = 11.25 | — | A1.5 card (input resolution) | 2026-10-03 |

Further cited facts (needed for G1/G2/G3):
- **AR1 closed-loop protocol.** Per §6.1, AlpaSim score is the average km driven between events (off-road or close encounter). At-fault variants exclude rear-end encounters. The simulation ends at the first event. Events where the ego deviates more than **4 m** from the recorded trajectory are excluded from all metrics. An MPC controller tracks trajectories on an extended bicycle model, and traffic agents replay their logs. — [arXiv v2 §6.1](https://arxiv.org/html/2511.00088v2)
- **AR1 internal training/eval data.** 80,000 h from more than 2,500 cities in 25 countries, with geo-fenced held-out evaluation. The CoC dataset has 700K video segments. All of this is internal, not PhysicalAI-AV. — [arXiv v2 §6.1](https://arxiv.org/html/2511.00088v2); [A1 card](https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/dd4a24cacefc9a6477a6dfc7354de2443401409d/README.md)
- **The released A1 is not RL post-trained** and has no route conditioning, meta-actions or VQA. The repo is in limited maintenance. — [NVlabs/alpamayo README @11a0e01](https://github.com/NVlabs/alpamayo/blob/11a0e01c13a5622377c45ee37d653351453ec43b/README.md)

### Inferences
- **The released Alpamayo 1 and 1.5 both use a Qwen3-VL-architecture VLM.** Both configs declare `vlm_backend: qwenvl3`. The paper's scaling ablations instead used Qwen2.5-VL-family backbones. Architecture claims about "Alpamayo" should cite the card or config, not the paper.
- **Parameter reconciliation** (inference). 8.767B (Cosmos-Reason2-8B, including a ~0.58B ViT) + ≈2.28B expert + ≈0.03B of extra trajectory-token embedding rows (vocab 155,697 vs 151,936, untied embedding and LM head) ≈ 11.08B, which matches the safetensors total of 11,078,526,194. The expert figure assumes 36 layers like the VLM, with 8 KV heads: 36 × (4.19M + 2.10M + 2.10M + 4.19M + 50.72M) = 2.28B. Under this reading, "Backbone 8.2B" counts only the Qwen3-8B-class LLM. The expert's layer count is not stated in config.
- 2,880 vision tokens applies to the released checkpoints, not to any configuration evaluated in the paper.

### Gaps
- Default flow-step count for A1.5: not in its card or README.
- Whether the A1.5 processor packs each camera's 4 frames with temporal patch 2: not checked in code.
- Cosmos-Reason2-8B config and preprocessor (gated): not read directly; derivations rely on the declared Qwen3-VL-8B base.

---

## 2. Alpamayo-R1 Table 14 latency (99 ms)

### Takeaway
Every Table 14 cell is exactly as the docs state. The hardware is stated (NVIDIA RTX 6000 Pro Blackwell). Model size, precision, camera count, image resolution and vision-token count are not. A third-party measurement of the **released Alpamayo 1.5-10B on the same GPU class gives 716.9 ms** (FlashDrive, Aug 2026). That strongly supports 04's inference that 99 ms is not the released 10B configuration.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 2.1 | 04: "99 ms end-to-end on the same GPU (RTX PRO 6000)" | VERIFIED | Table 14 caption (paraphrase): inference runtime breakdown on an NVIDIA RTX 6000 Pro Blackwell. AR1 row `Total 99ms`. §6.8 (paraphrase): 99 ms end-to-end, within the typical 100 ms real-time budget | This is **Table 13 in v1** | https://arxiv.org/html/2511.00088v2 §6.8 | 2026-10-03 |
| 2.2 | 04 / 03 Stage 0: "vision 3.43 + prefill 16.54 + reasoning 70 ms/40 tokens + trajectory 8.75 ms/5 steps" | VERIFIED | Row "Alpamayo-R1 (ours, flow matching)": `3.43ms` / `16.54ms` / `70ms (40 tokens)` / `8.75ms (5 steps)` / `99ms`. Trajectory-only baseline: `3.43ms` / `16.54ms` / `–` / `8.75ms (5 steps)` / `29ms`. AR-trajectory variant: `3.43ms` / `16.54ms` / `70ms (40 tokens)` / `222ms (127 tokens)` / `312ms` | Components sum to 98.72 ms | arXiv v2 Table 14 | 2026-10-03 |
| 2.3 | 04: "The table does not state model size or precision" | VERIFIED | Caption and §6.8 name only the GPU: no parameter count, precision, cameras, resolution or token count. The caption credits flow-matching decoding plus efficient vision encoding | Hardware is stated | arXiv v2 Table 14, §6.8 | 2026-10-03 |
| 2.4 | (context) abstract ties 99 ms to on-vehicle tests | INFO | Abstract (paraphrase): on-vehicle road tests confirm real-time performance (99 ms latency) | The 99 ms measurement is on an RTX 6000 Pro Blackwell, not the vehicle computer | https://arxiv.org/abs/2511.00088 | 2026-10-03 |
| 2.5 | 04: "1.75 ms per decoded token is 5× faster than the BF16 bandwidth bound for an 8.2B VLM" | VERIFIED (arithmetic) | 70 ms / 40 = 1.75 ms. 8.2e9 × 2 B / 1.79 TB/s = 9.2 ms, a 5.2× gap | The 1.79 TB/s figure was not checked here (sibling scope) | arXiv v2 Table 14 | 2026-10-03 |
| 2.6 | 04: "the 99 ms figure is a smaller or quantized on-vehicle configuration with far fewer vision tokens … open question" | UNVERIFIABLE (NVIDIA has not published the config); independent evidence supports it | FlashDrive Table 1, RTX PRO 6000, 1 trajectory, released A1.5-10B: `Encode 87.0` / `Prefill 165.3` / `Decode 271.7 (56.4 tok/s)` / `Action 192.9` / `Total 716.9` ms. System optimizations (CUDA graphs + fusion): `513.3` ms. Full FlashDrive + W4A8: `151.4` ms | Third-party measurement (arXiv 2608.12932 v1, 13 Aug 2026) | https://arxiv.org/html/2608.12932v1 Table 1 | 2026-10-03 |
| 2.7 | 04: derived "≈ 500 ms per tick (~2 Hz)" for A1.5 BF16 at B=1 | INFO | Measured 716.9 ms (1.4 Hz) unoptimized; 513.3 ms with system optimizations | Same order of magnitude as the derivation | FlashDrive Table 1 | 2026-10-03 |

### Inferences
- **The expert in Table 14 cannot be a 2–2.3B BF16 expert at B=1.** Each step would stream 4.0–4.6 GB of weights, about 2.2–2.6 ms at 1.79 TB/s, so 5 steps would take at least 11–13 ms. The table shows 8.75 ms. Likewise, prefill of 16.54 ms is about 6× below the compute floor for 2,880 tokens on an 8.2B model (~4.7e13 FLOP, about 94 ms at the docs' 504 TFLOPS peak). Both point to a smaller, quantized and/or fewer-token configuration. This rests on the docs' unverified GPU peak numbers.
- FlashDrive measures decode at about 17.7 ms per token unoptimized (56.4 tok/s) and about 11.2 ms optimized (89.5 tok/s). That brackets the docs' 9.2 ms bandwidth bound, so the docs' first-principles model holds up; Table 14's 1.75 ms per token does not describe the released model.

### Gaps
- NVIDIA has not published the configuration behind Table 14.
- FlashDrive's venue and peer-review status were not checked. Its code is said to be at github.com/z-lab/flashdrive (not opened).

---

## 3. Alpamayo-R1 small-model (0.5B) rows used as nanoAlpamayo acceptance (03 §2)

### Takeaway
All six numbers exist, but they come from **two different tables, two different evaluation domains, and possibly two different 0.5B models**.
- **Table 10 is the only public-data evaluation.** It covers 644 open-loop PhysicalAI-AV samples and 920 closed-loop NuRec scenarios, for a model named `Alpamayo-R1-0.5B`.
- **Table 9 is internal.** It runs on NVIDIA's CoC dataset, for "a 0.5B AR1 model fine-tuned on CoC data". Its metric is the ADE of the most likely of 6 samples, not minADE6.

The 644-sample and 920-scenario lists are not published.

### Cited Findings

| # | Claim as written (03 §2) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 3.1 | "minADE6 on 644 PAI samples: 0.913 m" | VERIFIED | Table 10, row `Alpamayo-R1-0.5B`: minADE6@6.4s `0.913`. Caption (paraphrase): open-loop on 644 examples from the PhysicalAI-AV evaluation set | Only in v2 (7 Jan 2026). AR1-10B: `0.849` (7.0% better) | https://arxiv.org/html/2511.00088v2 §6.4, Table 10 | 2026-10-03 |
| 3.2 | "RL post-training, ADE on CoC set: 2.12 → 1.92 m (−9.4%)" | VERIFIED | Table 9: SFT `2.12m`; SFT + RL(r_reason + r_consistency) `1.92m`. Text (paraphrase): a 9.4% reduction in most-likely-mode ADE | Metric is the **ADE of the most-likely of 6 rollouts**, not minADE6. Eval set is the full CoC dataset of §6.2 (internal, geo-fenced NVIDIA data). Model is a 0.5B AR1 fine-tuned on CoC (§6.3) | arXiv v2 §6.3, Table 9 | 2026-10-03 |
| 3.3 | "Reasoning quality 3.1 → 4.5" | VERIFIED | Table 9 Reasoning Grading: SFT `3.1`; +RL(r_reason) `4.5`; +RL(r_reason + r_consistency) `4.5`; +r_safety `4.4` | Graded by a large-reasoning-model critic (§5.3.2); internal | arXiv v2 Table 9 | 2026-10-03 |
| 3.4 | "consistency 0.62 → 0.85" | VERIFIED | Table 9: SFT `0.62`; r_reason only `0.53`; r_reason + r_consistency `0.85`; + r_safety `0.83` | Reasoning reward alone *lowers* consistency | arXiv v2 Table 9 | 2026-10-03 |
| 3.5 | (column the docs omit) Table 9 close-encounter rate | INFO | `6.9` (SFT) / `5.8` / `6.2` (the −9.4% ADE row) / `3.7` (with r_safety), in % | The row the docs cite cut close encounters only 6.9 → 6.2. The 3.7 comes from adding the safety reward | arXiv v2 Table 9 | 2026-10-03 |
| 3.6 | "Closed-loop (920 NuRec scenarios): close-encounter 9.0%" | VERIFIED | Table 10: `9.0 ± 1.0`. Caption: 920 scenarios from the PhysicalAI-AV NuRec dataset via AlpaSim; closed-loop metrics are at-fault | At-fault metric | arXiv v2 Table 10 | 2026-10-03 |
| 3.7 | "off-road 19%" | VERIFIED | Table 10: `19.0 ± 0.0` | At-fault | arXiv v2 Table 10 | 2026-10-03 |
| 3.8 | "AlpaSim 0.35" | VERIFIED | Table 10: `0.35 ± 0.01`; AR1-10B `0.72 ± 0.02` | AlpaSim score is km between events (§6.1) | arXiv v2 Table 10 | 2026-10-03 |
| 3.9 | Table header "Metric (0.5B, **public val**)" | CORRECTED | Table 9 rows are on internal CoC data (§6.1–§6.3) | Only rows 3.1 and 3.6–3.8 are public-data evaluations | arXiv v2 §6.1–6.4 | 2026-10-03 |
| 3.10 | 03 §2: small pretrained VLM backbone (0.5B–2B) "as NVIDIA does" | INFO | §6.5.1 (paraphrase): the 0.5B variant is a DINOv2 vision encoder plus the Qwen2.5-0.5B LLM; 3B and 7B are Qwen2.5-VL-3B/7B. §6.4 says only that Table 10's 0.5B uses a smaller backbone | The 0.5B is not a Cosmos-Reason model. The paper never says Table 9's 0.5B is Table 10's `Alpamayo-R1-0.5B` | arXiv v2 §6.4, §6.5.1 | 2026-10-03 |
| 3.11 | 03 Stage 2: "minADE6 ≤ 1.0 m on the 644-sample val" (assumes a public 644-sample set) | UNVERIFIABLE | The 644 IDs are not in the paper. NVlabs/alpamayo, alpamayo1.5 and alpamayo2 ship only `notebooks/clip_ids.parquet`: 1,181 IDs, purpose undocumented, identical across the three repos | Set is not reproducible as specified | arXiv v2; https://github.com/NVlabs/alpamayo1.5/tree/36aeb4c5938cbc2eb2aed33b22434773da4ab639/notebooks | 2026-10-03 |
| 3.12 | 03 §2: nanoAlpamayo uses "a PhysicalAI-AV subset with **public CoC labels**" | CORRECTED (scale) | PAI-AV card "Reasoning Labels" (human-refined CoC for curated OOD scenarios, `reasoning/ood_reasoning.parquet`): Train `1450` clips / `1728` labels; Val `290` / `349`; Test held out for the OOD Benchmark Challenge | About 1.7K public training labels, vs 700K (AR1), 3M (A1.5) and ~3.7M (A2S) internal traces | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles (repo SHA 33f9bf4) | 2026-10-03 |

### Inferences
- The acceptance target "within 20% relative, on the PhysLoop gate" compares a gate measurement with a number from another instrument. The 920-scenario list and AlpaSim version are unpublished, and NVIDIA's own scores for the same model moved by about 70% when its protocol changed (row 4.12). Only a re-run of Alpamayo-R1-0.5B under the gate would be a valid anchor, and those weights are not public (HF lists only the 10B).
- Table 9 cannot be reproduced on public data. "Same direction, with CI" needs a public stand-in: the ~1.7K OOD CoC labels, or CoC auto-labels from Alpamayo 2 Super or the open CoC autolabeler (subject to the dataset license, §7).

### Gaps
- The 644-sample list, the 920-scenario list, and the AlpaSim version used for Table 10.
- Table 9 eval-set size.
- Whether a 0.5B Alpamayo checkpoint will ever be released (none on HF as of 2026-10-03).

---

## 4. Alpamayo 1 vs Alpamayo 1.5: dates, parameters, backbone, license, checkpoint, model-card scores

### Takeaway
- **Release dates.** Alpamayo 1 is `nvidia/Alpamayo-R1-10B`: weights on HF 3 Dec 2025, renamed at CES (press release 5 Jan 2026). Alpamayo 1.5 went up on HF 19 Mar 2026 (GTC 2026).
- **Size.** Both cards state `8.2B` backbone + `2.3B` expert ("10B"); both checkpoints hold 11.08B BF16 parameters, 22.16 GB.
- **Backbone.** A1.5 is built on Cosmos-Reason2-8B, which is Qwen3-VL-8B-based.
- **License.** Weights have been under OpenMDW-1.1 only since **4 Aug 2026**; before that they were under a non-commercial NVIDIA License. The cards still carry a stale "non-commercial" sentence.
- **Model-card scores are protocol-dependent.** On 7 Aug 2026 the A1.5 card's AlpaSim score went from `0.81 ± 0.01` (910 scenarios) to `1.37 ± 0.10` (913) with no weight change. The A1 card is still on the old protocol.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 4.1 | 02 G3: "Alpamayo 1/1.5/2" are available | VERIFIED | HF `author=nvidia`, search "alpamayo", returns exactly 3 models: `Alpamayo-R1-10B`, `Alpamayo-1.5-10B`, `Alpamayo2-Super` | — | https://huggingface.co/api/models?author=nvidia&search=alpamayo | 2026-10-03 |
| 4.2 | (task) Alpamayo 1 "CES, Jan 2026?" | VERIFIED with nuance | CES press release dated `January 5, 2026` announces 10B Alpamayo 1 on HF. A1 card field `Release Date: Hugging Face 12/03/2025`. HF commits: model files `2025-12-03`; rename `2026-01-08` | Weights public 3 Dec 2025 as Alpamayo-R1-10B; renamed Alpamayo 1 at CES | https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development ; https://huggingface.co/nvidia/Alpamayo-R1-10B/commits/main | 2026-10-03 |
| 4.3 | (task) Alpamayo 1 card numbers | INFO | A1 card: `Backbone: 8.2B`, `Action Expert: 2.3B`; 4 cameras; AlpaSim `0.73 ± 0.01` on `910` scenarios; minADE6@6.4s `1.22m` on `937` challenging samples | Old evaluation protocol (see 4.12) | A1 card @dd4a24c | 2026-10-03 |
| 4.4 | Alpamayo 1.5 = "HF nvidia/Alpamayo-1.5-10B" (C1, 04) | VERIFIED | Card field `Release Date: Hugging Face 03/19/2026`; commit "Upload model checkpoint (7 files, 20.6 GB)" `2026-03-20T03:17Z` | NVIDIA's HF article (1 Jun 2026) says the platform was expanded at GTC 2026 | A1.5 card @7aba829; https://huggingface.co/nvidia/Alpamayo-1.5-10B/commits/main | 2026-10-03 |
| 4.5 | C1 / 04: "8.2B VLM + 2.3B expert", "10B" | VERIFIED (card) | `Backbone: 8.2B parameters`; `Action Expert: 2.3B parameters`; described as an open 10B-parameter model | safetensors metadata: `11,078,526,194` BF16 params | A1.5 card; https://huggingface.co/api/models/nvidia/Alpamayo-1.5-10B | 2026-10-03 |
| 4.6 | (task) backbone: Cosmos-Reason2 or Qwen3-VL-8B? | VERIFIED: both | A1.5 `base_model: nvidia/Cosmos-Reason2-8B`; Cosmos-Reason2-8B `base_model: Qwen/Qwen3-VL-8B-Instruct` | — | A1.5 card; https://huggingface.co/api/models/nvidia/Cosmos-Reason2-8B | 2026-10-03 |
| 4.7 | 04: "OpenMDW-1.1" | VERIFIED today; history differs | Current card: weights under OpenMDW-1.1. LICENSE replaced `2026-08-04` (commit 912a0f7). The original LICENSE (`2026-03-20`, commit 5dc9c5f) was the "NVIDIA License", whose §3.3 limits use to non-commercial, defined as research or evaluation only. The Alpamayo 1 repo followed the same path (eaaa25e → dd4a24c, 2026-08-04). NVIDIA blog of 4 Aug 2026 (paraphrase): OpenMDW is now applied across the entire Alpamayo family; earlier releases were R&D-only | **Stale text:** both cards still say the model is "ready for non-commercial use", and the GitHub READMEs' disclaimers say research and evaluation only. The OpenMDW LICENSE file is presumably controlling (inference) | https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/5dc9c5f26ea4bf995728df628e49966bb016bc4e/LICENSE ; …/blob/912a0f74846149e1f2aa45630b88b44ae34573c1/LICENSE ; https://blogs.nvidia.com/blog/alpamayo-2-super/ | 2026-10-03 |
| 4.8 | C2: "Alpamayo 1.5 BF16 ≈ 22 GB" | VERIFIED | Safetensors total `22,157,316,944` bytes (22.16 GB / 20.64 GiB). Quant recipe: `BF16 ~22 GB`. README: weights `22 GB` | — | https://huggingface.co/api/models/nvidia/Alpamayo-1.5-10B?blobs=true ; https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_5_quant/README.md | 2026-10-03 |
| 4.9 | 04 model list: "24 GB" | CORRECTED | Card: minimum `1 GPU with at least 24GB of VRAM`. alpamayo1.5 README, measured on H100: single-sample `~24 GB`; 16 samples `~40 GB`; 16 samples with CFG `~60 GB` | 24 GB is minimum VRAM. The checkpoint is 22.16 GB | A1.5 card; https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/README.md | 2026-10-03 |
| 4.10 | C2: "FP8 ≈ 11 GB" | VERIFIED | Recipe table: `FP8 ~11 GB 2.00x` | — | quant recipe README @c29dc61 | 2026-10-03 |
| 4.11 | 04: "ModelOpt 0.43: FP8, NVFP4, AutoQuant 6.5 bits; BF16 22 GB→9 GB … publishes no accuracy delta or latency"; 02 §0 "NVIDIA's PTQ recipe is open-loop only" | VERIFIED | `nvidia-modelopt==0.43.0`, `torch==2.8.0`; `Autoquant 6.5BPE ~9 GB 2.44x`. The table lists size only; accuracy is described only qualitatively. `eval.py` reports minADE (open-loop) only | AutoQuant's default budget is `4.8` bits; 6.5 is the worked example | quant recipe README @c29dc61 | 2026-10-03 |
| 4.12 | 04 eval lock: "model-card noise floor: AlpaSim 1.37 ± 0.10 for Alpamayo 1.5 on 913 scenarios" | VERIFIED (current card) — but **not stable** | Card @7aba829: AlpaSim `1.37 ± 0.10` on `913` scenarios. Changed `2026-08-07` (commits 68a8fda, 7aba829) from `0.81 ± 0.01` on `910` (card @22c3322, 2026-07-21). Weights unchanged | The card never defines "±". "Noise floor" is the docs' own interpretation | https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/22c33224a78a3d9343b954c8259a12d36cc276c8/README.md ; A1.5 card @7aba829 | 2026-10-03 |
| 4.13 | (task) A1.5 minADE | INFO | Current: minADE6@6.4s `0.916m` on `1434` challenging PAI-AV samples; LingoQA `74.2`. Before 2026-08-07: `1.11m` on `937` | — | Same two revisions | 2026-10-03 |
| 4.14 | 04: "Alpamayo 1.5 Nano" | VERIFIED (later NVIDIA naming) | "Alpamayo 1.5 Nano" and "Alpamayo 1 Nano" appear in the 1 Jun 2026 press release, the Table 1 reference column of the 4 Aug 2026 dev blog, the NVlabs/alpamayo1.5 and alpamayo repo descriptions, and the alpamayo-recipes model table | The HF card title is still "Alpamayo 1.5" | https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/ ; https://github.com/NVlabs/alpamayo-recipes | 2026-10-03 |
| 4.15 | 04: "public GRPO and NVFP4 recipes" | VERIFIED | alpamayo-recipes: `alpamayo1_x_rl` is open-loop RL with Cosmos-RL + GRPO (A1 and A1.5); `alpamayo1_5_quant` covers FP8 and NVFP4 + FP8 mixed precision. Also `alpamayo1_sft`, `alpamayo1_5_sft`, `alpamayo2_sft` | `alpamayo2_sft` is not mentioned in the docs | repo tree @c29dc61 | 2026-10-03 |
| 4.16 | 04: stock open-loop recipe "4 policy + 1 vLLM GPU"; "NVIDIA's 512 policy : 128 rollout GPU split" | VERIFIED | RL recipe README: the local test needs at least 5 GPUs with ≥80 GB (`dp_shard_size 4`, 1 rollout replica). Large scale: 512 policy GPUs + 128 rollout GPUs (640 GPUs, 80 nodes), global batch 2,560 | — | https://github.com/NVlabs/alpamayo-recipes/blob/c29dc61b7318169d2439f3e9d88e547242e301de/recipes/alpamayo1_x_rl/README.md | 2026-10-03 |
| 4.17 | 04: "Alpamayo 1.5 resizes inputs to 320×576" | VERIFIED | Card: 1080×1920 images downsampled by the processor to `320x576` | — | A1.5 card | 2026-10-03 |
| 4.18 | 04 trainer bound: "GRPO update on 10.5B parameters" | CORRECTED (minor) | 8.2 + 2.3 = 10.5 by the card, but the checkpoint holds `11,078,526,194` params | Use 11.08B in FLOP and memory math (~5% more) | HF API | 2026-10-03 |
| 4.19 | C2: "a 3B student BF16 ≈ 6 GB" | VERIFIED (arithmetic) | 3e9 × 2 bytes = 6 GB | Weights only | — | 2026-10-03 |

### Inferences
- **Do not rank card scores against each other.** The A1 card uses the old protocol (910 scenarios / 937 samples). The A1.5 and A2S cards use the new one (913 / 1,434). AR1 paper Table 10 is a third variant (920 / 644). The same A1.5 weights moved from 0.81 to 1.37. G3 has to re-run every policy under one pinned protocol (scenario list, AlpaSim version, seeds). That is exactly the instrument problem PhysLoop sets out to study, and NVIDIA's own cards illustrate it.
- Since 4 Aug 2026 the weights allow commercial use and derivatives (OpenMDW-1.1). The datasets are licensed separately and stay restrictive (§7).

### Gaps
- What changed in NVIDIA's evaluation protocol on 2026-08-07: AlpaSim version, scenario selection, or event rules. The card does not say.
- What "±" means on the cards (seeds? bootstrap? scenarios?).

---

## 5. Alpamayo 2 Super: parameters, hardware, scores, purpose, release timeline

### Takeaway
- **Timeline.** Announced 1 Jun 2026 at Computex (Taipei) as a "32-billion-parameter" model, expected "this summer". Weights committed as a release candidate on 31 Jul 2026. Official HF release and technical blog on **4 Aug 2026**.
- **Parameters.** The card and blog say 34B total: a 32B Cosmos 3 Super Reasoner plus an expert that the card lists as 2.3B and the description and blog call 2B. The safetensors total is 35.81B.
- **Hardware.** Tested on 1×H100 80 GB: peak `72,115 MiB` with 7 cameras × 4 frames, BF16, 10 diffusion steps, batch 1.
- **Scores.** AlpaSim `1.50 ± 0.13` (913 scenarios); minADE6 `0.911 m` (1,434 samples).
- **Purpose.** Explicitly a multi-task teacher, auto-labeler, critic and data engine.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 5.1 | 02 §7: Alpamayo 2 Super blog "(Aug 4 2026)" at the given URL | VERIFIED | Dev blog dated `Aug 04, 2026`, by Boris Ivanovic and Marco Pavone | — | https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/ | 2026-10-03 |
| 5.2 | 01 §2: "Alpamayo 2 Super 34B" | VERIFIED | Card: a 34B-parameter model with a 32B VLM backbone and a 2B diffusion expert. Blog: an open 34-billion-parameter VLA | 1 Jun press release: "32-billion-parameter" (backbone framing). safetensors: `35,813,934,322` params | https://huggingface.co/nvidia/Alpamayo2-Super/blob/00554695e729a6ff0b6281fd2c81b18d06e33dbe/README.md ; https://huggingface.co/api/models/nvidia/Alpamayo2-Super | 2026-10-03 |
| 5.3 | 04: "32B Cosmos 3 Reasoner + 2.3B expert" | VERIFIED (card fields); sources disagree | Card fields `Backbone: 32B`, `Action Expert: 2.3B`. Card description and blog: **2B** expert. `base_model: nvidia/Cosmos3-Super`; card says it is built on the Cosmos 3 Super Reasoner. Config: VLM `Qwen3VLForConditionalGeneration`, text `num_hidden_layers 64`, `hidden_size 5120`, vision `patch_size 16`. Expert `hidden_size 1536`, 64 layers, 16 heads, 8 KV heads | Expert size: 2B (prose) vs 2.3B (field) | A2S card + config.json @00554695 | 2026-10-03 |
| 5.4 | 04: "1×H100 80 GB" | VERIFIED | Card: tested on `1 NVIDIA H100 80GB HBM3`. Measured config: 7 cameras, 4 frames each, batch 1, 1 sample, BF16, SDPA, CFG off, 10 diffusion steps. Peak `72,115 MiB` device memory | Checkpoint 71.65 GB BF16 (66.72 GiB). The navigation-CFG demo needs 2×80 GB H100 (~67 + ~71 GiB) | A2S card; https://github.com/NVlabs/alpamayo2/blob/5e7975f4a2100ee8ac1a62b79239bbabefddcbef/README.md | 2026-10-03 |
| 5.5 | 04: "AlpaSim 1.50" | VERIFIED | Card and blog: `1.50 ± 0.13` on `913` PAI-AV NuRec scenarios; reference value for A1.5 Nano `1.37 ± 0.10` | New protocol | A2S card; dev blog Table 1 | 2026-10-03 |
| 5.6 | 04: "minADE6 0.911 m" | VERIFIED | `0.911m` at 6.4 s on `1,434` challenging PAI-AV samples | **Not comparable** with AR1-0.5B's 0.913 m, which uses 644 samples, a different subset and an older protocol | A2S card; dev blog Table 1 | 2026-10-03 |
| 5.7 | (task) stated purpose: auto-labeling, distillation teacher | VERIFIED | Card use cases: trajectory prediction, VQA, 2D grounding, auto-labeling. Blog (paraphrase): one model can serve as offline policy teacher, evaluation critic and data engine. CoC auto-label similarity `0.652` vs expert annotations on 8K internal clips (Qwen3-VL 32B: `0.450`). PAI-AV Reasoning Benchmark `0.433` (A1.5 Nano `0.414`). LingoQA `79.2` | — | dev blog; A2S card | 2026-10-03 |
| 5.8 | (task) release timeline | INFO | 1 Jun 2026 press release (Taipei): availability expected this summer, code on GitHub and weights on HF. HF commit "Add Alpamayo 2 Super release candidate" `2026-07-31`. Card `Release Date: Hugging Face 08/04/2026` | Not downloadable on 1 Jun, even though NVIDIA's 1 Jun HF article linked the repo | https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Launches-Alpamayo-2-Super-Open-Reasoning-Model-for-Robotaxis/default.aspx (read via a fetch tool's extraction) ; https://huggingface.co/nvidia/Alpamayo2-Super/commits/main | 2026-10-03 |
| 5.9 | (task) license | INFO | Card: OpenMDW-1.1. Blog (paraphrase): covers fine-tuning, derivative models and commercial redistribution; distilled models can be deployed commercially without further NVIDIA permission; model outputs carry no license conditions | Unlike the A1/A1.5 cards, there is no stale "non-commercial" sentence | dev blog; A2S card | 2026-10-03 |
| 5.10 | 03 P4: "auto-label CoC with Alpamayo 2 Super" | VERIFIED (capability) | `notebooks/autolabeling.ipynb` exists. README (paraphrase): by default auto-labeling conditions on the future ego trajectory, and it can auto-label from a predicted trajectory instead. A separate open pipeline, `NVlabs/alpamayo-coc-autolabeler`, has existed since 2026-05-24 | Flag: PAI-AV license §4.6 (no derivative works or distribution) may limit publishing auto-labels of PAI-AV clips | https://github.com/NVlabs/alpamayo2 ; https://api.github.com/orgs/NVlabs/repos | 2026-10-03 |
| 5.11 | (task) training data | INFO | Card: about 115,000 h of multi-camera video and about 3.7M CoC traces. Recipes README: 110,000+ hours | — | A2S card; recipes README | 2026-10-03 |
| 5.12 | (task) inputs | INFO | Card: public notebook profiles use 6 cameras × 4 frames. Blog: up to 7 cameras. AlpaSim driver config in the blog: 4 cameras, 4 frames, `max_batch_size: 1`, language-only navigation | — | A2S card; dev blog | 2026-10-03 |

### Inferences
- **A2S costs one full H100 per instance at batch 1.** The 72 GiB peak leaves essentially no headroom. Every G3 or G5 teacher-labeling hour on A2S therefore costs a full H100. NVIDIA has published no FP8 or NVFP4 A2S checkpoint; the only quantized version found is a third-party "NVFP4-language-smoke" repo.
- **Use the card figure, not 0.911 vs 0.913.** The near-identical numbers (A2S on 1,434 samples, AR1-0.5B on 644) are a coincidence across different sets. Any "teacher vs 0.5B" comparison must be re-measured.

### Gaps
- No published A2S latency.
- No NVIDIA-quantized A2S.
- Whether A2S closed-loop numbers used 4, 6 or 7 cameras (the blog's AlpaSim config shows 4).

---

## 6. Does NVIDIA explicitly position Alpamayo as a teacher to be distilled into compact runtime models? Is there a public distillation procedure?

### Takeaway
**Yes, explicitly and repeatedly, in primary sources:**
- CES press release, 5 Jan 2026;
- Computex press release, 1 Jun 2026;
- A2S dev blog, 4 Aug 2026, which names DRIVE AGX Thor;
- NVIDIA blog, 4 Aug 2026;
- the alpamayo-recipes README.

**There is no public distillation recipe** as of 2026-10-03. The "planned" quote is real, but it refers to a smaller *single-GPU AlpaGym training checkpoint*, not an in-vehicle Thor student.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 6.1 | C2 / 02 §0: "NVIDIA positions Alpamayo as a teacher distilled into compact runtime models on (DRIVE AGX) Thor" | VERIFIED | A2S dev blog, closing section (the one verbatim quote): "it can be distilled into compact models that run on NVIDIA DRIVE AGX Thor" | — | https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/ | 2026-10-03 |
| 6.2 | Same claim (earliest source) | VERIFIED | CES press release, 5 Jan 2026 (paraphrase): Alpamayo models do not run directly in the vehicle; they serve as large-scale teachers that developers fine-tune and distill into their AV stacks. Alpamayo 1 can be adapted into smaller runtime models | — | https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development | 2026-10-03 |
| 6.3 | Same claim | VERIFIED | 1 Jun 2026 press release (paraphrase, via fetch-tool extraction): A2S is designed as a teacher that can be distilled into compact models running on DRIVE Hyperion compute, with DRIVE AGX Thor inside the vehicle | Extracted by a summarizing fetch tool, so treat the exact wording as unconfirmed | investor.nvidia.com press release (row 5.8) | 2026-10-03 |
| 6.4 | Same claim | VERIFIED | NVIDIA blog, 4 Aug 2026 (paraphrase): A2S produces teacher outputs for distillation in the cloud; A1.5 and A1 are cheaper distillation options; distilled models are then optimized for in-vehicle real-time inference (a cloud-to-car workflow) | — | https://blogs.nvidia.com/blog/alpamayo-2-super/ | 2026-10-03 |
| 6.5 | 01 C2 / 02 §0: "No public distillation procedure exists ('planned' per NVIDIA's closed-loop blog)" | VERIFIED (quote and current absence); context note | Closed-loop blog, 31 May 2026 (paraphrase): training the 10B model does not fit on one GPU; a distillation script to a smaller single-GPU-compatible checkpoint is planned | Purpose is single-GPU AlpaGym training, not a Thor student. As of 2026-10-03, no path containing "distill" exists in NVlabs/alpamayo-recipes (275 files; recipes `alpamayo1_sft`, `alpamayo1_5_sft`, `alpamayo2_sft`, `alpamayo1_x_rl`, `alpamayo1_5_quant`) or NVlabs/alpagym (298 files). The recipes README still advertises "post-train, quantize or distill" | https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/ ; https://github.com/NVlabs/alpamayo-recipes/tree/c29dc61b7318169d2439f3e9d88e547242e301de | 2026-10-03 |
| 6.6 | 02 §0 (recipes ecosystem) | VERIFIED | alpamayo-recipes README (paraphrase): A2S can serve as a teacher for distillation and quantization into students that meet in-vehicle latency and safety requirements on DRIVE AGX Thor. NVIDIA's 1 Jun HF article lists "AV model distillation" (offline teacher, output or feature supervision) as a use case | — | https://github.com/NVlabs/alpamayo-recipes ; https://huggingface.co/blog/drmapavone/nvidia-alpamayo-2 | 2026-10-03 |

### Inferences
- G5 (teacher→student distillation) is still open ground in the public stack: no NVIDIA recipe and no NVIDIA student checkpoint. NVIDIA's messaging makes a release likely at some point. GTC Berlin (20–22 Oct 2026) is a near-term risk date ([NVIDIA blog banner](https://blogs.nvidia.com/blog/alpamayo-2-super/)).

### Gaps
- No NVIDIA timeline for the planned distillation script or any distilled checkpoint.

---

## 7. PhysicalAI-AV dataset: hours, cameras, rigs, countries, CoC labels, size, gating, license

### Takeaway
- **Contents.** 1,700 h: 306,152 clips of 20 s, 25 countries, 2,500+ cities. 7 cameras on every clip; LiDAR on 298,326; up to 10 radars on 160,761. Hyperion 8/8.1 rigs. 133 TB. Versions 25.10, 26.03, 26.09.
- **Gating.** Automatic approval on accepting the NVIDIA AV Dataset License, so access is immediate.
- **License scope.** AV use only, commercial or non-commercial; it is **not** research-only. But four clauses bear directly on PhysLoop:
  - purpose limited to *internal* AV/ADAS development *using NVIDIA technology*;
  - a confidentiality clause covering benchmarking, regression and performance data relating to the Dataset;
  - no derivative works or distribution;
  - expiry 12 months after first download.
- **CoC labels.** Public, but only for an OOD subset of about 1.7K training labels.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 7.1 | 03 P4: "PhysicalAI-AV (1,700 h)" | VERIFIED | Card: `1700 hours`. CES press release: `1,700+ hours` | — | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles (SHA 33f9bf4); CES press release | 2026-10-03 |
| 7.2 | (task) cameras and sensors | INFO | `306,152` clips × 20 s. Multi-camera `(7)` on all clips; LiDAR `(1)` on `298,326`; radar `(up to 10)` on `160,761`. Cameras: cross-left/right 120°, front-wide 120°, front-tele 30°, rear-left/right 70°, rear-tele 30°. 1080p mp4 at 30 fps; LiDAR at 10 Hz | — | PAI-AV card | 2026-10-03 |
| 7.3 | (task) sensor rigs | INFO | `platform_class` is hyperion_8 or hyperion_8.1. NuRec reconstruction is supported only for Hyperion 8.1 | — | PAI-AV card | 2026-10-03 |
| 7.4 | (task) countries | INFO | 25 countries, 2,500+ cities; e.g. US `155360` clips, Germany `43900`, France `10364` | — | PAI-AV card | 2026-10-03 |
| 7.5 | (task) size | INFO | Card: total size `133TB` | HF `usedStorage` (all revisions) ≈ 246 TB | PAI-AV card; https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles | 2026-10-03 |
| 7.6 | (task) versions | INFO | `25.10` initial release; `26.03` adds offline features for 97% of clips; `26.09` fixes offline calibration for 140 clips | Repo last modified 2026-09-12 | PAI-AV card | 2026-10-03 |
| 7.7 | 03 §2: CoC labels public? | CORRECTED (scale) | See row 3.12: Train `1450`/`1728`, Val `290`/`349`, Test held out | OOD-only, human-refined | PAI-AV card | 2026-10-03 |
| 7.8 | CLAUDE.md: "Gated assets must be requested by Huy"; 02 Week 1 "Gated access (PAI-AV …)" | VERIFIED (gated; approval is automatic) | HF API `gated: auto`; heading says you must agree to the NVIDIA AV Dataset License Agreement. HF docs (paraphrase): under automatic approval, access is granted immediately on request | Requires an HF account and acceptance by an adult of legal age. No manual review, so no waiting period | PAI-AV API; https://huggingface.co/docs/hub/models-gated | 2026-10-03 |
| 7.9 | (task) license terms: non-commercial? research-only? | INFO (decision-relevant) | Close paraphrase of the gated license text. **§1:** licensed solely for internal development of autonomous vehicles and automated-driving-assist systems using NVIDIA technology. **§3:** Confidential Information includes the Dataset's output and any benchmarking, competitive-analysis, regression or performance data relating to the Dataset. **§4.6:** no derivative works, and no sale, distribution or hosting. **§8:** expires 12 months after initial download. Card: commercial or non-commercial AV use per the license | Neither "research-only" nor "non-commercial". The binding constraints are §1, §3, §4.6 and §8 | PAI-AV card gating prompt (via HF API `cardData.extra_gated_prompt`); license link `…/blob/main/LICENSE.pdf` | 2026-10-03 |
| 7.10 | (task) other gated or ungated assets | INFO | `gated: auto`: PAI-AV, PAI-AV-NuRec, PAI-AV-NCore, Cosmos-Reason2-2B/8B, omni-dreams-models, omni-dreams-samples. Not gated: all three Alpamayo models, Cosmos-Reason2-32B, all Cosmos3-*, instant-nurec, omni-dreams-scenes | — | HF API queries | 2026-10-03 |
| 7.11 | (not in docs) NCore subset | INFO | PAI-AV-NCore: about 1.1K clips converted to NCore V4 with offline calibration, egomotion and cuboid labels; created `03/06/2026`; same license | — | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore (SHA 5f2fd45) | 2026-10-03 |

### Inferences
- **Blocker-class flag (not adjudicated).** §3 treats benchmarking, regression and performance data relating to the Dataset as NVIDIA Confidential Information, and the NuRec license has the same clause. Read literally, publishing G1–G3 results computed on PAI-AV or NuRec could conflict with it. Yet NVIDIA itself publishes numbers on these data and runs public challenges and leaderboards on them, so the practical reading is unclear. Get written clarification from NVIDIA (or legal review) before the GTC poster (10 Nov) or any preprint. Also, the 12-month expiry (§8) falls inside the 52-week mastery plan.
- §4.6 probably covers redistributing CoC auto-labels or derived clip sets. The plan should publish code, manifests of clip UUIDs, and aggregate statistics, not data.

### Gaps
- NVIDIA's interpretation of §3 for third-party publications (not found).
- The legal meaning of "using NVIDIA technology" for a GPU-rented, open-source workflow (out of scope).

---

## 8. NuRec scene assets, Instant NuRec, OmniDreams

### Takeaway
- **NuRec 26.04** is real: 1,607 reconstructed PAI-AV clips, about 2.9 TB (2.87 TB on HF), and the only release on the main branch. It carries the same restrictive AV dataset license, with automatic gating.
- **NVIDIA's evaluation scenario counts** (920, 910, 913) do not match the public 1,607, and the selection lists are unpublished.
- **Instant NuRec** (arXiv 2607.14203, NVIDIA, 15 Jul 2026) and **OmniDreams** (arXiv 2606.03159, NVIDIA, v1 2 Jun / v3 23 Sep 2026) both exist as cited.
- **OmniDreams' 68 FPS** (single view, one GB300, 704×1280) is verified. Real-time 4-camera rendering needs 16 GB300s.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 8.1 | 02 §3/§4: "NuRec 26.04", "~2.9 TB" | VERIFIED | Card version `26.04`: scenes built with NuRec 26.04 from `1607` PAI clips. Dataset Quantification: `approx 2.9 TB`. HF total file size `2.87 TB`. Main branch holds only `sample_set/26.04_release` (1,607 `.usdz`) | Card prose says "1500+" | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec (SHA be46448); HF API siblings | 2026-10-03 |
| 8.2 | (task) number of reconstructed scenes | INFO | 1,607 in 26.04. History: `26.01` cut the processed count to `918`; `26.02`; `25.05` added `labels.json` | Older releases are reachable through repo history | NuRec card | 2026-10-03 |
| 8.3 | (task) license and gating | INFO | NVIDIA Autonomous Vehicles NuRec Dataset License Agreement, with the same §1/§3/§4.6/§8 clauses as PAI-AV; `gated: auto`; card says commercial or non-commercial AV-only use | — | NuRec card / API | 2026-10-03 |
| 8.4 | (task) scene contents | INFO | Reconstructed from 6 cameras (front-wide 120°, front-tele 30°, cross L/R 120°, rear L/R 70°). USDZ holds a 3DGUT checkpoint, ClipGT parquets (lanes, obstacles, drivable space…), `map.xodr`, meshes. `clip_ratings_26.04.csv` carries a beta quality score | — | NuRec card | 2026-10-03 |
| 8.5 | (task) scenario counts NVIDIA evaluates on | INFO | AR1 v2: `920`. A1 card: `910`. A1.5 card: `910` → `913` (7 Aug 2026). A2S card: `913` | None equals 1,607; the lists are unpublished | sources in rows 3.6, 4.3, 4.12, 5.5 | 2026-10-03 |
| 8.6 | 04: "NuRec renders 100 FPS at 1080p × 6 cams on RTX 6000 Pro" | VERIFIED (wording ambiguous) | NuRec docs (paraphrase): NVIDIA's six-camera benchmark clip (`car2sim_6cam.yaml`) on an RTX6000Pro reaches 3.33 reconstructions per hour and 100 FPS rendering at 1080p | Does not say whether 100 FPS is the aggregate over 6 cameras (04 assumes aggregate) or per camera | https://docs.nvidia.com/nurec/best-practices/scaling-neural-reconstruction.html | 2026-10-03 |
| 8.7 | 02 §7 / 03 §7: "Instant NuRec (arXiv 2607.14203)" | VERIFIED | arXiv: NVIDIA (Jiahui Huang + 14); v1 `15 Jul 2026`. Feed-forward 3DGS from a multi-view log in one forward pass; a 10–20 s scene in about 1.5 s; +2.01 dB PSNR on Waymo over the best baseline; integrated into NuRec and AlpaSim-compatible | Weights: HF `nvidia/instant-nurec`, not gated, NVIDIA Open Model License, created 2026-05-12. Code availability not checked | https://arxiv.org/abs/2607.14203 ; https://huggingface.co/api/models/nvidia/instant-nurec | 2026-10-03 |
| 8.8 | 02 §7 / 04: "OmniDreams (arXiv 2606.03159 / html v2)" | VERIFIED | arXiv: "NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation"; 33 authors; v1 `2 Jun 2026`, v2 `23 Jul 2026`, v3 `23 Sep 2026`. Abstract (paraphrase): mid- and post-trained from Cosmos on 21k h of driving; runs closed-loop with Alpamayo 1 and AlpaSim; a world-action model built on it beats Alpamayo 1.5 with 1/5 the parameters | Docs cite v2; v3 is now the latest | https://arxiv.org/abs/2606.03159 | 2026-10-03 |
| 8.9 | 04: "OmniDreams (68 FPS single-view on GB300)" | VERIFIED | v2 and v3 (paraphrase): the single-camera 2B model renders 68 FPS at 720p (704×1280) on one GB300; the 4-camera model reaches up to 105 FPS per camera on 16 GB300s. Table 2: `118 ms` per 8-frame chunk = `68` effective FPS. Table 3 (4-view): `12` FPS on 1 GPU, `105` on 16 | — | https://arxiv.org/html/2606.03159v3 §5, Tables 2–3 | 2026-10-03 |
| 8.10 | 02 Week 1: "OmniDreams weights" are gated | VERIFIED | HF `nvidia/omni-dreams-models`: `gated: auto`, NVIDIA Open Model License, base `nvidia/Cosmos-Predict2.5-2B`. Datasets: `omni-dreams-samples` (gated auto), `omni-dreams-scenes` (not gated) | Automatic approval, so immediate | https://huggingface.co/api/models/nvidia/omni-dreams-models | 2026-10-03 |
| 8.11 | 02 G2: OmniDreams as a second renderer | VERIFIED (exists in AlpaSim) | alpamayo-recipes README (paraphrase): AlpaSim has two rendering backends, Omniverse NuRec and OmniDreams | — | https://github.com/NVlabs/alpamayo-recipes | 2026-10-03 |

### Inferences
- For G2's "renderer agreement (NuRec vs OmniDreams)": the policies use 4 cameras, and OmniDreams 4-view manages only about 12 FPS per camera on one GB300 (real time needs 16 GB300s). On rented single GPUs an OmniDreams tier will be well below real time and GB300-only. Budget it as an offline slice, not the main tier.
- The 920, 910 and 913 scenario sets are presumably subsets of NuRec releases. With only 1,607 public scenes, the challenge split quoted in 02 (1,761/441 = 2,202) cannot be drawn from the public 26.04 release alone. That is a sibling cross-check.

### Gaps
- How NVIDIA's 913 evaluation scenes map onto the 26.04 release.
- Whether NuRec's 100 FPS is aggregate or per camera.
- Instant NuRec code license.

---

## 9. Cosmos-Reason2 and Cosmos 3

### Takeaway
- **Cosmos-Reason2** exists in 2B, 8B and 32B sizes, all Qwen3-VL-based, under the NVIDIA Open Model License. The 2B and 8B are gated with automatic approval; the 32B is not gated. Initial release 19 Dec 2025, updated 10 Mar 2026.
- **The Cosmos 3 tech report** (arXiv 2606.02800, v4 23 Jun 2026) exists.
- **Cosmos 3 sizes:** Edge 4B (released 20 Jul 2026), Nano 16B and Super 64B (both 31 May 2026), all OpenMDW-1.1 and ungated. Alpamayo 2 Super's backbone is the 32B "Cosmos 3 Super Reasoner".

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 9.1 | 02 Week 1 / CLAUDE.md: Cosmos-Reason2 is gated | VERIFIED (2B, 8B) | `gated: auto`; license `nvidia-open-model-license` (NVIDIA Open Model License, version dated Sept 23, 2025) | 32B (repo created 2026-04-29) is not gated | https://huggingface.co/api/models/nvidia/Cosmos-Reason2-8B ; …/Cosmos-Reason2-32B | 2026-10-03 |
| 9.2 | (task) sizes | INFO | Parameters: 2B `2,438,696,960`; 8B `8,767,123,696`; 32B `33,357,390,064`. Bases: Qwen3-VL-2B/8B/32B-Instruct | — | HF API | 2026-10-03 |
| 9.3 | (task) release dates | INFO | Card: GitHub `12/19/2025`; HF `12/19/2025` initial release, `03/10/2026` update (better benchmarks, fewer hallucinations) | — | https://huggingface.co/nvidia/Cosmos-Reason2-8B (SHA a9fae2c) | 2026-10-03 |
| 9.4 | 04: "Cosmos 3 tech report arXiv 2606.02800 (v4)" | VERIFIED | "Cosmos 3: Omnimodal World Models for Physical AI"; NVIDIA, 296 authors; v1 `1 Jun 2026` through v4 `23 Jun 2026`. Abstract (paraphrase): mixture-of-transformers; code, checkpoints and data under OpenMDW-1.1 | v4 is the latest | https://arxiv.org/abs/2606.02800 | 2026-10-03 |
| 9.5 | 04: "Cosmos 3 Edge/Nano (4B/16B)" | VERIFIED | Cards: `Cosmos3-Edge: 4B` (released `07/20/2026`); `Cosmos3-Nano: 16B` and `Cosmos3-Super: 64B` (HF `05/31/2026`). License OpenMDW-1.1; not gated | safetensors: 3.86B / 15.75B / 64.6B | https://huggingface.co/nvidia/Cosmos3-Edge (SHA 344d602); https://huggingface.co/nvidia/Cosmos3-Super (SHA f543c56) | 2026-10-03 |
| 9.6 | 04 / A2S: "Cosmos 3 Super Reasoner 32B" | VERIFIED (via Alpamayo sources) | A2S card and blog: a 32B Cosmos 3 Super Reasoner. The Cosmos3-Super card states only `64B` trainable parameters (generator plus reasoner) | Reasoner-only size is not on the Cosmos3-Super card | A2S card; Cosmos3-Super card | 2026-10-03 |

### Inferences
- Cosmos3-Edge (4B) is one *candidate* in the right size class for a small student (G5) or a nanoAlpamayo backbone. No row here establishes that it suits driving, and its reasoner-only size is unknown. Separately, TensorRT-Edge-LLM 0.10.0 (Aug 2026) added Cosmos3-Edge support ([README](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/README.md)).

### Gaps
- Reasoner-only parameter counts for Cosmos3-Nano and Cosmos3-Edge.

---

## 10. Other Alpamayo-family releases (Jun–Oct 2026) and deployment-side Alpamayo claims in the docs

### Takeaway
NVIDIA has released **no student, distilled, edge or "Nano 2" Alpamayo model** as of 2026-10-03; HF shows only the three NVIDIA models. Releases the docs do not mention:
- an `alpamayo2_sft` recipe;
- the open CoC autolabeler repo;
- public OOD CoC labels plus a reasoning challenge and benchmark;
- the NCore subset;
- the family-wide OpenMDW license switch (4 Aug 2026);
- the A1.5 card's protocol change (7 Aug 2026);
- the "Nano" renaming;
- Cosmos3-Edge.

Deployment claims: TensorRT-Edge-LLM FP16-only is verified. FlashDrive's 717 → 151 ms with the expert left in BF16 is verified. The "unanswered OOM report" was not found. The "not available for AGX Thor" forum quote could not be located and looks superseded.

### Cited Findings

| # | Claim as written (doc §) | Status | Evidence | Correct value / note | Primary source | Accessed |
|---|---|---|---|---|---|---|
| 10.1 | (task) releases the docs do not mention | INFO | (a) `recipes/alpamayo2_sft`. (b) `NVlabs/alpamayo-coc-autolabeler` (created 2026-05-24, last push 2026-07-20). (c) PAI-AV OOD reasoning labels, the `PhysicalAI-AV-OOD-Reasoning-Challenge-2026` space, and the PAI-AV Reasoning Benchmark. (d) PAI-AV-NCore (Mar 2026). (e) OpenMDW for the whole family (4 Aug 2026). (f) A1.5 card protocol change (7 Aug 2026). (g) "Nano" renaming. (h) Cosmos3-Edge (20 Jul 2026). (i) TensorRT-Edge-LLM Alpamayo-R1 pipeline for Thor | — | recipes tree; https://api.github.com/orgs/NVlabs/repos ; PAI-AV card; NVIDIA blog 4 Aug 2026 | 2026-10-03 |
| 10.2 | (task) any NVIDIA student, distilled or edge Alpamayo | NOT FOUND | HF `author=nvidia` search "alpamayo" returns 3 models only | Third-party derivatives exist: z-lab `Alpamayo-1.5-10B-DFlash` / `-PARO`; `Renesas/Alpamayo-1.5-10B` (2026-08-05); `sina-1/alpamayo1_5_fp8`; `dima0000/Alpamayo2-Super-NVFP4-language-smoke`; MLX ports; `xiaomi-research/OneVL_AlpamayoR1` [third-party; not evaluated] | https://huggingface.co/api/models?search=alpamayo | 2026-10-03 |
| 10.3 | 02 §0: "TensorRT-Edge-LLM runs Alpamayo-R1 on Thor in FP16 only" | VERIFIED | `docs/.../vla/alpamayo.md` (paraphrase): only FP16 is supported for Alpamayo export; engines are built and run on the Thor device; the model is Alpamayo-R1-10B | — | https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/95515c2f87fba8982db5a519f9022277667b3cc9/docs/source/user_guide/examples/vla/alpamayo.md | 2026-10-03 |
| 10.4 | 02 §0: "…with an unanswered OOM report" | NOT FOUND | GitHub issue search finds 10 Alpamayo issues: export failure #166 (open), engine-build error #92 (open), quantized-export request #134 (open), "Alpamayo Support" #130 (open). None is titled OOM | Issue bodies were not searched | https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=alpamayo | 2026-10-03 |
| 10.5 | 04: Alpamayo is "not available for AGX Thor currently" (NVIDIA forum) | UNVERIFIABLE (post not located); likely superseded | Counter-evidence: the TensorRT-Edge-LLM Thor pipeline (10.3); FlashDrive benchmarks A1.5 on Jetson Thor (4.0× with 1 sample, 9.6× with 6); open issue alpamayo1.5 #22 "Installing Alpamayo 1.5 on Jetson Thor with JetPack 7.2" (2026-06-22) | — | https://arxiv.org/html/2608.12932v1 §4 ; https://github.com/NVlabs/alpamayo1.5/issues/22 | 2026-10-03 |
| 10.6 | 02 §0: "FlashDrive: Alpamayo 1.5 from 717 to 151 ms with W4A8 + speculative decoding, action expert kept in BF16" | VERIFIED (numbers and expert precision) | arXiv 2608.12932 v1 (`13 Aug 2026`): A1.5-10B goes from 717 to 151 ms (4.7×) on an RTX PRO 6000. VLM weights are 4-bit (ParoQuant) with 8-bit activations, and the action expert stays BF16. Speculative reasoning uses a diffusion drafter (accepted length 5.6). A1.5 needs about 31.6 GB in FP16 for 6 samples | Venue "CoRL 2026" not checked. Authors: Zekai Li … Zhijian Liu (affiliations not checked) | https://arxiv.org/abs/2608.12932 | 2026-10-03 |
| 10.7 | C2: "Thor bandwidth is 273 GB/s" | VERIFIED (Jetson AGX Thor only) | Jetson Thor blog spec table: `128 GB 256-bit LPDDR5X, 273 GB/s` (T5000); `64 GB … 273 GB/s` (T4000) | DRIVE AGX Thor not checked | https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/ | 2026-10-03 |
| 10.8 | C2: "challenge limit is 16 GiB and 0.1 s per call" | UNVERIFIABLE (out of scope; not checked) | The challenge space README (406 bytes) does not contain these limits | Sibling researcher | https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026 | 2026-10-03 |
| 10.9 | 04 tensor contract: AlpaGym reward preset (progress 1.0, collision −10, off-road −5, distance-to-GT −0.01) | UNVERIFIABLE (out of scope; not checked) | — | Sibling researcher (AlpaGym) | — | 2026-10-03 |
| 10.10 | 02 §0 / 04: AlpaGym "first commit 2026-06-16" | UNVERIFIABLE (out of scope; cross-check flag) | GitHub API: `NVlabs/alpagym` created `2026-05-19`; the closed-loop blog announcing it is dated 31 May 2026 | A first public commit could still be later than repo creation. Sibling should verify | https://api.github.com/orgs/NVlabs/repos | 2026-10-03 |

### Inferences
- The deployment gap the docs describe is real but narrower than stated. An official Thor path exists (TensorRT-Edge-LLM, FP16, Alpamayo-R1 only). The open requests are quantized export and Alpamayo 1.5 / 2 Super on Thor.

### Gaps
- Whether the cited forum post existed, and its date.
- Whether an OOM report exists in an issue body.
- FlashDrive's peer-review venue.

---

## 11. What is actually true now (2026-10-03): the model family and data a plan can be built on

### Takeaway
Three findings change the plan and should be fixed in the docs before any spending:
- **(1) NVIDIA's own AlpaSim numbers are protocol-dependent and cannot be ranked across cards.** The A1.5 score jumped 0.81 → 1.37 on 7 Aug 2026 with the same weights.
- **(2) The PAI-AV and NuRec licenses contain a confidentiality clause covering benchmarking, regression and performance data**, plus no-derivatives and a 12-month term. This needs clarification before G1–G3 are published.
- **(3) Public CoC labels cover only about 1.7K training clips**, so nanoAlpamayo's CoC-SFT/RL stage needs auto-labels or a smaller claim.

Everything else in the docs about Alpamayo is either verified or a minor correction (expert 2B vs 2.3B, "24 GB", egomotion shape, 10.5B vs 11.08B, step count).

### Cited Findings

**Model family (as of 2026-10-03)**

| Model | Release | Parameters (stated / in checkpoint) | Backbone | Inputs | Card scores (protocol) | License | Hardware |
|---|---|---|---|---|---|---|---|
| Alpamayo 1 ("1 Nano"), `nvidia/Alpamayo-R1-10B` | HF 2025-12-03; renamed at CES 2026-01-05 | 8.2B + 2.3B / 11.08B BF16, 22.16 GB | Cosmos-Reason (config `qwenvl3`) | 4 cameras × 4 frames, 320×576; 16-waypoint egomotion | AlpaSim 0.73 ± 0.01 (910, old); minADE6 1.22 m (937); not RL-trained | OpenMDW-1.1 since 2026-08-04 (was NVIDIA non-commercial) | ≥24 GB GPU |
| Alpamayo 1.5 ("1.5 Nano"), `nvidia/Alpamayo-1.5-10B` | HF 2026-03-19 (GTC) | 8.2B + 2.3B / 11.08B BF16, 22.16 GB | Cosmos-Reason2-8B (Qwen3-VL-8B) | 4 cameras by default, flexible count; navigation text; VQA | AlpaSim 1.37 ± 0.10 (913, new); minADE6 0.916 m (1,434); LingoQA 74.2 | Same | ~24 GB at 1 sample; 717 ms on RTX PRO 6000 (FlashDrive) |
| Alpamayo 2 Super, `nvidia/Alpamayo2-Super` | Announced 2026-06-01; HF 2026-08-04 | 32B + 2.3B (card field) or 2B (card prose and blog), "34B" / 35.81B BF16, 71.65 GB | Cosmos 3 Super Reasoner (Qwen3-VL-32B-class config) | Up to 7 cameras; 6-camera notebook profiles; AlpaSim config uses 4 | AlpaSim 1.50 ± 0.13 (913); minADE6 0.911 m (1,434); LingoQA 79.2 | OpenMDW-1.1, commercial use allowed | 1×H100 80 GB (72,115 MiB peak at batch 1) |

Sources: cards at the pinned SHAs above ([A1](https://huggingface.co/nvidia/Alpamayo-R1-10B/blob/dd4a24cacefc9a6477a6dfc7354de2443401409d/README.md), [A1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B/blob/7aba8293c09993f2e125c6819df05d7fa3e873ea/README.md), [A2S](https://huggingface.co/nvidia/Alpamayo2-Super/blob/00554695e729a6ff0b6281fd2c81b18d06e33dbe/README.md)); [CES press release](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development); [A2S dev blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/); [FlashDrive](https://arxiv.org/abs/2608.12932).

**Shared architecture (A1, A1.5, A2S)** — [AR1 v2](https://arxiv.org/html/2511.00088v2); configs:
- VLM writes CoC text; a flow-matching expert attends to the VLM KV cache.
- Output: 64 waypoints over 6.4 s at 10 Hz, as unicycle (a, κ) controls (a ∈ ±9.8, κ ∈ ±0.33, dt 0.1).
- 128 discrete training tokens per trajectory.
- Euler integration: 10 steps in the paper default and the A2S measurements; the paper's latency table uses 5.

**Data and scene assets**
- **PAI-AV.** 1,700 h; 306,152 × 20 s clips; 25 countries; 7 cameras, LiDAR and radar with partial coverage; 133 TB; v26.09. Automatic gating. AV-only license, commercial or non-commercial, with the §1/§3/§4.6/§8 constraints. — [card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- **CoC labels.** Public OOD subset only: 1,450 train clips / 1,728 labels; 290 val / 349; test held out. — [card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- **NuRec 26.04.** 1,607 scenes, 2.87 TB, 6-camera 3DGUT reconstructions on Hyperion 8.1; same license, automatic gating. — [card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
- **OmniDreams.** 2B, from Cosmos-Predict2.5-2B. 68 FPS single view on 1 GB300; 4-view real time needs 16 GB300s. Weights gated (auto). — [arXiv 2606.03159](https://arxiv.org/abs/2606.03159)
- **Instant NuRec.** About 1.5 s per scene, feed-forward; weights ungated. — [arXiv 2607.14203](https://arxiv.org/abs/2607.14203)
- **Cosmos-Reason2** 2B/8B gated (auto), 32B open; **Cosmos 3** Edge 4B / Nano 16B / Super 64B under OpenMDW. — [HF API](https://huggingface.co/api/models?author=nvidia&search=Cosmos3)

**NVIDIA's positioning**
- Alpamayo models are cloud teachers, critics and autolabelers, distilled into compact in-vehicle models on DRIVE AGX Thor (rows 6.1–6.4).
- No public distillation recipe or student checkpoint exists (row 6.5).

### Inferences
- **G3 must re-measure everything under one pinned protocol.** Published card numbers (0.73, 1.37, 1.50) and paper numbers (0.35, 0.72) come from at least three protocols and cannot be pooled or ranked. Record the AlpaSim version, scenario list, seeds and the 4 m deviation exclusion with every number.
- **nanoAlpamayo acceptance should be restated.**
  - Keep minADE6 ≤ 1.0 m only after defining a public evaluation subset; the 644 list is not public.
  - Replace the Table 9 targets (internal CoC data) with public analogs: the 349-label OOD val set, or auto-labels.
  - Treat 9.0% / 19% / 0.35 as a gate-measured re-baseline of an Alpamayo checkpoint, not a reproduction target (the 0.5B weights are unreleased).
- **Latency.** The 99 ms figure does not describe any released checkpoint. The released A1.5 measures 717 ms (BF16, RTX PRO 6000) and 151 ms with W4A8 plus the full FlashDrive stack. The challenge's "0.1 s per call" (sibling scope) is therefore far out of reach for a 10B model without heavy optimization, which strengthens G5/G6.
- **Teacher costs.** A2S needs one H100 80 GB per instance at batch 1, so G5 teacher labeling is costed per H100-hour with no batching gains unless quantized. A1.5 (22 GB) is the cheap teacher NVIDIA itself recommends for distillation.
- **License.** Model weights are now commercial-friendly (OpenMDW-1.1). The datasets are not freely publishable (§3 confidentiality, §4.6 no derivatives, §8 12-month term). Resolve before the GTC poster deadline.

### Gaps
- NVIDIA's evaluation lists (644, 920, 910/913, 937/1,434) and the AlpaSim versions behind every card number.
- The configuration behind AR1 Table 14.
- Alpamayo 1.5's default flow-step count.
- NVIDIA's reading of the dataset confidentiality clause for third-party publications.
- Cross-checks for sibling researchers:
  - AlpaGym repo created 2026-05-19 vs the docs' "first commit 2026-06-16".
  - The challenge split 1,761 + 441 = 2,202 exceeds the 1,607 public NuRec 26.04 scenes.
  - The challenge's 16 GiB / 0.1 s limits are not verified here.
  - Thor bandwidth (273 GB/s) is verified for Jetson AGX Thor only, not DRIVE AGX Thor.
- Possible new releases at GTC Berlin (20–22 Oct 2026), after this snapshot.
