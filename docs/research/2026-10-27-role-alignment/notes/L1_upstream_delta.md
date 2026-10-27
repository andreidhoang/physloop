# L1 — Upstream delta and race check (2026-10-25 → 2026-10-27)

**Lane.** NVIDIA's open Alpamayo / AlpaSim / AlpaGym / Cosmos / Thor ecosystem and NVIDIA's newest AV research, as a dated delta against the 2026-10-25 ledger (`docs/research/2026-10-25-teacher-to-thor/verification.md`, lanes A and D) and plan v4 (`docs/02_PHYSLOOP_PLAN.md`).

**When and how.** All probes ran on **2026-10-27, 08:20–08:45 UTC**:
- Hugging Face public API;
- the GitHub REST API (unauthenticated) and the GitHub MCP, both read-only;
- the NGC public catalog search API;
- the arXiv API;
- vendor and NVIDIA pages via WebFetch.

Nothing was signed into, no licence was accepted, no gated weights or PAI-AV/NuRec data were downloaded, and nothing was posted. Quotes are a few words at most.

**Structure.** The coordinator's six required sections are the top level. Inside §2 (facts ledger), each key question Q1–Q8 has a one-line **Takeaway**, a facts table, **Inferences** and **Gaps**, as the researcher template asks.

**Raw evidence** (scratchpad, not the repo): `/private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/l1_raw/`. It holds:
- the HF JSON, the AlpaSim issue and comment JSON, and the commit lists;
- the AlpaGym README and default config;
- the AlpaSim Slurm skill and VIDEO_MODEL doc;
- the Cosmos3 cards;
- the NGC search JSON and the arXiv scan;
- the vast.ai offers in `l1_raw/vast_2026-10-27/`. `python3 -I /private/tmp/claude-501/-Users-danghuyhoang-Desktop-physloop/6a1c1084-894a-4933-a900-90a396bc7621/scratchpad/stats_vast.py <that dir>` reproduces the price table. It is the same `stats.py` as the 10-25 ledger and uses the same filters: on-demand, rentable, not rented, all verification states.

**Three things the writer must not miss:**
1. **The earlier ledger's "no external AlpaSim merge since #149 (08-17)" holds for `main` only.** On `e2e_challenge`, PR #167 (author_association NONE) merged 2026-08-31, and three more PRs from non-NVIDIA handles merged 08-24/08-31. Plan §2 G1 already cites #167 as upstream, so the docs contradict themselves. See §3 D1.
2. **AlpaSim #224 is closed (10-23).** NVIDIA tied disclosure of the withheld "simulation choices" to *its next release*, not to the 10-31 close. S2's timing text needs rewording. See §3 D2.
3. **TacitVLA (NVIDIA authors, 10-13) already reports AlpaSim closed-loop results on the 441-scene public split for a 2B-backbone reasoning VLA.** It compares explicit reasoning, "tacit" reasoning (trained with reasoning, never decoded at inference) and no-reasoning training, with H100 latency. This narrows S3's novelty wording and S5(c)'s research-question wording. It does not remove either. See L1-2.

---

## 1. Findings table

JR2010863 lines as defined by the coordinator: I1, I2, D1–D5, Q3–Q5, X1–X3. Kinds:
- **a**: do an existing stage so that it also yields role evidence;
- **b**: design doc plus real measurement;
- **c**: gated proposal;
- **fact update**: no new work, the stage text changes.

Prices are vast.ai on-demand on 2026-10-27 unless stated. "★" marks Huy-owned cores.

| ID | JR lines | Stage · deliverable | Kind | What changes in the stage card | GPU-h and $ delta (price, date) | Hours delta | Gate | Owner | Sources |
|---|---|---|---|---|---|---|---|---|---|
| **L1-1** | X2, D3 | S3 · G5; KC-E | fact update | **KC-E text:** as of 10-27, Edge is not public on any of the three channels checked (HF, NGC, GitHub), and the blog is unchanged ("later this month"). **Add a release-day capture list to KC-E:** repo ID and revision SHA; licence text on evaluation, fine-tuning and publishing derivatives; backbone; action expert and flow steps; whether runtime CoC can be switched off; camera count; disclosed teacher; any NVIDIA closed-loop number. **Pre-write only** the PR #42-pilot experiment file, with prediction slots `____`. No training before KC-E resolves. Kill and order are unchanged | 0 | +0.5 (agent) | KC-E | agent (check); ★ Huy (decision) | M1–M8, M11 |
| **L1-2** | D3, D5, X2 | S3 · G5; S5(c) · G5, G3·R7 | fact update + design input | **S3 Why / Upstream value:** narrow "first closed-loop evaluation of an NVIDIA student" to "first **paired, stratified, student-vs-teacher** closed-loop verdict of an NVIDIA student". Cite TacitVLA (not Edge, not a distilled student, mean ± SD over 3 rollouts, no paired test). **§1 research question / S5(c) Build:** TacitVLA measured the endpoints, not a token-budget sweep, so replace "measured as a curve rather than a binary" with wording that positions against it. **Label the n = 0 point by architecture:** "expert conditioned on an empty CoC" (Alpamayo-style expert that attends to CoC) is not "tacit-trained". **Derivation inputs for Huy only** (no values proposed): TacitVLA scene scores 46.4 / 52.3 / 52.5 %; SD across rollouts 0.3–2.2 pp; H100 reasoning cost 143 ms. **Prediction slot unchanged:** curve shape `____` | 0 | +1.5–2.5 (Huy design; agent cites) | none (KC-L for publication) | ★ Huy | R1, R2 |
| **L1-3** | D3, D4, X2 | S2 · G1, G2 | a | **Build / Manifest:** pin the `e2e_challenge` scorer at or after #215 (merge 1b024f0f, 10-20) and record the scorer-file SHA. Add #198's minimal reproduction as a **known-answer failing test** in S2's determinism suite: two stationary-tail ground-truth (GT) polylines; pre-fix scorer gives 5.0 m lateral, expected ≈ 0. **Note in the report:** the leaderboard was recomputed on 10-22 and a frozen "v1" view is kept, so public scores from before the fix used the buggy scorer. **Validity-record semantics (★ Huy):** whether corridor exits within the GT terminal segment get a flag. **Acceptance:** the test passes on the pinned scorer | 0 | +1–2 (agent) | none | agent (test, manifest); ★ Huy (semantics) | A1, A2 |
| **L1-4** | D3, D4, X2 | S2 · G1 | a (+ optional c) | **Manifest adds:** harmonizer on/off; deployment topology and concurrency (the official run is `8gpu_32rollouts`, 16 replicas × 2); policy clock source (sim time vs wall clock); renderer-OOM re-queue count per rollout. The retry audit now covers re-queues, not only NaN retries. **Card/leaderboard match stays diagnostic:** NVIDIA says the PAI internal test set is a different, harder distribution. **Optional c:** a paired harmonizer on/off sub-sweep is a new variable, so it is its own experiment | 0. Optional sub-sweep: 34–85 GPU-h = $34–133 at $1.00–1.56 (RTX PRO 6000 WS, 10-27) | +1–2 (agent) | P-005 (optional sub-sweep only) | agent; ★ Huy (whether to run) | A3 |
| **L1-5** | D3, X2 | S2 taxonomy; S3 strata; S4 hacking audit · G1, G5, G9 | fact update (★ derivation input) | **Candidate stratum and failure mode for Huy to consider (no values):** a stationary or slow in-path lead. It is documented for Alpamayo 1.5 in native AlpaSim (#224 → alpamayo1.5 #40) and for Super in CARLA (#39). An NVIDIA-coauthored paper finds 1.5's longitudinal plan tracks ego-speed history more than hazard range. **Interaction with the S4 audit:** stalling avoids stationary-lead collisions, so the stall scorer and this stratum must be read together. **Race:** inference-time hazard guidance is NVIDIA-coauthored and G4 stays deferred | 0 | +0.5–1 (Huy) | none | ★ Huy | A4, R3 |
| **L1-6** | D2, D4, X3 | S2 · G7 | a | **"First measured throughput"** uses AlpaSim's own metric: stabilized full-run `seconds_per_rollout`. Rollout production starts about 5 min after launch; the first 10–15 min are congested; the metric stabilizes after 30–45 min. Report alongside it rollouts/GPU-h, per-service GPU utilization, RPC queue depth and DCGM tensor-active. **§5 cost model:** replace the single-stream "34–85 GPU-h per sweep" bound with the measured value. **Add a topology-invariance negative control:** 100 scenes × 1 seed under two concurrency settings must give the same verdict within Huy's tolerance `____`. This tests NVIDIA's #222 caveat that concurrency and clock are not guaranteed neutral | +2.6–6.4 GPU-h ≈ $3–10 at $1.00–1.56 (RTX PRO 6000 WS, 10-27) | +4–6 (agent) +0.5 (Huy) | P-005 (inside the S2 tier) | agent; ★ Huy (tolerance) | A5, A3 |
| **L1-7** | Q5, D4, D1 | S4 (and S2) · G9, G7 | a | **Run S4's one-node GRPO through AlpaGym's shipped Slurm path** (config-level only) on a single-node Slurm install: `deploy=slurm` with `topology=slurm_full_node_1_3_4` (one 8-GPU node: 1 policy, 3 rollout, 4 AlpaSim GPUs) or `slurm_partial_node_1_2_1` (4 GPUs). Optionally run S2's sweeps through AlpaSim's wizard Slurm deployment as well. **Measure autoresume:** force one time-limit requeue and check checkpoint continuity (step counter and reward curve). **Kill:** if not running after 8 agent-hours, revert to the local backend and keep an issue-ready failure note. AlpaGym's README itself says the Slurm presets are not expected to run out of the box | Bring-up: **16 GPU-h** ($16–25; 4 × RTX PRO 6000 WS for 4 h at $1.00–1.56) or **16–32 H100-h** ($64–128; 8 × H100 node for 2–4 h at $32.00/h), 10-27. The GRPO run itself is unchanged | +8–12 (agent) | P-005 | agent; Huy approves | B1, B2, B3, A5 |
| **L1-8** | D1, Q5, X1, D4 | S4 · G9 | **c** | **Gated proposal: one multi-node scaling measurement.** Run the same AlpaGym config on 1 node (`slurm_full_node_1_3_4`) and on the smallest shipped multi-node preset, `slurm_distributed_1_1_1`. That preset is **3 nodes / 24 GPUs**: one policy, one rollout and one AlpaSim node. A 2-node run needs a custom topology YAML, which is still config-level. Run 2–3 h at steady state each. **Measure:** rollouts/h, trainer idle %, NCCL transfer time. **★ Prediction slot:** scaling efficiency `____`. **Where:** RunPod Instant Clusters are the smallest self-serve multi-node option (2–8 nodes, per-second billing, no minimum lease, Slurm supported). H100/B200 cluster prices are "contact sales"; H200 is listed at $6.35 with the unit unstated | **2 nodes:** 32–48 GPU-h, ≈ $101.6/h → $203–305 for 2–3 h. **3 nodes:** 48–72 GPU-h, ≈ $152.4/h → $305–457. **Plus a 1-node baseline** if it is not already an S4 run: 16–24 GPU-h (8 GPUs × 2–3 h), $64–96 at vast 8 × H100 $32.00/h. Assumes $6.35 per GPU-hour (RunPod pricing page, 10-27) | +10–16 | P-005 (new budget line). Multi-node is c only | ★ Huy (prediction, go/no-go); agent | P3, P4, B2 |
| **L1-9** | D4 | S2 · G1 | fact update | **#207 (lockfile)** is still open with no NVIDIA reply since 10-14. **#208 (n_sim_steps)** was acknowledged by NVIDIA on 10-15 (add requested and actual to `results-summary`). The reporter offered a PR; none is filed as of 10-27. **Upstream value line:** do not file a competing PR. Keep both fields in PhysLoop's manifest, and switch to the upstream field if it lands | 0 | 0 | none | agent | A6, A7 |
| **L1-10** | D4, X1, X3 | S4, S5 · G9, G5 | a | **Training logs use NVIDIA's draft measurement definitions** (recipes PR #44, 10-21): wall-clock s/step from log timestamps; GPU-hours from the **mean** wall clock (never the median); MFU, HFU and OFU; the CUDA-event split into arrival, forward, backward and host residual; per-rank stall detection. This makes PhysLoop's training-efficiency numbers comparable with NVIDIA's. **Race:** NVIDIA's chapter lists regression bands, a benchmark harness and a profiler as coming, so do not build a generic harness | 0 | +2–3 (agent) | none | agent | C1 |
| **L1-11** | D5, Q3 | S5(a) · G5 | fact update (race + derivation input) | **Cite OPTED** (09-17, NVIDIA authors; code "coming soon"). A privileged RL teacher on vectorized inputs supervises camera students on-policy in AlpaSim. It reaches 95 % of the teacher's scene score in about 6.8k closed-loop episodes, vs about 6.2M for RL fine-tuning of the same student. **Setting of that comparison unverified:** it is likely the vectorized-student WOMD controlled study, not AlpaSim camera students (6.2M rendered AlpaSim episodes would be implausible). Do not use it for S5 sizing until checked. **S5(a) stays as written:** the released VLA teacher labels the states the student visits. Applying OPTED to an Alpamayo or Edge student is a plausible NVIDIA follow-up, so PhysLoop should not. **Derivation input** for Huy's S5 pilot sizing: OPTED's episode counts, only once their setting is confirmed | 0 | +0.5 | none | ★ Huy | R4 |
| **L1-12** | D3, X2 | S3, S6 · G5, G6 | a | **Manifest:** pin and record the controller and any trajectory post-processing (identity and version). **Diagnostics:** per-rollout motion-limit-violation diagnostics; Huy defines the semantics. Together these split a paired student-vs-teacher (or format-vs-format) difference into plan geometry and controller tracking vs decisions. **Evidence:** ECO (09-25, NVIDIA authors) raised AlpaSim scene scores by +123 % for VaVAM and +22 % for DiffusionDrive with no policy change | 0 (computed from logs) | +1–2 (agent) | none | agent; ★ Huy | R5 |
| **L1-13** | Q4 (context) | S6 · G6, G7 | fact update | **S6 Why unchanged:** no TensorRT-Edge-LLM release after v0.11.0 (09-29); PR #260 still open ("blocked", last updated 10-21); JetPack 7.2.1 still newest (TensorRT 10.16.2); TensorRT 11.4.0 is current (DRIVE). **Note for the release day:** TensorRT-Edge-LLM has supported **Cosmos3-Edge** (4B MoT) since v0.10.0, which matters only if Edge turns out to be built on it (unknown). Re-check at S6 start (12-07) | 0 | 0 | P-002 unchanged | agent | T1–T5, M10 |
| **L1-14** | — (licence) | S3 fallback, S5 · G5 | fact update | **Licence of the KC-E fallback student:** the PR #42 student's VLM, `nvidia/Cosmos-Reason2-2B`, is under the **NVIDIA Open Model License**, not OpenMDW-1.1. The fallback student, and S5's prerequisite that the licence permits fine-tuning, must be read against that licence. If derived weights are ever to be published, add it to the P-004 letter scope | 0 | +0.5 (Huy) | KC-L2 / P-004 | ★ Huy | M9 |
| **L1-15** | — (budget) | §5; S2–S5 | fact update | **Price basis 10-27** (§2 Q8). Stage costs at 10-27 prices (inference):<br>• S2: $136–531 (was $136–544).<br>• S3: $102–531, plus the Super sweep at $61–204.<br>• S4: training $402–1,080 (was $434–1,080); evaluation $341–1,327.<br>• **S5:** an 8 × H100 node lists at $31.92–36.00/h. These are list prices; 24–48 h availability was verified only for the vast $32.00 offer. A RunPod community 8-GPU pod at $21.52/h would be cheaper, but its existence is unverified. That gives **$516–1,728** for 24–48 h (was $360–1,536). Today's cheap $19.20/h vast offer shows 0.2 days maximum rental and `bw_nvlink 0.0`, so it is unusable and suspect | S5: +$156 (low end, with the unverified $21.52) to +$192 (high end). Using only the availability-verified vast $32.00 offer, the low end is $768, i.e. +$408 | 0 | P-005 re-price | agent | P1–P5 |
| **L1-16** | X2, D2 | S2 · G1; docs | fact update (doc correction) | **Correct the merge claim** in `CLAUDE.md`, `AGENTS.md`, `04` §5 and the ledger: "no external merge since 08-17" is true for `main` only. On `e2e_challenge`, #167 (author_association NONE) merged on 08-31; #111, #161 and #169 from non-collaborator handles merged 08-24/08-31 (their association is unverified). `main` has had no commit since 08-18; `e2e_challenge` none since 10-20. **S2 Timing text:** replace "re-read AlpaSim for post-challenge disclosures" with "check for a new AlpaSim release or sync at S2 start; if none, label results provisional" (#224: disclosure at "next release") | 0 | +0.5 (doc edit under the contract; `CLAUDE.md` and `AGENTS.md` change together) | none | agent drafts; Huy approves | A2, A4, A8 |
| **L1-17** | I2 | none (G3·R2 deferred) | fact update | **No plan change; record for the I2-coverage discussion.** Cosmos3-Nano-Sim-Auto (10-20; OpenMDW-1.1; not gated; 16B) is pitched for closed-loop policy evaluation but is **not integrated in AlpaSim**. AlpaSim's generative-renderer path is OmniDreams via FlashDreams (`deploy=managed_flashdreams`), which needs 96 GB VRAM with Alpamayo 1.5. See §5 for why a renderer arm is rejected | 0 | 0 | none | — | W1–W4, W7, A9 |

**Totals.** The a-kind and fact-update rows (L1-1…L1-7, L1-9…L1-16) add about **+21–33 focused hours**, about 7–10 % of the ~320 h plan. Their GPU cost is about **$19–138**:
- L1-6: $3–10;
- L1-7 bring-up: $16–128;
- L1-4's optional sub-sweep: $34–133 more, if Huy runs it.

In GPU-hours, the a-kind rows add about 2.6–6.4 GPU-h (L1-6), 16 GPU-h or 16–32 H100-h (L1-7) and, optionally, 34–85 GPU-h (L1-4).

The one c row (L1-8) adds +10–16 h, **32–72 GPU-h (plus 16–24 for a 1-node baseline if it is not an S4 run) and $203–457 (+$64–96)**, gated on P-005.

---

## 2. Facts ledger (by key question)

Confidence: H = read directly from a primary source today; M = automated reading of a long primary page, or a secondary source; L = snippet or unverified. P = primary source, S = secondary source. All access dates are 2026-10-27 unless stated.

### Q1 — Alpamayo 2 Edge (decides KC-E on 10-31); Alpamayo 2 Super

**Takeaway.** Edge is still **not public** on 10-27, six days after the announcement and four days before the promised "later this month" window closes. No repo ID, card, licence, size, architecture, latency, closed-loop number or teacher is published, and NVIDIA has published no closed-loop evaluation of Edge against its teacher. The first public hint that a 2B Alpamayo 2 checkpoint exists internally is in an NVIDIA paper (AD-Memo).

| ID | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| M1 | The HF API search (author=nvidia, "alpamayo") returns 3 models only: `nvidia/Alpamayo-R1-10B` (created 2025-11-22), `nvidia/Alpamayo-1.5-10B` (2026-03-03) and `nvidia/Alpamayo2-Super` (2026-05-27). Their last modifications are 08-04 and 08-07. This is unchanged since the 10-25 ledger | https://huggingface.co/api/models?author=nvidia&search=alpamayo&full=true | P | H |
| M2 | Nine guessed IDs return **HTTP 401** (not public, or nonexistent) at 08:20 UTC: `nvidia/Alpamayo2-Edge`, `Alpamayo-2-Edge`, `Alpamayo2-Edge-2B`, `Alpamayo-2-Edge-2B`, `Alpamayo2Edge`, `Alpamayo-Edge`, `Alpamayo2-Edge-BF16`, `Alpamayo2-Edge-FP8` and `Alpamayo-2-Edge-NVFP4` | `https://huggingface.co/api/models/nvidia/<id>` | P | H |
| M3 | The newest nvidia models on HF by creation date: the newest is `nvidia/MotionBricks` (10-23). None is Edge-like. A global "alpamayo" search shows only community re-uploads | https://huggingface.co/api/models?author=nvidia&sort=createdAt&direction=-1&limit=60 | P | H |
| M4 | The NGC public catalog has **0 Alpamayo model resources**. Containers: `nim/nvidia/alpamayo-1-5-10b` (created 07-29, modified 08-26) and `nim/nvidia/alpamayo2-super` (created 10-19, modified 10-20). There are no collections or Helm charts | https://api.ngc.nvidia.com/v2/search/catalog/resources/MODEL?q={"query":"alpamayo"} (and CONTAINER, COLLECTION, HELM_CHART) | P | H |
| M5 | GitHub NVlabs has 5 alpamayo repos, none for Edge:<br>• alpamayo-recipes (pushed 10-21);<br>• alpamayo2 (pushed 10-07);<br>• alpamayo1.5 and alpamayo (pushed 09-09);<br>• alpamayo-coc-autolabeler (07-20) | GitHub repository search `alpamayo org:NVlabs` (MCP, read-only) | P | H |
| M6 | NVIDIA's Alpamayo page lists Edge as a 2B VLA for in-car reasoning, with no date, licence or link. Its "Download on Hugging Face" button points to Alpamayo2-Super. developer.nvidia.com/drive/alpamayo 301-redirects to this page | https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/ | P | M |
| M7 | The 10-21 blog still promises GitHub/HF availability "later this month", calls Edge "optimized for" DRIVE AGX Thor and "built on NVIDIA Cosmos", and does not mention distillation. It has no update or editor's note on 10-27 | https://blogs.nvidia.com/blog/open-models-europe-industry/ | P | M |
| M8 | The HF collection `nvidia/alpamayo` holds R1-10B, 1.5-10B, PAI-AV and NuRec. Its lastUpdated 10-20T13:09 matches every other nvidia collection, so it is a bulk timestamp (inference) | https://huggingface.co/api/collections?owner=nvidia&q=alpamayo | P | H |
| M9 | `nvidia/Cosmos-Reason2-2B` (the PR #42 student VLM) carries the **NVIDIA Open Model License**, `gated: auto`, last modified 2026-04-30 | https://huggingface.co/api/models/nvidia/Cosmos-Reason2-2B | P | H |
| M10 | `nvidia/Cosmos3-Edge` is a **4B** Mixture-of-Transformers (MoT) model under OpenMDW-1.1, released on HF 07-20. Its reasoner, in eager Transformers on Jetson AGX Thor T5000 (MAXN), takes 3.17–3.60 s end to end for ~900–1,700-token prompts. TensorRT-Edge-LLM has supported it since v0.10.0 (08-12) | https://huggingface.co/nvidia/Cosmos3-Edge ; T1 | P | H |
| M11 | The AD-Memo paper (NVIDIA authors, 09-29) builds on a 2B, single-camera-focused Alpamayo 2 checkpoint with no action expert, and reports **open-loop** results only | https://arxiv.org/abs/2609.38641 (HTML v1) | P | M |
| M12 | The Alpamayo 2 Super card has not changed since 08-07; its AlpaSim score is 1.50 ± 0.13 on 913 scenarios, per the ledger. A Super NIM container was published on NGC (10-19/10-20). No new Super weights | M1; M4 | P | H |
| — | **No NVIDIA closed-loop evaluation of Edge, or of any distilled student, against its teacher was found** in: HF cards; NVlabs repos and PRs; arXiv API searches for Alpamayo/AlpaSim since 09-01; NVIDIA blog search; the AVG publications page. The nearest is TacitVLA (R1): NVIDIA authors, Qwen3-VL-2B backbone, not Edge, not stated to be distilled | R1, R10 | P | H |

**Inferences:**
- 2B parameters plus "built on Cosmos" fits Cosmos-Reason2-2B, the PR #42 student VLM, better than Cosmos3-Edge, which is 4B. Unverified. If true, Edge would carry the NVIDIA Open Model License terms of its base, not OpenMDW.
- TacitVLA shows NVIDIA researchers obtaining reasoning benefits without decoding the reasoning at runtime, at H100 latency 127 vs 269 ms. If Edge follows that design, S3's "reasoning on vs off" arm and S5(c)'s runtime-budget sweep may have no runtime switch to vary on Edge. This is speculation to be checked at release.
- AD-Memo's internal 2B Alpamayo 2 checkpoint is single-camera-focused. If Edge is related, its camera contract may differ from the 4-camera assumption in the owner's targets (p99 with 4 cameras). Unverified.

**Gaps:**
- Edge's repo ID, size, backbone, action expert, flow steps, reasoning switch, licence terms, latency, closed-loop numbers and teacher: none are published. Where I looked: M1–M7, plus the NGC catalog and NVlabs.
- Alpamayo 2 Super has no change to report.

### Q2 — NVlabs/alpasim (`main`, `e2e_challenge`, issues, PRs, challenge)

**Takeaway.** There has been **no commit on any branch since 2026-10-20** (`pushed_at` 10-20T13:35). Activity since 10-25 is challenge logistics plus three substantive NVIDIA statements:
- a scorer bug was fixed and the leaderboard recomputed;
- the internal test set is harder than the public one and the harmonizer will be on for final runs;
- simulation choices that affect inference are withheld until the next release.

The ledger missed three things that ship on `main`: Slurm tooling, a throughput skill, and a generative-renderer mode.

| ID | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| A8 | **Commits.**<br>• `main`: last commit `affc2eab` on 2026-08-18 (#154, CI); none since.<br>• `e2e_challenge`: two commits since 10-03, both on 10-20: `1b024f0f` (#215, lateral-corridor bugfix) and `e121e591` (#217, "Enable report uploading"). #217 is absent from the ledger.<br>• Repo `pushed_at` is 2026-10-20T13:35:17Z | https://api.github.com/repos/NVlabs/alpasim/commits?sha=main&since=2026-08-01 ; …?sha=e2e_challenge&since=2026-10-03 ; https://api.github.com/repos/NVlabs/alpasim | P | H |
| A8b | **External merges.**<br>• #149 (08-17) was the last non-NVIDIA merge to `main`.<br>• On `e2e_challenge`, **#167** (01Dami23, author_association **NONE**; curated 1,761/441 NuRec split) merged 2026-08-31.<br>• Also merged: #111 (WCJ-BERT, 08-24), #161 (hctian713, 08-24) and #169 (hctian713, 08-31). Their association was not verified; the handles are not mwatson-nvidia or jarcherNV.<br>• Open community PRs to `main` (#127, #129, #148, #179 by amtellezfernandez) are unmerged.<br>• #176 ("Apptainer support") closed unmerged 09-07 | https://github.com/NVlabs/alpasim/pull/167 ; PR list via MCP (sort=updated) | P | H (#167) / M (others) |
| A1 | **#198 (closed 10-22).** Jitter in a stationary GT tail made forward overshoot count as a lateral corridor exit (≥ 4 m), forcing a scene score of 0.<br>• NVIDIA's fix extrapolates the final orientation from the last 1 m of real movement.<br>• Metrics were **recalculated**, a frozen "v1" view stays visible, and all future submissions include the fix.<br>• The issue contains a minimal reproduction (two LineString cases that return 5.0 m lateral) | https://github.com/NVlabs/alpasim/issues/198 | P | H |
| A2 | **#215** merged 10-20 into `e2e_challenge` (the corridor fix); **#217** merged 10-20 (report uploading) | https://github.com/NVlabs/alpasim/pull/215 ; /pull/217 | P | H |
| A3 | **#222 (closed 10-22).** A team reports leaderboard scores 0.10–0.15 below local scores, and at-fault distance 3–4× lower on the leaderboard. NVIDIA's answer (mwatson) is that this is expected for the PAI track, because the internal test set comes from a different, harder distribution. Further points from the answer:<br>• a policy using the system clock rather than sim time could differ, with no guarantee;<br>• rendering version and defaults match, except topology;<br>• the **harmonizer will be enabled for final runs**;<br>• renderer OOMs are retried by putting the scene back on the queue;<br>• the official topology is `8gpu_32rollouts` (16 replicas × 2) | https://github.com/NVlabs/alpasim/issues/222 | P | H |
| A4 | **#224 (closed 10-23, "completed").** A stationary-lead collision report for Alpamayo 1.5 in native AlpaSim (alpamayo1.5 #40) and for 2 Super in CARLA (#39). NVIDIA closed it as not actionable. It said recently discovered simulation choices can affect inference, to be disclosed "until our next release" because the competition is ongoing | https://github.com/NVlabs/alpasim/issues/224 | P | H |
| A6 | **#207** (lockfile) is open, created 10-14, with **0 comments**. No NVIDIA reply | https://github.com/NVlabs/alpasim/issues/207 | P | H |
| A7 | **#208** (`n_sim_steps`) is open. NVIDIA collaborator maximilianigl said on 10-15 that the field ideally would be `max_n_sim_steps` and that adding it to `results-summary` makes sense. The reporter offered a PR on 10-15; none exists as of 10-27 | https://github.com/NVlabs/alpasim/issues/208 | P | H |
| A5 | **Slurm tooling on `main`** (ledger miss):<br>• the wizard's `deployment/slurm.py` and `slurm_enroot.py`;<br>• `src/tools/run-on-slurm/submit.sh` and `resume_slurm_job.sh`;<br>• `telemetry/slurm_process_exporter.py`;<br>• an agent skill, `.agents/skills/optimize-slurm-topology/`, synced 2026-07-01 (#114).<br>The skill defines the primary throughput metric as the stabilized full-run `seconds_per_rollout`: early congestion lasts 10–15 min and the metric stabilizes at 30–45 min. Its bottleneck signal is `rpc_queue_depth_at_start` max/min, plus DCGM tensor-active and GPU utilization. It references NVIDIA's internal IAD Lustre paths | https://github.com/NVlabs/alpasim/blob/main/.agents/skills/optimize-slurm-topology/SKILL.md ; …/references/metrics.md ; code search `slurm repo:NVlabs/alpasim` | P | H |
| A9 | **Generative renderer on `main`** (ledger miss): a "Video Model Renderer" that plugs NVIDIA OmniDreams into the renderer endpoint via FlashDreams (`deploy=managed_flashdreams`). It needs **48 GB VRAM with VaVAM and 96 GB with Alpamayo 1.5**, or a separate machine. FlashDreams images must be built locally. There is **no Cosmos3 integration**: a code search finds "cosmos" only in data scripts and the tutorial | https://github.com/NVlabs/alpasim/blob/main/docs/VIDEO_MODEL.md ; code searches | P | H |
| A10 | **#72 (closed "not_planned" 10-18).** NRE is a proprietary NGC container, obfuscated intentionally. NVIDIA runs it through Slurm `srun` with enroot. There is no short-term path for Kubernetes without Docker-in-Docker; staff suggested running NRE as a separate pod | https://github.com/NVlabs/alpasim/issues/72 | P | H |
| A11 | **#219 (nuPlan/MTGS track).** The RPC leaves the distortion coefficients empty while the rendered images keep lens distortion. In the PAI track, all cameras are f-theta with parameters filled. The issue was closed on 10-23; a follow-up comment from 10-26 asking how drivers should get per-session coefficients is unanswered. NVIDIA shares diagnostic summaries, not logs | https://github.com/NVlabs/alpasim/issues/219 | P | H |
| A12 | **New since 10-25:** #225 (nuPlan warm-up allowance; closed 10-26) and #226 (tech-report workflow; closed 10-26). Logistics only | issues API, since=2026-10-15 | P | H |
| A13 | **Challenge.** The public leaderboard closes and final submissions and technical reports are due 2026-10-31. The page makes no post-challenge disclosure statement. The ledger had nuPlan due 11-16 | https://opendrivelab.com/alpasim2026 | P | M |
| — | **Not found** in AlpaSim since 10-25 (no commits): paired or statistical comparison; new seeds work; stall, oscillation or close-encounter scorers; CAT-K changes; Cosmos3 integration; Kubernetes run method | A8 | P | H |

**Inferences:**
- Both NVIDIA leaderboard events (the #215 fix with recomputed metrics, and the harmonizer turned on for final runs) are evaluator or protocol changes inside the challenge window. That strengthens S2's premise that instrument noise and protocol must be measured and pinned.
- vast.ai pods are containers; AlpaSim's docker-compose/NRE stack and a Slurm daemon need a VM-type instance. This is an inference from #72 and the vastai VM/pod split, to be verified at bring-up.

**Gaps:**
- The withheld "simulation choices" are unknown until the next AlpaSim release.
- Author association for #111, #161 and #169 was not checked: the unauthenticated API quota ran out.

### Q3 — NVlabs/alpagym

**Takeaway.** There has been **no commit since 2026-07-21** (docs; code last changed 07-08). The ledger lacked the scale facts:
- Slurm presets exist, from a 4-GPU partial node up to 5 × 8-GPU nodes;
- the smallest multi-node preset is 3 nodes;
- Slurm autoresume ships;
- the README says the Slurm presets will not run out of the box.

There is no roadmap section and no published closed-loop gains.

| ID | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| B1 | **Commits.** The latest is `972d160e` on 07-21 (#5, docs). The 07-06/07-08 sync added: Slurm autoresume (requeue on time limit plus checkpoint resume), NCCL for the remaining presets, batched deterministic rollouts, and fetching the pinned AlpaSim commit by SHA. **README:** the default 10B model needs two GPUs; `deploy=slurm` and the `topology=slurm_*` presets are starting points, not expected to run out of the box (partition, account, image, mounts and wizard preset must be set). The host control plane runs on the login node | https://github.com/NVlabs/alpagym (commits via MCP) ; https://github.com/NVlabs/alpagym/blob/main/README.md | P | H |
| B2 | **Topology presets:**<br>• `local_colocated_1gpu` and `local_disaggregated_2gpu`;<br>• `slurm_partial_node_1_2_1`: 4 GPUs (1 policy, 2 rollout, 1 AlpaSim);<br>• `slurm_full_node_1_3_4`: one 8-GPU node (1 policy, 3 rollout, 4 AlpaSim);<br>• `slurm_distributed_1_1_1`: **3 × 8-GPU nodes**, the smallest multi-node preset;<br>• `slurm_distributed_1_2_2`: 5 nodes;<br>• `slurm_distributed_shared_cosmos_2_3`: 5 nodes. Its comments say AlpaSim rendering is the per-node bottleneck and cite a "0.42 baseline". It uses `control_timestep_us` 200,000, `n_sim_steps` 30 and 22 trainer replay rows | https://github.com/NVlabs/alpagym/tree/main/packages/host/src/alpagym_host/conf/topology | P | H |
| B3 | **Defaults:** `reward: distance_to_gt`; `expected_valid_steps` 8; control timestep 100 ms; GRPO ratio clip 0.2/0.2; `kl_beta` 0.0; `optm_lr` 1e-6; `n_generation` 1; `batch_size` 1 | https://github.com/NVlabs/alpagym/blob/main/packages/host/src/alpagym_host/conf/default.yaml | P | H |
| B4 | **Closed-loop blog (05-31):**<br>• AlpaGym is designed to scale from a single GPU to multi-node clusters without code changes;<br>• tested on 2 × RTX 6000 Ada (50 GB);<br>• no throughput figures and no closed-loop RL gains;<br>• a distillation script to a smaller single-GPU checkpoint is "planned" | https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/ | P | M |

**Inference.** The "0.42 baseline" in B2 is unlabelled in the preset. It is probably a throughput or utilization figure, but its unit is unknown, so do not cite it as a number.

**Gaps:**
- No AlpaGym roadmap was found.
- External PR and issue counts were not re-queried: the ledger had 0 and 0, and there is no activity since 07-21.

### Q4 — NVlabs/alpamayo-recipes

**Takeaway.** PRs #42, #43 and #45 are still open and unchanged since 10-20/10-21. The only new PR the ledger lacks is **#44**, NVIDIA's training-efficiency "Measurement" chapter. Nothing has merged since #41 (10-16), and there are no new RL, closed-loop, Edge or Thor PRs.

| ID | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| C2 | **#42** (distillation): open, last updated 2026-10-20 23:31 | https://github.com/NVlabs/alpamayo-recipes/pull/42 | P | H |
| C3 | **#45** (5 Hz inference for the 2B AM1.5/AM1 students): open, created and last updated 2026-10-21 17:58 | https://github.com/NVlabs/alpamayo-recipes/pull/45 | P | H |
| C4 | **#43** (speculative-decoding drafts): open, updated 10-20. **#18** (AlpaGym RL): still open, last updated 06-30. **#41** merged 10-16; no merge since | PR list via MCP | P | H |
| C1 | **#44** (open, opened 10-21, updated 10-22, author annz96, "blocked"). It adds chapter 05 "Measurement" of the training-efficiency docs. Items 1–5:<br>• per-rank telemetry and severe-stall detection;<br>• wall-clock s/step from log timestamps;<br>• GPU-hours from the **mean** wall clock;<br>• MFU, HFU and OFU (OFU via `nvidia-smi dmon --gpm-metrics`);<br>• a CUDA-event split into arrival, forward, backward and host residual.<br>Items 6–10 are placeholders: regression bands, forensics, benchmark harness, per-component loss logging, profiler. The internal reference numbers come from a 2B model on 16 nodes. Its sibling #38 (overview and data loading) merged 09-23 | https://github.com/NVlabs/alpamayo-recipes/pull/44 | P | H |

### Q5 — TensorRT-Edge-LLM; Jetson AGX Thor; DRIVE AGX Thor

**Takeaway.** Nothing changed. TensorRT-Edge-LLM's latest release is still v0.11.0, PR #260 is still open, JetPack 7.2.1 (TensorRT 10.16.2) is still the newest JetPack, and TensorRT 11.4.0 is current. No newer DriveOS could be confirmed.

| ID | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| T1 | The latest release is **v0.11.0 (2026-09-29)**. It does not mention Alpamayo. It adds native video input for the Cosmos3-Edge reasoner and Thor-optimized NVFP4-A16 kernels. v0.10.0 (08-12) added Cosmos3-Edge reasoning plus experimental policy and action generation. There is no release after v0.11.0 | https://github.com/NVIDIA/TensorRT-Edge-LLM/releases | P | H |
| T2 | **PR #260** (Alpamayo fused gate/up plugin for SM110; head on fork `jin-yc10`) is open, `mergeable_state` "blocked", last updated 10-21 17:39 | https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/260 | P | H |
| T6 | **New PRs:** #261 (port pre-commit to GitHub Actions; updated 10-27) and #262 ("wait for trusted internal pipeline status"; updated 10-26) are NVIDIA CI work. #251 (external; fixes the Cosmos3-Edge visual engine patch layout, #232) is open | PR list via MCP | P | H |
| T3 | **JetPack 7.2.1** is the only JetPack listed: Jetson Linux 39.2.1 (dated 2026-08-11), CUDA 13.2.2, TensorRT 10.16.2, cuDNN 9.20.0. It covers the Jetson AGX Thor dev kit, T5000 and T4000. No 7.2.2, 7.3 or 8.x is mentioned | https://developer.nvidia.com/embedded/jetpack/downloads ; https://developer.nvidia.com/embedded/jetpack | P | H |
| T4 | **TensorRT 11.4.0** is "Latest" (support matrix last updated 2026-10-05). DLA is not supported for DriveOS deployments in 11.4.0 | https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html | P | M |
| T5 | **DriveOS:**<br>• 7.2.5 was announced on the forum (CUDA 13.3, cuDNN 9.23.0, TensorRT 11);<br>• the TensorRT 10→11 migration guide says DriveOS 7.2.5+ in one revision and 7.2.6+ in another (11.3.0+);<br>• the ledger cites DriveOS 7.2.6 from PR #45's README;<br>• **the newest DriveOS is unverified**, because the DRIVE downloads page is JS-rendered | https://forums.developer.nvidia.com/t/announcement-from-nvidia-nvidia-driveos-7-2-5-now-available/379371 (S, via snippet) ; https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-driveos.html | P/S | M/L |

**Gap.** Issue #130 (Alpamayo 1.5 export) was not re-queried; the ledger had it open.

### Q6 — Cosmos ("Cosmos3-Nano-Sim-Auto"); NuRec/NRE since 10-25

**Takeaway.** Cosmos3-Nano-Sim-Auto is a **downloadable, ungated, OpenMDW-1.1** 16B causal-distilled multi-view (plus LiDAR) world model. NVIDIA explicitly lists closed-loop driving-policy evaluation among its uses. It reports no correlation with real or NuRec closed-loop scores and has no AlpaSim integration. NuRec and NRE have had no release since 10-25.

| ID | Fact | Source | P/S | Conf. |
|---|---|---|---|---|
| W1 | **Card (V1.0, dated 10/20/2026):**<br>• **16B** MoT, causal-distilled from Cosmos3-Nano-Transfer-Auto (itself post-trained from Cosmos3-Nano) into a few-step chunked generator;<br>• conditioned on text and World Scenario Map (WSM) controls rendered from an AV policy's planned trajectory (or a steering wheel);<br>• up to 11 camera views (Hyperion 8.1 set) plus 128-beam LiDAR, 480p/720p.<br>Use case includes "closed-loop driving policy evaluation and verification". Commercial and non-commercial use allowed; **OpenMDW-1.1**; runs on PyTorch or vLLM-Omni; supported hardware is Hopper and Blackwell. **Single-H100 throughput** (4 denoising steps, no CFG, 201 frames, no LiDAR):<br>• 1 view at 480p: 10.2 fps;<br>• 4 views at 480p: 2.5 fps;<br>• 7 views at 480p: 1.4 fps.<br>"Multi-GPU optimization is underway". The card calls the technical report "coming soon" | https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto (README) | P | H |
| W1b | HF metadata: repo **created 2026-08-31**, last modified 10-21, `gated: false`, 664 downloads. The creation date predates the card's 10/20 release date: the repo was probably private until release (inference). Recording both dates | https://huggingface.co/api/models/nvidia/Cosmos3-Nano-Sim-Auto | P | H |
| W2 | **Cosmos Lab page:** Sim-Auto at 16.7 FPS on 16 GB200 GPUs; AlpaSim and Alpamayo are not mentioned; no sim-vs-real or sim-vs-NuRec correlation; a technical-report PDF link is present. The page is inconsistent on views ("7 RGB views + LiDAR" vs eleven cameras) | https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/ | P | M |
| W3 | **NVIDIA forum announcement (2026-10-21):** four Cosmos 3 Nano (16B) example checkpoints "for closed-loop simulation" (robotics and AV) | https://forums.developer.nvidia.com/t/new-nvidias-cosmos-3-tools-simulation-specific-models-multimodal-and-multisensory-data-processing-pipelines-and-an-open-framework-for-physical-ai/386457 | P | M |
| W4 | The evaluation dataset `nvidia/Cosmos3-Driving-Simulation-Evaluation` is OpenMDW-1.1, ungated, created 10-14 | https://huggingface.co/api/datasets/nvidia/Cosmos3-Driving-Simulation-Evaluation | P | H |
| W5 | **NuRec and NRE: no release since 10-25.**<br>• NGC `nvidia/nre/nre-ga` last modified 2026-08-07;<br>• HF NuRec dataset last modified 2026-07-30;<br>• PAI-AV last modified 2026-10-23 (as in the ledger);<br>• the Cosmos3 NIM container was modified 10-20 | NGC search API ; https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec ; …/PhysicalAI-Autonomous-Vehicles | P | H |
| W6 | **Foundation4D** (arXiv 2610.30340, 2026-10-21, author "NVIDIA"): a feed-forward model for geometry, detection, occupancy and Gaussian reconstruction, used as priors for lidarless NuRec. Its dataset, `PhysicalAI-Autonomous-Vehicles-Foundation4D`, was created 10-15, modified 10-21, `gated: auto` | https://arxiv.org/abs/2610.30340 ; HF dataset API | P | H |
| W7 | **OmniDreams v3 (2026-09-23):** built on Cosmos diffusion and trained on 21k h of driving. It runs closed loop with Alpamayo 1 and the AlpaSim orchestrator. A world-action model post-trained from it reportedly beats Alpamayo 1.5 on NuRec with one fifth of the parameters (preliminary). The v3 comment links GitHub and HF weights (`nvidia/omni-dreams-models`, created 04-21) | https://arxiv.org/abs/2606.03159v3 ; HF API | P | M |

**Inference.** At AlpaSim's 10 Hz, a 20 s rollout with 4 views at 480p would need about **80 s of one H100** for Sim-Auto generation alone, about 4× slower than real time. Over 441 scenes that is about **9.8 H100-h per seed** of rendering only, before integration overhead. This assumes the card's "fps" counts synchronized multi-view frames, which the card does not define.

**Gap.** The card does not report Sim-Auto throughput on the RTX PRO 6000 (it lists Hopper and Blackwell datacenter parts).

### Q7 — NVIDIA AV research since 2026-09-01 that overlaps PhysLoop's artifacts

**Takeaway.** Five NVIDIA-authored papers since 09-01 touch PhysLoop:
- **TacitVLA** (10-13) hits S5(c), S3's reasoning arm and S6 latency;
- **OPTED** (09-17) hits S5(a);
- **"Perceives but Doesn't Brake"** (10-20) hits S2/S3 strata and the S4 audit;
- **ECO** (09-25) hits S3/S6 interpretation;
- **AD-Memo** (09-29) bears on S4's RL estimator design and on Q1.

None publishes a paired, stratified student-vs-teacher closed-loop verdict, a seed test–retest of AlpaSim, AlpaGym RL gains, or a 2B student at 10 Hz on Jetson AGX Thor.

**How NVIDIA's role is established.** By author overlap with NVIDIA-authored artifacts, all primary sources:
- AlpaGym's README names Karkus, Tian, Nauman and Packer as core contributors. Tian is a TacitVLA author; Karkus, Naumann and Packer are AD-Memo authors; Karkus is also an OPTED author.
- AlpaGym commit metadata carries pkarkus@nvidia.com.
- AlpaSim sync #114 is co-authored by migl@nvidia.com (Igl, an OPTED author).
- The AR1 arXiv entry is authored "NVIDIA" and lists Yan Wang, Wenjie Luo and Yulong Cao, who are all TacitVLA authors.

Affiliation *at submission time* is still an inference: arXiv abstract pages do not print affiliations, except Foundation4D, which is authored "NVIDIA".

| ID | Item (date) | NVIDIA role | Releases | What it shows (numbers as stated) | PhysLoop overlap |
|---|---|---|---|---|---|
| R1 | **TacitVLA**, arXiv 2610.17540 (v1 2026-10-13) | Authors include Ivanovic, Pavone, Yan Wang, Yulong Cao and Wenjie Luo (inference: NVIDIA AV group) | Project page tacitvla.github.io. Code and weights are not stated. Driving data is a proprietary set of 3.8M events | **Driving model:** Qwen3-VL-2B backbone plus a flow-matching expert (10 Euler steps), about 3.86–3.90B parameters.<br>**Protocol:** AlpaSim 2026 challenge, 441-scene public split, 20 s at 10 Hz, 3 rollouts per scene; "±" is the **SD across rollouts, not a CI**; no paired tests.<br>**Scene score:** no-reasoning 46.4 ± 2.2; explicit reasoning 52.3 ± 1.4; tacit 52.5 ± 0.3.<br>**At-fault collisions:** 6.9 / 5.5 / 6.7 %.<br>**Latency (1 × H100, bf16):** 127 ms (no-reasoning and tacit) vs 269 ms (explicit: 126 action + 143 reasoning).<br>**Not done:** no reasoning-token sweep for driving.<br>**Reference point:** the paper cites Alpamayo-R1 at 55.0 % on the organizers' reference rollouts | S5(c) CoC-budget curve (endpoints exist; no sweep); S3 "reasoning on/off" arm; S6 latency framing; S2 rollout-variance inputs |
| R4 | **OPTED**, arXiv 2609.20756 (2026-09-17) | Igl, Karkus, Chitta, Ivanovic and Pavone, plus Da Col, Schindler and Sakaridis (ETH; inference) | Code "coming soon" at github.com/01Dami23/opted | A privileged teacher, trained with RL on vectorized inputs, supervises on-policy fine-tuning of camera students in AlpaSim on 441 held-out NuRec scenes. **Scene score:** LTFv6 26.5 → 41.8; VaVAM 3.9 → 37.1. **Driving score:** ×1.6 (TransFuser) and ×9.5 (VaVAM). **Efficiency:** 95 % of the teacher in about 6.8k episodes vs 6.2M for RL fine-tuning. The setting of that comparison is unverified; it is likely the vectorized WOMD controlled study | S5(a) on-policy correction (already cited in plan v4); S4 (RL vs on-policy sample efficiency) |
| R3 | **Perceives but Doesn't Brake**, arXiv 2610.28184 (2026-10-20) | Pavone co-author (inference) | None stated | In Alpamayo-1.5, hazards are encoded in the visual representation, but the longitudinal plan is dominated by ego-speed history. Injecting a hazard-conditioned energy into the frozen flow sampler cuts footprint intrusion from 58.1 % to 14.5 % and removes collisions in their braking-relevant closed-loop battery | S2 failure taxonomy; S3 strata; S4 hacking audit; G4 (deferred) |
| R5 | **ECO**, arXiv 2609.31383 (2026-09-25) | Ivanovic and Chitta, plus Toronto co-authors (inference) | None stated | A post-processing layer that keeps the predicted endpoint and reshapes the intermediate waypoints. AlpaSim scene score: VaVAM +123 %, DiffusionDrive +22 %. HUGSIM: VaVAM 18.1 → 31.0 | S3/S6 paired-verdict interpretation (controller/geometry confound); S2 protocol pinning |
| R6 | **AD-Memo**, arXiv 2609.38641 (2026-09-29) | Ivanovic, Pavone, Karkus, Y. Wang and others, plus UIUC (inference) | Project page; release not stated | "Da Capo" semi-closed-loop RL with trajectory-level advantages for memory and step-level advantages for driving. It replays GT states with no simulator. The base is a **2B single-camera Alpamayo 2 checkpoint**. Open-loop only | S4 estimator design (context only); Q1 |
| W6 | Foundation4D (10-21) | Authored "NVIDIA" | PAI-Foundation4D dataset | Reconstruction priors for lidarless NuRec | Low (renderer and data side) |
| W7 | OmniDreams v3 (09-23 update) | NVIDIA | GitHub and HF weights | A Cosmos-based world model running closed loop with Alpamayo in AlpaSim | G3·R2 (deferred); I2 |
| W1 | Cosmos3-Nano-Sim-Auto (10-20; a model, not a paper) | NVIDIA | Weights (OpenMDW-1.1) | Pitched for closed-loop policy evaluation; no correlation reported | G3·R2 (deferred); I2 |

**Non-NVIDIA but relevant to "first" claims (race context):**
- R7. **TRUST**, arXiv 2610.00601 (09-30; Somil Bansal's group). On Alpamayo 1.5 in AlpaSim, CoT monitoring and steering reaches 88.9 % monitoring accuracy, raises reasoning correctness from 75.9 % to 90.0 %, and cuts collision rate by 30.4 % on a challenging subset. It concludes that reasoning-correctness gains do not automatically become embodied gains. Relevant to S5(c) related work.
- R8. arXiv **2609.00718** (09-01). A stage-wise closed-loop evaluation of a PPO driving policy through pruning, distillation and quantization in Gym-Duckietown. Pruning is where capability is first lost. A generic precedent for "closed-loop evaluation of compression", so S3/S6 must not claim novelty for the generic idea; their claim is NVIDIA's student, paired and stratified, in AlpaSim.
- R9. **Qwen-Drive-1.0**, arXiv 2609.00111 (08-31; Alibaba). The visible text shows **open-loop** RL (group-relative advantage, rollouts from the flow sampler, NAVSIM/WOD-E2E rewards) and mentions AlpaSim only as promise. A search-snippet claim of an at-fault AlpaSim score of 0.27 → 0.37 is **unverified** (the text was truncated). It does not contradict S4's "no published closed-loop RL gains from AlpaGym".
- Other closed-loop or world-model papers in the scan (RoXDrive, ZYT-World, DreamStream, NavSafe-∞, CounterPlay, DriveZero) are non-NVIDIA. They do not touch AlpaSim pairing, AlpaGym or Thor.

**R10 method.** arXiv API queries sorted by submission date and filtered to ≥ 2026-09-01: `all:AlpaSim`, `all:Alpamayo`, `all:AlpaGym` (0 hits), `all:NuRec`, plus abstract queries combining closed-loop, driving and distillation; world model; reinforcement learning; and end-to-end driving with on-policy. Endpoint: https://export.arxiv.org/api/query. Plus a web search for TacitVLA.

**Gaps:**
- arXiv metadata search misses papers that name AlpaSim only in the body.
- The NVIDIA AVG publications page shows no 2026 entries in its first 100k characters.
- TacitVLA and OPTED code status: "not stated" and "coming soon" respectively.

### Q8 — Prices on 2026-10-27 (on-demand list)

**Takeaway.** Compared with 10-25:
- H100 SXM median is flat at $2.40, cheapest $1.79;
- RTX PRO 6000 Server is cheaper at the median ($1.34);
- RTX PRO 6000 Workstation offer count fell from 49 to 33;
- B200 and B300 medians rose.

An 8 × H100 node lists at about $31.92–36.00/h. These are list prices; 24–48 h availability was verified only for the vast $32.00 offer. The smallest self-serve multi-node option is RunPod Instant Clusters: 2 nodes, no minimum term. Lambda's smallest cluster needs a 2-week, sales-assisted commitment, about $33.1k for 16 × H100.

**vast.ai, on-demand, 2026-10-27 08:31 UTC.** $ per GPU-hour, `dph_total / num_gpus`; median per offer. Source P1: https://console.vast.ai/api/v0/bundles/ with q = {gpu_name, rentable: true, rented: false, type: on-demand}. Saved JSON is in `l1_raw/vast_2026-10-27/`.

| GPU | Offers (GPUs) | Cheapest | Median | 10-25 cheapest / median |
|---|---|---|---|---|
| RTX PRO 6000 Workstation | 33 (58) | 1.00 | 1.56 | 1.00 / 1.60 (49 offers) |
| RTX PRO 6000 Server | 30 (79) | 1.27 | 1.34 | 1.24 / 1.47 (47) |
| RTX PRO 6000 Max-Q | 41 (82) | 0.80 | 1.44 | 0.64 / 1.48 (49) |
| H100 SXM | 17 (52) | 1.79 | 2.40 | 1.93 / 2.40 (17) |
| B200 | 12 (43) | 6.49 | 10.25 | 7.50 / 9.38 (9) |
| B300 | 8 (14) | 8.98 | 11.25 | 9.01 / 9.01 (5) |
| H200 | 2 (6) | 6.43 | 6.43 | 3.96 / 4.86 (6) |

**Full 8 × H100 node per hour:**
- **vast.ai.** Two 8-GPU offers.
  - $19.20 (Japan, unverified). It reports `bw_nvlink` 0.0 and a maximum rental duration of about 0.2 days, so it is **unusable for 24–48 h and suspect**.
  - **$32.00** (Germany, unverified, NVLink 478 GB/s, about 10.2 days available).
- **Lambda.** $3.99/GPU-h → **$31.92**. 1 × H100 SXM is $4.29; 8 × B200 is $6.69/GPU-h (P2: https://lambda.ai/pricing).
- **RunPod.** H100 SXM $3.99 Secure / $2.69 Community per GPU-h, so **$31.92 / $21.52** for 8 if available; an 8-GPU pod price is not listed. B200 $7.99 / $5.98; RTX PRO 6000 $2.49 / $1.69 (P3: https://www.runpod.io/pricing).
- **Nebius.** H100 $4.50/GPU-h → **$36.00**. H200 $5.40, B200 $8.50, B300 $9.50, RTX PRO 6000 $1.80. Managed Kubernetes and Managed Soperator (Slurm on Kubernetes) are free, with consumption-based pricing. Large clusters are by contact (P5: https://nebius.com/prices).

**Smallest rentable 2-node (16-GPU) cluster:**
- **RunPod Instant Clusters** (P4). Anyone can get 2 nodes / 16 GPUs on demand, up to 8 nodes / 64 GPUs. Billing is per second (the structured data says per millisecond), with no minimum lease. InfiniBand or RoCE v2 at 1,600–3,200 Gbps. Slurm is supported; Kubernetes is not. The product page lists no prices. The pricing page shows H100 and B200 clusters as "Contact sales" and **H200 SXM at $6.35, unit unstated**. If that is per GPU-hour, 16 × H200 ≈ **$101.6/h**. Sources: https://www.runpod.io/product/instant-clusters ; https://docs.runpod.io/instant-clusters
- **Lambda 1-Click Clusters.** 16 to 2,000+ GPUs; H100 $6.16/GPU-h, B200 $9.86/GPU-h; **2-week minimum**; "Talk to our team" (not self-serve). 16 × H100 × 336 h ≈ **$33,116** (computed).
- **Nebius.** Cluster pricing by contact. **vast.ai.** No cluster product appeared in the offer data (`cluster_id` null on every 8-GPU offer).

**Gaps:**
- The unit of RunPod's $6.35, and RunPod's H100 cluster price, need a console quote (Huy's action; it needs an account).
- RunPod 8-GPU community pod availability was not verified.
- Lambda and Nebius pages carry no effective date.

---

## 3. Deltas against the 2026-10-25 ledger and plan v4

| # | Earlier text (where) | Delta, 2026-10-27 | Source |
|---|---|---|---|
| D1 | "No external merge since #149 (08-17)" (ledger B3; `04` §5; `CLAUDE.md`, `AGENTS.md` "none has merged since 2026-08-17"; plan §8 risk row) | **Contradicted for `e2e_challenge`:** #167 (author_association NONE) merged 08-31; #111, #161 and #169 (non-collaborator handles) merged 08-24/08-31. **Still true for `main`.** Plan §2 G1 already lists #167 as upstream, so the docs contradict themselves | A8b |
| D2 | "NVIDIA withholds some 'simulation choices' until the challenge ends on 10-31" (plan S2 Why; ledger B3) | #224 is **closed (10-23)**. The disclosure is tied to "our next release", not to a date. S2 Timing should check for a new release or sync rather than assume disclosure after 10-31 | A4 |
| D3 | "`e2e_challenge` corridor scorer fixed mid-challenge (#215, 10-20)" | Adds the mechanism (stationary GT tail jitter) and NVIDIA's fix (orientation from the last 1 m of real movement), and that **metrics were recalculated on 10-22** with a frozen v1 view kept. #217 (report uploading) also merged 10-20 | A1, A2 |
| D4 | (absent) | #222, NVIDIA's statements: the PAI internal test set is a different, harder distribution; the harmonizer is on for final runs; renderer-OOM scenes are re-queued; the official topology is `8gpu_32rollouts`; system-clock policies carry no guarantee | A3 |
| D5 | Manifest needs the lockfile (#207) and `n_sim_steps` (#208) | Unchanged in substance. #207 is still unanswered. #208 is **acknowledged by NVIDIA** (10-15), and the reporter offered a PR that has not been filed | A6, A7 |
| D6 | AlpaSim "ships `results-summary.json`, Prometheus telemetry…" (`04` §5) | **Ledger miss.** AlpaSim also ships Slurm deployment (slurm.py, slurm_enroot.py, submit and resume scripts, a Slurm process exporter) and the `optimize-slurm-topology` skill, whose throughput metric is the stabilized full-run `seconds_per_rollout` (synced 2026-07-01) | A5 |
| D7 | G3 row: "OmniDreams (n = 4)" | **Ledger miss.** AlpaSim `main` integrates OmniDreams as a renderer via FlashDreams (`deploy=managed_flashdreams`; 96 GB VRAM with Alpamayo 1.5). OmniDreams v3 (09-23) links weights. There is no Cosmos3 integration | A9, W7 |
| D8 | (absent) | **Ledger miss.** #72 (closed 10-18): NRE is a proprietary, obfuscated container; NVIDIA runs it under Slurm with enroot; Kubernetes without Docker-in-Docker is unsupported | A10 |
| D9 | AlpaGym "needs 2 GPUs ≥ 40 GB", code-frozen (ledger B1; `04` §5) | Confirmed: no commits since 07-21. **Added:** Slurm presets from a 4-GPU partial node to 5 × 8-GPU nodes, with **3 nodes the smallest multi-node preset**; Slurm autoresume (07-06); the README warns the presets will not run out of the box; the blog claims single-GPU-to-multi-node scaling with no numbers | B1–B4 |
| D10 | Recipes: open PRs #42, #43, #45 (ledger B2) | **New: #44** (10-21), NVIDIA's training-efficiency "Measurement" chapter. #42, #43 and #45 are unchanged; no merge since 10-16 | C1–C4 |
| D11 | Plan §0 "What survived": no published closed-loop evaluation of any NVIDIA 2B student. §1 research question: "curve rather than a binary" | Still literally true, because TacitVLA's model is not an NVIDIA student of a released teacher. But **NVIDIA authors have published AlpaSim closed-loop results (441 split) for a 2B-backbone reasoning VLA**, with explicit, tacit and no-reasoning arms and H100 latency. S3's novelty wording and the S5(c)/§1 research-question wording must cite and position against it | R1 |
| D12 | S5 Why cites "RoaD and OPTED report large sample-efficiency gains" | OPTED's details: code not yet released; about 6.8k vs 6.2M episodes; tested on TransFuser, VaVAM and LTFv6, not on a VLA or Alpamayo | R4 |
| D13 | (absent) | New NVIDIA-authored papers: ECO (09-25), AD-Memo (09-29), "Perceives but Doesn't Brake" (10-20), Foundation4D (10-21). Third-party Alpamayo/AlpaSim papers: TRUST (09-30) and a compression closed-loop study (09-01) | R3, R5, R6, R7, R8, W6 |
| D14 | Berlin: "Cosmos 3 Nano variants" (ledger lane A §2) | Specifically: **Cosmos3-Nano-Sim-Auto** (16B, OpenMDW-1.1, ungated, pitched for closed-loop policy evaluation; H100 2.5 fps at 4 views/480p; no AlpaSim integration; no correlation study). Repo created 08-31; card release 10-20 | W1–W4 |
| D15 | (absent) | Licence of the PR #42 student VLM (`Cosmos-Reason2-2B`): **NVIDIA Open Model License**, not OpenMDW | M9 |
| D16 | Prices (ledger C5; plan §5) | H100 SXM cheapest $1.93 → $1.79 (median flat at $2.40). RTX PRO 6000 Server median $1.47 → $1.34. RTX PRO 6000 Workstation median $1.60 → $1.56, with offers down 49 → 33. B200 $7.50/$9.38 → $6.49/$10.25. B300 $9.01/$9.01 → $8.98/$11.25. An 8 × H100 node now lists at $31.92–36.00/h; 24–48 h availability was verified only for the vast $32.00 offer. The plan's "$15–32/h" low end is not available today for a ≥ 24 h rental | P1–P5 |
| D17 | Unchanged (re-confirmed 10-27) | Edge not public; Super card unchanged since 08-07; TensorRT-Edge-LLM v0.11.0 and PR #260 open; JetPack 7.2.1 and TensorRT 10.16.2; TensorRT 11.4.0; NuRec 07-30; NRE GA 08-07; PAI-AV 10-23; AlpaGym unchanged; alpamayo2 repo last pushed 10-07 | M1–M12, T1–T4, W5, B1 |

---

## 4. First-principles rationale per finding

- **L1-1 (KC-E capture list).** A verdict on a student is only interpretable if the artifact is pinned: weights revision, licence and architecture. Edge's licence decides whether S3 (evaluation) and S5 (fine-tuning, derivatives) are lawful. Its architecture decides which S3 and S5 arms exist (a reasoning switch, flow steps). Recording these on release day costs minutes and prevents a later re-run. Pre-writing the fallback experiment file keeps "prediction before run" intact without spending on a student that may become redundant days later.
- **L1-2 (TacitVLA).** Novelty claims must survive the strongest prior result. TacitVLA measures the two endpoints of the CoC-budget question, decoded vs not decoded, on the same split and simulator PhysLoop will use. What remains open is the **dose–response** between the endpoints, under a real on-device latency budget, on the deployable student, with paired and stratified inference.
  - Architecture matters. In TacitVLA the action expert never attends to reasoning, so n = 0 is in-distribution. In an Alpamayo-style expert conditioned on CoC tokens, n = 0 feeds the expert an input it never saw in training.
  - So the n = 0 point measures a different thing per architecture, and the curve's intercept is not comparable across models unless labelled.
  - TacitVLA's SDs over three rollouts of a 441-scene aggregate (0.3–2.2 pp) bound the run-to-run noise of the *aggregate*. They are not the paired per-scene σ_d that S1's power tables need. Huy derives that; these are inputs only.
- **L1-3 (#198 test).** A scorer is part of the instrument. A known-answer case with an analytically obvious truth (≈ 0 m lateral for pure forward overshoot) is the cheapest guard against silently running a pre-fix scorer. Because the leaderboard was recomputed, any externally quoted number needs its scorer version to be comparable.
- **L1-4 (manifest fields).** Closed-loop scores are functions of the policy *and* the environment protocol. NVIDIA names three protocol variables that can move scores: harmonizer, concurrency or topology, and clock source. Unrecorded protocol variables are confounds. Recording them is free; varying them is a separate one-variable experiment.
- **L1-5 (stationary lead).** Strata should follow mechanisms. The perception–action binding gap is a known mechanism: the longitudinal plan follows speed history. Its signature scenario is a stationary in-path lead. If the student inherits the teacher's weakness, the paired difference in that stratum may be near zero while absolute failure stays high. If RL "fixes" it by stalling, the progress test catches it only if stall is scored. Values are Huy's.
- **L1-6 (throughput metric).** Throughput measured before steady state is biased by start-up congestion, and NVIDIA's own rule is to wait 30–45 min. Adopting NVIDIA's metric makes PhysLoop's numbers commensurable with the people who will read them.
  - The topology-invariance control tests whether the instrument is time-invariant: a verdict that changes with concurrency means timing leaks into the policy.
  - #222 says this is possible for wall-clock policies. It is a cheap reliability measurement (D4).
- **L1-7 (single-node Slurm).** The role's cluster duties (Q5, D4) cannot be done at full scale here, but the *mechanism* can be exercised on one node: Slurm submission, requeue and autoresume are node-count-independent control paths. Measuring checkpoint continuity across a forced requeue is a real reliability measurement, not a signal. The kill rule bounds the risk the README itself flags.
- **L1-8 (multi-node, c).** Scaling efficiency (rollouts/h per GPU, 1 → N nodes) is the quantity a cluster RL framework is judged on. AlpaGym's shipped presets encode NVIDIA's own view of the bottleneck (rendering). A short 1-node vs 3-node measurement tests that claim directly. It costs hundreds of dollars and new hours, so it is gated; Lambda's 2-week minimum makes it infeasible there.
- **L1-9 (no competing PR).** Racing a community contributor on an acknowledged issue adds no evidence. Recording the fields in PhysLoop's manifest delivers the measurement value regardless.
- **L1-10 (training-efficiency definitions).** GPU-hours from mean (not median) wall clock is the unbiased estimator of total cost when step times are right-skewed (stalls). MFU and the CUDA-event decomposition localize the bottleneck. Using NVIDIA's definitions avoids "our MFU vs their MFU" ambiguity.
- **L1-11 (OPTED).** On-policy distillation from a privileged teacher reduces the variance of the learning signal compared with RL, which explains the three-orders-of-magnitude episode gap. That is NVIDIA's line of work. S5's DAgger with the released VLA teacher stays distinct because the teacher is the deployable model's own teacher, not a privileged planner. The overlap is in method family, not in artifact.
- **L1-12 (controller and geometry confound).** A closed-loop score depends on plan → controller → vehicle. ECO shows that changing only plan geometry moves scores by +22–123 %. A student–teacher difference can therefore come from trajectory smoothness rather than decisions. Pinning the controller and post-processing, and logging motion-limit violations, keeps the paired verdict interpretable.
- **L1-13 (toolchain unchanged).** S6's porting estimate rests on TensorRT-Edge-LLM lacking Alpamayo 1.5 support and on the forked path in PR #45. Neither changed, so neither does S6's cost.
- **L1-14 (licence).** Derivative-weights publication depends on the base model's licence as well as the data licence. The fallback student's base is not OpenMDW.
- **L1-15 (prices).** Every stage is re-priced at its start (§5 rule). A 24–48 h job needs an offer whose host-declared availability exceeds the job. Ignoring the duration field understates cost.
- **L1-16 (doc correction).** Contradictory facts inside the contract erode trust in every other number. Fixing them is cheap.
- **L1-17 (Sim-Auto).** A world model pitched for policy evaluation is only an instrument once its agreement with a reference (NuRec or real) is measured. NVIDIA reports none and has not integrated it with AlpaSim. Building that bridge would race NVIDIA's technical report and re-open deferred scope.

## 5. Rejected options (with reasons)

1. **A Cosmos3-Nano-Sim-Auto renderer arm in S2 or S3.**
   - It is not integrated in AlpaSim, so it needs a WSM-control bridge from AlpaSim's ego plan, which means building an integration.
   - G3·R2 (renderer agreement) is deferred in plan v4.
   - NVIDIA's technical report and "multi-GPU optimization" are in flight, so this is a race risk.
   - Inferred cost is about 9.8 H100-h per 441-scene seed for rendering alone at 4 views/480p.
   - An optional ≤ 2 GPU-h single-scene throughput probe is possible after S2 starts under P-005, but it adds little beyond the card's own H100 table. Not recommended.
2. **An OmniDreams/FlashDreams renderer-swap sub-sweep in S2.** It re-opens deferred G3·R2. NVIDIA's OmniDreams paper already runs closed loop with Alpamayo in AlpaSim. It needs 96 GB VRAM with Alpamayo 1.5, which is borderline on a 96 GB RTX PRO 6000. If Huy wants I2 coverage later, this is the only zero-new-integration path, and it should go to `DECISIONS.md` as a proposal.
3. **Training a TacitVLA-style "tacit" student for S5(c).** It is new training and a new variable, it is NVIDIA's own research line (likely follow-ups), and it is not needed to measure the curve on the released student. Label the n = 0 point instead (L1-2).
4. **Applying OPTED to the Edge or PR #42 student.** This is a plausible NVIDIA follow-up, and OPTED's code is not released ("coming soon").
5. **Filing a PR for #208** (or #207). The reporter offered #208, NVIDIA has not merged to `main` since 08-18, and it adds no measurement.
6. **A Kubernetes deployment of AlpaSim or NRE as the Q5 stand-in.** #72 says NRE is proprietary and obfuscated and that Kubernetes without Docker-in-Docker is unsupported. NVIDIA's own path is Slurm with enroot. Hours are high and evidence per hour is low next to L1-7. Nebius's free managed Kubernetes and Soperator do not remove the NRE constraint.
7. **Lambda 1-Click Clusters for any multi-node work.** 16 GPUs × a 2-week minimum ≈ $33.1k at $6.16/GPU-h, and not self-serve.
8. **The vast.ai $19.20/h 8 × H100 offer for S5.** It declares about 0.2 days of availability and reports `bw_nvlink` 0.0. It cannot host a 24–48 h run, and its interconnect is suspect.
9. **Starting a PR #42 pilot training before KC-E (10-31).** Edge may still land "this month"; only pre-write the experiment file.
10. **A PhysLoop inference-time hazard-guidance monitor** (from R3). G4 is deferred, the work is NVIDIA-coauthored, and the "Halos" naming rule applies.
11. **A generic training-efficiency or benchmark harness.** PR #44 lists NVIDIA's own regression bands, benchmark harness and profiler chapters as coming. Apply the definitions instead (L1-10).
12. **Re-deriving claims from Qwen-Drive's AlpaSim RL numbers.** They are unverified (snippet only), and the visible text shows open-loop RL.

## 6. Open questions and what would resolve them

| Question | Why it matters | Resolver |
|---|---|---|
| Will Edge be public by 10-31? Repo ID, revision, licence terms (evaluation, fine-tuning, derivatives), backbone (Cosmos-Reason2-2B?), action expert and flow steps, runtime-CoC switch, camera count, teacher, any NVIDIA closed-loop number | KC-E; S3 arms; S5 legality; S6 export path | KC-E check on 10-31: HF API (author search plus the M2 IDs), NGC catalog API, NVlabs repo search, blog and Alpamayo page; again at S3 start |
| What are the withheld AlpaSim "simulation choices"? | S2 protocol validity; comparability with any pre-release number | Next AlpaSim release or sync. Watch `pushed_at` on NVlabs/alpasim and the `main` commits |
| Is the NuRec Harmonizer available for local runs, and which version runs in the final evaluation? | Manifest field (L1-4); possible protocol difference from the leaderboard | NGC/NIM docs for the NuRec Harmonizer at S2 start; challenge README after 10-31 |
| RunPod cluster pricing: the unit of "H200 $6.35" and the H100 cluster price | L1-8 cost | A RunPod console quote. Huy's action (needs an account); not for agents |
| Can AlpaGym's Slurm presets run on a single rented VM node (slurmctld and slurmd plus enroot/pyxis)? | L1-7 feasibility | A bounded bring-up under L1-7's 8-hour kill rule |
| Did Qwen-Drive-1.0 run closed-loop RL in AlpaSim (0.27 → 0.37)? | S4 novelty wording | Read §3.3 and the appendices of arXiv 2609.00111 |
| Will TacitVLA or OPTED release code or weights? | Whether S5 can reuse them as baselines vs only cite them | tacitvla.github.io; github.com/01Dami23/opted |
| Was OPTED's "6.8k vs 6.2M episodes" comparison run in AlpaSim with camera students, or in the vectorized WOMD study? | Whether it is a valid input for S5 pilot sizing (L1-11) | OPTED arXiv 2609.20756, experiments section |
| What is the newest DriveOS (7.2.5 vs 7.2.6+), and its TensorRT 11 minor? | Claims hygiene for any DRIVE comparison (S6 does Jetson only) | DRIVE OS documentation portal (the JS page did not render for automated fetch) |
| Sim-Auto: meaning of "fps" for multi-view, per-step latency at 10 Hz with WSM chunking, any closed-loop agreement study | I2 discussion; rejected option 1 | The Cosmos simulation technical-report PDF linked from the Cosmos Lab page |
| Author association of `e2e_challenge` PRs #111, #161, #169 | Exact wording of the external-merge correction (D1) | The GitHub issues API with a fresh quota |
| Was the vast RTX PRO 6000 Workstation drop from 49 to 33 offers in two days transient? | S2 cost and supply (P-005) | Re-sample at S2 start with the same `stats.py` |
| Does AD-Memo's "2B single-camera Alpamayo 2" relate to Edge? | S3 camera contract; S6 "4 cameras" target | The Edge card at release |
