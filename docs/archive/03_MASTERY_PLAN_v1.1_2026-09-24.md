> **ARCHIVED — historical, superseded on 2026-10-03 (DECISIONS D-010, D-011). Never act on this file; the live version is listed in `DOC_AUTHORITY.md`.**

# 03 — AI 2.0 Driving Engineer: Mastery Plan v1.1 (24 September 2026)

**Target:** the capabilities behind NVIDIA's "Principal Deep Learning Engineer, End-to-End
Autonomous Driving" posting (AI 2.0: LLM/VLM/VLA agents that drive, deployed on NVIDIA
chips in production vehicles). Credential lines (12 years, PhD) are ignored by design;
every requirement becomes a skill with a measurable acceptance test and a public artifact.
**Spine:** PhysLoop (release gate, distillation, deployment — `02_PHYSLOOP_PLAN_v2.md`)
and nanoAlpamayo (a driving VLA built and post-trained by us).
**Method:** first principles, prediction before measurement, one variable per experiment,
negative results kept, a mastery gate at the end of every stage.

**v1.1 changes (Sep 24):** synthetic-truth gate validation added before any GPU run
(Stage 1a); the three written predictions from the system-design lesson attached to Stage 1;
explicit mapping matrix (§3); dependency and priority rule between the two spines and the
combined budget (§5); G4 and G7 placed; per-stage mastery gates (§6); data contracts and
variance decomposition added to pillar P5.

---

## 0. Read the posting as a system

```
 [3] Data engine ──► [1][2] Train VLA (IL → SFT → RL) ──► [4] Evaluate & release ──► [5] Firmware / runtime
        ▲                                                        │
        └──────────────── failures found in [4] and on the road ─┘
```

"AI 1.0 → AI 2.0" is NVIDIA's framing: from hand-built driver modules to a reasoning agent
(Alpamayo: a VLM writes a Chain-of-Causation trace, a flow-matching expert emits the
trajectory), distilled to run on DRIVE AGX Thor next to a classical safety stack.

**"Principal", stripped of tenure:** own one link of that loop end-to-end, decide with
measurements, write the design doc others build on, ship code that survives review by the
people who wrote the stack. Evidence is public and checkable: merged PRs, reproduced
numbers, preprints, challenge entries.

---

## 1. Seven skill pillars

| Pillar | Posting bullets | You must be able to… | The trap |
|---|---|---|---|
| P1 Driving as a decision problem | planning and reasoning; behavior/motion planning | Derive the unicycle model and why Alpamayo predicts (accel, curvature) × 64 waypoints at 10 Hz; write an MPC/sampling baseline; explain open- vs closed-loop metrics | Optimizing open-loop L2 (ρ = −0.36 vs closed-loop score, Bosch 2026) |
| P2 VLA modeling from scratch | build, pre-train, fine-tune LLM/VLM/VLA | Multi-camera VLM (180 tokens/image at 320×576 → 2,880 tokens for 4 cams × 4 frames), discrete trajectory tokenizer, flow-matching expert reading the VLM KV cache; train end-to-end | Treating the flow expert as a black box; not knowing which path is deployed |
| P3 Learning algorithms | generative, imitation, RL | DAgger, flow matching (conditional OT path, Euler sampler, SDE variant with log-probs), GRPO with reward design and KL control, distillation, privileged-teacher RL (OPTED) | Reward improves a proxy while close-encounters do not fall; rollout/trainer log-prob mismatch |
| P4 Data engine | novel data generation and collection | Curate PhysicalAI-AV (1,700 h), auto-label CoC with Alpamayo 2 Super, reconstruct scenes (NuRec / Instant NuRec), targeted synthetic variants (Cosmos Transfer), mine failure strata without leakage | Location leakage; synthetic data that fixes one stratum and regresses another |
| P5 Simulation and evaluation | performance, safety, reliability standards | Closed-loop AlpaSim; the gate as a decision function with two error types; **variance decomposition** (policy / simulator / sampling); **data contracts** (identity tuple, simulation-step admission predicate, append-only manifest); validity envelope; paired statistics; measuring whether the simulator's ranking is trustworthy | Novel-view collapse scored as real failures; seeds counted as independent scenes; wall-clock used as the authority in a paused simulator |
| P6 Deployment and real-time | real-time in resource-constrained environments; C++ safety-critical | Roofline on Thor-class silicon (273 GB/s); FP8/NVFP4 incl. expert; TensorRT engines; CUDA graphs; C++ runtime with fixed memory and deterministic timing; action-chunk pacing | Batch-1 eager latency called deployable; VLM quantized but not the memory-bound expert |
| P7 Safety and production engineering | deploy in production, safety-critical | Dual stack and guardrails; release notes with allowed claims; what SOTIF / ISO PAS 8800 / UN R171 ask of ML validation; postmortems | Claiming "safe" from simulation; single aggregate metrics |

Ladder for every pillar: L5 user-visible behavior → L4 system → L3 algorithm/math →
L2 kernel, memory, precision → L1 silicon. Mastered only when you can go down to L2 and
back up with numbers.

---

## 2. The two spine projects and the rule between them

### nanoAlpamayo — "I built and post-trained a driving VLA"
Build the architecture, tokenizer, training loop, SFT and GRPO yourself on a small
pretrained VLM backbone (0.5B–2B), on a PhysicalAI-AV subset with public CoC labels.
("From scratch" = the system and the training loop; the backbone is initialized from a
pretrained VLM, as NVIDIA does.) Acceptance = reproducing NVIDIA's published small-model
numbers on the public validation set (Alpamayo-R1, 0.5B rows):

| Metric (0.5B, public val) | Published | Acceptance |
|---|---|---|
| minADE6 on 644 PAI samples | 0.913 m | ≤ 1.0 m |
| RL post-training, ADE on CoC set | 2.12 → 1.92 m (−9.4%) | ≥ 5% relative gain, 3 seeds |
| Reasoning quality / consistency | 3.1 → 4.5 / 0.62 → 0.85 | Same direction, with CI |
| Closed-loop (920 NuRec scenarios) | close-encounter 9.0%, off-road 19%, AlpaSim 0.35 | Within 20% relative, **on the PhysLoop gate** |

Then the unpublished part: RL through the flow expert on our model (Flow-SDE, as AlpaGym
does) vs token-only RL, with the expert-drift monitor.

### PhysLoop v2 — "I built the release gate and the Thor-student path"
G1 gate → G2 validity → G3 gate validity study → G4 guardrail (C++) → G5 distillation →
G6 precision → G7 throughput PRs → G8 flywheel spec.

### Dependency and priority rule (new in v1.1)
- **Evaluation before model claims.** Every nanoAlpamayo number that is not open-loop is
  produced by the PhysLoop gate. nanoAlpamayo closed-loop work cannot start before G1 v1
  exists (end of week 2), and its RL claims (Stage 3) cannot be made before G2 exists.
- **When hours collide, PhysLoop G1–G3 wins** through week 6; nanoAlpamayo data pipeline
  and architecture (Stage 2) fill the gaps while gate runs are executing on rented GPUs.
- **Compute in the same 12 weeks:** PhysLoop ≈ $7.5–8.5k (02 §4) + nanoAlpamayo Stages 2–3
  ≈ $4–6k → **≈ $12–14k**, front-loaded on the gate. If the monthly budget is lower, the
  nanoAlpamayo SFT run is what slips, never the gate.
- **Hours assumption:** 25–30 h/week solo. *Unconfirmed — record the real number in
  `DECISIONS.md` (proposal P-003).*

---

## 3. Mapping matrix (new in v1.1)

| Stage | Weeks | PhysLoop G | Pillars | Core concepts (`01_SYSTEM_DESIGN.md`) | Posting bullet | Artifact | Mastery gate |
|---|---|---|---|---|---|---|---|
| 0 Foundations | 0–2 | — | P1, P2, P3 | C1, C2, C3 | 1 | notebook + note | derive unicycle + flow-matching OT path from memory |
| 1a Gate on synthetic truth (no GPU) | 1–2 | G1 core | P5 | C6, C7 | 4 | `gate/` decision function + synthetic-truth test | false-GO ≤ 5% at a −30% effect on a 2% stratum; INSUFFICIENT appears below noise floor |
| 1b Closed-loop evaluation | 2–6 | G1, G2, G3 | P5, P7 | C4, C5, C6, C7, C8 | 4 | preprint draft + GTC poster (Nov 10) | three written predictions reconciled; ρ with CI; inversions listed |
| 2 nanoAlpamayo v0: imitation | 3–8 | (uses G1) | P2, P3, P4 | C1, C3 | 1, 2 | reproducible training repo | minADE6 ≤ 1.0 m; teach-back on covariate shift |
| 3 nanoAlpamayo v1: reasoning + RL | 7–12 | (uses G1, G2) | P3 | C1, C10 | 1 | ablation table + write-up | ≥ 5% ADE gain; transfer question on reward hacking |
| 4 Distill and deploy | 9–16 | G4, G5, G6, G7 | P6, P7 | C2, C9, C10 | 4, 5 | C++ runtime + latency report; AlpaGym profile + PR | student within 10% of teacher; ≤ 16 GiB / ≤ 0.1 s; PR opened |
| 5 Data engine and flywheel | 14–22 | G8 | P4 | C4, C7 | 3 | flywheel note | targeted stratum improves with no paired regression |
| 6 Safety and production discipline | 18–26 | — | P7 | C9 | 4, 5 | release note + postmortem | release note survives a safety-minded reviewer |
| 7 Scale and voice | 26–52 | all | all | all | every "standout" line | preprints, GTC talk, merged PRs | paper-grade results; merged PR |

Core concepts: C1 two-brain VLA · C2 teacher/student · C3 open- vs closed-loop ·
C4 neural reconstruction · C5 validity envelope · C6 paired statistics · C7 stratification
and taxonomy · C8 measuring the instrument · C9 guardrail in the loop · C10 precision and
throughput.

---

## 4. Stage plan

| Stage | Weeks | Learn | Build | Acceptance (predict first) |
|---|---|---|---|---|
| 0 Foundations | 0–2 | Unicycle kinematics + Euler discretization; transformer/KV cache internals (rebuild a nanoGPT-class model from memory); flow matching from Lipman et al. guide (arXiv 2412.06264) | Run Alpamayo 1.5 inference; toy flow model on 2-D trajectories; re-derive the 99 ms Table 14 budget and find the model size that makes it consistent | Toy flow model within 2% MSE of the analytic OT path; latency derivation within 20% of measurement |
| **1a Gate on synthetic truth** | 1–2 | The gate as a decision function; two error types; variance decomposition (policy / simulator / sampling); identity tuple and simulation-step admission predicate; exact McNemar; log-clustered bootstrap; rule of three | Manifest schema + `is_pair()`; `verdict(pairs_by_stratum, thresholds)` → GO / NO-GO / INSUFFICIENT; a synthetic generator with known effect sizes; 1,000-run Monte Carlo of the gate | false-GO ≤ 5% for a −30% effect on a 2% stratum at 7% base failure; INSUFFICIENT when the effect is below the noise floor; removing stratification measurably raises false-GO |
| 1b Closed-loop evaluation (G1–G3) | 2–6 | Open- vs closed-loop; simulator credibility (NATM); Waymo approval logic; validity envelope mechanics | Gate on 441 scenes × 3 seeds; validity records; policy zoo ≥ 10 | **Written before any run:** (i) test–retest discordance rate of Alpamayo 1.5 across seeds; (ii) % rollouts beyond a 2 m deviation cap and whether it differs between good and bad policies; (iii) scenarios/s per RTX PRO 6000 at 1080p vs 576×320. Then: ρ with CI; inversions listed |
| 2 nanoAlpamayo v0: imitation | 3–8 | BC, covariate shift, DAgger; multi-camera video tokenization; 128 trajectory tokens (64 × {accel, curvature}) | Data pipeline on 100–300 h PAI-AV; VLM + tokens + flow expert; SFT on 8×H100 | minADE6 ≤ 1.0 m on the 644-sample val; overfit-one-batch and loss-at-init checks first |
| 3 nanoAlpamayo v1: reasoning + RL | 7–12 | CoC labels and consistency rewards; GRPO (DeepSeekMath 2402.03300); Flow-SDE log-probs; reward-hacking checks | SFT with CoC; GRPO on tokens; GRPO through the expert; log KL(train‖infer), ratio histograms, lengths | ≥ 5% ADE gain, consistency up, close-encounter not worse **on the gate (paired)** |
| 4 Distill and deploy (G4–G7 + C++) | 9–16 | Distillation; OPTED; roofline of memory-bound expert (OI 54 vs ridge ≈ 281); FP8/NVFP4 (ModelOpt); TensorRT; FlashDrive as competitor on the expert; AlpaGym `perf/` | ≤ 3B student; FP8/NVFP4 incl. expert; C++ runtime: TensorRT engine + gRPC driver, fixed memory, CUDA graphs, timing histograms; G4 guardrail in C++; G7 profile + upstream PRs | Student within 10% of teacher on the gate; ≤ 16 GiB and ≤ 0.1 s per call on RTX PRO 6000; deterministic p99; ≥ 1 PR opened |
| 5 Data engine and flywheel (G8, small) | 14–22 | Auto-labeling with Alpamayo 2 Super; Instant NuRec; Cosmos-Drive-Dreams; leakage-aware splits | Mine worst gate stratum → reconstruct/augment → retrain student → re-gate | Targeted stratum improves with no paired regression elsewhere; leakage audit passes |
| 6 Safety and production discipline | 18–26 | Dual stack, Halos; SOTIF / PAS 8800 / R171 basics; allowed-claims writing; postmortems | Release note for one student checkpoint; OTA-style regression run vs previous checkpoint; one postmortem | Release note survives a safety-minded reviewer; regression run reproducible |
| 7 Scale and voice | 26–52 | Multi-node training (FSDP/EP), WSD ablations; OmniDreams as second renderer; Jetson AGX Thor as DRIVE Thor proxy | nanoAlpamayo 2–3B multi-node; OmniDreams tier in the gate; Thor latency; second preprint; upstream design docs and PR reviews | Paper-grade results; PR merged in AlpaGym / alpamayo-recipes / AlpaSim |

Stage 1a is deliberately GPU-free: the decision function is validated on data whose truth
is known before a single rented GPU-hour is spent. Stages overlap on purpose thereafter,
under the priority rule in §2.

---

## 5. Operating system
- Weekly: one active ablation; prediction before every run; three-sentence log each
  evening; Sunday review of predictions vs results.
- Evidence: every number carries git hash, config, seeds, manifest; nothing in a write-up
  that a fresh clone cannot reproduce.
- Design docs: one page per stage — hypothesis, falsifiable predictions, scope, kill
  criteria, metric-to-decision mapping (`../experiments/TEMPLATE.md`).
- Compute (rented): Stage 1 ~$1.5k; Stages 2–3 ~$4–6k; Stage 4 ~$1k; Stage 5 ~$2k;
  Stage 7 ~$8–15k. Year ≈ $20–30k; the first 12 weeks ≈ $12–14k combined with PhysLoop.
  Jetson AGX Thor ($5,499) only in Stage 7.
- Kill criteria: Stage 1a gate fails its synthetic test → fix the statistics before any
  AlpaSim run; Stage 1b gate invalid (ρ < 0.6) → publish the negative result, keep building
  models, stop trusting the gate; Stage 2 misses minADE6 ≤ 1.0 m after three ablation
  rounds → shrink the subset and fix the pipeline before any RL; Stage 4 student misses
  16 GiB / 0.1 s → smaller student, not a faster kernel.

---

## 6. Mastery gates (new in v1.1)

A stage is closed only when all three pass, in this order:
1. **Teach-back:** explain the stage's mechanism from memory with one hand-worked numeric
   example (e.g., the McNemar 12-vs-6 case; the B* = b·F_peak/(2·BW) crossover).
2. **Transfer question:** apply the mechanism to an unfamiliar shape or failure mode
   (e.g., the kernel-CI analog of the gate; a stratum with 5 scenes; an FP8 student whose
   score rises while 18% of its rollouts leave the envelope).
3. **Evidence gate:** the stage's artifact exists, reproduces from a fresh clone, and its
   prediction-vs-measurement reconciliation is written, misses explained causally.

No stage is closed by a feeling of understanding.

---

## 7. Reference shelf
- Alpamayo-R1 (arXiv 2511.00088); Alpamayo 2 Super blog; NVlabs alpamayo-recipes, alpasim,
  alpagym; closed-loop post-training blog.
- Flow matching: Lipman et al., Flow Matching Guide and Code (arXiv 2412.06264);
  facebookresearch/flow_matching.
- RL: GRPO (DeepSeekMath, arXiv 2402.03300); OPTED (arXiv 2609.20756); RoaD (arXiv 2512.01993).
- Evaluation: nuPlan misalignment (arXiv 2306.07962); pseudo-simulation (arXiv 2506.04218);
  Bosch cross-benchmark (arXiv 2605.00066); Waymo approval guidelines (arXiv 2505.09880);
  RAND Driving to Safety (RR1478).
- Simulation: NuRec docs; Instant NuRec (arXiv 2607.14203); OmniDreams (arXiv 2606.03159);
  EUVS novel-view limits (ICCV 2025).
- Deployment: VLA-Perf (arXiv 2602.18397); FlashDrive (arXiv 2608.12932); Jetson Thor
  specs; TensorRT-Edge-LLM Alpamayo pipeline.
- Safety: Halos; UNECE NATM (WP.29-187-08); ISO/PAS 8800 summary (UL).
- Classical texts (standard, not re-verified): LaValle, Planning Algorithms; Thrun et al.,
  Probabilistic Robotics; Sutton & Barto; Ross et al., DAgger (2011).

---

## Tóm tắt tiếng Việt (v1.1)

Bản v1.1 sửa năm chỗ hở. Một, thêm Stage 1a không cần GPU: kiểm chứng hàm quyết định của
gate trên dữ liệu tổng hợp có sự thật biết trước, đo false-GO và false-NO-GO, và chứng
minh rằng bỏ phân tầng làm false-GO tăng; chưa qua bài này thì chưa thuê GPU. Hai, ba dự
đoán viết trước khi đo (noise floor giữa các seed, tỉ lệ rollout vượt deviation cap, và
throughput theo độ phân giải render) trở thành tiêu chí chấp nhận của Stage 1b. Ba, ma trận
ánh xạ stage ↔ G ↔ pillar ↔ C1–C10 ↔ bullet ↔ artifact, đặt G4 và G7 vào Stage 4. Bốn, quy
tắc phụ thuộc: mọi số closed-loop của nanoAlpamayo phải đi qua gate, nanoAlpamayo không
bắt đầu trước khi G1 v1 tồn tại, khi va chạm giờ thì gate thắng, và ngân sách 12 tuần đầu
cộng lại khoảng $12–14k. Năm, mỗi stage đóng bằng ba cổng làm chủ: teach-back có ví dụ số
tự tính, câu hỏi chuyển giao sang tình huống lạ, và cổng bằng chứng tái lập được từ clone
mới. Hiểu nhầm cần tránh: cảm giác "đã hiểu" không phải tiêu chí đóng stage.
