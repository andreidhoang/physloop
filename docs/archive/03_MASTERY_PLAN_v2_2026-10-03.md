# 03 — Mastery plan v2 (3 October 2026)

**Status:** authoritative for the skill pillars, the per-stage curriculum, the mastery gates and the skill-coverage map. It replaces v1.1 (archived). It contains **no schedule**: stages S1–S8, their order, costs and kill checks live only in `02_PHYSLOOP_PLAN.md`. The evidence is in `research/2026-10-03-verification/report.md`; posting frequencies come from the role notes in that folder.

**Target.** The capabilities behind NVIDIA's live end-to-end autonomous-driving postings, read on 2026-10-03 (`05_ROLE_MAPPING.md`). Credential lines (years, PhD) are ignored by design. Every requirement becomes a skill with a measurable acceptance test and a public artifact.

**"Principal", stripped of tenure:**
- own one link of the loop end to end;
- decide with measurements;
- write the design doc others build on;
- ship code that survives review by the people who wrote the stack.

The evidence must be public and checkable: an artifact published from this repo, issues and PRs in AlpaSim, a preprint, reproduced numbers.

---

## 1. Read the role as a system

```
 [3] Data engine ──► [1][2] Train VLA (IL → SFT → closed-loop correction → RL) ──► [4] Evaluate & release ──► [5] Runtime on Thor
        ▲                                                                              │
        └──────────────────────── failures found in [4] and on the road ───────────────┘
```

This is NVIDIA's "AI 2.0" framing: a reasoning VLA replaces hand-built modules and is distilled to run on DRIVE AGX Thor next to a classical safety stack. PhysLoop owns link [4] end to end. nanoAlpamayo (S7) exercises links [1]–[3].

## 2. Seven skill pillars, with verified demand

The demand figure is the share of qualification sections that mention the cluster: 43 comparable NVIDIA AV postings and 212 postings from 14 competitors, read 2026-10-03, ±5–10 points.

| Pillar | Demand (NVIDIA / competitors) | You must be able to… | Built in | The trap |
|---|---|---|---|---|
| **P1** Driving as a decision problem | Motion planning 30% / 29% | Derive the unicycle model and AR1's exact heading update; explain why (accel, curvature) × 64 at 10 Hz; explain open- vs closed-loop metrics | S6, S7 | Optimizing open-loop L2, which shows no significant correlation with closed-loop score (Bosch, n = 7) |
| **P2** VLA modeling | VLA/VLM/LLM 49% / 27% | Build a multi-camera VLM with 180 tokens per image (patch 16 × merge 2), a discrete trajectory tokenizer, and a flow expert that reads the KV cache | S7 | Treating the flow expert as a black box; not knowing which path is deployed |
| **P3** Learning algorithms | RL 23% / 16%; imitation / E2E 23% / 7% | DAgger, flow matching (CFM, Euler sampler, ODE→SDE for log-probs), GRPO with reward design and KL control, distillation (GKD), RoaD / OPTED closed-loop correction | S7 | The proxy reward improves while behaviour degrades (AR1's r_reason row); rollout and trainer log-prob mismatch |
| **P4** Data engine | Data 23% / 17%; world models / neural recon 21% / 13% | Location-grouped splits, auto-labeling CoC with a teacher, 3DGS rendering limits | S3, S7 | Frame-level random splits: with k = 20 frames per location and f = 0.2, 98.85% of locations land in both splits |
| **P5** Simulation and evaluation | Evaluation 28% / 39% (51% / 57% of responsibilities); simulation 35% / 24% | Treat the gate as a decision function; paired, cluster-adjusted non-inferiority; power; reliability (VIM, ICC); agreement (Bland–Altman, κ); construct validity | S1–S5 | Seeds counted as independent scenes; aggregates hiding strata; agreement on the same reconstruction read as validity |
| **P6** Deployment and real time | C++ 67% / 58%; CUDA 40% / 13%; real-time 35% / 18%; quantization / TensorRT 7% / 16% | Roofline on Thor (273 GB/s); FP8/NVFP4 formats; closed-loop precision verdicts; deterministic C++ | S4, S4b, S6, S8 | Assuming "4-bit is 4× smaller"; batch-1 eager latency called deployable; reading emulated FP8 on Thor as real |
| **P7** Safety and production | Safety / V&V / standards 19% / 19% | Dual-stack guardrails; allowed-claims writing; what ISO/PAS 8800, SOTIF, UL 4600 and UN R[185] ask of ML validation; release notes | S4 (release note), S6 | Claiming "safe" from simulation; calling your monitor "Halos" |

**Ladder rule.** Every pillar descends L5 user-visible behaviour → L4 system → L3 algorithm/math → L2 kernel, memory, precision → L1 silicon. A pillar is mastered only when you can go down to L2 and back up with numbers.

## 3. Ownership and the learning method

**★ Huy writes, from blank:** the measurement instruments, the statistics, the RL loss math, the verifier logic, every prediction, the tolerances and every causal diagnosis. **Agents write** plumbing, harnesses, scaffolds, failing tests, sweeps, plots and reviews. `02` §3 marks the ★ cores per stage.

**Order inside every rep:** concept and derivation first; then Huy's written prediction; then measurement; then reconciliation. A prediction comes after the mechanism has been derived. A prediction written before that is a guess and is logged as one.

**Why this split** (verified learning evidence):
- In a randomized Anthropic study, AI assistance cut quiz mastery by 17% (d = 0.738, p = 0.010) with no significant speed gain. The cognitively engaged interaction patterns preserved learning (Shen & Tamkin, arXiv 2601.20245).
- In a high-school field study, unguarded GPT access made students 17% worse once access was removed (Bastani et al. 2025, PNAS).
- Retrieval practice beats restudy on delayed tests (Roediger & Karpicke 2006).

## 4. Curriculum by stage

Every exercise has a known answer that can be checked on a CPU. The answers were computed and verified in the research notes. Do the exercise *before* the stage's build work.

| Stage | Primary sources | Derivation to own | Exercise (known answer) | Trap |
|---|---|---|---|---|
| **S1** Gate core | McNemar 1947; Clopper & Pearson 1934; Hanley & Lippman-Hand 1983; Efron 1979; Cameron, Gelbach & Miller 2008; Blackwelder 1982; Tango 1998; Berger 1982; Durkalski 2003 | Exact McNemar null; zero-failure bound (1 − p)ⁿ = α ⇒ p ≈ 3/n | b = 12, c = 6 ⇒ Σ_{k≤6} C(18,k) = 31,180; one-sided p = 0.11894, two-sided 0.23788. 0/441 ⇒ 0.677% one-sided, 0.833% two-sided | Uncorrected χ² gives 0.157; an unpaired z-test gives 0.399; calling 3/n a two-sided 95% CI |
| **S2** Reliability | JCGM 200:2012 (VIM); Shrout & Fleiss 1979; Koo & Li 2016; NHTSA pre-crash typology DOT HS 810 767 | Var(mean) = σ²/(nm)·[1 + (m − 1)ρ] | m = 3 seeds, ρ = 0.6 ⇒ design effect 2.2 (simulation gives 2.236) | Counting seeds as scenes; ranking card scores from different protocols (0.73 / 1.37 / 1.50 vs 0.35 / 0.72) |
| **S3** Envelope and agreement | NeRF Eq. 3 (arXiv 2003.08934); 3DGS Eq. 5 (arXiv 2308.04079); EUVS (arXiv 2412.05256); Bland & Altman 1986; Campbell & Fiske 1959 | Alpha compositing from dT/dt = −σT; EWA projection Σ′ = J W Σ Wᵀ Jᵀ, so an on-axis splat of std σ projects to fσ/z pixels | σ = [1, 2], δ = [0.5, 0.5] ⇒ α = [0.3935, 0.6321], T = [1, 0.6065], weights [0.3935, 0.3834], opacity 0.7769 | Reading agreement on one reconstruction as validity |
| **S4** Degradation ladders and precision | FP8 formats (arXiv 2209.05433); MX formats (arXiv 2310.10537); NVIDIA NVFP4 blog; SmoothQuant (arXiv 2211.10438); Roofline (Williams et al. 2009); PMPP 5th ed. (2026); Kocmi et al. 2021; Deutsch et al. 2023 | B* = b·F / (2·BW); NVFP4's effective bits are 4 + 8/16 = 4.5 | NVFP4 saves 16/4.5 = 3.556× vs FP16 and 1.778× vs FP8; E4M3 max = (1 + 6/8)·2⁸ = 448 | "4-bit = 4× smaller"; "E4M3 tops out at 240"; assuming every metric moves the same way under quantization |
| **S5** Construct validity | Cronbach & Meehl 1955; Messick 1995; Kane 2013; Fieller, Hartley & Pearson 1957; Bonett & Wright 2000; SIMPLER (arXiv 2405.05941); Kadian et al. (arXiv 1912.06321); PolaRiS (arXiv 2512.16881) | Fisher-z CI for Spearman's ρ and the n it implies | Gate [1..5] vs reference [2,1,3,5,4] ⇒ ρ = 0.8; exact P(ρ ≥ 0.8 \| H0, n = 5) = 8/120; at n = 10, CI ≈ [0.24, 0.96] | Counting degraded variants of one parent as extra n; tuning the simulator or policies on the validation set |
| **S6** Guardrail | RSS Lemma 2 (arXiv 1708.06374); Halos page; UL 4600 Ed. 3; ISO/PAS 8800:2024 metadata; NHTSA SGO data dictionary (no fault field) | Worst-case kinematics behind the RSS d_min | v_r = v_f = 20 m/s, ρ = 0.5 s, a_accel = 3, a_min,brake = 4, a_max,brake = 8 m/s² ⇒ **d_min = 43.16 m** | Assuming the lead car brakes no harder than you (gives 18.16 m); calling the monitor "Halos" |
| **S7** nanoAlpamayo | AR1 §3.2.2 and Eq. 5; Ross & Bagnell 2010; DAgger (arXiv 1011.0686); Flow Matching (arXiv 2210.02747) and its Guide (arXiv 2412.06264); Flow-GRPO (arXiv 2505.05470); π_RL (arXiv 2510.25889); PPO (arXiv 1707.06347); DeepSeekMath (arXiv 2402.03300); Schulman's KL note; GKD (arXiv 2306.13649); RoaD (arXiv 2512.01993); OPTED (arXiv 2609.20756) | Exact heading update Δθ = κ(vΔT + aΔT²/2); BC's T²ε vs DAgger's uTε; the CFM marginal field; ODE→SDE drift | Constant arc v = 10 m/s, κ = 0.02 m⁻¹, T = 6.4 s ⇒ forward-Euler endpoint error **0.5972 m**; AR1 Eq. 5 gives **0.00199 m**. Alternate: 1-D Gaussian flow, 10 Euler steps ⇒ std 0.43078, relative MSE 1.92% | A forward-Euler decoder adds a 0.60 m systematic error, the same order as the 0.913 m target; an inverted k3 ratio; computing KV with 32 heads instead of 8 KV heads |
| **S8** Throughput | AlpaSim TELEMETRY.md and OPERATIONS.md; NuRec scaling page; PhyAI (arXiv 2608.03682); VLA-Perf (arXiv 2602.18397, modeled only) | Batching crossover with KV reads included; balance equation N_gen·g ≈ N_train·t | Alpamayo 1.5 BF16 reads 15.17 GB per token ⇒ 15.17 / 273 GB/s = **55.6 ms/token** on Thor; a 3B FP8 student ⇒ **11.0 ms** | Reading NuRec's "100 FPS" as per-camera; profiling AlpaGym's 8-step smoke preset as production; quoting VLA-Perf's modeled Thor numbers as measured |

**Free lecture series** (latest editions verified in the curriculum notes): Tübingen Self-Driving Cars (Geiger) for P1/P4; Berkeley CS285 for P3; Stanford CS231n and CS336 for P2. Use them to support a stage, never as a separate track.

## 5. Mastery gates: a stage closes only when all three pass, in order

1. **Teach-back.** Explain the stage's mechanism from memory with one hand-worked numeric example, for example the McNemar 12-vs-6 case or the B* crossover.
2. **Transfer question.** Apply the mechanism to an unfamiliar shape or failure mode: a stratum with 5 scenes; an FP8 student whose score rises while 18% of its rollouts leave the envelope; a monitor that masks a regression.
3. **Evidence gate.** The artifact exists and reproduces from a fresh clone, and its prediction-vs-measurement reconciliation is written, with misses explained causally.

**Owner proof for ★ cores.** Either a cold reconstruction of a representative core from blank, or a diagnosis of a deliberately planted defect. Either one establishes ownership without redoing every delegated line. A failed owner proof means redoing the core. No stage is closed by a feeling of understanding.

## 6. Skill coverage and honest gaps

● = primary evidence, ◐ = partial.

| Cluster | NVIDIA / comp. | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|---|---|---|---|---|---|---|---|---|---|
| C++ | 67% / 58% | | | | | | ● | ◐ | |
| Python | 65% / 64% | ● | ● | ● | ● | ● | ◐ | ● | ● |
| VLA / VLM / LLM | 49% / 27% | | | | ◐ | | | ● | |
| Evaluation / metrics | 28% / 39% | ● | ● | ● | ● | ● | ● | ◐ | |
| Simulation | 35% / 24% | | ● | ● | ● | ● | ◐ | ◐ | ◐ |
| Motion planning | 30% / 29% | | | | | ◐ | ◐ | ◐ | |
| RL / post-training | 23% / 16% | | | | | | | ● | ◐ |
| World models / neural recon | 21% / 13% | | ◐ | ● | | | | ◐ | |
| CUDA / kernels | 40% / 13% | | | | ◐ (S4b) | | | | ◐ |
| Quantization / TensorRT | 7% / 16% | | | | ● | | | ● | |
| Distributed training | 14% / 18% | | | | | | | ◐ (one node) | |
| Safety / V&V | 19% / 19% | ◐ | ◐ | ◐ | ◐ | | ● | | |
| Publications | 33% / 24% | | | | | ● | | ◐ | |
| Open source | 14% / 4% | ● | ● | ● | ◐ | ◐ | ◐ | ◐ | ◐ |

**Gaps, named honestly.** They stay open until a later stage is added to `02` through a `DECISIONS.md` entry.
- **C++, the most common hard requirement.** Only S6 covers it fully. The next C++ artifact would be the AlpaSim monitor service, or later a DriveOS-style TensorRT runtime.
- **Multi-node distributed training (14% / 18%).** Uncovered: S7 uses one 8×H100 node.
- **Real-time work on actual Thor hardware.** Partial without a device. The Jetson AGX Thor kit is $5,499, collaborator tier only (P-002).
- **Kubernetes / SLURM.** Optional, through AlpaSim's `SLURM_ENROOT` run method.
- **Shipped production and years of experience.** Not addressable by a solo project. 13 of the 43 NVIDIA postings (30%) require 12+ years; none of the 212 competitor postings does. Postings treat public evidence as a complement, never as a substitute.

**Beyond S8, not scheduled.** The data-engine flywheel at scale (G8 implemented), multi-node training of a 2–3B nanoAlpamayo, OmniDreams as a second renderer tier, and on-device Thor latency. Each enters `02` only through a new `DECISIONS.md` entry.

---

## Tóm tắt tiếng Việt

- **Vai trò của bản v2.** Đây chỉ là bản đồ kỹ năng và giáo trình; lịch, chi phí và kill check nằm duy nhất trong `02`.
- **Bảy trụ cột P1–P7.** Mỗi trụ cột gắn với tần suất thực tế trong 43 tin tuyển dụng AV của NVIDIA và 212 tin của đối thủ, đọc ngày 2026-10-03. Kỹ năng được đòi nhiều nhất là C++ (67%), VLA/VLM (49%), CUDA (40%) và evaluation (51% trong phần trách nhiệm).
- **Giáo trình theo stage S1–S8.** Mỗi stage có nguồn gốc, phép dẫn xuất phải tự làm, một bài tập có đáp số kiểm tra được trên CPU, và cái bẫy hay gặp. Làm bài tập trước khi build.
- **Đóng một stage.** Cần qua ba cổng: teach-back có ví dụ số, câu hỏi chuyển giao, và bằng chứng tái lập từ clone mới. Phần ★ phải có owner proof (dựng lại từ trắng, hoặc chẩn đoán lỗi cài sẵn).
- **Khoảng trống còn mở.** C++ ngoài S6, huấn luyện multi-node, và phần cứng Thor thật.
