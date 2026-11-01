# Decision log (append-only; newest at the bottom)

Format: `D-nnn · date · decision · why · what it changes`. A decision here overrides older
text in any doc until that doc is revised.

- **D-001 · 2026-09-22 · Field pick.** Physical-AI post-training systems, autonomous driving
  on NVIDIA's open Alpamayo stack as the beachhead (not humanoids, world models as
  research, kernels alone, or AI-for-science). Why: Jensen-style criteria — existence proof
  + unbuilt stack, insanely hard, workload→architecture, beachhead with real revenue,
  intersection of domains. Changes: PhysLoop core plan created.
- **D-002 · 2026-09-23 · Hackathon goal dropped.** The Mise/GR00T/SO-101 fork existed only for
  the Nebius × NVIDIA hackathon and is retired; Mise's specification layer (evidence
  standard, contracts v2, tape-diff parity, failure taxonomy) is ported into the gate.
  Note: the hackathon rules do allow simulation-only Physical AI entries (rules re-read
  verbatim); an earlier claim that hardware was mandatory was wrong.
- **D-003 · 2026-09-23 · Correction.** AlpaGym (NVlabs, first commit 2026-06-16) already
  runs closed-loop GRPO on Alpamayo's flow-matching expert with SDE log-probs and bounded
  staleness. Our earlier claim ("NVIDIA never trains the expert") held only for the
  open-loop recipe. Changes: v1 deliverables D4 (async engine) and D5 (RL through the
  expert) demoted; RLinf fork dropped.
- **D-004 · 2026-09-23 · PhysLoop v2.** The project is an open, validity-checked release gate
  for the Alpamayo teacher → Thor-student path (G1–G8). Headline result is G3, the gate's
  own predictive validity. Kill check Dec 14 2026.
- **D-005 · 2026-09-23 · Second spine.** nanoAlpamayo added so that posting bullets 1–2
  (design/train models; build/fine-tune VLAs) have evidence; acceptance = reproduce
  NVIDIA's published 0.5B numbers on the public validation set.
- **D-006 · 2026-09-23 · Repo conventions.** No code before `/reread`; predictions before
  runs; append-only `evidence/`; upstream PRs preferred over forks; allowed claims per
  `02_PHYSLOOP_PLAN_v2.md` §5.

- **D-007 · 2026-09-24 · Mastery plan v1.1.** Stage 1a added: the gate's decision function is
  validated on synthetic data with known effect sizes (false-GO / false-NO-GO) before any
  rented GPU-hour. Dependency rule: every nanoAlpamayo closed-loop number comes from the
  gate; nanoAlpamayo starts after G1 v1 exists; when hours collide, G1–G3 win through
  week 6. Combined first-12-week compute ≈ $12–14k. Every stage closes by three mastery
  gates (teach-back, transfer question, evidence gate).

- **D-008 · 2026-09-24 · Parked.** *[SUPERSEDED in full by D-009 on 2026-10-03.]* PhysLoop is parked, not killed, by the owner's decision 3 of the
  cross-project "Frontier Lab Lens" review (recorded in `ladders/worklogs/2026-09-24_vlad_lens_v7.md`).
  Why: the six-month target was fixed as frontier LLM labs (kernels/serving of a real LLM and
  rigorous agent evaluation), which this project does not serve; its first 12 weeks price at $12–14k
  against a $2,500 envelope; gated assets are requested, not granted; the challenge needs an org
  email; it was the third field pick in 21 days. What stays: every document as written, the free
  gated-asset requests, this log. What stops: Stage 1a and all G1–G8 work, the Oct 31 challenge,
  the Nov 10 GTC poster. Re-entry test on **2026-11-20**: re-enter only if the six-month target flips
  to NVIDIA autonomous driving, or a collaborator brings data and compute; otherwise the park
  continues. The Dec 14 kill check is suspended with the project. Changes: `00_START_HERE.md`
  "Where we are" block.

- **D-009 · 2026-10-03 · Standalone and active.** By the owner's instruction, PhysLoop is its own
  project, entirely separate from any other program. Nothing is parked. It takes no hours, budget,
  deadline or decision from another project, and it links to none. Supersedes D-008 in full: the park,
  the 2026-11-20 re-entry test and the suspension of the kill check. Changes: `DOC_AUTHORITY.md`
  (park row removed), `00_START_HERE.md` (where we are), `CLAUDE.md`.
- **D-010 · 2026-10-03 · Plan v3 adopted from the primary-source verification**
  (`research/2026-10-03-verification/`, claim ledger in `report.md` §10). Why: the verification
  overturned load-bearing assumptions:
  - G3 was unfalsifiable: no named reference, and a ρ < 0.6 rule cannot be decided at n = 10.
  - Challenge registration closes 2026-10-18 and needs an organization email.
  - AlpaGym accepts no outside PRs.
  - NVIDIA's quantization recipe already covers the action expert.
  - NVIDIA has announced a distillation script three times.
  - The PAI-AV and NuRec licenses treat benchmarking data as confidential.
  - The v2 cost model and prices were wrong.

  What it changes:
  - G3 is restated as construct validity with a pre-registered PASS / KILL / INSUFFICIENT rule at
    n ≈ 28–34 checkpoints.
  - G5 merges into nanoAlpamayo, with a pivot-to-evaluation kill.
  - G6 becomes closed-loop per-format verdicts.
  - G7 becomes a measurement report; upstream PRs target AlpaSim.
  - The allowed claims are corrected (one-sided bounds; R[185] and the ADS GTR added; no
    "predictive validity").
  - The fixed 2026-12-14 kill check is replaced by KC-1…KC-7 relative to D (default 2026-10-05).
  - D-007's fixed $12–14k is replaced by budget tiers ($0 / ≤ $2.5k / ≤ $6k / collaborator).
  - A license publication gate (KC-L) is added; no PAI-AV or NuRec download before S2.
  - Stages S1–S8 replace the 12-week table.
  - P-001 is closed.
- **D-011 · 2026-10-03 · One source of truth per topic.**
  - `02_PHYSLOOP_PLAN.md` holds the plan; the version lives in the text, not the file name.
  - `03` holds pillars, curriculum and mastery gates, with no schedule.
  - `04` holds verified technical facts and the preserved 22-Sep predictions.
  - `05` holds the live posting map.
  - Superseded versions (01 v1, 02 v2, 03 v1.1, 04 v1, 05 v1) and the rebuild-time `START HERE.txt`
    move to `archive/`. The verification evidence lives in `research/`. Entry point: `README.md` →
    `docs/00_START_HERE.md`.

- **D-012 · 2026-10-25 · Plan v4 adopted: the teacher → student → Thor loop, measured by the gate.**
  Input: the owner's decision doc "Alpamayo-Grade AV Engineering: #1 Project Decision"
  (`research/2026-10-25-teacher-to-thor/decision_input_2026-10-25.md`). It was verified by four
  research lanes (`verification.md`) and reviewed against plan v3 (`review.md`).

  Why:
  - The doc's strategic read held. The released Alpamayo checkpoints are cloud teachers, and the
    open gaps are the teacher → student step and real-time serving on Thor.
  - NVIDIA moved first on the student:
    - alpamayo-recipes PR #42, a public, unmerged distillation recipe to a Cosmos-Reason2-2B
      student (2026-10-20);
    - PR #45, 2B students at 171 ms FP8 on DRIVE AGX Thor;
    - Alpamayo 2 Edge, announced 2026-10-21 with weights promised "later this month".

    That activates the doc's own contingency ("use Edge as the P2 baseline").
  - The doc's gates can only be decided with the gate's statistics (review §3):
    - the RL gate (≥ 20% relative) is likely INSUFFICIENT on 441 scenes at low base rates;
    - the ≥ 90% gate must name its metric, because the card score (km between events) and the
      challenge `scene_score` (in [0, 1]) differ;
    - by the doc's own formula, the 2B reasoning student needs 139 ms against a 100 ms budget.

  What it changes. v4 overrides v3 in each of these places:
  - **Headline.** v3's G3 construct-validity study is replaced by the loop, with the gate as its
    instrument. G3 narrows to reliability (R1) plus the known-order ladders the pipeline produces
    (R7/R9). R2–R6 are deferred.
  - **Stages.** S1 and S2 are kept. The rest are redefined:
    - S3: closed-loop verdict on NVIDIA's 2B student;
    - S4: teacher RL in AlpaGym;
    - S5: on-policy correction and the CoC-budget curve;
    - S6: Jetson AGX Thor serving;
    - S7: HIL with latency injection;
    - S8: shadow mode, optional.

    v3's S3 (envelope and agreement), S5 (cross-paradigm), S6 (guardrail) and S8 (throughput
    report) are deferred. Their text stays in `archive/02_PHYSLOOP_PLAN_v3_2026-10-03.md`.
  - **Deliverables.**
    - G4 is deferred.
    - G5 becomes the headline.
    - G6 moves onto the device.
    - G7 is split across S2, S6 and S7.
    - G8 becomes the shadow-mode data engine.
    - G9 (teacher RL study) is new.
  - **Student.** NVIDIA's own 2B student (Edge) is measured first, instead of a student we distill.
    nanoAlpamayo becomes the correction and CoC-budget work on top of it. v3's KC-7(a) pivot is
    now the default.
  - **Order differs from the doc**, which ran RL before distillation:
    - The doc's order assumed we would distill from the RL'd teacher. With NVIDIA's student that
      premise is gone.
    - S3 costs only sweeps, and it holds up whatever NVIDIA releases next, because NVIDIA
      publishes open-loop error and latency only.
    - S3 and S4 depend only on S2 and can be swapped. If KC-E finds no Edge weights, S4 runs
      first, which is the doc's order.
  - **RL arms.** The doc's AlpaJudge reasoning–action reward runs only in the open-loop recipe,
    because AlpaGym freezes the VLM (`cotrain_vlm = False`).
  - **Thor bug fix.** The doc's "empty-CoC Thor bug fix" is dropped. That was an Alpamayo 1 setup
    issue on DRIVE AGX Thor, resolved by 2026-10-06.
  - **Thresholds.** The doc's thresholds stay as Huy's targets. They are restated in paired form
    on a named metric before pre-registration. Matching the card score is a diagnostic, not a gate.
  - **Kill checks.**
    - KC-L is kept.
    - KC-L2 is added: no weights or labels trained on PAI-AV/NuRec-derived data are published
      before an answer on license §4.6.
    - KC-E is added: re-check Edge on 2026-10-31 and at S3's start.
    - KC-1…KC-8 are re-dated to D = 2026-10-26. v3's S1 had not started (no `gate/`, `journal/`
      or experiment file on 2026-10-25).
  - **Manifest.** It adds the resolved lockfile, `n_sim_steps` requested and actual, the score
    definition, and the overlap with PAI-AV's `alpamayo_aligned` split.
  - **Budget.** The cloud tiers stay. The Jetson AGX Thor (P-002) and the shadow-mode rig (P-006)
    become separate hardware lines. P-005 stays open.
  - **Cross-project question.** The doc raises a priority question about another project. Under
    D-009 that is outside PhysLoop and is recorded nowhere in this repository.
- **D-013 · 2026-10-25 · Documentation set revised to plan v4.**
  - `01` v3 adds the loop to the goal and the map, and two concepts: C11, the Thor latency budget,
    and C12, shadow mode is open-loop.
  - `03` v3 re-keys the curriculum to the v4 stages, computes the new known answers in
    `research/2026-10-25-teacher-to-thor/notes/arith_check.py`, and adds the Think Autonomous
    course as an optional supplement.
  - `04` v2.1 corrects the record:
    - TensorRT 11.x supports DRIVE AGX Thor only; Jetson AGX Thor stays on TensorRT 10.16.2;
    - AlpaSim has merged no external PR since 2026-08-17;
    - the distillation PR #42 exists;
    - stage tags are re-mapped, and §9 adds the deltas.
  - `05` v2.1 re-maps stages to postings.
  - `nano_alpamayo/README.md` is re-based on NVIDIA's student.
  - The previous versions are archived.
  - The decision doc moved from the repository root to `research/2026-10-25-teacher-to-thor/`.

- **D-014 · 2026-10-25 · Starred cores are tool-enforced; AGENTS.md mirrors the contract for Codex.**
  Why: the generator–verifier operating model needs ownership to be a mechanism, not a memory. If one
  agent context writes both a core and its tests, the two share its misreadings, so a test can pass on
  a wrong core (common-mode failure).
  What it changes:
  - A manifest (`.claude/starred-cores.txt`, starting with `gate/verdict.py` and `gate/stats/`) lists
    the files only Huy writes.
  - A Claude Code hook (`.claude/hooks/guard_starred.py`, wired in `.claude/settings.json`) enforces it:
    - it denies Edit, Write, MultiEdit and NotebookEdit on listed paths;
    - it denies Bash commands that name a listed path next to a write verb;
    - it snapshots listed files before every Bash command and restores any change afterwards, moving
      the displaced version to `.claude/starred-guard-backups/` so nothing is lost;
    - it fails closed.
  - The manifest, the hooks directory and the project settings protect themselves. Only Huy unlocks
    an entry, by commenting it out in his editor.
  - `AGENTS.md` carries the same contract for Codex, where no hook runs, so there the rule is binding by
    instruction. `CLAUDE.md` and `AGENTS.md` change together.
  - Verified: 38 pipe tests in a sandbox; a live Edit denied; a live Bash probe reverted.
  - Known limits:
    - background Bash commands escape the post-check;
    - user-level or managed settings (`disableAllHooks`) can switch hooks off;
    - the static Bash check can refuse a command that only mentions a starred path in text;
    - on case-insensitive macOS (APFS), a file-tool write to `Gate/Verdict.py` reaches
      `gate/verdict.py` (same inode) and is not denied (verified 2026-10-25). Fix, Huy's edit, since
      the script protects itself: in `is_starred()`, compare `relpath.lower()` with `e.lower()` when
      `sys.platform == "darwin"`.

- **D-015 · 2026-10-27 · Plan v4.1: corrections, savings, and evidence for the target role.**
  Input: the owner's instruction of 2026-10-27, which said:
  - optimize the plan so that, finished end to end, it maximizes checkable evidence of exceptional engineering and research skill for NVIDIA's JR2010863 (Senior ML and Simulation Engineer – AV, AV Simulation team);
  - leave out years, degree and location;
  - use the 2026 NVIDIA stack.

  Seven read-only research lanes covered upstream changes, RL, simulation quality, statistics, the student and data, the cluster, and Thor (`research/2026-10-27-role-alignment/`, with notes L1–L7, `report.md`, `verification.md` and `notes/arith_check.py`).

  Why:
  - **The instrument had unmeasured knobs.** Verified 2026-10-27:
    - on `e2e_challenge`, the Alpamayo driver ignores the session seed, so S2's "seeds" are unseeded replicates;
    - the Cosmos-based Harmonizer is on or off depending on the config;
    - `e2e_challenge` has no CAT-K;
    - the evaluator was fixed mid-challenge (#198/#215);
    - post-processing alone moved AlpaSim scores by +22% to +123% (ECO).
  - **v4 stated four things wrongly:**
    - `alpamayo_aligned` labels every clip, so "exclude it" excludes everything;
    - AlpaGym's estimator is a per-step-row geometric-mean ratio with a closed-form Gaussian KL, off by default, not a trajectory ratio with k3;
    - the comfort/jerk arm is not configurable;
    - AlpaGym's default `n_generation` of 1 trains nothing.
  - **Upstream moved:**
    - TacitVLA (NVIDIA, 10-13) published unpaired AlpaSim scores for 2B-backbone research variants;
    - #224 closed, with disclosure tied to NVIDIA's next release;
    - PRs from non-collaborator accounts had merged on `e2e_challenge`, among them #167, from an OPTED co-author;
    - a 2B Alpamayo policy had been timed inside AlpaGym (alpamayo1.5 PR #36);
    - AlpaGym's public branches moved until 2026-09-03;
    - S5's list price rose;
    - the 441-scene protocol and the tooling sit on different AlpaSim branches.
  - **Savings exist:**
    - NVIDIA's expert CUDA graph (alpamayo1.5 PR #36, bitwise-checked by NVIDIA);
    - PR #42's one-node stage-2 trainer as S5's learner;
    - AlpaSim's own external-driver mode and delay buffer for S7;
    - reusing S2's teacher sweeps in S3;
    - a re-scoped KC-E fallback, since a pilot PR #42 student would never reach the recipe's on-policy phase.
  - **The role's weak lines can be served inside existing stages.** D1, D2, D4, Q4, Q5 and X3 can be measured on one node, on NVIDIA's own presets, run methods and measurement definitions, with no new stage and no new framework.

  What it changes:
  - **`02` v4 → v4.1** (v4 archived):
    - §0.1 lists corrections K1–K14, savings V1–V6, the tiers [C] / [O] / [G] and the cut order;
    - every stage card gains [C] / [O] / [G] items and a Role evidence line;
    - S2 runs replicates and pre-registers its seed mode, Harmonizer state and run method;
    - the S2 manifest gains the simulator configuration;
    - the S4 ★ row and arms are corrected;
    - KC-E is re-scoped;
    - §5 prices are re-verified on 10-27;
    - §6 adds claims rules on replicates and on scale and origin;
    - §9 adds P-012 to P-019 and the new pre-registered choices.
  - **Effort.** About 320 h becomes about 370–405 h of [C] work, with up to about 40 h of [O]. KC-1 re-forecasts; the cut order is S8, then [O], then the [C] role items. The corrections are never cut.
  - **`05` v2.1 → v3** (archived): adds §3, the JR2010863 evidence contract (13 paraphrased lines, v4 vs v4.1 marks, the artifacts that carry each), and refreshes §1 from the 2026-10-27 sweep.
  - **`03` v3 → v3.1:**
    - pillar P8, training systems and cluster operations;
    - curriculum rows S1–S5 and S7 corrected and extended with known answers computed in `notes/arith_check.py`;
    - §6's coverage and gaps updated.
  - **`01` v3 → v3.1:** C2, C4, C5, C6 and C10 corrected; C13 added, on the closed loop as a throughput system.
  - **`04` v2.1 → v2.2:** the AlpaSim and AlpaGym rows corrected; premise notes on E1, E2 and E4; §10 adds the facts verified on 2026-10-27.
  - **`nano_alpamayo/README.md`:**
    - the A1 rule is keyed on `alpamayo_aligned`'s labels, with a chunk-aware manifest;
    - A7 is added: every round reproduces from its manifest;
    - the learner, the ladder and the CoC-curve steps follow S5.
  - **`00_START_HERE.md`, `DOC_AUTHORITY.md`, `CLAUDE.md` + `AGENTS.md`** (changed together):
    - the current state;
    - the external-merge fact;
    - the replicate wording in the statistics defaults;
    - the manifest fields;
    - two never-claims: unseeded replicates called seeds, and single-node or solo work presented as cluster-scale or industry experience.

  Binding:
  - no new stage;
  - AlpaGym is used as shipped;
  - ★ cores, predictions, tolerances and margins stay Huy's;
  - P-002, P-004 (KC-L, KC-L2), P-005, P-006 and KC-E are unchanged;
  - every gated addition is a proposal below.

  Rejected, with reasons in the lane notes:
  - a Kubernetes path: AlpaSim ships no Kubernetes run method, and NVIDIA had no short-term fix for running its renderer container under Kubernetes without Docker-in-Docker (#72);
  - a Cosmos3-Nano-Sim-Auto adapter and NuRec-vs-Cosmos ranking: it would race NVIDIA, does not run on the plan's setup, and is INSUFFICIENT at n = 3;
  - switching S4 to NeMo-RL: it has no flow-matching or VLA path;
  - fixed IRT scene subsets: they change the estimand;
  - rare-event acceleration: replay traffic, and an unseeded CAT-K;
  - confidence sequences for one-shot verdicts: they cost 1.55–1.85× the width;
  - vLLM or SGLang inside the driver;
  - FP8 as a sweep speed-up: it changes what is measured;
  - Holoscan for S7.

  Notes on open proposals, from the 2026-10-27 evidence. The proposals' text is unchanged:
  - **P-007 is stronger.**
    - FlashDrive's 113.9 ms expert time already includes CUDA graphs and fusion, so the remaining gap to the bound is on the GPU side.
    - Split H1 into H1a (CPU launch overhead) and H1b (per-kernel ramp and tail), and run E-K0 with graphs on and off.
    - A fix speeds up S2–S5's sweeps before it helps on Thor.
  - **P-008 is stronger.** TensorRT-Edge-LLM already ships three of the items: media-aware KV reuse since v0.10.0; FP8 ViT attention and a ViT-embedding cache since v0.10.1. None is stated as validated for Alpamayo, but building them would race NVIDIA.
  - **P-009 is adopted in part by this decision.** Its numerical-correctness ladder (iv) is the protocol for any driver change used in a verdict sweep (P-015). Its glass-to-action latency (ii) is what AlpaSim's `planner_delay_us` models in S7. Items (i) and (iii) remain proposals.
  - **P-010 is weak on its own.** Keep it as P-007's parallelism-starvation (H2) test.

## Proposals (not decided)
- P-001 · ~~Submit a gate-vetted FP8 Alpamayo 1.5 to the AlpaSim E2E Closed-Loop Challenge.~~
  **Closed by D-010.** Registration closes 2026-10-18 and requires an organization email.
  Only through a collaborator, at most 3 private-set submissions.
- P-002 · Buy a Jetson AGX Thor dev kit ($5,499 MSRP, re-verified 2026-10-25; one retailer lists
  $6,269, out of stock) for on-device numbers. **Under plan v4 it blocks S6 and S7** (it was
  collaborator-tier only under v3). Its numbers do not transfer 1:1 to DRIVE AGX Thor: the silicon
  and bandwidth are the same, but the software differs (JetPack 7.2.1 with TensorRT 10.16.2 vs
  DriveOS 7.2 with TensorRT 11.x).
- P-003 · Weekly hours. `02` §3 assumes 25–30 h/week. Record the real number; all windows are
  re-forecast at KC-1.
- P-004 · Huy sends NVIDIA a written request asking:
  - whether PAI-AV/NuRec license §3 permits publishing aggregate regression results computed on
    those datasets;
  - whether the fallback (scene-ID manifests + aggregate statistics) is permitted;
  - **(extended 2026-10-25)** whether §4.6 permits publishing weights or labels trained on
    PAI-AV/NuRec-derived data, such as an on-policy-corrected student.

  It blocks KC-L and KC-L2, which puts it on the critical path for every public number from S2 on.
- P-005 · Budget tier for this project: $0 / ≤ $2.5k / ≤ $6k / collaborator (`02` §5). Blocks any
  spend in S2.
  The tiers are re-stated in `02` v4 §5. The hardware lines (P-002, P-006) are separate.
- P-006 · Shadow mode (S8, optional). It needs three things:
  - the rig budget (~$5–8k estimated from verified parts);
  - Huy's own confirmation of the local rules for mounting recording equipment. The decision
    doc's caveat stands: an engineering reading, not legal advice;
  - a privacy plan (notice; anonymize faces and plates before publication).

  It blocks S8.
- P-007 · *(proposed 2026-10-26, inference/kernel review)* Split S6 into S6a (pre-device) and S6b
  (on Jetson AGX Thor).
  - S6a re-measures, then diagnoses, the flow expert on a rented RTX PRO 6000. FlashDrive's Table 1
    was measured on that GPU class, so the work is reproducible and cheap (a few GPU-hours).
  - **Correction it carries:** "the expert runs at ~20% of its bandwidth bound" is **not stated in
    FlashDrive**. It is the 2026-10-03 ledger's own arithmetic (≈ 22.3 ms bound ÷ 113.9 ms
    measured, 8 steps). It depends on three choices:
    - the table row (the baseline row's 192.9 ms gives ~12%);
    - the Workstation vs Server bandwidth;
    - a full KV re-read per step.

    Label it derived. The first S6a step is to re-measure, not to diagnose.
  - The first experiment (E-K0):
    - setup: official Alpamayo 1.5, BF16, B = 1, synthetic inputs, CoC fixed at 16 tokens, K = 8
      and 10;
    - measurements: `nsys` over the tick, `ncu` over one expert step, a copy microbenchmark for
      the achieved bandwidth.
  - The decision it informs: whether S6 contains a ★ kernel, and which one. The candidate causes
    and their fixes:
    - H1, many small kernels → CUDA graph / megakernel;
    - H2, parallelism starvation → split-K / batching samples;
    - H3, attention → split-KV;
    - H4/H5, copies or a wrong bound → no kernel.
  - Preconditions: KC-1 passed; P-005 recorded. Huy writes the mechanism and the predictions before
    the run.
- P-008 · *(proposed 2026-10-26)* Narrow S6's build list from seven items to four:
  1. a real engine with a per-stage timeline;
  2. **one** diagnosed bottleneck plus **one** fix, with a kernel only if the measurement requires
     it;
  3. FP8 vs NVFP4;
  4. the 30-minute sustained run.

  Two of the current items move or become conditional:
  - **Move "streaming reuse of history frames" to S5.** FlashDrive needs expert fine-tuning for it
    (≈ 0.2 m minADE6 drift without), so it is a model change, not serving.
  - Keep "camera-batched ViT" only if the profile puts the ViT on the critical path.

  The "fused flow-step kernel" becomes conditional on the S6a diagnosis.
- P-009 · *(proposed 2026-10-26)* Add four items to the S6/S7 measurement protocol:
  - (i) a **bandwidth-contention arm**: a synthetic process consuming X GB/s of the shared
    273 GB/s, standing in for the rest of the stack;
  - (ii) an explicit **glass-to-action** latency (oldest frame's exposure → command takes effect),
    kept distinct from model latency;
  - (iii) **pinned power mode and clocks**, recorded in the manifest;
  - (iv) a **numerical-correctness ladder**, with each tolerance set by its consequence at the next
    level and owned by Huy:
    - E0a: bitwise repeatability;
    - E0b: per-layer comparison against FP32;
    - E0c: action error in physical units;
    - E1: minADE6;
    - E2: paired closed-loop verdict per stratum.
- P-010 · *(proposed 2026-10-26)* Add a GPU-execution exercise with a known answer to `03` §4 S6. It
  currently covers the roofline only. The exercise:
  - write a GEMV / skinny GEMM (M = 64, the expert's shape);
  - measure achieved DRAM bandwidth vs grid size on a rented GPU;
  - reproduce the parallelism-starvation curve. On the RTX PRO 6000 (188 SMs), a 16-CTA grid must
    keep ~56–90 KB in flight per SM to saturate 1.792 TB/s, by Little's law at an illustrative
    latency of 500–800 ns.
- P-011 · *(proposed 2026-10-26)* Enter the Meta Global AI Developer Hackathon (Indie track) with a
  teacher → student gate entry. Huy reports the APPROVED email (2026-10-26).
  - **The event** (Official Rules: the Cerebral Valley event waiver dated 2026-10-25, read
    2026-10-26):
    - build window 10-26 12:00 ET → 11-04 21:00 ET (≈ 11-05 09:00 ICT; submit on the evening of
      11-04 ICT); judging 11-04 → 11-08; live finals 11-09 ET;
    - scoring: Impact 30%, Use of Meta AI tooling 25%, Technical depth and execution 20%, Demo 25%;
    - Indie prizes $150k / $100k / $75k / $50k / $25k; solo or teams of up to 4;
    - the entrant keeps ownership but grants the organizers a perpetual, irrevocable licence. Everything
      shared during the event (Discord, office hours) is non-confidential.
  - **The entry.** PhysLoop's gate decides whether Muse Glimmer 30B (Apache-2.0; reported as distilled
    from Muse Spark) can replace Muse Spark at short-horizon forecasting on Ho Chi Minh City traffic:
    - footage Huy owns, with faces and plates blurred before any API call or upload;
    - both models see the same frames and answer the same forecast question;
    - the label is what happened 2–3 s later, taken from SAM 3.1 tracks and audited on a
      hand-checked subset. Tracking errors are a different kind of failure from the models' reasoning
      errors, which is why the audit is required;
    - paired outcomes, stratified (density, time of day, rain, junction type) and cluster-adjusted by
      road; a one-sided non-inferiority bound; INSUFFICIENT blocks GO; a power table per stratum.
  - **Fallback (option D):** the same gate on public text and vision benchmarks, with no footage.
  - **Why this entry.** It reuses S1's core unchanged. It rehearses S3's question (a distilled
    student against its teacher) on a licence-clean pair. It tests whether the gate works beyond
    AlpaSim. PhysLoop itself cannot be entered:
    - PAI-AV/NuRec §1, §3 and §4.6 conflict with the submission warranty and the organizers' licence;
    - the S2 gates fall inside the window;
    - the rubric's 25% needs Meta models doing essential work.
  - **Binding if adopted:**
    - S1 runs as written. No real-model verdict until `verdict()` holds false-GO ≤ α on synthetic
      truth (the KC-1 rule). KC-E and KC-1 keep their dates;
    - ★ cores stay Huy's. Agents write the model harness, the SAM 3.1 pipeline, the demo UI and the
      video tooling;
    - nothing from PAI-AV, NuRec, AlpaSim or Alpamayo goes into the entry or any hackathon channel;
    - never film while riding; no vehicle-mounted rig (P-006);
    - claims: an open-loop forecasting measurement on N clips. Never "safe", no AV validation, no Thor
      numbers;
    - Meta and partner credits only. Any cash spend is P-005.
  - **Cost (planning inference).** S1 ≈ 20 h plus ≈ 20–30 h for the entry, against ≈ 34–40 h of
    capacity in the window at 25–30 h/week (P-003). Expect KC-1 to slip 0–3 days, or longer weeks.
  - **Hour cap:** ____ h
  - **Kill checks (Huy fills):**
    - K0 · Accepted (APPROVED email). Result: ____
    - K1 · A published problem statement fits the entry (checked at kickoff, 10-26). Result: ____
    - K2 · Enough own footage, anonymized. Minimum: ____ · Due: ____ · Result: ____
    - K3 · SAM 3.1 label error on the hand-checked subset. Tolerance: ____ · Due: ____ · Result: ____
    - K4 · Synthetic false-GO ≤ α before any real-model verdict. Result: ____
    - K5 · Hours spent against the cap. If exceeded: ____ · Result: ____
  - **Open (Huy's call):**
    - where the entry's code lives: a separate public repository that imports `gate/`, or a folder
      here, which would need CLAUDE.md's repo layout updated;
    - its open-source licence (Apache-2.0 or MIT), chosen before submission.
  - Decide before agents scaffold anything for the entry.
- P-012 · *(proposed 2026-10-27, D-015)* **One multi-node scaling measurement.**
  - **What it shows.** It is the only multi-node evidence for JR2010863's D1 and Q5. It runs only after the one-node models exist: S4's AlpaGym balance and scaling model, and S5's 1→8-GPU ladder.
  - **Option A (S5).** S5's FSDP2/DDP trainer on 2 × 8 H100 with InfiniBand for 4–6 h:
    - measurements: inter-node `nccl-tests`, strong and weak scaling at 8 vs 16 GPUs, a multi-node Slurm launch, and NCCL RAS plus the Flight Recorder on an injected cross-node hang;
    - cost: 64–96 H100-h, about $300–640. Nebius lists $72/h for 2 × 8 H100 on 10-27. RunPod Instant Clusters cost more if the listed H200 price is per GPU (unconfirmed).
  - **Option B (S4).** AlpaGym's shipped 3-node preset (`slurm_distributed_1_1_1`, 3 × 8 GPUs) for 2–3 h at steady state, against the one-node 1/3/4 preset:
    - it validates the S4 model's predicted 3-node speed-up;
    - cost: about $305–457, plus a $64–96 one-node baseline if that is not already an S4 run. A 2-node run needs a custom topology YAML.
  - **Predictions** (Huy): inter-node bus bandwidth `____`; the 16- vs 8-GPU (or 3- vs 1-node) efficiency `____`.
  - **Kill:** if inter-node bus bandwidth is below `____` within 45 minutes, or spend reaches `____`.
  - **Preconditions:** KC-1 passed; room in the P-005 tier.
  - **What it cannot show:** behaviour at 100+ GPUs, or production scheduling.
- P-013 · *(proposed 2026-10-27, D-015)* **A minimal R3: log replay vs CAT-K.**
  - **Setup:** both arms on one SHA (`main@affc2eab`, or the post-10-31 release), with only the traffic setting changed. Pass the 441 scene IDs explicitly, and score both arms with one evaluator (port the post-#215 corridor scorer, or re-evaluate the logs).
  - **Runs:** Alpamayo 1.5 BF16, replay × 1 and CAT-K × 3. CAT-K ignores its seed, so it gets repeats.
  - **Report:** κ, the flip rate per failure type and the paired difference, per deviation bin, each beside S2's replicate κ for the same bin. Rank claims are INSUFFICIENT at n ≤ 3 policies.
  - **Seeding PR:** a PR that threads the seed into CAT-K is Huy's outward action, and only if NVIDIA's next release lacks it.
  - **Cost:** 45–113 GPU-h, about $45–180 for one policy.
  - **Why:** it is the only route to the posting's "traffic models" clause (I2). It also tells S4 whether non-yielding replayed followers make `collision_any` punish legitimate braking.
  - **Gates:** un-defers v3's R3; needs P-005 and KC-L.
- P-014 · *(proposed 2026-10-27, D-015)* **A minimal R2/R4 for the renderer, in three tiers:**
  - (i) **A render-fidelity curve with no novel-view ground truth,** about $4–18:
    - render the 4 policy cameras at lateral offsets of 0, 1, 2 and 4 m;
    - project ClipGT obstacle cuboids and report detector recall against the offset;
    - compute KID against the recorded front-wide video;
    - Harmonizer off on all 441 scenes, on for a 100-scene subset.
  - (ii) **A policy-sensitivity probe,** about $3–13: the plan change caused by appearance (Harmonizer off vs on) at matched poses, against the replicate floor.
  - (iii) **A closed-loop Harmonizer-agreement arm,** about $23–90; up to about $64–323 if the Harmonizer cost scales with pixels:
    - one flipped-Harmonizer sweep per policy (Alpamayo 1.5 BF16 and FP8), with scene-level κ by bin;
    - whether S2's BF16-vs-FP8 verdict survives the flip.
  - **Gates:** un-defers parts of R2 and R4; needs P-005 and KC-L.
  - **Race risk:** partial. NVIDIA holds Harmonizer-off and Harmonizer-on challenge runs, so re-check the challenge reports in November.
- P-015 · *(proposed 2026-10-27, D-015)* **A GPU-batched `predict_batch` for the Alpamayo driver** (promoting `04` §7 E2):
  - **Build:** padded prompts and masks, with a per-sample generator seeded from each session's seed, so per-session random streams survive batching.
  - **Acceptance:** P-009's ladder (E0a–E1), plus a paired closed-loop verdict of the batched vs unbatched driver on the same checkpoint. That is a known null the gate must not flag, so it doubles as a validity test of the gate.
  - **Cost:** one equivalence sweep, about $34–132.
  - **Saving:** −263 to −964 GPU-h on S3–S5, only if Alpamayo's batch speed-up matches published non-Alpamayo analogs (2.26–2.84×). The Amdahl bound for Alpamayo is about 2.0–2.6×, by inference. It does not add to the expert-graph saving.
  - **Conditional sub-option:** a static KV cache with graphed decode, only if S2's profile shows decode dominates and batching is not adopted.
- P-016 · *(proposed 2026-10-27, D-015)* **A third S4 arm,** after the S4 pilot. Either:
  - (a) a Lagrangian collision constraint, where an outer loop updates the collision scale at checkpoint boundaries through config-only restarts (the optimizer reset is recorded); or
  - (b) reactive CAT-K traffic during training, re-enabled through AlpaGym's AlpaSim override arguments. That this works is unverified.

  Cost per arm: +112–225 H100-h × 3 seeds' worth, about $217–540. Gates: P-005; KC-L2 before releasing any checkpoint.
- P-017 · *(proposed 2026-10-27, D-015)* **A PR #42 stage-1 pilot student as S3's fallback.** This was v4's KC-E fallback.
  - **Price** (inference from NVIDIA's own 1.77 s/step at 128 GPUs, recipes PR #44): about 629 H100-h, $1.1–1.5k at $1.79–2.40, for a 10% schedule. The full schedule is about 6,290 H100-h, $11–15k, and collaborator-tier.
  - **Why the pilot falls short:** it never reaches the recipe's on-policy phase, which starts at step 75,000 of 100,000 in the reproduced config.
  - **Race risk:** a full-scale run would duplicate NVIDIA's own student, Edge, which is announced.
  - **Licence:** the student's VLM, Cosmos-Reason2-2B, is under the NVIDIA Open Model License.
  - Decide only if KC-E fails and the deployment-lever path is judged insufficient.
- P-018 · *(proposed 2026-10-27, D-015)* **Few-step distillation of the flow expert,** only if S5's ladder shows a cliff above the K that S6's latency budget allows.
  - **Method:** an expert-only shortcut-model or consistency fine-tune on one node. MeanFlow-style training from scratch is out.
  - **Cost:** tens of H100-h ($20–150) plus 2–3 sweeps ($68–395).
  - **Race check:** Edge may already be few-step.
- P-019 · *(proposed 2026-10-27, D-015)* **One C++ driver for two targets.**
  - **What:** S6a's runtime (TensorRT C++ engines or the TensorRT-Edge-LLM runtime, plus the C++ `EgodriverService`) is built first on a rented RTX PRO 6000. Edge-LLM lists x86 SM120 as a development platform. It then serves AlpaSim's student sweeps through `driver_source=external_static`.
  - **Acceptance:** P-009's ladder against the PyTorch driver, plus a paired closed-loop verdict. A cloud FP8 verdict never stands in for the Thor-engine verdict.
  - **Cost:** one equivalence sweep ($34–132) plus a few GPU-hours.
  - **Needs:** P-007.
- P-020 · *(proposed 2026-10-29, capstone selection)* **Capstone #1: the simulator as a measured deployment instrument (re-cuts S7's injection and S5(c); the floor is P-022, the upstream change is P-021).**
  - **What.** Three layers. (1) The floor, P-022. (2) The instrument layer: E0 CPU known-answer tests of `DelayBuffer` (⌈L/Δt⌉; warm-up GT ticks); a week-1 AlpaSim issue that also asks why the 200 ms default was dropped on 24.03.26 and which published numbers used it; E1 on 20 scenes × 3 replicates × {0, 1, 2, 3, 10} ticks, the 10-tick arm being the positive control. (3) The conditional headline: a power pre-check on S2's measured σ_d and ρ against the smallest loss worth reporting ____ (★); E2 on dynamic-actor strata sampled from the 2,202-scene curated pool minus the contamination exclusions (stationary or slow lead; crossing or merging replayed actor; a third ★ stratum), 60 scenes each × 3 replicates × {1, 2} ticks, the per-stratum MDE printed beside every Δ; E3t at the n the pre-check requires (163 scenes per stratum for Δ = 5 pp at an assumed σ_d of 0.3, m = 3 [F13]); E3w on 441 × 3 × {1, 2, 3} as the optional whole-set step function; E4 plan FIFO vs command FIFO at d = 2 and E5 sub-tick arms only with P-021. S5(c) becomes score(n | d) at a fixed tick band on one checkpoint at full prefill; the history-frame-reuse prerequisite is dropped (reuse is a separate model-change arm, KC-L2). The x-axis is applied staleness in ticks; the device point comes from P-023 or is labelled "not measured on Thor".
  - **Why.** AlpaSim applies a delayed plan once per control tick, so 139/171/200 ms are all 200 ms of staleness [AS-2, AS-3]; the Alpamayo driver is not queried during the 1.7 s warm-up and the buffer holds GT [AS-1]; the MPC tracks from 1.0 s ahead, so staleness acts through inputs [AS-8]; every run is lockstep by code [AS-4]; the per-stratum MDE on 9–30-scene strata is 8–36 pp, so the 441 split cannot power the strata where the mechanism predicts the loss [F13]; a null without a positive control is uninterpretable; the published curves are CARLA, command-hold or command-FIFO, without CIs [N-1, C-30]; KC-5 (01-11) precedes KC-6 (01-18), so S5(c)'s latency axis could never have come from S6 in time [E-§7.1].
  - **Carries over.** S7's constant arms, lockstep semantics and TOST (★); S5(b)'s K set and noise pairing; S5(c)'s natural-length-first rule, budget forcing and the n = 0 architecture label; S3's teacher-lever fallback.
  - **Gates.** P-005 before any run; KC-L before any public closed-loop number, the E1 positive-control result included, since it is scored on NuRec scenes (code, CPU known answers, scene-ID manifests and the latency axis on synthetic inputs are publishable now); KC-E for student arms only; interim looks by S1's group-sequential rule.
  - **Cost.** E1 $8–28; E2 $28–101; E3t $75–276 (E3w $102–373); E4, E5 $14–51 each (DR1) [F12]. Hours +8–10 (E0–E5) plus P-021.
  - **Predictions (★).** loss at 1 / 2 / 3 / 10 ticks ____ / ____ / ____ / ____; per stratum ____; the smallest loss worth reporting ____; margins ____; the warm-up rule ____; plan minus command FIFO at d = 2 ____.
  - **Evidence.** `docs/research/2026-10-28-capstone-selection/`: `report.md` (§0–§10); Tracks A–E, the analysis and the red team in `notes/`; corrections made at filing in `verification.md`. Bracketed IDs in P-020…P-029 resolve in `report.md` §10.
- P-021 · *(proposed 2026-10-29, capstone selection)* **AlpaSim upstream: a sub-tick / trace-driven `DelayBuffer` mode and a command-FIFO option (a named deliverable of P-020).**
  - **What.** (a) A per-step delay source (a trace or a distribution) applied inside the controller service so that L in (100, 200] ms resolves below one control tick without changing the policy's query rate or the camera clocks; (b) a command-FIFO option, ã_t = a_{t−τ}: the same `DelayBuffer` logic applied to the MPC's output commands (Bench2Drive-Robust's main mode), so the two interfaces compare at matched applied staleness; command-hold (repeat the last command, refresh every d ticks: 2601.07393's and B2D-Robust's dynamic mode) only as an optional third arm, because its mean staleness is (d − 1)/2 ticks. Filed as an issue in week 1 with the CPU known answers; as a PR after re-checking NVIDIA's post-10-31 release and only if Huy's rule says the constant arms are insufficient (`02` S7).
  - **Why.** Finer resolution is not a config change: policy and controller share `dt`, camera clocks are separate, `assert_zero_decision_delay` fails at 50 ms steps [E-§3.2]; the deployment question (171 vs 100 ms) is sub-tick; this is the capstone brief's MISSION (a), an accepted upstream contribution, on AlpaSim's driver path; AlpaSim accepts outside PRs [B-C08]; it is not #208's field [ver27 V21]; no measurement compares the two interfaces on one simulator [C-31].
  - **Carries over.** S7's "trace-driven delay as an AlpaSim PR" text; nothing is forked.
  - **Gates.** Re-check the next AlpaSim release; never compete with an offered PR; KC-L for any score shown in the PR (synthetic scenes or scene-ID manifests).
  - **Cost.** +8–12 h; E4/E5 $14–51 each.
  - **Predictions (★).** plan-FIFO minus command-FIFO loss at d = 2 ____; loss difference between 139 and 200 ms applied ____.
- P-022 · *(proposed 2026-10-29, capstone selection)* **The floor: a bitwise-invariant batched Alpamayo evaluation driver (amends P-015).**
  - **What.** Port the batching that NVIDIA's AlpaGym already runs across overlapping rollouts (`InferenceEngine`, preset `max_batch_size: 4`, `force_determinism: true` [AG-1, AG-2]) into the AlpaSim evaluation driver's `predict_batch`, behind a flag, consuming alpamayo1.5 PR #26's per-row generators [A-Q3.3]. Two arms: (i) invariant: batch-invariant GEMM, RMSNorm and attention reduction order; accepted only if B = k is bitwise identical to the **reseeded** B = 1 driver on every compared tick (on `e2e_challenge` the stock driver ignores the seed, so the reference is the driver with `main`'s per-tick reseed ported, `02` §0.1 K1); (ii) non-invariant: the plain batch under P-009's ladder E0a–E1 plus one known-null paired closed-loop verdict. Report the invariance tax, GPU-hours per rollout vs B at natural and fixed CoC lengths, and the gate's flag rate under the null. The bitwise arm needs no verdict; S2's same-policy re-run is its negative control.
  - **Why.** The evaluation driver runs eager at B = 1 with a sequential `predict_batch` [L7 4.3–4.4]; the policy is ≈ 73–87.5 % of step time [L7 §1]; batching changes numerics and RNG streams; invariance costs 34 % to > 2× on LLM decode [C-36i, C-37, C-39], net 14–54 % per sweep [F7]; no such study exists for flow policies in closed loop [C-44]; the nearest neighbour is bitwise branching for world-action models without a batched comparison [N-2]; the claim is narrowed to bitwise batch invariance and its tax, and the work consumes NVIDIA's own engine (D5) rather than duplicating it.
  - **Carries over.** P-015's build, acceptance and the conditional static-KV sub-option; L7-4's cost lines; the saving still assumes S3–S5 all run (21–22 sweeps), which the ≤ $2.5k tier does not fund.
  - **Gates.** P-005 (one extra sweep); KC-L for the verdict's numbers; a comment on PR #26 instead of a duplicate PR; re-check NVIDIA's post-10-31 sync before the AlpaSim PR (its CHANGELOG moved batching out of the runtime on 13.03.25 [AS-5]).
  - **Cost.** B1 $4–13; B3 $34–124; +16–22 h. Saving on S3–S5 if all run: 99–258 GPU-h (invariant, worst) to 384–1,001 GPU-h (non-invariant, best) [F7].
  - **Predictions (★).** invariance tax ____×; per-sample speed-up at B = 8 / 16 ____ / ____; E2 verdict for the non-invariant arm ____; the gate's false-flag rate under the known null ____.
- P-023 · *(proposed 2026-10-29, capstone selection)* **Capstone #3: the Jetson AGX Thor measurement artifact and one C++ driver for two targets (re-cuts S6; merges P-008 and P-019; amends P-009 (iii)).**
  - **What.** S6's build list becomes: (1) a real engine with a per-stage timeline, labelled real or emulated, device and versions named; (2) one diagnosed bottleneck and one fix, a kernel only if the profile puts a stage above its bound by Huy's margin ____; (3) FP8 vs NVFP4 as policy rows with the ladder; (4) the protocol: per-stage p50/p99/p99.9, glass-to-action (P-028), J/tick, a 30-minute sustained run at the default 120 W and at MAXN pinned, P-009's contention arm, on synthetic or own inputs; (5) the C++ `EgodriverService` built first on x86 SM120 and then on Jetson (strongly typed TensorRT, `IPluginV3`, CI on 10.16 and 11.x, golden bitwise replay, sanitizers, p99.9 histograms). Dropped: camera-batched ViT, FP8 ViT, ViT cache, streaming reuse. A PR #45 configuration re-built on Jetson is a self-made baseline and is labelled so; FlashRT `fast` is the published Jetson baseline for the 10B. Device-in-loop verdicts run on a 100-scene subset only.
  - **Why.** Edge-LLM ships the vision-side items and its matrix rejects action-runner KV reuse [A-Q1.1, Q1.5]; on Jetson the teacher's expert is at 66–79 % of bound [A-Q4.6]; under serial decode the 2B's expert step is ≤ 1.7 ms, under Domino the headroom question is reopened [F3, F14]; no glass-to-action, sustained or J/tick figure exists for Alpamayo on either Thor [A-§4.1 #6–7]; a full sweep with the device in the loop costs 7.3 h at a 100 ms budget and 59.6 h at FlashRT's 811 ms [F5]; KC-6's 150 ms sits mid-band [E-§7.20]; the TRT 11 API changes are known [L7 1.5].
  - **Carries over.** P-008's four items; P-019's acceptance (ladder with injected reference `x_0`, paired verdict; a cloud FP8 verdict never stands in for Thor); S6's owner targets; the TRT-11-ready C++ and the C++ server specification (L7-7, L7-9); the proxy-transfer section (why a fix on 188 SMs may regress on 20 [D-F4.1, F4.3]).
  - **Gates.** P-002 for every device line (P-024); P-005; P-009 (i)(iii) with a recorded-clocks fallback; KC-L if NuRec frames feed any published latency; KC-6 read in tick bands.
  - **Cost.** x86 10–20 GPU-h ($10–29); optional B200 ISA check ($75–88); Jetson $5,499 + import ____; +12–20 h (deferrable, §5).
  - **Predictions (★).** p99 per stage, no-CoC and CoC arms ____; glass-to-action minus model time ____; 30-minute p99 at 120 W minus 20-request MAXN p99 ____; J/tick ____; expert % of bound ____; NVFP4 vs FP8 verdict ____.
- P-024 · *(proposed 2026-10-29, capstone selection)* **A trigger rule for P-002 (amends P-002; the decision is Huy's).**
  - **What.** Order the Jetson AGX Thor dev kit when KC-E yields a public 2B student whose licence permits evaluation (then #3 has a model and the curve a device point in the 1–2-tick regime). Decide in week 1 (11-02 → 11-08), right after KC-E: at the quoted 30–45-day lead time [D-F6.8] a week-1 order lands 12-02 → 12-23, before T3 (week 9). The last order date that still lands the device before T4 (week 10, from 01-04) is 11-20 at 45 days or 12-05 at 30 days; a later arrival collapses T4 and T5 to the x86 half plus a labelled latency model. If KC-E fails, defer and re-decide at Edge's release.
  - **Why.** No Thor is rentable [D-F6.7]; the dev kit is out of stock at one reseller [D-DR6]; for the 10B teacher the Jetson point is 9 ticks and FlashRT owns the behaviour-preserving reference [F4; B-R03]; S6's window starts 12-07.
  - **Carries over.** P-002's price and the Jetson-vs-DRIVE warning.
  - **Gates.** P-005 recorded; the hardware line is separate from the cloud tier (`02` §5).
  - **Cost.** $5,499 + import ____; lead time ____.
  - **Predictions (★).** none; a decision rule.
- P-025 · *(proposed 2026-10-29, capstone selection)* **P-007 re-cut as a sweep-cost diagnosis on sm_120, with P-010 merged as its H2 test (amends P-007 and P-010).**
  - **What.** E-K0 on the rented RTX PRO 6000 in week 1: `nsys` over the tick, `ncu` over one expert step if counters are exposed (else recorded clocks and CUDA events), graphs on and off, with P-010's GEMV/skinny-GEMM curve as the H2 discriminator. Its output decides whether a sweep-side fix exists and feeds P-022; it makes no Thor claim. For the student, the expert-headroom question is reopened only under Domino decode (F14) and is answered by T3's profile, not here.
  - **Why.** The "~20 % of bound" is the ledger's derivation on sm_120 [DEC P-007]; on Jetson the teacher's expert runs at 66–79 % of bound [A-Q4.6]; PR #36 already removed CPU launch overhead on the PyTorch path [L7 4.7]; any sm_120 fix is Amdahl-bounded at 11–25 % of a sweep [L7 4 Q4].
  - **Carries over.** E-K0's setup; the H1a/H1b/H2/H3 discriminators; P-010's known answer (56–90 KB in flight per CTA at 16 CTAs) [L7 §4 KA-7].
  - **Gates.** KC-1 passed; P-005.
  - **Cost.** 4–8 GPU-h ($4–12); 2–4 h.
  - **Predictions (★).** expert % of bound on sm_120 with graphs on ____; the top kernel class ____.
- P-026 · *(proposed 2026-10-29, capstone selection)* **P-018 stays conditional and outside the window.**
  - **What.** Re-decide at Edge's release and after E7 shows a cliff above the K that the tick band allows; no few-step training before 01-25.
  - **Why.** K = 5 with fixed noise is deployed [A-Q2.2]; Edge may be few-step [DEC P-018]; a one-step head cut LIBERO-Long to 26 % [C-28i]; weights are KC-L2 material; under serial decode K does not move a tick for the 2B [F3].
  - **Carries over.** P-018's method and cost lines. **Gates.** KC-L2; P-005. **Cost.** Deferred. **Predictions (★).** ____ (none until E7).
- P-027 · *(proposed 2026-10-29, capstone selection)* **S8 deferred beyond the window; P-006's checklist extended.**
  - **What.** No rig work before 01-25. P-006 adds: the Decree 356/2025 Art. 18 cross-border dossier (Form 01b, within 60 days of any transfer) and the Art. 19 processing dossier if identifiable footage is processed on foreign GPU hosts (Art. 17(1)(c)); on-device anonymization before upload as the design lever; the smart-vehicle definition and the modification question for external cameras; re-verified rig prices (the 9295 cameras and HSB stock are unverified; the deserializer PSU is backordered). S8's real-sensor glass-to-action moves to P-023/P-028.
  - **Why.** S8's window starts the day the 12 weeks end [E-§5]; P-006 and P-002 are unrecorded; the legal reading is an engineering reading, not legal advice [D-F8.3–F8.11]; shadow mode is open-loop and never validation (`01` C12).
  - **Carries over.** S8's rig design, the within-rig contrast and the trigger-classifier specification (★). **Gates.** P-006, P-002, KC-8. **Cost.** $0 in the window. **Predictions (★).** disagreement rate, dense mixed traffic vs arterials ____.
- P-028 · *(proposed 2026-10-29, capstone selection)* **P-009 (ii) glass-to-action split into a device term and a simulated term (amends P-009; keeps (i), (iii), (iv)).**
  - **What.** Device term: oldest-frame exposure → command available, measured on the device by the C++ server's host monotonic clock and CUDA events (the frame-age convention is ★: with a 0.4 s, 4-frame history the "oldest frame" adds ≈ 300 ms by definition [E-§7.14]). Simulated term: ⌈L/Δt⌉ ticks in AlpaSim, with L either model time or the device term, as two arms. (iii) keeps pinned clocks with a recorded-clocks fallback on rented hosts; (i) and (iv) stand as adopted by D-015.
  - **Why.** `planner_delay_us` covers capture → planner output → controller and is applied in whole ticks [AS-2]; actuation lives in the vehicle model, not in the delay [E-§7.14]. **Carries over.** P-009's ladder as the gate's protocol for every driver change. **Gates.** P-002 for the device term. **Cost.** 0 beyond P-023. **Predictions (★).** device term minus model time ____.
- P-029 · *(proposed 2026-10-29, capstone selection)* **Cut order and hours for the capstone window (amends §0.1's cut order for 11-02 → 01-25 only).**
  - **What.** The capstone adds +44–64 h; the net slack is −104 to +10 h; at 25 h/week the plan is 50–104 h over, at 30 h/week from 44 h over to 10 h under [F9]. The [O] items are outside the [C] total and free nothing. If hours run short: S8 (already outside the window); then the [O] items; then, before the "S3 and S6–S7, integration and C++" list that the literal reverse order would cut first, these [C] role items in order, with their lane-note hours: L2-1 S4 scaling model and multi-configuration sweep (12–18 h), L6-6 S5 scaling and resiliency ladder (10–14 h), L5-5 manifest-driven pipeline metrics (8–12 h), L6-3 DCGM and goodput dashboard (8–11 h), L6-2 throughput measurement (6–8 h), and #3's x86 half (12–20 h), a pool of 56–83 h. At 25 h/week the window also extends by up to two weeks, or one of #1's layers waits. The corrections and the S3/S4 verdicts are never cut.
  - **Why.** `02` §3 prices 370–405 h at 12–16 weeks; the literal cut order removes the floor and #3 first [E-§5]; P-011 (20–30 h inside 10-26 → 11-04) may slip KC-1 [DEC P-011].
  - **Carries over.** §0.1's three tiers and "corrections are never cut". **Gates.** KC-1's re-forecast from S1's actual hours (P-003). **Cost.** 0. **Predictions (★).** none.

## S1 autonomous execution

- **D-016 · 2026-10-31 · Task-specific autonomous S1 execution.** Huy instructed the agent to handle all of S1 autonomously, including reasoning, specification engineering, implementation and tests, and report when finished. For this task this supersedes the earlier Huy-first restriction on S1 cores, predictions, tolerances and diagnoses. Work is agent-authored, not Huy's mastery evidence. The protected manifest and hook configuration remain untouched; no general permission to modify them or future-stage cores is inferred. No commit, public message, dataset download, hardware purchase or rented run is authorized.
  - Engineering choice made under that instruction: `docs/S1_IMPLEMENTATION.md` defines a conservative finite-sample road-level authorization bound, with separately controlled Holm regression flags and finite-look alpha spending. This explicitly amends the S1 decision-method requirement: Tango and road bootstrap are implemented as reference/diagnostic methods, canonical Gaussian sequential boundaries as planning tables. They do not authorize a release on arbitrary cluster data. Spent-alpha stopping bounds replace stage-wise-ordering intervals for this v0. The estimand is equal-road, then equal-scene and equal-replicate within road; it is not silently presented as a scene-weighted mean.
  - The agent's preregistration is `experiments/2026-10-31-s1-autonomous.md`; Huy's original blank worksheet is preserved. Exact probability checks, known answers, independent code review and preregistered Monte Carlo are separate evidence layers; none proves unconditional correctness or real-world validity.
  - S1 engineering qualification is distinct from human mastery, publication and S2 admission. KC-1's statistical requirement still binds. P-005, S2 pre-registration and all dataset/license gates remain unchanged. Later proposals remain unadopted. S2 needs a measured road/scene sampling design compatible with the proved assumptions.

## S2 preparation and research allocation

- **D-017 · 2026-10-31 · Autonomous free S2 preparation.** Huy requested detailed engineering specifications, an implementation plan, everything feasible without spending, and the exact decisions needed before GPU rental. This authorizes current-stage documentation, public-source research, mechanical admission/normalization plumbing, failing tests, CPU verification and budget recommendations. It does not extend D-016 to S2 scientific cores, predictions, tolerances or diagnoses, and does not authorize rentals, licenses, asset downloads, public messages or commits. The deliverables serve S2/G1, G2, G3·R1 and G7.
  - Implementation homes: `docs/S2_ENGINEERING_SPEC.md`, `docs/S2_IMPLEMENTATION_PLAN.md`; schema-v2 pilot/full drafts preserve pending owner choices. Synthetic approvals are test fixtures only. The validator checks logical consistency; it never authenticates approval or authorizes launch/publication. The adapter checks an inspected source schema and trusted sidecars; actual log extraction/physical validity remain unverified.
  - Source-audit correction: per-rollout JSON uses `score`; infrastructure failures also get zero; aggregated metrics may be truncated and missing offroad may become zero. Full trajectory/metric/schedule provenance is therefore required. Exact public pre-fix/fixed geometry helpers are tested on CPU; no full evaluator/runtime correctness is claimed.
  - Two registrations resolve the apparent circularity in the S2 card: register the five-scene Harmonizer off/on cost pilot first, then freeze its selected protocol before the full sweep. Pilot observations do not enter full inference and pilot policy scores cannot select scenes, strata or margins. Outcome-informed redesign must be marked exploratory with justified independent confirmation.
  - The proposed $100 pilot / $1,200 S2 / $2,500 initial project ceilings are nested recommendations, **not P-005 approval**. First compute is proposed at ≤8 GPU-hours and ≤$2.50/GPU-hour, all setup/idle included; other costs remain inside the all-in cap. Full-sweep spend requires a measured forecast and a separate release of remaining funds. A later $6,000 tier may be proposed for measured S4/S5 work, not assumed now.
  - Public-source evidence dated 2026-10-31 did not verify Edge weights/license. No KC-E pass or alternate model adoption is inferred. The existing fallback order remains conditional. No optional S2/future-stage experiment is adopted by preparation.
  - Source-archive plumbing now retains supporting JSON/docs and upstream fixture files needed by the expanded tests. Existing S1 evidence, its protected inference sources and method are unchanged. The CPU development lock adds Shapely 2.1.2 for helper verification; it is not a runtime AlpaSim lock.

- **P-030 · proposed 2026-10-31 · Focus Huy's time on AV model research; delegate evaluator operations.** `docs/RESEARCH_OPERATING_MODEL.md` recommends 70% hypothesis/model/training work, 20% ablation/interpretation and 10% critical measurement review after S2, with agent-owned reproducibility, manifests, orchestration, metadata/contamination audits, telemetry, test scaffolds and independent reviews. Recommended target after a specific ownership amendment: agents also implement evaluator reliability/validity/power cores under independent review and known-answer controls; Huy retains scientific definitions, predictions, margin/strata, interpretation and the RL/distillation loss/reward/estimator design. Until that amendment, current S2 ★ ownership remains binding. Train/dev/confirmation separation and checkpoint-selection multiplicity must be explicit. This is not a weekly-hours decision, new stage, paid platform adoption or general unlock of ★ cores. Any permanent ownership change needs Huy's recorded amendment, synchronized contract edits and Huy's own manifest unlock where applicable.


## Adopted evaluator ownership and budget ceilings

- **D-018 · 2026-10-31 · P-030 and P-005 adopted; concrete execution approval still pending.** Huy instructed: “Adopt P-030 and the $100 pilot / $1,200 S2 / $2,500 project caps. Take ownership of evaluator engineering, update the contracts, and prepare the exact GPU rental and license steps for my approval before spending.” This supersedes the unadopted ownership/cash-cap status in D-017 and P-030; their historical text/evidence is retained.
  - **Engineering ownership:** agents own evaluator implementation and technical design within the adopted scientific contract: statistics/reliability, validity instrumentation, power/reporting, admission/verifier logic, data lineage, orchestration, telemetry, reproducibility and engineering diagnosis. Independent review, known-answer controls and negative controls remain required. Huy retains scientific definitions, estimands, strata/validity/contamination rules, margins/tolerances, predictions, interpretation/claims and model-research loss/reward/estimator, latency-model and kernel design. Agents may propose scientific amendments but cannot silently change them or label agent-authored work Huy's mastery. Stage admission/scope restrictions remain.
  - **Guard boundary:** contracts are updated together; the protected manifest, hooks and settings remain unchanged. Existing `gate/verdict.py` and `gate/stats/` remain file-locked until Huy unlocks the relevant entries. No relocation or copy of protected logic may evade that restriction; no unlock is required merely to complete this contract/rental-preparation task.
  - **Recorded P-005 tier:** `up_to_2500`, currency USD, project cloud ceiling **$2,500**, S2 all-in ceiling **$1,200**, pilot all-in ceiling **$100**, nested rather than additive. Include paid setup/idle time, storage/retention, transfer, tax and ancillary charges. Hardware remains separately gated. These are spending ceilings, not commitments to spend or a claim all later training fits.
  - **Action approval remains pending by Huy's explicit instruction:** no rental, paid credit purchase/card charge, license acceptance or asset download now. Prepare the concrete provider/SKU/region/resources, cost/retention/stop plan and actual terms for approval. A prepaid credit purchase counts as cash outlay; usage paid from that credit must not be double-counted as a second payment. Track cash paid, resource charges accrued, unused credit and remaining liability separately.
  - **Still proposed, not attributed to Huy's approval:** the phase-specific ≤8 GPU-hours / ≤$2.50 per GPU-hour operational envelope and technical pilot defaults. The cash ceilings are recorded in both schema-v2 drafts; phase compute/rate approval fields, scientific registration/predictions, licenses, host checks and full-sweep release remain pending. Full S2 execution requires measured pilot forecasting and an explicit release of the remaining stage balance; $1,100 in the full draft is a conservative planning reservation after a maximum $100 pilot, not money already spent.
  - **Current work:** S2/G1, G2, G3·R1 and G7 contract and admission preparation; no scientific experiment is run. CPU engineering requirements may be implemented/tested without inventing a scientific prediction. P-002/P-003/P-004/P-006, KC-L/KC-L2, release/challenge checks and the 2026-11-02 S2 start are unchanged.

## Vast host qualification authorized

- **D-019 · 2026-11-01 · Bounded Vast.ai rental, engineering qualification and cleanup.** Huy instructed: “Find the best Vast.ai VM for S2, verify everything possible by renting and setup and test all conclusion above via vastai cli and cancle for saving money and make final real valiadation conclusion.” This authorizes selecting a suitable current Vast offer, renting it, setting up and testing its capabilities, and cancelling/deleting the task's rental to prevent further charges. Do not re-request the rental authority already granted. The nested D-018 cash ceilings remain.
  - **Scope today:** S2 [C] host/runtime/monitoring preparation for G1, G2, G3·R1 and G7. This is an engineering check using public software and synthetic inputs, with written known-answer acceptance criteria; it may proceed now under this task-specific instruction. It does not move the November 2 scientific-study admission, invent Huy's predictions, approve licensed-data transfer or accept licenses on Huy's behalf.
  - **Agent execution limits:** one VM at a time; first candidate one RTX PRO 6000 Blackwell Workstation 96 GB, on-demand, 160 GB instance disk, no extra volume or reserved commitment; compute rate at most $2.50/hour. Target completion within one hour; external stop deadline at 60 minutes from rental request; startup failure budget 20 minutes. This task's operational cash allowance is at most $10 within the adopted $100 pilot cap, including setup/idle, storage, transfer, tax and fees. These are conservative agent-imposed limits, not new Huy scientific choices or authorization to use the full $100. No automatic retry rental or extension if the host fails.
  - **Money/account:** use existing authorized account credit if adequate; do not change billing settings or purchase credit without a concrete funding decision if needed. Record charges separately from cash deposited. Never print API/private keys or copy the provider API key into the rented VM.
  - **Lifecycle:** prepare and test a local control-plane stop watchdog before creation, bind the exact new instance ID and unique label immediately, verify it is armed before bootstrap, and stop immediately if it cannot be armed. Export allowed evidence incrementally, verify local copies, then destroy only this task's VM/instance storage and verify absence. Existing account resources are outside scope. No commitments, additional volumes or provider template startup scripts from untrusted community templates.
  - **Claims:** a passing synthetic host test establishes only those capabilities on that delivered VM. No NuRec/AlpaSim full-pipeline pass, licensed-data host permission, scientific reliability, long-duration uptime or future-offer guarantee follows. Keep failures and unsupported metrics. The full pilot/full-study drafts remain unadmitted unless their independent conditions are satisfied.

- **D-020 · 2026-11-01 · Actual Vast host check completed; allocation closed.** Outcome recorded by the agent under D-019, not a new scientific protocol or budget release. Offer 56500949 / machine 140620 / host 395870 in South Korea delivered instance 58628722: KVM Ubuntu 22.04, root, one RTX PRO 6000 Blackwell Workstation 96 GB class GPU, driver 580.95.05, 15 visible vCPUs and 96.996 GiB guest RAM. Listing driver/CPU/RAM differed; future sizing must use the guest and actual pipeline measurements. The hourly quote for numeric 160 disk was $1.451111 compute+storage.
  - **Evidence:** public CUDA integer reference/repeat/corruption controls, approximately 70.82 GiB VRAM integrity, buildx/Compose networking/mount/PID/GPU checks, actual Nsight counters/CUDA trace, exact synthetic GEMM, responsive DCGM and loopback exporter scrapes. XID subscription succeeds, but DCGM reports NO_DATA and fault delivery is unverified. No project assets, licensed-data transfer, model experiment, scientific prediction or AlpaSim instrument qualification occurred. Normal public-software terms applied during authorized setup; no project-asset acceptance action was performed. See `docs/research/2026-11-01-vast-host-validation.md` and `evidence/s2-vast-host-20261101-v1/`.
  - **Real defect retained and fixed:** the early external watchdog stopped the VM, but retried because Vast's actual terminal state was `exited`, with explicit stopped intent and next state. The final predicate recognizes that structured state while rejecting queued starts; 26 watchdog tests and independent review cover the correction. Original runtime source/logs and failed DCGM-binding probes remain. The final watchdog correction was tested locally against the observed shape, not via another paid allocation.
  - **Lifecycle and charges:** 161 guest evidence files were exported and hash-verified before stop. Destruction was confirmed at 07:21:57 UTC; a later inventory had no instances or separate volumes. The post-delete task ledger recorded **$0.600 usage**, from existing credit; new cash paid/deposited **$0**. Amounts are provider-rounded snapshots, not an assurance against later settlement adjustments. No rental, retained disk, reservation, automatic retry or additional allocation remains authorized by this completed check.
  - **Next admission remains separate:** retain Vast KVM as an actually tested host route and Nebius as an untested fallback; do not declare an operator approved for licensed data, a future offer guaranteed, or full-pipeline memory/throughput measured. Project-asset terms/access, the November 2 scientific admission, Huy's pilot prediction/registration, actual model/renderer/Harmonizer compatibility and the later measured full-sweep release remain unchanged. No protected file, hook or setting was unlocked or edited; no commit or external message was made.

## Proposed focused RL research path

- **P-031 · proposed 2026-11-01 · Make G9 teacher RL the primary research path after S2.** In response to Huy's request to assess a dedicated RL focus, the agent recommends S1 → S2 → S4, with an explicit stop after a reproducible RL study. The detailed proposal is `docs/research/2026-11-01-rl-research-path-proposal.md`. This is not adoption of a new plan or authority to execute S4.
  - **Question:** when does expert-only closed-loop RL improve held-out collision outcomes while preserving progress/off-road performance, and when is reward improvement explained by stalling, proxy exploitation or simulator artifacts? Begin with one shipped reward arm and three independent training runs against the frozen teacher; add a one-variable reward/exploration/horizon contrast only after development evidence identifies a mechanism and measured cost funds the comparison. No numeric scientific prediction is supplied on Huy's behalf.
  - **Scope amendment proposed:** promote G9 above G5; defer S3/S5 and S6–S8 from the first focused study, broad S2 FP8/TransFuser comparative sweeps beyond required integration/reliability checks, and S4's one-to-eight-GPU scaling study. Existing requirements and gates remain unchanged until explicitly amended; S8 is already optional. Instrumentation, validity, power, contamination, known-answer/negative controls, held-out evaluation and cost evidence are retained. S3/S4 are already reversible after S2, but this broader reprioritization is new.
  - **Verified implementation distinction:** AlpaGym `972d160eed0e23d388497851504a3a233fec5879` uses a frozen-VLM continuous expert path through recipes PR #18's pinned code. Its advertised two-GPU smoke uses `n_generation=2`; the one-candidate setting is the base default. The open-loop reasoning recipe at `2e907dab31e5353f42b2c8a2f79d7e39b6e0d42c` documents a different VLM/discrete-action path and five 80 GB GPUs. The completed single-GPU synthetic host check qualifies neither training configuration. AlpaJudge is initially a diagnostic; adding a constant score to a group cannot train the expert, and adapting its action inputs requires validation.
  - **Budget and ownership unchanged:** $100 within $1,200 within $2,500. At maximum S2 cost, $1,300 remains; a proposed $650 training / $400 independent evaluation / $250 ancillary-and-failure reservation is a conditional envelope, not a measured quote, funding release or promise that two arms fit. Retain the one-arm pilot or seek a separately approved expansion if the forecast fails. D-018 ownership, file protections, S2 admission, concrete rental/license approval, KC-L and KC-L2 remain. No future-stage code, protected edits, asset download, rental, publication or commit is authorized by this proposal.

## Focused RL research plan adopted after independent review

- **D-021 · 2026-11-01 · Plan v5 adopts the verified RL-first scope of P-031.** Huy requested multiple research agents to verify the direction against legitimate papers and official technical sources before changing the plan/context, then explicitly answered: “Update the plan after verification (Recommended).” Three independent lanes and the coordinating review produced `docs/research/2026-11-01-rl-direction-review/`; cross-review corrections were applied before adoption. This is authority for the planning/context revision, not a scientific registration or a paid run.
  - **Primary path:** S1 → S2 → S4 → research conclusion; G9 teacher RL is primary, with G1/G2/G3·R1/G7 as the instrument and evidence. The initial model is the released Alpamayo 1.5 continuous expert with visual/language weights frozen. The question asks when an audited update improves held-out outcomes and, when transfer fails, whether one demonstrated correction changes it. It does not presuppose a failure, gain, novelty or efficiency superiority.
  - **Explicit scope changes:** defer full S2 FP8/TransFuser comparative sweeps and policy-panel ranking, while retaining teacher/re-export controls, reliability, complete validity/lineage, operational power and cost measurement. Defer the mandatory S4 1→8-GPU scaling study, broad searches, S3/S5 and S6–S8 from the first focused conclusion; S8 was already optional. All previous stage cards remain in the exact pre-amendment v4.1 archive. The S2 five-scene pilot, existing 441-scene full preflight and minimum three simulator replicates remain until Huy explicitly registers a different scientific sampling design. Existing blocked drafts are not automatically valid under the new matrix.
  - **Research sequencing:** qualify load/export/version/likelihood and reward semantics, then one shipped arm with at least three independent training runs if its measured envelope fits. A one-variable second arm is conditional on development mechanism evidence and its own frozen scientific contract. Three runs are a minimum diagnostic floor, not a guarantee of a calibrated method-level interval. Training-procedure confirmation needs fresh independent training runs after selection or a reviewed selection-aware design; new roads alone do not remove selection on exploratory training runs. Price those runs in addition to pilots.
  - **Corrections affecting admission:** online Gym reward aggregation differs from processed evaluation; preserve timestamped raw metrics before the ordinary episode object loses dense fields. Staleness configuration alone is not a hard version fence; assert actual identity during parity and account for later asynchronous data. Preserve attempted, retried and dropped scenes before cleanup. A capped front/lateral reward is not currently a free configuration-only substitution. The geometric-mean likelihood surrogate differs from a joint ratio within a replay row. Source inspection establishes these implementation facts, not runtime behavior or learning.
  - **Preserved scientific choices:** D-016's bounded-road equal-road gate and file locks remain. No reward arm, new estimand, margin, stratum, numeric prediction, checkpoint-selection criterion or algorithm-level inference method is adopted silently. The historical 20% relative collision target remains a target. The legacy KC-4 “score varies by more than 0.1 after 3 training seeds” remains a stop/review trigger where the same score applies; Huy must define its spread statistic or explicitly replace it before training. Its automatic open-loop-GRPO fallback is removed because it changes the scientific and resource problem.
  - **Admission and money:** S2 scientific admission remains November 2 plus its protocol/prediction/rights/concrete action gates. After accepted S2, a separately approved and capped S4 engineering qualification can measure fit/cost; those measurements precede release of the replicated scientific study. No new rental, funding, asset download, license acceptance, public message, commit or future-stage implementation occurred or is authorized here. The $100/$1,200/$2,500 nested caps, D-018 ownership, KC-L/KC-L2 and hardware gates remain. The obsolete $4,000 trigger and “finish the stage before stopping” rule are superseded: stop before the approved cap/deadline, reserve cleanup/evaluation, and reconcile both cash outlay and resource liabilities.
  - **Context synchronization:** authoritative plan v5; entry point, README, system map, curriculum focus, technical corrections, role-map scope, research allocation and both agent contracts updated together. The broad role map is historical evidence of possible extensions, not a requirement to add device/cluster work back to this study. Agent-authored research, specifications and reviews are not Huy's personal mastery or NVIDIA employment evidence. Validation and remaining runtime gaps are recorded in the review's `verification.md`.

- **D-022 · 2026-11-01 · Plan v5 amended at S4 after a gap review; direction unchanged.** Huy asked for multiple research agents to verify the new RL direction against papers and official technical resources before updating the plan and context, without redoing finished work, and to change nothing if the plan was already correct. Two agents covered the literature absent from the v5 review, and the coordinating agent re-checked consequential claims against primary sources and pinned code. The record is `docs/research/2026-11-01-rl-direction-review/gap-review.md`; pre-amendment copies, pinned files and the toy script are in `evidence/rl-gap-review-20261101-v1/`.
  - **Change rule:** a finding qualifies only if it falsifies a stated fact, makes a `[C]` commitment infeasible or ill-posed, or names a well-established failure mechanism absent from the plan. Five qualified; other papers are recorded as precedent only.
  - **Corrections:** one trainer step is many optimizer steps, so strict parity is compared before the first one and update counts name their unit. `progress_safety` fixes the reward, not the learning run, and no shipped preset defines a two-GPU multi-scene run. With KL off, logged KL is zero by construction, so the reference for drift is the saved zero-update checkpoint, and retention is registered.
  - **Named mechanisms:** noise-ranked or magnitude-blind groups, because the stock advantage is close to a sign at group size two; and unanchored drift from the pretrained expert. Both join S4.5, KC-4, §8 and `protocol-design.md` §6 with diagnostics and E19–E21. Neither is measured in driving. No driving paper has measured collapse without an anchor, and RAD's RL-only arm did not collapse, though its initialization is unstated. The Alpamayo-R1 paper describes a KL term to a reference policy; the released Gym presets set it to zero.
  - **Preserved:** direction, S1 → S2 → S4, direct expert RL as first baseline, the shipped reward as baseline A, S2, budget caps, admissions, ownership and file locks. No learning rate, group size, anchor, estimator, tolerance or prediction was chosen; the new registration fields are blank for Huy. No training, rental, license acceptance, gated download or spending occurred, and this does not record a scientific registration.
