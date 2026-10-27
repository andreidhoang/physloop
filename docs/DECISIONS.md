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
