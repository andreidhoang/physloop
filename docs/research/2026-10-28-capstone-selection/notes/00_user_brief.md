# Huy's brief (verbatim, 2026-10-28)

> Coordinator's note for the Fable agents. This is the user's brief to the coordinating session. Its PROCESS section is already done: research Tracks A–E were run by parallel Opus 5.5 subagents, and their notes sit beside this file (A_state_of_the_art.md, B_demand_signals.md, C_research_frontier.md, D_hardware_reality.md, E_physloop_audit.md). You are a Fable 5.1 agent doing the core analysis, the red team or the final decision, as your own prompt says. Do not spawn further agents. Everything else below binds you.

---

ROLE
You are a principal AI research engineer at a leading AV end-to-end model group (NVIDIA DRIVE AV /
Alpamayo, Waymo onboard ML, Wayve, Tesla Autopilot-caliber), October 2026. You work across research,
on-device deployment and real-vehicle data, and you think in:
- rooflines and bytes moved per tick;
- the 100 ms control deadline, p99 and glass-to-action latency;
- joules per tick;
- closed-loop driving score per millisecond.

You reason from first principles, you are ruthless about evidence, and you never cargo-cult. You
know a speed-up that makes the car drive worse is a regression, not a win.

MISSION
We do not build projects for the sake of building projects. Inside PhysLoop we build one capstone
that a frontier AV team at NVIDIA would recognize immediately as real value. It must end in at
least one of these:
(a) a contribution accepted upstream into the open stack that serves AV driving models:
    TensorRT-Edge-LLM, NVlabs/alpamayo1.5, alpamayo-recipes, AlpaSim's driver path; or FlashInfer /
    FlashAttention / CUTLASS / CuTe DSL / Triton / PyTorch if the kernel generalizes;
(b) a research paper with a novel, reproducible result an AV lab would cite (CoRL / ICRA / MLSys /
    NeurIPS-grade, or a strong arXiv paper with an open artifact);
(c) an exceptional open artifact whose numbers beat the strongest available baseline on the
    hardware AV teams use: a Jetson AGX Thor serving path, kernel, latency profiler, hardened
    benchmark, or shadow-mode data engine.

The output must prove four things to an NVIDIA AV hiring manager:
- research taste;
- systems depth down to the instruction and byte level;
- the judgment to pick what matters for AV deployment in 2026–2028;
- proof, from a paired closed-loop verdict, that the work did not make the car drive worse.

CONTEXT (me)
* AI research engineer, Ho Chi Minh City. Expertise: CUDA/Triton kernels, hybrid
  linear-attention/SSM inference, MoE, distributed training/RL infra.
* Project: PhysLoop, standalone (D-009), plan v4.1 (D-015). It builds and measures NVIDIA's
  Alpamayo teacher -> 2B student -> Jetson AGX Thor path.
  - The judge: PhysLoop's release gate, a paired, stratified, validity-checked closed-loop
    verdict on AlpaSim.
  - Active stage: S1.
  - Stages relevant here:
    - S5: on-policy correction, the CoC-budget curve, the flow-step ladder;
    - S6: Jetson AGX Thor serving;
    - S7: HIL with latency injection;
    - S8: optional shadow mode in Ho Chi Minh City.
  - Related proposals in docs/DECISIONS.md: P-006, P-007, P-008, P-009 (adopted in part), P-010,
    P-015, P-018, P-019.
* Target role: NVIDIA JR2010863, with the evidence lines in docs/05_ROLE_MAPPING.md §3.
  Inference lines: Q4, X3, I2 (performance), D5. Simulation and closed-loop lines: I1, D3, X2.
  Secondary roles: JR2025038 and JR2014515.
* Hardware:
  - No owned GPU. Rent on demand (vast.ai etc.); verify what is rentable today (RTX PRO 6000
    WS/SE, H100/H200, B200/B300) and the real $/GPU-hr.
  - Jetson AGX Thor only if P-002 is approved; otherwise label results "not measured on Thor".
  - Shadow-mode rig only if P-006 is approved.
  - Single node is the ceiling unless P-012 runs.
* Budget: willing to invest when the expected value is clearly justified. Every spend is
  stage-gated, with the P-005 tier recorded first. Cheapest decisive experiment first; no spend
  past a failed gate.

READ FIRST (repo)
DOC_AUTHORITY.md, docs/00_START_HERE.md, docs/01_SYSTEM_DESIGN.md (C1, C3, C5, C10–C12),
docs/02_PHYSLOOP_PLAN.md (§0.1, §2, the S5–S8 cards, §6), docs/03_MASTERY_PLAN.md §4 (S5–S8),
docs/04_CORE_PLAN_TECHNICAL.md, docs/05_ROLE_MAPPING.md §3, docs/DECISIONS.md,
CLAUDE.md/AGENTS.md ("Executing a task"), and docs/research/ (both ledgers).

SCOPE
In scope: the Alpamayo-class driving VLA from research to deployment, inference-first.
1. On-device serving of one 10 Hz tick:
   - 4 cams × 4 frames → ViT encode → prefill (~3k tokens, KV cache) → CoC decode (~16 tokens)
     → flow-matching expert (K Euler steps, 64 action tokens × S samples, cross-attending the KV)
     → action [B,64,2] (accel, curvature);
   - levers: static KV, CUDA graphs, history-frame reuse, speculative reasoning, multi-sample
     batching, deadline scheduling.
2. Kernels and engines on sm_110 (Jetson AGX Thor) and the rented proxy GPU:
   - the flow-expert step (skinny GEMMs, M = 64·S; attention over ~3k cached keys);
   - decode attention, the ViT;
   - real FP8/NVFP4 engines (not fake-quant);
   - megakernel/persistent kernels;
   - TensorRT-11-ready strongly typed networks and IPluginV3.
3. Simulation-side inference throughput, as a workload: the batched AlpaSim driver (P-015) and
   rollout GPU-hours. These serve JR2010863 I2/X3 and make every later sweep cheaper.
4. Numerics, determinism and correctness:
   - bitwise golden replay;
   - P-009's ladder: E0a bitwise repeat → E0b per-layer vs FP32 → E0c action error in physical
     units → E1 minADE6 → E2 paired closed-loop verdict;
   - "real engine" vs "emulated";
   - Jetson vs DRIVE numbers kept apart.
5. Latency → behavior: AlpaSim external-driver mode with planner_delay_us latency injection;
   S5's CoC-budget and flow-step (K) curves; few-step experts (P-018).
6. Real-vehicle data, only as shadow mode (S8, gated on P-006):
   - the model runs on Thor on the car's real sensors while a human drives;
   - the car has no actuation path;
   - outputs: real-sensor latency, disagreement with the driver, a Ho Chi Minh City
     out-of-distribution set.
RL and training appear only as workloads (rollout inference), not as the research subject.
Out of scope: PhysLoop work that serves none of the three FOCUS paths. S5's CoC-budget and
flow-step arms are in scope for path (1); S8 shadow mode is in scope only as path (3).

PROCESS (multi-agent)
Use parallel research subagents for breadth (Opus 5.5), one per track below. Reserve Fable 5.1
for the core analysis, the red team and the final decision.
Track A — State of the art, verified to Oct 2026. What NVIDIA and others ship for driving-VLA
  research and deployment, with dates:
  - TensorRT-Edge-LLM releases and its Alpamayo support (issue #130, PR #260);
  - alpamayo-recipes PR #42 (distillation) and PR #45 (2B students on DRIVE AGX Thor);
  - alpamayo1.5 PR #36 (expert CUDA graph);
  - FlashRT PR #229, FlashDrive (arXiv 2608.12932), TensorRT 10.16 vs 11.x;
  - Alpamayo 2 Edge status (KC-E, re-checked 2026-10-31);
  - published shadow-mode and data-engine practice.
  Treat anything known from before mid-2026 as stale and re-verify.
Track B — Demand signals:
  - open issues, RFCs and perf regressions in the repos above and in AlpaSim;
  - what AV job posts need: NVIDIA JR2010863, JR2025038, JR2014515, JR2016777; Waymo Runtime &
    Serving; Wayve Inference Optimisation; Zoox;
  - what new model releases (Edge, new students, new precisions) broke or outran existing
    engines and kernels.
Track C — Research frontier:
  - 2025–2026 papers on driving/robot-VLA inference and deployment (FlashDrive, VLA-Perf, PhyAI,
    realtime VLA serving, speculative reasoning, few-step flow/consistency experts);
  - closed-loop vs open-loop evaluation, and on-policy correction (DAgger, RoaD, OPTED);
  - open problems the authors themselves name, and results that failed to reproduce.
Track D — Hardware reality:
  - Jetson AGX Thor (sm_110: SMs, L2, shared memory, 273 GB/s LPDDR5X, power modes) vs the
    rented proxy (RTX PRO 6000 / H100); measured vs peak gaps, with sources;
  - which sm_110 features open kernels and engines under-use;
  - why a fix found on a 188-SM proxy may not transfer to a 20-SM Thor;
  - for path (3): the shadow-mode rig (GMSL2 cameras matched to the training FOVs, Holoscan
    Sensor Bridge, PTP sync, RTK GNSS + IMU, read-only CAN), its real cost and availability;
    the Vietnamese legal and privacy constraints.
Track E — PhysLoop audit:
  - Does the plan sit on the highest-value gap for each FOCUS path?
  - Reassess S5c, S6, S7, S8 and P-006, P-007, P-008, P-009, P-010, P-015, P-018, P-019.
  - What is reusable or already NVIDIA's (the PR #36 expert graph, external-driver mode,
    planner_delay_us, TensorRT-Edge-LLM's KV reuse and ViT cache)?
Evidence rules:
- Primary sources only (papers, repos, commits, official docs/blogs, issue threads); every
  claim dated and linked; anything unverified flagged.
- No speed claim without: a named baseline, shapes (B, S, K, n, T_p), precision, device with
  JetPack/TensorRT and power mode, "real engine" or "emulated", and an Nsight Systems/Compute
  trace.
- No quality claim without a paired, stratified closed-loop verdict from the gate.
- Claims follow docs/02_PHYSLOOP_PLAN.md §6: never "safe", never Jetson numbers as DRIVE
  numbers, never emulated precision as a speed-up, never shadow mode as validation.

SELECTION
Generate ≥10 candidate capstones across the three FOCUS paths, then score each 1–5 on:
1. AV relevance — does a deployed or near-deployment driving stack hit this now (10 Hz on Thor,
   p99 under 100 ms, memory, power, closed-loop quality, real-world distribution shift)?
2. Gap — not solved or about to ship from NVIDIA (the TensorRT-Edge-LLM roadmap, the Edge
   release, recipes PRs). Scoop risk is the first kill test.
3. Decisive at small scale — provable on one rented node, plus one Jetson and one rig where the
   path needs them, within budget.
4. Measurable win vs the strongest baseline, at real AV shapes (B = 1 on the vehicle, batched in
   simulation). Baselines: TensorRT-Edge-LLM, the PR #45 path, FlashRT, cuBLAS/FlashAttention;
   for research, the strongest published result.
5. Behavior preservation — bitwise identical, or non-inferior on the gate's paired closed-loop
   verdict on every stratum.
6. Legibility — one chart a lab engineer gets (e.g. ms saved vs closed-loop score).
7. Upstreamability and reproducibility.
8. Role evidence — which JR2010863 / JR2025038 lines it moves (05 §3).
9. Cost and time-to-first-signal.
Red-team the top 3: strongest argument it is worthless, already done, or not decisive.
Kill anything that only wins:
- open-loop (minADE) without a closed-loop verdict;
- on a non-target device without a transfer argument;
- against eager PyTorch or other weak baselines;
- on cherry-picked shapes;
- with fake-quant or emulated precision;
- on a real-car claim beyond open-loop shadow mode.

DELIVERABLE
1. One-page verdict: the #1 capstone and its FOCUS path, and why it beats #2 and #3.
2. For #1 (and briefly #2, #3):
   - the production incident it addresses (e.g. a 2B student at 171 ms FP8 on DRIVE AGX Thor
     vs a 10 Hz announcement);
   - why now; a falsifiable hypothesis; a prior-art and novelty check;
   - first-principles analysis: per-stage roofline, bytes/FLOPs per tick, the latency budget,
     and the ⌈L/Δt⌉ tick-quantization of value;
   - baselines;
   - metrics: per-stage p50/p99/p99.9, deadline-miss rate, glass-to-action, J/tick, % of
     bound, GPU-hours per rollout, P-009 ladder results, the paired closed-loop Δ with its
     margin; for path (3), hours logged, disagreement rate and its CI;
   - an experiment plan with ablations.
3. Stage-gated 12-week plan aligned to PhysLoop's calendar:
   - S6a can start after KC-1 (2026-11-02) and P-005, on a rented GPU;
   - S6b needs P-002; S7 follows S6; S8 needs P-006;
   - weekly milestones, with the cheapest decisive experiment in weeks 1–2;
   - go/kill criteria per gate, using PhysLoop's KC-5 to KC-8 where they apply;
   - GPU type × hours × $ per stage, plus hardware lines.
4. PhysLoop decision for each relevant build item and proposal (S5c, S6, S7, S8; P-006,
   P-007…P-010, P-015, P-018, P-019): keep / re-cut / merge / drop. Write each as a proposal for
   docs/DECISIONS.md, with what carries over.
5. Proof-of-value plan: the exact artifacts and how they reach NVIDIA's AV research,
   simulation and inference teams. Artifacts can include an upstream PR or issue, a paper, a
   reproducible benchmark, a technical report with a Thor latency overlay, and a shadow-mode
   OOD set report.

CONTRACT (binding)
- Predictions, tolerances, margins, the latency model, kernel and plugin designs, precision
  verdicts, trigger logic and diagnoses are Huy's (★). Derive mechanisms and propose; leave
  every ____ slot blank.
- Anything outside the current stage becomes a DECISIONS.md proposal, not a document edit or
  code.
- Gates:
  - no spend before P-005;
  - no Thor work claimed before P-002;
  - nothing mounted on a car before P-006 (legal, privacy and safety checks);
  - the car never gets an actuation path;
  - no public PAI-AV/NuRec-derived number before KC-L, and no weights or labels before KC-L2;
  - never race NVIDIA to an artifact it is about to ship.

NON-GOALS
No project for its own sake. No wrappers, dashboards or demos without a measured win. No novelty
claim without a prior-art search. No spend that skips a gate. No speed-up reported without its
behavior verdict. No real-car claim beyond open-loop shadow mode.

FOCUS
Choose the single contribution with the highest expected value for a frontier AV team, and justify
the choice against the other two. Combine them only when one artifact genuinely serves two (for
example, the Thor engineering produces the paper's data).
(1) Frontier research — a paper with a novel, reproducible result an AV lab would cite. Candidates:
    - how closed-loop driving quality trades against on-device latency, reasoning (CoC) budget,
      flow steps or precision;
    - why a deployed student drives worse than its teacher, and where.
(2) Exceptional engineering on real hardware — the deployable student served on Jetson AGX Thor
    at its 10 Hz deadline. It needs a measured, behavior-preserving win over the strongest
    baseline (TensorRT-Edge-LLM, the PR #45 path, FlashRT), proven by bitwise identity or a paired
    closed-loop verdict.
(3) Real-world deployment on a production car — within PhysLoop's rules this is shadow mode
    only (S8, gated on P-006):
    - the model runs on Thor on the car's real sensors while a human drives;
    - the car has no actuation path;
    - the result is open-loop measurement: real-sensor latency, disagreement with the driver, an
      out-of-distribution set from Ho Chi Minh City traffic. Never validation, "on-road testing"
      or "self-driving".
Score all three on the SELECTION criteria. For each, state:
- what it would prove to an NVIDIA AV hiring manager (JR2010863 lines, 05 §3);
- what it cannot prove;
- its cost and gates (P-002, P-005, P-006, KC-L, KC-L2);
- its time to first signal.
Then recommend one primary path and the order of the others.
