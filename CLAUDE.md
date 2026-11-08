# PhysLoop — agent contract

You work with Dang Huy on PhysLoop, a standalone project researching **closed-loop post-training of driving action policies** on NVIDIA's open stack. It has two parts:
- **the gate:** an open, paired, stratified, validity-checked release gate on AlpaSim, serving as the measuring instrument;
- **PhysLoop-AV:** the focused S4/G9 study of Alpamayo 1.5 continuous-expert RL, with visual/language weights frozen, a reproducible baseline, one conditional mechanism-driven intervention and independent held-out evaluation.

The active path is **S1 → S2 → S4 → research conclusion**. Student comparison/correction (S3/S5), device serving/HIL (S6/S7) and optional shadow mode (S8) are deferred from the first focused study. Reasoning-text RL, Super, reactive traffic and broad scaling are separate future decisions.

PhysLoop is independent of every other project: it takes no hours, budget, deadline or decision from another program (DECISIONS D-009).

**Current state (2026-11-01).** Plan v5 is adopted under D-021 after independent literature/source review and Huy's explicit instruction to update the plan after verification. It incorporates P-031's RL-first focus with corrected reward, estimator, freshness, data-accounting and confirmation requirements. See `docs/research/2026-11-01-rl-direction-review/`; papers and source inspection do not establish a completed runtime or training result.

S1 is technically qualified and S2 free preparation is implemented; the scientific S2 pilot and full study remain unadmitted. D-018 adopts P-030 evaluator ownership and P-005 cash ceilings: $100 pilot within $1,200 S2 within $2,500 project cloud. D-019 authorized the bounded Vast host rental/test/cleanup, completed and closed under D-020 on 2026-11-01: synthetic host checks, $0.60 recorded usage from existing credit, instance and storage deleted. It is not authority for another allocation, account funding, project-asset license acceptance/downloads or the scientific pilot, and does not move its November 2 admission. Standard public-tool terms applied during the authorized setup. D remains 2026-10-26; current admission status is in `docs/00_START_HERE.md`.

## Before you touch code, every session
1. Read `DOC_AUTHORITY.md`, then `docs/00_START_HERE.md` in full.
2. Read `docs/02_PHYSLOOP_PLAN.md`: §2 (deliverables), the current stage card in §3, and §6 (claims).
3. Read `docs/DECISIONS.md` and the latest file in `journal/`, if there is one.
4. State in chat which stage (S1–S8) and deliverable (G1–G9) today's work serves. For a scientific experiment, name its written prediction; if missing, prepare the experiment file and ask Huy before dependent execution. Reversible documentation, engineering specifications and CPU known-answer tests may proceed without inventing a scientific prediction; record their scope and deterministic expected outcomes. Paid GPU engineering checks additionally require their recorded execution authority; D-019 supplied it for the now-completed bounded host check in `experiments/2026-11-01-vast-host-check.md`.

`/reread` performs steps 1–4. The command texts are in `context/claude-instructions/commands/`.

## How we work (non-negotiable)
- **Evaluation before model claims.** No result is reported without replicates and seed mode, git hash, config, manifest, AlpaSim SHA(s), scene list and deviation convention.
- **One variable per experiment.** The prediction is written before the run, never after, and only once the mechanism has been derived.
- **A number that beats its prediction is investigated as a possible bug** before it is celebrated.
- **Negative results are kept,** together with the mechanism that explains the miss.
- **Reuse NVIDIA's open pieces.** They include:
  - AlpaSim, with its Slurm run methods, external-driver mode and `planner_delay_us` delay;
  - alpamayo-recipes, including the distillation PR #42 and its one-node stage-2 trainer;
  - AlpaGym, with its presets and `perf` telemetry;
  - AlpaJudge and Drive-IRT;
  - the expert CUDA graph (alpamayo1.5 PR #36);
  - the measurement definitions in recipes PR #44.

  Build only what `02` §2 marks as open; never race NVIDIA to an artifact it is about to ship.
- **Upstream contributions** go to AlpaSim. It accepts outside PRs, although none has merged on `main` since 2026-08-17; four from non-collaborator accounts merged on `e2e_challenge` from 08-24 to 08-31. Re-check NVIDIA's latest release before any PR (#224), and never compete with an offered PR (#208). AlpaGym accepts none.
- **Allowed claims** are listed in `02` §6. Never claim:
  - "safe", per-mile rates, human parity, "above human", predictive validity or real-world validation;
  - compliance with any standard or regulation, including UN R[185] and the ADS GTR;
  - shadow mode as validation (it is open-loop);
  - Jetson AGX Thor numbers as DRIVE AGX Thor numbers;
  - emulated precision as a speed-up;
  - unseeded replicates as seeds;
  - single-node or rented-cluster results as cluster or production scale, or this solo project as industry infrastructure experience.
- **Statistics defaults:**
  - paired and cluster-adjusted by road;
  - one-sided bounds labelled as one-sided;
  - INSUFFICIENT blocks GO;
  - out-of-envelope (≥ 4 m) rollouts are reported separately, never silently scored or dropped;
  - repeats are unseeded replicates unless the manifest records a seeded mode. On `e2e_challenge`, the Alpamayo driver ignores the session seed;
  - interim looks follow the pre-registered group-sequential rule from S1, or none are taken.
- **Gates before action:**
  - P-005 is recorded under D-018: $100 pilot, $1,200 S2 and $2,500 project cloud, nested all-in cash ceilings. Concrete rental/account-funding/license actions need recorded approval; the ceilings, read-only preflight or public prices alone do not provide it. D-019 is the recorded rental authorization for the bounded host check, not the scientific pilot;
  - D-019's one public-software/synthetic Vast VM check is completed/closed under D-020. Its limits were $10 all-in within the $100 cap, compute at most $2.50/hour, a 20-minute startup failure limit and an external stop deadline at 60 minutes. Preserve its lifecycle requirements for any separately admitted rental: bind exact ID/unique label, verify evidence export, destroy only task resources and verify absence. No automatic retry rental, extension, new funding, project assets or scientific run follows from the completed check;
  - no hardware purchase without its recorded decision (Jetson AGX Thor: P-002; shadow-mode rig: P-006);
  - no public number derived from PAI-AV or NuRec before KC-L passes (P-004);
  - no weights or labels trained on PAI-AV/NuRec-derived data published before KC-L2 passes (P-004, license §4.6);
  - no PAI-AV or NuRec download before S2 starts;
  - no scientific S2 runs before the challenge closes on 2026-10-31 and the November 2 stage admission; D-019's engineering-only check is the explicit earlier exception;
  - nothing mounted on a car before P-006 is recorded. The car never gets an actuation path.
- **Scope creep is the top risk.** Anything outside the current stage goes to `docs/DECISIONS.md` as a proposal, not into code.
- **Lessons follow `docs/03_MASTERY_PLAN.md` §8–§9.** When Huy asks to learn, explain, derive, trace or dive into a topic, place it on the §9 map first. Then teach it in §8's order: goal → derivation from first principles → code → an ASCII trace of every tensor element, rendered by a script from the real code → tests → production view → Huy's part. The row's mode decides who writes code. Starred cores stay read-only, and S4 lessons read pinned upstream code until S4 is admitted. Lesson scripts live in git-ignored `study/` and use tiny synthetic inputs only. Agent-written code and traces are delivery, not Huy's mastery evidence.
- **Ownership (P-030 adopted, D-018).** Agents own evaluator engineering: statistical/reliability implementation, validity instrumentation, power analysis, admission/verifier implementation, data-lineage audits, orchestration, telemetry, reproducibility and engineering defect diagnosis. Evaluator cores require independent review and known-answer/negative-control tests. Huy retains scientific definitions and claims, estimands, strata/contamination/validity conventions, margins and tolerances, scientific predictions and interpretation, plus RL/distillation/correction loss and reward/estimator design, latency-model and kernel-design research. Agents may derive and recommend evaluator methods and technical defaults, but do not silently change a frozen protocol or fill Huy's scientific prediction slots. Agent-authored work is attributed as such, not Huy's mastery evidence. This ownership allocation applies when each stage is admitted; it does not authorize future-stage work early.
- **File protection is separate from engineering ownership.** P-030 does not unlock entries in `.claude/starred-cores.txt`, authorize edits to hooks/settings, or allow moving/duplicating protected logic to evade the guard. Existing listed paths remain read/test/review-only until Huy explicitly unlocks the relevant entry. New evaluator work outside those paths may proceed under D-018; model-training and other retained Huy cores are not reassigned.
- **Starred cores are tool-enforced.** `.claude/starred-cores.txt` lists the files only Huy writes. Agents may read, test, explain and review them, never write them.
  - In Claude Code, `.claude/hooks/guard_starred.py` enforces this:
    - Edit, Write, MultiEdit and NotebookEdit are denied;
    - Bash commands that write those paths are denied, and any change a Bash command makes anyway is restored, with the displaced version kept in `.claude/starred-guard-backups/`.
  - The manifest, the hooks and the project settings protect themselves. Only Huy unlocks an entry, by commenting it out in his editor.
  - `AGENTS.md` mirrors this contract for Codex. Change both files in the same edit.

## Executing a task
Run every task through these steps in order. Each step only applies the rules above; none adds a new one.
1. **Place it.** Name the stage (S1–S8), the deliverable (G1–G9) and the line of the stage card in `02` §3 that the task implements. If no line covers it, the task is outside the current stage: write a proposal in `docs/DECISIONS.md` and stop.
2. **Check ownership.** Read `.claude/starred-cores.txt` and the stage's ★ list in `02` §3 with the D-018 allocation. Evaluator implementation is agent-owned, subject to the unchanged file guard and independent verification. Scientific choices/predictions and retained model-research cores remain Huy's; propose changes for his record. Do not treat old ★ labels as a blanket ban on the evaluator work D-018 delegates, or delegation as an unlock of listed paths.
3. **Check the experiment.** Open `experiments/<date>-<slug>.md` for scientific runs; missing Huy prediction blocks dependent execution. Engineering preparation uses explicit input/output requirements or known answers and does not populate that slot.
4. **Check the gates.** Use the recorded action authority as well as P-005 ceilings: D-019 approved the now-completed bounded host rental/test/cleanup; it does not authorize another rental. Scientific-pilot rental, new funding and license actions remain separately gated. Do not re-request authorization already granted within an uncompleted recorded scope. Hardware needs P-002 or P-006. Dataset downloads, licence terms (KC-L, KC-L2) and dated checks apply as listed above. Estimate GPU-hours, storage, transfer, tax and other charges before any rented run; full S2 spend also needs the measured pilot forecast and release of the remaining cap.
5. **Build.** Write the failing test first. Where `03` §4 gives a known answer, that answer is the test. Then make the smallest change that passes. Change one variable per experiment, and pin the environment (resolved `uv` lockfile).
6. **Run and record.** Write outputs to `evidence/`, which is append-only and never edited by hand. Engineering-only checks retain their acceptance contract, source/config/image identities, delivered hardware, raw results, costs and cleanup evidence; they do not claim simulator qualification. Each scientific rollout record carries the manifest from `02` §3 S2:
   - git hash, config, replicates and seed mode (`force_determinism`);
   - AlpaSim branch and SHA(s), evaluator SHA, resolved lockfile, `n_sim_steps` requested and actual;
   - simulator configuration: Harmonizer state, renderer image, traffic mode, run method and topology;
   - driver settings: expert graph, batch size, replicas, and the resolved `alpamayo1_5` commit;
   - GPU edition, scene list, score definition, deviation convention.

   Every latency number also names its device, with JetPack and TensorRT versions, and says "real engine" or "emulated".
7. **Verify before reporting.**
   - A number that beats its prediction, or its roofline bound, is investigated as a bug first.
   - Time GPU work only after warm-up, with device synchronization.
   - Correctness needs more than one loose `allclose`: repeat bitwise and compare against a reference.
   - A speed-up that should keep numerics must be bitwise identical. One that changes numerics (batching, a lower precision) needs P-009's ladder plus a paired verdict before any verdict sweep uses it.
   - Report ≥ 4 m rollouts separately, and treat INSUFFICIENT as a result.
8. **Hand back.** Report what changed, reproduction commands, evidence and unverified assumptions. Agents own engineering defect analysis and prepare prediction-versus-measurement evidence; Huy owns the scientific reconciliation, causal interpretation and claims. Commit only when Huy asks.

## Repo layout
Directories are created by the stage that first needs them.

| Path | Contents |
|---|---|
| `docs/` | Authoritative plans: the reading order is in `docs/00_START_HERE.md` and precedence in `DOC_AUTHORITY.md` |
| `docs/research/` | Verification evidence and the claim ledger |
| `docs/archive/` | Superseded versions; never act on these |
| `experiments/` | One design doc per experiment (template: `experiments/TEMPLATE.md`) |
| `journal/` | One file per day, three sentences: tested, result, next hypothesis |
| `gate/` | S1–S3: the decision core, AlpaSim layer, validity records, reliability, student verdicts |
| `teacher_rl/` | S4: AlpaGym runs, reward variants, the reward-hacking audit |
| `nano_alpamayo/` | S5: the student's on-policy correction, flow-step ladder, CoC-budget arms |
| `thor/` | S6–S7: Jetson AGX Thor engines and kernels (C++/CUDA), the latency profiler, HIL with latency injection |
| `shadow/` | S8 (optional): rig configuration, logging, trigger classifier, OOD set |
| `evidence/` | Manifests, scored runs, plots; append-only, never edited by hand |
| `study/` | Lesson trace scripts and their printed output (`03` §8); git-ignored, tiny synthetic inputs only, never evidence |
| `.claude/` | `starred-cores.txt` (the ★ manifest), `hooks/guard_starred.py` (the guard), `settings.json` (hook wiring). Protected from agents |
| `AGENTS.md` | This contract for Codex; kept identical to this file apart from its tooling notes |

## Environment facts (verified 2026-10-03; Thor re-verified 2026-10-25)
- **Compute.** No local GPU. Compute is rented: vast.ai, RunPod, Nebius, Lambda. The default workhorse is an RTX PRO 6000 (record Workstation vs Server edition). Every run has a cost and a kill criterion, and is re-priced at every stage start. Record delivered guest RAM, GPU/VRAM and driver separately from offer claims; VM administrator access or a synthetic pass does not establish renderer compatibility, monitoring support or full-pipeline memory fit.
- **Python** via `uv`, pinned to the Alpamayo recipes: torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, ModelOpt 0.43.0.
- **Gated assets** (PAI-AV, NuRec, Cosmos-Reason2, OmniDreams weights) are approved automatically on Hugging Face once the license is accepted. Accepting a license is Huy's action. The PAI-AV and NuRec licenses:
  - allow internal use only (§1);
  - treat benchmarking data as confidential (§3);
  - forbid derivatives and distribution (§4.6);
  - expire 12 months after first download (§8).
- **Thor names two devices.** Always name which.
  - Jetson AGX Thor: 128 GB; JetPack 7.2.1, CUDA 13.2.2, TensorRT 10.16.2; no native GMSL.
  - DRIVE AGX Thor: 64 GB, ASIL-D design, DriveOS 7.2, TensorRT 11.x.

  They have the same silicon and 273 GB/s of bandwidth, but numbers do not transfer 1:1 between them.

## User context
Preferences shared across Huy's projects are preserved in `context/CLAUDE.user.md`.
