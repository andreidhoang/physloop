# PhysLoop — agent contract

You work with Dang Huy on PhysLoop, a standalone project. It builds and measures NVIDIA's Alpamayo teacher → Thor-student path, in two parts:
- **the gate**: an open, paired, stratified, validity-checked **release gate** on AlpaSim. It is the measuring instrument for everything else;
- **PhysLoop-AV**, the headline application, in six steps:
  - a closed-loop verdict on NVIDIA's own 2B student (Alpamayo 2 Edge) against its 10B teacher;
  - closed-loop RL of the teacher;
  - on-policy correction of the student (nanoAlpamayo) and its runtime-reasoning budget curve;
  - serving on Jetson AGX Thor;
  - hardware-in-the-loop;
  - optional shadow mode in Ho Chi Minh City.

PhysLoop is independent of every other project: it takes no hours, budget, deadline or decision from another program (DECISIONS D-009).

**Current state (2026-10-25).** Plan v4 is adopted (D-012), replacing v3 (D-010). The folder holds plans, specifications, verification evidence and Claude context, but **no implementation yet**. The active stage is S1, with D = 2026-10-26. It is in `docs/00_START_HERE.md` under "Where we are", along with the dated KC-E check on 2026-10-31.

## Before you touch code, every session
1. Read `DOC_AUTHORITY.md`, then `docs/00_START_HERE.md` in full.
2. Read `docs/02_PHYSLOOP_PLAN.md`: §2 (deliverables), the current stage card in §3, and §6 (claims).
3. Read `docs/DECISIONS.md` and the latest file in `journal/`, if there is one.
4. State in chat which stage (S1–S8) and deliverable (G1–G9) today's work serves, and the written prediction for the experiment. If no prediction exists, create the experiment file in `experiments/` and stop to ask Huy for it.

`/reread` performs steps 1–4. The command texts are in `context/claude-instructions/commands/`.

## How we work (non-negotiable)
- **Evaluation before model claims.** No result is reported without seeds, git hash, config, manifest, AlpaSim SHA(s), scene list and deviation convention.
- **One variable per experiment.** The prediction is written before the run, never after, and only once the mechanism has been derived.
- **A number that beats its prediction is investigated as a possible bug** before it is celebrated.
- **Negative results are kept,** together with the mechanism that explains the miss.
- **Reuse NVIDIA's open pieces** (AlpaSim, alpamayo-recipes including the distillation PR #42, AlpaGym, AlpaJudge, Drive-IRT). Build only what `02` §2 marks as open; never race NVIDIA to an artifact it is about to ship. Upstream contributions go to AlpaSim, which accepts outside PRs although none has merged since 2026-08-17. AlpaGym accepts none.
- **Allowed claims** are listed in `02` §6. Never claim:
  - "safe", per-mile rates, human parity, "above human", predictive validity or real-world validation;
  - compliance with any standard or regulation, including UN R[185] and the ADS GTR;
  - shadow mode as validation (it is open-loop);
  - Jetson AGX Thor numbers as DRIVE AGX Thor numbers;
  - emulated precision as a speed-up.
- **Statistics defaults:**
  - paired and cluster-adjusted by road;
  - one-sided bounds labelled as one-sided;
  - INSUFFICIENT blocks GO;
  - out-of-envelope (≥ 4 m) rollouts are reported separately, never silently scored or dropped.
- **Gates before action:**
  - no spend without a recorded budget tier (P-005);
  - no hardware purchase without its recorded decision (Jetson AGX Thor: P-002; shadow-mode rig: P-006);
  - no public number derived from PAI-AV or NuRec before KC-L passes (P-004);
  - no weights or labels trained on PAI-AV/NuRec-derived data published before KC-L2 passes (P-004, license §4.6);
  - no PAI-AV or NuRec download before S2 starts;
  - no S2 runs before the 2026 challenge closes on 2026-10-31;
  - nothing mounted on a car before P-006 is recorded. The car never gets an actuation path.
- **Scope creep is the top risk.** Anything outside the current stage goes to `docs/DECISIONS.md` as a proposal, not into code.
- **Ownership.** Huy writes the measurement instruments, the statistics, the RL and distillation/correction loss math, the latency model, the kernel designs, the verifier logic, the predictions, the tolerances and the diagnoses. The ★ cores for each stage are listed in `02` §3. Agents write plumbing, harnesses, scaffolds, failing tests, sweeps, plots and reviews. Agents never fill a prediction slot.
- **Starred cores are tool-enforced.** `.claude/starred-cores.txt` lists the files only Huy writes. Agents may read, test, explain and review them, never write them.
  - In Claude Code, `.claude/hooks/guard_starred.py` enforces this:
    - Edit, Write, MultiEdit and NotebookEdit are denied;
    - Bash commands that write those paths are denied, and any change a Bash command makes anyway is restored, with the displaced version kept in `.claude/starred-guard-backups/`.
  - The manifest, the hooks and the project settings protect themselves. Only Huy unlocks an entry, by commenting it out in his editor.
  - `AGENTS.md` mirrors this contract for Codex. Change both files in the same edit.

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
| `.claude/` | `starred-cores.txt` (the ★ manifest), `hooks/guard_starred.py` (the guard), `settings.json` (hook wiring). Protected from agents |
| `AGENTS.md` | This contract for Codex; kept identical to this file apart from its tooling notes |

## Environment facts (verified 2026-10-03; Thor re-verified 2026-10-25)
- **Compute.** No local GPU. Compute is rented: vast.ai, RunPod, Nebius, Lambda. The default workhorse is an RTX PRO 6000 (record Workstation vs Server edition). Every run has a cost and a kill criterion, and is re-priced at every stage start.
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
