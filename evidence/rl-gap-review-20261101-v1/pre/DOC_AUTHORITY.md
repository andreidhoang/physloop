# Document authority

One table decides which document wins when two disagree. Each topic has exactly one live home. Historical documents live only under `docs/archive/`.

| Path | Status | Owns | Wins over | Last verified |
|---|---|---|---|---|
| `docs/02_PHYSLOOP_PLAN.md` | **authoritative** (plan v5, D-021; earlier versions archived) | What we build, in what order, budget tiers, allowed claims, kill checks, risks, open decisions | everything | 2026-11-01 |
| `docs/01_SYSTEM_DESIGN.md` | authoritative (v3.2) | System map and core concepts C1–C13 | 03, 04, 05 | 2026-11-01 |
| `docs/03_MASTERY_PLAN.md` | authoritative (v3.2) | Skill pillars P1–P8, per-stage curriculum, mastery gates, skill coverage. No schedule | 04, 05 | 2026-11-01 |
| `docs/04_CORE_PLAN_TECHNICAL.md` | active reference (v2.3) | Verified model, hardware, cost and tooling facts; preserved 22-Sep predictions. No sequencing | — | 2026-11-01 |
| `docs/05_ROLE_MAPPING.md` | active reference (v3.1) | October 27 posting snapshot and prior role map; November 1 focused scope note, not a fresh hiring-status check | — | 2026-10-27 |
| `nano_alpamayo/README.md` | deferred reference (re-based 2026-10-25; A1 corrected and A7 added 2026-10-27) | The student's acceptance criteria A1–A7 and order | — (02 wins) | 2026-10-27 |
| `CLAUDE.md` + `AGENTS.md` | agent contract (Claude Code, Codex); identical apart from tooling notes, changed together | How agents work here: reading ritual, gates, ownership, the starred-core rule | — (02 wins on plan content; D-018 updates ownership, D-019/D-020 close the bounded host check and D-021 adopts the research focus) | 2026-11-01 |
| `.claude/starred-cores.txt` | the starred-core manifest; only Huy edits it | Which paths remain file-locked until Huy unlocks them, independently of D-018 evaluator ownership; enforced by `.claude/hooks/guard_starred.py` in Claude Code | — | 2026-10-25 |
| `docs/DECISIONS.md` | append-only decision log | A decision here overrides older text in any doc until that doc is revised | all docs, until revised | rolling |
| `docs/research/2026-11-01-rl-direction-review/` | source review and independent verification behind D-021 | 34 selected paper sources, official NVIDIA code audit, study-design recommendations and review resolutions; no runtime qualification | — (02 and decisions own scope; source dates/limits are explicit) | 2026-11-01 |
| `docs/research/2026-11-01-vast-host-validation.md` | actual host evidence under D-019/D-020 | Selected/delivered Vast resources, public synthetic checks, observed defects, recorded usage and verified cleanup; no scientific S2 admission | earlier untested-host/$0 status for this task only | 2026-11-01 |
| `docs/S1_IMPLEMENTATION.md`, `docs/S1_CORRECTNESS.md`, `docs/S1_HANDOFF.md` | S1 implementation contract, conditional proof and engineering qualification under D-016 | The explicit conservative v0 method amendment, its assumptions, evidence and S2 handoff; no authority over later-stage gates | older S1 method text as amended by D-016 | 2026-10-31 |
| `docs/S2_ENGINEERING_SPEC.md`, `docs/S2_IMPLEMENTATION_PLAN.md` | S2 preparation detail under D-017/D-018 | Offline interfaces/tests, remaining implementation, draft protocol and adopted cash ceilings; scientific pilot admission remains pending, while D-019 separately authorizes its bounded host check | — (02 and recorded decisions win) | 2026-11-01 |
| `docs/RESEARCH_OPERATING_MODEL.md` | P-030 adopted by D-018; focus amended by D-021 | Agent-owned evaluator engineering and Huy-owned scientific/model research decisions; file guards and stage gates remain | older evaluator ownership text, within D-018 scope | 2026-11-01 |
| `docs/S2_RENTAL_APPROVAL.md` | scientific pilot approval packet | Concrete rental/license sequence and outstanding owner actions under adopted ceilings; the packet alone does not authorize execution | — (D-018, D-019 and actual approvals win; D-019 rental authority was used and closed) | 2026-11-01 |
| `experiments/2026-11-01-vast-host-check.md` | engineering acceptance contract under D-019 | Public-software/synthetic VM checks and deterministic expected outcomes; evidence in `evidence/s2-vast-host-20261101-v1/` records only the delivered host and tested capabilities | — (D-019 owns rental scope/limits; no scientific pilot or licensed-data admission follows) | 2026-11-01 |
| `docs/research/2026-10-27-role-alignment/` | evidence record | Seven research lanes, the synthesis report, the verification ledger (with an independent check of 33 claims), the arithmetic check and the price snapshot behind plan v4.1 | — (live docs win where they differ) | 2026-10-27 |
| `docs/research/2026-10-25-teacher-to-thor/` | evidence record | The owner's decision input, the review, the verification ledger and the arithmetic check behind plan v4 | — (live docs win where they differ) | 2026-10-25 |
| `docs/research/2026-10-03-verification/` | evidence record | The claim ledger (`report.md` §10) and seven research notes behind plan v3; still valid where the 2026-10-25 ledger does not supersede it | — (live docs win where they differ) | 2026-10-03 |
| `docs/archive/*` | historical; never act on these | 01 v1–v2, 02 v2–v4.1, 03 v1.1–v2, 04 v1–v2, 05 v1–v2.1, nano README v2, the rebuild `START HERE` | none | — |

Rule: if two live documents conflict, record the conflict in `docs/DECISIONS.md` and stop. Do not pick one silently.

Recovery: the full pre-rebuild context backup is outside this folder, at `/Users/danghuyhoang/Documents/Codex/2026-10-01/can/outputs/project-recovery-backup-20261001-020531`. It is not the implementation of this project.
