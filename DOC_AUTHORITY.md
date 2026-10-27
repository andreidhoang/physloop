# Document authority

One table decides which document wins when two disagree. Each topic has exactly one live home. Historical documents live only under `docs/archive/`.

| Path | Status | Owns | Wins over | Last verified |
|---|---|---|---|---|
| `docs/02_PHYSLOOP_PLAN.md` | **authoritative** (plan v4, D-012) | What we build, in what order, budget tiers, allowed claims, kill checks, risks, open decisions | everything | 2026-10-25 |
| `docs/01_SYSTEM_DESIGN.md` | authoritative (v3) | System map and core concepts C1–C12 | 03, 04, 05 | 2026-10-25 |
| `docs/03_MASTERY_PLAN.md` | authoritative (v3) | Skill pillars, per-stage curriculum, mastery gates, skill coverage. No schedule | 04, 05 | 2026-10-25 |
| `docs/04_CORE_PLAN_TECHNICAL.md` | active reference (v2.1) | Verified model, hardware, cost and tooling facts; preserved 22-Sep predictions. No sequencing | — | 2026-10-25 |
| `docs/05_ROLE_MAPPING.md` | active reference (v2.1) | Live NVIDIA postings and the stage → posting map | — | 2026-10-25 |
| `nano_alpamayo/README.md` | active (re-based 2026-10-25) | The student's acceptance criteria A1–A6 and order | — (02 wins) | 2026-10-25 |
| `CLAUDE.md` + `AGENTS.md` | agent contract (Claude Code, Codex); identical apart from tooling notes, changed together | How agents work here: reading ritual, gates, ownership, the starred-core rule | — (02 wins on plan content) | 2026-10-25 |
| `.claude/starred-cores.txt` | the starred-core manifest; only Huy edits it | Which files only Huy writes; enforced by `.claude/hooks/guard_starred.py` in Claude Code | — | 2026-10-25 |
| `docs/DECISIONS.md` | append-only decision log | A decision here overrides older text in any doc until that doc is revised | all docs, until revised | rolling |
| `docs/research/2026-10-25-teacher-to-thor/` | evidence record | The owner's decision input, the review, the verification ledger and the arithmetic check behind plan v4 | — (live docs win where they differ) | 2026-10-25 |
| `docs/research/2026-10-03-verification/` | evidence record | The claim ledger (`report.md` §10) and seven research notes behind plan v3; still valid where the 2026-10-25 ledger does not supersede it | — (live docs win where they differ) | 2026-10-03 |
| `docs/archive/*` | historical; never act on these | 01 v1–v2, 02 v2–v3, 03 v1.1–v2, 04 v1–v2, 05 v1–v2, nano README v2, the rebuild `START HERE` | none | — |

Rule: if two live documents conflict, record the conflict in `docs/DECISIONS.md` and stop. Do not pick one silently.

Recovery: the full pre-rebuild context backup is outside this folder, at `/Users/danghuyhoang/Documents/Codex/2026-10-01/can/outputs/project-recovery-backup-20261001-020531`. It is not the implementation of this project.
