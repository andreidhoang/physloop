# Start here

## Reading order
On the first read, read all of it. Every session, re-read steps 1, 3, 5 and 6.

1. `../DOC_AUTHORITY.md` — 2 minutes. Which document wins.
2. `01_SYSTEM_DESIGN.md` — the system map, the three system laws, the twelve core concepts.
3. `02_PHYSLOOP_PLAN.md` — **the plan**: deliverables G1–G9, stages S1–S8, budget tiers, allowed claims, kill checks, risks, open decisions.
4. `03_MASTERY_PLAN.md` — the skill pillars, the curriculum for each stage, the mastery gates, skill coverage.
5. `DECISIONS.md` — what was decided and why. Newest entries are at the bottom.
6. `../journal/` — the last entry: what was tested, what happened, the next hypothesis. Created at the first after-build ritual.

Reference, when needed:
- `04_CORE_PLAN_TECHNICAL.md` — verified model, hardware and cost facts, plus the preserved 22-Sep predictions.
- `05_ROLE_MAPPING.md` — live NVIDIA postings.
- `research/2026-10-25-teacher-to-thor/` — the decision input, the review and the verification ledger behind plan v4.
- `research/2026-10-03-verification/` — the evidence and claim ledger behind plan v3, still valid where not superseded.

## The before-build ritual (do not skip)
1. Name the stage (S1–S8) and the deliverable (G1–G9) that today's work serves.
2. Do the stage's curriculum exercise in `03` §4 if you have not done it yet.
3. Open or create `../experiments/<date>-<slug>.md` from `TEMPLATE.md`. Write the hypothesis, the mechanism (derived, with numbers), the falsifiable prediction with a number, the scope, the kill criterion, and the metric-to-decision rule. The prediction comes after the mechanism is derived, never before.
4. Estimate the cost (GPU-hours × the current price; re-price at every stage start) and write it down.
5. Check the gates:
   - Does this spend money (P-005; hardware: P-002, P-006)?
   - Does it publish a number derived from PAI-AV or NuRec (KC-L, P-004)?
   - Does it publish weights or labels trained on such data (KC-L2)?
6. Only then: code.

## The after-build ritual
1. Reconcile prediction against measurement in the experiment file. Explain every miss causally, and keep negative results.
2. Write three sentences in `../journal/<date>.md`: what was tested, the result (with the number), and the next hypothesis.
3. If anything changed a decision, append it to `DECISIONS.md`.

## Where we are (update this block when the stage changes)
- **Status:** standalone and active (D-009). **Plan v4 adopted on 2026-10-25 (D-012):** the teacher → student → Thor loop, with the gate as its measuring instrument. No implementation exists yet: the folder holds plans, specifications, evidence and Claude context only.
- **Stage:** **S1**, the gate decision core on synthetic truth plus a power analysis. It costs $0, runs on CPU and needs no dataset. v3's D (2026-10-05) passed without S1 starting. **D = 2026-10-26, so KC-1 falls on 2026-11-02.** The day-by-day plan is in `02` §4.
- **Next output:** the public `physloop-gate` v0, its power tables (including what S3's and S4's gates can certify), and an AlpaSim issue proposing the paired-verdict layer.
- **Dated check:** **KC-E on 2026-10-31.** Are Alpamayo 2 Edge weights public, and under what license? This decides whether S3 or S4 runs first.
- **Open decisions:**
  - P-002, the Jetson AGX Thor purchase: blocks S6–S7.
  - P-003, weekly hours: re-forecast at KC-1.
  - P-004, the license request to NVIDIA (§3 numbers, §4.6 weights): blocks every public number from S2 on. Sending it early is cheap.
  - P-005, the cloud budget tier: needed before any S2 spend.
  - P-006, the shadow-mode go: blocks S8.
- **Do not:**
  - download PAI-AV or NuRec before S2 starts (license term §8);
  - start S2 runs before the challenge closes on 2026-10-31 (AlpaSim issue #224);
  - register for the 2026 challenge (it needs an organization email).
