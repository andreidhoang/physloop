# S2 free software preparation record — 2026-10-31

Serves S2/G1, G2, G3·R1 and G7: stage-card manifest, AlpaSim per-rollout integration, pre-registration plumbing and cost planning. This is an agent-authored software preparation record, **not** a registered scientific experiment or Huy's prediction. No simulator, model, licensed dataset or GPU was run.

## Known-answer software requirements

Before implementation, synthetic contract tests specified exact outcomes: complete source-faithful records normalize; infrastructure `metrics.error` blocks admission even when null; missing/duplicate/contradictory records block; a zero-progress pass stays a zero continuous score; an unapproved draft stays blocked. These are deterministic input/output specifications, not empirical forecasts. Subsequent independent reviews added failing counterexamples for malformed identity, oversized numeric values, contradictory score provenance, road-spanning strata, missing predictions, missing full-sweep release and uncovered budget caps.

Source: the immutable public writer/scorer audit in `docs/research/2026-10-31-s2-upstream.md`. CPU environment: resolved `uv.lock`, extended with pinned Shapely 2.1.2 for exact public upstream geometry-helper tests. Cost: cloud $0. Protected files and S1 evidence are unchanged by this S2 work. Verification writes a fresh immutable evidence directory with code hashes and test logs; no evidence is edited by hand.

## Scientific work remains unregistered

Huy's pilot prediction (runtime, Harmonizer overhead, expected failure/stop mechanism): **pending — do not run**.

Huy's full-study predictions, margin, strata, reliability rule, stabilization and diagnosis: **pending — do not run**.

Templates `S2_PILOT_DRAFT.json` and `S2_FULL_DRAFT.json` deliberately preserve null values and pending approvals. The pilot is separate from the full protocol so its cost evidence can inform the Harmonizer decision without pretending that decision existed before the pilot. No fixture's fictional approval authorizes a rental.

## Result and next action

Reproduce software checks with `scripts/verify_s2.py`; inspect the generated `verification.json` and logs. The final output reports the actual test count and whether both drafts remained blocked. Review the concrete budget/protocol choices in `docs/S2_IMPLEMENTATION_PLAN.md`; then Huy can record the missing decisions. S2 hardware qualification and scientific execution remain future work under the registered protocol.
