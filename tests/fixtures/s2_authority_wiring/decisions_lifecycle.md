# Synthetic decision log for the G1 authority-wiring tests

Test fixture only; it is not `docs/DECISIONS.md` and records no real decision. Each hash is
`gate.s2_identity.identity_hash` of `tests.test_s2_execution.record(scope="actual")` with the
named `approval.decision_id` (asserted literally in `tests/test_s2_lifecycle_cli.py`).

## Log

- **D-019 · 2026-11-01 · Synthetic copy of a closed rental authority.** Closed statically (D-020).
  - authorizes: s2_action_record sha256:9950b2edd06a18ad9c726d1a3b2cb112067ad5e4ee9ff058fcb8052785dc96a3
- **D-024 · 2026-11-08 · Synthetic decision that authorizes no rental.** It has no authorizes line.
- **D-030 · 2026-11-12 · Synthetic rental approval.** Binds exactly the D-030 test record.
  - authorizes: s2_action_record sha256:e59b6f706e36c63e715f06b73da37af2218b0c4d9ff68ab8fb3871b07b357a89
- **D-031 · 2026-11-12 · Synthetic approval naming another record.** Binds the D-030 record, not its own.
  - authorizes: s2_action_record sha256:e59b6f706e36c63e715f06b73da37af2218b0c4d9ff68ab8fb3871b07b357a89
