# Synthetic decision log template for the G1 bootstrap, launch and watchdog tests

Test fixture only; it is not `docs/DECISIONS.md` and records no real decision. The plan and
launch hashes embed producer source hashes, so tests render this template into a temporary
file with `string.Template`: `$grants` becomes `authorizes:` lines for the artifact under test,
`$wrong_grants` the same kinds bound to a hash of 64 zeros, `$records` the `records:` lines for
its gated `terms:`/`acquire:` keys.

## Log

- **D-019 · 2026-11-01 · Synthetic copy of a closed authority.** Closed statically (D-020).
$grants
$records
- **D-024 · 2026-11-08 · Synthetic decision that authorizes no artifact.** Records only.
$records
- **D-032 · 2026-11-12 · Synthetic approval of exactly this artifact.** Grants and records.
$grants
$records
- **D-033 · 2026-11-12 · Synthetic approval of another artifact hash.** Records present.
$wrong_grants
$records
- **D-034 · 2026-11-12 · Synthetic approval without its terms and acquisition records.**
$grants
