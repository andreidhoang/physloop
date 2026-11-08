# S2 synthetic evidence ingestion — engineering acceptance

Agent-authored under D-017/D-018 and Huy's November 7 instruction to implement the
reviewed next step. Serves [C] S2.1/S2.3 and G1/G2/G3·R1/G7. This is a CPU software
qualification, not a scientific experiment, registration, or S2 admission.

## Deterministic acceptance

- The active schedule contains ten pilot jobs, or 2,646 full jobs at three
  replicates, with explicit scene/road/stratum/policy/protocol identities.
- Artifact checks bind the exact bytes parsed, reject changed checksums and
  ambiguous JSON, and do not follow artifact symlinks.
- A planned job survives failure before UUID creation. Every attempt and retry
  remains visible. Conflicting events and unresolved retry selection block.
- Source-shaped synthetic Parquet metrics plus an explicitly synthetic execution
  trace derive observation sidecars for the existing adapter. Missing offroad
  streams, invalid metric samples, truncated coverage, missing frames, nonfinite
  poses, corrupt artifacts and unexplained short horizons block admission.
- A full deviation of 4.2 m after an earlier summary cutoff remains 4.2 m and
  out-of-envelope. Exactly 4 m is also out-of-envelope.
- The current exact-horizon rule and protected inference implementation stay
  unchanged. Failed batches expose no admitted records.
- Hashes establish integrity, not origin. The synthetic trace format is not an
  AlpaSim ASL decoder; no test pass establishes actual runtime capture or physical
  validity. Real log integration and scientific definitions remain separate.

## Resources and limits

Local CPU only; zero GPU-hours and zero cloud spend. No license acceptance,
gated assets, rental, public message, protected-file edit or commit. Public source
inspection and development dependencies are permitted engineering preparation.
Independent review and the full CPU/lint suite precede qualification. The earlier
four-sweep cost arithmetic remains historical; an active matrix report names the
current two-arm scope without inventing measured throughput or prices.

## Evidence

The final generated verification bundle records source hashes, test/lint results,
synthetic fixture outputs, negative controls and remaining qualification limits.
It is append-only. Huy's scientific prediction slot is not populated by this work.

## Observed engineering results

Generated bundle: `evidence/s2-ingestion-20261107-v1/`. All **1,046 tests** in
the captured working tree passed, including concurrent pilot-preparation tests;
Ruff and whitespace checks passed. The dedicated synthetic control produced
score 0.8 and full deviation 4.2 m, while absent raw offroad data blocked the
whole batch. Regression tests also cover a genuine synthetic collision cutoff
followed by later deviation, exactly 4 m, failed launches, retries and corrupt
artifacts. All 25 bundle-file checksums verified, sources stayed unchanged during
checks, and protected S1 hashes match their qualification record.

Independent review found finite but impossible event/distance values were
accepted; failing regressions reproduced this before binary/nonnegative domain
checks fixed it. It also corrected the collision fixture to retain its computed
progress subscore while its overall failure score is zero. These are engineering
defects and fixture corrections, not scientific effect estimates.

Installed direct dependency versions match the updated lock, including PyArrow
25.0.1 from verified existing resolved metadata. A clean frozen rebuild remains
unverified because the host uv crashes before resolution; provenance and exact
limitations are recorded in `docs/S2_INGESTION.md`. Both scientific drafts remain
blocked (24 pilot and 37 full-study blockers), and runtime qualification remains
false. No model/GPU/asset run or cloud spending occurred. The source archive
preserves the acceptance document before this post-run result appendix.
