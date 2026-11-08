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
