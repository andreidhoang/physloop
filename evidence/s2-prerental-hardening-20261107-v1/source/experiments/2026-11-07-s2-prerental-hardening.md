# S2 pre-rental CPU hardening

Agent-authored engineering work under D-017/D-018, 2026-11-07. Implements
S2.1 registration plumbing, S2.2 bounded pilot preparation, S2.3 exact attempt
identity and S2.7 telemetry; serves G1/G2/G3·R1/G7. No scientific prediction,
registration, admission, licensed download or paid execution is supplied here.

## Deterministic acceptance contract

1. An ASL scene/session identity reused by different attempts is ambiguous even
   when their external run UUIDs differ. Preserve both records and fail capture
   inspection; copying the log to another path must not bypass this check.
2. The exporter probe must reject an occupied host metrics port and an exited
   task exporter. An unrelated HTTP metrics server cannot establish that the
   task exporter works. Never stop an unrelated host service.
3. A pilot packet must reconcile one explicit five-scene schedule with its
   launch request, including identical horizon, one replicate and both
   Harmonizer states. Require an explicit permutation of all ten job IDs;
   never invent an execution order. An alternating declaration must alternate
   states. A caller-declared preregistered permutation is not authenticated.
4. Emit one sequential command unit per supplied job, with a singleton scene,
   its job/protocol/asset identities and unique output directories. Wizard,
   external driver and Compose paths must agree. The five-scene preparation API
   stays bounded. Commands are printed, never run; wizard preparation still
   fetches assets and requires separate authority. Reusing the external driver
   port requires the preceding unit's services to have stopped. Include every
   fresh startup in later cost accounting.
5. The integrated CPU rehearsal must bind commands to the very same ten jobs
   whose synthetic evidence is ingested. A mismatched scene, horizon or order
   must fail before any command is emitted for execution.
6. A clean-environment verifier must use a newly isolated environment and the
   frozen lock, record installer/source identities and fail visibly if install,
   dependency checks or CPU checks fail. Existing installed packages or matching
   version strings cannot count as a clean rebuild. No Python download is implied.
7. Apply the exact pinned upstream quaternion consumer checks while preserving
   raw values. Invalid norms and ambiguous float32 backend boundary cases block
   inspection. Source-defined direct service session UUIDs must match the ASL
   attempt; unrelated identifier types are not reinterpreted as session UUIDs.
8. The ASL file limit is an explicit operator/API/CLI bound, defaulting to the
   existing 256 MiB. Reject oversized receipts before reading; propagate an
   explicitly raised bound to the parser and retain it in the report. Changing
   this bound does not establish memory fit for a real log.
9. Managed-service port assignment must start above the external driver port,
   preventing a prepared driver from colliding with a managed service before
   the driver is running. Constrain the supported high end using the pinned
   allocator's search budget. Actual host port availability remains a runtime
   check, and no socket is opened by command preparation.

Regression tests precede fixes. Independent reviews inspect the changes before
one final frozen-source CPU verification. Outputs are written to a fresh
append-only evidence directory. Protected S1 inference files remain unchanged.

## Limits and remaining inputs

This contract fixes engineering failures that can be reproduced locally. It
does not select real scenes/roads/strata, scientific horizon/order/predictions,
licensed assets, runtime images, provider resources or a rental. The actual
AlpaSim scorer/capture, clipped-horizon semantics, renderer/model fit and fault
telemetry still need separately admitted qualification. The CPU evaluator lock
does not identify the AlpaSim runtime environment.

## Verification record

Pending implementation and independent review. Preserve historical evidence;
record exact final commands, results and unresolved environmental blockers here.
