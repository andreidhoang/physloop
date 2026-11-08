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

Completed the scoped code fixes with failing regression tests first and
independent source/code reviews. The new packet had no implementation before
its tests; the rehearsal and CLI failed for missing integration. The ASL reused
session control returned an incorrect pass before the fix; invalid quaternion
and service-session cases likewise failed their new expectations. Seven new
exporter controls reproduced acceptance of foreign/disappeared endpoints, and
the port allocation tests reproduced the unsupported collision case. A separate
review found cross-unit output-directory ancestry overlap, also fixed with a
regression. No protected inference logic changed.

Final reproduction (use a new output path on subsequent runs):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/rehearse_s2_pilot.py \
  --output evidence/s2-prerental-hardening-20261107-v1
git diff --check
```

[Frozen evidence](../evidence/s2-prerental-hardening-20261107-v1/verification.json):
**1,360 tests passed in 55.94 s**, Ruff and whitespace checks passed. All **329**
payload receipts verified. The **186-file** source snapshot matched the working
tree after the run, before this results-only record update. Protected S1 hashes
matched their qualified baseline. Git HEAD was
`05c439f19f60bbcd8700b32170176e69709ba907`; uncommitted implementation is identified
by the archived source manifest, not that commit alone.

The synthetic packet bound all ten command units to the same ingested schedule;
wrong horizon and missing order blocked preparation. Existing synthetic scores
remained 0.8 with full 4.2 m deviations preserved. ASL completion and runtime
qualification remain unknown/false. No prepared command, model, GPU, scientific
experiment, cloud allocation or gated download was executed; cloud spend was $0.

The [new clean-environment verifier](../scripts/verify_clean_s2_environment.py)
passed 16 focused positive/negative controls. Its actual isolated install attempt
is retained in [separate evidence](../evidence/s2-clean-rebuild-20261107-v1/verification.json):
uv 0.8.17 exited 101 during macOS SystemConfiguration initialization, before
dependency installation or fresh tests. No fresh-rebuild pass is claimed.
[Pinned-source diagnosis](../evidence/s2-environment-20261107-v3/diagnosis.json)
shows that proxy environment overrides cannot avoid this version's unconditional
system configuration access; no complete original wheel route or alternative
working local uv executable was available in the inspected tool/cache locations.
Dependency declarations and lock were unchanged in this hardening task.

Remaining pre-rental boundaries are a working installer/package retrieval path
for the genuine clean CPU rebuild; actual scientific scenes/roads/strata,
horizon/order/prediction and terms/access records; and the concrete itemized
resource/action/cutoff/retention packet. Actual runtime images/lock/configuration,
model-renderer fit, completion/coverage/scorer qualification and host fault
telemetry resolve during separately admitted, bounded host work. Source-derived
templates and fake-provider checks do not close those boundaries. The existing
Vast watchdog's 60-minute stop-only scope is unchanged; a longer allocation or
different provider still needs a tested cutoff mechanism. No commit was made.
