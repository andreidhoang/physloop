# S2 CPU handoff and ASL inspection

Agent-authored engineering specification, 2026-11-07. Implements plan v5
**S2.1/S2.2/S2.3/S2.7 [C], G1/G2/G3·R1/G7**, under D-017/D-018 and the
owner's instruction to specify and execute the remaining CPU preparation.
This is a bounded continuation of the minimal pilot runbook, not the fourteen
package backlog. Cloud cost is $0; no scientific prediction is supplied.

## Objective

Make the existing ten-job synthetic rehearsal reproducible, inspect actual-format
AlpaSim log bytes without calling them scored evidence, and prepare source-checked
pilot commands whose unresolved inputs are explicit. The accepted scientific
protocol, protected inference code and existing evidence remain unchanged.

The smallest useful integration keeps three different outcomes separate:

| Check | What success establishes | What remains unestablished |
|---|---|---|
| Fresh CPU rebuild | Dependencies install from the unchanged resolved lock in a new environment | Linux/CUDA simulator environment |
| ASL inspection and receipt binding | Bytes parse under the pinned schema and match the declared attempt/session | Physical provenance, completed horizon, camera/map coverage, scorer correctness |
| Pilot command preparation | Source-checked explicit five-scene/off-on commands and required input inventory | Scene selection rationale, permissions, effective host configuration and execution |

## Implementation contract

### E0 — Environment

Use a functioning public uv distribution, verify its provenance, and create a new
environment from `uv.lock`. Never count copied installed packages or a matching
version list as a fresh frozen rebuild. Add only the pinned protobuf runtime needed
by the ASL decoder; resolve and archive the resulting dependency change before the
final rebuild. Keep the CPU evaluator lock distinct from the AlpaSim runtime lock.
Retain exact commands, exit status, tool/interpreter identities and lock hashes.
If this environment cannot resolve or fetch dependencies, retain the failed
attempts and report the precise boundary; successful existing-environment tests
remain separately useful.

### E1 — Pinned ASL parser

Reuse the protobuf runtime and a descriptor generated from the exact public
`e121e59141dc391f6c295513d6dadc840a419149` schema and its imports. Preserve source
licenses, hashes and a reproducible descriptor-generation command. Do not
substitute JSON or an invented wire schema for ASL.

Decode four-byte big-endian length framing, with explicit frame/resource limits.
Reject truncated headers/bodies, invalid protobuf and conflicting metadata.
Expose source identities and all original pose samples with timestamps and
coordinate semantics; preserve unexamined message identities/receipts rather
than silently claiming their contents were checked. Separate metadata's effective
`n_sim_steps`, caller-requested steps, pose count and observed timestamp span.
Initial/context/intermediate poses do not count as executed simulation steps.

`actual_steps`, completion, renderer coverage, black-frame coverage and map
coverage stay unknown unless their own inspected evidence establishes them.
This implementation does not establish those properties. Metadata timestamps
must not silently become policy-start timestamps. ASL session/scene identifiers
are bound only according to verified upstream relationships, never guessed from
the names `run_uuid`, `clipgt_id` or `rollout_id`.

### E2 — Capture inspection integration

Provide a separate offline CLI/API for ASL inspection. Use the existing strict
JSON, path/size/hash receipts, explicit schedule and complete attempt ledger.
Preserve failed/pre-UUID attempts and retry diagnostics; inspect available logs
even when the ledger cannot admit a batch. Do not select a successful retry.
Require explicit per-attempt ASL bindings and reject duplicates, unknown attempts,
missing expected logs, receipt corruption and mismatched metadata. Preserve all
diagnostics. The report never emits score records or a gate verdict.

The existing `synthetic_source_contract_v1` ingestion remains synthetic. A decoded
ASL file must never be relabelled as its `world_m` trace. A complete receipt check
does not authenticate a producer or prove the simulator finished. Every report
retains `runtime_qualified=false`, `score_admitted=false` and
`cloud_launch_authorized=false`.

### E3 — Five-scene pilot preparation

Re-check official source/branch state, retain the reviewed evaluator pin unless a
change is separately qualified, and document differences. Verify actual Hydra
selection/override keys, external-driver startup, camera/control settings and
Harmonizer off/on behavior from pinned source. Produce argv as lists, not shell
interpolation. Never use the 441-scene example as the five-scene pilot.

Complete public-source-resolvable inventory facts. Leave scientific scene/road/
stratum choices, authorization records, gated asset bytes, effective runtime
locks/images/configs and delivered hardware unknown until real evidence exists.
Templates clearly separate required inputs from literal executable arguments.
No new rental or asset acquisition occurs to fill this inventory.

The pinned wizard later joins command values and uses a container shell. This
v0 therefore supports portable ASCII paths and scene identifiers only; it must
reject unsupported names before emitting commands. JSON quoting at the first
argv boundary is not a guarantee of correct downstream shell handling. Immutable
base-image references require the active physics/controller/runtime services to
use their `external_image=true` route instead of automatic image building.
Both renderer arms include the source-documented `--enable-timing` instrumentation
required by the existing S2 engineering pilot specification.

### E4 — Verification and evidence

Write failing tests before implementation. Positive ASL fixtures contain invented
poses encoded with the actual upstream schema; their origin is always synthetic.
Independent review checks parser/source correspondence and integration failure
behavior. Extend/reuse the existing rehearsal and source snapshot machinery,
then run the full CPU suite, Ruff and whitespace checks. Verify protected S1
hashes are unchanged and use a fresh append-only evidence directory.

### E5 — Pre-rental integration and hardening

The follow-up [acceptance contract](../experiments/2026-11-07-s2-prerental-hardening.md)
requires ten command units bound to the same schedule used by ingestion, in an
explicit caller-supplied order. `gate.s2 prepare-pilot` accepts exactly
`schema_version=1`, `launch_request`, `schedule_request` and `execution_order`.
The latter contains `method` (`alternating` or `preregistered_permutation`) and
all ten unique `job_ids`. Missing order, a scene/horizon mismatch or incomplete
coverage emits no units. Alternation is checked mechanically; a declared
preregistered permutation is neither authenticated nor evidence of randomness.

Each unit selects one scene, preserves job/protocol/asset identity, and uses
distinct run and driver-output directories. Execute units serially only after
admission, with prior Compose and driver processes stopped before reusing the
external port. A new driver per unit means ten startup periods to include in
later paid-time accounting. Templates establish no measured performance.
The preparer sets `wizard.baseport=driver_port+1`, so managed services cannot
take the external driver's port before it starts. The supported driver range is
1–59,535: the pinned setup constructs six managed-service instances across its
deployment definitions and may scan 1,000 ports per allocation. Actual port
availability is still checked on the admitted host; source preparation opens no
sockets. See the launch audit for the allocation source and retained pin.

ASL inspection also rejects ambiguous scene/session reuse across attempts even
when an external run UUID differs. Raw quaternions remain unchanged; compatibility
checks use the pinned upstream consumer's numerical conditions, not a newly
adopted scientific tolerance. Service payloads remain semantically uninspected
beyond their explicit source-defined session bindings. The operator may supply
`inspect-capture --max-asl-bytes N`; the default is 2 GiB per file, an agent
technical default (D-018) provisional until the first measured ASL. Receipts
exceeding the chosen bound are rejected before reading. Files are streamed in
1 MiB chunks and hash-verified, never loaded whole; the decoded report is kept in memory.

The exporter probe must bind metrics to its exact running task container and
listener processes. An occupied port, missing process visibility, exited
container or unrelated metrics listener blocks the probe without stopping host
services. Host execution of this check remains unqualified until admitted.

`scripts/verify_clean_s2_environment.py` supplies a separate fresh-environment
check with a new empty cache, an existing Python interpreter and the frozen lock.
It records installer failure independently of current-interpreter test results.

## Deterministic acceptance cases

1. A valid framed protobuf fixture preserves every initial/intermediate pose and
   timestamp in input order; requested/effective steps stay separate and actual
   completion remains unknown.
2. Torn framing, oversized frames, malformed protobuf, wrong/duplicate metadata,
   identity mismatch and nonfinite pose data produce explicit failures.
3. Missing/corrupt/wrong-attempt logs cannot produce a complete capture report;
   failed jobs and pre-UUID crashes remain in the ledger. A later successful
   retry does not erase or select away its predecessor.
4. A source-format parse never admits scores, manufactures zero render/map errors,
   or changes the existing synthetic ingestion contract.
5. Five explicit scene inputs produce exactly two Harmonizer arms and ten jobs;
   missing/extra/duplicate scenes or unresolved required command inputs block
   executable command preparation. Source-checked templates are not launch rights.
6. The existing ten-job rehearsal retains ten scores of 0.8 and ten full deviations
   of 4.2 m; missing artifacts/export corruption/cost/deadline controls still fail.
7. A clean frozen rebuild, if achievable, matches the final lock and runs the same
   checks. Otherwise the report states the unresolved rebuild instead of claiming
   end-to-end reproducibility.

## Completion boundary

CPU implementation is complete when these contracts have tests, independent
review, reproducible commands and recorded results. The real pilot still needs
Huy's scientific registration/prediction and concrete access/action approval,
then actual host/capture/scorer/teacher-renderer-Harmonizer qualification.
No RL training, protected edit, publication or commit follows from this task.

## End-to-end CPU workflow

All commands below run from the repository root and perform local preparation
only. Input examples with invented values are emitted by the rehearsal; they
must never be mistaken for registered scene selections or fetched asset identities.

1. **Prepare arm templates:** `python -B -m gate.s2 prepare-launch REQUEST.json`.
   The exact request fields and pinned source interfaces are documented in
   [the launch audit](research/2026-11-07-pilot-launch-audit.md). Success emits two
   five-scene arm templates and no subprocess execution. `prepared=true` checks
   the supported input/command contract, not paths, images, permissions or fit.
   Invoking the emitted wizard later can download assets even with run method NONE.
2. **Bind ordered work:** use `gate.s2 schedule` to identify ten jobs, then
   `python -B -m gate.s2 prepare-pilot PACKET_REQUEST.json` to reconcile the
   schedule, launch request and explicit registered order. It emits ten serial
   singleton command units; the rehearsal supplies only synthetic examples.
   Retain every actual start/failure/completion in the existing attempt ledger;
   this preparer does not observe execution or manufacture runtime UUIDs.
3. **Inspect captured ASL:** `python -B -m gate.s2 inspect-capture REQUEST.json
   --root ARTIFACT_DIRECTORY`. Receipt inputs have `path`, `sha256` and `size_bytes`.
   The request has exactly `schema_version=1`,
   `evidence_kind="alpasim_asl_inspection_v1"`, `capture_origin` (either
   `synthetic_fixture` or `runtime_claim_unverified`), the pinned `evaluator_sha`,
   `schedule`, `schedule_request`, an `events` receipt, and `artifacts` entries
   containing `attempt_id` and an `asl` receipt. Pilot/full schedules must reproduce
   from `schedule_request`; only explicitly synthetic schedules may omit it.
4. **Read the boundaries:** the result retains the canonical request/hash, original
   ledger and all inspected captures. `capture_inspection_passed` requires ledger
   admission plus valid receipts/schema/identity checks. It never supplies score
   records. ASL `session_uuid` binds to `rollout_id`, `scene_id` to `clipgt_id` and
   scheduled scene; the external `run_uuid` cannot be recovered from these fields.
   Local-to-AABB poses remain untransformed. The wrapper streams the ASL with a
   provisional 2 GiB declared-size bound and limits events JSON to 16 MiB before
   opening files; these limits have not been qualified against actual simulator log sizes.
5. **Rehearse and verify:**

   ```sh
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/rehearse_s2_pilot.py \
     --output evidence/s2-cpu-handoff-UNIQUE-ID
   ```

   Use a new output directory. It contains source hashes, the source snapshot,
   tests/lint logs, ten-job synthetic scoring/export/cost/cutoff controls,
   actual-format synthetic ASL inspection with a torn-frame control, and pure
   launch preparation with a missing-scene control. Its `cpu_handoff_ready` field
   covers the current interpreter and source checks only; the separate
   `clean_frozen_sync_verified` field must remain false until a genuine rebuild
   is demonstrated. This is not a scientific or GPU launch readiness decision.

Missing real scenes, registered scientific choices, runtime image/lock/config
receipts, permissions and action records remain in `experiments/S2_SETUP_DRAFT.json`.
CPU verification closes none of these fields by assumption. Actual raw metric
coverage, scorer replay and renderer/map qualification remain host work after
admission, rather than assertions made by the ASL parser.
