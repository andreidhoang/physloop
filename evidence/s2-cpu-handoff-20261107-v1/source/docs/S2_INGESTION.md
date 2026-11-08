# S2 synthetic evidence ingestion

Agent-authored engineering preparation, 2026-11-07, serving S2.1/S2.3 and G1/G2/G3·R1/G7 under D-017/D-018. This implements local identity, schedule, attempt reconciliation and synthetic raw-file checks. It does not admit the scientific pilot or adopt P-032.

The reader consumes actual Parquet bytes using PyArrow and an explicit **PhysLoop synthetic capture contract**. Its execution trace is not AlpaSim's ASL format. A successful report establishes the checked fixture contracts; it does not qualify actual AlpaSim capture, the scorer, renderer, policy or scientific protocol. Environment-lock setup and the complete verification record are separate checks; this document does not claim a successful frozen rebuild.

Environment note, November 7: PyArrow is pinned to **25.0.1**. The installed
package's 754 hashed RECORD entries were verified, and its unchanged package
entry was copied from previously resolved local `uv.lock` metadata; it has no
declared transitive dependencies. The source-lock SHA-256 was
`9296e1e02dbba831aeb1d63a0a001a398d37bb5e0b8ae4c09edcd64005457109`
and the copied package-block SHA-256 was
`80c6cb2299a10087a6ef53c80d7c6fab3d30cfc6b0f13dc236d459d4f6a9989b`.
Other locked package entries were unchanged. Local Parquet round-trip and
installed-versus-locked version checks passed. The host's `uv 0.8.17` crashes in
macOS SystemConfiguration before resolution, including offline mode, so a clean
`uv sync --frozen` with a working newer uv remains unverified. No fresh resolver
run or clean environment rebuild is inferred from the metadata checks.

The subsequent [CPU handoff](S2_CPU_HANDOFF_SPEC.md) adds a **separate** pinned
ASL inspection command and protobuf runtime. It does not change this synthetic
scoring contract or enable actual AlpaSim score admission. The environment
attempts in `evidence/s2-environment-20261107-v1/` and `-v2/` retain the dependency
provenance and failed clean-rebuild checks; current installed versions are checked
separately by the rehearsal.

## Commands and outputs

From the repository root, with the declared CPU dependencies installed:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m gate.s2 matrix --replicates 3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m gate.s2 schedule REQUEST.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m gate.s2 ingest REQUEST.json --root DIR
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/verify_s2_ingestion.py --output evidence/s2-ingestion-NEW-ID
```

`matrix` reports 10 pilot policy rollouts and 2,646 full-sweep policy rollouts at three replicates, excluding diagnostics and retries. Independent road count remains unknown. These are schedule counts, not measured runtime or prices. The historical `cost-table` remains a separately labelled dated scenario.

`schedule` reads the complete builder request below. `ingest` reads a different request describing a retained bundle under `DIR`. Both operate locally; they create no rental, run no model and download no asset. The CLI prints JSON; unsuccessful checks return exit 2. Existing `normalize` remains available for caller-supplied sidecars, whose authenticity it does not verify.

The verifier requires a new output directory and refuses to overwrite one. It
creates separately retained positive and missing-offroad fixtures, runnable
`request.json` examples, the active matrix and blocked-draft reports. It runs the
full CPU suite, lint and whitespace checks, archives the actual source bytes,
checks installed direct dependency versions against the lock, and verifies that
the protected S1 sources still match their qualification hashes. Its reports
explicitly distinguish these checks from a clean frozen environment rebuild.

## Schedule request

The JSON object supplies exactly the keyword inputs of `gate.s2_schedule.build_schedule`:

| Field | Contract |
|---|---|
| `study_id` | Explicit nonempty study identity |
| `phase` | `harmonizer_pilot` or `full_sweep` |
| `scenes` | Rows with exactly `scene_id`, `road_id`, `stratum`, `asset_sha256`; five pilot scenes or 441 full scenes |
| `policies` | Pilot: `{"A": "teacher-id"}`; full: distinct policy IDs under `A` and `B` |
| `protocol_hashes` | Pilot: distinct hashes under `off` and `on`; full: one hash under `fixed` |
| `replicates` | Pilot: integer 1; full: integer ≥3 |
| `requested_steps` | Explicit positive integer; no inferred horizon |

Scene IDs are unique and each road belongs to one stratum. Digests are 64 hexadecimal characters, normalized to lowercase. The builder sorts scene/arm/protocol keys for stable output. That serialization order does not replace the registered randomized or alternating runtime order.

Each emitted job has exactly `job_id`, `phase`, `arm`, `policy_id`, `scene_id`, `road_id`, `stratum`, `replicate_id`, `protocol_hash` and `requested_steps`. Its canonical SHA-256 identity also binds the study ID and scene asset digest from the retained request. It does not establish that a declared asset or protocol is authentic.

For pilot/full ingestion, retain the entire builder request as `schedule_request`. The reader rebuilds it and requires exact equality with `schedule`; editing a compact job row cannot bypass its source request. Tiny engineering fixtures may instead use jobs with `phase="synthetic"` and `schedule_request=null`. That exception does not reduce scientific scene counts.

## Bundle request and file receipts

The ingestion request has exactly these fields:

| Field | Contract |
|---|---|
| `schema_version` | Integer 1 |
| `evidence_kind` | `synthetic_source_contract_v1` only |
| `evaluator_sha` | `e121e59141dc391f6c295513d6dadc840a419149` |
| `schedule` | Complete list of planned jobs |
| `schedule_request` | Full builder request for pilot/full, or null for tiny synthetic fixtures |
| `events` | File receipt for a JSON array of recorded attempt events |
| `batches` | Rows with exactly `arm`, `protocol_hash`, `summary`; summary is a file receipt |
| `artifacts` | Rows with exactly `attempt_id`, `metrics`, `trace`, `completion`; the latter three are file receipts |

A file receipt has exactly `path`, lowercase `sha256` and nonnegative integer `size_bytes`. Paths are relative POSIX paths under `DIR`; absolute paths, traversal, empty components, symlinks and non-regular files are rejected. The reader checks byte length and SHA-256 and parses the same bytes it checked. A real directory path without symlink components is required for `DIR` as well.

Canonical JSON identities use strict UTF-8, sorted object keys and compact separators. Duplicate JSON keys, nonfinite numbers and malformed JSON are rejected. These checks bind bytes and declarations; hashes are integrity checks, not proof of origin or permission to use an asset. Capture authenticity remains unqualified.

## Attempt ledger

Every event has `event_id`, `job_id`, `attempt_id`, `type` and integer `timestamp_us`. Events stay in recorder order within one monotonic clock domain; they are not sorted to repair inconsistent evidence.

| Event type | Additional exact fields and conditions |
|---|---|
| `started` | `parent_attempt_id`, `retry_reason`, `upstream_identity`; parent/reason are both null for the first attempt |
| `completed` | `reason="exact_horizon"`, non-null `upstream_identity` |
| `failed`, `cancelled` | Nonempty `reason`, `upstream_identity` which may be null |

An upstream identity has exactly `run_uuid`, `clipgt_id` and `rollout_id`. A first start can precede UUID allocation; a failure then remains a scheduled failed attempt. A started attempt without a terminal event remains incomplete. Repeated event IDs are idempotent only for identical canonical JSON values; conflicting duplicates block.

Retries retain parent, reason and all prior attempts. **Every multiple-attempt job blocks current selection**, including a failed attempt followed by a successful one. No best-score or last-success rule is supplied. Admission requires one completed first attempt for every scheduled job and no ledger issue. A ledger failure returns the retained schedule/events and diagnostics before raw-file normalization.

## Synthetic capture contract

The completion file is an empty touch marker. Its presence alone never proves completion.

The trace JSON has exactly `schema_version=1`, `kind="synthetic_execution_trace_v1"`, `upstream_identity`, `coordinate_frame="world_m"`, `start_timestamp_us`, `step_interval_us` and `samples`. It contains an initial sample plus every requested step: `requested_steps + 1` samples with exact consecutive indices and timestamps. Each sample has exactly `step_index`, `timestamp_us`, `ego_xyz_m`, `render_status`, `black_frame` and `map_available`.

The reader checks finite three-coordinate poses, render status `ok`, boolean black-frame observations and available map evidence. It preserves separate render, black-frame, pose and map diagnostics. These are fixture observations supplied by the synthetic capture, not image-based black-frame detection or an actual runtime monitor. Missing/partial observations block; unknown is not converted to zero.

The long-format Parquet columns are exactly `name`, `timestamps_us`, `values`, `valid`, `time_aggregation`, `clipgt_id`, `rollout_id`, `run_uuid` and `run_name`. Timestamps use integer columns, values float64, validity boolean, and other columns strings. The checked required streams are:

| Raw metric | Required aggregation label |
|---|---|
| `progress_rel_to_total` | `last` |
| `progress_rel` | `min` |
| `gt_dist_traveled_m` | `last` |
| `collision_front`, `collision_lateral` | `max` |
| `offroad`, `left_corridor_laterally` | `max` |
| `dist_to_gt_trajectory`, `lateral_dist_to_gt_trajectory` | `max` |

Every required stream must cover the entire synthetic trace exactly once, in increasing timestamp order, with finite valid values and matching upstream identity. Missing offroad is rejected even if the supplied summary says zero. Full deviation comes from the full `dist_to_gt_trajectory` stream, including samples after summary truncation; ≥4 m observations remain visible.

The pinned source also constrains collision/offroad/corridor events to zero or
one, and GT/trajectory distances to nonnegative values. These domain checks were
added after independent review reproduced admission of impossible finite values.
The regression suite separately models a collision cutoff followed by a 4.2 m
deviation, retaining the overall collision score of zero and the full deviation.

The existing adapter still checks the supplied summary's internal score/status/metric consistency. This reader does **not** recompute the scorer or prove that the supplied summary is the aggregation of the Parquet data. A raw metric presence check, a source hash and a self-consistent summary are separate evidence checks.

## Result boundary and remaining work

`admitted=true` means the whole synthetic bundle passed these checks. Every normalization batch is cleared of admitted records if any bundle issue remains. The report retains ledger data, observations, source receipts and structured issues. `runtime_qualified`, `summary_recomputed`, `cloud_launch_authorized` and `publication_authorized` remain false.

Actual AlpaSim ASL/capture integration, scorer replay, real renderer/map coverage, recording-end clipping and a registered retry-selection rule remain outstanding. The exact-horizon requirement is unchanged: unexplained shortening and legitimate but unsupported terminal clipping both stay blocked pending the applicable protocol and implementation. Scientific definitions, predictions, license/access and concrete rental actions retain their existing gates. Protected inference code is unchanged.
