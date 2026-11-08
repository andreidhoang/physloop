# Public AlpaSim ASL source contract

Exact Apache-2.0 sources at `NVlabs/alpasim` commit
`e121e59141dc391f6c295513d6dadc840a419149`, retrieved 2026-11-07.
`provenance.json` records immutable URLs, Git blob IDs and SHA-256 hashes.
`LICENSE` is the exact upstream license. No model, dataset, runtime output or
project-gated asset is included. The tests generate invented pose samples.

The eight complete `.proto` files form the entire transitive import closure of
`logging.LogEntry`. The packaged descriptor in `gate/asl_schema/e121e591.pb` was
generated with **libprotoc 33.4**:

```sh
protoc --proto_path=tests/fixtures/alpasim_asl/proto --include_imports \
  --descriptor_set_out=gate/asl_schema/e121e591.pb alpasim_grpc/v0/logging.proto
```

The packaged provenance records the compiler and descriptor hash. The protobuf
runtime is a separate pinned dependency; generated Python source is unnecessary.

## Load-bearing source relationships

- `logs_e121e591.py.txt`: `write_pb_stream` serializes `LogEntry` with a four-byte
  big-endian unsigned length prefix; `async_read_pb_stream` reads that framing.
- `scenario_evaluator_e121e591.py.txt`: `session_metadata.scene_id` is `clipgt_id`;
  `session_metadata.session_uuid` is `rollout_id`. `run_uuid` comes from separate
  `scenario_input.run_uuid`, so a session UUID is not a run UUID.
- `unbound_rollout_e121e591.py.txt`: `_build_rollout_timing` clips requested steps
  to ground-truth availability. `get_log_metadata` records this effective count,
  while `start_timestamp_us` records the **egomotion context start**. The caller's
  requested count and the control/policy start are distinct facts.
- `step_e121e591.py.txt`: logs initial context and first-policy poses plus all
  intermediate ego timestamps during execution. Poses use **local-to-AABB**
  coordinates, with actor `EGO`; `common.proto` orders quaternion fields **wxyz**.
  Preserve all samples, actor ordering and duplicates. Pose count is not steps.
- `event_loop_e121e591.py.txt` and `autoresume_e121e591.py.txt`: the empty
  `_complete` marker is written after runtime evaluation returns. It is outside
  ASL and does not by itself establish horizon, camera or map coverage.

## Inspection boundary

`gate.s2_asl.inspect_asl` checks framing, the full pinned protobuf schema, selected
metadata and original actor poses. Unknown fields are explicitly unsupported
under this pin. Duplicate singular/oneof fields are rejected as ambiguous even
though a permissive protobuf decoder would merge or choose the last value.
Other known service entries receive byte offsets, payload lengths, hashes and
type names, with `semantic_inspection=false`; their service semantics are not
validated. Nine source-defined direct `session_uuid` fields are checked against
the bound rollout, with a separate `session_identity_checked` receipt field;
the video-model service's distinct `session_id` is not compared to a rollout ID.
Original bytes remain the caller's artifact and are identified by
the whole-file receipt. These receipts do not authenticate their producer.

An absent pose submessage stays `null`; explicit scalar zeros stay zero. Proto3
scalar absence/default ambiguity is not turned into fabricated presence data.
Nonfinite pose values are retained as protobuf JSON tokens (`NaN`, `Infinity`,
`-Infinity`) and block inspection validity. No pose normalization, interpolation,
rig-frame conversion, deviation calculation or step-count inference occurs.

The exact upstream Rust geometry consumer also rejects near-zero and non-unit
quaternions. The `quaternion_compat/` sources pin that consumer and its glam
dependency. Inspection applies the same float32 numerical conditions without
modifying the raw quaternion. These are upstream format/consumer compatibility
conditions, not a newly selected scientific tolerance. Because the pinned SSE2,
scalar and NEON reductions can disagree at the boundary, an ambiguous result
blocks portable inspection explicitly; it does not choose a host backend or
widen the threshold. No compiler, floating-point mode or runtime is qualified.

File/frame/count limits default to 256 MiB/64 MiB/1,000,000 and are configurable
CPU engineering limits, not verified runtime capacity. A report may preserve
decoded prefix diagnostics after malformed trailing data; `parsed=false` and
`inspection_valid=false` prevent treating that prefix as a complete parse.
Even `inspection_valid=true` leaves actual steps, completion, renderer coverage,
black-frame coverage and map coverage unknown. Runtime qualification, score
admission and cloud-launch authorization are always false.
`inspect_capture(..., max_asl_bytes=N)` exposes the file bound explicitly and
passes it to the parser after checking the receipt before file I/O. The bound is
an operator choice, not permission supplied by the artifact. This implementation
materializes file bytes and the decoded report in memory, so increasing the
limit requires a suitable RAM budget. Scene/session reuse across two attempts is
ambiguous even when the external run UUIDs differ and therefore blocks capture.
