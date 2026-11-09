# Actual-extraction fixtures (scope: public_source_fixture)

Hand-written inputs for `tests/test_s2_actual_extract.py` (`gate.s2_actual_extract`, D-024
module F2). Agent-authored (D-018/D-024). They contain no licensed driving data, model
weights, real rollout output or credentials, and they claim no runtime qualification.

`scenario.json` describes seven units of one batch in the output shapes of pinned AlpaSim
`e121e59141dc391f6c295513d6dadc840a419149` (sources vendored and hash-checked under
`tests/fixtures/alpasim_s2_raw/`, plus the read-only R2 lane clone):

| Unit | Literal case |
|---|---|
| `pair_a` | arm A, `clip-s1`, complete; full-stream deviation maximum 0.8 m; summary pass 0.8 |
| `pair_b_late` | arm B, `clip-s1`, collision at 200000 us; deviation 4.2 m at 400000 us; summary keeps rows up to the collision (modifiers.py:99-101, processing.py:63), so its `dist_to_gt_trajectory` is 1.0 |
| `invalid_samples` | one `valid=False` sample in `collision_front` (required) and one in `img_is_black` |
| `missing_offroad` | no `offroad` stream; the summary carries the upstream zero-fill 0.0 (processing.py:752-760) |
| `failed_pre_uuid` | `Rollout FAILED ... uuid=N/A`; summary failure row `failed-0` with `run_uuid` null |
| `incomplete_folder` | rollout folder without `_complete` and without metrics; failure row with `run_uuid` null; Compose exit 0 |
| `internal_retry` | first rollout fails (retryable), scheduler retry line, second rollout completes; two rollout UUIDs |

How the test materialises a unit:

- `run_metadata.yaml`: `run_name` and `run_uuid` lines, PyYAML sorted order (configuration.py:274-305).
- `eval-config.yaml`: `fail_on_driver_termination: true` as the pinned e2e_challenge preset (dev.yaml:51), written by the wizard to the log dir (configuration.py:241-249, 320-325); the horizon is reported only under it.
- `physloop-capture/compose-up.stdout.txt`: the `log` lines; message formats are the pinned runtime lines checked in `tests/test_s2_capture.py`.
- unit records: the exact `_Run._envelope` shape, with a `compose_up` step, a `run_dir_hash` step carrying the manifest `files_sha256` (`symlink_policy="record"`), `reconcile` and `ledger_terminal`.
- `rollouts/<clipgt_id>/<rollout_uuid>/`: `rollout.asl` placeholder bytes (never decoded), `_complete` (empty) when `complete`, and `metrics.parquet` written with pyarrow from `base_streams` with the per-rollout `drop`/`override`, in the long layout of `create_metrics_dataframe` (data.py:2027-2077): one row per sample in metric blocks, `timestamps_us` int64, `values` float64, `valid` bool, strings `large_string`.
- `aggregate/results-summary.json`: scored rows in the shape of processing.py:293-337 (`base_summary_metrics` plus `metrics_override`) and failure rows in the shape of failed_rollouts.py:44-80. Summary numbers are fixture values, not a recomputation of upstream aggregation.

`provenance.json` holds the SHA-256 of `README.md` and `scenario.json`.
