# Synthetic actual-capture inspection inputs (scope: synthetic)

Hand-written inputs for `tests/test_s2_actual.py` (build-spec B1, `gate.s2_actual`).
They contain no licensed driving data, model weights, real rollout output or
credentials, and they claim no runtime qualification.

Shapes follow the pinned AlpaSim e121e591 sources already vendored and hash-checked in
this repository:

| File | Shape source |
|---|---|
| `run/run_metadata.yaml` | `run_uuid:` line written by configuration.py:274-305 (see `gate.s2_capture.bind_run_uuid`) |
| `run/generated-user-config-0.yaml` | `simulation_config.n_sim_steps` and `scenes[].scene_id` read by `gate.s2_launch.check_generated_configs` |
| `run/physloop-capture/compose-up.stdout.txt` | runtime formatter and messages checked against source in `tests/test_s2_capture.py::test_pinned_message_formats_and_line_numbers` |
| `run/aggregate/results-summary.json` | scored row and failed_rollouts.py:48-64 runtime-failure rows (`run_uuid: null`; `failed-<idx>` without an allocated UUID) |

The `metrics.parquet` files are written by the test with pyarrow in the long-format
column layout of the pinned writer (`tests/fixtures/alpasim_s2_raw/data_e121e591.py.txt`,
columns `gate.s2_extract.PARQUET_COLUMNS`), including one preserved `valid=False`
sample, one required stream with partial coverage and one required stream absent.

Literal scenario: rollout `...0001` completed with 3 logged steps; rollout `...0002`
logged 2 steps (fewer than the 3 requested) and failed, leaving a folder without
`_complete`; one failure row `failed-1` has no rollout UUID. `provenance.json` holds the
SHA-256 of every file here.
