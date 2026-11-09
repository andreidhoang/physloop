# Pinned public source for S2 raw metric fixtures

These are unchanged public AlpaSim source files at commit
`e121e59141dc391f6c295513d6dadc840a419149`, retrieved on 2026-11-07.
`provenance.json` records repository paths, exact Git blob hashes, SHA-256 hashes
and source URLs; `LICENSE` retains the upstream Apache-2.0 terms. No licensed
driving data, model weights or real rollout outputs are included.

`data_e121e591.py.txt` defines `MetricReturn`, `AggregationType` and
`create_metrics_dataframe`. The writer creates a long-format parquet table:

| Column | Writer value |
|---|---|
| `name` | Metric name string |
| `timestamps_us` | One integer timestamp per sample |
| `values` | Float64, including originally boolean metrics |
| `valid` | Boolean validity mask |
| `time_aggregation` | Lowercase `mean`, `median`, `max`, `min` or `last` |
| `clipgt_id`, `rollout_id`, `run_uuid`, `run_name` | Strings identifying the run |

Each metric's lists are exploded together. Rows are emitted in metric blocks;
timestamps need not be globally monotonic across different metrics. Check
monotonicity and duplicate samples within each identity and metric stream.

The raw scorer requirements differ from the summary's fields:

| Summary field | Raw source | Raw aggregation |
|---|---|---|
| `progress_clipped_rel` | `progress_rel_to_total` | `last` |
| `collision_at_fault` | `collision_front` OR `collision_lateral` | `max` |
| `gt_dist_traveled_m` | Same name; constant full GT path length | `last` |
| `offroad` | Same name | `max` |
| `left_corridor_laterally` | Same name | `max` |
| `dist_to_gt_trajectory` | Same name | `max` |
| `lateral_dist_to_gt_trajectory` | Same name | `max` |

`processing_e121e591.py.txt` derives the first two summary fields during
aggregation. Its modifiers remove warm-up samples and truncate at events, and it
can insert zero for missing offroad data. Preserve the original parquet bytes;
do not infer full trajectory coverage from summary fields or recalculate a
summary over all raw samples as though the modifiers were absent.

Ground-truth metrics heuristically initialize the first two timestamp values.
The image scorer emits `img_is_black` only where an image is available, for its
configured single camera, using threshold 10 on all pixel channels. It does not
establish all-camera render coverage. Required metric validity masks and raw
timestamp coverage must be checked explicitly; no missing sample becomes zero.

These source snapshots establish the metric-file schema and origins. Synthetic
PhysLoop execution traces are a separate, explicitly labelled test contract.
They do not establish parsing of AlpaSim's ASL protobuf logs, effective runtime
configuration, map coverage, actual rendering completeness, actual simulation
step counts, or a real rollout's qualification.

`modifiers_e121e591.py.txt`, `main_e121e591.py.txt` and `scene_score_e121e591.py.txt` (`src/eval/src/eval/aggregation/`) were added on 2026-11-09 from a clean local public clone at the same commit via `git cat-file`; their blob SHA-1 and SHA-256 are in `provenance.json`, whose per-entry `retrieval` field supersedes the top-level one for these three. CPU-12 executes the modifier row semantics and the >= 4 m modifier of `main` from them; `scene_score` is hash-checked only.
