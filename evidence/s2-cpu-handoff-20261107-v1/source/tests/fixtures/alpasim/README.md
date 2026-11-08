# Public AlpaSim geometry source fixtures

These are **unmodified complete source files** from NVIDIA's public AlpaSim repository.
They preserve the original Apache-2.0 SPDX and copyright lines. The upstream Apache
license is included in `LICENSE`; exact source URLs, commits, Git blob IDs and SHA-256
hashes are in `provenance.json`.

- `ground_truth_20235345.py.txt`: before the #215 stationary-tail fix, as linked in #198.
- `ground_truth_e121e591.py.txt`: the audited challenge HEAD, including #215.

The `.txt` suffix marks immutable source data for formatters and import scanners;
the contents remain byte-for-byte identical to the original `.py` files.

Do not format or edit these snapshots. A new upstream revision needs a new named
snapshot and explicit provenance; a hash mismatch must fail the tests.

## What executes

`tests/test_alpasim_ground_truth_upstream.py` verifies source bytes and compiles only
the original `_end_direction` and `_lateral_distance_to_gt` function AST nodes, without
rewriting them. It supplies real NumPy and Shapely objects. It does not import the
full evaluator or replace the geometry with a locally implemented equivalent.

The original two #198 geometries must reproduce 5 m with the old helper and less than
0.00001 m with the corrected helper. Additional known answers cover the 1 m supported
direction boundary, duplicate/short stationary tails, endpoint position and forward
versus backward extension, lateral distances around 4 m, and the union with earlier
recorded path segments. The 4 m cases check returned geometry distances, not execution
of the scorer's corridor flag or its aggregation.

Run with the resolved project CPU environment:

```sh
uv run --frozen python -B -m pytest -p no:cacheprovider -q tests/test_alpasim_ground_truth_upstream.py
```

## Scope

This is a CPU, source-helper known-answer check on invented geometry. It uses no
licensed scene data, model weights, simulation, renderer, GPU, container or rental.
It does **not** verify the full `GroundTruthScorer`, metric aggregation, scene-score
logic, deployed container bytes, actual rollout geometry, seed behavior or physical
validity. Those remain separate S2 qualifications.
