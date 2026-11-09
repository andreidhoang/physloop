# Pinned S4 upstream source fixtures

Unmodified files from four Apache-2.0 repositories at the S4 engineering pins, each stored with
`.txt` appended so formatters, importers and test collectors ignore it. Each repository folder
keeps its own `LICENSE`. `provenance.json` records the repository, commit, path, byte count,
SHA-256, Git blob SHA and source URL of every file. Do not edit or reformat these files; a new
upstream revision needs a new snapshot and new provenance, and a hash mismatch must fail tests.

| Folder | Repository | Commit |
|---|---|---|
| `alpagym/` | NVlabs/alpagym | `972d160eed0e23d388497851504a3a233fec5879` |
| `cosmos-rl/` | nvidia-cosmos/cosmos-rl | `d2a2c57c4bd6496482bc42da19a59b4432705eda` |
| `alpamayo-recipes/` | NVlabs/alpamayo-recipes | `1d99fc50370c96637157455da386323a44663c9d` |
| `alpasim-gympin/` | NVlabs/alpasim | `46b401beaec27977cbb86826df8e39cec2d49e49` |

Licence check (build spec §3): every repository's `LICENSE` is Apache License 2.0, which
permits redistribution with the licence retained; none of the four has a `NOTICE` file.

## What executes

`tests/test_s4_capture.py` executes only selected AST nodes of stdlib-only pinned code, in a
restricted namespace: the `StreamingRolloutWorker` retry/collect class and its two dataclasses,
`AlpagymRollout.rollout_generation`, `filter_trainable_rollouts`, the host's
`_build_cosmos_command`/`_build_cosmos_launcher_command`, `ArtifactPaths` with
`build_artifact_paths` (run-directory layout, `tests/test_s4_capture.py`), and
`_validate_cosmos_grpo_batch_geometry` from `config_validation.py` (the batch-geometry oracle,
`tests/test_s4_registration.py`). Everything else is parsed only: wrap
target and fake signatures are compared with the pinned `FunctionDef`s, and every compiled Hydra
or Cosmos key is resolved against the pinned dataclass/Pydantic fields
(`tests/test_s4_registration.py`). No upstream package is imported, no model, dataset, GPU,
container or network is used, and nothing here establishes runtime behaviour on a host.
