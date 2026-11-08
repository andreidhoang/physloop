# Five-scene pilot launch source audit

Agent-authored, 2026-11-07. Implements plan v5 **S2.1/S2.2/S2.3/S2.7 [C],
G1/G2/G3·R1/G7** under the [CPU handoff contract](../S2_CPU_HANDOFF_SPEC.md).
This is public-source inspection and command preparation, not a simulator run,
scientific registration, asset acquisition or spending authority. No selected
scene, road, stratum, horizon, prediction or accepted use terms is supplied here.

## Revision check

The official GitHub API was checked on 2026-11-07:

| Surface | Observed identity | Consequence |
|---|---|---|
| [`e2e_challenge`](https://api.github.com/repos/NVlabs/alpasim/branches/e2e_challenge) | `297b4d00cfdd2c4f636b9d9bcee7a85c699174c0` | One commit after the reviewed pin |
| [`main`](https://api.github.com/repos/NVlabs/alpasim/branches/main) | `affc2eab209fa43bdfa2f26c0f8d437922d78a68` | No substitution of the evaluation branch |
| [Latest release endpoint](https://api.github.com/repos/NVlabs/alpasim/releases/latest) | `v2026.5`, published 2026-06-04 | This endpoint is a release record, not the challenge head |

The [exact comparison](https://github.com/NVlabs/alpasim/compare/e121e59141dc391f6c295513d6dadc840a419149...297b4d00cfdd2c4f636b9d9bcee7a85c699174c0)
adds the NuPlan reference-score data, its reference manifest, local-evaluation
README text and evaluation tests. It changes no runtime, driver, wizard or scorer
implementation. Retain **`e121e59141dc391f6c295513d6dadc840a419149`** for this
packet; this finding does not qualify a new runtime. Re-check official state at
actual admission. The model implementation candidate remains
**`36aeb4c5938cbc2eb2aed33b22434773da4ab639`** in `NVlabs/alpamayo1.5`.

The [public-source fixtures](../../tests/fixtures/alpasim_launch_upstream/README.md)
retain the 24 load-bearing source/license files with original contents. Their
[manifest](../../tests/fixtures/alpasim_launch_upstream/manifest.json) records
immutable URLs, Git blob identities, SHA-256 and sizes; all saved files matched
their upstream Git blob identities. Python copies use `.py.txt` filenames and
are never imported as the simulator. These receipts establish inspected source
bytes, not runtime behavior or asset identity.

## Exact scene selection and launch stages

The pinned [curated preset](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/nurec_scenes/curated_val.yaml)
loads both `sim_scenes.csv` and `sim_scenes_2604.csv`; its default suite selects
the 441-scene population. The pilot keeps those catalogs and explicitly overrides
**both** `scenes.test_suite_id=null` and `scenes.scene_ids=[ID1,ID2,ID3,ID4,ID5]`,
with `scenes.limit_to_first_n=0`. The five identifiers are required registered
inputs, not the first five records of a suite. The
[selector](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/context.py#L55-L131)
rejects simultaneous suite/ID selection and sorts selected scenes by ID.
[Catalog lookup](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/scenes/sceneset.py#L272-L292)
rejects unknown IDs and deduplicates scene versions. An ID-only command therefore
does not pin the selected USDZ bytes: preserve the resolved artifact UUID,
dataset revision, file path, byte size and hash alongside the road/stratum map.

**Do not invoke the wizard during this CPU-only task.**
[`WizardContext.create`](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/context.py#L195-L209)
calls `fetch_artifacts` before deployment selection. Consequently
`wizard.run_method=NONE` and `wizard.dry_run=true` are **not asset-download
barriers**. `NONE` suppresses service launch after artifact resolution; it does
not make the wizard an offline configuration validator.

The [dev preset](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/e2e_challenge/dev.yaml)
selects local deployment, one-GPU topology and
[`external_static`](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/driver_source/external_static.yaml).
The wizard does not start that external driver. After separately authorized
setup and asset resolution, the source-supported sequence is:

1. Generate one arm's configs with `alpasim_wizard` and `wizard.run_method=NONE`.
   Preserve all resolved configs and the emitted `docker-compose.yaml`.
2. Start the external stock Alpamayo 1.5 driver against the generated
   `driver-config.yaml`, in the pinned AlpaSim environment.
3. From that arm's run directory, run
   `docker compose up --exit-code-from runtime-0` with the external driver already
   ready. Preserve process logs, attempted jobs and failures.

The [configuration writer](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/configuration.py#L252-L260)
resolves the driver subtree into `driver-config.yaml`.
The [driver entrypoint](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/driver/src/alpasim_driver/main.py#L1106)
accepts Hydra's config path/name; the
[Compose generator](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/deployment/docker_compose.py#L42-L94)
documents the final command. These are reviewed source interfaces, not a tested
five-scene deployment.

## Required command inputs

`gate.s2_launch.build_pilot_launch(request)` prepares argv lists without executing
them. The request is deliberately separate from scientific registration and the
complete-attempt ledger. Its required keys are:

| Key | Required value |
|---|---|
| `schema_version` | `1` |
| `alpasim_sha` | Exact retained AlpaSim SHA above |
| `alpasim_checkout` | Absolute path to the eventual pinned Linux checkout |
| `scene_ids` | Exactly five distinct registered scene IDs |
| `requested_steps` | Positive integer from the scientific protocol; no default chosen |
| `driver_port` | Explicit port in 1–59,535, shared by driver and runtime endpoint; actual availability remains unverified |
| `teacher_snapshot` | Absolute path to the verified local teacher snapshot |
| `data_root` | Absolute data/cache root on the eventual host |
| `renderer_image`, `base_image` | Pullable immutable `repository@sha256:...` references |
| `arms` | Two objects, each with `harmonizer` (`false`/`true`), distinct absolute `run_dir` and `driver_log_dir` |

This preparer supports portable ASCII absolute paths using letters, digits,
`_`, `-`, `.`, and `/`, with no empty, `.` or `..` path segments. Scene IDs use
letters, digits, `_`, `-` and `.`, excluding the complete IDs `.` and `..`.
Spaces, shell metacharacters, backslashes and OmegaConf interpolation are outside
this narrow input contract. JSON quoting alone is not a Hydra round-trip proof.

The pure builder can reject malformed inputs; it cannot establish that paths,
images or assets exist, match their claimed identities, are permitted, or fit the
host. Two five-scene arm templates establish **ten requested jobs**, not a
registered physical execution order. The wizard's scene sorting is not randomized
order. Freeze the permitted off/on ordering and bind each actual attempt to the
registered schedule before execution; do not silently equate grouped arm commands
with an interleaved order.

The following is an **argv template**, not an executable request. Angle-bracket
tokens must be resolved, validated and preserved as individual arguments. Never
paste it through shell interpolation. Both wizard and driver commands use the
AlpaSim checkout as their working directory.

```json
[
  "uv", "run", "--no-sync", "alpasim_wizard",
  "driver=alpamayo1_5", "+e2e_challenge=dev", "+nurec_scenes=curated_val",
  "scenes.test_suite_id=null", "scenes.scene_ids=<JSON_LIST_OF_FIVE_IDS>",
  "scenes.limit_to_first_n=0",
  "runtime.simulation_config.n_rollouts=1",
  "runtime.simulation_config.n_sim_steps=<REQUESTED_STEPS>",
  "runtime.enable_autoresume=false", "wizard.nr_retries=0",
  "driver.port=<DRIVER_PORT>",
  "wizard.baseport=<DRIVER_PORT_PLUS_ONE>",
  "wizard.external_services.driver=[localhost:<DRIVER_PORT>]",
  "driver.model.checkpoint_path=<TEACHER_SNAPSHOT>",
  "driver.output_dir=<DRIVER_LOG_DIR>", "driver.inference.max_batch_size=1",
  "defines.filesystem=<DATA_ROOT>", "defines.base_image=<IMMUTABLE_BASE_IMAGE>",
  "services.physics.external_image=true",
  "services.controller.external_image=true",
  "services.runtime.external_image=true",
  "services.renderer.image=<IMMUTABLE_RENDERER_IMAGE>",
  "services.renderer.command=<JSON_RENDERER_COMMAND_LIST>",
  "wizard.log_dir=<RUN_DIR>", "wizard.run_method=NONE"
]
```

Driver argv: `["uv", "run", "--no-sync", "-m", "alpasim_driver.main",
"--config-path=<RUN_DIR>", "--config-name=driver-config.yaml"]`.
Compose argv: `["docker", "compose", "up", "--exit-code-from", "runtime-0"]`,
with working directory `<RUN_DIR>`. `--no-sync` assumes the separate resolved
runtime environment has already been installed; it is not environment setup.
Explicit driver endpoint selection prevents an inherited
`ALPASIM_DRIVER_PORT`/`ALPASIM_DRIVER_HOST` value changing the connection.

The managed-service allocator starts at `driver_port + 1`, preventing it from
assigning the external driver's port while that driver has not started yet.
The pinned [allocator](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/context.py#L218-L259)
only advances and searches at most 1,000 ports per allocation. This dev route
allocates three managed RPC endpoints, then constructs
[`SlurmDeployment`](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/deployment/slurm.py#L27-L34),
which allocates three more even for `NONE`. The supported maximum
`65,535 − 6 × 1,000 = 59,535` keeps all six search windows inside the TCP port
range. Host availability and races with other processes remain unverified; this
technical routing guard is not a socket reservation or runtime qualification.

This host driver invocation is assembled from the base container command,
generated config filename and driver Hydra entrypoint; it is not an upstream
documented end-to-end host launch example. Absolute config/output paths adapt
the container's `/mnt/output` to the external host process.

Argv lists protect the preparer-to-wizard process boundary only. Upstream
[`services.py`](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/services.py#L200-L221)
joins service command strings, and the
[Compose writer](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/deployment/docker_compose.py#L130-L144)
runs them through `bash -c`. Inspect the emitted configuration before any
authorized execution; the pure builder does not qualify arbitrary downstream
shell input.

The explicit `external_image=true` overrides on physics, controller and runtime
prevent the
[Compose writer](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/alpasim_wizard/deployment/docker_compose.py#L120-L128)
from attaching an automatic build recipe and a digest reference as a build tag.
Renderer already sets this flag in base source. The external driver is not a
managed container; traffic simulation is skipped by the selected topology. This
selects prebuilt immutable images, but does not prove they are cached: the source
pull policy still permits fetching a missing image during separately authorized
setup.

## Harmonizer and effective controls

Use the same dev configuration for both arms, changing only the Harmonizer flag.
The base [renderer command](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/base_config.yaml#L133-L146)
includes `--enable-harmonizer`; the
[EC2 source comment and command](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/e2e_challenge/ec2.yaml#L32-L48)
identify omission as the unharmonized path. Do not select the EC2 preset for the
off arm: it also changes topology, renderer worker count and render bundling.
The following shared command uses constants resolved from dev/base/1gpu source,
plus its documented `--enable-timing` option, required by the existing S2
engineering pilot specification. Insert `"--enable-harmonizer"` immediately after
`"--no-enable-nrend"` for on:

```json
[
  "/app/internal/scripts/pycena/runtime/pycena_nrm_full", "serve-grpc",
  "--port={port}", "--host=0.0.0.0",
  "--artifact-glob=/mnt/nre-data/{sceneset}/**/*.usdz",
  "--egocar-hood-dir=/mnt/ego-hoods", "--no-enable-nrend",
  "--download-cache-dir /tmp/nre-cache-dir", "--cache-size=5",
  "--max-workers=4", "--enable-editing-actors", "--enable-timing"
]
```

`{port}` and `{sceneset}` are upstream wizard substitutions, not user placeholder
syntax. The renderer command list contains one source string with a space for
`--download-cache-dir`; upstream constructs the container command from this list.
Actual image behavior, Harmonizer loading and absence of unexpected fetches still
need qualified host evidence. Image source/tag metadata does not establish
checkpoint content or whether dependencies are bundled.

| Control | Pinned source behavior to preserve and record |
|---|---|
| Rendered cameras | Six: front-wide, front-tele, cross-left, cross-right, rear-left-70fov, rear-right-70fov; the dev comment saying four is stale |
| Camera requests | Height 1080, width 1900, frame interval 100,000 µs, shutter 30,000 µs; request dimensions are not proof of delivered pixel dimensions |
| Driver input | Four cameras: cross-left, front-wide, cross-right, front-tele; four temporal frames; batch one; subsample factor one |
| Policy path | Stock Alpamayo 1.5 BF16, SDPA, CFG navigation off, trajectory optimizer off |
| Control and warm-up | 100,000 µs tick; zero planner delay; 1,700,000 µs forced-GT context; driver skipped during forced-GT context |
| Simulator | Render bundling `NONE`; route-start offset 40 m; `EGO_ONLY` physics; traffic service skipped/replayed traffic; recording ground truth not sent to driver |
| Concurrency | One GPU topology, one runtime worker, driver endpoint concurrency one; this is not evidence of memory fit |
| Repetition | One rollout per selected scene/arm; unseeded stock-driver replicates |
| Expert graph | Off in the inspected stock path; no invented Hydra switch |

Sources: [six-camera configuration](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/cameras/6cam_1080.yaml),
[Alpamayo driver configuration](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/driver/alpamayo1_5.yaml),
[shared driver controls](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/driver/alpamayo_configs.yaml),
[one-GPU topology](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/topology/1gpu.yaml),
[base controls](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/wizard/configs/base_config.yaml#L272-L320),
[model wrapper](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/driver/src/alpasim_driver/models/alpamayo1_5_model.py).
At the pinned model dependency the
[graph field starts as `None`](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/alpamayo1_5.py#L138-L159)
and requires explicit enabling, absent from the stock wrapper. The inspected
driver schema exposes neither an expert-graph switch nor `force_determinism`.
Recording a false determinism label is not a source setting or a seeded run.

`wizard.nr_retries=0` and `runtime.enable_autoresume=false` make those selected
controls explicit; they do not prove every service is free of internal retries.
All attempts, internal restarts and lost work still need capture and accounting.
A successful retry cannot erase its failed predecessor or select itself into
current inference.

## Runtime and asset boundary

The pinned [onboarding requirements](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/docs/ONBOARDING.md)
name uv ≥0.9.17, Rust, Docker Engine/Compose/buildx, NVIDIA Container Toolkit,
CUDA 12.8 compatibility and the 570.x-or-later driver family for NRE. These are
upstream requirements, not observations of a delivered host.
The [driver project](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/src/driver/pyproject.toml)
requires Python ≥3.12,<3.13 and declares unversioned Git sources including
`alpamayo1_5`. The complete pinned tree contains no `uv.lock`. The CPU evaluator's
lock cannot resolve this runtime; archive a separately resolved runtime lock
whose actual Git dependencies match the reviewed candidates.

The upstream [Dockerfile](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/Dockerfile)
uses a tagged CUDA base, mutable `uv:latest` and dependency resolution during its
build. A source commit therefore does not identify a reproducible image by itself.
Supply the actual immutable base and renderer image references, preserve platform
and fetched digests, and capture built environment identities. Compile the
upstream protobuf packages as its setup instructions require; the CPU ASL
descriptor is not a substitute runtime installation.

The stock wrapper calls teacher `from_pretrained` without a revision argument;
use the verified local snapshot to avoid silently selecting a later model head.
This alone does not make all loads offline. The pinned
[processor helper](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/helper.py#L190-L199)
names `Qwen/Qwen3-VL-2B-Instruct`, and the
[base model](https://github.com/NVlabs/alpamayo1.5/blob/36aeb4c5938cbc2eb2aed33b22434773da4ab639/src/alpamayo1_5/models/base_model.py#L248-L266)
also resolves `vlm_name_or_path` from model configuration. Inspect the exact
checkpoint config and record all needed config/tokenizer/processor files before
authorizing acquisition. Do not infer an additional full VLM weight download
from a config/processor load. The earlier
[asset audit](2026-10-31-s2-license-approval.md) retains the separate unresolved
Harmonizer checkpoint/backbone and conditional loader dependencies; neither
Cosmos-Predict2 nor Cosmos-Reason2 becomes an unconditional allowlist entry here.

Remaining inputs are explicit in
[`S2_SETUP_DRAFT.json`](../../experiments/S2_SETUP_DRAFT.json): registered
five-scene/road/stratum mapping and prediction; exact authorized asset files,
revisions, hashes and use terms; runtime lock and generated configuration;
delivered hardware, image/driver/Harmonizer qualification; actual raw-capture and
scorer checks; quote, concrete action authority and bounded lifecycle. Public
source preparation resolves none of those by assumption.
