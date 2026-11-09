# Minimal S2 pilot preparation

Agent-authored, 2026-11-07. Serves **S2.1/S2.2/S2.3/S2.7 [C], G1/G2/G3·R1/G7**. This is the reduced preparation path approved in this conversation; the 14-package specification remains a backlog. Plan v5/D-022 and recorded action approvals govern execution.

The immediate deliverable is a checked CPU rehearsal, a setup inventory, and a bounded pilot packet. The scientific pilot remains five scenes × Harmonizer off/on × one replicate = **ten jobs**; retries and diagnostic runs are additional accounted attempts/work. Pilot approval never releases the 2,646-job full study.

## 1. Run the CPU rehearsal

The full synthetic rehearsal of the whole pipeline (CPU-13, D-024) is `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/rehearse_s2.py --fixture complete --output evidence/s2-rehearsal-UNIQUE-ID`. The reduced pilot rehearsal below remains for the ten-job packet path.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/rehearse_s2_pilot.py \
  --output evidence/s2-pilot-cpu-UNIQUE-ID
```

Use a fresh output name. Review its report, source identities, synthetic ten-job accounting, independent rejection controls and export checks. This command incurs no cloud charge and loads no model. Read [S2 ingestion](S2_INGESTION.md) for its precise limitations: it consumes a PhysLoop synthetic capture contract, not actual AlpaSim ASL, and does not recompute the summary from raw metrics.

The command runs the fixture, full CPU suite and Ruff from its included source snapshot. `cpu_checks_passed` is separate from `cpu_handoff_ready`; exit 2 preserves a lock/declaration/installed-version mismatch even when tests pass. The later ingestion delivery resolved PyArrow 25.0.1 in `uv.lock`, matching the installed version; the original pilot evidence retains its historical mismatch. A clean `uv sync --frozen` in a new environment is still unverified in that delivery. Matching installed versions alone do not establish a fresh rebuild. The bounded follow-up contract is [S2 CPU handoff](S2_CPU_HANDOFF_SPEC.md).

`experiments/S2_SETUP_DRAFT.json` is an **engineering inventory**, not a new scientific protocol schema. Null fields and empty allowlists remain unresolved. `experiments/S2_PILOT_DRAFT.json` remains the separate scientific schema-v2 draft; a populated inventory cannot approve it.

## 2. Resolve the setup packet before ordering

| Item | Prepared fact | Required completion |
|---|---|---|
| Simulator/evaluator | Retained `e2e_challenge` commit `e121e59141dc391f6c295513d6dadc840a419149`; November 7 official head differs only in NuPlan reference data/docs/tests | Repeat source check at admission; qualify any changed runtime pin separately |
| Model implementation | Audited alpamayo1.5 commit `36aeb4c5938cbc2eb2aed33b22434773da4ab639` | Resolve actual driver dependency and archive its runtime lock |
| Teacher | Candidate model revision `7aba8293c09993f2e125c6819df05d7fa3e873ea` | Exact file list, bytes/hashes, applicable use terms and loader dependencies |
| Renderer | Historical NRE `26.04` catalog digest candidate in inventory | Actual platform-specific fetched image identity and governing terms |
| Harmonizer/processor | Named candidate dependencies in inventory | Immutable revisions, required files and whether NRE fetches them separately |
| Five scenes | Count fixed; identities intentionally absent | Authorized metadata, registered scene/road/stratum mapping and exact artifact allowlist |
| Runtime | Source-checked Linux amd64 Compose/stock external-driver templates with explicit five-scene selection and Harmonizer controls | Actual immutable images/runtime lock, registered inputs, resolved configs and effective host settings |
| Host and spending | Adopted nested $100/$1,200/$2,500 ceilings | Current itemized quote, exact resources, concrete action scope, cash/usage/liability ledger, cutoff and retention |

The local `uv.lock` belongs to the CPU evaluator. The audited AlpaSim source has no committed runtime lock; resolve and archive that environment separately. Old host image IDs cannot identify images on a new machine. The host-probe tools must receive the run's validated immutable image reference.

The later [CPU handoff evidence](../evidence/s2-cpu-handoff-20261107-v1/verification.json)
records 1,228 passing tests and matching direct dependency versions. Its clean
rebuild remains explicitly unverified. The subsequent
[pre-rental hardening record](../experiments/2026-11-07-s2-prerental-hardening.md)
tracks the ordered-packet, capture and exporter fixes and their final verification.
Historical evidence below is retained with its original status.

Historical pilot implementation evidence: [the CPU verification record](../evidence/s2-pilot-cpu-20261107-v1/verification.json) records **1,046 passing tests**, successful Ruff and ten-job rehearsal from a frozen source copy. It deliberately reports `cpu_handoff_ready=false` because of that snapshot's dependency lock mismatch. [The engineering record](../experiments/2026-11-07-s2-minimal-pilot-preparation.md) records scope, known answers, review corrections and limitations. The later ingestion lock correction does not rewrite this evidence or establish a fresh rebuild.

After authorized setup has pre-staged the images, invoke the monitoring probes as `scripts/vast_monitor_control.sh OUT WORK CUDA_IMAGE_REF` and `scripts/vast_exporter_control.sh OUT WORK CUDA_IMAGE_REF`. `CUDA_IMAGE_REF` must be a pullable `repository@sha256:...` reference already cached on that host; the probes verify its digest and execute its local ID. Pre-stage the pinned exporter image in the inventory too. Probes do not pull images. Exporter refuses an active DCGM/`nv-hostengine` conflict and does not stop host services; resolve any conflict separately within the recorded setup authority.

Do not fill unknown asset sizes with zero. Disk capacity must include images, weights, five scenes, caches, extraction scratch, raw evidence and temporary export copies. Conditional Cosmos dependencies remain outside the download allowlist until the selected loader/image establishes that they are needed. No raw PAI camera/lidar corpus or Harmonizer training dataset is required by this packet.

Sources: [November 7 launch-source audit](research/2026-11-07-pilot-launch-audit.md), [earlier upstream audit](research/2026-10-31-s2-upstream.md), [asset/access audit](research/2026-10-31-s2-license-approval.md), [rental packet](S2_RENTAL_APPROVAL.md). Source inspections are not fresh quotes, current terms acceptance or runtime qualifications.

## 3. Upstream setup and launch boundary

Use the pinned [onboarding instructions](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/docs/ONBOARDING.md) and [local evaluation route](https://github.com/NVlabs/alpasim/blob/e121e59141dc391f6c295513d6dadc840a419149/e2e_challenge/local_evaluation/README.md). The audited route requires an external driver, prepared assets and GPU services. Its `dev` preset inherits renderer settings that differ from `ec2`.

The documented `+nurec_scenes=curated_val` example selects the entire **441-scene** set. The [source-checked packet](research/2026-11-07-pilot-launch-audit.md) instead requires exactly five explicit IDs, `scenes.test_suite_id=null`, `scenes.scene_ids=[...]` and `scenes.limit_to_first_n=0`. It defines the separate driver startup and identical dev renderer commands with/without `--enable-harmonizer`. The pure command builder emits argv lists; it neither launches them nor validates actual host assets. The inventory retains `pilot_argv=null` because registered scenes/horizon and concrete host/image inputs remain unresolved. Its `request_draft` names each required input without supplying invented values.

Do not invoke the upstream wizard as a CPU-only check: both `run_method=NONE` and `dry_run` still resolve/fetch scene assets before service deployment. `NONE` is suitable only during authorized setup. Two five-scene arm templates give ten requested jobs, not a registered execution order; source sorting and arm grouping do not satisfy interleaving/randomization requirements by themselves.

Use `python -B -m gate.s2 prepare-pilot PACKET_REQUEST.json` to bind the exact
launch/schedule inputs to all ten job IDs in a supplied order. The request and
checks are specified in [S2 CPU handoff §E5](S2_CPU_HANDOFF_SPEC.md#e5--pre-rental-integration-and-hardening).
The packet emits one scene per unit with distinct output directories; it never
launches a command or authenticates scientific registration. Run units serially
after admission, stopping the previous Compose services and external driver
before reusing the port. Account for a fresh driver/service startup per unit.
Preserve failed attempts and bind actual UUIDs from observed execution records.

The preparer intentionally accepts only portable ASCII paths and scene IDs. Upstream later joins service commands into `bash -c`, so an argv list is not a qualification of arbitrary downstream shell input. Physics/controller/runtime explicitly use prebuilt external images to avoid an automatic build recipe for the immutable base image; renderer already does. Preserve and review the emitted configs on the admitted host. The [source fixture manifest](../tests/fixtures/alpasim_launch_upstream/manifest.json) binds the inspected public files, not the actual runtime or licensed asset bytes.

Collect the complete actual argv and resolved configuration before scientific execution. Record the six-camera simulation request separately from the driver's four consumed cameras; the audited dev comment saying four rendered cameras is stale. As configured, this mismatch is predicted to fail every rollout (the stock driver rejects unconfigured cameras; lane R2 L1): the unit runner now blocks before any GPU command until Huy's camera fix makes the generated renderer camera list equal the driver's `use_cameras`. Keep the stock graph-off path, batch one and the actual unseeded classification unless an independently qualified registered protocol specifies otherwise. This driver exposes no `force_determinism` or graph Hydra knob; a label does not implement one. A disabled wizard retry count/autoresume does not prove that every service has no internal retries.

## 4. Bounded execution after recorded authority

Updated 2026-11-09 for the D-024 build. Every command below exists and was checked against its `--help`; none is runnable until the owner inputs exist. `CTRL` is the controller (this Mac, repository root, `.venv/bin/python -B`); `GUEST` is the rented VM (inside the unpacked bundle, `uv run --frozen --no-dev python`). Synthetic success never qualifies a host.

**How authority works now.** Executing paths do not read prose. A decision entry in `docs/DECISIONS.md` authorizes an artifact only when it contains the exact line printed by:

```sh
.venv/bin/python -B -m gate.s2 authority-line --kind KIND ARTIFACT.json
```

for `KIND` in `s2_action_record`, `s2_bootstrap_plan`, `s2_pilot_packet` (or `s2_full_sweep_packet`), `s2_readiness_report`, `s2_full_release`, `s4_bootstrap_plan`, `s4_launch_plan` and `s4_registration`. Licence and acquisition steps need lines of the form `records: terms:ROLE` and `records: acquire:ROLE`; the guest bootstrap prints the exact set it needs as `authority_request` when run without `--execute`. Huy writes these lines into his own new decision entry when he approves; agents never write them. A decision closed by a later `closes_authority: D-NNN` line, or D-019 (closed by D-020), authorizes nothing. The real log authorizes nothing today.

1. **Before create (CTRL).**
   - Rehearse: `.venv/bin/python -B scripts/rehearse_s2.py --fixture complete --output evidence/s2-rehearsal-NEW-ID`.
   - Record the owner decisions in [the completeness audit's open list](research/2026-11-08-codebase-build/completeness-audit.md#owner-inputs-still-open), led by the camera fix (§3), the Harmonizer source, the licences (now including `nvidia/Cosmos-Reason2-8B`, required by the teacher config) and the pilot registration.
   - Admit the action record: `.venv/bin/python -B scripts/s2_lifecycle.py validate-action-record ACTION.json`. The record names one provider (only Vast is implemented), exact resources (disk at least 300 GB), the maximum rate, an absolute window of at most 12 hours, retention and the export destination.
   - Compile the plan: `.venv/bin/python -B -m gate.s2 compile-bootstrap INVENTORY.json ACTION.json --phases A B C D E`.
   - Prepare the packet: `.venv/bin/python -B -m gate.s2 prepare-pilot PACKET_REQUEST.json`.
   - Build the readiness report: `.venv/bin/python -B -m gate.s2 readiness REQUEST.json --root ROOT`.
   - Print the authority lines for the action record, plan, packet and readiness report. Huy writes them, plus the `records:` lines, into one new decision entry.
2. **Create, bind and arm (CTRL).**
   - Get an intent ID: `scripts/s2_lifecycle.py intent-id`.
   - Create: `scripts/s2_lifecycle.py step --journal J --record ACTION.json --intent-id ID --watchdog-events W --execute --authority-ref D-NNN`. It refuses unless that decision carries the record's hash line; an uncertain create is reconciled by its unique label and never repeated.
   - Arm the watchdog: `scripts/vast_stop_watchdog.py --instance-id N --expected-label LABEL --deadline-utc T --max-window-seconds S --events W --first-arm`. `S` equals the approved window. On a restart use `--prior-events W` instead of `--first-arm`; the deadline is never renewed.
   - Then run `scripts/s2_lifecycle.py observe-watchdog ...`. No bootstrap starts without the durable `armed` receipt.
3. **Watchdog scope.** It is stop-only and external, not a provider-enforced cap. After the deadline it stops by exact ID even when the inventory fails, and exits 3 (`stop_unconfirmed`) if a stop is never confirmed. Storage stays billable after stop.
4. **Check the delivered host (GUEST then CTRL).**
   - Seal the source on CTRL: `.venv/bin/python -B scripts/s2_prepare_bundle.py build --root . --out-dir OUT --scope actual`. Verify it after transfer with `scripts/s2_prepare_bundle.py verify --bundle-sha256 H ARCHIVE`.
   - Phase A runs on the guest's system Python 3.10: `python3 scripts/s2_guest_bootstrap.py PLAN --action-record ACTION.json --inventory INVENTORY.json --journal-dir JD --decision-id D-NNN --decisions DECISIONS.md --execute`.
   - Phase A installs the pinned uv and the uv-managed CPython 3.12, runs the host probe (`bash scripts/vast_s2_host_probe.sh /abs/new/evidence-dir REPOSITORY@sha256:DIGEST /abs/data-root`, digest-only pulls) and the host-fit check against the action record (`gate/s2_host_fit.py`).
   - A shortfall stops without a replacement host. XID delivery remains unverified unless qualified or accepted by amendment.
5. **Acquire permitted assets (GUEST, phases D–E).** Gated stages compile only with their `records:` lines. Downloads are pinned, and every file gets a byte and SHA-256 receipt. The HF token and the NGC key come from `--secret-file NAME=PATH` and are redacted. Packaging larger than priced stops acquisition.
6. **Qualify actual input.** Run any declared diagnostic qualification unit first, before the ten jobs and outside them. For example: `uv run --frozen --no-dev python scripts/s2_run_units.py PACKET --authority A --run-config RC --budget-file B --journal-dir JD --unit-selection diagnostics_only --clock-uncertainty-us U --decisions DECISIONS.md --decision-id D-NNN --execute`. This needs `engineering_only_diagnostic: true` in the decision. The runner blocks before any GPU command if the generated renderer cameras differ from the driver's `use_cameras`, or if Compose differs from the packet expectations.
7. **Run ten jobs (GUEST).**
   - Run the same command with `--unit-selection all --readiness-report R`.
   - Before each unit, headroom is recomputed from the cost ledger; a full sweep additionally needs `--release-record` and `--release-decision-id`.
   - Captures happen before teardown. Internal retries become separate attempts. Exit 0 is never success evidence. No retry is selected.
   - Optional telemetry runs alongside: `scripts/s2_gpu_sampler.py OUT --scope actual --device-key uuid --device ID --clock-id C --interval-s I --duration-s D --timeout-s T` and `scripts/s2_dcgm_scraper.py OUT --scope actual --clock-id C --interval-s I --duration-s D --timeout-s T`.
8. **Export and verify.**
   - Guest manifest: `scripts/s2_guest_manifest.py ROOT --scope actual --output M`.
   - After the copy, verify on CTRL and record the export with `scripts/s2_lifecycle.py observe ... --observation OBS --export-root EXPORT`. The verification report is re-checked against the bytes on disk.
   - Seal: `.venv/bin/python -B -m gate.s2 seal INDEX --root EXPORT`, then `verify-bundle --root EXPORT`.
   - A transport exit code alone is never success.
9. **Stop, destroy and reconcile (CTRL).**
   - Run `scripts/s2_lifecycle.py step ... --execute --authority-ref D-NNN` until the journal reaches `absent_verified`.
   - An armed instance is never destroyed without a verified export: it is held under retention or escalated to manual review. Absence needs exact-ID NotFound twice.
   - Reconcile: `scripts/s2_lifecycle.py charges ... --execute --authority-ref D-NNN`, then `reconcile-cost --events E --quote Q --scope S`. Keep cash, usage, credit and liabilities separate, with settlement pending until the invoice.
10. **At a hard cutoff.** Stop GPU work even if export is incomplete. Retain storage only within the retention authority, and report the incomplete export. Never extend time, add resources, create another VM or release the full study automatically.

## 5. Handoff after the pilot

Produce measured fit, complete-job throughput, failures, raw-input qualification and all-in cost evidence. Huy chooses/finalizes the scientific full-study protocol without selecting favorable scenes/margins from pilot scores; agents prepare its measured forecast and unresolved blockers. Full-study funds need their own release. Reliability, population/contamination and operational-power work are required before the corresponding full-study/S4 decisions, not all as new reusable modules before this pilot.

An engineering pass means the tested preparation works on the named fixtures or host. It does not establish simulator reliability, real-world validation, publication rights or S4 admission. No rental, license action, gated download or scientific run is performed by this runbook.

## 6. Rebuild the CPU environment separately

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/verify_clean_s2_environment.py \
  --output evidence/s2-clean-rebuild-UNIQUE-ID --uv /absolute/path/to/uv \
  --python /absolute/path/to/python3.12
```

This uses a new environment and empty cache, and fetches only the public CPU
dependencies in the frozen lock. It does not download Python, models or datasets.
Read the result before treating the rebuild as verified: a failing installer
does not run fresh-environment tests or become a pass because the existing venv
works. The 11-07 uv 0.8.17 SystemConfiguration failure was environmental: the same
binary passed a clean frozen rebuild on 2026-11-08 (`evidence/s2-clean-rebuild-20261108-v1/`).
The D-024 lock changes (`s4` group, PyYAML) need the final D-024 rebuild record; read
its verification.json before calling the current lock reproducible from scratch.
