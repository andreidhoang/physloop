# Minimal S2 pilot preparation

Agent-authored, 2026-11-07. Serves **S2.1/S2.2/S2.3/S2.7 [C], G1/G2/G3·R1/G7**. This is the reduced preparation path approved in this conversation; the 14-package specification remains a backlog. Plan v5/D-022 and recorded action approvals govern execution.

The immediate deliverable is a checked CPU rehearsal, a setup inventory, and a bounded pilot packet. The scientific pilot remains five scenes × Harmonizer off/on × one replicate = **ten jobs**; retries and diagnostic runs are additional accounted attempts/work. Pilot approval never releases the 2,646-job full study.

## 1. Run the CPU rehearsal

From the repository root, after the reduced-scope implementation is integrated:

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

Collect the complete actual argv and resolved configuration before scientific execution. Record the six-camera simulation request separately from the driver's four consumed cameras; the audited dev comment saying four rendered cameras is stale. Keep the stock graph-off path, batch one and the actual unseeded classification unless an independently qualified registered protocol specifies otherwise. This driver exposes no `force_determinism` or graph Hydra knob; a label does not implement one. A disabled wizard retry count/autoresume does not prove that every service has no internal retries.

## 4. Bounded execution after recorded authority

1. **Before create:** inspect the current exact quote and action record, approved asset/private-host access arrangement, scientific registration/prediction, cutoff, retention and export destination. Pending scientific fields do not prevent CPU work; they block dependent runs. D-019's old allocation is closed.
2. **Bind and arm:** create only the authorized allocation, record its unique label and exact ID, persist the original creation-request time/absolute cutoff, and wait for the external watchdog's durable `armed` receipt before bootstrap. If arming fails, stop via the control plane. An uncertain create response requires inventory reconciliation, never an automatic second rental.
3. **Respect watchdog scope:** `scripts/vast_stop_watchdog.py` supports Vast and at most 60 minutes ahead. It stops only; storage can remain billable. It is not a provider-enforced cap. A longer approved pilot or another provider needs its own tested cutoff mechanism before create. Restarting must not renew the original allowance.
4. **Check delivered resources:** GPU edition/VRAM, guest RAM/CPU/disk, driver, admin access, Docker/Compose/buildx and runtime visibility. Run the existing public synthetic checks within the allowance. Preserve missing XID/profiler status; a successful scrape does not establish fault delivery. Resolve mandatory monitoring failures or record their explicit acceptance amendment.
5. **Acquire permitted assets:** only the approved exact subset, after terms/access and download authority. Record fetched bytes, immutable revisions and first delivery for restricted datasets. New dependencies or larger packaging stop acquisition until reviewed. Keep provider credentials off the guest and secrets out of command/profiler logs.
6. **Qualify actual input:** verify simultaneous teacher/renderer/Harmonizer fit, actual raw capture, evaluator path, required metric coverage, requested/actual horizon and failure preservation. The synthetic ingestion pass does not satisfy this. Declare/budget extra diagnostic jobs separately; preserve any pilot job used for qualification with its role explicit.
7. **Run ten jobs:** only after scientific preflight and actual evidence checks pass. Record every attempt, including pre-UUID crashes. No successful-retry selection rule is currently admitted; a retry is preserved and blocks current inference. Include setup/acquisition/warm-up/idle/retry/tail time in paid usage. Check next-unit cost and cleanup headroom before each bounded unit.
8. **Export and verify:** incrementally copy permitted evidence to the private destination; compare full relative file set, lengths and SHA-256 hashes against the source manifest. Seal only after the required files and local verification pass. Preserve partial-export status on failure; do not mark success from the transport exit alone.
9. **Stop, destroy, reconcile:** stop the exact bound task resource; verify its structured terminal state, then destroy only authorized task compute/storage after verified export on the normal path. Verify absence from compute and volume inventories. Failed inventory/API calls never prove absence. Record settlement pending separately from final charges.
10. **At a hard cutoff:** stop GPU work even if export is incomplete. Retain storage only within the recorded retention authority; report the incomplete export and any evidence-loss risk. Do not extend time, increase resources, create another VM or release the full study automatically.

Keep cash paid, accrued usage, remaining credit and liabilities separate. A credit deposit followed by consumption is one cash payment. Stop is not deletion; retained storage, transfer and unsettled charges still count. Preserve teardown headroom and any recorded independent-evaluation reserve.

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
works. On this host, the installed uv 0.8.17 has a macOS SystemConfiguration
initialization failure; the diagnosis and retained attempts are separate from
the source test results. Use a working installer/network environment to close
that boundary before calling the CPU preparation reproducible from scratch.
