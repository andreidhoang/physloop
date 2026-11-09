"""Pure command/schedule binding for the five-scene off/on S2 pilot and the full sweep.

F1 (build-spec B2): ``build_full_sweep_packet`` binds a registered 441-scene list,
replicate count (2 arms x 441 x 3 = 2,646 units at three replicates) and execution order
-- all required owner inputs with no defaults -- to the same serial unit shape, with
distinct per-unit directories and driver ports. The pilot path below is unchanged.

Tickets: S2 CPU handoff E5; build-spec A4 R2 L6 (ego-hood staging receipt), L7
(``wizard.nr_retries`` is a no-op here), P1 (a unit binds a list of upstream
attempts), P2 (``run_uuid`` from the unit's run_metadata.yaml), P3 (captures
required before Compose teardown), P5 (inherits L1-L5, L8 through the launch
command group) and the required ``shm_size`` field. Serves S2.1-S2.7 [C].

The caller supplies both requests and all ten job IDs in an explicit order.
Preparation binds declared inputs, not their origin or scientific registration.
It performs no reads (apart from hashing its own source once at import for
producer identity), launches, asset retrieval, UUID inference or approval. The
packet ``scope`` is a required input equal to ``launch_request.scope``; an
``actual`` packet requires an ``actual`` ego-hood staging receipt and any other
scope must not carry one. Each unit's check expectations carry that scope and
the renderer mounts its argv requests (ego hoods; pinned Harmonizer host_dir).
On-arm units need the launch request's recorded ``terms:harmonizer`` reference
(review A2-F12); without it no unit is prepared (``harmonizer_terms_unrecorded``).

Pinned source e121e59141dc391f6c295513d6dadc840a419149 supports a singleton
scene list (wizard schema.py/context.py), writes Compose/configs for run method
NONE (wizard.py), and requires a fresh driver config path/output for each unit.
The source fixtures and their Git/SHA256 manifests are retained under
tests/fixtures/alpasim_launch_upstream. Actual host execution is unqualified.

These units reuse the external port and host network. They require strictly
serial execution, prior Compose/driver teardown, and a new driver per unit.
All ten startup periods are paid work; no measured cost or efficiency follows.
Exit codes are never success evidence: the unit's attempts are reconciled later
by ``gate.s2_capture.reconcile_unit_attempts`` from captured logs, folders and
summary rows, and a runtime-internal retry yields a second attempt in one unit.

Verification level: ``build_pilot_packet`` L1 binding logic, L2 argv/paths
against pinned source; capture/teardown argv are Docker CLI templates (L3,
R2 F5/F6). ``shm_size`` is a required owner/technical input with no default:
``docker_default`` (no key; Docker's 64 MiB) or a Docker size such as ``64g``
(NRE documents ``--shm-size=64g``, grpc_api_guide_2604.html.txt:65). The
generator never writes ``shm_size`` (docker_compose.py:96-173); any non-default
value needs an operator Compose override and is checked by
``gate.s2_launch.check_generated_compose`` on the effective configuration.

Diagnostic qualification units (completeness audit, runbook step 6 gap; readiness
spec section 9 item 7: "any extra diagnostic rollout must be declared and budgeted
separately from the ten pilot jobs"). The pilot request may carry an optional
``diagnostic_units`` list, for example the camera/config check lane R2 recommends before
the pilot. Each declaration is exactly ``{diagnostic_id, role, source_job_id,
budget_line: {line_id, cap_usd}}``: it repeats the commands of one scheduled job
(``source_job_id``) under its own job ID ``diagnostic-<diagnostic_id>`` with phase
``diagnostic_qualification`` (``DIAGNOSTIC_PHASE``, review G2-04) and its own
``<arm run_dir>/diagnostic-<id>`` and ``<arm driver_log_dir>/diagnostic-<id>``
directories. Diagnostic units live only in ``packet["diagnostic_units"]`` (each with a
``diagnostic`` block: role, budget line, ``counts_toward_pilot_jobs: false``,
``inference_input: false``), never in ``units`` or ``schedule``, so they are never one
of the ten pilot jobs and never an inference input; they run before the pilot units and
are recorded in their own ledger (``scripts/s2_run_units.py``). The role, the budget line
and its cap are declared inputs with no default (a cap is a budget value, Huy's). When
the key is absent the packet is byte-identical to a packet built before diagnostic
support (``tests/test_s2_packet.py`` pins its identity hash, producer excluded); an empty
list is refused so "none declared" has exactly one spelling. The full-sweep request does
not accept diagnostic units (its field set is exact). Verification level: L1.

This module never executes, reads files or environment, chooses an owner
decision, verifies staged bytes on a host or selects among retried attempts.
"""

import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.s2_launch import (
    CAMERA_PRESETS,
    COMPOSE_SERVICES,
    EGO_HOOD_CONTAINER_DIR,
    HARMONIZER_CONTAINER_DIR,
    SCOPES,
    _command_group,
    _overlap,
    build_pilot_launch,
    check_scene_ids,
    check_sweep_launch,
    docker_bytes,
    sweep_unit_request,
)
from gate.s2_schedule import build_schedule

SCHEMA_VERSION = 2
_FIELDS = {"schema_version", "scope", "launch_request", "schedule_request", "execution_order",
           "ego_hood_staging", "shm_size"}
# Optional pilot request field (declared diagnostic qualification units); see the docstring.
DIAGNOSTIC_FIELD = "diagnostic_units"
DIAGNOSTIC_JOB_PREFIX = "diagnostic-"
# Review G2-04: a diagnostic job's ledger phase (gate.s2_ledger PHASES), so its own ledger
# never carries the source pilot job's ``harmonizer_pilot``.
DIAGNOSTIC_PHASE = "diagnostic_qualification"
_DIAGNOSTIC_FIELDS = {"diagnostic_id", "role", "source_job_id", "budget_line"}
_DIAGNOSTIC_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_DIAGNOSTIC_ROLE = re.compile(r"[a-z][a-z0-9_]{2,63}")
_BUDGET_LINE_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_POSITIVE_USD = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:\.[0-9]{1,6})?")


def _read_source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


# Hashed once at import so packet preparation itself performs no file I/O.
_SOURCE_SHA256 = _read_source_sha256()
_SCHEDULE_FIELDS = {
    "study_id", "phase", "scenes", "policies", "protocol_hashes", "replicates",
    "requested_steps",
}
# Pinned ego hoods: AlpaSim e121e591 tree data/nre-artifacts/ego-hoods/ (git ls-tree -l;
# regular blobs, not LFS). defines.sensordata = <filesystem>/nre-artifacts
# (base_config.yaml:31) mounts <sensordata>/ego-hoods at /mnt/ego-hoods (:126); the
# runtime selects rig hyperion_8_1 (:287). Without staging, the dev preset does not
# validate mounts (dev.yaml:23) and rendering proceeds without a hood (R2 L6).
EGO_HOOD_FILES = (
    {"path": "hyperion_8/camera_front_wide_120fov.png", "size_bytes": 808_775,
     "git_blob_sha1": "1e4cbd9f0a023f3cd3578b707380e61a9d4a39ca",
     "sha256": "04fa3f354940d03984a37eb755ebf845be4b4bdae39c178b8de990d0b045392d"},
    {"path": "hyperion_8_1/camera_front_wide_120fov.png", "size_bytes": 425_624,
     "git_blob_sha1": "1a07fea50bf774883fc3a7b9df8555d97b9dae98",
     "sha256": "b7c36c0d179e82cc0d820a202fb9b0017e4006aa815cc850fbd3ea1848d9a8cd"},
)
_STAGING_FIELDS = {"schema_version", "artifact_kind", "scope", "staged_root", "files"}
# L7 and P1 documentation, cited to pinned source.
RETRY_CONTROLS = {
    "wizard.nr_retries": {
        "value": 0, "effect_on_run_method_NONE": "none",
        "source": "deployment/slurm.py:66-102 is the only reader (SlurmDeployment launch)",
    },
    "runtime_internal_retry": {
        "max_retry_attempts": 1, "configurable": False,
        "trigger": "RetryableRendererInfrastructureError",
        "new_rollout_uuid_per_retry": True,
        "source": ("worker/ipc.py:50; worker/main.py:166; daemon/scheduler.py:181-213 "
                   "(session_uuid='' at :199 allocates a fresh rollout UUID)"),
        "evidence": "Compose stdout 'Retrying rollout after retryable infrastructure failure'",
    },
    "runtime.enable_autoresume": {
        "value": False,
        "reason": "true deletes rollout folders lacking _complete (autoresume.py:59-76)",
    },
}
COMPOSE_PULL_NEVER = {
    "argv_flag": ["--pull", "never"], "support": "qualify_on_host",
    "reason": "generated services default to pull_policy missing (schema.py:157)",
    "fallback_requirement": ("If the host Compose lacks 'up --pull', the effective config "
                             "must set pull_policy: never for every service before up."),
}


def _issue(report, code, path, message, severity="blocking"):
    report["issues"].append({"code": code, "path": path, "severity": severity,
                             "evidence_refs": [], "message": message})


def _validate_staging(staging, data_root):
    if type(staging) is not dict or set(staging) != _STAGING_FIELDS:
        return f"Ego-hood staging receipt requires exactly {sorted(_STAGING_FIELDS)}"
    if type(staging["schema_version"]) is not int or staging["schema_version"] != 1:
        return "Unsupported ego-hood staging receipt version"
    if staging["artifact_kind"] != "ego_hood_staging_receipt":
        return "artifact_kind must be ego_hood_staging_receipt"
    if staging["scope"] not in ("synthetic", "actual"):
        return "scope must be synthetic or actual"
    expected_root = f"{data_root}/nre-artifacts/ego-hoods" if type(data_root) is str else None
    if expected_root is None or staging["staged_root"] != expected_root:
        return "staged_root must be <data_root>/nre-artifacts/ego-hoods (base_config.yaml:31,126)"
    files = staging["files"]
    if type(files) is not list or not all(type(item) is dict for item in files):
        return "files must be a list of file receipts"
    expected = sorted(json.dumps(item, sort_keys=True) for item in EGO_HOOD_FILES)
    declared = sorted(json.dumps(item, sort_keys=True) for item in files)
    if declared != expected:
        return ("Staged files must equal the pinned ego-hood blobs exactly "
                "(path, size_bytes, git_blob_sha1, sha256)")
    return None


def _validate_shm(shm):
    if type(shm) is not dict or sorted(shm) != list(COMPOSE_SERVICES):
        return f"shm_size requires exactly {list(COMPOSE_SERVICES)}; no default is chosen"
    for name, value in shm.items():
        if value == "docker_default":
            continue
        if type(value) is not str or re.fullmatch(r"[1-9][0-9]{0,5}[mg]", value) is None or (
                docker_bytes(value) is None):
            return f"shm_size[{name!r}] must be 'docker_default' or a size like '64g'"
    return None


def _capture_plan(run_dir, env_allowlist, env_set):
    """Docker CLI capture steps that must finish before teardown (R2 F5; L3)."""
    capture_dir = run_dir + "/physloop-capture"

    def step(step_id, argv, filename):
        return {"id": step_id, "cwd": run_dir, "argv": argv,
                "stdout_path": f"{capture_dir}/{filename}",
                "env_allowlist": list(env_allowlist), "env_set": dict(env_set)}

    captures = [
        {"id": "compose_up_stdout", "argv": None, "source": "compose.stdout_and_stderr",
         "stdout_path": f"{capture_dir}/compose-up.stdout.txt",
         "why": "only place runtime parent retry lines appear (simulate/__main__.py:296-300)"},
        step("compose_logs", ["docker", "compose", "logs", "--no-color", "--timestamps"],
             "compose-logs.txt"),
        step("compose_ps", ["docker", "compose", "ps", "-a", "--format", "json"],
             "compose-ps.json"),
        step("compose_container_ids", ["docker", "compose", "ps", "-a", "-q"],
             "compose-container-ids.txt"),
        {"id": "container_inspect", "cwd": run_dir, "argv": None,
         "argv_template": ["docker", "inspect", "{container_ids}"],
         "placeholder": "{container_ids}", "placeholder_source": "compose_container_ids",
         "stdout_path": f"{capture_dir}/docker-inspect.json",
         "env_allowlist": list(env_allowlist), "env_set": dict(env_set)},
    ]
    return capture_dir, captures, step


def _unit_record(*, index, job, mode, asset_sha256, unit_request, isolated_arm, scope, hashes,
                 cameras, shm, renderer_volumes, pinned):
    """One serial command unit (shared by the pilot and the full sweep)."""
    commands = _command_group(unit_request, isolated_arm, [job["scene_id"]])
    commands.pop("harmonizer")
    run_dir = isolated_arm["run_dir"]
    compose_env = commands["compose"]
    capture_dir, captures, step = _capture_plan(
        run_dir, compose_env["env_allowlist"], compose_env["env_set"])
    return {
        "index": index,
        "job": dict(job),
        "harmonizer_mode": mode,
        "asset_sha256": asset_sha256,
        "upstream_attempts": [],
        "upstream_attempts_status": "not_run",
        "upstream_attempt_schema": {
            "fields": ["rollout_uuid", "clipgt_ids", "state", "effective_steps",
                       "evidence_refs"],
            "producer": "gate.s2_capture.reconcile_unit_attempts",
        },
        "run_uuid_binding": {"source": run_dir + "/run_metadata.yaml#run_uuid",
                             "status": "not_run", "run_uuid": None},
        "launch_request_sha256": hashes["launch_request"],
        "schedule_request_sha256": hashes["schedule_request"],
        "commands_sha256": identity_hash(commands),
        "requires_previous_unit_teardown": True,
        "capture_dir": capture_dir,
        "driver_output_path": isolated_arm["driver_log_dir"] + "/physloop-driver-output.txt",
        "generated_config_expectations": {
            "scope": scope,
            "scene_ids": [job["scene_id"]],
            "requested_steps": unit_request["requested_steps"],
            "driver_port": unit_request["driver_port"],
            "teacher_snapshot": unit_request["teacher_snapshot"],
            "renderer_cameras": list(cameras),
        },
        "compose_config": step("compose_config",
                               ["docker", "compose", "config", "--format", "json"],
                               "compose-config.json"),
        "compose_expectations": {
            "scope": scope,
            "renderer_image": unit_request["renderer_image"],
            "base_image": unit_request["base_image"],
            "harmonizer": isolated_arm["harmonizer"],
            "harmonizer_cache": HARMONIZER_CONTAINER_DIR if pinned else None,
            "shm_size": dict(shm),
            "renderer_volumes": dict(renderer_volumes),
        },
        "required_captures_before_teardown": captures,
        "teardown": {
            **step("compose_down", ["docker", "compose", "down", "--timeout", "10"],
                   "compose-down.txt"),
            "requires": [capture["id"] for capture in captures],
        },
        **commands,
    }


def _validate_diagnostics(declarations, jobs):
    """Problems with the declared diagnostic units (empty when valid). Pure, L1."""
    if type(declarations) is not list or not declarations:
        return ["diagnostic_units must be a nonempty list; omit the key to declare none"]
    problems, seen, lines = [], set(), {}
    for position, item in enumerate(declarations):
        path = f"diagnostic_units[{position}]"
        if type(item) is not dict or set(item) != _DIAGNOSTIC_FIELDS:
            problems.append(f"{path} requires exactly {sorted(_DIAGNOSTIC_FIELDS)}")
            continue
        ident = item["diagnostic_id"]
        if type(ident) is not str or not _DIAGNOSTIC_ID.fullmatch(ident):
            problems.append(f"{path}.diagnostic_id must match [a-z0-9][a-z0-9-]{{0,39}}")
        elif ident in seen:
            problems.append(f"{path}.diagnostic_id {ident!r} is declared twice")
        else:
            seen.add(ident)
        if type(item["role"]) is not str or not _DIAGNOSTIC_ROLE.fullmatch(item["role"]):
            problems.append(f"{path}.role must be a declared identifier [a-z][a-z0-9_]{{2,63}}; "
                            "no default role is chosen")
        if type(item["source_job_id"]) is not str or item["source_job_id"] not in jobs:
            problems.append(f"{path}.source_job_id must name a scheduled pilot job")
        line = item["budget_line"]
        if type(line) is not dict or set(line) != {"line_id", "cap_usd"}:
            problems.append(f"{path}.budget_line requires exactly line_id and cap_usd")
            continue
        if type(line["line_id"]) is not str or not _BUDGET_LINE_ID.fullmatch(line["line_id"]):
            problems.append(f"{path}.budget_line.line_id must match [a-z][a-z0-9_]{{0,63}}")
            continue
        cap = line["cap_usd"]
        if type(cap) is not str or not _POSITIVE_USD.fullmatch(cap) or not any(
                char in "123456789" for char in cap):
            problems.append(f"{path}.budget_line.cap_usd must be a positive decimal string; "
                            "no default cap is chosen")
            continue
        if lines.setdefault(line["line_id"], cap) != cap:
            problems.append(f"{path}.budget_line {line['line_id']!r} is declared with two caps")
    return problems


def build_pilot_packet(request):
    """Validate exact pilot coverage and emit ordered singleton command units.

    ``prepared`` establishes consistency of declared requests and source-checked
    command templates. A caller's ``preregistered_permutation`` label is retained
    as a declaration; neither registration nor randomness is authenticated.
    Any blocking issue leaves ``units`` empty. Valid JSON inputs remain in the
    report for diagnosis, including invalid schedules/orders; non-JSON input is
    rejected. Launch warnings are carried through without blocking.
    """
    report = {
        "artifact_kind": "s2_pilot_packet",
        "schema_version": SCHEMA_VERSION,
        "scope": None,
        "producer": {"module": "gate.s2_packet", "function": "build_pilot_packet",
                     "source_sha256": _SOURCE_SHA256},
        "prepared": False,
        "issues": [],
        "inputs": None,
        "schedule": [],
        "units": [],
        "cloud_launch_authorized": False,
        "publication_authorized": False,
        "runtime_qualified": False,
        "execution_order_registration_verified": False,
        "randomization_verified": False,
        "asset_identities_verified": False,
        "ego_hood_staging_host_verified": False,
        "commands_executed": False,
        "execution_model": "serial_units_driver_restart_capture_before_compose_teardown",
        "startup_cost_basis": "unmeasured_per_unit_startup_included_in_paid_work",
        "validation_scope": "declared_input_binding_and_pinned_source_command_preparation",
        "unit_completion_rule": ("_complete marker + metrics.parquet + one non-runtime-failure "
                                 "summary row; exit code ignored (gate.s2_capture)"),
        "retry_controls": RETRY_CONTROLS,
        "compose_pull_never": COMPOSE_PULL_NEVER,
    }

    try:
        snapshot = json.loads(canonical_json_bytes(request))
    except (ValueError, TypeError, UnicodeError) as exc:
        _issue(report, "invalid_request", "request", str(exc))
        return report
    report["inputs"] = snapshot
    report["request_sha256"] = identity_hash(snapshot)
    if type(snapshot) is not dict or set(snapshot) - {DIAGNOSTIC_FIELD} != _FIELDS:
        present = set(snapshot) if type(snapshot) is dict else set()
        _issue(report, "invalid_request", "request",
               f"Provide exactly the packet fields; missing {sorted(_FIELDS - present)}, "
               f"unexpected {sorted(present - _FIELDS - {DIAGNOSTIC_FIELD})}.")
        return report
    declared = snapshot.get(DIAGNOSTIC_FIELD) if DIAGNOSTIC_FIELD in snapshot else None
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != SCHEMA_VERSION:
        _issue(report, "unsupported_schema", "schema_version",
               "Only packet version 2 is supported (ego-hood staging and shm_size required).")
    scope = snapshot["scope"]
    if type(scope) is str and scope in SCOPES:
        report["scope"] = scope
        launch_scope = (snapshot["launch_request"].get("scope")
                        if type(snapshot["launch_request"]) is dict else None)
        staging_scope = (snapshot["ego_hood_staging"].get("scope")
                         if type(snapshot["ego_hood_staging"]) is dict else None)
        if launch_scope != scope:
            _issue(report, "scope_mismatch", "launch_request.scope",
                   f"Launch request scope {launch_scope!r} differs from packet scope {scope!r}")
        # An actual packet needs an actual staging receipt; a synthetic or
        # public-source-fixture packet must not borrow an actual one.
        if type(staging_scope) is str and (staging_scope == "actual") != (scope == "actual"):
            _issue(report, "scope_mismatch", "ego_hood_staging.scope",
                   f"Staging receipt scope {staging_scope!r} is inconsistent with packet "
                   f"scope {scope!r}")
    else:
        _issue(report, "invalid_scope", "scope",
               f"Declare the packet scope as one of {list(SCOPES)}; no default is chosen.")
    launch_request = snapshot["launch_request"]
    schedule_request = snapshot["schedule_request"]
    order = snapshot["execution_order"]
    report["input_hashes"] = {
        key: identity_hash(snapshot[key])
        for key in ("launch_request", "schedule_request", "execution_order",
                    "ego_hood_staging", "shm_size")
    }
    if DIAGNOSTIC_FIELD in snapshot:
        report["input_hashes"][DIAGNOSTIC_FIELD] = identity_hash(snapshot[DIAGNOSTIC_FIELD])
    launch = build_pilot_launch(launch_request)
    for item in launch["issues"]:
        # Review A2-F12: the on-arm licence gate keeps its own code so it is not hidden.
        keep = item["severity"] != "blocking" or item["code"] == "harmonizer_terms_unrecorded"
        _issue(report, item["code"] if keep else "invalid_launch_request",
               "launch_request." + item["path"], item["message"], item["severity"])
    data_root = launch_request.get("data_root") if type(launch_request) is dict else None
    problem = _validate_staging(snapshot["ego_hood_staging"], data_root)
    if problem:
        _issue(report, "ego_hood_staging_required", "ego_hood_staging", problem)
    problem = _validate_shm(snapshot["shm_size"])
    if problem:
        _issue(report, "invalid_shm_size", "shm_size", problem)
    try:
        if type(schedule_request) is not dict or set(schedule_request) != _SCHEDULE_FIELDS:
            raise ValueError("Provide exactly the schedule builder's required fields.")
        if schedule_request["phase"] != "harmonizer_pilot":
            raise ValueError("The packet supports only the five-scene Harmonizer pilot.")
        schedule = build_schedule(**schedule_request)
    except (ValueError, TypeError, UnicodeError) as exc:
        _issue(report, "invalid_schedule_request", "schedule_request", str(exc))
        return report
    report["schedule"] = schedule
    report["schedule_sha256"] = identity_hash(schedule)
    if launch["prepared"] and (
        set(launch_request["scene_ids"]) != {job["scene_id"] for job in schedule}
        or launch_request["requested_steps"] != schedule_request["requested_steps"]
    ):
        _issue(report, "launch_schedule_mismatch", "launch_request",
               "Launch and schedule must name exactly the same scenes and requested horizon.")
    jobs = {job["job_id"]: job for job in schedule}
    protocols = {
        value.lower(): mode for mode, value in schedule_request["protocol_hashes"].items()
    }
    if (
        type(order) is not dict
        or set(order) != {"method", "job_ids"}
        or order["method"] not in ("alternating", "preregistered_permutation")
        or type(order["job_ids"]) is not list
        or len(order["job_ids"]) != 10
        or not all(type(job_id) is str for job_id in order["job_ids"])
        or set(order["job_ids"]) != set(jobs)
    ):
        _issue(report, "invalid_execution_order", "execution_order",
               "Supply all ten unique scheduled job IDs and an explicit supported order method.")
    elif order["method"] == "alternating":
        modes = [protocols[jobs[job_id]["protocol_hash"]] for job_id in order["job_ids"]]
        if any(first == second for first, second in pairwise(modes)):
            _issue(report, "invalid_execution_order", "execution_order.job_ids",
                   "An alternating order must change Harmonizer mode after every unit.")
    if DIAGNOSTIC_FIELD in snapshot:
        for problem in _validate_diagnostics(declared, jobs):
            _issue(report, "invalid_diagnostic_units", DIAGNOSTIC_FIELD, problem)
    if any(item["severity"] == "blocking" for item in report["issues"]):
        return report

    assets = {row["scene_id"]: row["asset_sha256"].lower() for row in schedule_request["scenes"]}
    arms = {"on" if arm["harmonizer"] else "off": arm for arm in launch_request["arms"]}
    pinned = launch_request["harmonizer_source"]["kind"] == "pinned_mount"
    cameras = list(CAMERA_PRESETS[launch_request["renderer_camera_override"] or "6cam_1080"])
    # Review A4-F5: the mounts the argv requests, checked on the effective Compose
    # config. The ego-hood source is the staged root (base_config.yaml:31,126).
    renderer_volumes = {EGO_HOOD_CONTAINER_DIR: snapshot["ego_hood_staging"]["staged_root"]}
    if pinned:
        renderer_volumes[HARMONIZER_CONTAINER_DIR] = (
            launch_request["harmonizer_source"]["host_dir"])
    report["ego_hood_staging"] = {
        "receipt_sha256": report["input_hashes"]["ego_hood_staging"],
        "scope": snapshot["ego_hood_staging"]["scope"],
        "declared_matches_pin": True, "host_verified": False,
    }
    units, output_paths = [], []
    hashes = {"launch_request": report["input_hashes"]["launch_request"],
              "schedule_request": report["input_hashes"]["schedule_request"]}
    for index, job_id in enumerate(order["job_ids"]):
        job = jobs[job_id]
        mode = protocols[job["protocol_hash"]]
        arm = arms[mode]
        isolated_arm = {
            "harmonizer": arm["harmonizer"],
            "run_dir": arm["run_dir"] + "/" + job_id,
            "driver_log_dir": arm["driver_log_dir"] + "/" + job_id,
        }
        current_paths = [isolated_arm["run_dir"], isolated_arm["driver_log_dir"]]
        if any(_overlap(first, second) for first in current_paths for second in output_paths):
            _issue(report, "overlapping_unit_outputs", "launch_request.arms",
                   "Derived output directories of different jobs must not equal or contain "
                   "one another.")
        output_paths.extend(current_paths)
        units.append(_unit_record(
            index=index, job=job, mode=mode, asset_sha256=assets[job["scene_id"]],
            unit_request=launch_request, isolated_arm=isolated_arm, scope=scope, hashes=hashes,
            cameras=cameras, shm=snapshot["shm_size"], renderer_volumes=renderer_volumes,
            pinned=pinned))
    diagnostics = []
    for index, item in enumerate(declared or []):
        source = jobs[item["source_job_id"]]
        mode = protocols[source["protocol_hash"]]
        arm = arms[mode]
        job = {**source, "job_id": DIAGNOSTIC_JOB_PREFIX + item["diagnostic_id"],
               "phase": DIAGNOSTIC_PHASE}
        isolated_arm = {
            "harmonizer": arm["harmonizer"],
            "run_dir": arm["run_dir"] + "/" + job["job_id"],
            "driver_log_dir": arm["driver_log_dir"] + "/" + job["job_id"],
        }
        current_paths = [isolated_arm["run_dir"], isolated_arm["driver_log_dir"]]
        if any(_overlap(first, second) for first in current_paths for second in output_paths):
            _issue(report, "overlapping_unit_outputs", DIAGNOSTIC_FIELD,
                   "Diagnostic unit directories must not equal or contain any other unit's.")
        output_paths.extend(current_paths)
        unit = _unit_record(
            index=index, job=job, mode=mode, asset_sha256=assets[job["scene_id"]],
            unit_request=launch_request, isolated_arm=isolated_arm, scope=scope, hashes=hashes,
            cameras=cameras, shm=snapshot["shm_size"], renderer_volumes=renderer_volumes,
            pinned=pinned)
        unit["diagnostic"] = {
            "diagnostic_id": item["diagnostic_id"], "role": item["role"],
            "source_job_id": item["source_job_id"], "budget_line": dict(item["budget_line"]),
            "counts_toward_pilot_jobs": False, "inference_input": False,
        }
        diagnostics.append(unit)
    if any(item["severity"] == "blocking" for item in report["issues"]):
        return report
    report["units"] = units
    if DIAGNOSTIC_FIELD in snapshot:
        report["diagnostic_units"] = diagnostics
        report["diagnostics"] = {
            "unit_count": len(diagnostics),
            "budget_lines": {item["budget_line"]["line_id"]: item["budget_line"]["cap_usd"]
                             for item in declared},
            "execution": "before_pilot_units_serially",
            "counts_toward_pilot_jobs": False, "inference_input": False,
            "ledger": "separate from the pilot schedule's attempt ledger",
        }
    report["prepared"] = True
    return report


FULL_SWEEP_SCHEMA_VERSION = 1
_SWEEP_FIELDS = {"schema_version", "artifact_kind", "scope", "launch", "schedule_request",
                 "execution_order", "ego_hood_staging", "shm_size"}


def build_full_sweep_packet(request):
    """Bind a registered full sweep (F1) to ordered singleton command units; never execute.

    Required owner inputs, no defaults: the 441-scene list (``schedule_request.scenes``
    with road, stratum and asset identity), the replicate count
    (``schedule_request.replicates``; ``gate.s2_schedule`` requires at least three), the
    execution order (``execution_order.job_ids``, every scheduled job exactly once; the
    ``preregistered_permutation`` label is a declaration, not authenticated
    randomization), the Harmonizer setting and source, the camera decision and the A/B
    policy snapshots. Every scheduled scene ID must pass the pilot's portable-ASCII rule
    (``invalid_scene_selection`` otherwise), because the pinned wizard joins argv into
    shell commands. With three replicates the matrix is 2 arms x 441 scenes x 3 = 2,646
    units. Unit ``i`` (execution index) gets driver port ``driver_port_base + i`` and its
    own ``<policy run_dir>/<job_id>`` and ``<policy driver_log_dir>/<job_id>``
    directories, so every unit has distinct directories and ports; units still run
    strictly serially with teardown in between, as in the pilot.

    The unit shape is the pilot's (``build_pilot_packet``) plus ``policy_arm``; the
    executor (``gate.s2_executor``) currently accepts only ``s2_pilot_packet`` and must be
    extended before a full-sweep packet can be executed. Verification: L1 binding, L2
    argv against the pinned source, L3 execution.
    """
    report = {
        "artifact_kind": "s2_full_sweep_packet",
        "schema_version": FULL_SWEEP_SCHEMA_VERSION,
        "scope": None,
        "producer": {"module": "gate.s2_packet", "function": "build_full_sweep_packet",
                     "source_sha256": _SOURCE_SHA256},
        "prepared": False,
        "issues": [],
        "inputs": None,
        "schedule": [],
        "units": [],
        "sweep": None,
        "cloud_launch_authorized": False,
        "publication_authorized": False,
        "runtime_qualified": False,
        "execution_order_registration_verified": False,
        "randomization_verified": False,
        "asset_identities_verified": False,
        "ego_hood_staging_host_verified": False,
        "commands_executed": False,
        "execution_model": "serial_units_driver_restart_capture_before_compose_teardown",
        "startup_cost_basis": "unmeasured_per_unit_startup_included_in_paid_work",
        "validation_scope": "declared_input_binding_and_pinned_source_command_preparation",
        "unit_completion_rule": ("_complete marker + metrics.parquet + one non-runtime-failure "
                                 "summary row; exit code ignored (gate.s2_capture)"),
        "retry_controls": RETRY_CONTROLS,
        "compose_pull_never": COMPOSE_PULL_NEVER,
    }
    try:
        snapshot = json.loads(canonical_json_bytes(request))
    except (ValueError, TypeError, UnicodeError) as exc:
        _issue(report, "invalid_request", "request", str(exc))
        return report
    report["inputs"] = snapshot
    report["request_sha256"] = identity_hash(snapshot)
    if type(snapshot) is not dict or set(snapshot) != _SWEEP_FIELDS:
        present = set(snapshot) if type(snapshot) is dict else set()
        _issue(report, "invalid_request", "request",
               f"Provide exactly the full-sweep fields; missing {sorted(_SWEEP_FIELDS - present)}, "
               f"unexpected {sorted(present - _SWEEP_FIELDS)}.")
        return report
    if (snapshot["artifact_kind"] != "s2_full_sweep_request"
            or type(snapshot["schema_version"]) is not int
            or snapshot["schema_version"] != FULL_SWEEP_SCHEMA_VERSION):
        _issue(report, "unsupported_schema", "schema_version",
               "Require an s2_full_sweep_request of schema version 1.")
        return report
    scope = snapshot["scope"]
    if type(scope) is str and scope in SCOPES:
        report["scope"] = scope
        staging_scope = (snapshot["ego_hood_staging"].get("scope")
                         if type(snapshot["ego_hood_staging"]) is dict else None)
        if type(staging_scope) is str and (staging_scope == "actual") != (scope == "actual"):
            _issue(report, "scope_mismatch", "ego_hood_staging.scope",
                   f"Staging receipt scope {staging_scope!r} is inconsistent with packet "
                   f"scope {scope!r}")
    else:
        _issue(report, "invalid_scope", "scope",
               f"Declare the packet scope as one of {list(SCOPES)}; no default is chosen.")
    report["input_hashes"] = {
        key: identity_hash(snapshot[key])
        for key in ("launch", "schedule_request", "execution_order", "ego_hood_staging",
                    "shm_size")
    }
    launch, schedule_request = snapshot["launch"], snapshot["schedule_request"]
    order = snapshot["execution_order"]
    try:
        if type(schedule_request) is not dict or set(schedule_request) != _SCHEDULE_FIELDS:
            raise ValueError("Provide exactly the schedule builder's required fields.")
        if schedule_request["phase"] != "full_sweep":
            raise ValueError("The full-sweep packet supports only phase full_sweep.")
        schedule = build_schedule(**schedule_request)
    except (ValueError, TypeError, UnicodeError) as exc:
        _issue(report, "invalid_schedule_request", "schedule_request", str(exc))
        return report
    report["schedule"] = schedule
    report["schedule_sha256"] = identity_hash(schedule)
    scene_ids = [row.get("scene_id") if type(row) is dict else None
                 for row in schedule_request["scenes"]]
    for position in check_scene_ids(scene_ids):
        _issue(report, "invalid_scene_selection", f"schedule_request.scenes[{position}].scene_id",
               "Scene IDs use only [A-Za-z0-9._-] (not '.' or '..'): the pinned wizard joins "
               "argv into shell commands, as in build_pilot_launch.")
    for item in check_sweep_launch(launch, len(schedule)):
        _issue(report, item["code"], "launch." + item["path"], item["message"], item["severity"])
    data_root = launch.get("data_root") if type(launch) is dict else None
    problem = _validate_staging(snapshot["ego_hood_staging"], data_root)
    if problem:
        _issue(report, "ego_hood_staging_required", "ego_hood_staging", problem)
    problem = _validate_shm(snapshot["shm_size"])
    if problem:
        _issue(report, "invalid_shm_size", "shm_size", problem)
    if type(launch) is dict and launch.get("requested_steps") != schedule_request[
            "requested_steps"]:
        _issue(report, "launch_schedule_mismatch", "launch.requested_steps",
               "Launch and schedule must name the same requested horizon.")
    jobs = {job["job_id"]: job for job in schedule}
    job_ids = order.get("job_ids") if type(order) is dict else None
    if (
        type(order) is not dict
        or set(order) != {"method", "job_ids"}
        or order["method"] != "preregistered_permutation"
        or type(job_ids) is not list
        or len(job_ids) != len(jobs)
        or not all(type(job_id) is str for job_id in job_ids)
        or set(job_ids) != set(jobs)
    ):
        _issue(report, "invalid_execution_order", "execution_order",
               f"Supply all {len(jobs)} unique scheduled job IDs once each under the "
               "preregistered_permutation method; no default order is chosen.")
    if any(item["severity"] == "blocking" for item in report["issues"]):
        return report

    assets = {row["scene_id"]: row["asset_sha256"].lower() for row in schedule_request["scenes"]}
    pinned = launch["harmonizer_source"]["kind"] == "pinned_mount"
    cameras = list(CAMERA_PRESETS[launch["renderer_camera_override"] or "6cam_1080"])
    renderer_volumes = {EGO_HOOD_CONTAINER_DIR: snapshot["ego_hood_staging"]["staged_root"]}
    if pinned:
        renderer_volumes[HARMONIZER_CONTAINER_DIR] = launch["harmonizer_source"]["host_dir"]
    mode = "on" if launch["harmonizer"] else "off"
    hashes = {"launch_request": report["input_hashes"]["launch"],
              "schedule_request": report["input_hashes"]["schedule_request"]}
    report["ego_hood_staging"] = {
        "receipt_sha256": report["input_hashes"]["ego_hood_staging"],
        "scope": snapshot["ego_hood_staging"]["scope"],
        "declared_matches_pin": True, "host_verified": False,
    }
    units = []
    for index, job_id in enumerate(job_ids):
        job = jobs[job_id]
        policy = launch["policies"][job["arm"]]
        isolated_arm = {"harmonizer": launch["harmonizer"],
                        "run_dir": policy["run_dir"] + "/" + job_id,
                        "driver_log_dir": policy["driver_log_dir"] + "/" + job_id}
        unit = _unit_record(
            index=index, job=job, mode=mode, asset_sha256=assets[job["scene_id"]],
            unit_request=sweep_unit_request(launch, job["arm"], launch["driver_port_base"] + index),
            isolated_arm=isolated_arm, scope=scope, hashes=hashes, cameras=cameras,
            shm=snapshot["shm_size"], renderer_volumes=renderer_volumes, pinned=pinned)
        unit["policy_arm"] = job["arm"]
        units.append(unit)
    # Distinct by construction (unique job IDs under disjoint policy roots); checked anyway.
    directories = [unit[key] for unit in units for key in ("capture_dir", "driver_output_path")]
    ports = [unit["generated_config_expectations"]["driver_port"] for unit in units]
    if len(set(directories)) != len(directories) or len(set(ports)) != len(ports):
        _issue(report, "overlapping_unit_outputs", "launch.policies",
               "Every unit needs its own directories and driver port.")
        return report
    report["sweep"] = {
        "scene_count": len(schedule_request["scenes"]),
        "arm_count": len(schedule_request["policies"]),
        "replicates": schedule_request["replicates"],
        "unit_count": len(units),
        "harmonizer_mode": mode,
        "driver_port_range": [ports[0], ports[-1]],
        "counts_exclude_diagnostics_and_retries": True,
    }
    report["units"] = units
    report["prepared"] = True
    return report
