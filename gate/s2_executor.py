"""Pure per-unit S2 execution plan and attempt-recorder events from a prepared packet.

Tickets: build-spec B1 (runbook section 4 steps 6-8; lane R1 P4 "serial executor +
attempt recorder", P4b "neutral process-completion reason"; W2 producer for
``gate.s2_ledger``; lane R2 section 1 phase F, F1-F6, and section 4 P1-P3, C1-C3).
Agent-authored evaluator engineering under D-018/D-024; serves S2.1-S2.7 [C]
(G1, G2, G3-R1, G7).

``build_execution_plan(packet, run_config)`` validates a prepared ``gate.s2_packet`` packet
of a kind in ``PACKET_KINDS`` -- ``s2_pilot_packet`` schema 2 or ``s2_full_sweep_packet``
schema 1 (the pilot unit shape plus ``policy_arm``, a driver port per unit); any other kind
or version is ``unsupported_packet`` -- (``prepared``, exact capture plan, untampered
``commands_sha256``) and emits, per unit in
the registered order, the fixed step list ``UNIT_STEPS`` that
``scripts/s2_run_units.py`` executes verbatim::

    budget_check -> ledger_started -> run_dirs_absent -> wizard_prepare
    -> generated_config_check -> compose_shm_override -> compose_config_check
    -> driver_port_free -> driver_start -> driver_ready -> compose_up
    -> capture:{compose_up_stdout, compose_logs, compose_ps, compose_container_ids,
                container_inspect} -> run_metadata_capture -> run_dir_hash
    -> compose_down -> driver_stop -> driver_port_released -> reconcile -> ledger_terminal

Every argv is the packet's own, and the packet is bound before any plan is emitted:
``request_sha256`` must equal ``identity_hash(packet["inputs"])`` and the kind's own builder
(``gate.s2_packet.build_pilot_packet`` or ``build_full_sweep_packet``) applied to
``packet["inputs"]`` must reproduce the packet's ``artifact_kind``, ``units`` and
``schedule`` exactly. So every executed argv, cwd, env allowlist and env value
(wizard, driver, Compose, compose config, captures, inspect template, teardown) follows from
the inputs whose ``request_sha256`` the run authority names; ``commands_sha256`` alone binds
only wizard/driver/Compose and is not relied on for that. Steps carry ``on_failure``: ``stop_run`` (a pre-GPU block, a driver that never
becomes ready, a teardown that cannot be verified) or ``record`` (captures; failures are
evidence). Teardown steps carry ``always_after`` and run whenever the driver or Compose
was started.

Recorder semantics (ledger event schema v2, ``gate.s2_ledger``):

- ``started_event`` for attempt 1 is appended before any side effect of the unit, with
  ``upstream_identity: None`` (nothing upstream exists yet).
- ``unit_terminal_events`` binds every upstream attempt that
  ``gate.s2_capture.reconcile_unit_attempts`` found: the first upstream attempt closes
  attempt 1; each further one (a runtime-internal retry) is a new ``started`` (parent =
  previous attempt, ``retry_reason`` ``runtime_internal_retry``) plus its terminal; a
  failure before a rollout UUID is a failed attempt with ``upstream_identity: None``.
  Attempts are ordered by their first line in the first captured log (Compose stdout),
  then the following logs, then rollout UUID; the ledger blocks selection among them
  regardless (``retry_selection_unregistered`` is Huy's rule).
- ``completed`` (reason ``process_completed``) is emitted only for an attempt whose
  capture state is ``complete`` and whose unit report has no blocking issue other than
  ``retry_selection_unregistered``/``unit_not_complete``; otherwise ``failed`` with a
  descriptive reason (``completion_not_established:<codes>`` when evidence conflicts), or
  ``cancelled`` with the recorder's interruption reason when the runner stopped Compose.
  Success is never inferred from an exit code.
- ``cancelled_event`` closes attempt 1 when the unit stops before Compose (blocked
  generated config, driver not ready, wizard failure, internal error).

Verification level per function (build-spec section 1): ``build_execution_plan``,
``started_event``, ``cancelled_event``, ``unit_terminal_events``,
``has_upstream_attempt_evidence``, ``terminal_established``, ``shm_override``,
``container_ids``, ``inspect_argv``, ``worst_case_unit_seconds`` L1 (literal known
answers); ``driver_ready_line`` L2 for the message against the pinned driver source
(main.py:1048-1054) at AlpaSim e121e591, accepting two line prefixes: Hydra's default
job_logging ``[%(asctime)s][%(name)s][%(levelname)s] - `` (main.py:1106 is
``@hydra.main``, whose run_job configures job logging before the task function, so the
basicConfig at :1116-1119 without ``force=True`` is a no-op; hydra-core 1.3.8
conf/hydra/job_logging/default.yaml) and that basicConfig form in case logging is
reconfigured. The exact prefix the admitted host prints is ``qualify_on_host``; the
Compose implicit override file name (``docker-compose.override.yaml``
beside the wizard's ``docker-compose.yaml``) is L2 against Compose's documented default
file lookup and is verified on the effective ``docker compose config`` by
``gate.s2_launch.check_generated_compose`` before GPU work; actual host behaviour is L3.

Declared diagnostic qualification units (``gate.s2_packet`` ``diagnostic_units``; pilot
packets only). When a packet carries them, each must be a unit of the same shape whose job
ID starts with ``diagnostic-`` and whose phase is ``diagnostic_qualification`` (review
G2-04), is absent from the schedule, carries the packet's
``diagnostic`` block (role, budget line, ``counts_toward_pilot_jobs: false``,
``inference_input: false``) and has its own run directory; the rebuild check covers them
too. They are planned in ``diagnostic_unit_plans`` (key present only when declared, so a
packet without diagnostics plans byte-identically) with the same ``UNIT_STEPS``, journal
work directories ``diagnostics/NNN`` and a ``diagnostic`` block, and
``diagnostic_schedule`` lists their jobs in the ledger's job schema for their own ledger
(``diagnostic_schedule``; never merged into the pilot schedule). Verification level L1.

Run configuration timeouts and the poll interval are required inputs with no defaults.
This module never executes, reads files or environment variables, chooses an owner value,
selects among retried attempts, admits a score or treats an exit code as success.
"""

import hashlib
import math
import re
from pathlib import Path, PurePosixPath

from gate.s2_identity import identity_hash
from gate.s2_ledger import PROCESS_COMPLETED
from gate.s2_packet import (
    DIAGNOSTIC_JOB_PREFIX,
    DIAGNOSTIC_PHASE,
    FULL_SWEEP_SCHEMA_VERSION,
    build_full_sweep_packet,
    build_pilot_packet,
)
from gate.s2_packet import SCHEMA_VERSION as PACKET_SCHEMA_VERSION

ARTIFACT_KIND = "s2_unit_execution_plan"
# Executable packet kinds: artifact_kind -> (schema version, the builder that must reproduce
# the packet's units and schedule from packet["inputs"]). Any other kind is refused.
PACKET_KINDS = {
    "s2_pilot_packet": (PACKET_SCHEMA_VERSION, build_pilot_packet),
    "s2_full_sweep_packet": (FULL_SWEEP_SCHEMA_VERSION, build_full_sweep_packet),
}
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
LEDGER_EVENT_SCHEMA_VERSION = 2
CAPTURE_IDS = ("compose_up_stdout", "compose_logs", "compose_ps", "compose_container_ids",
               "container_inspect")
UNIT_STEPS = (
    "budget_check", "ledger_started", "run_dirs_absent", "wizard_prepare",
    "generated_config_check", "compose_shm_override", "compose_config_check",
    "driver_port_free", "driver_start", "driver_ready", "compose_up",
    *(f"capture:{capture}" for capture in CAPTURE_IDS),
    "run_metadata_capture", "run_dir_hash", "compose_down", "driver_stop",
    "driver_port_released", "reconcile", "ledger_terminal",
)
COMPOSE_UP_ARGV = ["docker", "compose", "up", "--pull", "never", "--exit-code-from", "runtime-0"]
COMPOSE_OVERRIDE_NAME = "docker-compose.override.yaml"
# Compose's implicit override names; any pre-existing one would make the merge ambiguous.
COMPOSE_OVERRIDE_NAMES = ("compose.override.yml", "compose.override.yaml",
                          "docker-compose.override.yml", "docker-compose.override.yaml")
RUN_CONFIG_FIELDS = (
    "wizard_prepare_timeout_s", "compose_config_timeout_s", "driver_ready_timeout_s",
    "driver_poll_interval_s", "compose_up_timeout_s", "capture_timeout_s",
    "teardown_timeout_s", "port_release_timeout_s",
)
# Reasons the recorder writes; codes the unit report may carry beside a complete attempt.
RETRY_REASON = "runtime_internal_retry"
OTHER_ATTEMPT_REASON = "additional_upstream_attempt_evidence"
_TOLERATED_FOR_COMPLETION = {"retry_selection_unregistered", "unit_not_complete"}
_UNIT_FIELDS = {
    "index", "job", "wizard_prepare", "driver", "compose", "compose_config",
    "generated_config_expectations", "compose_expectations",
    "required_captures_before_teardown", "teardown", "capture_dir", "driver_output_path",
    "commands_sha256",
}
_DIAGNOSTIC_BLOCK = {"diagnostic_id", "role", "source_job_id", "budget_line",
                     "counts_toward_pilot_jobs", "inference_input"}
_LOG_REF = re.compile(r"^logs\[(\d+)\]:line:(\d+)$")
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}")


# Runner calls that may each run to their timeout and then escalate SIGINT/SIGTERM/SIGKILL:
# wizard, compose config, compose up, the four capture commands and compose down.
_RUN_CALL_TIMEOUTS = ("wizard_prepare_timeout_s", "compose_config_timeout_s",
                      "compose_up_timeout_s", *("capture_timeout_s",) * 4,
                      "teardown_timeout_s")


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _issue(issues, code, path, message, severity="blocking"):
    issues.append({"code": code, "path": path, "severity": severity, "evidence_refs": [],
                   "message": message})


def _abs(value):
    return (type(value) is str and value.startswith("/") and "\x00" not in value
            and all(part not in ("", ".", "..") for part in value.split("/")[1:]))


def _within(path, parent):
    return _abs(path) and _abs(parent) and PurePosixPath(parent) in PurePosixPath(path).parents


def _check_run_config(config, issues):
    if type(config) is not dict or set(config) != {"schema_version", *RUN_CONFIG_FIELDS}:
        _issue(issues, "invalid_run_config", "run_config",
               f"Run config requires exactly schema_version and {list(RUN_CONFIG_FIELDS)}; "
               "no timeout is defaulted")
        return False
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        _issue(issues, "invalid_run_config", "run_config.schema_version", "Only version 1")
        return False
    for name in RUN_CONFIG_FIELDS:
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            _issue(issues, "invalid_run_config", f"run_config.{name}",
                   "Must be a positive finite number of seconds")
            return False
    if config["driver_poll_interval_s"] >= config["driver_ready_timeout_s"]:
        _issue(issues, "invalid_run_config", "run_config.driver_poll_interval_s",
               "The poll interval must be shorter than the readiness timeout")
        return False
    return True


def _check_diagnostic(packet, unit, path, issues):
    """A declared diagnostic unit is never a scheduled job and never an inference input."""
    job, block = unit["job"], unit.get("diagnostic")
    if unit["job"] in packet["schedule"] or type(job) is not dict or any(
            type(item) is dict and item.get("job_id") == job.get("job_id")
            for item in packet["schedule"]):
        _issue(issues, "diagnostic_unit_scheduled", path,
               "A diagnostic unit's job must not be one of the scheduled pilot jobs")
    if type(job) is not dict or type(job.get("job_id")) is not str or not job[
            "job_id"].startswith(DIAGNOSTIC_JOB_PREFIX):
        _issue(issues, "invalid_diagnostic_unit", f"{path}.job.job_id",
               f"A diagnostic job ID starts with {DIAGNOSTIC_JOB_PREFIX!r}")
    if (type(block) is not dict or set(block) != _DIAGNOSTIC_BLOCK
            or block.get("counts_toward_pilot_jobs") is not False
            or block.get("inference_input") is not False
            or type(job) is not dict
            or job.get("phase") != DIAGNOSTIC_PHASE
            or job.get("job_id") != DIAGNOSTIC_JOB_PREFIX + str(block.get("diagnostic_id"))):
        _issue(issues, "invalid_diagnostic_unit", f"{path}.diagnostic",
               "A diagnostic unit carries its role and budget line, never counts toward the "
               "pilot jobs and is never an inference input")


def _check_unit(packet, unit, position, issues, run_dirs, *, diagnostic=False):
    path = f"{'diagnostic_units' if diagnostic else 'units'}[{position}]"
    if type(unit) is not dict or not _UNIT_FIELDS <= set(unit):
        _issue(issues, "invalid_packet_unit", path, f"Unit requires {sorted(_UNIT_FIELDS)}")
        return
    if unit["index"] != position:
        _issue(issues, "unit_order_mismatch", path, "Unit index differs from its position")
    if diagnostic:
        _check_diagnostic(packet, unit, path, issues)
    elif unit["job"] not in packet["schedule"]:
        _issue(issues, "unit_job_unscheduled", path, "Unit job is not in the packet schedule")
    commands = {key: unit[key] for key in ("wizard_prepare", "driver", "compose")}
    try:
        bound = identity_hash(commands) == unit["commands_sha256"]
    except (TypeError, ValueError):
        bound = False
    if not bound:
        _issue(issues, "commands_hash_mismatch", path,
               "wizard_prepare/driver/compose do not hash to the unit's commands_sha256")
    compose = unit["compose"]
    if (type(compose) is not dict or compose.get("argv") != COMPOSE_UP_ARGV
            or compose.get("exit_code_is_success_evidence") is not False):
        _issue(issues, "unexpected_compose_command", f"{path}.compose",
               f"Compose must be exactly {COMPOSE_UP_ARGV} with exit code not success evidence")
        return
    run_dir = compose.get("cwd")
    if not _abs(run_dir):
        _issue(issues, "invalid_run_dir", f"{path}.compose.cwd", "Run dir must be absolute")
        return
    if run_dir in run_dirs:
        _issue(issues, "duplicate_run_dir", f"{path}.compose.cwd", "Run dirs must be unique")
    run_dirs.add(run_dir)
    capture_dir = run_dir + "/physloop-capture"
    if unit["capture_dir"] != capture_dir:
        _issue(issues, "unexpected_capture_dir", f"{path}.capture_dir",
               "Capture dir must be <run_dir>/physloop-capture")
    captures = unit["required_captures_before_teardown"]
    ids = [item.get("id") for item in captures] if type(captures) is list and all(
        type(item) is dict for item in captures) else None
    if ids != list(CAPTURE_IDS):
        _issue(issues, "unexpected_capture_plan", f"{path}.required_captures_before_teardown",
               f"Captures must be exactly {list(CAPTURE_IDS)} in order")
    elif not all(_within(item.get("stdout_path"), capture_dir) for item in captures):
        _issue(issues, "unexpected_capture_plan", f"{path}.required_captures_before_teardown",
               "Every capture writes under the unit capture dir")
    teardown = unit["teardown"]
    if (type(teardown) is not dict or teardown.get("id") != "compose_down"
            or teardown.get("requires") != list(CAPTURE_IDS)
            or teardown.get("cwd") != run_dir):
        _issue(issues, "unexpected_teardown", f"{path}.teardown",
               "Teardown must be compose_down in the run dir requiring every capture")
    config_step = unit["compose_config"]
    if type(config_step) is not dict or config_step.get("cwd") != run_dir or not _within(
            config_step.get("stdout_path"), capture_dir):
        _issue(issues, "unexpected_compose_config", f"{path}.compose_config",
               "compose_config must run in the run dir and write under the capture dir")
    driver = unit["driver"]
    if type(driver) is not dict or f"--config-path={run_dir}" not in (driver.get("argv") or []):
        _issue(issues, "unexpected_driver_command", f"{path}.driver",
               "Driver must read the unit's generated driver-config.yaml")
    if not _abs(unit["driver_output_path"]):
        _issue(issues, "invalid_driver_output_path", f"{path}.driver_output_path",
               "Driver output path must be absolute")
    for key in ("generated_config_expectations", "compose_expectations"):
        expected = unit[key]
        if type(expected) is not dict or expected.get("scope") != packet["scope"]:
            _issue(issues, "scope_mismatch", f"{path}.{key}", "Expectation scope differs")
    expected = unit["generated_config_expectations"]
    if type(expected) is dict and (type(expected.get("driver_port")) is not int
                                   or not 0 < expected["driver_port"] < 65536):
        _issue(issues, "invalid_driver_port", f"{path}.generated_config_expectations",
               "driver_port must be a TCP port")


def worst_case_unit_seconds(run_config, *, kill_grace_s):
    """Upper bound in seconds of one unit's runner time under ``run_config`` (L1).

    Sum of the eight runner calls' timeouts, each plus three ``kill_grace_s`` waits
    (SIGINT, SIGTERM, SIGKILL escalation); one ``kill_grace_s`` for the Compose output pump
    join; readiness timeout plus one poll overshoot; driver stop (teardown timeout plus
    SIGTERM/SIGKILL waits); port release timeout plus one poll overshoot. Local file I/O and
    hashing are excluded. ``kill_grace_s`` is the runner's agent technical default (D-018).
    """
    if not _check_run_config(run_config, []):
        raise ValueError("Invalid run config")
    if type(kill_grace_s) not in (int, float) or not math.isfinite(kill_grace_s) \
            or kill_grace_s < 0:
        raise ValueError("kill_grace_s must be a nonnegative finite number")
    poll = run_config["driver_poll_interval_s"]
    total = sum(run_config[name] + 3 * kill_grace_s for name in _RUN_CALL_TIMEOUTS)
    total += kill_grace_s
    total += run_config["driver_ready_timeout_s"] + poll
    total += run_config["teardown_timeout_s"] + 2 * kill_grace_s
    total += run_config["port_release_timeout_s"] + poll
    return total


def _check_rebuild(packet, issues, builder):
    """Bind the packet to its inputs: request hash, then a byte-identical rebuild.

    ``builder`` is the packet kind's own builder (``PACKET_KINDS``); the rebuilt packet must
    also carry the same ``artifact_kind``.
    """
    name = builder.__name__
    try:
        request_hash = identity_hash(packet.get("inputs"))
    except (TypeError, ValueError):
        request_hash = None
    if request_hash is None or request_hash != packet.get("request_sha256"):
        _issue(issues, "request_hash_mismatch", "packet.request_sha256",
               "request_sha256 is not the identity hash of packet.inputs")
        return
    try:
        rebuilt = builder(packet["inputs"])
        same = (rebuilt.get("prepared") is True
                and rebuilt.get("artifact_kind") == packet["artifact_kind"]
                and identity_hash(rebuilt.get("units")) == identity_hash(packet["units"])
                and identity_hash(rebuilt.get("schedule")) == identity_hash(packet["schedule"])
                and ("diagnostic_units" in rebuilt) == ("diagnostic_units" in packet)
                and identity_hash(rebuilt.get("diagnostic_units")) == identity_hash(
                    packet.get("diagnostic_units")))
    except Exception as exc:  # noqa: BLE001 - an unbuildable input set is a refusal
        _issue(issues, "packet_rebuild_failed", "packet.inputs",
               f"{name}(inputs) raised {type(exc).__name__}")
        return
    if not same:
        _issue(issues, "packet_rebuild_mismatch", "packet.units",
               f"units/schedule differ from {name}(packet.inputs): the packet was "
               "edited after preparation")


def shm_override(shm_size):
    """Compose override mapping for non-default ``shm_size`` values; None when all default.

    ``gate.s2_launch.check_generated_compose`` verifies the merged effective value.
    """
    if type(shm_size) is not dict or not all(type(k) is str for k in shm_size):
        raise ValueError("shm_size must map Compose services to declared values")
    services = {name: {"shm_size": value} for name, value in sorted(shm_size.items())
                if value != "docker_default"}
    if any(type(item["shm_size"]) is not str for item in services.values()):
        raise ValueError("Non-default shm_size values must be Docker size strings")
    return {"services": services} if services else None


def _unit_plan(unit, config, index, *, work_root="units"):
    job = unit["job"]
    run_dir = unit["compose"]["cwd"]
    capture_dir = unit["capture_dir"]
    driver_log_dir = str(PurePosixPath(unit["driver_output_path"]).parent)
    port = unit["generated_config_expectations"]["driver_port"]
    captures = {item["id"]: item for item in unit["required_captures_before_teardown"]}
    work = f"{work_root}/{index:03d}"

    def step(step_id, stage, on_failure="stop_run", **fields):
        return {"id": step_id, "stage": stage, "on_failure": on_failure,
                "always_after": None, **fields}

    teardown_after = ["driver_start", "compose_up"]
    steps = [
        step("budget_check", "pre_side_effect", job_id=job["job_id"]),
        step("ledger_started", "pre_side_effect", attempt_id=attempt_id(job["job_id"], 1),
             evidence_refs=[f"packet_unit:{unit['commands_sha256']}"]),
        step("run_dirs_absent", "pre_side_effect", paths=[run_dir, driver_log_dir]),
        step("wizard_prepare", "pre_gpu", command=unit["wizard_prepare"],
             stdout_journal_path=f"{work}/wizard-prepare.stdout.txt",
             timeout_s=config["wizard_prepare_timeout_s"]),
        step("generated_config_check", "pre_gpu", run_dir=run_dir,
             generated_user_config="generated-user-config-0.yaml",
             driver_config="driver-config.yaml",
             expected=unit["generated_config_expectations"],
             report_path=capture_dir + "/generated-config-check.json"),
        step("compose_shm_override", "pre_gpu", run_dir=run_dir, name=COMPOSE_OVERRIDE_NAME,
             conflicting_names=list(COMPOSE_OVERRIDE_NAMES),
             content=shm_override(unit["compose_expectations"]["shm_size"])),
        step("compose_config_check", "pre_gpu", command=unit["compose_config"],
             expected=unit["compose_expectations"],
             report_path=capture_dir + "/compose-check.json",
             timeout_s=config["compose_config_timeout_s"]),
        step("driver_port_free", "pre_gpu", port=port),
        step("driver_start", "gpu", command=unit["driver"],
             stdout_path=unit["driver_output_path"]),
        step("driver_ready", "gpu", port=port, log_path=unit["driver_output_path"],
             timeout_s=config["driver_ready_timeout_s"],
             poll_interval_s=config["driver_poll_interval_s"]),
        step("compose_up", "gpu", "record", command=unit["compose"],
             stdout_path=captures["compose_up_stdout"]["stdout_path"],
             timeout_s=config["compose_up_timeout_s"]),
    ]
    for capture_id in CAPTURE_IDS:
        steps.append(step(f"capture:{capture_id}", "capture", "record",
                          capture=captures[capture_id], timeout_s=config["capture_timeout_s"]))
    steps += [
        step("run_metadata_capture", "capture", "record", run_dir=run_dir,
             name="run_metadata.yaml"),
        step("run_dir_hash", "capture", "record", root=run_dir,
             manifest_path=capture_dir + "/run-dir-manifest.json"),
        step("compose_down", "teardown", command=unit["teardown"],
             timeout_s=config["teardown_timeout_s"]),
        step("driver_stop", "teardown", timeout_s=config["teardown_timeout_s"]),
        step("driver_port_released", "teardown", port=port,
             timeout_s=config["port_release_timeout_s"],
             poll_interval_s=config["driver_poll_interval_s"]),
        step("reconcile", "record", "record", run_dir=run_dir,
             scene_id=job["scene_id"], requested_steps=job["requested_steps"],
             summary_name="aggregate/results-summary.json",
             report_path=capture_dir + "/unit-attempt-evidence.json"),
        step("ledger_terminal", "record", "record", job_id=job["job_id"]),
    ]
    for item in steps:
        if item["stage"] == "teardown":
            item["always_after"] = list(teardown_after)
    assert [item["id"] for item in steps] == list(UNIT_STEPS)
    plan = {"index": index, "job_id": job["job_id"], "scene_id": job["scene_id"],
            "run_dir": run_dir, "driver_log_dir": driver_log_dir, "capture_dir": capture_dir,
            "port": port, "steps": steps}
    if "diagnostic" in unit:
        plan["diagnostic"] = dict(unit["diagnostic"])
    return plan


def build_execution_plan(packet, run_config):
    """Validate a prepared packet and the run config; emit the ordered unit plans (L1)."""
    issues = []
    try:
        packet_hash = identity_hash(packet)
    except (TypeError, ValueError):
        packet_hash = None
    try:
        config_hash = identity_hash(run_config)
    except (TypeError, ValueError):
        config_hash = None
    report = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION, "scope": None,
        "producer": {"module": "gate.s2_executor", "function": "build_execution_plan",
                     "source_sha256": _SOURCE_SHA256},
        "input_hashes": {"packet": packet_hash, "run_config": config_hash},
        "units": {"timeouts": "seconds"},
        "status": "blocked", "issues": issues, "unit_plans": [],
        "commands_executed": False, "cloud_launch_authorized": False,
        "ledger_event_schema_version": LEDGER_EVENT_SCHEMA_VERSION,
    }
    if packet_hash is None or type(packet) is not dict:
        _issue(issues, "invalid_packet", "packet", "Packet must be strict JSON")
        return report
    kind = packet.get("artifact_kind")
    if type(kind) is not str or kind not in PACKET_KINDS:
        _issue(issues, "unsupported_packet", "packet",
               "Require one of: " + "; ".join(
                   f"an {name} of schema version {version}"
                   for name, (version, _builder) in PACKET_KINDS.items()))
        return report
    version, builder = PACKET_KINDS[kind]
    if type(packet.get("schema_version")) is not int or packet["schema_version"] != version:
        _issue(issues, "unsupported_packet", "packet",
               f"Require an {kind} of schema version {version}")
        return report
    if packet.get("scope") not in SCOPES:
        _issue(issues, "invalid_scope", "packet.scope", "Packet scope must be declared")
        return report
    report["scope"] = packet["scope"]
    if packet.get("prepared") is not True or any(
            type(item) is dict and item.get("severity") == "blocking"
            for item in packet.get("issues") or []):
        _issue(issues, "packet_not_prepared", "packet.prepared",
               "Only a prepared packet without blocking issues can be executed")
        return report
    units, schedule = packet.get("units"), packet.get("schedule")
    if type(units) is not list or not units or type(schedule) is not list:
        _issue(issues, "invalid_packet", "packet.units", "Require nonempty units and schedule")
        return report
    job_ids = [unit.get("job", {}).get("job_id") if type(unit) is dict and type(
        unit.get("job")) is dict else None for unit in units]
    if len(set(job_ids)) != len(job_ids) or None in job_ids:
        _issue(issues, "duplicate_or_missing_job", "packet.units", "Job IDs must be unique")
    run_dirs = set()
    for position, unit in enumerate(units):
        _check_unit(packet, unit, position, issues, run_dirs)
    diagnostics = packet.get("diagnostic_units")
    if "diagnostic_units" in packet:
        if kind != "s2_pilot_packet" or type(diagnostics) is not list or not diagnostics:
            _issue(issues, "invalid_diagnostic_unit", "packet.diagnostic_units",
                   "Diagnostic units are a nonempty list on a pilot packet only")
            diagnostics = []
        ids = [unit.get("job", {}).get("job_id") if type(unit) is dict and type(
            unit.get("job")) is dict else None for unit in diagnostics]
        if len(set(ids)) != len(ids) or None in ids or set(ids) & set(job_ids):
            _issue(issues, "duplicate_or_missing_job", "packet.diagnostic_units",
                   "Diagnostic job IDs must be unique and distinct from the pilot jobs")
        for position, unit in enumerate(diagnostics):
            _check_unit(packet, unit, position, issues, run_dirs, diagnostic=True)
    _check_rebuild(packet, issues, builder)
    if not _check_run_config(run_config, issues) or issues:
        return report
    try:
        report["unit_plans"] = [_unit_plan(unit, run_config, index)
                                for index, unit in enumerate(units)]
        if "diagnostic_units" in packet:
            report["diagnostic_unit_plans"] = [
                _unit_plan(unit, run_config, index, work_root="diagnostics")
                for index, unit in enumerate(diagnostics)]
            report["diagnostic_schedule"] = [dict(unit["job"]) for unit in diagnostics]
    except (KeyError, TypeError, ValueError) as exc:
        _issue(issues, "invalid_packet_unit", "packet.units", f"{type(exc).__name__}: {exc}")
        report["unit_plans"] = []
        report.pop("diagnostic_unit_plans", None)
        report.pop("diagnostic_schedule", None)
        return report
    report["worst_case_unit_seconds_excluding_kill_grace"] = worst_case_unit_seconds(
        run_config, kill_grace_s=0)
    report["status"] = "passed"
    return report


# ----------------------------------------------------------------- pure step helpers

_READY_PREFIXES = (
    # Hydra default job_logging: ``[%(asctime)s][%(name)s][%(levelname)s] - `` with the
    # logging default asctime ``YYYY-MM-DD HH:MM:SS,mmm``.
    r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]\[[^\[\]\s]+\]\[INFO\] - ",
    # driver main.py:1116-1119 basicConfig: ``%(asctime)s.%(msecs)03d %(levelname)s:\t``.
    r"\d{2}:\d{2}:\d{2}\.\d{3} INFO:\t",
)


def driver_ready_line(text, port):
    """True when the pinned driver start line for ``port`` appears in ``text`` (L2).

    main.py:1048-1054 logs ``Starting %s driver on %s (external IP: %s:%d)`` after
    ``server.start()`` with model_type ``alpamayo1_5`` and address ``<host>:<port>`` (host
    ``0.0.0.0``, configs/driver/alpamayo1_5.yaml); the message must end the line. The line
    prefix is Hydra's default job_logging (the driver is ``@hydra.main``) or main.py's
    basicConfig format; which one the host prints is ``qualify_on_host``.
    """
    if type(text) is not str or type(port) is not int:
        return False
    message = (rf"Starting alpamayo1_5 driver on 0\.0\.0\.0:{port} "
               rf"\(external IP: [^\s()]+:{port}\)")
    pattern = re.compile(rf"^(?:{'|'.join(_READY_PREFIXES)}){message}$")
    return any(pattern.match(line.removesuffix("\r")) for line in text.split("\n"))


def container_ids(text):
    """Container IDs from ``docker compose ps -a -q`` stdout; ValueError when malformed."""
    if type(text) is not str:
        raise ValueError("Expected decoded text")
    ids = [line.strip() for line in text.splitlines() if line.strip()]
    if not ids or not all(_CONTAINER_ID.fullmatch(item) for item in ids) or len(
            set(ids)) != len(ids):
        raise ValueError("Expected one or more unique lowercase hex container IDs")
    return ids


def inspect_argv(capture, ids):
    """Expand the packet's ``container_inspect`` template with the captured IDs."""
    template, placeholder = capture.get("argv_template"), capture.get("placeholder")
    if type(template) is not list or template.count(placeholder) != 1:
        raise ValueError("container_inspect needs a template with one placeholder")
    position = template.index(placeholder)
    return [*template[:position], *ids, *template[position + 1:]]


# ----------------------------------------------------------------- ledger v2 events

def attempt_id(job_id, number):
    return f"{job_id}/attempt-{number}"


def _event(job_id, number, kind, timestamp_us, refs, **fields):
    return {"schema_version": LEDGER_EVENT_SCHEMA_VERSION,
            "event_id": f"{attempt_id(job_id, number)}:{kind}", "job_id": job_id,
            "attempt_id": attempt_id(job_id, number), "type": kind,
            "timestamp_us": timestamp_us, "evidence_refs": list(dict.fromkeys(refs)), **fields}


def started_event(job_id, number, timestamp_us, refs, *, parent=None, retry_reason=None):
    return _event(job_id, number, "started", timestamp_us, refs,
                  parent_attempt_id=None if parent is None else attempt_id(job_id, parent),
                  retry_reason=retry_reason, upstream_identity=None)


def cancelled_event(job_id, number, timestamp_us, reason, refs):
    if type(reason) is not str or not reason.strip():
        raise ValueError("A cancellation needs a reason")
    return _event(job_id, number, "cancelled", timestamp_us, refs, reason=reason,
                  upstream_identity=None)


def _order_key(entry, log_count):
    lines = [[] for _ in range(log_count)]
    for ref in entry.get("evidence_refs") or []:
        match = _LOG_REF.match(ref) if type(ref) is str else None
        if match and int(match[1]) < log_count:
            lines[int(match[1])].append(int(match[2]))
    big = float("inf")
    return (*(min(found) if found else big for found in lines),
            entry.get("rollout_uuid") or entry.get("job") or "")


def _upstream_attempts(report, log_count):
    entries = [dict(item, _kind="uuid") for item in report.get("attempts") or []]
    entries += [dict(item, _kind="pre_uuid") for item in report.get("pre_uuid_failures") or []
                if item.get("kind") == "rollout_failed"]
    entries.sort(key=lambda item: _order_key(item, log_count))
    if not entries and any(item.get("kind") == "summary_failure_row"
                           for item in report.get("pre_uuid_failures") or []):
        entries.append({"_kind": "summary_row_only", "evidence_refs": []})
    return entries


def has_upstream_attempt_evidence(report):
    """True when the reconcile report holds any upstream attempt the recorder binds (L1).

    False means ``unit_terminal_events`` closes attempt 1 with no upstream entry at all
    (``no_upstream_attempt_evidence``, or ``cancelled`` with the recorder's interruption
    reason when it stopped Compose); the runner treats that unit as systemic.
    """
    if type(report) is not dict or report.get("artifact_kind") != "s2_unit_attempt_evidence":
        raise ValueError("Expected a reconcile_unit_attempts report")
    return bool(_upstream_attempts(report, 0))


# Terminal reasons whose upstream evidence establishes how the attempt ended.
_ESTABLISHED_FAILURE_PREFIXES = ("upstream_rollout_failed:",
                                 "upstream_failure_before_rollout_uuid:")


def terminal_established(event):
    """True when a terminal event's evidence establishes how its attempt ended (L1).

    ``completed``/``process_completed``, or ``failed`` with a log-backed upstream failure
    reason. A ``cancelled`` attempt, ``completion_not_established:*``, an incomplete attempt,
    a summary failure row without a log line and an absent attempt are not established.
    """
    if type(event) is not dict or type(event.get("reason")) is not str:
        return False
    if event.get("type") == "completed":
        return event["reason"] == PROCESS_COMPLETED
    return (event.get("type") == "failed"
            and event["reason"].startswith(_ESTABLISHED_FAILURE_PREFIXES))


def unit_terminal_events(job_id, report, timestamp_us, evidence_ref, *, log_count,
                         interruption=None):
    """Ledger v2 events binding every upstream attempt of one unit (L1).

    ``report`` is the unmodified ``reconcile_unit_attempts`` output; ``evidence_ref``
    names the stored report (e.g. ``unit_attempt_evidence:<sha256>``); ``log_count`` is
    the number of logs given to the reconciler (orders attempts by line); ``interruption``
    is the recorder's reason when it stopped Compose (timeout), else None.
    """
    if type(report) is not dict or report.get("artifact_kind") != "s2_unit_attempt_evidence":
        raise ValueError("Expected a reconcile_unit_attempts report")
    blocking = sorted({item["code"] for item in report.get("issues") or []
                       if item.get("severity") == "blocking"})
    other = [code for code in blocking if code not in _TOLERATED_FOR_COMPLETION]
    binding = report.get("run_uuid_binding") or {}
    run_uuid = binding.get("run_uuid") if binding.get("status") == "passed" else None
    entries = _upstream_attempts(report, log_count)
    events = []
    if not entries:
        reason = interruption or "no_upstream_attempt_evidence"
        kind = "cancelled" if interruption else "failed"
        return [_event(job_id, 1, kind, timestamp_us, [evidence_ref], reason=reason,
                       upstream_identity=None)]
    retry = RETRY_REASON if report.get("internal_retries") else OTHER_ATTEMPT_REASON
    for number, entry in enumerate(entries, 1):
        refs = [evidence_ref, *sorted(set(entry.get("evidence_refs") or []))]
        if number > 1:
            events.append(started_event(job_id, number, timestamp_us, refs, parent=number - 1,
                                        retry_reason=retry))
        identity = None
        if entry["_kind"] == "uuid":
            clips = entry.get("clipgt_ids") or []
            if run_uuid and len(clips) == 1:
                identity = {"run_uuid": run_uuid, "clipgt_id": clips[0],
                            "rollout_id": entry["rollout_uuid"]}
            state = entry.get("state")
            if state == "complete" and not other and identity is not None:
                kind, reason = "completed", PROCESS_COMPLETED
            elif state == "complete":
                kind = "failed"
                reason = "completion_not_established:" + ",".join(
                    other or ["upstream_identity_unbound"])
            elif interruption and state != "failed":
                kind, reason = "cancelled", interruption
            elif state == "failed":
                kind = "failed"
                reason = f"upstream_rollout_failed:retryable={entry.get('retryable')}"
            else:
                kind, reason = "failed", "upstream_attempt_incomplete"
        elif entry["_kind"] == "pre_uuid":
            kind = "failed"
            reason = f"upstream_failure_before_rollout_uuid:retryable={entry.get('retryable')}"
        else:
            kind, reason = "failed", "upstream_failure_row_without_log"
        events.append(_event(job_id, number, kind, timestamp_us, refs, reason=reason,
                             upstream_identity=identity))
    return events
