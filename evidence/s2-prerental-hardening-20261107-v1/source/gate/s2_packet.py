"""Pure command/schedule binding for the five-scene off/on S2 pilot.

The caller supplies both requests and all ten job IDs in an explicit order.
Preparation binds declared inputs, not their origin or scientific registration.
It performs no reads, launches, asset retrieval, UUID inference or approval.

Pinned source e121e59141dc391f6c295513d6dadc840a419149 supports a singleton
scene list (wizard schema.py/context.py), writes Compose/configs for run method
NONE (wizard.py), and requires a fresh driver config path/output for each unit.
The source fixtures and their Git/SHA256 manifest are retained under
tests/fixtures/alpasim_launch_upstream. Actual host execution is unqualified.

These units reuse the external port and host network. They require strictly
serial execution, prior Compose/driver teardown, and a new driver per unit.
All ten startup periods are paid work; no measured cost or efficiency follows.
"""

import json
from itertools import pairwise

from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.s2_launch import _command_group, _overlap, build_pilot_launch
from gate.s2_schedule import build_schedule

_FIELDS = {"schema_version", "launch_request", "schedule_request", "execution_order"}
_SCHEDULE_FIELDS = {
    "study_id", "phase", "scenes", "policies", "protocol_hashes", "replicates",
    "requested_steps",
}


def build_pilot_packet(request):
    """Validate exact pilot coverage and emit ordered singleton command units.

    ``prepared`` establishes consistency of declared requests and source-checked
    command templates. A caller's ``preregistered_permutation`` label is retained
    as a declaration; neither registration nor randomness is authenticated.
    Any issue leaves ``units`` empty. Valid JSON inputs remain in the report for
    diagnosis, including invalid schedules/orders; non-JSON input is rejected.
    """
    report = {
        "schema_version": 1,
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
        "commands_executed": False,
        "execution_model": "serial_units_with_driver_restart_and_compose_teardown",
        "startup_cost_basis": "unmeasured_per_unit_startup_included_in_paid_work",
        "validation_scope": "declared_input_binding_and_pinned_source_command_preparation",
    }

    def issue(code, path, message):
        report["issues"].append({"code": code, "path": path, "message": message})

    try:
        snapshot = json.loads(canonical_json_bytes(request))
    except (ValueError, TypeError, UnicodeError) as exc:
        issue("invalid_request", "request", str(exc))
        return report
    report["inputs"] = snapshot
    report["request_sha256"] = identity_hash(snapshot)
    if type(snapshot) is not dict or set(snapshot) != _FIELDS:
        issue("invalid_request", "request", "Provide exactly the four packet fields.")
        return report
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1:
        issue("unsupported_schema", "schema_version", "Only packet version 1 is supported.")
    launch_request = snapshot["launch_request"]
    schedule_request = snapshot["schedule_request"]
    order = snapshot["execution_order"]
    report["input_hashes"] = {
        key: identity_hash(snapshot[key])
        for key in ("launch_request", "schedule_request", "execution_order")
    }
    launch = build_pilot_launch(launch_request)
    for item in launch["issues"]:
        issue("invalid_launch_request", "launch_request." + item["path"], item["message"])
    try:
        if type(schedule_request) is not dict or set(schedule_request) != _SCHEDULE_FIELDS:
            raise ValueError("Provide exactly the schedule builder's required fields.")
        if schedule_request["phase"] != "harmonizer_pilot":
            raise ValueError("The packet supports only the five-scene Harmonizer pilot.")
        schedule = build_schedule(**schedule_request)
    except (ValueError, TypeError, UnicodeError) as exc:
        issue("invalid_schedule_request", "schedule_request", str(exc))
        return report
    report["schedule"] = schedule
    report["schedule_sha256"] = identity_hash(schedule)
    if launch["prepared"] and (
        set(launch_request["scene_ids"]) != {job["scene_id"] for job in schedule}
        or launch_request["requested_steps"] != schedule_request["requested_steps"]
    ):
        issue(
            "launch_schedule_mismatch", "launch_request",
            "Launch and schedule must name exactly the same scenes and requested horizon.",
        )
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
        issue(
            "invalid_execution_order", "execution_order",
            "Supply all ten unique scheduled job IDs and an explicit supported order method.",
        )
    elif order["method"] == "alternating":
        modes = [protocols[jobs[job_id]["protocol_hash"]] for job_id in order["job_ids"]]
        if any(first == second for first, second in pairwise(modes)):
            issue(
                "invalid_execution_order", "execution_order.job_ids",
                "An alternating order must change Harmonizer mode after every unit.",
            )
    if report["issues"]:
        return report

    assets = {row["scene_id"]: row["asset_sha256"].lower() for row in schedule_request["scenes"]}
    arms = {"on" if arm["harmonizer"] else "off": arm for arm in launch_request["arms"]}
    units, output_paths = [], []
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
            issue(
                "overlapping_unit_outputs", "launch_request.arms",
                "Derived output directories of different jobs must not equal or contain one another.",
            )
        output_paths.extend(current_paths)
        commands = _command_group(launch_request, isolated_arm, [job["scene_id"]])
        commands.pop("harmonizer")
        units.append({
            "index": index,
            "job": dict(job),
            "harmonizer_mode": mode,
            "asset_sha256": assets[job["scene_id"]],
            "upstream_identity": None,
            "launch_request_sha256": report["input_hashes"]["launch_request"],
            "schedule_request_sha256": report["input_hashes"]["schedule_request"],
            "commands_sha256": identity_hash(commands),
            "requires_previous_unit_teardown": True,
            **commands,
        })
    if report["issues"]:
        return report
    report["units"] = units
    report["prepared"] = True
    return report
