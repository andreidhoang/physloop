"""Invented complete packet inputs; no asset, registration or owner-decision claim.

The ego-hood receipt repeats the pinned AlpaSim blob identities with
``scope: synthetic``: it declares, and does not prove, host staging. The
``shm_size`` values are synthetic placeholders matching the generator's output.
"""

import copy

from s2_launch_fixture import launch_request

from gate.s2_packet import EGO_HOOD_FILES
from gate.s2_schedule import build_schedule


def ego_hood_staging(data_root="/synthetic/data"):
    return {
        "schema_version": 1,
        "artifact_kind": "ego_hood_staging_receipt",
        "scope": "synthetic",
        "staged_root": data_root + "/nre-artifacts/ego-hoods",
        "files": copy.deepcopy(list(EGO_HOOD_FILES)),
    }


def shm_size():
    return {name: "docker_default"
            for name in ("controller-0", "physics-0", "renderer-0", "runtime-0")}


def diagnostic_unit(source_job_id, diagnostic_id="camera-check", **changes):
    """One invented diagnostic qualification declaration (the R2 camera/config check).

    The role, line and cap are synthetic test inputs, not an owner budget decision.
    """
    return {"diagnostic_id": diagnostic_id, "role": "camera_config_qualification",
            "source_job_id": source_job_id,
            "budget_line": {"line_id": "pilot_diagnostics", "cap_usd": "5"}, **changes}


def packet_request():
    launch = launch_request()
    launch["requested_steps"] = 3
    schedule = {
        "study_id": "synthetic-pilot-rehearsal",
        "phase": "harmonizer_pilot",
        "scenes": [
            {
                "scene_id": f"synthetic-scene-{index}",
                "road_id": f"synthetic-road-{index}",
                "stratum": "synthetic-urban",
                "asset_sha256": str(index) * 64,
            }
            for index in range(5)
        ],
        "policies": {"A": "synthetic-teacher"},
        "protocol_hashes": {"off": "a" * 64, "on": "b" * 64},
        "replicates": 1,
        "requested_steps": 3,
    }
    return {
        "schema_version": 2,
        "scope": "synthetic",
        "launch_request": launch,
        "schedule_request": schedule,
        "execution_order": {
            "method": "alternating",
            "job_ids": [job["job_id"] for job in build_schedule(**schedule)],
        },
        "ego_hood_staging": ego_hood_staging(launch["data_root"]),
        "shm_size": shm_size(),
    }


def full_sweep_request(replicates=3, scene_count=441):
    """Invented registered full sweep: synthetic scenes, roads, snapshots and order.

    Scene IDs are zero-padded so the schedule's sorted order equals the listed order.
    Every value here is a synthetic test input, not a registered owner decision.
    """
    schedule = {
        "study_id": "synthetic-full-sweep",
        "phase": "full_sweep",
        "scenes": [
            {
                "scene_id": f"synthetic-scene-{index:03d}",
                "road_id": f"synthetic-road-{index // 3:03d}",
                "stratum": "synthetic-urban" if (index // 3) % 2 else "synthetic-highway",
                "asset_sha256": f"{index:064x}",
            }
            for index in range(scene_count)
        ],
        "policies": {"A": "synthetic-teacher-bf16", "B": "synthetic-teacher-reexport"},
        "protocol_hashes": {"fixed": "c" * 64},
        "replicates": replicates,
        "requested_steps": 3,
    }
    launch = launch_request()
    sweep_launch = {
        key: launch[key] for key in (
            "alpasim_sha", "alpasim_checkout", "data_root", "hf_home", "renderer_image",
            "base_image", "renderer_camera_override", "harmonizer_source",
            "harmonizer_terms_authorization_ref")
    }
    sweep_launch.update({
        "requested_steps": 3,
        "driver_port_base": 20000,
        "harmonizer": False,
        "policies": {
            "A": {"snapshot": "/synthetic/teacher/snapshot", "run_dir": "/synthetic/runs/A",
                  "driver_log_dir": "/synthetic/driver-logs/A"},
            "B": {"snapshot": "/synthetic/reexport/snapshot", "run_dir": "/synthetic/runs/B",
                  "driver_log_dir": "/synthetic/driver-logs/B"},
        },
    })
    try:
        job_ids = [job["job_id"] for job in build_schedule(**schedule)]
    except ValueError:
        job_ids = []
    return {
        "schema_version": 1,
        "artifact_kind": "s2_full_sweep_request",
        "scope": "synthetic",
        "launch": sweep_launch,
        "schedule_request": schedule,
        "execution_order": {"method": "preregistered_permutation", "job_ids": job_ids},
        "ego_hood_staging": ego_hood_staging(launch["data_root"]),
        "shm_size": shm_size(),
    }
