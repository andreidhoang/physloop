"""Invented complete packet inputs; no asset or registration claim."""

from s2_launch_fixture import launch_request

from gate.s2_schedule import build_schedule


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
        "schema_version": 1,
        "launch_request": launch,
        "schedule_request": schedule,
        "execution_order": {
            "method": "alternating",
            "job_ids": [job["job_id"] for job in build_schedule(**schedule)],
        },
    }
