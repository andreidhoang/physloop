"""Invented paths/scene labels for pure pilot-command preparation tests only."""


def launch_request():
    return {
        "schema_version": 1,
        "alpasim_sha": "e121e59141dc391f6c295513d6dadc840a419149",
        "alpasim_checkout": "/synthetic/alpasim",
        "scene_ids": [f"synthetic-scene-{i}" for i in range(5)],
        "requested_steps": 17,
        "driver_port": 6789,
        "teacher_snapshot": "/synthetic/teacher/snapshot",
        "data_root": "/synthetic/data",
        "renderer_image": "nvcr.io/nvidia/nre/nre-ga@sha256:" + "a" * 64,
        "base_image": "registry.example/synthetic-alpasim@sha256:" + "b" * 64,
        "arms": [
            {
                "harmonizer": False,
                "run_dir": "/synthetic/runs/off",
                "driver_log_dir": "/synthetic/driver-logs/off",
            },
            {
                "harmonizer": True,
                "run_dir": "/synthetic/runs/on",
                "driver_log_dir": "/synthetic/driver-logs/on",
            },
        ],
    }
