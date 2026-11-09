"""Invented paths/scene labels for pure pilot-command preparation tests only.

``scope`` declares the invented inputs synthetic (shared contract rule 2.1).
Owner-decision fields carry synthetic placeholders that keep the pinned preset
and default Harmonizer download; they are test inputs, not chosen decisions. The
Harmonizer terms reference is an invented label, not a recorded acceptance.
"""


def launch_request():
    return {
        "schema_version": 2,
        "scope": "synthetic",
        "alpasim_sha": "e121e59141dc391f6c295513d6dadc840a419149",
        "alpasim_checkout": "/synthetic/alpasim",
        "scene_ids": [f"synthetic-scene-{i}" for i in range(5)],
        "requested_steps": 17,
        "driver_port": 6789,
        "teacher_snapshot": "/synthetic/teacher/snapshot",
        "data_root": "/synthetic/data",
        "hf_home": "/synthetic/hf-home",
        "renderer_image": "nvcr.io/nvidia/nre/nre-ga@sha256:" + "a" * 64,
        "base_image": "registry.example/synthetic-alpasim@sha256:" + "b" * 64,
        "renderer_camera_override": None,
        "harmonizer_source": {"kind": "upstream_default_download"},
        "harmonizer_terms_authorization_ref": "SYNTHETIC-harmonizer-terms",
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


def pinned_harmonizer_source():
    """Synthetic pinned-mount record; revision/hash values are invented."""
    return {
        "kind": "pinned_mount",
        "repository": "nvidia/Harmonizer",
        "revision": "c" * 40,
        "file": "harmonizer_nontemporal.pt",
        "sha256": "d" * 64,
        "size_bytes": 1,
        "host_dir": "/synthetic/harmonizer",
    }
