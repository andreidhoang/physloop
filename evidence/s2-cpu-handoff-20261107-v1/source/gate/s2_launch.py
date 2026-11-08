"""Pure five-scene pilot command preparation for the reviewed AlpaSim pin.

Source anchors: ``docs/research/2026-11-07-pilot-launch-audit.md`` and upstream
``src/wizard/configs/{base_config,e2e_challenge/dev,e2e_challenge/ec2,
driver/alpamayo1_5,topology/1gpu,nurec_scenes/curated_val}.yaml`` at ALPASIM_SHA.
The external driver invocation is assembled from base_config's container command,
configuration.py's driver-config output and driver/main.py's Hydra interface.
Host paths replace /mnt/output; --no-sync presumes an already prepared environment.
This assembled template has not been runtime tested.

Only portable ASCII paths and scene labels are supported. The pinned wizard
joins service arguments into downstream shell commands; Python argv boundaries
alone cannot establish correct handling of arbitrary names through that path.
Spaces, quotes, backslashes, interpolation and shell metacharacters are rejected.
Both arms request the source-documented --enable-timing option required by the
local S2 pilot contract; its actual runtime behavior still requires qualification.

No wizard import, filesystem access, subprocess, dependency resolution or launch
occurs here. Wizard run_method=NONE STILL FETCHES ASSETS; the emitted preparation
command is not a harmless dry run. All paths/images and effective host state remain
unverified. Graph off means the pinned stock wrapper makes no graph-enable call;
there is no invented graph/seed override. The unchanged driver remains unseeded.
"""

import json
import re
from pathlib import PurePosixPath

from gate.s2_identity import identity_hash

ALPASIM_SHA = "e121e59141dc391f6c295513d6dadc840a419149"
_FIELDS = {
    "schema_version",
    "alpasim_sha",
    "alpasim_checkout",
    "scene_ids",
    "requested_steps",
    "driver_port",
    "teacher_snapshot",
    "data_root",
    "renderer_image",
    "base_image",
    "arms",
}
_ARM_FIELDS = {"harmonizer", "run_dir", "driver_log_dir"}
_RENDERER_BEFORE_HARMONIZER = [
    "/app/internal/scripts/pycena/runtime/pycena_nrm_full",
    "serve-grpc",
    "--port={port}",
    "--host=0.0.0.0",
    "--artifact-glob=/mnt/nre-data/{sceneset}/**/*.usdz",
    "--egocar-hood-dir=/mnt/ego-hoods",
    "--no-enable-nrend",
]
_RENDERER_AFTER_HARMONIZER = [
    "--download-cache-dir /tmp/nre-cache-dir",
    "--cache-size=5",
    "--max-workers=4",
    "--enable-editing-actors",
    "--enable-timing",
]


def _text(value):
    """Block OmegaConf interpolation and control bytes; never shell-escape argv."""
    return (
        type(value) is str
        and bool(value.strip())
        and "${" not in value
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _path(value):
    if not _text(value) or re.fullmatch(r"/[A-Za-z0-9._/-]+", value) is None:
        return False
    path = PurePosixPath(value)
    return (
        path.is_absolute()
        and str(path) != "/"
        and all(component not in ("", ".", "..") for component in value.split("/")[1:])
    )


def _scene(value):
    return (
        _text(value)
        and value not in (".", "..")
        and re.fullmatch(r"[A-Za-z0-9._-]+", value) is not None
    )


def _image(value):
    return (
        _text(value)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", value) is not None
    )


def _overlap(first, second):
    first, second = PurePosixPath(first), PurePosixPath(second)
    return first == second or first in second.parents or second in first.parents


def _hydra(value):
    # Values here contain only validated portable input alphabets or fixed source
    # constants. JSON escaping is NOT a general Hydra string round-trip codec:
    # Hydra retains some backslashes that JSON would unescape. Never broaden the
    # accepted alphabet without qualifying both Hydra and downstream shell use.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def build_pilot_launch(request):
    """Return explicit argv lists for two five-scene arms, never execute them.

    ``prepared`` means syntax and pinned command-contract consistency only. It
    does not establish scene selection, protocol registration, existing/verified
    snapshot bytes, installed runtime, access rights, image availability or fit.
    The fixed off/on representation does not register an execution ordering; the
    wizard sorts scene selection, so pilot order still needs explicit review.
    """
    report = {
        "schema_version": 1,
        "prepared": False,
        "alpasim_sha": ALPASIM_SHA,
        "command_origin": "assembled_from_pinned_source_interfaces_not_runtime_tested",
        "validation_scope": "pure source-contract command preparation",
        "cloud_launch_authorized": False,
        "publication_authorized": False,
        "runtime_qualified": False,
        "paths_verified": False,
        "images_verified": False,
        "execution_order_registered": False,
        "commands_executed": False,
        "input_name_scope": "portable_ASCII_paths_and_scene_labels_only",
        "issues": [],
        "pilot": None,
        "arms": [],
    }

    def issue(code, path, message):
        report["issues"].append({"code": code, "path": path, "message": message})

    if type(request) is not dict or set(request) != _FIELDS:
        issue("invalid_request", "request", "Provide exactly the required pilot command fields.")
        return report
    try:
        report["request_sha256"] = identity_hash(request)
    except (ValueError, TypeError, UnicodeError) as exc:
        issue("invalid_request", "request", str(exc))
        return report
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        issue(
            "unsupported_schema", "schema_version", "Only command request version 1 is supported."
        )
    if request["alpasim_sha"] != ALPASIM_SHA:
        issue(
            "unsupported_source",
            "alpasim_sha",
            "Command contract requires the reviewed source pin.",
        )
    for name in ("alpasim_checkout", "teacher_snapshot", "data_root"):
        if not _path(request[name]):
            issue(
                "invalid_path",
                name,
                "Require an absolute non-root path using only ASCII letters, digits, slash, dot, underscore and hyphen; no empty/dot/traversal segments.",
            )
    scenes = request["scene_ids"]
    if (
        type(scenes) is not list
        or len(scenes) != 5
        or not all(_scene(scene) for scene in scenes)
        or (all(type(scene) is str for scene in scenes) and len(set(scenes)) != 5)
    ):
        issue(
            "invalid_scene_selection",
            "scene_ids",
            "Provide exactly five unique scene IDs using ASCII letters, digits, dot, underscore or hyphen; dot/traversal labels are unsupported.",
        )
    if type(request["requested_steps"]) is not int or request["requested_steps"] <= 0:
        issue(
            "invalid_requested_steps",
            "requested_steps",
            "Provide a positive requested horizon; no default is adopted.",
        )
    port = request["driver_port"]
    if type(port) is not int or not 1 <= port <= 65535:
        issue("invalid_port", "driver_port", "Provide an explicit port in 1..65535.")
    for name in ("renderer_image", "base_image"):
        if not _image(request[name]):
            issue(
                "invalid_image",
                name,
                "Require an immutable image reference ending in @sha256:<64 lowercase hex>.",
            )
    arms = request["arms"]
    if type(arms) is not list or len(arms) != 2:
        issue("invalid_arms", "arms", "Provide exactly the off and on arm definitions.")
    else:
        paths_valid = True
        for index, arm in enumerate(arms):
            if type(arm) is not dict or set(arm) != _ARM_FIELDS:
                issue(
                    "invalid_arm",
                    f"arms[{index}]",
                    "Arm fields are harmonizer, run_dir, driver_log_dir.",
                )
                paths_valid = False
                continue
            if type(arm["harmonizer"]) is not bool or arm["harmonizer"] != bool(index):
                issue(
                    "invalid_arm",
                    f"arms[{index}].harmonizer",
                    "Represent off first and on second; this does not register execution order.",
                )
            for name in ("run_dir", "driver_log_dir"):
                if not _path(arm[name]):
                    issue(
                        "invalid_path",
                        f"arms[{index}].{name}",
                        "Require an absolute portable-ASCII output path without spaces, shell characters or empty/dot/traversal segments.",
                    )
                    paths_valid = False
        if paths_valid:
            for first in (arms[0]["run_dir"], arms[0]["driver_log_dir"]):
                for second in (arms[1]["run_dir"], arms[1]["driver_log_dir"]):
                    if _overlap(first, second):
                        issue(
                            "overlapping_arm_outputs",
                            "arms",
                            "Off/on output directories must be disjoint, including ancestors.",
                        )
    if report["issues"]:
        return report

    report["pilot"] = {
        "scene_ids": list(scenes),
        "scene_count": 5,
        "replicates": 1,
        "policy_rollouts": 10,
        "requested_steps": request["requested_steps"],
        "seed_mode": "unseeded",
        "graph_mode": "stock_wrapper_no_graph_enable_call",
        "max_batch_size": 1,
        "schedule_status": "cardinality_only_execution_order_not_registered",
        "source_defaults": {
            "camera_config": "6cam_1080",
            "control_timestep_us": 100_000,
            "force_gt_duration_us": 1_700_000,
            "planner_delay_us": 0,
            "render_bundling": "NONE",
            "traffic_mode": "log_replay",
        },
    }
    for arm in arms:
        renderer = list(_RENDERER_BEFORE_HARMONIZER)
        if arm["harmonizer"]:
            renderer.append("--enable-harmonizer")
        renderer.extend(_RENDERER_AFTER_HARMONIZER)
        argv = [
            "uv",
            "run",
            "--no-sync",
            "alpasim_wizard",
            "driver=alpamayo1_5",
            "+e2e_challenge=dev",
            "+nurec_scenes=curated_val",
            "scenes.test_suite_id=null",
            f"scenes.scene_ids={_hydra(scenes)}",
            "scenes.limit_to_first_n=0",
            "runtime.simulation_config.n_rollouts=1",
            f"runtime.simulation_config.n_sim_steps={request['requested_steps']}",
            f"driver.port={port}",
            f"wizard.external_services.driver={_hydra([f'localhost:{port}'])}",
            f"driver.model.checkpoint_path={_hydra(request['teacher_snapshot'])}",
            f"driver.output_dir={_hydra(arm['driver_log_dir'])}",
            "driver.inference.max_batch_size=1",
            f"wizard.log_dir={_hydra(arm['run_dir'])}",
            "wizard.run_method=NONE",
            "wizard.nr_retries=0",
            "runtime.enable_autoresume=false",
            f"defines.filesystem={_hydra(request['data_root'])}",
            f"services.renderer.image={_hydra(request['renderer_image'])}",
            f"defines.base_image={_hydra(request['base_image'])}",
            "services.physics.external_image=true",
            "services.controller.external_image=true",
            "services.runtime.external_image=true",
            f"services.renderer.command={_hydra(renderer)}",
        ]
        report["arms"].append(
            {
                "harmonizer": arm["harmonizer"],
                "wizard_prepare": {
                    "cwd": request["alpasim_checkout"],
                    "argv": argv,
                    "may_download_assets": True,
                },
                "driver": {
                    "cwd": request["alpasim_checkout"],
                    "argv": [
                        "uv",
                        "run",
                        "--no-sync",
                        "-m",
                        "alpasim_driver.main",
                        f"--config-path={arm['run_dir']}",
                        "--config-name=driver-config.yaml",
                    ],
                },
                "compose": {
                    "cwd": arm["run_dir"],
                    "argv": [
                        "docker",
                        "compose",
                        "up",
                        "--exit-code-from",
                        "runtime-0",
                    ],
                },
            }
        )
    report["prepared"] = True
    return report
