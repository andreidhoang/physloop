"""Pure five-scene pilot command preparation and generated-config checks.

Tickets: S2 CPU handoff E3; build-spec A4 R2 L1 (camera override and generated
camera check), L2 (registry-digest or local-build image references, Compose
``--pull never``), L4 (``hydra.run.dir``), L5 (per-command environment
allowlist), L8 (Harmonizer source as a required decision) and the Compose
``shm_size`` check. Serves S2.1-S2.7 [C] (G1, G2, G3-R1, G7).

Source anchors: ``docs/research/2026-11-07-pilot-launch-audit.md``,
``docs/research/2026-11-08-codebase-build/lane-r2-alpasim-runtime.md`` and the
hash-checked pinned excerpts under ``tests/fixtures/alpasim_launch_upstream``
(``manifest.json`` and ``manifest_runtime_capture.json``) at ALPASIM_SHA. Every
argv token below cites its pinned source line. Host paths replace /mnt/output;
``--no-sync`` presumes an already prepared environment. Not runtime tested.

Only portable ASCII paths and scene labels are supported. The pinned wizard
joins service arguments into downstream shell commands (services.py:207,
docker_compose.py:130-144); Python argv boundaries alone cannot establish
correct handling of arbitrary names through that path. Spaces, quotes,
backslashes, interpolation and shell metacharacters are rejected in inputs.
Both arms request the source-documented --enable-timing option required by the
local S2 pilot contract; its actual runtime behavior still requires qualification.

Owner decisions are required inputs with no default (build-spec section 6):
``renderer_camera_override`` (``null`` keeps dev's six-camera preset,
dev.yaml:11; ``"4cam_1080"`` adds the Hydra override ``cameras=4cam_1080``,
configs/cameras/4cam_1080.yaml) and ``harmonizer_source``
(``upstream_default_download`` or ``pinned_mount``). Driver-side camera changes
(R2 options b/c) are not built. ``hf_home`` is a required host path.

Licence gate (review A2-F12): every on-arm unit (``--enable-harmonizer``) uses
nvidia/Harmonizer under its licence; the upstream default downloads it in each
on-arm renderer and ``pinned_mount`` mounts a checkpoint bootstrap acquired.
``harmonizer_terms_authorization_ref`` is therefore a required input with no
default: the recorded reference that ``gate.s2_bootstrap`` resolves for
authorization id ``terms:harmonizer``. A missing, blank, placeholder or
credential-like value blocks preparation (``harmonizer_terms_unrecorded``). It is
bound to the on arm only and never enters argv or environments, so the off arm's
commands are unaffected. This checks that a reference is declared, not that it
matches the bootstrap plan or that the terms were accepted.

Agent technical default (D-018): the pinned Harmonizer mount uses container
path ``/mnt/harmonizer`` passed as ``--harmonizer-cache=/mnt/harmonizer``
(NRE 26.04 gRPC guide, nre-docs/grpc_api_guide_2604.html.txt:93, retrieved
copy sha256 bb26d3fd...e762; HTML sha256 a0bb2781...e736). Whether NRE reuses a
pre-placed checkpoint there without network access is ``qualify_on_host``.

F1 (build-spec B2): ``check_sweep_launch`` validates the full-sweep launch block (A/B
policy snapshots, one registered Harmonizer setting, per-unit driver ports from a
required ``driver_port_base``) and ``sweep_unit_request`` adapts it to the same pinned
command group; the five-scene pilot behaviour is unchanged.

Verification level per function: ``build_pilot_launch`` and ``check_sweep_launch`` L1
input logic and L2 argv against pinned source; ``sweep_unit_request`` L1; ``check_generated_configs``,
``check_generated_compose`` L2 over parsed generated files (the host runner
parses YAML; no YAML dependency is locked); ``filter_environment`` L1. Actual
execution, Hydra override resolution and Compose flag support are L3.

No wizard import, filesystem access (apart from hashing this module's own source
once at import for producer identity), subprocess, dependency resolution or launch
occurs here. Every report carries ``artifact_kind``, a declared ``scope``
(required input, no default) and ``producer`` (shared contract rule 2.1). Wizard run_method=NONE STILL FETCHES ASSETS; the emitted preparation
command is not a harmless dry run. All paths/images and effective host state remain
unverified. Graph off means the pinned stock wrapper makes no graph-enable call;
there is no invented graph/seed override. The unchanged driver remains unseeded.
This module never executes commands, reads environment variables, embeds
secrets, chooses owner decisions or infers success from an exit code.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from gate.s2_identity import identity_hash
from gate.s2_runtime import is_placeholder, secret_like

ALPASIM_SHA = "e121e59141dc391f6c295513d6dadc840a419149"
SCHEMA_VERSION = 2
# Shared contract rule 2.1 scopes; every report echoes a declared, validated scope.
SCOPES = ("synthetic", "public_source_fixture", "actual")


def _read_source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


# Hashed once at import so the builders and checks themselves perform no file I/O.
_SOURCE_SHA256 = _read_source_sha256()


def _producer(function):
    return {"module": "gate.s2_launch", "function": function, "source_sha256": _SOURCE_SHA256}


# The pinned wizard's NONE path builds three managed service endpoints twice:
# DockerComposeDeployment then SlurmDeployment. Each forward-only search spans
# at most 1000 ports. Leave all six windows within the TCP range. This does not
# prove host availability or reserve sockets against competing processes.
MAX_DRIVER_PORT = 65535 - 6 * 1000
_FIELDS = {
    "schema_version",
    "scope",
    "alpasim_sha",
    "alpasim_checkout",
    "scene_ids",
    "requested_steps",
    "driver_port",
    "teacher_snapshot",
    "data_root",
    "hf_home",
    "renderer_image",
    "base_image",
    "renderer_camera_override",
    "harmonizer_source",
    "harmonizer_terms_authorization_ref",
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
# Cameras (pinned YAML): dev.yaml:11 selects 6cam_1080 (cameras/6cam_1080.yaml:13-47);
# cameras/4cam_1080.yaml:12-32 lists four; the stock driver accepts only
# inference.use_cameras (driver/alpamayo1_5.yaml:32-36), and rejects any other
# camera image with ValueError (driver main.py:787).
CAMERA_PRESETS = {
    "6cam_1080": (
        "camera_front_wide_120fov", "camera_front_tele_30fov", "camera_cross_left_120fov",
        "camera_cross_right_120fov", "camera_rear_left_70fov", "camera_rear_right_70fov",
    ),
    "4cam_1080": (
        "camera_front_wide_120fov", "camera_front_tele_30fov", "camera_cross_left_120fov",
        "camera_cross_right_120fov",
    ),
}
STOCK_DRIVER_CAMERAS = (
    "camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov",
    "camera_front_tele_30fov",
)
_CAMERA_OVERRIDES = (None, "4cam_1080")
# Renderer volumes replace the whole list when overridden (services.py:169 builds
# them; VolumeMount.from_str splits on ":" so ":ro" is rejected, services.py:40-45).
# These two are the pinned base entries (base_config.yaml:125-126), kept verbatim.
_BASE_RENDERER_VOLUMES = (
    "${scenes.scene_cache}:/mnt/nre-data",
    "${defines.sensordata}/ego-hoods:/mnt/ego-hoods",
)
HARMONIZER_CONTAINER_DIR = "/mnt/harmonizer"
# Authorization id under which gate.s2_bootstrap records the Harmonizer terms reference.
HARMONIZER_TERMS_AUTHORIZATION_ID = "terms:harmonizer"
_HARMONIZER_PINNED_FIELDS = {"kind", "repository", "revision", "file", "sha256", "size_bytes",
                             "host_dir"}
# Environment (L5). Names inherited from the host are only those listed; values in
# _ENV_SET override. Excluded by construction: HF_TOKEN and other credentials,
# ALPASIM_DRIVER_HOST/PORT (dev.yaml:28), SLURM_ARRAY_TASK_ID/COUNT (services.py:217,
# configuration.py:333-357), NRE_IMAGE/CONTESTANT_IMAGE (dev.yaml:32,35), CUDA_*.
_ENV_INHERITED = ("HOME", "PATH")
_FORBIDDEN_ENV = re.compile(
    r"(?:.*(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_KEY|APIKEY).*|HF_TOKEN.*|"
    r"HUGGING_FACE_HUB_TOKEN|ALPASIM_.*|SLURM_.*|NRE_IMAGE|CONTESTANT_IMAGE|CUDA_.*)",
    re.IGNORECASE,
)
# Generated Compose services (R2 F2; service keys are container uuids,
# docker_compose.py:188-214); exact names are qualify_on_host.
COMPOSE_SERVICES = ("controller-0", "physics-0", "renderer-0", "runtime-0")


def _issue(report, code, path, message, severity="blocking", evidence_refs=()):
    report["issues"].append({"code": code, "path": path, "severity": severity,
                             "evidence_refs": list(evidence_refs), "message": message})


def _blocking(report):
    return any(item["severity"] == "blocking" for item in report["issues"])


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


def image_kind(value):
    """Classify an immutable image reference; ``None`` for anything mutable.

    ``registry_digest``: pullable ``repository@sha256:<64 hex>``.
    ``local_build``: a bare local image ID ``sha256:<64 hex>`` (the alpasim base
    image is built locally, base_config.yaml:26 ``alpasim-base:${repo-version:}``,
    and is never pullable). Generated services default to ``pull_policy:
    missing`` (schema.py:157), so the Compose argv adds ``--pull never``.
    """
    if not _text(value):
        return None
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", value):
        return "registry_digest"
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        return "local_build"
    return None


def _overlap(first, second):
    first, second = PurePosixPath(first), PurePosixPath(second)
    return first == second or first in second.parents or second in first.parents


def _hydra(value):
    # Values here contain only validated portable input alphabets or fixed source
    # constants. JSON escaping is NOT a general Hydra string round-trip codec:
    # Hydra retains some backslashes that JSON would unescape. Never broaden the
    # accepted alphabet without qualifying both Hydra and downstream shell use.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _validate_harmonizer(source, report):
    if type(source) is dict and source == {"kind": "upstream_default_download"}:
        return True
    if type(source) is not dict or source.get("kind") != "pinned_mount":
        _issue(report, "invalid_harmonizer_source", "harmonizer_source",
               "Owner decision required: {'kind': 'upstream_default_download'} or a "
               "pinned_mount record; no default is chosen.")
        return False
    if set(source) != _HARMONIZER_PINNED_FIELDS:
        _issue(report, "invalid_harmonizer_source", "harmonizer_source",
               f"pinned_mount requires exactly {sorted(_HARMONIZER_PINNED_FIELDS)}.")
        return False
    valid = (
        source["repository"] == "nvidia/Harmonizer"
        and source["file"] == "harmonizer_nontemporal.pt"
        and type(source["revision"]) is str
        and re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is not None
        and type(source["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None
        and type(source["size_bytes"]) is int and source["size_bytes"] > 0
        and _path(source["host_dir"])
    )
    if not valid:
        _issue(report, "invalid_harmonizer_source", "harmonizer_source",
               "pinned_mount needs nvidia/Harmonizer, harmonizer_nontemporal.pt, a 40-hex "
               "revision, a 64-hex sha256, positive size_bytes and a portable host_dir.")
    return valid


def _harmonizer_terms_recorded(ref):
    return _text(ref) and not is_placeholder(ref) and not secret_like(ref)


def build_pilot_launch(request):
    """Return explicit argv lists for two five-scene arms, never execute them.

    ``prepared`` means syntax and pinned command-contract consistency only. It
    does not establish scene selection, protocol registration, existing/verified
    snapshot bytes, installed runtime, access rights, image availability or fit.
    The fixed off/on representation does not register an execution ordering; the
    wizard sorts scene selection, so pilot order still needs explicit review.
    Warnings (``severity: warning``) never block preparation.
    """
    report = {
        "artifact_kind": "s2_pilot_launch",
        "schema_version": SCHEMA_VERSION,
        "scope": None,
        "producer": _producer("build_pilot_launch"),
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

    if type(request) is not dict or set(request) != _FIELDS:
        present = set(request) if type(request) is dict else set()
        _issue(report, "invalid_request", "request",
               f"Provide exactly the required pilot command fields; missing "
               f"{sorted(_FIELDS - present)}, unexpected {sorted(present - _FIELDS)}.")
        return report
    try:
        report["request_sha256"] = identity_hash(request)
    except (ValueError, TypeError, UnicodeError) as exc:
        _issue(report, "invalid_request", "request", str(exc))
        return report
    if type(request["scope"]) is str and request["scope"] in SCOPES:
        report["scope"] = request["scope"]
    else:
        _issue(report, "invalid_scope", "scope",
               f"Declare the input scope as one of {list(SCOPES)}; no default is chosen.")
    if type(request["schema_version"]) is not int or request["schema_version"] != SCHEMA_VERSION:
        _issue(report, "unsupported_schema", "schema_version",
               "Only command request version 2 is supported; version 1 lacked the "
               "required camera, Harmonizer and environment decisions.")
    if request["alpasim_sha"] != ALPASIM_SHA:
        _issue(report, "unsupported_source", "alpasim_sha",
               "Command contract requires the reviewed source pin.")
    for name in ("alpasim_checkout", "teacher_snapshot", "data_root", "hf_home"):
        if not _path(request[name]):
            _issue(report, "invalid_path", name,
                   "Require an absolute non-root path using only ASCII letters, digits, slash, "
                   "dot, underscore and hyphen; no empty/dot/traversal segments.")
    scenes = request["scene_ids"]
    if (
        type(scenes) is not list
        or len(scenes) != 5
        or not all(_scene(scene) for scene in scenes)
        or (all(type(scene) is str for scene in scenes) and len(set(scenes)) != 5)
    ):
        _issue(report, "invalid_scene_selection", "scene_ids",
               "Provide exactly five unique scene IDs using ASCII letters, digits, dot, "
               "underscore or hyphen; dot/traversal labels are unsupported.")
    if type(request["requested_steps"]) is not int or request["requested_steps"] <= 0:
        _issue(report, "invalid_requested_steps", "requested_steps",
               "Provide a positive requested horizon; no default is adopted.")
    port = request["driver_port"]
    if type(port) is not int or not 1 <= port <= MAX_DRIVER_PORT:
        _issue(report, "invalid_port", "driver_port",
               "Provide a port in 1..59535, leaving six upstream service-search windows above it.")
    for name in ("renderer_image", "base_image"):
        if image_kind(request[name]) is None:
            _issue(report, "invalid_image", name,
                   "Require an immutable registry digest (repository@sha256:<64 hex>) or a "
                   "local build image ID (sha256:<64 hex>); mutable tags are rejected.")
    override = request["renderer_camera_override"]
    if not (override is None or (type(override) is str and override in _CAMERA_OVERRIDES)):
        _issue(report, "invalid_camera_override", "renderer_camera_override",
               "Owner decision required: null (keep dev 6cam_1080) or '4cam_1080'.")
    harmonizer_ok = _validate_harmonizer(request["harmonizer_source"], report)
    arms = request["arms"]
    on_arms = ([f"arms[{index}]" for index, arm in enumerate(arms)
                if type(arm) is dict and arm.get("harmonizer") is True]
               if type(arms) is list else [])
    if on_arms and not _harmonizer_terms_recorded(request["harmonizer_terms_authorization_ref"]):
        _issue(report, "harmonizer_terms_unrecorded", "harmonizer_terms_authorization_ref",
               f"{', '.join(on_arms)} (--enable-harmonizer) uses nvidia/Harmonizer under its "
               f"licence for either source; give the recorded {HARMONIZER_TERMS_AUTHORIZATION_ID} "
               "authorization reference (non-placeholder, not a credential); no default is "
               "chosen.")
    if type(arms) is not list or len(arms) != 2:
        _issue(report, "invalid_arms", "arms", "Provide exactly the off and on arm definitions.")
    else:
        paths_valid = True
        for index, arm in enumerate(arms):
            if type(arm) is not dict or set(arm) != _ARM_FIELDS:
                _issue(report, "invalid_arm", f"arms[{index}]",
                       "Arm fields are harmonizer, run_dir, driver_log_dir.")
                paths_valid = False
                continue
            if type(arm["harmonizer"]) is not bool or arm["harmonizer"] != bool(index):
                _issue(report, "invalid_arm", f"arms[{index}].harmonizer",
                       "Represent off first and on second; this does not register "
                       "execution order.")
            for name in ("run_dir", "driver_log_dir"):
                if not _path(arm[name]):
                    _issue(report, "invalid_path", f"arms[{index}].{name}",
                           "Require an absolute portable-ASCII output path without spaces, "
                           "shell characters or empty/dot/traversal segments.")
                    paths_valid = False
        if paths_valid:
            for first in (arms[0]["run_dir"], arms[0]["driver_log_dir"]):
                for second in (arms[1]["run_dir"], arms[1]["driver_log_dir"]):
                    if _overlap(first, second):
                        _issue(report, "overlapping_arm_outputs", "arms",
                               "Off/on output directories must be disjoint, including "
                               "ancestors.")
            source = request["harmonizer_source"]
            if harmonizer_ok and source["kind"] == "pinned_mount":
                outputs = [arm[key] for arm in arms for key in ("run_dir", "driver_log_dir")]
                if any(_overlap(source["host_dir"], output) for output in outputs):
                    _issue(report, "overlapping_harmonizer_mount", "harmonizer_source.host_dir",
                           "The pinned checkpoint directory must be disjoint from arm outputs.")
    if _blocking(report):
        return report

    preset = override or "6cam_1080"
    renderer_cameras = list(CAMERA_PRESETS[preset])
    if set(renderer_cameras) != set(STOCK_DRIVER_CAMERAS):
        _issue(report, "renderer_driver_camera_mismatch_predicted", "renderer_camera_override",
               f"Preset {preset} renders {len(renderer_cameras)} cameras; the stock driver "
               f"accepts {len(STOCK_DRIVER_CAMERAS)} (alpamayo1_5.yaml:32-36) and raises on "
               "others (driver main.py:787). check_generated_configs will block until the "
               "owner's camera decision makes them equal.", severity="warning")
    pinned = request["harmonizer_source"]["kind"] == "pinned_mount"
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
        "owner_decisions": {
            "renderer_camera_override": override,
            "harmonizer_source": dict(request["harmonizer_source"]),
        },
        "image_kinds": {name: image_kind(request[name])
                        for name in ("renderer_image", "base_image")},
        "renderer_cameras": renderer_cameras,
        "stock_driver_cameras": list(STOCK_DRIVER_CAMERAS),
        "harmonizer_checkpoint_pinned": pinned,
        "harmonizer_terms_authorization": {
            "authorization_id": HARMONIZER_TERMS_AUTHORIZATION_ID,
            "authorization_ref": request["harmonizer_terms_authorization_ref"],
            "required_by_arms": ["on"],
        },
        "source_defaults": {
            "camera_config": preset,
            "control_timestep_us": 100_000,
            "force_gt_duration_us": 1_700_000,
            "planner_delay_us": 0,
            "render_bundling": "NONE",
            "traffic_mode": "log_replay",
        },
    }
    report["arms"] = [_command_group(request, arm, scenes) for arm in arms]
    report["prepared"] = True
    return report


_SWEEP_FIELDS = {
    "alpasim_sha", "alpasim_checkout", "requested_steps", "driver_port_base", "data_root",
    "hf_home", "renderer_image", "base_image", "renderer_camera_override", "harmonizer",
    "harmonizer_source", "harmonizer_terms_authorization_ref", "policies",
}
SWEEP_POLICIES = ("A", "B")
_SWEEP_POLICY_FIELDS = {"snapshot", "run_dir", "driver_log_dir"}


def check_sweep_launch(launch, unit_count):
    """Issues for a full-sweep launch block (F1); an empty list means usable (L1/L2).

    Same pinned command contract as the pilot (``build_pilot_launch``), generalized to
    policies A (teacher) and B (re-export), one registered Harmonizer setting for every
    unit (``harmonizer``: required boolean, no default) and per-unit driver ports
    ``driver_port_base + execution index``. Every port window must stay in range:
    ``driver_port_base + unit_count - 1 <= MAX_DRIVER_PORT``. Policy output directories
    must be disjoint (including ancestors) from each other and from a pinned Harmonizer
    mount. With the Harmonizer off, ``harmonizer_terms_authorization_ref`` is null or a
    recorded non-placeholder, non-credential reference (it is copied into the packet
    inputs). The pilot's ``renderer_driver_camera_mismatch_predicted`` warning is emitted
    for the same camera presets. Warnings never block. Scene IDs are checked by the
    caller with ``check_scene_ids``. Never executes, reads files or chooses a value.
    """
    report = {"issues": []}
    if type(launch) is not dict or set(launch) != _SWEEP_FIELDS:
        present = set(launch) if type(launch) is dict else set()
        _issue(report, "invalid_request", "launch",
               f"Provide exactly the full-sweep launch fields; missing "
               f"{sorted(_SWEEP_FIELDS - present)}, unexpected {sorted(present - _SWEEP_FIELDS)}.")
        return report["issues"]
    if launch["alpasim_sha"] != ALPASIM_SHA:
        _issue(report, "unsupported_source", "alpasim_sha",
               "Command contract requires the reviewed source pin.")
    for name in ("alpasim_checkout", "data_root", "hf_home"):
        if not _path(launch[name]):
            _issue(report, "invalid_path", name,
                   "Require an absolute portable-ASCII path without empty/dot/traversal segments.")
    if type(launch["requested_steps"]) is not int or launch["requested_steps"] <= 0:
        _issue(report, "invalid_requested_steps", "requested_steps",
               "Provide a positive requested horizon; no default is adopted.")
    base = launch["driver_port_base"]
    if (type(base) is not int or type(unit_count) is not int or unit_count < 1
            or not 1 <= base <= MAX_DRIVER_PORT - (unit_count - 1)):
        _issue(report, "invalid_port", "driver_port_base",
               f"Per-unit ports base..base+{unit_count - 1 if type(unit_count) is int else '?'} "
               f"must stay within 1..{MAX_DRIVER_PORT}, leaving six service-search windows.")
    for name in ("renderer_image", "base_image"):
        if image_kind(launch[name]) is None:
            _issue(report, "invalid_image", name,
                   "Require an immutable registry digest or a local build image ID.")
    override = launch["renderer_camera_override"]
    if not (override is None or (type(override) is str and override in _CAMERA_OVERRIDES)):
        _issue(report, "invalid_camera_override", "renderer_camera_override",
               "Owner decision required: null (keep dev 6cam_1080) or '4cam_1080'.")
    else:
        # Review B2-09: the same predicted-mismatch warning as build_pilot_launch.
        preset = override or "6cam_1080"
        if set(CAMERA_PRESETS[preset]) != set(STOCK_DRIVER_CAMERAS):
            _issue(report, "renderer_driver_camera_mismatch_predicted",
                   "renderer_camera_override",
                   f"Preset {preset} renders {len(CAMERA_PRESETS[preset])} cameras; the stock "
                   f"driver accepts {len(STOCK_DRIVER_CAMERAS)} (alpamayo1_5.yaml:32-36) and "
                   "raises on others (driver main.py:787). check_generated_configs will block "
                   "until the owner's camera decision makes them equal.", severity="warning")
    if type(launch["harmonizer"]) is not bool:
        _issue(report, "invalid_harmonizer_setting", "harmonizer",
               "Owner decision required: the one registered Harmonizer setting (true/false).")
    harmonizer_ok = _validate_harmonizer(launch["harmonizer_source"], report)
    terms_ref = launch["harmonizer_terms_authorization_ref"]
    if launch["harmonizer"] is True and not _harmonizer_terms_recorded(terms_ref):
        _issue(report, "harmonizer_terms_unrecorded", "harmonizer_terms_authorization_ref",
               f"--enable-harmonizer uses nvidia/Harmonizer under its licence; give the "
               f"recorded {HARMONIZER_TERMS_AUTHORIZATION_ID} reference; no default is chosen.")
    elif terms_ref is not None and not _harmonizer_terms_recorded(terms_ref):
        # Review B2-09: the reference is copied into the packet inputs even when the
        # Harmonizer is off, so it is null or a recorded non-placeholder, non-credential
        # reference; never a credential-shaped or placeholder value.
        _issue(report, "harmonizer_terms_unrecorded", "harmonizer_terms_authorization_ref",
               "With the Harmonizer off the terms reference is null or a recorded, "
               "non-placeholder, non-credential reference.")
    policies = launch["policies"]
    if type(policies) is not dict or set(policies) != set(SWEEP_POLICIES) or not all(
            type(policies[name]) is dict and set(policies[name]) == _SWEEP_POLICY_FIELDS
            for name in SWEEP_POLICIES):
        _issue(report, "invalid_policies", "policies",
               "Provide policies A and B, each exactly {snapshot, run_dir, driver_log_dir}.")
        return report["issues"]
    outputs = []
    for name in SWEEP_POLICIES:
        for key in sorted(_SWEEP_POLICY_FIELDS):
            if not _path(policies[name][key]):
                _issue(report, "invalid_path", f"policies.{name}.{key}",
                       "Require an absolute portable-ASCII path.")
        outputs.append([policies[name]["run_dir"], policies[name]["driver_log_dir"]])
    if not any(item["path"].startswith("policies.") for item in report["issues"]):
        if any(_overlap(first, second) for first in outputs[0] for second in outputs[1]):
            _issue(report, "overlapping_arm_outputs", "policies",
                   "Policy A/B output directories must be disjoint, including ancestors.")
        if policies["A"]["snapshot"] == policies["B"]["snapshot"]:
            _issue(report, "identical_policy_snapshots", "policies",
                   "A and B load the same snapshot path; the re-export control needs its own "
                   "exported files.", severity="warning")
        source = launch["harmonizer_source"]
        if harmonizer_ok and source["kind"] == "pinned_mount" and any(
                _overlap(source["host_dir"], path) for pair in outputs for path in pair):
            _issue(report, "overlapping_harmonizer_mount", "harmonizer_source.host_dir",
                   "The pinned checkpoint directory must be disjoint from policy outputs.")
    return report["issues"]


def check_scene_ids(scene_ids):
    """Positions of scene IDs outside the pilot's portable-ASCII alphabet (review B2-01).

    The pinned wizard joins argv into shell commands, so every scheduled scene ID must
    pass the same rule as ``build_pilot_launch`` (no spaces, shell metacharacters,
    quotes or OmegaConf interpolation).
    """
    return [index for index, scene in enumerate(scene_ids) if not _scene(scene)]


def sweep_unit_request(launch, policy, port):
    """Pilot-shaped command request for one full-sweep unit (validated launch only)."""
    return {
        "alpasim_checkout": launch["alpasim_checkout"],
        "requested_steps": launch["requested_steps"],
        "driver_port": port,
        "teacher_snapshot": launch["policies"][policy]["snapshot"],
        "data_root": launch["data_root"],
        "hf_home": launch["hf_home"],
        "renderer_image": launch["renderer_image"],
        "base_image": launch["base_image"],
        "renderer_camera_override": launch["renderer_camera_override"],
        "harmonizer_source": launch["harmonizer_source"],
    }


def _renderer_command(request, harmonizer):
    renderer = list(_RENDERER_BEFORE_HARMONIZER)
    if harmonizer:
        renderer.append("--enable-harmonizer")
    renderer.extend(_RENDERER_AFTER_HARMONIZER)
    if request["harmonizer_source"]["kind"] == "pinned_mount":
        # Both arms carry the cache flag so the arms differ only by --enable-harmonizer.
        renderer.append(f"--harmonizer-cache={HARMONIZER_CONTAINER_DIR}")
    return renderer


def _env(request, *, hugging_face):
    """Explicit process environment: names allowed, plus fixed values (L5)."""
    env_set = {"LANG": "C.UTF-8"}
    if hugging_face:
        # Offline: assets are pre-staged by a separately authorized stage; any
        # attempted download then fails closed instead of using a token.
        env_set.update({"HF_HOME": request["hf_home"], "HF_HUB_OFFLINE": "1"})
    return sorted({*_ENV_INHERITED, *env_set}), env_set


def _command_group(request, arm, scenes):
    """Assemble already-validated inputs; private singleton support for packets.

    Public callers validate the entire five-scene pilot before reaching here.
    The packet supplies one of those scenes and job-specific child output paths.
    """
    port = request["driver_port"]
    argv = [
        "uv",
        "run",
        "--no-sync",
        "alpasim_wizard",
        "driver=alpamayo1_5",
        "+e2e_challenge=dev",
        "+nurec_scenes=curated_val",
    ]
    if request["renderer_camera_override"] is not None:
        # Overrides dev.yaml:11 "/cameras: 6cam_1080"; replacement of the nested
        # camera list is INFERRED and verified by check_generated_configs.
        argv.append(f"cameras={request['renderer_camera_override']}")
    argv += [
        "scenes.test_suite_id=null",
        f"scenes.scene_ids={_hydra(scenes)}",
        "scenes.limit_to_first_n=0",
        "runtime.simulation_config.n_rollouts=1",
        f"runtime.simulation_config.n_sim_steps={request['requested_steps']}",
        f"driver.port={port}",
        f"wizard.baseport={port + 1}",
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
        f"services.renderer.command={_hydra(_renderer_command(request, arm['harmonizer']))}",
    ]
    if request["harmonizer_source"]["kind"] == "pinned_mount":
        # Source interpolations are fixed constants, not inputs; their resolution
        # inside a CLI list override is INFERRED (qualify_on_host).
        volumes = [*_BASE_RENDERER_VOLUMES,
                   f"{request['harmonizer_source']['host_dir']}:{HARMONIZER_CONTAINER_DIR}"]
        argv.append(f"services.renderer.volumes={_hydra(volumes)}")
    # L4: keep Hydra outputs out of the checkout (setup_omegaconf.py:262-266 binds
    # hydra.main; base_config.yaml:17-19 keeps chdir false; default run dir is
    # outputs/<date>/<time> under the working directory).
    argv.append(f"hydra.run.dir={_hydra(arm['run_dir'] + '/.hydra-wizard')}")
    wizard_env, wizard_set = _env(request, hugging_face=True)
    driver_env, driver_set = _env(request, hugging_face=True)
    compose_env, compose_set = _env(request, hugging_face=False)
    return {
        "harmonizer": arm["harmonizer"],
        "wizard_prepare": {
            "cwd": request["alpasim_checkout"],
            "argv": argv,
            "may_download_assets": True,
            "env_allowlist": wizard_env,
            "env_set": wizard_set,
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
                # driver main.py:1106-1110 is a hydra.main entrypoint.
                f"hydra.run.dir={_hydra(arm['driver_log_dir'] + '/.hydra')}",
            ],
            "env_allowlist": driver_env,
            "env_set": driver_set,
        },
        "compose": {
            "cwd": arm["run_dir"],
            # wizard.py prints this command for run_method NONE; "--pull never"
            # (Docker Compose CLI, L3) stops missing-image pulls (schema.py:157).
            "argv": ["docker", "compose", "up", "--pull", "never",
                     "--exit-code-from", "runtime-0"],
            "env_allowlist": compose_env,
            "env_set": compose_set,
            "exit_code_is_success_evidence": False,
        },
    }


def filter_environment(command, host_env):
    """Return the exact environment for one prepared command; pure.

    Only names in ``env_allowlist`` survive; ``env_set`` values win over the
    host; inherited names must exist on the host. Credential-like names are
    rejected even if a caller adds them to a command.
    """
    if type(command) is not dict or type(host_env) is not dict:
        raise ValueError("Expected a prepared command and a host environment mapping")
    allowlist, env_set = command.get("env_allowlist"), command.get("env_set")
    if type(allowlist) is not list or type(env_set) is not dict:
        raise ValueError("Command lacks env_allowlist/env_set")
    if not set(env_set) <= set(allowlist):
        raise ValueError("env_set names must be allowlisted")
    result = {}
    for name in allowlist:
        if type(name) is not str or _FORBIDDEN_ENV.fullmatch(name):
            raise ValueError(f"Environment name {name!r} is not allowed")
        if name in env_set:
            result[name] = env_set[name]
        elif name in host_env and type(host_env[name]) is str:
            result[name] = host_env[name]
        else:
            raise ValueError(f"Required inherited environment variable {name} is missing")
    return result


def _input_hash(*values):
    """Canonical identity of JSON-native check inputs; None when not JSON-native."""
    try:
        return identity_hash(list(values))
    except (ValueError, TypeError, UnicodeError):
        return None


_CONFIG_EXPECTED = {"scope", "scene_ids", "requested_steps", "driver_port", "teacher_snapshot",
                    "renderer_cameras"}


def _get(mapping, *keys):
    for key in keys:
        if type(mapping) is not dict or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def _camera_list(value):
    return (type(value) is list and bool(value)
            and all(type(item) is str and item for item in value)
            and len(set(value)) == len(value))


def _config_expectation_problem(expected):
    """Validate expectation values before use (review A4-F4); None when valid."""
    if expected["scope"] not in SCOPES or type(expected["scope"]) is not str:
        return f"scope must be one of {list(SCOPES)}"
    scenes = expected["scene_ids"]
    if not (type(scenes) is list and scenes and all(_scene(scene) for scene in scenes)
            and len(set(scenes)) == len(scenes)):
        return "scene_ids must be a nonempty list of unique portable scene labels"
    for key in ("requested_steps", "driver_port"):
        if type(expected[key]) is not int or expected[key] <= 0:
            return f"{key} must be a positive integer"
    if not _path(expected["teacher_snapshot"]):
        return "teacher_snapshot must be a portable absolute path"
    if not _camera_list(expected["renderer_cameras"]):
        return "renderer_cameras must be a nonempty list of unique camera ID strings"
    return None


def check_generated_configs(generated_user_config, driver_config, expected):
    """Compare parsed generated-user-config-0.yaml and driver-config.yaml (L1 check).

    Blocks when the renderer camera set (``simulation_config.cameras[].logical_id``,
    configuration.py:97-108) differs from the driver's ``inference.use_cameras``
    (configuration.py:252-260 writes the driver subtree) or from the expected
    preset, and when scene, horizon, port or checkpoint differ from the unit.
    Sets are compared because the driver builds a set (driver main.py:224).
    """
    report = {"artifact_kind": "s2_generated_config_check", "schema_version": 1,
              "scope": None, "producer": _producer("check_generated_configs"),
              "input_sha256": _input_hash(generated_user_config, driver_config, expected),
              "status": "blocked", "issues": [], "observed": None}
    if (type(generated_user_config) is not dict or type(driver_config) is not dict
            or type(expected) is not dict or set(expected) != _CONFIG_EXPECTED):
        _issue(report, "invalid_check_input", "inputs",
               f"Provide two parsed mappings and expected fields {sorted(_CONFIG_EXPECTED)}.")
        return report
    problem = _config_expectation_problem(expected)
    if problem:
        _issue(report, "invalid_check_input", "expected", problem)
        return report
    report["scope"] = expected["scope"]
    cameras = _get(generated_user_config, "simulation_config", "cameras")
    renderer = ([camera.get("logical_id") if type(camera) is dict else None for camera in cameras]
                if type(cameras) is list else None)
    driver = _get(driver_config, "inference", "use_cameras")
    scenes = _get(generated_user_config, "scenes")
    observed = {
        "renderer_cameras": renderer,
        "driver_use_cameras": driver,
        "scene_ids": ([scene.get("scene_id") if type(scene) is dict else None
                       for scene in scenes] if type(scenes) is list else None),
        "n_sim_steps": _get(generated_user_config, "simulation_config", "n_sim_steps"),
        "n_rollouts": _get(generated_user_config, "simulation_config", "n_rollouts"),
        "driver_port": _get(driver_config, "port"),
        "checkpoint_path": _get(driver_config, "model", "checkpoint_path"),
        "max_batch_size": _get(driver_config, "inference", "max_batch_size"),
    }
    report["observed"] = observed

    if not _camera_list(renderer):
        _issue(report, "generated_cameras_invalid", "generated_user_config",
               "simulation_config.cameras must list unique nonempty logical_id values")
    if not _camera_list(driver):
        _issue(report, "driver_cameras_invalid", "driver_config",
               "inference.use_cameras must list unique nonempty camera IDs")
    if _camera_list(renderer) and _camera_list(driver) and set(renderer) != set(driver):
        _issue(report, "renderer_driver_camera_mismatch", "cameras",
               f"Renderer sends {len(renderer)} cameras {sorted(renderer)}; driver accepts "
               f"{len(driver)} {sorted(driver)}. Every extra camera image raises in the "
               "stock driver (main.py:787).")
    if _camera_list(renderer) and set(renderer) != set(expected["renderer_cameras"]):
        _issue(report, "renderer_cameras_unexpected", "generated_user_config",
               "Generated camera list differs from the registered preset/override")
    for key, value in (("scene_ids", expected["scene_ids"]),
                       ("n_sim_steps", expected["requested_steps"]), ("n_rollouts", 1),
                       ("driver_port", expected["driver_port"]),
                       ("checkpoint_path", expected["teacher_snapshot"]),
                       ("max_batch_size", 1)):
        if observed[key] != value or type(observed[key]) is bool:
            _issue(report, f"generated_{key}_mismatch", key,
                   f"Generated {key} {observed[key]!r} differs from expected {value!r}")
    report["status"] = "blocked" if _blocking(report) else "passed"
    return report


_DOCKER_UNITS = {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def docker_bytes(value):
    """Docker byte-size (go-units RAMInBytes: binary units, optional 'b'); None if invalid."""
    if type(value) is int and value > 0:
        return value
    if type(value) is not str:
        return None
    match = re.fullmatch(r"([1-9][0-9]{0,15})([bkmgt]?)b?", value.strip().lower())
    return None if match is None else int(match[1]) * _DOCKER_UNITS[match[2]]


_COMPOSE_EXPECTED = {"scope", "renderer_image", "base_image", "harmonizer", "harmonizer_cache",
                     "shm_size", "renderer_volumes"}
EGO_HOOD_CONTAINER_DIR = "/mnt/ego-hoods"
# Renderer mount targets whose host source the unit pins (review A4-F5). The
# /mnt/nre-data source is not checked: context.py:117 rewrites scenes.scene_cache
# to the sceneset directory when local files are used, so its host side is not a
# function of the launch request.
_MANAGED_RENDERER_TARGETS = (EGO_HOOD_CONTAINER_DIR, HARMONIZER_CONTAINER_DIR)


def _compose_expectation_problem(expected):
    """Validate compose expectation values before use; None when valid."""
    if expected["scope"] not in SCOPES or type(expected["scope"]) is not str:
        return f"scope must be one of {list(SCOPES)}"
    for key in ("renderer_image", "base_image"):
        if type(expected[key]) is not str:
            return f"{key} must be a string"
    if type(expected["harmonizer"]) is not bool:
        return "harmonizer must be a boolean"
    cache = expected["harmonizer_cache"]
    if cache not in (None, HARMONIZER_CONTAINER_DIR):
        return f"harmonizer_cache must be null or {HARMONIZER_CONTAINER_DIR!r}"
    if type(expected["shm_size"]) is not dict:
        return "shm_size must be a mapping of service to declared size"
    volumes = expected["renderer_volumes"]
    if (type(volumes) is not dict or EGO_HOOD_CONTAINER_DIR not in volumes
            or not set(volumes) <= set(_MANAGED_RENDERER_TARGETS)
            or not all(_path(source) for source in volumes.values())):
        return (f"renderer_volumes must map {list(_MANAGED_RENDERER_TARGETS)} targets "
                f"(always {EGO_HOOD_CONTAINER_DIR!r}) to portable host paths")
    if (HARMONIZER_CONTAINER_DIR in volumes) != (cache is not None):
        return "A Harmonizer mount is expected exactly when the pinned cache flag is"
    return None


def _volume_pairs(volumes):
    """(source, target) pairs from short ``src:dst[:mode]`` or long bind mappings.

    The generator writes short strings (services.py:47-48, docker_compose.py:114);
    ``docker compose config`` emits long ``{type: bind, source, target}`` mappings
    (the host-emitted shape is qualify_on_host). None marks an unrecognised entry.
    """
    pairs = []
    for entry in volumes:
        if type(entry) is str and 2 <= len(entry.split(":")) <= 3:
            source, target = entry.split(":")[:2]
            pairs.append((source, target))
        elif (type(entry) is dict and entry.get("type") == "bind"
              and type(entry.get("source")) is str and type(entry.get("target")) is str):
            pairs.append((entry["source"], entry["target"]))
        else:
            pairs.append(None)
    return pairs


def check_generated_compose(compose, expected):
    """Check a parsed effective Compose config (``docker compose config``) for one unit.

    Blocks unless: services are exactly COMPOSE_SERVICES; each uses host
    networking (dev.yaml:24-25 use_localhost; docker_compose.py:108-111); images
    equal the unit's immutable references with no build recipe
    (docker_compose.py:115-128); the renderer command contains
    ``--enable-harmonizer`` only in the on arm and the pinned cache flag only when
    pinned; the renderer mounts exactly the declared ego-hood source at
    /mnt/ego-hoods and, only when pinned, the Harmonizer host_dir at
    /mnt/harmonizer (review A4-F5: the cache flag alone does not prove the mount);
    and every service's ``shm_size`` equals the declared value. The
    generator never writes ``shm_size`` (docker_compose.py:96-173), so Docker's
    64 MiB default applies unless an operator override supplies it; NRE's own
    documented launch uses ``--shm-size=64g`` (grpc_api_guide_2604.html.txt:65,
    202). ``pull_policy`` other than ``never`` is a warning: the Compose argv
    carries ``--pull never`` (support is qualify_on_host).
    """
    report = {"artifact_kind": "s2_generated_compose_check", "schema_version": 1,
              "scope": None, "producer": _producer("check_generated_compose"),
              "input_sha256": _input_hash(compose, expected),
              "status": "blocked", "issues": [], "observed": None}
    if (type(compose) is not dict or type(expected) is not dict
            or set(expected) != _COMPOSE_EXPECTED or type(compose.get("services")) is not dict
            or not all(type(name) is str for name in compose["services"])):
        _issue(report, "invalid_check_input", "inputs",
               f"Provide a parsed Compose mapping with services and expected fields "
               f"{sorted(_COMPOSE_EXPECTED)}.")
        return report
    problem = _compose_expectation_problem(expected)
    if problem:
        _issue(report, "invalid_check_input", "expected", problem)
        return report
    report["scope"] = expected["scope"]
    services = compose["services"]
    report["observed"] = {"services": sorted(services)}
    if sorted(services) != list(COMPOSE_SERVICES):
        _issue(report, "unexpected_compose_services", "services",
               f"Expected exactly {list(COMPOSE_SERVICES)}, observed {sorted(services)}")
    shm = expected["shm_size"]
    for name in COMPOSE_SERVICES:
        service = services.get(name)
        if type(service) is not dict:
            continue
        path = f"services.{name}"
        if service.get("network_mode") != "host":
            _issue(report, "network_mode_not_host", path, "Generated service must use host")
        image = expected["renderer_image"] if name == "renderer-0" else expected["base_image"]
        if service.get("image") != image:
            _issue(report, "image_mismatch", path, f"Image {service.get('image')!r} != {image!r}")
        if "build" in service:
            _issue(report, "unexpected_build_recipe", path, "external_image must suppress build")
        if service.get("pull_policy") != "never":
            _issue(report, "pull_policy_not_never", path,
                   f"pull_policy {service.get('pull_policy')!r}; relies on --pull never",
                   severity="warning")
        declared = shm.get(name) if type(shm) is dict else None
        actual = service.get("shm_size")
        if declared == "docker_default":
            if actual is not None:
                _issue(report, "shm_size_mismatch", path,
                       f"Declared Docker default but service sets shm_size {actual!r}")
        elif docker_bytes(declared) is None:
            _issue(report, "shm_size_undeclared", path, "No valid declared shm_size")
        elif actual is None or docker_bytes(actual) != docker_bytes(declared):
            _issue(report, "shm_size_mismatch", path,
                   f"Declared shm_size {declared!r}; effective config has {actual!r} "
                   "(generator never writes shm_size)")
    renderer = services.get("renderer-0")
    if type(renderer) is dict:
        _check_renderer_volumes(report, renderer.get("volumes", []),
                                expected["renderer_volumes"])
    command = renderer.get("command") if type(renderer) is dict else None
    if type(command) is list and len(command) == 2 and command[0] == "-c" and type(
            command[1]) is str:
        tokens = command[1].split()
        if ("--enable-harmonizer" in tokens) != (expected["harmonizer"] is True):
            _issue(report, "harmonizer_flag_mismatch", "services.renderer-0.command",
                   "--enable-harmonizer presence differs from the unit's arm")
        cache = expected["harmonizer_cache"]
        flags = [token for token in tokens if token.startswith("--harmonizer-cache")]
        if flags != ([] if cache is None else [f"--harmonizer-cache={cache}"]):
            _issue(report, "harmonizer_cache_mismatch", "services.renderer-0.command",
                   f"Harmonizer cache flags {flags} differ from the declared source")
    else:
        _issue(report, "renderer_command_unrecognized", "services.renderer-0.command",
               "Expected ['-c', '<script>'] from docker_compose.py:130-144")
    report["status"] = "blocked" if _blocking(report) else "passed"
    return report


def _check_renderer_volumes(report, volumes, expected):
    path = "services.renderer-0.volumes"
    if type(volumes) is not list:
        _issue(report, "renderer_volume_unrecognized", path, "volumes must be a list")
        return
    pairs = _volume_pairs(volumes)
    for index, pair in enumerate(pairs):
        if pair is None:
            _issue(report, "renderer_volume_unrecognized", f"{path}[{index}]",
                   "Expected 'host:container[:mode]' or a bind mapping with source/target")
    for target in _MANAGED_RENDERER_TARGETS:
        sources = [pair[0] for pair in pairs if pair is not None and pair[1] == target]
        if target not in expected:
            if sources:
                _issue(report, "renderer_volume_unexpected", path,
                       f"{target} is mounted from {sources} but the unit declares no such mount")
        elif not sources:
            _issue(report, "renderer_volume_missing", path,
                   f"Expected {expected[target]}:{target}; the renderer would run without it")
        elif sources != [expected[target]]:
            _issue(report, "renderer_volume_mismatch", path,
                   f"{target} is mounted from {sources}; expected exactly [{expected[target]!r}]")
