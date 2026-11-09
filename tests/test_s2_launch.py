"""Literal source-pinned command oracles; no wizard, service or shell execution."""

import ast
import hashlib
import json
import re
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from s2_launch_fixture import launch_request, pinned_harmonizer_source

from gate.s2_launch import (
    CAMERA_PRESETS,
    STOCK_DRIVER_CAMERAS,
    build_pilot_launch,
    check_generated_compose,
    check_generated_configs,
    docker_bytes,
    filter_environment,
    image_kind,
)

SOURCE_ROOT = Path(__file__).parent / "fixtures" / "alpasim_launch_upstream"


def upstream_source(path):
    return (SOURCE_ROOT / "alpasim" / path).read_text()


def test_public_source_fixtures_match_both_pinned_git_blobs_and_sha256():
    manifest = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    expected_pins = {
        "NVlabs/alpasim": "e121e59141dc391f6c295513d6dadc840a419149",
        "NVlabs/alpamayo1.5": "36aeb4c5938cbc2eb2aed33b22434773da4ab639",
    }
    assert manifest["sources"]
    for source in manifest["sources"]:
        assert source["commit"] == expected_pins[source["repository"]]
        data = (SOURCE_ROOT / source["fixture_path"]).read_bytes()
        assert len(data) == source["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        header = f"blob {len(data)}\0".encode()
        assert hashlib.sha1(header + data).hexdigest() == source["git_blob_sha1"]


def test_renderer_command_is_derived_from_pinned_public_yaml():
    base = upstream_source("src/wizard/configs/base_config.yaml")
    dev = upstream_source("src/wizard/configs/e2e_challenge/dev.yaml")
    topology = upstream_source("src/wizard/configs/topology/1gpu.yaml")
    renderer_block = base.split("\n  renderer:\n", 1)[1].split("\n  driver:\n", 1)[0]
    command_block = renderer_block.split("    command:\n", 1)[1]
    # Parse only the reviewed double-quoted scalar list, not arbitrary YAML.
    source_command = [
        json.loads(match)
        for match in re.findall(r'^      - ("(?:[^"\\]|\\.)*")', command_block, re.MULTILINE)
    ]
    assert len(source_command) == 12
    entrypoint = json.loads(re.search(r"^  renderer_entrypoint: (.+)$", dev, re.MULTILINE)[1])
    cache = re.search(r"^  nre_cache_size: ([0-9]+)", topology, re.MULTILINE)[1]
    workers = re.search(r"^  nre_max_workers: ([0-9]+)", base, re.MULTILINE)[1]
    resolved = [
        item.replace("${defines.renderer_entrypoint}", entrypoint)
        .replace("${defines.nre_cache_size}", cache)
        .replace("${defines.nre_max_workers}", workers)
        for item in source_command
    ]
    report = build_pilot_launch(launch_request())
    for arm in report["arms"]:
        rendered = next(
            value
            for value in arm["wizard_prepare"]["argv"]
            if value.startswith("services.renderer.command=")
        )
        expected = (
            resolved
            if arm["harmonizer"]
            else [value for value in resolved if value != "--enable-harmonizer"]
        )
        # Pilot instrumentation is an explicit local requirement; the pinned
        # source documents the option, which is not enabled in its stock list.
        assert '# - "--enable-timing"' in command_block
        assert json.loads(rendered.split("=", 1)[1]) == [*expected, "--enable-timing"]
    ec2 = upstream_source("src/wizard/configs/e2e_challenge/ec2.yaml")
    assert "--enable-harmonizer" not in ec2


def test_pinned_sources_explain_downstream_shell_and_prebuilt_image_guards():
    services = upstream_source("src/wizard/alpasim_wizard/services.py.txt")
    compose = upstream_source("src/wizard/alpasim_wizard/deployment/docker_compose.py.txt")
    base = upstream_source("src/wizard/configs/base_config.yaml")
    assert 'command = " ".join(service_config.command)' in services
    assert 'ret["entrypoint"] = "bash"' in compose
    assert 'ret["command"] = ["-c", command]' in compose
    assert "if not container.service_config.external_image:" in compose
    assert '"tags": [container.service_config.image]' in compose
    for name in ("physics", "controller", "runtime"):
        assert f"\n  {name}:\n    image: ${{defines.base_image}}" in base
    report = build_pilot_launch(launch_request())
    for arm in report["arms"]:
        for name in ("physics", "controller", "runtime"):
            assert f"services.{name}.external_image=true" in arm["wizard_prepare"]["argv"]


def test_pinned_driver_config_output_matches_assembled_host_command():
    config = upstream_source("src/wizard/alpasim_wizard/configuration.py.txt")
    base = upstream_source("src/wizard/configs/base_config.yaml")
    assert 'self._write_config("driver-config.yaml", driver_config)' in config
    assert '"uv run -m alpasim_driver.main"' in base
    assert '"--config-path=/mnt/output"' in base
    assert '"--config-name=driver-config.yaml"' in base
    report = build_pilot_launch(launch_request())
    assert report["command_origin"] == "assembled_from_pinned_source_interfaces_not_runtime_tested"
    assert all(
        "--config-name=driver-config.yaml" in arm["driver"]["argv"] for arm in report["arms"]
    )


def test_five_scene_source_contract_emits_two_arms_and_no_authority():
    request = launch_request()
    before = deepcopy(request)
    report = build_pilot_launch(request)
    assert report["prepared"], report["issues"]
    # The synthetic fixture keeps the six-camera preset: a predicted mismatch warning.
    assert [(i["code"], i["severity"]) for i in report["issues"]] == [
        ("renderer_driver_camera_mismatch_predicted", "warning")]
    assert report["pilot"]["scene_count"] == 5
    assert report["pilot"]["replicates"] == 1
    assert report["pilot"]["policy_rollouts"] == 10
    assert report["pilot"]["seed_mode"] == "unseeded"
    assert report["cloud_launch_authorized"] is False
    assert report["runtime_qualified"] is False
    assert report["paths_verified"] is False
    assert report["images_verified"] is False
    assert report["execution_order_registered"] is False
    assert report["commands_executed"] is False
    assert [arm["harmonizer"] for arm in report["arms"]] == [False, True]
    assert request == before
    json.dumps(report, allow_nan=False)


def test_literal_off_arm_wizard_argv_matches_pinned_source():
    report = build_pilot_launch(launch_request())
    wizard = report["arms"][0]["wizard_prepare"]
    assert wizard["cwd"] == "/synthetic/alpasim"
    assert wizard["argv"] == [
        "uv",
        "run",
        "--no-sync",
        "alpasim_wizard",
        "driver=alpamayo1_5",
        "+e2e_challenge=dev",
        "+nurec_scenes=curated_val",
        "scenes.test_suite_id=null",
        (
            'scenes.scene_ids=["synthetic-scene-0","synthetic-scene-1","synthetic-scene-2",'
            '"synthetic-scene-3","synthetic-scene-4"]'
        ),
        "scenes.limit_to_first_n=0",
        "runtime.simulation_config.n_rollouts=1",
        "runtime.simulation_config.n_sim_steps=17",
        "driver.port=6789",
        "wizard.baseport=6790",
        'wizard.external_services.driver=["localhost:6789"]',
        'driver.model.checkpoint_path="/synthetic/teacher/snapshot"',
        'driver.output_dir="/synthetic/driver-logs/off"',
        "driver.inference.max_batch_size=1",
        'wizard.log_dir="/synthetic/runs/off"',
        "wizard.run_method=NONE",
        "wizard.nr_retries=0",
        "runtime.enable_autoresume=false",
        'defines.filesystem="/synthetic/data"',
        'services.renderer.image="nvcr.io/nvidia/nre/nre-ga@sha256:' + "a" * 64 + '"',
        'defines.base_image="registry.example/synthetic-alpasim@sha256:' + "b" * 64 + '"',
        "services.physics.external_image=true",
        "services.controller.external_image=true",
        "services.runtime.external_image=true",
        (
            'services.renderer.command=["/app/internal/scripts/pycena/runtime/pycena_nrm_full",'
            '"serve-grpc","--port={port}","--host=0.0.0.0",'
            '"--artifact-glob=/mnt/nre-data/{sceneset}/**/*.usdz",'
            '"--egocar-hood-dir=/mnt/ego-hoods","--no-enable-nrend",'
            '"--download-cache-dir /tmp/nre-cache-dir","--cache-size=5",'
            '"--max-workers=4","--enable-editing-actors","--enable-timing"]'
        ),
        'hydra.run.dir="/synthetic/runs/off/.hydra-wizard"',
    ]
    assert wizard["may_download_assets"] is True
    assert wizard["env_allowlist"] == ["HF_HOME", "HF_HUB_OFFLINE", "HOME", "LANG", "PATH"]
    assert wizard["env_set"] == {"HF_HOME": "/synthetic/hf-home", "HF_HUB_OFFLINE": "1",
                                 "LANG": "C.UTF-8"}


def test_renderer_arms_differ_only_by_explicit_harmonizer_flag():
    report = build_pilot_launch(launch_request())
    commands = []
    for arm in report["arms"]:
        override = next(
            arg
            for arg in arm["wizard_prepare"]["argv"]
            if arg.startswith("services.renderer.command=")
        )
        commands.append(json.loads(override.split("=", 1)[1]))
    assert "--enable-harmonizer" not in commands[0]
    assert commands[1].count("--enable-harmonizer") == 1
    assert [item for item in commands[1] if item != "--enable-harmonizer"] == commands[0]


@pytest.mark.parametrize("port", [1, 6000, 6789, 59535])
def test_managed_ports_begin_strictly_above_external_driver(port):
    request = launch_request()
    request["driver_port"] = port
    result = build_pilot_launch(request)
    assert result["prepared"], result["issues"]
    for arm in result["arms"]:
        argv = arm["wizard_prepare"]["argv"]
        assert f"driver.port={port}" in argv
        assert f"wizard.baseport={port + 1}" in argv


@pytest.mark.parametrize("port", [59536, 64000, 65535])
def test_driver_port_leaves_source_allocator_search_headroom(port):
    request = launch_request()
    request["driver_port"] = port
    result = build_pilot_launch(request)
    assert not result["prepared"]
    assert result["arms"] == []
    assert "invalid_port" in {issue["code"] for issue in result["issues"]}


def test_pinned_allocator_six_search_windows_stay_above_driver_and_within_tcp_range():
    source = upstream_source("src/wizard/alpasim_wizard/context.py.txt")
    tree = ast.parse(source)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_find_next_open_port", "create_port_assigner")
    ]
    assert len(functions) == 2
    available = [60535, 61535, 62535, 63535, 64535, 65535]
    probes = []

    def port_available(port):
        assert 59535 < port <= 65535
        probes.append(port)
        return port in available

    namespace = {
        "Iterator": Iterator,
        "logger": SimpleNamespace(debug=lambda *args: None),
        "_is_port_open": port_available,
    }
    # Execute only the pinned allocator helpers with an injected synthetic port
    # probe. No upstream imports, socket calls or simulator code runs.
    exec(  # noqa: S102 - pinned helper AST only, with an injected in-memory probe.
        compile(ast.Module(body=functions, type_ignores=[]), "pinned-port-helpers", "exec"),
        namespace,
    )
    allocator = namespace["create_port_assigner"](59536)
    assert [next(allocator) for _ in range(6)] == available
    assert len(probes) == 6000
    wizard = upstream_source("src/wizard/alpasim_wizard/wizard.py.txt")
    slurm = upstream_source("src/wizard/alpasim_wizard/deployment/slurm.py.txt")
    assert "slurm_deployment = SlurmDeployment(self.context)" in wizard
    assert 'self.container_set = build_container_set(context, use_address_string="0.0.0.0")' in slurm


def test_driver_and_compose_use_verified_config_and_working_directories():
    report = build_pilot_launch(launch_request())
    for label, arm in zip(("off", "on"), report["arms"], strict=True):
        assert arm["driver"] == {
            "cwd": "/synthetic/alpasim",
            "argv": [
                "uv",
                "run",
                "--no-sync",
                "-m",
                "alpasim_driver.main",
                f"--config-path=/synthetic/runs/{label}",
                "--config-name=driver-config.yaml",
                f'hydra.run.dir="/synthetic/driver-logs/{label}/.hydra"',
            ],
            "env_allowlist": ["HF_HOME", "HF_HUB_OFFLINE", "HOME", "LANG", "PATH"],
            "env_set": {"HF_HOME": "/synthetic/hf-home", "HF_HUB_OFFLINE": "1",
                        "LANG": "C.UTF-8"},
        }
        assert arm["compose"] == {
            "cwd": f"/synthetic/runs/{label}",
            "argv": ["docker", "compose", "up", "--pull", "never",
                     "--exit-code-from", "runtime-0"],
            "env_allowlist": ["HOME", "LANG", "PATH"],
            "env_set": {"LANG": "C.UTF-8"},
            "exit_code_is_success_evidence": False,
        }


def test_shell_metacharacters_are_rejected_at_downstream_shell_boundary():
    request = launch_request()
    path = '/synthetic/a path/$(touch marker);`whoami` "quoted" \\literal'
    request["teacher_snapshot"] = path
    request["arms"][0]["run_dir"] = "/synthetic/runs/off arm;echo literal"
    report = build_pilot_launch(request)
    assert not report["prepared"]
    assert report["arms"] == []
    assert report["issues"]


@pytest.mark.parametrize(
    "suffix",
    [
        "space here",
        "semi;colon",
        "$(whoami)",
        "`whoami`",
        "back\\slash",
        'double"quote',
        "single'quote",
        "uniçode",
    ],
)
@pytest.mark.parametrize(
    "field",
    ["alpasim_checkout", "teacher_snapshot", "data_root", "run_dir", "driver_log_dir", "scene_ids"],
)
def test_only_documented_portable_input_alphabet_is_prepared(field, suffix):
    request = launch_request()
    if field == "scene_ids":
        request[field][0] = suffix
    elif field in ("run_dir", "driver_log_dir"):
        request["arms"][0][field] = "/synthetic/" + suffix
    else:
        request[field] = "/synthetic/" + suffix
    report = build_pilot_launch(request)
    assert not report["prepared"]
    assert report["arms"] == []


@pytest.mark.parametrize(
    "path", ["/synthetic//double", "//synthetic/path", "/synthetic/./dot", "/synthetic/end/"]
)
def test_ambiguous_path_components_are_rejected(path):
    request = launch_request()
    request["data_root"] = path
    assert not build_pilot_launch(request)["prepared"]


@pytest.mark.parametrize("scene_id", [".", "..", "nested/scene"])
def test_scene_ids_cannot_be_path_expressions(scene_id):
    request = launch_request()
    request["scene_ids"][0] = scene_id
    assert not build_pilot_launch(request)["prepared"]


def test_builder_never_reads_paths_or_invokes_subprocess(monkeypatch):
    import builtins
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("Command preparation must remain pure")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert build_pilot_launch(launch_request())["prepared"]


@pytest.mark.parametrize(
    "scenes",
    [
        None,
        [],
        ["s"] * 5,
        [f"s{i}" for i in range(4)],
        [f"s{i}" for i in range(6)],
        [f"s{i}" for i in range(441)],
        ["a", "b", "c", "d", ""],
        ["a", "b", "c", "d", None],
    ],
)
def test_scene_count_is_explicit_without_full_sweep_fallback(scenes):
    request = launch_request()
    request["scene_ids"] = scenes
    result = build_pilot_launch(request)
    assert not result["prepared"] and result["arms"] == []
    assert result["issues"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", True),
        ("alpasim_sha", "main"),
        ("alpasim_checkout", "relative"),
        ("alpasim_checkout", "/tmp/../other"),
        ("teacher_snapshot", None),
        ("teacher_snapshot", "/tmp/${oc.env:HOME}"),
        ("data_root", "/tmp/new\nline"),
        ("requested_steps", None),
        ("requested_steps", 0),
        ("requested_steps", True),
        ("driver_port", 0),
        ("driver_port", 65536),
        ("driver_port", True),
        ("base_image", "mutable:latest"),
        ("base_image", "alpasim-base:0.89.0"),
        ("base_image", "sha256:" + "B" * 64),
        ("base_image", "sha256:" + "b" * 63),
        ("renderer_image", "image@sha256:pending"),
        ("renderer_camera_override", "6cam_1080"),
        ("renderer_camera_override", "4cam"),
        ("renderer_camera_override", ""),
        ("renderer_camera_override", False),
        ("harmonizer_source", None),
        ("harmonizer_source", {}),
        ("harmonizer_source", "upstream_default_download"),
        ("harmonizer_source", {"kind": "upstream_default_download", "revision": "main"}),
        ("harmonizer_source", {"kind": "pinned_mount"}),
        ("hf_home", None),
        ("hf_home", "relative/hf"),
        ("schema_version", 1),
        ("arms", []),
        ("arms", [None, None]),
    ],
)
def test_invalid_required_values_block_preparation(key, value):
    request = launch_request()
    request[key] = value
    report = build_pilot_launch(request)
    assert not report["prepared"] and report["arms"] == []
    assert report["issues"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("key", list(launch_request()))
def test_missing_required_inputs_never_get_defaults(key):
    request = launch_request()
    del request[key]
    assert not build_pilot_launch(request)["prepared"]


@pytest.mark.parametrize(
    "mutation",
    [
        "reversed",
        "same_mode",
        "same_run",
        "same_log",
        "nested_run",
        "cross_output",
        "extra_field",
        "extra_arm",
    ],
)
def test_arm_modes_and_output_directories_are_isolated(mutation):
    request = launch_request()
    arms = request["arms"]
    if mutation == "reversed":
        arms.reverse()
    elif mutation == "same_mode":
        arms[1]["harmonizer"] = False
    elif mutation == "same_run":
        arms[1]["run_dir"] = arms[0]["run_dir"]
    elif mutation == "same_log":
        arms[1]["driver_log_dir"] = arms[0]["driver_log_dir"]
    elif mutation == "nested_run":
        arms[1]["run_dir"] = arms[0]["run_dir"] + "/child"
    elif mutation == "cross_output":
        arms[1]["driver_log_dir"] = arms[0]["run_dir"]
    elif mutation == "extra_field":
        arms[0]["force_determinism"] = True
    else:
        arms.append(deepcopy(arms[0]))
    report = build_pilot_launch(request)
    assert not report["prepared"] and report["arms"] == []


@pytest.mark.parametrize("payload", [None, [], {}, {"unknown": True}])
def test_malformed_request_returns_plain_report(payload):
    result = build_pilot_launch(payload)
    assert not result["prepared"]
    assert result["cloud_launch_authorized"] is False
    json.dumps(result, allow_nan=False)


# --- A4: R2 L1/L2/L4/L5/L8 and the generated-config/Compose checks ----------------------


def yaml_cameras(path):
    """Read logical_id lines from a pinned camera/driver YAML, in file order."""
    text = upstream_source(path)
    return re.findall(r"^\s*logical_id: (\S+)$", text, re.MULTILINE)


def test_camera_constants_are_derived_from_pinned_yaml():
    assert list(CAMERA_PRESETS["6cam_1080"]) == yaml_cameras(
        "src/wizard/configs/cameras/6cam_1080.yaml")
    assert list(CAMERA_PRESETS["4cam_1080"]) == yaml_cameras(
        "src/wizard/configs/cameras/4cam_1080.yaml")
    driver = upstream_source("src/wizard/configs/driver/alpamayo1_5.yaml").splitlines()
    assert driver[31] == "  use_cameras:"
    assert tuple(line.strip(" -") for line in driver[32:36]) == STOCK_DRIVER_CAMERAS
    assert upstream_source("src/wizard/configs/e2e_challenge/dev.yaml").splitlines()[10] == (
        "  - /cameras: 6cam_1080")
    main = upstream_source("src/driver/src/alpasim_driver/main.py.txt").splitlines()
    assert "not in desired cameras" in main[786]
    assert "set(cfg.inference.use_cameras)" in main[223]


def test_camera_override_is_explicit_and_changes_only_the_preset_token():
    request = launch_request()
    request["renderer_camera_override"] = "4cam_1080"
    report = build_pilot_launch(request)
    assert report["prepared"] and report["issues"] == []
    baseline = build_pilot_launch(launch_request())
    for arm, base in zip(report["arms"], baseline["arms"], strict=True):
        argv = arm["wizard_prepare"]["argv"]
        assert argv[7] == "cameras=4cam_1080"
        assert argv[:7] + argv[8:] == base["wizard_prepare"]["argv"]
    assert report["pilot"]["source_defaults"]["camera_config"] == "4cam_1080"
    assert report["pilot"]["renderer_cameras"] == list(CAMERA_PRESETS["4cam_1080"])
    assert report["pilot"]["owner_decisions"]["renderer_camera_override"] == "4cam_1080"
    assert baseline["pilot"]["owner_decisions"]["renderer_camera_override"] is None


@pytest.mark.parametrize("key", ["renderer_camera_override", "harmonizer_source", "hf_home"])
def test_missing_owner_decision_is_named(key):
    request = launch_request()
    del request[key]
    report = build_pilot_launch(request)
    assert not report["prepared"]
    assert key in report["issues"][0]["message"]


def test_local_build_and_registry_digest_image_kinds():
    assert image_kind("sha256:" + "b" * 64) == "local_build"
    assert image_kind("nvcr.io/nvidia/nre/nre-ga@sha256:" + "a" * 64) == "registry_digest"
    for mutable in ("alpasim-base:0.89.0", "nvcr.io/nvidia/nre/nre-ga:26.04", "sha256:abc",
                    "repo@sha256:" + "A" * 64, None, "", "${oc.env:NRE_IMAGE}"):
        assert image_kind(mutable) is None
    request = launch_request()
    request["base_image"] = "sha256:" + "b" * 64
    report = build_pilot_launch(request)
    assert report["prepared"]
    assert report["pilot"]["image_kinds"] == {"renderer_image": "registry_digest",
                                              "base_image": "local_build"}
    assert ('defines.base_image="sha256:' + "b" * 64 + '"') in (
        report["arms"][0]["wizard_prepare"]["argv"])
    schema = upstream_source("src/wizard/alpasim_wizard/schema.py.txt").splitlines()
    assert schema[156].strip() == 'pull_policy: str = "missing"'
    assert 'base_image: "${defines.image_registry}alpasim-base:${repo-version:}"' in (
        upstream_source("src/wizard/configs/base_config.yaml"))


def test_hydra_run_dirs_are_under_unit_outputs_and_source_backed():
    report = build_pilot_launch(launch_request())
    for label, arm in zip(("off", "on"), report["arms"], strict=True):
        assert arm["wizard_prepare"]["argv"][-1] == (
            f'hydra.run.dir="/synthetic/runs/{label}/.hydra-wizard"')
        assert arm["driver"]["argv"][-1] == f'hydra.run.dir="/synthetic/driver-logs/{label}/.hydra"'
    setup = upstream_source("src/wizard/alpasim_wizard/setup_omegaconf.py.txt").splitlines()
    assert "main_with_bound_config = hydra.main(" in setup[261]
    base = upstream_source("src/wizard/configs/base_config.yaml").splitlines()
    assert base[16:19] == ["hydra:", "  job:", "    chdir: false"]
    main = upstream_source("src/driver/src/alpasim_driver/main.py.txt").splitlines()
    assert main[1105] == "@hydra.main("


def test_pinned_harmonizer_mount_applies_to_both_arms_and_keeps_base_volumes():
    request = launch_request()
    request["harmonizer_source"] = pinned_harmonizer_source()
    report = build_pilot_launch(request)
    assert report["prepared"], report["issues"]
    base = upstream_source("src/wizard/configs/base_config.yaml").splitlines()
    source_volumes = [json.loads(line.strip()[2:]) for line in base[124:126]]
    commands = []
    for arm in report["arms"]:
        argv = arm["wizard_prepare"]["argv"]
        volumes = next(a for a in argv if a.startswith("services.renderer.volumes="))
        assert json.loads(volumes.split("=", 1)[1]) == [
            *source_volumes, "/synthetic/harmonizer:/mnt/harmonizer"]
        command = next(a for a in argv if a.startswith("services.renderer.command="))
        commands.append(json.loads(command.split("=", 1)[1]))
        assert commands[-1][-1] == "--harmonizer-cache=/mnt/harmonizer"
    assert [c for c in commands[1] if c != "--enable-harmonizer"] == commands[0]
    assert report["pilot"]["harmonizer_checkpoint_pinned"] is True
    services = upstream_source("src/wizard/alpasim_wizard/services.py.txt")
    assert 'host, container = string.split(":")' in services


def test_default_download_source_adds_no_mount_or_cache_flag():
    report = build_pilot_launch(launch_request())
    for arm in report["arms"]:
        argv = arm["wizard_prepare"]["argv"]
        assert not any(a.startswith("services.renderer.volumes=") for a in argv)
        assert "--harmonizer-cache" not in " ".join(argv)
    assert report["pilot"]["harmonizer_checkpoint_pinned"] is False


@pytest.mark.parametrize("key,value", [
    ("repository", "other/Harmonizer"), ("file", "other.pt"), ("revision", "main"),
    ("sha256", "e" * 63), ("size_bytes", 0), ("size_bytes", True), ("host_dir", "relative"),
    ("host_dir", "/synthetic/runs/off/harmonizer"), ("extra", 1),
])
def test_pinned_harmonizer_negative_controls(key, value):
    request = launch_request()
    request["harmonizer_source"] = {**pinned_harmonizer_source(), key: value}
    assert not build_pilot_launch(request)["prepared"]


CANARY = "hf_synthetic_canary_DO_NOT_LEAK_0123456789"


def test_environment_allowlist_filters_secret_canary():
    report = build_pilot_launch(launch_request())
    host = {"PATH": "/usr/bin", "HOME": "/root", "HF_TOKEN": CANARY,
            "HUGGING_FACE_HUB_TOKEN": CANARY, "ALPASIM_DRIVER_PORT": "1", "SLURM_JOB_ID": "9",
            "SLURM_ARRAY_TASK_COUNT": "4", "CUDA_VISIBLE_DEVICES": "1", "HF_HOME": "/elsewhere",
            "LANG": "fr_FR.UTF-8", "NRE_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04"}
    for arm in report["arms"]:
        driver = filter_environment(arm["driver"], host)
        assert driver == {"HF_HOME": "/synthetic/hf-home", "HF_HUB_OFFLINE": "1",
                          "HOME": "/root", "LANG": "C.UTF-8", "PATH": "/usr/bin"}
        assert filter_environment(arm["compose"], host) == {
            "HOME": "/root", "LANG": "C.UTF-8", "PATH": "/usr/bin"}
        assert CANARY not in json.dumps(filter_environment(arm["wizard_prepare"], host))
    assert CANARY not in json.dumps(report)


@pytest.mark.parametrize("mutation", ["token", "missing_path", "set_outside", "bad_shape"])
def test_environment_filter_fails_closed(mutation):
    command = deepcopy(build_pilot_launch(launch_request())["arms"][0]["driver"])
    host = {"PATH": "/usr/bin", "HOME": "/root"}
    if mutation == "token":
        command["env_allowlist"].append("HF_TOKEN")
    elif mutation == "missing_path":
        del host["PATH"]
    elif mutation == "set_outside":
        command["env_set"]["EXTRA"] = "1"
    else:
        command = {"argv": []}
    with pytest.raises(ValueError):
        filter_environment(command, host)


def generated(cameras, *, driver=STOCK_DRIVER_CAMERAS, scene="synthetic-scene-0", steps=17):
    user = {"scenes": [{"scene_id": scene}],
            "simulation_config": {"n_rollouts": 1, "n_sim_steps": steps,
                                  "cameras": [{"logical_id": c, "height": 1080} for c in cameras]}}
    driver_config = {"port": 6789, "model": {"checkpoint_path": "/synthetic/teacher/snapshot"},
                     "inference": {"use_cameras": list(driver), "max_batch_size": 1}}
    return user, driver_config


def expectation(preset="4cam_1080"):
    return {"scope": "synthetic",
            "scene_ids": ["synthetic-scene-0"], "requested_steps": 17, "driver_port": 6789,
            "teacher_snapshot": "/synthetic/teacher/snapshot",
            "renderer_cameras": list(CAMERA_PRESETS[preset])}


def test_generated_six_cameras_against_four_driver_cameras_block():
    user, driver = generated(CAMERA_PRESETS["6cam_1080"])
    report = check_generated_configs(user, driver, expectation("6cam_1080"))
    assert report["status"] == "blocked"
    assert [i["code"] for i in report["issues"]] == ["renderer_driver_camera_mismatch"]
    assert "6 cameras" in report["issues"][0]["message"]
    assert "accepts 4" in report["issues"][0]["message"]


def test_four_cameras_in_a_different_order_pass():
    user, driver = generated(CAMERA_PRESETS["4cam_1080"])
    report = check_generated_configs(user, driver, expectation())
    assert report["status"] == "passed", report["issues"]
    assert report["observed"]["renderer_cameras"] != report["observed"]["driver_use_cameras"]


@pytest.mark.parametrize("mutation,code", [
    ("override_not_applied", "renderer_cameras_unexpected"),
    ("scene", "generated_scene_ids_mismatch"),
    ("steps", "generated_n_sim_steps_mismatch"),
    ("duplicate_camera", "generated_cameras_invalid"),
    ("no_driver_cameras", "driver_cameras_invalid"),
])
def test_generated_config_negative_controls(mutation, code):
    cameras = list(CAMERA_PRESETS["4cam_1080"])
    kwargs = {}
    if mutation == "duplicate_camera":
        cameras.append(cameras[0])
    elif mutation == "scene":
        kwargs["scene"] = "synthetic-scene-1"
    elif mutation == "steps":
        kwargs["steps"] = 16
    user, driver = generated(cameras, **kwargs)
    expected = expectation()
    if mutation == "override_not_applied":
        user, driver = generated(CAMERA_PRESETS["6cam_1080"],
                                 driver=CAMERA_PRESETS["6cam_1080"])
    elif mutation == "no_driver_cameras":
        driver["inference"]["use_cameras"] = []
    report = check_generated_configs(user, driver, expected)
    assert report["status"] == "blocked"
    assert code in {i["code"] for i in report["issues"]}


@pytest.mark.parametrize("args", [(None, {}, {}), ({}, {}, {"scene_ids": []})])
def test_generated_config_check_rejects_malformed_inputs(args):
    assert check_generated_configs(*args)["status"] == "blocked"


def compose_config(*, harmonizer=False, cache=None, shm=None):
    renderer_cmd = "umask 0000\n/app/x serve-grpc --port=6790 --no-enable-nrend"
    if harmonizer:
        renderer_cmd += " --enable-harmonizer"
    if cache:
        renderer_cmd += f" --harmonizer-cache={cache}"
    services = {}
    for name in ("renderer-0", "physics-0", "controller-0", "runtime-0"):
        services[name] = {
            "image": ("nvcr.io/nvidia/nre/nre-ga@sha256:" + "a" * 64 if name == "renderer-0"
                      else "sha256:" + "b" * 64),
            "network_mode": "host", "pull_policy": "never",
            "command": ["-c", renderer_cmd if name == "renderer-0" else "umask 0000\nrun"],
        }
        if name == "renderer-0":
            services[name]["volumes"] = [
                "/synthetic/data/nre-artifacts:/mnt/nre-data",
                "/synthetic/data/nre-artifacts/ego-hoods:/mnt/ego-hoods"]
            if cache:
                services[name]["volumes"].append(f"/synthetic/harmonizer:{cache}")
        if shm and name in shm:
            services[name]["shm_size"] = shm[name]
    return {"services": services}


def compose_expectation(shm=None, *, harmonizer=False, cache=None):
    return {"scope": "synthetic", "renderer_image": "nvcr.io/nvidia/nre/nre-ga@sha256:" + "a" * 64,
            "base_image": "sha256:" + "b" * 64, "harmonizer": harmonizer,
            "harmonizer_cache": cache,
            "shm_size": shm or {name: "docker_default" for name in
                                ("controller-0", "physics-0", "renderer-0", "runtime-0")},
            "renderer_volumes": {"/mnt/ego-hoods": "/synthetic/data/nre-artifacts/ego-hoods",
                                 **({cache: "/synthetic/harmonizer"} if cache else {})}}


def test_compose_check_known_answers():
    assert check_generated_compose(compose_config(), compose_expectation())["status"] == "passed"
    shm = {"controller-0": "docker_default", "physics-0": "docker_default",
           "renderer-0": "64g", "runtime-0": "docker_default"}
    # The generator never writes shm_size: a declared 64g blocks on generated output.
    blocked = check_generated_compose(compose_config(), compose_expectation(shm))
    assert [i["code"] for i in blocked["issues"]] == ["shm_size_mismatch"]
    effective = compose_config(shm={"renderer-0": "68719476736"})
    assert check_generated_compose(effective, compose_expectation(shm))["status"] == "passed"
    on = check_generated_compose(compose_config(harmonizer=True, cache="/mnt/harmonizer"),
                                 compose_expectation(harmonizer=True, cache="/mnt/harmonizer"))
    assert on["status"] == "passed"


@pytest.mark.parametrize("mutation,code", [
    ("flag", "harmonizer_flag_mismatch"), ("cache", "harmonizer_cache_mismatch"),
    ("bridge", "network_mode_not_host"), ("image", "image_mismatch"),
    ("build", "unexpected_build_recipe"), ("services", "unexpected_compose_services"),
    ("shm_set", "shm_size_mismatch"), ("shm_undeclared", "shm_size_undeclared"),
    ("command", "renderer_command_unrecognized"),
])
def test_compose_check_negative_controls(mutation, code):
    compose, expected = compose_config(), compose_expectation()
    services = compose["services"]
    if mutation == "flag":
        expected["harmonizer"] = True
    elif mutation == "cache":
        expected = compose_expectation(cache="/mnt/harmonizer")
        services["renderer-0"]["volumes"].append("/synthetic/harmonizer:/mnt/harmonizer")
    elif mutation == "bridge":
        services["physics-0"]["network_mode"] = "bridge"
    elif mutation == "image":
        services["runtime-0"]["image"] = "alpasim-base:0.89.0"
    elif mutation == "build":
        services["controller-0"]["build"] = {"context": "."}
    elif mutation == "services":
        services["trafficsim-0"] = dict(services["physics-0"])
    elif mutation == "shm_set":
        services["runtime-0"]["shm_size"] = "1g"
    elif mutation == "shm_undeclared":
        expected["shm_size"] = {}
    else:
        services["renderer-0"]["command"] = "serve"
    report = check_generated_compose(compose, expected)
    assert report["status"] == "blocked"
    assert code in {i["code"] for i in report["issues"]}


def test_pull_policy_missing_is_a_warning_because_argv_forces_never():
    compose = compose_config()
    compose["services"]["runtime-0"]["pull_policy"] = "missing"
    report = check_generated_compose(compose, compose_expectation())
    assert report["status"] == "passed"
    assert [(i["code"], i["severity"]) for i in report["issues"]] == [
        ("pull_policy_not_never", "warning")]


def test_docker_byte_sizes_known_answers():
    assert docker_bytes("64g") == 68_719_476_736
    assert docker_bytes("64m") == 67_108_864
    assert docker_bytes("1gb") == 1_073_741_824
    assert docker_bytes(67_108_864) == 67_108_864
    for invalid in ("0", "-1g", "64 x", "", None, True, 1.5, "1e3"):
        assert docker_bytes(invalid) is None


def test_compose_generator_sets_no_shm_size_and_host_network():
    compose = upstream_source("src/wizard/alpasim_wizard/deployment/docker_compose.py.txt")
    assert "shm" not in compose
    lines = compose.splitlines()
    assert lines[110].strip() == 'ret["network_mode"] = "host"'
    assert lines[114].strip() == 'ret["pull_policy"] = container.service_config.pull_policy'
    dev = upstream_source("src/wizard/configs/e2e_challenge/dev.yaml").splitlines()
    assert dev[23:25] == ["  debug_flags:", "    use_localhost: true"]


def test_runtime_capture_source_manifest_matches_git_blobs_and_sha256():
    manifest = json.loads((SOURCE_ROOT / "manifest_runtime_capture.json").read_text())
    assert manifest["license_fixture_path"] == "alpasim/LICENSE"
    assert len(manifest["sources"]) == 8
    for source in manifest["sources"]:
        assert source["commit"] == "e121e59141dc391f6c295513d6dadc840a419149"
        data = (SOURCE_ROOT / source["fixture_path"]).read_bytes()
        assert len(data) == source["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        header = f"blob {len(data)}\0".encode()
        assert hashlib.sha1(header + data).hexdigest() == source["git_blob_sha1"]


# --- Review A4-F4/F5 regression controls -------------------------------------------------


@pytest.mark.parametrize("key,value", [
    ("renderer_cameras", None), ("renderer_cameras", 7), ("renderer_cameras", ["a", "a"]),
    ("renderer_cameras", [["camera"]]), ("renderer_cameras", []),
    ("scene_ids", None), ("scene_ids", [None]), ("requested_steps", True),
    ("requested_steps", 0), ("driver_port", "6789"), ("teacher_snapshot", None),
])
def test_malformed_expectation_values_return_a_blocked_report(key, value):
    # A-F4 repro R6: renderer_cameras None/int raised TypeError.
    user, driver = generated(CAMERA_PRESETS["4cam_1080"])
    expected = expectation()
    expected[key] = value
    report = check_generated_configs(user, driver, expected)
    assert report["status"] == "blocked"
    assert [i["code"] for i in report["issues"]] == ["invalid_check_input"]


EGO_HOOD_MOUNT = ("/synthetic/data/nre-artifacts/ego-hoods", "/mnt/ego-hoods")
HARMONIZER_MOUNT = ("/synthetic/harmonizer", "/mnt/harmonizer")


def with_volumes(compose, mounts, *, long_form=False):
    entries = []
    for source, target in mounts:
        entries.append({"type": "bind", "source": source, "target": target,
                        "bind": {"create_host_path": True}} if long_form
                       else f"{source}:{target}")
    compose["services"]["renderer-0"]["volumes"] = [
        "/synthetic/data/nre-artifacts:/mnt/nre-data", *entries]
    return compose


def volume_expectation(*, pinned=False, harmonizer=False):
    expected = compose_expectation(harmonizer=harmonizer,
                                   cache="/mnt/harmonizer" if pinned else None)
    expected["renderer_volumes"] = {EGO_HOOD_MOUNT[1]: EGO_HOOD_MOUNT[0]}
    if pinned:
        expected["renderer_volumes"][HARMONIZER_MOUNT[1]] = HARMONIZER_MOUNT[0]
    return expected


@pytest.mark.parametrize("long_form", [False, True])
def test_compose_check_verifies_renderer_volumes_in_short_and_long_form(long_form):
    unpinned = with_volumes(compose_config(), [EGO_HOOD_MOUNT], long_form=long_form)
    assert check_generated_compose(unpinned, volume_expectation())["status"] == "passed"
    pinned = with_volumes(compose_config(harmonizer=True, cache="/mnt/harmonizer"),
                          [EGO_HOOD_MOUNT, HARMONIZER_MOUNT], long_form=long_form)
    report = check_generated_compose(pinned, volume_expectation(pinned=True, harmonizer=True))
    assert report["status"] == "passed", report["issues"]


@pytest.mark.parametrize("mutation,code", [
    ("no_volumes", "renderer_volume_missing"),
    ("no_harmonizer_mount", "renderer_volume_missing"),
    ("wrong_harmonizer_source", "renderer_volume_mismatch"),
    ("duplicate_target", "renderer_volume_mismatch"),
    ("unexpected_harmonizer_mount", "renderer_volume_unexpected"),
    ("unrecognised_entry", "renderer_volume_unrecognized"),
])
def test_compose_check_blocks_missing_or_wrong_renderer_volumes(mutation, code):
    # A-F5 repro R7: a pinned unit without its mounts previously passed.
    pinned = mutation in ("no_harmonizer_mount", "wrong_harmonizer_source")
    compose = compose_config(harmonizer=True, cache="/mnt/harmonizer" if pinned else None)
    mounts = [EGO_HOOD_MOUNT, HARMONIZER_MOUNT] if pinned else [EGO_HOOD_MOUNT]
    if mutation == "no_harmonizer_mount":
        mounts = [EGO_HOOD_MOUNT]
    elif mutation == "wrong_harmonizer_source":
        mounts = [EGO_HOOD_MOUNT, ("/synthetic/other", "/mnt/harmonizer")]
    elif mutation == "duplicate_target":
        mounts = [EGO_HOOD_MOUNT, ("/synthetic/other", "/mnt/ego-hoods")]
    elif mutation == "unexpected_harmonizer_mount":
        mounts = [EGO_HOOD_MOUNT, HARMONIZER_MOUNT]
    compose = with_volumes(compose, mounts)
    if mutation == "no_volumes":
        del compose["services"]["renderer-0"]["volumes"]
    elif mutation == "unrecognised_entry":
        compose["services"]["renderer-0"]["volumes"].append({"type": "volume"})
    report = check_generated_compose(compose, volume_expectation(pinned=pinned, harmonizer=True))
    assert report["status"] == "blocked"
    assert code in {i["code"] for i in report["issues"]}


# --- Review A4-F6: shared artifact contract (scope, producer, status vocabulary) ----------

CONTRACT_STATUSES = {"passed", "blocked", "not_run", "not_estimable", "unverified",
                     "not_applicable", "qualify_on_host"}


def test_launch_report_carries_kind_scope_and_producer_identity():
    report = build_pilot_launch(launch_request())
    assert report["prepared"], report["issues"]
    assert report["artifact_kind"] == "s2_pilot_launch"
    assert report["scope"] == "synthetic"
    source = Path(__file__).parents[1] / "gate" / "s2_launch.py"
    assert report["producer"] == {
        "module": "gate.s2_launch", "function": "build_pilot_launch",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


@pytest.mark.parametrize("scope", [None, "assumed", "SYNTHETIC", 1])
def test_launch_scope_is_a_required_declared_input(scope):
    request = launch_request()
    request["scope"] = scope
    report = build_pilot_launch(request)
    assert not report["prepared"] and report["scope"] is None
    assert "invalid_scope" in {i["code"] for i in report["issues"]}
    del request["scope"]
    assert "invalid_request" in {i["code"] for i in build_pilot_launch(request)["issues"]}


def test_generated_file_checks_carry_scope_producer_and_contract_status():
    user, driver = generated(CAMERA_PRESETS["4cam_1080"])
    configs = check_generated_configs(user, driver, expectation())
    compose = check_generated_compose(compose_config(), compose_expectation())
    for report, function in ((configs, "check_generated_configs"),
                             (compose, "check_generated_compose")):
        assert report["status"] == "passed", report["issues"]
        assert report["scope"] == "synthetic"
        assert report["producer"]["module"] == "gate.s2_launch"
        assert report["producer"]["function"] == function
        assert report["status"] in CONTRACT_STATUSES


@pytest.mark.parametrize("scope", [None, "assumed"])
def test_generated_file_checks_require_a_valid_scope(scope):
    user, driver = generated(CAMERA_PRESETS["4cam_1080"])
    expected = expectation()
    expected["scope"] = scope
    configs = check_generated_configs(user, driver, expected)
    expected = compose_expectation()
    expected["scope"] = scope
    compose = check_generated_compose(compose_config(), expected)
    for report in (configs, compose):
        assert report["status"] == "blocked" and report["scope"] is None
        assert [i["code"] for i in report["issues"]] == ["invalid_check_input"]


@pytest.mark.parametrize("key,value", [
    ("harmonizer", None), ("harmonizer_cache", "/mnt/other"), ("renderer_image", 1),
    ("shm_size", None), ("renderer_volumes", None), ("renderer_volumes", {}),
    ("renderer_volumes", {"/mnt/ego-hoods": "relative/path"}),
    ("renderer_volumes", {"/mnt/ego-hoods": "/a", "/mnt/harmonizer": "/b"}),
])
def test_malformed_compose_expectation_values_return_a_blocked_report(key, value):
    expected = compose_expectation()
    expected[key] = value
    report = check_generated_compose(compose_config(), expected)
    assert report["status"] == "blocked"
    assert [i["code"] for i in report["issues"]] == ["invalid_check_input"]


# --- Review A2-F12: Harmonizer licence gate on the on arm ----------------------------


@pytest.mark.parametrize("source", ["upstream_default_download", "pinned_mount"])
@pytest.mark.parametrize(
    "ref", [None, "", "   ", "TBD", "placeholder", 7, ["ref"], "hf_" + "a" * 34, "${oc.env:X}",
            "ref\nwith-newline"])
def test_on_arm_requires_recorded_harmonizer_terms(source, ref):
    request = launch_request()
    if source == "pinned_mount":
        request["harmonizer_source"] = pinned_harmonizer_source()
    request["harmonizer_terms_authorization_ref"] = ref
    report = build_pilot_launch(request)
    assert not report["prepared"] and report["arms"] == [] and report["pilot"] is None
    blocking = [(i["code"], i["path"]) for i in report["issues"] if i["severity"] == "blocking"]
    assert blocking == [("harmonizer_terms_unrecorded", "harmonizer_terms_authorization_ref")]
    assert "terms:harmonizer" in report["issues"][0]["message"]
    assert "arms[1]" in report["issues"][0]["message"]
    json.dumps(report, allow_nan=False)


def test_harmonizer_terms_ref_is_bound_to_the_on_arm_and_never_enters_commands():
    first = build_pilot_launch(launch_request())
    request = launch_request()
    request["harmonizer_terms_authorization_ref"] = "SYNTHETIC-other-harmonizer-terms"
    second = build_pilot_launch(request)
    assert first["prepared"] and second["prepared"]
    assert first["pilot"]["harmonizer_terms_authorization"] == {
        "authorization_id": "terms:harmonizer",
        "authorization_ref": "SYNTHETIC-harmonizer-terms",
        "required_by_arms": ["on"],
    }
    # Off arm unaffected and the reference never reaches argv or environments.
    assert first["arms"] == second["arms"]
    assert "SYNTHETIC-harmonizer-terms" not in json.dumps(first["arms"])
