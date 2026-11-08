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
from s2_launch_fixture import launch_request

from gate.s2_launch import build_pilot_launch

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
    assert report["issues"] == []
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
    ]
    assert wizard["may_download_assets"] is True


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
            ],
        }
        assert arm["compose"] == {
            "cwd": f"/synthetic/runs/{label}",
            "argv": ["docker", "compose", "up", "--exit-code-from", "runtime-0"],
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
        ("renderer_image", "image@sha256:pending"),
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
