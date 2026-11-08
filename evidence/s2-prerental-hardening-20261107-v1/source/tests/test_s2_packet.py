"""Known-answer command/schedule binding; nothing launches or reads host paths."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from s2_packet_fixture import packet_request

from gate.s2_identity import identity_hash
from gate.s2_launch import build_pilot_launch
from gate.s2_packet import build_pilot_packet
from gate.s2_schedule import build_schedule


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def override(unit, key):
    return next(
        item.split("=", 1)[1]
        for item in unit["wizard_prepare"]["argv"]
        if item.startswith(key + "=")
    )


def test_known_answer_packet_binds_every_job_to_one_scene_and_matching_mode():
    request = packet_request()
    before = deepcopy(request)
    report = build_pilot_packet(request)
    assert report["prepared"], report["issues"]
    assert report["issues"] == []
    assert request == before
    assert report["inputs"] == request
    assert report["request_sha256"] == identity_hash(request)
    assert report["schedule"] == build_schedule(**request["schedule_request"])
    assert report["schedule_sha256"] == identity_hash(report["schedule"])
    assert len(report["units"]) == 10
    assert [unit["job"]["job_id"] for unit in report["units"]] == request["execution_order"]["job_ids"]
    paths = set()
    for index, unit in enumerate(report["units"]):
        job = unit["job"]
        mode = "off" if index % 2 == 0 else "on"
        assert unit["harmonizer_mode"] == mode
        assert job["arm"] == "A"
        assert job["protocol_hash"] == request["schedule_request"]["protocol_hashes"][mode]
        assert unit["asset_sha256"] == job["scene_id"].rsplit("-", 1)[1] * 64
        assert unit["upstream_identity"] is None
        assert json.loads(override(unit, "scenes.scene_ids")) == [job["scene_id"]]
        assert override(unit, "scenes.test_suite_id") == "null"
        assert override(unit, "runtime.simulation_config.n_rollouts") == "1"
        assert override(unit, "runtime.simulation_config.n_sim_steps") == "3"
        run_dir = f"/synthetic/runs/{mode}/{job['job_id']}"
        log_dir = f"/synthetic/driver-logs/{mode}/{job['job_id']}"
        assert json.loads(override(unit, "wizard.log_dir")) == run_dir
        assert json.loads(override(unit, "driver.output_dir")) == log_dir
        assert f"--config-path={run_dir}" in unit["driver"]["argv"]
        assert unit["compose"]["cwd"] == run_dir
        assert unit["wizard_prepare"]["cwd"] == "/synthetic/alpasim"
        assert unit["driver"]["cwd"] == "/synthetic/alpasim"
        renderer = json.loads(override(unit, "services.renderer.command"))
        assert ("--enable-harmonizer" in renderer) == (mode == "on")
        assert "--enable-timing" in renderer
        assert unit["launch_request_sha256"] == identity_hash(request["launch_request"])
        assert unit["schedule_request_sha256"] == identity_hash(request["schedule_request"])
        commands = {key: unit[key] for key in ("wizard_prepare", "driver", "compose")}
        assert unit["commands_sha256"] == identity_hash(commands)
        assert unit["requires_previous_unit_teardown"] is True
        paths.update((run_dir, log_dir))
    assert len(paths) == 20
    assert report["execution_model"] == "serial_units_with_driver_restart_and_compose_teardown"
    assert report["startup_cost_basis"] == "unmeasured_per_unit_startup_included_in_paid_work"
    for field in (
        "cloud_launch_authorized", "publication_authorized", "runtime_qualified",
        "execution_order_registration_verified", "commands_executed", "asset_identities_verified",
    ):
        assert report[field] is False
    json.dumps(report, allow_nan=False)


def test_packet_snapshots_inputs_without_references_to_mutable_caller_objects():
    request = packet_request()
    report = build_pilot_packet(request)
    expected = deepcopy(report)
    request["launch_request"]["scene_ids"].clear()
    request["schedule_request"]["scenes"][0]["scene_id"] = "changed"
    request["execution_order"]["job_ids"].clear()
    assert report == expected


def test_explicit_permutation_is_retained_without_claiming_randomness():
    request = packet_request()
    order = request["execution_order"]
    order["method"] = "preregistered_permutation"
    order["job_ids"] = order["job_ids"][::2] + order["job_ids"][1::2]
    report = build_pilot_packet(request)
    assert report["prepared"]
    assert [unit["job"]["job_id"] for unit in report["units"]] == order["job_ids"]
    assert report["execution_order_registration_verified"] is False
    assert report["randomization_verified"] is False


def test_alternating_order_can_begin_with_either_mode():
    request = packet_request()
    request["execution_order"]["job_ids"].reverse()
    report = build_pilot_packet(request)
    assert report["prepared"]
    assert report["units"][0]["harmonizer_mode"] == "on"


def test_scene_input_order_is_not_mistaken_for_execution_order():
    request = packet_request()
    request["launch_request"]["scene_ids"].reverse()
    request["schedule_request"]["scenes"].reverse()
    report = build_pilot_packet(request)
    assert report["prepared"]
    assert [unit["job"]["job_id"] for unit in report["units"]] == request["execution_order"]["job_ids"]


def test_mixed_case_declared_hashes_bind_normalized_jobs_and_preserve_original_input():
    request = packet_request()
    source = request["schedule_request"]
    source["protocol_hashes"] = {"off": "aA" * 32, "on": "bB" * 32}
    source["scenes"][0]["asset_sha256"] = "cC" * 32
    request["execution_order"]["job_ids"] = [job["job_id"] for job in build_schedule(**source)]
    report = build_pilot_packet(request)
    assert report["prepared"]
    assert report["inputs"] == request
    assert [unit["harmonizer_mode"] for unit in report["units"]] == ["off", "on"] * 5
    assert {unit["asset_sha256"] for unit in report["units"][:2]} == {"c" * 64}
    assert report["units"][0]["job"]["protocol_hash"] == "a" * 64


@pytest.mark.parametrize("field", ["requested_steps", "scene_ids"])
def test_disconnected_launch_and_schedule_are_rejected_without_partial_units(field):
    request = packet_request()
    if field == "requested_steps":
        request["launch_request"][field] = 17
    else:
        request["launch_request"][field][0] = "other-scene"
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert report["units"] == []
    assert "launch_schedule_mismatch" in codes(report)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "wrong_method", "extra", "wrong_type", "not_alternating"])
def test_bad_order_fails_closed(mutation):
    request = packet_request()
    order = request["execution_order"]
    if mutation == "missing":
        order["job_ids"].pop()
    elif mutation == "duplicate":
        order["job_ids"][1] = order["job_ids"][0]
    elif mutation == "unknown":
        order["job_ids"][0] = "f" * 64
    elif mutation == "wrong_method":
        order["method"] = "automatic"
    elif mutation == "extra":
        order["seed"] = 1
    elif mutation == "wrong_type":
        order["job_ids"][0] = {}
    else:
        order["job_ids"][1], order["job_ids"][2] = order["job_ids"][2], order["job_ids"][1]
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert report["units"] == []
    assert "invalid_execution_order" in codes(report)


@pytest.mark.parametrize("value", [None, [], {}, "x", True, 1])
@pytest.mark.parametrize("field", ["launch_request", "schedule_request", "execution_order"])
def test_malformed_nested_input_always_returns_json_report(field, value):
    request = packet_request()
    request[field] = value
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert report["units"] == []
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("phase", "full_sweep"), ("phase", []), ("replicates", 2),
    ("protocol_hashes", {"off": "a" * 64, "on": "a" * 64}),
    ("scenes", [{"scene_id": []}]), ("study_id", None),
])
def test_schedule_contract_errors_are_reported(field, value):
    request = packet_request()
    request["schedule_request"][field] = value
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert report["units"] == []
    assert "invalid_schedule_request" in codes(report)


@pytest.mark.parametrize("value", [None, [], {}, 1, float("nan"), {1: "x"}])
def test_bad_outer_requests_are_total(value):
    report = build_pilot_packet(value)
    assert not report["prepared"]
    assert report["units"] == []
    json.dumps(report, allow_nan=False)


def test_cyclic_input_is_explicitly_rejected():
    request = packet_request()
    request["launch_request"]["scene_ids"].append(request)
    report = build_pilot_packet(request)
    assert not report["prepared"]
    json.dumps(report, allow_nan=False)


def test_singleton_internal_units_do_not_weaken_public_five_scene_api():
    request = packet_request()
    request["launch_request"]["scene_ids"] = request["launch_request"]["scene_ids"][:1]
    assert not build_pilot_launch(request["launch_request"])["prepared"]
    assert not build_pilot_packet(request)["prepared"]


def test_stale_job_ids_after_asset_identity_change_cannot_prepare():
    request = packet_request()
    request["schedule_request"]["scenes"][0]["asset_sha256"] = "f" * 64
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert "invalid_execution_order" in codes(report)


def test_derived_output_ancestry_collision_fails_without_partial_units():
    request = packet_request()
    first_job = request["execution_order"]["job_ids"][0]
    arm = request["launch_request"]["arms"][0]
    arm["driver_log_dir"] = arm["run_dir"] + "/" + first_job
    assert build_pilot_launch(request["launch_request"])["prepared"]
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert report["units"] == []
    assert "overlapping_unit_outputs" in codes(report)


def test_equal_run_and_driver_base_within_one_arm_preserves_cross_unit_isolation():
    request = packet_request()
    for arm in request["launch_request"]["arms"]:
        arm["driver_log_dir"] = arm["run_dir"]
    assert build_pilot_packet(request)["prepared"]


def test_source_supports_single_scene_selection_and_none_compose_generation():
    root = Path(__file__).parent / "fixtures/alpasim_launch_upstream/alpasim"
    context = (root / "src/wizard/alpasim_wizard/context.py.txt").read_text()
    schema = (root / "src/wizard/alpasim_wizard/schema.py.txt").read_text()
    wizard = (root / "src/wizard/alpasim_wizard/wizard.py.txt").read_text()
    assert "scene_ids: list[str] | None" in schema
    assert "artifacts = manager.query_by_scene_ids(scene_ids)" in context
    assert "artifacts = sorted(artifacts, key=lambda x: x.scene_id)" in context
    assert "docker_compose_deployment.generate_docker_compose()" in wizard
    assert "(NONE generates docker-compose files that will be run manually)" in wizard
    assert '"Run \'docker compose up --exit-code-from runtime-0\' in %s "' in wizard


def test_packet_is_pure_and_does_not_import_wizard_or_execute_commands(monkeypatch):
    import builtins
    import subprocess

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("alpasim")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("packet preparation must be pure")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert build_pilot_packet(packet_request())["prepared"]
