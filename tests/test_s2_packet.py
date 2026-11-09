"""Known-answer command/schedule binding; nothing launches or reads host paths."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from s2_packet_fixture import diagnostic_unit, ego_hood_staging, packet_request

from gate.s2_identity import identity_hash
from gate.s2_launch import build_pilot_launch, check_generated_compose
from gate.s2_packet import EGO_HOOD_FILES, SCHEMA_VERSION, build_pilot_packet
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
    assert [(i["code"], i["severity"]) for i in report["issues"]] == [
        ("renderer_driver_camera_mismatch_predicted", "warning")]
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
        assert unit["upstream_attempts"] == []
        assert unit["upstream_attempts_status"] == "not_run"
        assert unit["run_uuid_binding"] == {
            "source": f"/synthetic/runs/{mode}/{job['job_id']}/run_metadata.yaml#run_uuid",
            "status": "not_run", "run_uuid": None}
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
    assert report["execution_model"] == (
        "serial_units_driver_restart_capture_before_compose_teardown")
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


# --- A4: L6 ego hoods, L7 retries, P1/P3 attempts and captures, shm_size ----------------

UPSTREAM = Path(__file__).parent / "fixtures" / "alpasim_launch_upstream" / "alpasim"


def test_ego_hood_pin_matches_literal_git_tree_and_source_mounts():
    # Literal `git ls-tree -l e121e591 data/nre-artifacts/ego-hoods` and sha256 of each blob.
    assert [(f["path"], f["size_bytes"], f["git_blob_sha1"], f["sha256"])
            for f in EGO_HOOD_FILES] == [
        ("hyperion_8/camera_front_wide_120fov.png", 808775,
         "1e4cbd9f0a023f3cd3578b707380e61a9d4a39ca",
         "04fa3f354940d03984a37eb755ebf845be4b4bdae39c178b8de990d0b045392d"),
        ("hyperion_8_1/camera_front_wide_120fov.png", 425624,
         "1a07fea50bf774883fc3a7b9df8555d97b9dae98",
         "b7c36c0d179e82cc0d820a202fb9b0017e4006aa815cc850fbd3ea1848d9a8cd"),
    ]
    base = (UPSTREAM / "src/wizard/configs/base_config.yaml").read_text().splitlines()
    assert base[30].strip() == 'sensordata: "${defines.filesystem}/nre-artifacts"'
    assert base[125].strip() == '- "${defines.sensordata}/ego-hoods:/mnt/ego-hoods"'
    assert 'ego_mask_rig_config_id: "hyperion_8_1"' in base[286]
    dev = (UPSTREAM / "src/wizard/configs/e2e_challenge/dev.yaml").read_text().splitlines()
    assert dev[22].strip() == "validate_mount_points: false"


def test_packet_records_declared_staging_without_claiming_host_verification():
    report = build_pilot_packet(packet_request())
    assert report["ego_hood_staging"]["declared_matches_pin"] is True
    assert report["ego_hood_staging"]["host_verified"] is False
    assert report["ego_hood_staging"]["scope"] == "synthetic"
    assert report["ego_hood_staging_host_verified"] is False


@pytest.mark.parametrize("mutation", [
    "missing", "wrong_hash", "missing_file", "extra_file", "wrong_root", "bad_scope", "kind",
    "duplicate",
])
def test_unit_preparation_requires_pinned_ego_hood_staging(mutation):
    request = packet_request()
    staging = request["ego_hood_staging"]
    if mutation == "missing":
        del request["ego_hood_staging"]
    elif mutation == "wrong_hash":
        staging["files"][0]["sha256"] = "f" * 64
    elif mutation == "missing_file":
        staging["files"].pop()
    elif mutation == "extra_file":
        staging["files"].append({**staging["files"][0], "path": "hyperion_9/x.png"})
    elif mutation == "wrong_root":
        staging["staged_root"] = "/synthetic/data/ego-hoods"
    elif mutation == "bad_scope":
        staging["scope"] = "assumed"
    elif mutation == "kind":
        staging["artifact_kind"] = "receipt"
    else:
        staging["files"].append(dict(staging["files"][0]))
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert report["units"] == []
    expected = "invalid_request" if mutation == "missing" else "ego_hood_staging_required"
    assert expected in codes(report)


def test_staged_root_follows_the_launch_data_root():
    request = packet_request()
    request["launch_request"]["data_root"] = "/other/data"
    assert not build_pilot_packet(request)["prepared"]
    request["ego_hood_staging"] = ego_hood_staging("/other/data")
    assert build_pilot_packet(request)["prepared"]


@pytest.mark.parametrize("value", [None, {}, {"renderer-0": "64g"}, "64g"])
def test_shm_size_is_required_for_every_service_without_default(value):
    request = packet_request()
    request["shm_size"] = value
    report = build_pilot_packet(request)
    assert not report["prepared"]
    assert "invalid_shm_size" in codes(report)


@pytest.mark.parametrize("size", ["0g", "64x", "64", "64gb", " 64g", "1234567g"])
def test_shm_size_rejects_malformed_values(size):
    request = packet_request()
    request["shm_size"]["renderer-0"] = size
    assert "invalid_shm_size" in codes(build_pilot_packet(request))


def test_declared_shm_size_reaches_each_unit_compose_expectation_and_check():
    request = packet_request()
    request["shm_size"]["renderer-0"] = "64g"
    report = build_pilot_packet(request)
    assert report["prepared"], report["issues"]
    unit = report["units"][0]
    expected = unit["compose_expectations"]
    assert expected["shm_size"]["renderer-0"] == "64g"
    assert expected["harmonizer"] is (unit["harmonizer_mode"] == "on")
    assert expected["harmonizer_cache"] is None
    services = {
        name: {"image": (request["launch_request"]["renderer_image"] if name == "renderer-0"
                         else request["launch_request"]["base_image"]),
               "network_mode": "host", "pull_policy": "never",
               "command": ["-c", "umask 0000\n/app/x serve-grpc"]}
        for name in ("controller-0", "physics-0", "renderer-0", "runtime-0")}
    services["renderer-0"]["volumes"] = [
        "/synthetic/data/nre-artifacts:/mnt/nre-data",
        "/synthetic/data/nre-artifacts/ego-hoods:/mnt/ego-hoods"]
    generated = check_generated_compose({"services": services}, expected)
    assert [i["code"] for i in generated["issues"]] == ["shm_size_mismatch"]
    services["renderer-0"]["shm_size"] = "64g"
    assert check_generated_compose({"services": services}, expected)["status"] == "passed"


def test_units_require_captures_before_teardown_and_never_trust_exit_codes():
    report = build_pilot_packet(packet_request())
    unit = report["units"][0]
    run_dir = unit["compose"]["cwd"]
    capture_dir = run_dir + "/physloop-capture"
    assert unit["capture_dir"] == capture_dir
    assert [(c["id"], c["argv"]) for c in unit["required_captures_before_teardown"]] == [
        ("compose_up_stdout", None),
        ("compose_logs", ["docker", "compose", "logs", "--no-color", "--timestamps"]),
        ("compose_ps", ["docker", "compose", "ps", "-a", "--format", "json"]),
        ("compose_container_ids", ["docker", "compose", "ps", "-a", "-q"]),
        ("container_inspect", None),
    ]
    first = unit["required_captures_before_teardown"][0]
    assert first["stdout_path"] == capture_dir + "/compose-up.stdout.txt"
    inspect = unit["required_captures_before_teardown"][-1]
    assert inspect["argv_template"] == ["docker", "inspect", "{container_ids}"]
    assert inspect["placeholder_source"] == "compose_container_ids"
    assert unit["teardown"]["argv"] == ["docker", "compose", "down", "--timeout", "10"]
    assert unit["teardown"]["requires"] == [
        "compose_up_stdout", "compose_logs", "compose_ps", "compose_container_ids",
        "container_inspect"]
    assert unit["compose_config"]["argv"] == ["docker", "compose", "config", "--format", "json"]
    assert unit["compose"]["argv"] == ["docker", "compose", "up", "--pull", "never",
                                       "--exit-code-from", "runtime-0"]
    assert unit["compose"]["exit_code_is_success_evidence"] is False
    for step in (*unit["required_captures_before_teardown"][1:], unit["teardown"]):
        assert step["cwd"] == run_dir
        assert step["env_allowlist"] == ["HOME", "LANG", "PATH"]
    assert "exit code ignored" in report["unit_completion_rule"]


def test_retry_controls_document_noop_nr_retries_and_internal_retry_with_sources():
    controls = build_pilot_packet(packet_request())["retry_controls"]
    assert controls["wizard.nr_retries"]["effect_on_run_method_NONE"] == "none"
    assert controls["runtime_internal_retry"]["max_retry_attempts"] == 1
    assert controls["runtime_internal_retry"]["configurable"] is False
    slurm = (UPSTREAM / "src/wizard/alpasim_wizard/deployment/slurm.py.txt").read_text()
    assert "self.context.cfg.wizard.nr_retries" in slurm.splitlines()[66]
    wizard_sources = [p for p in (UPSTREAM / "src/wizard/alpasim_wizard").rglob("*.py.txt")
                      if "nr_retries" in p.read_text()]
    assert sorted(p.name for p in wizard_sources) == ["schema.py.txt", "slurm.py.txt"]
    ipc = (UPSTREAM / "src/runtime/alpasim_runtime/worker/ipc.py.txt").read_text()
    assert ipc.splitlines()[49].strip() == "max_retry_attempts: int = 1"
    scheduler = (UPSTREAM / "src/runtime/alpasim_runtime/daemon/scheduler.py.txt").read_text()
    assert scheduler.splitlines()[198].strip() == 'session_uuid="",'


def test_generated_config_expectations_follow_the_camera_decision():
    request = packet_request()
    request["launch_request"]["renderer_camera_override"] = "4cam_1080"
    report = build_pilot_packet(request)
    assert report["prepared"] and report["issues"] == []
    expected = report["units"][0]["generated_config_expectations"]
    assert expected == {
        "scope": "synthetic",
        "scene_ids": [report["units"][0]["job"]["scene_id"]], "requested_steps": 3,
        "driver_port": 6789, "teacher_snapshot": "/synthetic/teacher/snapshot",
        "renderer_cameras": ["camera_front_wide_120fov", "camera_front_tele_30fov",
                             "camera_cross_left_120fov", "camera_cross_right_120fov"]}
    assert "cameras=4cam_1080" in report["units"][0]["wizard_prepare"]["argv"]


def test_schema_version_one_packets_are_rejected():
    request = packet_request()
    request["schema_version"] = 1
    assert "unsupported_schema" in codes(build_pilot_packet(request))


# --- Review A4-F5/F6 ------------------------------------------------------------------


def test_compose_expectations_name_the_renderer_mounts_the_argv_requests():
    from s2_launch_fixture import pinned_harmonizer_source

    report = build_pilot_packet(packet_request())
    assert report["units"][0]["compose_expectations"]["renderer_volumes"] == {
        "/mnt/ego-hoods": "/synthetic/data/nre-artifacts/ego-hoods"}
    request = packet_request()
    request["launch_request"]["harmonizer_source"] = pinned_harmonizer_source()
    report = build_pilot_packet(request)
    assert report["prepared"], report["issues"]
    for unit in report["units"]:
        assert unit["compose_expectations"]["renderer_volumes"] == {
            "/mnt/ego-hoods": "/synthetic/data/nre-artifacts/ego-hoods",
            "/mnt/harmonizer": "/synthetic/harmonizer"}
        assert "/synthetic/harmonizer:/mnt/harmonizer" in json.loads(
            override(unit, "services.renderer.volumes"))


def test_packet_report_carries_kind_scope_producer_and_unit_scopes():
    report = build_pilot_packet(packet_request())
    assert report["prepared"], report["issues"]
    assert report["artifact_kind"] == "s2_pilot_packet"
    assert report["scope"] == "synthetic"
    assert report["producer"]["module"] == "gate.s2_packet"
    assert report["producer"]["function"] == "build_pilot_packet"
    assert len(report["producer"]["source_sha256"]) == 64
    for unit in report["units"]:
        assert unit["generated_config_expectations"]["scope"] == "synthetic"
        assert unit["compose_expectations"]["scope"] == "synthetic"


@pytest.mark.parametrize("mutation", ["missing", "invalid", "launch_differs", "actual_packet",
                                      "actual_staging"])
def test_packet_scope_is_required_and_consistent(mutation):
    request = packet_request()
    if mutation == "missing":
        del request["scope"]
    elif mutation == "invalid":
        request["scope"] = "assumed"
    elif mutation == "launch_differs":
        request["launch_request"]["scope"] = "public_source_fixture"
    elif mutation == "actual_packet":
        request["scope"] = request["launch_request"]["scope"] = "actual"
    else:
        request["ego_hood_staging"]["scope"] = "actual"
    report = build_pilot_packet(request)
    assert not report["prepared"] and report["units"] == []
    expected = {"missing": "invalid_request", "invalid": "invalid_scope"}.get(
        mutation, "scope_mismatch")
    assert expected in codes(report)


# --- Review A2-F12: Harmonizer licence gate on on-arm units --------------------------


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("ref", [None, "", "TBD"])
def test_on_arm_units_require_recorded_harmonizer_terms(pinned, ref):
    from s2_launch_fixture import pinned_harmonizer_source

    request = packet_request()
    if pinned:
        request["launch_request"]["harmonizer_source"] = pinned_harmonizer_source()
    request["launch_request"]["harmonizer_terms_authorization_ref"] = ref
    report = build_pilot_packet(request)
    assert not report["prepared"] and report["units"] == []
    blocking = [(i["code"], i["path"]) for i in report["issues"] if i["severity"] == "blocking"]
    assert blocking == [("harmonizer_terms_unrecorded",
                         "launch_request.harmonizer_terms_authorization_ref")]


def test_terms_reference_leaves_off_and_on_unit_commands_unchanged():
    first = build_pilot_packet(packet_request())
    request = packet_request()
    request["launch_request"]["harmonizer_terms_authorization_ref"] = "SYNTHETIC-other-terms"
    second = build_pilot_packet(request)
    assert first["prepared"] and second["prepared"]
    for one, two in zip(first["units"], second["units"], strict=True):
        assert one["commands_sha256"] == two["commands_sha256"]
        assert one["launch_request_sha256"] != two["launch_request_sha256"]


# ------------------------------------------------------------ diagnostic qualification units

# Identity of the fixture's ten-job pilot packet, pinned before diagnostic support existed
# (computed on the unmodified builder, 2026-11-09). ``producer`` is excluded because it
# carries this module's own source hash, which every edit changes.
PILOT_PACKET_SHA256_WITHOUT_PRODUCER = (
    "bd5b0ed35014a4ef4c09f9b8613937233fe79aa325abdeb5e32c18645943a9e9")
PILOT_REQUEST_SHA256 = "56222db27747d70f88de76fb9183993d7a838efe8167126aab12345077283232"
PILOT_UNITS_SHA256 = "99b5e2da360f2c54b63ef66630de0ad71bf4828168e9c988fd1370655702dc8d"
PILOT_SCHEDULE_SHA256 = "fd8173944e65dba4a9eacdb1e821a569d991d8d14550c33840f546a843f56b98"


def without_producer(packet):
    return {key: value for key, value in packet.items() if key != "producer"}


def test_ten_job_packet_without_diagnostics_is_byte_identical_to_the_pin():
    packet = build_pilot_packet(packet_request())
    assert packet["prepared"] and SCHEMA_VERSION == 2
    assert identity_hash(without_producer(packet)) == PILOT_PACKET_SHA256_WITHOUT_PRODUCER
    assert packet["request_sha256"] == PILOT_REQUEST_SHA256
    assert identity_hash(packet["units"]) == PILOT_UNITS_SHA256
    assert identity_hash(packet["schedule"]) == PILOT_SCHEDULE_SHA256
    assert "diagnostic_units" not in packet and "diagnostics" not in packet
    assert "diagnostic_units" not in packet["input_hashes"]


def diagnostic_packet(**changes):
    request = packet_request()
    first = build_pilot_packet(packet_request())["units"][0]["job"]["job_id"]
    request["diagnostic_units"] = [diagnostic_unit(first, **changes)]
    return request, build_pilot_packet(request), first


def test_declared_diagnostic_unit_is_separate_from_the_ten_pilot_jobs():
    request, packet, source = diagnostic_packet()
    assert packet["prepared"], packet["issues"]
    # The ten pilot jobs, their units and the schedule are unchanged by the declaration.
    assert len(packet["units"]) == 10 and len(packet["schedule"]) == 10
    assert identity_hash(packet["units"]) == PILOT_UNITS_SHA256
    assert identity_hash(packet["schedule"]) == PILOT_SCHEDULE_SHA256
    assert packet["request_sha256"] == identity_hash(request) != PILOT_REQUEST_SHA256
    assert packet["input_hashes"]["diagnostic_units"] == identity_hash(
        request["diagnostic_units"])
    [unit] = packet["diagnostic_units"]
    assert unit["job"]["job_id"] == "diagnostic-camera-check"
    # Review G2-04: its own ledger phase, never the source pilot job's.
    assert unit["job"]["phase"] == "diagnostic_qualification"
    assert {job["phase"] for job in packet["schedule"]} == {"harmonizer_pilot"}
    assert unit["job"]["job_id"] not in {job["job_id"] for job in packet["schedule"]}
    assert unit["diagnostic"] == {
        "diagnostic_id": "camera-check", "role": "camera_config_qualification",
        "source_job_id": source,
        "budget_line": {"line_id": "pilot_diagnostics", "cap_usd": "5"},
        "counts_toward_pilot_jobs": False, "inference_input": False}
    assert packet["diagnostics"] == {
        "unit_count": 1, "budget_lines": {"pilot_diagnostics": "5"},
        "execution": "before_pilot_units_serially", "counts_toward_pilot_jobs": False,
        "inference_input": False, "ledger": "separate from the pilot schedule's attempt ledger"}
    pilot = packet["units"][0]
    assert pilot["job"]["job_id"] == source
    # Same scene, mode and horizon as its source job, in its own directories.
    for key in ("scene_id", "protocol_hash", "requested_steps", "road_id", "stratum"):
        assert unit["job"][key] == pilot["job"][key]
    assert unit["harmonizer_mode"] == pilot["harmonizer_mode"]
    assert unit["compose"]["cwd"] == pilot["compose"]["cwd"].rsplit("/", 1)[0] + (
        "/diagnostic-camera-check")
    assert unit["capture_dir"] == unit["compose"]["cwd"] + "/physloop-capture"
    assert unit["driver_output_path"].endswith(
        "/diagnostic-camera-check/physloop-driver-output.txt")
    assert override(unit, "scenes.scene_ids") == override(pilot, "scenes.scene_ids")
    assert json.loads(override(unit, "wizard.log_dir")) == unit["compose"]["cwd"]
    assert unit["commands_sha256"] != pilot["commands_sha256"]


@pytest.mark.parametrize(("mutate", "fragment"), [
    (lambda r, s: r.update(diagnostic_units=[]), "nonempty list"),
    (lambda r, s: r.update(diagnostic_units={"x": 1}), "nonempty list"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit("0" * 64)]), "scheduled pilot job"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(s), diagnostic_unit(s)]),
     "declared twice"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(s, role="")]), "no default role"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(s, role="TBD")]), "role"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(s, diagnostic_id="../x")]),
     "diagnostic_id"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(
        s, budget_line={"line_id": "pilot_diagnostics", "cap_usd": "0"})]), "positive"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(
        s, budget_line={"line_id": "pilot_diagnostics", "cap_usd": "0.00"})]), "positive"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(
        s, budget_line={"line_id": "pilot_diagnostics", "cap_usd": 5})]), "positive"),
    (lambda r, s: r.update(diagnostic_units=[diagnostic_unit(
        s, budget_line={"line_id": "pilot_diagnostics"})]), "line_id and cap_usd"),
    (lambda r, s: r.update(diagnostic_units=[
        diagnostic_unit(s),
        diagnostic_unit(s, diagnostic_id="other",
                        budget_line={"line_id": "pilot_diagnostics", "cap_usd": "6"})]),
     "two caps"),
    (lambda r, s: r.update(diagnostic_units=[{**diagnostic_unit(s), "counts": True}]),
     "requires exactly"),
])
def test_invalid_diagnostic_declarations_prepare_nothing(mutate, fragment):
    request = packet_request()
    source = build_pilot_packet(packet_request())["units"][0]["job"]["job_id"]
    mutate(request, source)
    packet = build_pilot_packet(request)
    assert not packet["prepared"] and packet["units"] == []
    assert "diagnostic_units" not in packet
    messages = [item["message"] for item in packet["issues"]
                if item["code"] == "invalid_diagnostic_units"]
    assert any(fragment in message for message in messages), packet["issues"]


def test_two_diagnostics_may_share_one_budget_line_and_keep_declared_order():
    request = packet_request()
    jobs = [unit["job"]["job_id"] for unit in build_pilot_packet(packet_request())["units"]]
    request["diagnostic_units"] = [diagnostic_unit(jobs[1], diagnostic_id="b-on"),
                                   diagnostic_unit(jobs[0], diagnostic_id="a-off")]
    packet = build_pilot_packet(request)
    assert packet["prepared"], packet["issues"]
    assert [unit["job"]["job_id"] for unit in packet["diagnostic_units"]] == [
        "diagnostic-b-on", "diagnostic-a-off"]
    assert [unit["index"] for unit in packet["diagnostic_units"]] == [0, 1]
    assert [unit["harmonizer_mode"] for unit in packet["diagnostic_units"]] == [
        packet["units"][1]["harmonizer_mode"], packet["units"][0]["harmonizer_mode"]]
    assert packet["diagnostics"]["budget_lines"] == {"pilot_diagnostics": "5"}
