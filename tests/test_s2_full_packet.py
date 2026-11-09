"""F1 known answers: the registered full sweep binds 2 x 441 x 3 = 2,646 serial units.

Inputs are the invented ``full_sweep_request`` fixture (synthetic scene list, order and
snapshots); no value here is a registered owner decision. The pilot packet keeps its
own tests in ``tests/test_s2_packet.py``.
"""

import copy
from collections import Counter

import pytest
from s2_packet_fixture import full_sweep_request, packet_request

from gate.s2_identity import identity_hash
from gate.s2_launch import MAX_DRIVER_PORT
from gate.s2_packet import build_full_sweep_packet, build_pilot_packet


@pytest.fixture(scope="module")
def prepared():
    request = full_sweep_request()
    return request, build_full_sweep_packet(request)


def codes(report):
    return {item["code"] for item in report["issues"]}


def test_full_sweep_has_literal_unit_count_and_registered_order(prepared):
    request, packet = prepared
    assert packet["prepared"], packet["issues"]
    assert len(packet["units"]) == 2 * 441 * 3 == 2646
    assert packet["sweep"] == {
        "scene_count": 441, "arm_count": 2, "replicates": 3, "unit_count": 2646,
        "harmonizer_mode": "off", "driver_port_range": [20000, 22645],
        "counts_exclude_diagnostics_and_retries": True}
    assert [unit["job"]["job_id"] for unit in packet["units"]] == (
        request["execution_order"]["job_ids"])
    assert [unit["index"] for unit in packet["units"]] == list(range(2646))
    assert Counter(unit["policy_arm"] for unit in packet["units"]) == {"A": 1323, "B": 1323}
    assert Counter(unit["job"]["replicate_id"] for unit in packet["units"]) == {
        0: 882, 1: 882, 2: 882}
    assert len({unit["job"]["scene_id"] for unit in packet["units"]}) == 441
    assert packet["commands_executed"] is False
    assert packet["cloud_launch_authorized"] is False


def test_every_unit_has_distinct_directories_and_ports(prepared):
    _request, packet = prepared
    units = packet["units"]
    ports = [unit["generated_config_expectations"]["driver_port"] for unit in units]
    assert ports == list(range(20000, 22646))
    for key in ("capture_dir", "driver_output_path"):
        assert len({unit[key] for unit in units}) == 2646
    assert len({unit["compose"]["cwd"] for unit in units}) == 2646
    first = units[0]
    job_id = first["job"]["job_id"]
    assert first["compose"]["cwd"] == f"/synthetic/runs/{first['policy_arm']}/{job_id}"
    assert "driver.port=20000" in first["wizard_prepare"]["argv"]
    assert "wizard.baseport=20001" in first["wizard_prepare"]["argv"]
    assert f"scenes.scene_ids=[\"{first['job']['scene_id']}\"]" in first["wizard_prepare"]["argv"]


def test_policy_snapshot_follows_the_unit_arm(prepared):
    _request, packet = prepared
    for unit in packet["units"][:6]:
        snapshot = {"A": "/synthetic/teacher/snapshot",
                    "B": "/synthetic/reexport/snapshot"}[unit["policy_arm"]]
        assert unit["generated_config_expectations"]["teacher_snapshot"] == snapshot
        assert f'driver.model.checkpoint_path="{snapshot}"' in unit["wizard_prepare"]["argv"]
        assert unit["harmonizer_mode"] == "off"
        assert unit["compose_expectations"]["harmonizer"] is False
        assert "--enable-harmonizer" not in " ".join(unit["wizard_prepare"]["argv"])


def test_a_different_registered_order_is_kept_verbatim():
    request = full_sweep_request()
    request["execution_order"]["job_ids"] = list(reversed(request["execution_order"]["job_ids"]))
    packet = build_full_sweep_packet(request)
    assert packet["prepared"]
    assert [unit["job"]["job_id"] for unit in packet["units"]] == (
        request["execution_order"]["job_ids"])


def test_full_sweep_is_deterministic(prepared):
    request, packet = prepared
    assert identity_hash(build_full_sweep_packet(copy.deepcopy(request))) == identity_hash(packet)


@pytest.mark.parametrize("change, code", [
    (lambda r: r["schedule_request"]["scenes"].pop(), "invalid_schedule_request"),
    (lambda r: r["schedule_request"].update(replicates=2), "invalid_schedule_request"),
    (lambda r: r["schedule_request"].pop("replicates"), "invalid_schedule_request"),
    (lambda r: r["schedule_request"].pop("scenes"), "invalid_schedule_request"),
    (lambda r: r["schedule_request"].update(phase="harmonizer_pilot"),
     "invalid_schedule_request"),
    (lambda r: r["execution_order"]["job_ids"].pop(), "invalid_execution_order"),
    (lambda r: r["execution_order"]["job_ids"].__setitem__(
        1, r["execution_order"]["job_ids"][0]), "invalid_execution_order"),
    (lambda r: r["execution_order"].update(job_ids=[]), "invalid_execution_order"),
    (lambda r: r["execution_order"].update(method="alternating"), "invalid_execution_order"),
    (lambda r: r.pop("execution_order"), "invalid_request"),
    (lambda r: r["launch"].update(harmonizer=None), "invalid_harmonizer_setting"),
    (lambda r: r["launch"].update(harmonizer=True,
                                  harmonizer_terms_authorization_ref="TBD"),
     "harmonizer_terms_unrecorded"),
    (lambda r: r["launch"].update(driver_port_base=MAX_DRIVER_PORT - 2645 + 1), "invalid_port"),
    (lambda r: r["launch"].update(requested_steps=4), "launch_schedule_mismatch"),
    (lambda r: r["launch"]["policies"]["B"].update(run_dir="/synthetic/runs/A/nested"),
     "overlapping_arm_outputs"),
    (lambda r: r["launch"]["policies"].pop("B"), "invalid_policies"),
    (lambda r: r["launch"].update(renderer_image="nvcr.io/nvidia/nre/nre-ga:latest"),
     "invalid_image"),
    (lambda r: r.update(scope="actual"), "scope_mismatch"),
    (lambda r: r.update(schema_version=2), "unsupported_schema"),
])
def test_owner_inputs_are_required_and_defects_block(change, code):
    request = full_sweep_request()
    change(request)
    packet = build_full_sweep_packet(request)
    assert not packet["prepared"] and packet["units"] == []
    assert code in codes(packet)


def test_largest_port_base_is_accepted():
    request = full_sweep_request()
    request["launch"]["driver_port_base"] = MAX_DRIVER_PORT - 2645
    packet = build_full_sweep_packet(request)
    assert packet["prepared"]
    assert packet["sweep"]["driver_port_range"] == [56890, 59535]


def test_harmonizer_on_sweep_carries_the_flag_in_every_unit():
    request = full_sweep_request()
    request["launch"]["harmonizer"] = True
    packet = build_full_sweep_packet(request)
    assert packet["prepared"]
    assert all(unit["harmonizer_mode"] == "on" for unit in packet["units"])
    renderer = next(arg for arg in packet["units"][0]["wizard_prepare"]["argv"]
                    if arg.startswith("services.renderer.command="))
    assert "--enable-harmonizer" in renderer


def test_identical_snapshots_warn_without_choosing_for_the_owner():
    request = full_sweep_request()
    request["launch"]["policies"]["B"]["snapshot"] = request["launch"]["policies"]["A"][
        "snapshot"]
    packet = build_full_sweep_packet(request)
    assert packet["prepared"]
    warnings = [item for item in packet["issues"] if item["code"] == "identical_policy_snapshots"]
    assert warnings and warnings[0]["severity"] == "warning"


def test_pilot_packet_is_unchanged_by_the_generalization():
    packet = build_pilot_packet(packet_request())
    assert packet["prepared"] and packet["artifact_kind"] == "s2_pilot_packet"
    assert len(packet["units"]) == 10
    assert all("policy_arm" not in unit for unit in packet["units"])
    assert {unit["generated_config_expectations"]["driver_port"] for unit in packet["units"]} == {
        6789}


# Independent pin of the ten-unit pilot units (review B2-11): the hash was computed from
# the pilot fixture before the full-sweep fixes, so any change to a pilot unit fails here.
PILOT_UNITS_SHA256 = "99b5e2da360f2c54b63ef66630de0ad71bf4828168e9c988fd1370655702dc8d"


def test_pilot_units_hash_is_pinned():
    packet = build_pilot_packet(packet_request())
    assert identity_hash(packet["units"]) == PILOT_UNITS_SHA256


def _with_scene_id(request, index, scene_id):
    from gate.s2_schedule import build_schedule

    request["schedule_request"]["scenes"][index]["scene_id"] = scene_id
    request["execution_order"]["job_ids"] = [
        job["job_id"] for job in build_schedule(**request["schedule_request"])]
    return request


@pytest.mark.parametrize("scene_id", [
    "synthetic-scene-000;touch_x", "synthetic scene 001", "${oc.env:HOME}", "$(id)",
    "a'b\"c", "..",
])
def test_unsafe_scene_ids_block_like_the_pilot(scene_id):
    """Review B2-01: the pinned wizard joins argv into shell commands."""
    packet = build_full_sweep_packet(_with_scene_id(full_sweep_request(), 0, scene_id))
    assert not packet["prepared"] and packet["units"] == []
    assert "invalid_scene_selection" in codes(packet)


@pytest.mark.parametrize("ref", ["hf_" + "Q" * 34, "TBD", "  "])
def test_off_sweep_terms_reference_is_still_validated(ref):
    """Review B2-09: a credential-shaped or placeholder reference never reaches inputs."""
    request = full_sweep_request()
    request["launch"]["harmonizer_terms_authorization_ref"] = ref
    packet = build_full_sweep_packet(request)
    assert not packet["prepared"] and packet["units"] == []
    assert "harmonizer_terms_unrecorded" in codes(packet)


def test_off_sweep_accepts_a_null_terms_reference():
    request = full_sweep_request()
    request["launch"]["harmonizer_terms_authorization_ref"] = None
    assert build_full_sweep_packet(request)["prepared"]


def test_sweep_warns_on_renderer_driver_camera_mismatch_like_the_pilot(prepared):
    _request, packet = prepared
    pilot = build_pilot_packet(packet_request())
    warning = "renderer_driver_camera_mismatch_predicted"
    for report in (pilot, packet):
        found = [item for item in report["issues"] if item["code"] == warning]
        assert found and found[0]["severity"] == "warning"
    request = full_sweep_request()
    request["launch"]["renderer_camera_override"] = "4cam_1080"
    assert warning not in codes(build_full_sweep_packet(request))
