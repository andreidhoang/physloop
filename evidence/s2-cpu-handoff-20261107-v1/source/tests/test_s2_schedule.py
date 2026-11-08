"""Known-answer contracts for the active v5 schedule; no simulator execution."""

import re
from copy import deepcopy

import pytest

from gate.s2_schedule import active_matrix_report, build_schedule


def request(phase="full_sweep"):
    count = 441 if phase == "full_sweep" else 5
    return {
        "study_id": "synthetic-study",
        "phase": phase,
        "scenes": [
            {
                "scene_id": f"scene-{index:03d}",
                "road_id": f"road-{index // 3:03d}",
                "stratum": "synthetic-stratum",
                "asset_sha256": "a" * 64,
            }
            for index in range(count)
        ],
        "policies": {"A": "teacher", "B": "reexport"}
        if phase == "full_sweep"
        else {"A": "teacher"},
        "protocol_hashes": {"fixed": "b" * 64}
        if phase == "full_sweep"
        else {"off": "b" * 64, "on": "c" * 64},
        "replicates": 3 if phase == "full_sweep" else 1,
        "requested_steps": 200,
    }


def test_active_matrix_has_literal_v5_counts_without_measurement_or_authority():
    report = active_matrix_report()
    assert report["harmonizer_pilot"]["scheduled_policy_rollouts"] == 10
    assert report["full_sweep"]["scheduled_policy_rollouts"] == 2646
    assert report["full_sweep"]["arm_count"] == 2
    assert report["full_sweep"]["scene_count"] == 441
    assert report["independent_road_count"] is None
    assert report["runtime_measurement_status"] == "not_measured"
    assert report["counts_exclude_diagnostics_and_retries"] is True
    assert report["spend_authorized"] is False
    assert report["cloud_launch_authorized"] is False
    assert active_matrix_report(4)["full_sweep"]["scheduled_policy_rollouts"] == 3528


@pytest.mark.parametrize("replicates", [0, 1, 2, True, 3.0, "3", None])
def test_active_matrix_rejects_invalid_full_replicates(replicates):
    with pytest.raises(ValueError):
        active_matrix_report(replicates)


def test_full_schedule_exact_jobs_and_ledger_shape():
    inputs = request()
    before = deepcopy(inputs)
    rows = build_schedule(**inputs)
    assert inputs == before
    assert len(rows) == 2646
    assert len({row["job_id"] for row in rows}) == 2646
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["job_id"]) for row in rows)
    assert set(rows[0]) == {
        "job_id",
        "phase",
        "arm",
        "policy_id",
        "scene_id",
        "road_id",
        "stratum",
        "replicate_id",
        "protocol_hash",
        "requested_steps",
    }
    first_scene = [row for row in rows if row["scene_id"] == "scene-000"]
    assert {(row["arm"], row["replicate_id"]) for row in first_scene} == {
        ("A", 0),
        ("A", 1),
        ("A", 2),
        ("B", 0),
        ("B", 1),
        ("B", 2),
    }
    assert {row["road_id"] for row in first_scene} == {"road-000"}
    assert {row["protocol_hash"] for row in rows} == {"b" * 64}
    assert {row["requested_steps"] for row in rows} == {200}


def test_pilot_has_each_scene_in_both_harmonizer_modes_once():
    rows = build_schedule(**request("harmonizer_pilot"))
    assert len(rows) == 10
    assert {row["arm"] for row in rows} == {"A"}
    assert {row["replicate_id"] for row in rows} == {0}
    assert len({row["job_id"] for row in rows}) == 10
    for scene_id in {row["scene_id"] for row in rows}:
        assert {row["protocol_hash"] for row in rows if row["scene_id"] == scene_id} == {
            "b" * 64,
            "c" * 64,
        }


def test_input_order_does_not_change_schedule():
    inputs = request("harmonizer_pilot")
    expected = build_schedule(**inputs)
    inputs["scenes"].reverse()
    inputs["protocol_hashes"] = {"on": "c" * 64, "off": "b" * 64}
    assert build_schedule(**inputs) == expected
    inputs = request()
    expected = build_schedule(**inputs)
    inputs["scenes"].reverse()
    inputs["policies"] = {"B": "reexport", "A": "teacher"}
    assert build_schedule(**inputs) == expected


@pytest.mark.parametrize("changed", ["asset", "protocol", "study", "steps", "policy", "road"])
def test_job_identity_binds_asset_config_and_logical_assignment(changed):
    inputs = request("harmonizer_pilot")
    original = build_schedule(**inputs)[0]["job_id"]
    if changed == "asset":
        inputs["scenes"][0]["asset_sha256"] = "d" * 64
    elif changed == "protocol":
        inputs["protocol_hashes"]["off"] = "d" * 64
    elif changed == "study":
        inputs["study_id"] = "other-study"
    elif changed == "steps":
        inputs["requested_steps"] = 201
    elif changed == "policy":
        inputs["policies"]["A"] = "other-teacher"
    else:
        inputs["scenes"][0]["road_id"] = "other-road"
    assert build_schedule(**inputs)[0]["job_id"] != original


@pytest.mark.parametrize(
    "changed",
    [
        "phase",
        "study",
        "scenes_type",
        "scene_count",
        "scene_type",
        "scene_keys",
        "duplicate",
        "road_strata",
        "empty_scene",
        "asset",
        "policies_type",
        "arms",
        "policy",
        "same_policy",
        "protocols_type",
        "protocol_keys",
        "protocol_hash",
        "replicates",
        "steps",
    ],
)
def test_invalid_full_schedule_fails_before_returning_jobs(changed):
    inputs = request()
    if changed == "phase":
        inputs["phase"] = "training"
    elif changed == "study":
        inputs["study_id"] = " "
    elif changed == "scenes_type":
        inputs["scenes"] = None
    elif changed == "scene_count":
        inputs["scenes"].pop()
    elif changed == "scene_type":
        inputs["scenes"][0] = None
    elif changed == "scene_keys":
        inputs["scenes"][0]["unregistered_field"] = 0
    elif changed == "duplicate":
        inputs["scenes"][1] = deepcopy(inputs["scenes"][0])
    elif changed == "road_strata":
        inputs["scenes"][1]["stratum"] = "different-stratum"
    elif changed == "empty_scene":
        inputs["scenes"][0]["scene_id"] = ""
    elif changed == "asset":
        inputs["scenes"][0]["asset_sha256"] = "unresolved"
    elif changed == "policies_type":
        inputs["policies"] = []
    elif changed == "arms":
        del inputs["policies"]["B"]
    elif changed == "policy":
        inputs["policies"]["B"] = ""
    elif changed == "same_policy":
        inputs["policies"]["B"] = inputs["policies"]["A"]
    elif changed == "protocols_type":
        inputs["protocol_hashes"] = None
    elif changed == "protocol_keys":
        inputs["protocol_hashes"] = {"on": "b" * 64}
    elif changed == "protocol_hash":
        inputs["protocol_hashes"]["fixed"] = "b" * 63
    elif changed == "replicates":
        inputs["replicates"] = 2
    else:
        inputs["requested_steps"] = True
    with pytest.raises(ValueError):
        build_schedule(**inputs)


@pytest.mark.parametrize("changed", ["replicates", "scene_count", "arms", "same_hash"])
def test_pilot_contract_cannot_silently_change(changed):
    inputs = request("harmonizer_pilot")
    if changed == "replicates":
        inputs["replicates"] = 3
    elif changed == "scene_count":
        inputs["scenes"].pop()
    elif changed == "arms":
        inputs["policies"]["B"] = "reexport"
    else:
        inputs["protocol_hashes"]["on"] = inputs["protocol_hashes"]["off"].upper()
    with pytest.raises(ValueError):
        build_schedule(**inputs)
