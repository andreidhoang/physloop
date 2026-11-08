"""Exercise the bounded rehearsal without recursively running its full suite."""

import json

import pytest

from scripts.rehearse_s2_pilot import declaration_check, run_demo


def test_ten_job_rehearsal_has_literal_answers_and_no_runtime_authority(tmp_path):
    result = run_demo(tmp_path / "rehearsal")
    assert result["passed"]
    assert result["jobs"] == result["records"] == result["out_of_envelope"] == 10
    assert result["scores"] == [0.8] * 10
    assert result["deviations_m"] == [4.2] * 10
    assert result["negative_controls"] == {
        "missing_last_artifact": True, "export_corruption": True,
        "budget_boundary": True, "deadline_boundary": True,
    }
    assert result["export_verified"]
    assert not result["runtime_qualified"]
    assert not result["cloud_launch_authorized"]
    assert not result["scientific_draft_ready"]
    assert result["asl_capture"]["passed"]
    assert result["asl_capture"]["pose_events"] == 4
    assert result["asl_capture"]["actual_steps"] is None
    assert result["asl_capture"]["torn_frame_rejected"]
    assert result["launch_preparation"]["prepared"]
    assert result["launch_preparation"]["policy_rollouts"] == 10
    assert result["launch_preparation"]["missing_scenes_rejected"]
    assert result["launch_preparation"]["commands_executed"] is False
    assert result["pilot_packet"]["prepared"]
    assert result["pilot_packet"]["bound_job_count"] == 10
    assert result["pilot_packet"]["matches_ingestion_schedule"]
    assert result["pilot_packet"]["horizon_mismatch_rejected"]
    assert result["pilot_packet"]["missing_order_rejected"]
    assert result["pilot_packet"]["commands_executed"] is False
    saved = json.loads((tmp_path / "rehearsal/demo.json").read_text())
    assert saved == result


def test_rehearsal_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    assert list(tmp_path.iterdir()) == []


def environment_fixture():
    return {
        "pyproject.toml": b'''[project]
name = "physloop-gate"
requires-python = "==3.12.*"
dependencies = ["numpy==2.5.4"]
''',
        "uv.lock": b'''requires-python = "==3.12.*"
[[package]]
name = "physloop-gate"
dependencies = [{name = "numpy"}]
[package.metadata]
requires-dist = [{name = "numpy", specifier = "==2.5.4"}]
''',
    }


def test_environment_checks_declarations_not_just_installed_versions():
    snapshot = environment_fixture()
    assert declaration_check(snapshot)["matched"]
    assert not declaration_check(snapshot)["frozen_sync_verified"]
    snapshot["pyproject.toml"] = snapshot["pyproject.toml"].replace(b"2.5.4", b"2.5.5")
    result = declaration_check(snapshot)
    assert not result["matched"]
    assert "runtime_declarations_differ_from_lock" in result["issues"]


def test_unknown_requirement_or_python_constraint_blocks_environment():
    for before, after in ((b"numpy==2.5.4", b"numpy>=2"), (b"3.12", b"3.13")):
        snapshot = environment_fixture()
        snapshot["pyproject.toml"] = snapshot["pyproject.toml"].replace(before, after)
        assert not declaration_check(snapshot)["matched"]


def test_matching_dev_metadata_cannot_hide_missing_resolved_dependency():
    snapshot = environment_fixture()
    snapshot["pyproject.toml"] += b'\n[dependency-groups]\ndev = ["pytest"]\n'
    snapshot["uv.lock"] += b'\n[package.metadata.requires-dev]\ndev = [{name="pytest"}]\n'
    assert not declaration_check(snapshot)["matched"]
