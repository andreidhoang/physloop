"""Literal controls for the complete synthetic ten-job pilot fixture."""

import json

import pytest
from s2_ingestion_fixture import json_receipt
from s2_pilot_fixture import build_pilot_fixture

from gate.s2_extract import extract_bundle
from gate.s2_schedule import build_schedule


def test_pilot_fixture_exercises_ten_jobs_and_both_protocol_batches(tmp_path):
    request = build_pilot_fixture(tmp_path)
    assert request == json.loads((tmp_path / "request.json").read_text())
    assert request["schedule"] == build_schedule(**request["schedule_request"])
    assert len(request["schedule"]) == 10
    assert {job["phase"] for job in request["schedule"]} == {"harmonizer_pilot"}
    assert len({job["scene_id"] for job in request["schedule"]}) == 5
    assert {job["replicate_id"] for job in request["schedule"]} == {0}
    assert {job["requested_steps"] for job in request["schedule"]} == {3}
    report = extract_bundle(tmp_path, request)
    assert report["admitted"], report["issues"]
    assert report["ledger"]["counts"]["scheduled_jobs"] == 10
    assert report["ledger"]["counts"]["attempts"] == 10
    assert report["ledger"]["counts"]["completed_attempts"] == 10
    assert len(report["batches"]) == 2
    assert [len(batch["normalization"]["records"]) for batch in report["batches"]] == [5, 5]
    records = [row for batch in report["batches"] for row in batch["normalization"]["records"]]
    assert [row["value"] for row in records] == [0.8] * 10
    assert [row["deviation_m"] for row in records] == [4.2] * 10
    assert sum(batch["normalization"]["counts"]["out_of_envelope"]
               for batch in report["batches"]) == 10
    assert len({observation["rollout_id"] for observation in report["observations"]}) == 10
    assert {observation["actual_steps"] for observation in report["observations"]} == {3}
    for name in ("runtime_qualified", "summary_recomputed", "cloud_launch_authorized"):
        assert report[name] is False


def test_missing_final_job_keeps_schedule_and_blocks_all_admission(tmp_path):
    request = build_pilot_fixture(tmp_path)
    events = json.loads((tmp_path / "events.json").read_text())
    request["events"] = json_receipt(tmp_path, "events.json", events[:-2])
    report = extract_bundle(tmp_path, request)
    assert not report["admitted"]
    assert report["ledger"]["counts"]["scheduled_jobs"] == 10
    assert report["ledger"]["counts"]["attempts"] == 9
    assert "missing_job" in {issue["code"] for issue in report["issues"]}
    assert all(not batch["normalization"]["records"] for batch in report["batches"])


def test_fixture_refuses_nonempty_directory_without_overwriting(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        build_pilot_fixture(tmp_path)
    assert marker.read_text() == "original"
    assert list(tmp_path.iterdir()) == [marker]


def test_fixture_refuses_symlink_root(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        build_pilot_fixture(linked)
    assert not list(actual.iterdir())
