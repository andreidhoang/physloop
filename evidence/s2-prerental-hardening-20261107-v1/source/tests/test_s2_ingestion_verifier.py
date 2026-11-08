"""Cheap verifier controls; the verifier's full check runner is never nested."""

import json

import pytest

from scripts.verify_s2_ingestion import run_demo


def test_demo_records_literal_positive_and_missing_metric_known_answers(tmp_path):
    output = tmp_path / "new-evidence"
    report = run_demo(output)
    assert report["passed"]
    assert report["positive"]["admitted"]
    assert report["positive"]["score"] == 0.8
    assert report["positive"]["deviation_m"] == 4.2
    assert report["positive"]["out_of_envelope"] == 1
    assert not report["missing_offroad"]["admitted"]
    assert "missing_raw_metric" in report["missing_offroad"]["issue_codes"]
    assert report["runtime_qualified"] is False
    assert all(not item["ready"] for item in report["drafts"].values())
    positive = json.loads((output / "synthetic" / "request.json").read_text())
    negative = json.loads((output / "missing-offroad" / "request.json").read_text())
    assert positive["artifacts"][0]["metrics"]["path"] == "metrics.parquet"
    assert negative["artifacts"][0]["metrics"]["path"] == "filtered/metrics.parquet"
    assert (output / "synthetic" / "result.json").is_file()
    assert (output / "missing-offroad" / "result.json").is_file()
    matrix = json.loads((output / "active_matrix.json").read_text())
    assert matrix["full_sweep"]["scheduled_policy_rollouts"] == 2646
    assert matrix["harmonizer_pilot"]["scheduled_policy_rollouts"] == 10


def test_demo_refuses_existing_output_before_writing(tmp_path):
    output = tmp_path / "existing-evidence"
    output.mkdir()
    marker = output / "immutable.txt"
    marker.write_text("keep this exact content")
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert marker.read_text() == "keep this exact content"
    assert list(output.iterdir()) == [marker]


def test_demo_refuses_existing_file_output(tmp_path):
    output = tmp_path / "existing-file"
    output.write_text("unchanged")
    with pytest.raises(FileExistsError):
        run_demo(output)
    assert output.read_text() == "unchanged"


def test_environment_check_requires_pyarrow_in_root_lock_and_exact_versions():
    from scripts.verify_s2_ingestion import check_locked_versions

    lock = {
        "package": [
            {"name": "physloop-gate", "dependencies": [{"name": "numpy"}]},
            {"name": "numpy", "version": "2.5.4"},
            {"name": "pyarrow", "version": "26.0.0"},
        ]
    }
    installed = {"numpy": "2.5.4", "pyarrow": "26.0.0"}
    assert not check_locked_versions(lock, installed)["matched"]
    lock["package"][0]["dependencies"].append({"name": "pyarrow"})
    assert check_locked_versions(lock, installed)["matched"]
    installed["pyarrow"] = "21.0.0"
    assert not check_locked_versions(lock, installed)["matched"]
