"""Integrated synthetic S2 rehearsal (CPU-13; build-spec B2) with literal answers.

The rehearsal runs once per module with process creation and outbound sockets disabled:
any attempt to start a real tool (``vastai``, ``docker``, ``hf``, ``nvidia-smi``, ``uv``)
or to open a network connection raises inside the run and fails these tests.
"""

import hashlib
import json
import socket
import subprocess

import pytest

from gate.s2_bundle import verify_bundle
from scripts.rehearse_s2 import main, run_rehearsal

EXPECTED_CONTROLS = {
    "pilot_missing_order", "full_sweep_440_scenes", "setup_draft_inventory_blocked",
    "bootstrap_without_authorization", "lifecycle_unadmitted_record_refused",
    "unit_runs_authority_mismatch_refused", "capture_torn_frame",
    "ingestion_missing_last_artifact", "probe_different_inputs_not_comparable",
    "gate_unregistered_margin_blocks", "next_unit_equality_boundary", "next_unit_deadline",
    "readiness_real_pilot_draft_blocked", "readiness_closed_d019",
    "scientific_draft_not_ready", "bundle_secret", "bundle_changed",
    "unit_runs_ledger_budget_stop", "report_unbound_row_blocks_cpu_rehearsed",
}


def _refuse(*_args, **_kwargs):
    raise AssertionError("the synthetic rehearsal must not start a process or open a socket")


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    base = tmp_path_factory.mktemp("rehearsal")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "Popen", _refuse)
        patch.setattr(socket.socket, "connect", _refuse)
        patch.setattr(socket, "create_connection", _refuse)
        result = run_rehearsal(base / "out")
    return base, result


def test_every_check_and_negative_control_passes(rehearsal):
    _base, result = rehearsal
    assert result["failed_checks"] == [], {
        name: result["checks"][name] for name in result["failed_checks"]}
    assert result["unrejected_controls"] == []
    assert set(result["negative_controls"]) == EXPECTED_CONTROLS
    assert result["passed"] and result["status"] == "passed"


def test_literal_pipeline_answers(rehearsal):
    _base, result = rehearsal
    observed = {name: item["observed"] for name, item in result["checks"].items()}
    assert observed["pilot_packet_units"] == 10
    assert observed["full_sweep_units"] == 2646
    assert observed["full_sweep_distinct_ports_and_dirs"] == [2646, 2646]
    assert observed["guest_bootstrap_stops_at_first_unverified_output"] == [
        "failed", "A1_host_probe", "verification_failed"]
    assert observed["lifecycle_reaches_verified_absence"] == [True, "absent_verified"]
    assert observed["unit_runs_attempt_counts"] == [11, 8, 3]
    assert observed["unit_runs_ledger_not_admitted"] == [
        False, ["job_not_completed", "retry_selection_unregistered"]]
    assert observed["ingestion_admitted_records"] == [True, 10]
    assert observed["ingestion_scores_and_deviations"] == [[0.8], [4.2]]
    assert observed["validity_deviation_bins"] == {
        "[0,1)": 0, "[1,2)": 0, "[2,4)": 0, "[4,inf)": 10, "unknown": 0}
    assert observed["telemetry_useful_work"] == ["1", "8", "4"]
    assert observed["power_three_roads_insufficient"] == [20, 0, 36]
    assert observed["synthetic_gate_forty_equal_roads"] == ["passed", "GO", "synthetic"]
    assert observed["cost_deposit_and_usage"] == ["10", "3", "7"]
    assert observed["bundle_replayed_analyses"] == 12
    assert observed["unit_runs_budget_from_cost_ledger"] == ["pilot", 11]
    assert observed["unit_runs_clock_anchor_recorded"] == [
        ["anchor_ref", "clock_id", "monotonic_us", "uncertainty_us", "utc"], 0]
    assert observed["diagnostic_unit_separate_from_pilot_jobs"] == [
        "completed", 1, 0, [], True, 10]
    # Telemetry producers over the runner's own outputs: 7 useful, 3 failed, 1 awaiting
    # Huy's retry-selection rule (its unit time is unknown, so the summary blocks).
    assert observed["telemetry_useful_rows"] == [11, 7, 3, 1]
    assert observed["telemetry_dcgm_scrapes"] == ["passed", 2]
    assert observed["telemetry_summary_reports_retry_gap"] == ["blocked", True]
    assert result["counts"] == {"pilot_units": 10, "pilot_attempts": 11,
                                "full_sweep_units": 2646, "bundle_entries": 106,
                                "replayed_analyses": 12}


def test_milestones_never_claim_host_or_acceptance(rehearsal):
    _base, result = rehearsal
    assert result["milestones"] == {
        "CPU_REHEARSED": "passed", "PACKET_COMPLETE": "not_applicable",
        "HOST_QUALIFIED": "qualify_on_host", "ACTUAL_INPUT_QUALIFIED": "qualify_on_host",
        "S2_ACCEPTED": "not_run", "S2_STOPPED_OR_INSUFFICIENT": "not_run"}
    assert result["host_qualified"] is False and result["runtime_qualified"] is False
    assert result["cloud_launch_authorized"] is False
    assert result["real_tools_started"] is False


def test_outputs_stay_inside_the_fresh_directory_and_bundle_verifies(rehearsal):
    base, result = rehearsal
    assert sorted(path.name for path in base.iterdir()) == ["out"]
    out = base / "out"
    assert sorted(path.name for path in out.iterdir()) == [
        "bundle", "controls", "rehearsal.json", "replay", "report.json", "report.md", "work"]
    assert json.loads((out / "rehearsal.json").read_text()) == result
    for name in ("bundle", "replay"):
        report = verify_bundle(out / name)
        assert report["verified"] and report["manifest_sha256"] == result[
            "bundle_manifest_sha256"]
    report = json.loads((out / "report.json").read_text())
    assert list(report["items"]) == [f"S2.{index}" for index in range(1, 8)]
    assert report["scientific_verdict"] is None
    assert "| S2_ACCEPTED | not_run |" in (out / "report.md").read_text()


def test_second_invocation_refuses_and_changes_nothing(rehearsal):
    base, _result = rehearsal
    out = base / "out"
    before = sorted(str(path.relative_to(out)) for path in out.rglob("*"))
    with pytest.raises(FileExistsError):
        run_rehearsal(out)
    assert sorted(str(path.relative_to(out)) for path in out.rglob("*")) == before
    assert main(["--fixture", "complete", "--output", str(out)]) == 2


def test_unit_ledger_rejection_is_cited_as_blocked(rehearsal):
    """Review B2-03: the unadmitted attempt ledger is never shown as a passing row."""
    base, result = rehearsal
    report = json.loads((base / "out/report.json").read_text())
    item = report["items"]["S2.2"]
    row = next(row for row in item["evidence"] if row["key"] == "unit_runs")
    assert row["path"] == "reports/unit-run-summary.json"
    assert row["artifact_kind"] == "s2_unit_run_summary"
    assert row["status"] == "blocked"
    assert row["blockers"] == ["job_not_completed", "retry_selection_unregistered"]
    assert item["engineering_status"] == "blocked"
    assert "retry_selection_unregistered" in item["engineering_blockers"]
    summary = json.loads((base / "out/bundle/reports/unit-run-summary.json").read_text())
    assert summary["ledger"]["admitted"] is False
    ledger = json.loads((base / "out/bundle/outputs/unit_ledger.json").read_text())
    assert ledger == summary["ledger"]
    assert result["checks"]["report_unit_ledger_blocked"]["passed"]


def test_no_secret_canary_file_is_left_in_the_output_tree(rehearsal):
    """Review B2-08: synthetic guest secrets live outside the evidence tree."""
    base, _result = rehearsal
    out = base / "out"
    assert [path for path in out.rglob("*.secret")] == []
    for path in out.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert b"synthetic-hf-canary" not in data and b"synthetic-ngc-canary" not in data


def test_bundle_carries_the_cpu11_minimum_and_replays_against_its_source(rehearsal):
    """Review B2-10: the rehearsal bundle has every CPU-11 minimum role."""
    from gate.s2_bundle import ACTUAL_MINIMUM_ROLES

    base, result = rehearsal
    manifest = json.loads((base / "out/bundle/bundle-manifest.json").read_text())
    assert set(ACTUAL_MINIMUM_ROLES) <= {entry["role"] for entry in manifest["entries"]}
    assert result["checks"]["bundle_replay_source_identical"]["observed"][0] == "identical"
    source = json.loads((base / "out/bundle/source/source-manifest.json").read_text())
    paths = {item["path"] for item in source["files"]}
    assert "tests/test_s2_rehearsal.py" in paths
    assert any(path.startswith("tests/fixtures/") for path in paths)


def test_every_cited_row_is_bound_with_the_producer_its_artifact_names(rehearsal):
    """Review F4-02: producer identities come from the cited bytes; all rows are bound."""
    base, result = rehearsal
    out = base / "out"
    report = json.loads((out / "report.json").read_text())
    manifest = json.loads((out / "bundle/bundle-manifest.json").read_text())
    sealed = {(entry["path"], entry["sha256"]) for entry in manifest["entries"]}
    rows = [row for item in report["items"].values() for row in item["evidence"]
            if row["status"] != "missing"]
    assert rows and report["evidence_binding"]["unverified"] == 0
    for row in rows:
        assert row["binding"] == {"status": "bound", "reasons": []}, row["key"]
        assert (row["path"], row["sha256"]) in sealed
        data = (out / "bundle" / row["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        named = json.loads(data)["producer"]
        expected = {"module": named["module"], "source_sha256": named.get("source_sha256")}
        if "function" in named:
            expected["function"] = named["function"]
        assert row["producer"] == expected, row["key"]
    producers = {row["key"]: row["producer"]["module"] for row in rows}
    assert producers["unit_runs"] == "scripts.s2_run_units"
    assert producers["pilot_packet"] == "gate.s2_packet"
    assert producers["capture"] == producers["lifecycle"] == "scripts.rehearse_s2"
    assert result["checks"]["report_every_cited_row_bound"]["passed"]
    assert result["negative_controls"]["report_unbound_row_blocks_cpu_rehearsed"] is True
