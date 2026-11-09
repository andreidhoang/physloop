"""Known answers and negative controls for gate.s2_readiness (CPU-02; build-spec B2).

All plans, owner records, approvals and quotes are invented synthetic fixtures. The
real drafts in ``experiments/`` are read only to show they stay blocked.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from s2_packet_fixture import packet_request
from test_s2_execution import record as action_record
from test_s2_preflight import manifest as preflight_manifest
from test_s2_runtime import fixture_inventory

from gate.s2_identity import canonical_json_bytes, identity_hash, read_json_bytes
from gate.s2_packet import build_pilot_packet
from gate.s2_readiness import OWNER_DECISIONS, SECTIONS, inspect_readiness

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2027-01-15T07:30:00+00:00"


def write_evidence(root, entry_id, kind, payload, **fields):
    data = canonical_json_bytes(payload) + b"\n"
    path = root / "evidence" / f"{entry_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    entry = {"id": entry_id, "kind": kind, "path": f"evidence/{entry_id}.json",
             "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
             "scope": "synthetic", "phase": "harmonizer_pilot", "host_id": None,
             "decision_id": None, "subject": None, "subject_sha256": None,
             "valid_until_utc": None}
    entry.update(fields)
    return entry


def complete_case(root):
    """A fully linked synthetic case: every pre-rental section passes."""
    packet = build_pilot_packet(packet_request())
    assert packet["prepared"]
    entries = [
        write_evidence(root, "owner-1", "owner_record", {"synthetic": "owner record"}),
        write_evidence(root, "approval-1", "action_approval", {"synthetic": "approval"},
                       decision_id="D-999"),
        write_evidence(root, "quote-1", "quote", {"synthetic": "quote", "usd_per_hour": "2.50"},
                       valid_until_utc="2027-01-15T09:00:00+00:00"),
    ]
    plan = {
        "artifact_kind": "s2_readiness_plan", "schema_version": 1, "scope": "synthetic",
        "phase": "harmonizer_pilot", "as_of_utc": AS_OF, "host_id": None,
        "scientific_manifest": preflight_manifest(), "runtime_inventory": fixture_inventory(),
        "packet": packet,
        "unit_run_authority": {
            "artifact_kind": "s2_unit_run_authority", "schema_version": 1, "scope": "synthetic",
            "action_record_ref": "SYNTHETIC-D-999", "packet_request_sha256":
                packet["request_sha256"],
            "job_ids": [unit["job"]["job_id"] for unit in packet["units"]]},
        "owner_decisions": {name: {"value_ref": f"SYNTHETIC {name}", "evidence_id": "owner-1"}
                            for name in OWNER_DECISIONS["harmonizer_pilot"]},
    }
    index = {"artifact_kind": "s2_evidence_index", "schema_version": 1, "scope": "synthetic",
             "entries": entries}
    return plan, index, packet["schedule"], action_record()


def inspect(root, plan, index, schedule, record):
    return inspect_readiness(plan, index, schedule, record, root=root)


def codes(report, section=None):
    sections = [section] if section else SECTIONS
    return {item["code"] for name in sections for item in report["sections"][name]["issues"]}


def admission_keys(value, path=""):
    """Every truthy key that could read as admission or authorization."""
    found = []
    if type(value) is dict:
        for key, item in value.items():
            if any(word in key for word in ("admitted", "authorized", "authenticated",
                                            "ready_for_planning")) and item:
                found.append(f"{path}.{key}")
            found += admission_keys(item, f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            found += admission_keys(item, f"{path}[{index}]")
    return found


def test_complete_synthetic_case_passes_pre_rental_sections_only(tmp_path):
    report = inspect(tmp_path, *complete_case(tmp_path))
    assert report["status"] == "passed", report["next_blocking_checks"]
    statuses = {name: report["sections"][name]["status"] for name in SECTIONS}
    assert statuses == {"evidence_integrity": "passed", "planning_consistency": "passed",
                        "owner_decisions": "passed", "scientific_admission": "unverified",
                        "action_scope": "passed", "host_qualification": "qualify_on_host"}
    assert report["next_blocking_checks"] == []
    assert report["sections"]["host_qualification"]["blocks_pre_rental_status"] is False
    assert report["sections"]["evidence_integrity"]["authenticity"] == "not_verified"
    assert report["cloud_launch_authorized"] is False
    assert report["authorization_authenticated"] is False
    assert admission_keys(report) == []
    assert report["sections"]["planning_consistency"]["packet"]["unit_count"] == 10


def test_report_is_deterministic(tmp_path):
    case = complete_case(tmp_path)
    assert identity_hash(inspect(tmp_path, *case)) == identity_hash(inspect(tmp_path, *case))


def test_real_drafts_remain_blocked(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["scientific_manifest"] = read_json_bytes(
        (ROOT / "experiments/S2_PILOT_DRAFT.json").read_bytes())
    plan["runtime_inventory"] = read_json_bytes(
        (ROOT / "experiments/S2_SETUP_DRAFT.json").read_bytes())["runtime_inventory"]
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["status"] == "blocked"
    assert report["sections"]["scientific_admission"]["status"] == "blocked"
    assert {"not_preregistered", "owner_choice_unfrozen"} <= codes(report, "scientific_admission")
    assert "runtime_inventory_blocked" in codes(report, "planning_consistency")
    assert admission_keys(report) == []


def test_nonexistent_and_changed_references_block(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    (tmp_path / "evidence/quote-1.json").write_bytes(b'{"synthetic":"edited"}\n')
    index["entries"].append({**index["entries"][0], "id": "ghost", "path": "evidence/ghost.json"})
    report = inspect(tmp_path, plan, index, schedule, record)
    assert {"changed_bytes", "nonexistent_reference"} <= codes(report, "evidence_integrity")
    assert "quote_missing" in codes(report, "action_scope")
    rows = {row["id"]: row["bytes"] for row in report["sections"]["evidence_integrity"]["entries"]}
    assert rows["quote-1"] == "changed" and rows["ghost"] == "missing"


def test_traversal_reference_is_invalid(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    index["entries"][0]["path"] = "../outside.json"
    assert "invalid_reference" in codes(inspect(tmp_path, plan, index, schedule, record),
                                        "evidence_integrity")


def test_expired_quote_blocks(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["as_of_utc"] = "2027-01-15T09:00:01+00:00"
    report = inspect(tmp_path, plan, index, schedule, record)
    assert "quote_expired" in codes(report, "action_scope")
    check = next(item for item in report["next_blocking_checks"] if item["code"] == "quote_expired")
    assert check["resolver"] == "agent"


def test_closed_d019_is_never_new_authority(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    record["approval"]["decision_id"] = "D-019"
    plan["unit_run_authority"]["action_record_ref"] = "D-019 host check"
    report = inspect(tmp_path, plan, index, schedule, record)
    closed = [item for item in report["sections"]["action_scope"]["issues"]
              if item["code"] == "closed_authority_reused"]
    assert len(closed) == 2 and report["status"] == "blocked"


def test_populated_fields_without_actual_approval_are_declarations(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    index["entries"] = [entry for entry in index["entries"] if entry["kind"] != "action_approval"]
    plan["owner_decisions"]["camera_fix"]["evidence_id"] = None
    plan["owner_decisions"]["seed_mode"]["evidence_id"] = "quote-1"
    report = inspect(tmp_path, plan, index, schedule, record)
    assert "approval_not_linked" in codes(report, "action_scope")
    decisions = report["sections"]["owner_decisions"]["decisions"]
    assert decisions["camera_fix"] == "declared_unlinked"
    assert decisions["seed_mode"] == "owner_record_unverified"
    assert {"owner_decision_unlinked", "owner_record_unverified"} <= codes(report,
                                                                          "owner_decisions")
    owners = {item["code"] for item in report["next_blocking_checks"]
              if item["resolver"] == "owner"}
    assert {"approval_not_linked", "owner_decision_unlinked"} <= owners


def test_missing_owner_decision_has_no_default(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["owner_decisions"]["camera_fix"] = None
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["sections"]["owner_decisions"]["decisions"]["camera_fix"] == "missing"
    del plan["owner_decisions"]["camera_fix"]
    assert "invalid_owner_decisions" in codes(inspect(tmp_path, plan, index, schedule, record),
                                              "owner_decisions")


def test_synthetic_fixtures_never_satisfy_an_actual_action(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["scope"] = "actual"
    report = inspect(tmp_path, plan, index, schedule, record)
    found = codes(report)
    assert {"scope_mismatch", "synthetic_evidence_for_actual_plan",
            "synthetic_action_for_actual_plan", "synthetic_authority_for_actual_plan"} <= found
    assert report["status"] == "blocked"


def test_missing_action_record_and_authority_are_owner_actions(tmp_path):
    plan, index, schedule, _record = complete_case(tmp_path)
    plan["unit_run_authority"] = None
    report = inspect(tmp_path, plan, index, schedule, None)
    assert report["sections"]["action_scope"]["status"] == "not_run"
    assert {"action_record_missing", "unit_run_authority_missing"} <= codes(report,
                                                                           "action_scope")


def test_authority_for_another_packet_is_rejected(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["unit_run_authority"]["job_ids"] = list(reversed(plan["unit_run_authority"]["job_ids"]))
    plan["unit_run_authority"]["packet_request_sha256"] = "0" * 64
    report = inspect(tmp_path, plan, index, schedule, record)
    mismatched = {item["path"] for item in report["sections"]["action_scope"]["issues"]
                  if item["code"] == "authority_packet_mismatch"}
    assert mismatched == {"plan.unit_run_authority.job_ids",
                          "plan.unit_run_authority.packet_request_sha256"}


def test_edited_packet_and_schedule_are_detected(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["packet"]["units"][0]["driver"]["argv"].append("--extra")
    edited = copy.deepcopy(schedule)
    edited[0]["requested_steps"] += 1
    report = inspect(tmp_path, plan, index, edited, record)
    assert "packet_rebuild_mismatch" in codes(report, "planning_consistency")
    plan, index, schedule, record = complete_case(tmp_path / "second")
    report = inspect(tmp_path / "second", plan, index, edited, record)
    assert "schedule_mismatch" in codes(report, "planning_consistency")


def test_frozen_manifest_changed_after_review(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    reviewed = identity_hash(plan["scientific_manifest"])
    index["entries"].append(write_evidence(tmp_path, "review-1", "review_receipt", {"ok": 1},
                                           subject="scientific_manifest",
                                           subject_sha256=reviewed))
    assert inspect(tmp_path, plan, index, schedule, record)["status"] == "passed"
    plan["scientific_manifest"]["registration"]["source_ref"] = "SYNTHETIC changed reference"
    assert "reviewed_artifact_changed" in codes(inspect(tmp_path, plan, index, schedule, record),
                                                "planning_consistency")


def test_host_smoke_receipts_never_qualify_model_or_renderer(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    index["entries"].append(write_evidence(tmp_path, "smoke-old", "host_receipt", {"smoke": 1},
                                           host_id="vast-58628722"))
    report = inspect(tmp_path, plan, index, schedule, record)
    assert "host_receipt_scope_mismatch" in codes(report, "host_qualification")
    assert report["status"] == "passed"  # host evidence never blocks pre-rental status
    plan["host_id"] = "vast-1"
    index["entries"].append(write_evidence(tmp_path, "smoke-new", "host_receipt", {"smoke": 2},
                                           host_id="vast-1"))
    report = inspect(tmp_path, plan, index, schedule, record)
    host = report["sections"]["host_qualification"]
    assert "host_smoke_not_model_renderer_qualification" in codes(report, "host_qualification")
    assert host["status"] == "blocked"  # the old receipt still names another host
    assert host["receipts"] == ["smoke-new"]
    index["entries"] = [entry for entry in index["entries"] if entry["id"] != "smoke-old"]
    host = inspect(tmp_path, plan, index, schedule, record)["sections"]["host_qualification"]
    assert host["status"] == "qualify_on_host"


@pytest.mark.parametrize("change", [
    lambda plan: plan.pop("owner_decisions"),
    lambda plan: plan.update(schema_version=2),
    lambda plan: plan.update(phase="pilot"),
    lambda plan: plan.update(as_of_utc="2027-01-15T07:30:00"),
])
def test_invalid_plan_fails_closed(tmp_path, change):
    plan, index, schedule, record = complete_case(tmp_path)
    change(plan)
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["status"] == "blocked" and report["sections"] == {}
    assert report["issues"][0]["code"] == "invalid_plan"


def test_full_sweep_plan_requires_its_own_packet_kind(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["phase"] = "full_sweep"
    plan["owner_decisions"] = {name: None for name in OWNER_DECISIONS["full_sweep"]}
    report = inspect(tmp_path, plan, index, schedule, record)
    assert "packet_phase_mismatch" in codes(report, "planning_consistency")
    assert "schedule_phase_mismatch" in codes(report, "planning_consistency")
    assert len([item for item in report["sections"]["owner_decisions"]["issues"]
                if item["code"] == "owner_decision_missing"]) == 20


def test_evidence_index_is_json_serializable_and_hash_bound(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["input_hashes"]["evidence_index"] == identity_hash(index)
    json.dumps(report, allow_nan=False)


def test_full_sweep_packet_is_rebuilt_and_bound(tmp_path):
    from s2_packet_fixture import full_sweep_request

    from gate.s2_packet import build_full_sweep_packet

    plan, index, _schedule, record = complete_case(tmp_path)
    packet = build_full_sweep_packet(full_sweep_request())
    assert packet["prepared"]
    plan.update(phase="full_sweep", packet=packet,
                scientific_manifest=preflight_manifest(phase="full_sweep"),
                owner_decisions={name: {"value_ref": f"SYNTHETIC {name}",
                                        "evidence_id": "owner-1"}
                                 for name in OWNER_DECISIONS["full_sweep"]})
    plan["unit_run_authority"].update(
        packet_request_sha256=packet["request_sha256"],
        job_ids=[unit["job"]["job_id"] for unit in packet["units"]])
    for entry in index["entries"]:
        entry["phase"] = "full_sweep"
    report = inspect(tmp_path, plan, index, packet["schedule"], record)
    planning = report["sections"]["planning_consistency"]
    assert planning["status"] == "passed", planning["issues"]
    assert planning["packet"]["unit_count"] == 2646
    edited = json.loads(json.dumps(packet))
    edited["units"][2645]["compose"]["argv"].append("--detach")
    plan["packet"] = edited
    assert "packet_rebuild_mismatch" in codes(inspect(tmp_path, plan, index, packet["schedule"],
                                                      record), "planning_consistency")


# ---------------------------------------------------------------- review fixes (B2)


@pytest.mark.parametrize("ref", ["hf_" + "Q" * 34, "TBD", "  "])
def test_unrecorded_or_credential_shaped_authority_reference_blocks(tmp_path, ref):
    """Review B2-06: the same reference rule as scripts/s2_run_units.validate_authority."""
    plan, index, schedule, record = complete_case(tmp_path)
    plan["unit_run_authority"]["action_record_ref"] = ref
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["status"] == "blocked"
    assert "authority_unrecorded" in codes(report, "action_scope")
    assert "hf_" + "Q" * 34 not in json.dumps(report["sections"]["action_scope"]["issues"])


@pytest.mark.parametrize("ref", ["some-unrelated-action-record-7", "SYNTHETIC-D-9990",
                                 "SYNTHETIC-XD-999"])
def test_authority_reference_must_name_the_validated_action_record(tmp_path, ref):
    """Review B2-06: an authority naming another record does not bind this one."""
    plan, index, schedule, record = complete_case(tmp_path)
    plan["unit_run_authority"]["action_record_ref"] = ref
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["status"] == "blocked"
    assert "authority_action_record_mismatch" in codes(report, "action_scope")


def test_authority_reference_may_be_the_action_record_hash(tmp_path):
    plan, index, schedule, record = complete_case(tmp_path)
    plan["unit_run_authority"]["action_record_ref"] = identity_hash(record)
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["status"] == "passed", report["next_blocking_checks"]


@pytest.mark.parametrize("field, value", [("phase", "full_sweep"), ("scope", "actual")])
def test_approval_for_another_phase_or_scope_does_not_link(tmp_path, field, value):
    """Review B2-06: CPU-02 'receipt for another phase'."""
    plan, index, schedule, record = complete_case(tmp_path)
    for entry in index["entries"]:
        if entry["kind"] == "action_approval":
            entry[field] = value
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["status"] == "blocked"
    assert "approval_scope_mismatch" in codes(report, "action_scope")


@pytest.mark.parametrize("field, value", [("phase", "full_sweep"), ("scope", "actual")])
def test_owner_record_for_another_phase_or_scope_does_not_link(tmp_path, field, value):
    plan, index, schedule, record = complete_case(tmp_path)
    for entry in index["entries"]:
        if entry["kind"] == "owner_record":
            entry[field] = value
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["status"] == "blocked"
    assert "owner_record_scope_mismatch" in codes(report, "owner_decisions")
    assert set(report["sections"]["owner_decisions"]["decisions"].values()) == {
        "owner_record_scope_mismatch"}


def test_malformed_shapes_are_issues_not_exceptions(tmp_path):
    """Review B2-07: unhashable evidence IDs and null unit jobs never raise."""
    plan, index, schedule, record = complete_case(tmp_path)
    plan["owner_decisions"]["camera_fix"]["evidence_id"] = ["owner-1"]
    report = inspect(tmp_path, plan, index, schedule, record)
    assert report["sections"]["owner_decisions"]["decisions"]["camera_fix"] == (
        "owner_record_unverified")
    plan, index, schedule, record = complete_case(tmp_path / "second")
    plan["packet"] = copy.deepcopy(plan["packet"])
    plan["packet"]["units"][0]["job"] = None
    report = inspect(tmp_path / "second", plan, index, schedule, record)
    assert report["status"] == "blocked"
    assert "authority_packet_mismatch" in codes(report, "action_scope")
    plan, index, schedule, record = complete_case(tmp_path / "third")
    index["entries"][0]["id"] = ["owner-1"]
    report = inspect(tmp_path / "third", plan, index, schedule, record)
    assert "invalid_evidence_entry" in codes(report, "evidence_integrity")
