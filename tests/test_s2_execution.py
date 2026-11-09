"""Known answers and negative controls for gate.s2_execution (CPU-10; build-spec A3).

No test invokes a real ``vastai`` or any network call: provider calls go through
``FakeProvider`` or a fake runner injected into ``VastCliClient``. Provider shapes
are copied from evidence/s2-vast-host-20261101-v1/ (order-result.json,
state-065200.json, stop-state-072043.json with the D-020 ``exited`` state,
post-delete-inventory.json) with only fields the module reads.
"""

import copy
import itertools
import json
import subprocess
from pathlib import Path

import pytest

from gate import s2_execution as ex

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "s2-vast-host-20261101-v1"
INTENT = "0f8e2c1a-5b7d-4c3e-9a10-2b3c4d5e6f70"
LABEL = "physloop-s2-0f8e2c1a-5b7d-4c3e-9a10-2b3c4d5e6f70"
WINDOW_START = 1_800_000_000  # 2027-01-15T08:00:00Z
WINDOW_END = WINDOW_START + 3600
ID = 58628722


def record(**overrides):
    value = {
        "artifact_kind": "s2_action_record",
        "schema_version": 1,
        "scope": "synthetic",
        "provider": "vast",
        "offer": {"offer_id": 56500949, "sku": "RTX PRO 6000 WS x1 KVM"},
        "region": "South Korea, KR",
        "resources": {
            "gpu_count": 1,
            "gpu_edition": "RTX PRO 6000 WS",
            "disk_gb": 160,
            "image": "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21",
            "volumes": [],
        },
        "max_hourly_usd": "2.50",
        "window": {"start_utc": "2027-01-15T08:00:00+00:00",
                   "end_utc": "2027-01-15T09:00:00+00:00"},
        "lifecycle_limits": {"startup_limit_seconds": 1200, "arm_limit_seconds": 120,
                             "export_attempt_limit": 2},
        "retention_after_stop": {"mode": "destroy_after_stop", "authority_ref": "D-999",
                                 "retain_until_utc": None},
        "export_destination": "evidence/s2-synthetic-export",
        "cap_headroom_snapshot": {
            "as_of_utc": "2027-01-15T07:00:00+00:00",
            "tier": "pilot",
            "caps_usd": {"pilot": "100", "stage": "1200", "project": "2500"},
            "used_usd": {"pilot": "0.6", "stage": "0.6", "project": "0.6"},
            "liability_usd": {"pilot": "0", "stage": "0", "project": "0"},
        },
        "approval": {"decision_id": "D-999",
                     "quoted_instruction": "Synthetic fixture instruction for tests only."},
        "license_attestation_refs": [],
        "scientific_registration_ref": None,
        "engineering_only": True,
    }
    for key, item in overrides.items():
        value[key] = item
    return value


def codes(report):
    return [issue["code"] for issue in report["issues"]]


# G1: an export_verified observation carries a report that re-verifies from its bytes.
EMPTY_JSON_SHA = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"  # b"{}"
RECEIPTS = [{"path": "attempts.json", "sha256": EMPTY_JSON_SHA, "size_bytes": 2}]
# sha256 of the literal canonical bytes
# [{"path":"attempts.json","sha256":"44136fa3...aff8a","size_bytes":2}]
RECEIPTS_HASH = "32148c41b40f074942ba4c7de1aab5a577b20589e556b955794b2b7e46f9d18f"


def envelope(receipts=None, **overrides):
    receipts = RECEIPTS if receipts is None else receipts
    value = {"artifact_kind": "s2_export_verification", "schema_version": 1,
             "producer": {"module": "gate.s2_execution", "function": "verify_export"},
             "receipts": receipts, "manifest_sha256": RECEIPTS_HASH,
             "result": {"verified": True, "file_count": len(receipts), "issues": [],
                        "cloud_cleanup_authorized": False},
             "status": "passed"}
    value.update(overrides)
    return value


def exported(**overrides):
    return {"kind": "export_verified", "manifest_hash": RECEIPTS_HASH,
            "verification": envelope(**overrides)}


def export_copy(tmp_path, data=b"{}"):
    """The local export copy ``RECEIPTS`` describes (G1-07: observe re-verifies on disk)."""
    root = tmp_path / "export"
    root.mkdir()
    (root / "attempts.json").write_bytes(data)
    return root


# ------------------------------------------------------------ action record (P0)


def test_complete_engineering_record_is_admitted_but_never_authenticated():
    report = ex.validate_action_record(record())
    assert report["admitted"] is True, report["issues"]
    assert report["status"] == "passed"
    assert report["authenticated"] is False
    assert report["artifact_kind"] == "s2_action_record_report"
    assert report["window_seconds"] == 3600
    assert report["input_hashes"]["record"] == ex.identity_hash(record())


@pytest.mark.parametrize(
    "key",
    ["provider", "offer", "region", "resources", "max_hourly_usd", "window",
     "lifecycle_limits", "retention_after_stop", "export_destination",
     "cap_headroom_snapshot", "approval", "license_attestation_refs",
     "scientific_registration_ref", "engineering_only"],
)
def test_every_field_is_required_without_defaults(key):
    value = record()
    del value[key]
    report = ex.validate_action_record(value)
    assert report["admitted"] is False
    assert report["status"] == "blocked"


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda r: r.update(provider="nebius"), "unsupported_provider"),
        (lambda r: r["offer"].update(offer_id=True), "invalid_field"),
        (lambda r: r["resources"].update(gpu_count=0), "invalid_field"),
        (lambda r: r["resources"].update(volumes=[{"size_gb": 10}]), "volumes_not_supported"),
        (lambda r: r.update(max_hourly_usd=2.5), "invalid_field"),
        (lambda r: r.update(max_hourly_usd="0"), "invalid_field"),
        (lambda r: r["window"].update(end_utc="2027-01-15T08:00:00+00:00"), "window_not_positive"),
        (lambda r: r["window"].update(end_utc="2027-01-15T20:00:01+00:00"),
         "window_exceeds_hard_bound"),
        (lambda r: r["window"].update(start_utc="2027-01-15T08:00:00"), "invalid_field"),
        (lambda r: r["approval"].update(quoted_instruction="TBD later"), "placeholder_value"),
        (lambda r: r["approval"].update(decision_id="approved"), "invalid_field"),
        (lambda r: r.update(engineering_only=False), "registration_or_engineering_only"),
        (lambda r: r.update(scientific_registration_ref="experiments/x.md"),
         "registration_or_engineering_only"),
        (lambda r: r["cap_headroom_snapshot"]["caps_usd"].update(pilot="1300"),
         "caps_not_nested"),
        (lambda r: r["cap_headroom_snapshot"]["used_usd"].update(pilot="5"),
         "budget_usage_not_nested"),
        (lambda r: r.update(max_hourly_usd="99.5"), "window_exposure_exceeds_headroom"),
        (lambda r: r["retention_after_stop"].update(mode="retain_until"), "invalid_field"),
        (lambda r: r["retention_after_stop"].update(authority_ref=""), "invalid_field"),
        (lambda r: r.update(unexpected=True), "unknown_field"),
        (lambda r: r.update(schema_version=2), "unknown_schema"),
        (lambda r: r.update(scope="production"), "invalid_field"),
    ],
)
def test_action_record_negative_controls(mutate, code):
    value = record()
    mutate(value)
    report = ex.validate_action_record(value)
    assert report["admitted"] is False
    assert code in codes(report), report["issues"]


def test_window_exposure_boundary_is_strict():
    # Pilot headroom = 100 - 0.6 - 0 = 99.4; exposure = rate x 1 h.
    assert ex.validate_action_record(record(max_hourly_usd="99.39"))["admitted"] is True
    assert ex.validate_action_record(record(max_hourly_usd="99.4"))["admitted"] is False


def test_twelve_hour_window_is_the_agent_hard_bound():
    value = record()
    value["window"]["end_utc"] = "2027-01-15T20:00:00+00:00"
    value["max_hourly_usd"] = "1"
    assert ex.validate_action_record(value)["admitted"] is True
    assert ex.MAX_WINDOW_SECONDS == 12 * 3600


def test_scientific_record_needs_licence_refs_registration_and_not_engineering_only():
    value = record(engineering_only=False, scientific_registration_ref="experiments/p.md")
    assert "license_attestation_required" in codes(ex.validate_action_record(value))
    value["license_attestation_refs"] = ["docs/research/licence-receipt.md"]
    assert ex.validate_action_record(value)["admitted"] is True


def test_retain_until_must_cover_the_window():
    value = record()
    value["retention_after_stop"] = {"mode": "retain_until", "authority_ref": "D-999",
                                     "retain_until_utc": "2027-01-15T08:30:00+00:00"}
    assert "retention_before_window_end" in codes(ex.validate_action_record(value))
    value["retention_after_stop"]["retain_until_utc"] = "2027-01-22T09:00:00+00:00"
    assert ex.validate_action_record(value)["admitted"] is True


def test_non_dict_record_is_blocked():
    assert ex.validate_action_record([1])["admitted"] is False


# ------------------------------------------------------------ state machine core


def fresh(**record_overrides):
    rec = record(**record_overrides)
    return ex.initial_state(rec, ex.validate_action_record(rec), INTENT,
                            watchdog_events="evidence/watchdog-events.jsonl")


def row(status="running", **overrides):
    # Shape from state-065200.json (fields the module reads; identity kept).
    value = {
        "id": ID, "label": LABEL, "actual_status": status, "intended_status": "running",
        "next_state": "running", "gpu_name": "RTX PRO 6000 WS", "num_gpus": 1,
        "disk_space": 160.0, "dph_total": 1.451111111111111,
        "image_uuid": "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21",
    }
    value.update(overrides)
    return value


def exited_row():
    # stop-state-072043.json: D-020 real terminal shape with a stale status_msg.
    shape = json.loads((EVIDENCE / "stop-state-072043.json").read_text())["instance"]
    return {**row(), **{k: shape[k] for k in ("actual_status", "intended_status",
                                             "next_state")}, "label": LABEL}


NO_LABEL = {"kind": "inventory", "status": "ok", "rows": []}
CLEAN_INVENTORY = {"status": "ok", "rows": []}


def drive(state, observation, now):
    out = ex.next_action(state, observation, now)
    return out["state"], out["action"], out


def to_state(target, now=WINDOW_START + 10):
    """Walk the normal path to a target state with literal observations."""
    state = fresh()
    path = [
        ("admitted", None),
        ("create_intent", NO_LABEL),
        ("create_requested", {"kind": "create_result", "outcome": "created",
                              "instance_id": ID}),
        ("bound", {"kind": "watchdog_receipt", "armed": True, "instance_id": ID,
                   "label": LABEL, "deadline_epoch": WINDOW_END}),
        ("armed", {"kind": "instance_state", "results": {str(ID): {"status": "row",
                                                                    "row": row()}}}),
        ("bootstrapping", {"kind": "bootstrap_complete"}),
        ("running", {"kind": "work_complete"}),
        ("exporting", exported()),
        ("export_verified", None),
        ("stopping", {"kind": "instance_state", "results": {str(ID): {"status": "row",
                                                                       "row": exited_row()}}}),
        ("stopped", None),
    ]
    for name, observation in path:
        if state["state"] == target:
            return state
        assert state["state"] == name
        state, _, _ = drive(state, observation, now)
    assert state["state"] == target
    return state


def test_normal_path_actions_and_terminal_reconciliation():
    now = WINDOW_START + 10
    state = fresh()
    assert state["label"] == LABEL
    state, action, _ = drive(state, None, now)
    assert (state["state"], action) == ("create_intent", {"action": "check_label_absent",
                                                          "label": LABEL})
    state, action, _ = drive(state, None, now)  # no inventory yet: never create
    assert (state["state"], action["action"]) == ("create_intent", "check_label_absent")
    state, action, _ = drive(state, {"kind": "inventory", "status": "error",
                                     "error": "inventory_failed"}, now)
    assert (state["state"], action["action"]) == ("create_intent", "check_label_absent")
    state, action, _ = drive(state, NO_LABEL, now)
    assert state["state"] == "create_requested"
    assert action == {"action": "create", "offer_id": 56500949, "label": LABEL,
                      "image": "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21",
                      "disk_gb": 160}
    assert state["create_requested_epoch"] == now
    assert state["cutoff_epoch"] == WINDOW_END
    state, action, _ = drive(state, {"kind": "create_result", "outcome": "created",
                                     "instance_id": ID}, now)
    assert state["state"] == "bound"
    assert action["action"] == "arm_watchdog"
    assert action["deadline_epoch"] == WINDOW_END
    state = to_state("stopped")
    state, action, _ = drive(state, None, now)
    assert state["state"] == "destroying"
    assert action == {"action": "destroy", "instance_ids": [ID], "volume_ids": [],
                      "verify_instance_ids": [ID]}
    notfound = {"kind": "absence_check",
                "instances": {str(ID): {"status": "not_found"}},
                "volumes": {"status": "ok", "ids": [], "label_matches": 0},
                "inventory": CLEAN_INVENTORY}
    state, action, _ = drive(state, notfound, now)
    assert state["state"] == "destroying"  # one NotFound is not absence
    assert action["instance_ids"] == []
    state, action, _ = drive(state, notfound, now)
    assert state["state"] == "absent_verified"
    assert action["action"] == "reconcile_cost"
    state, action, _ = drive(state, {"kind": "cost_reconciled", "ledger_status": "passed",
                                     "ledger_hash": "b" * 64,
                                     "settlement_status": "pending"}, now)
    assert state["state"] == "reconciled"
    assert action["action"] == "none"
    assert state["settlement_status"] == "pending"


def test_arm_watchdog_argv_is_exact():
    state = to_state("create_requested")
    out = ex.next_action(state, {"kind": "create_result", "outcome": "created",
                                 "instance_id": ID}, WINDOW_START + 10)
    assert out["action"]["argv"] == [
        "scripts/vast_stop_watchdog.py", "--instance-id", str(ID), "--expected-label", LABEL,
        "--deadline-utc", "2027-01-15T09:00:00+00:00", "--max-window-seconds", "3600",
        "--events", "evidence/watchdog-events.jsonl", "--first-arm",
    ]
    later = ex.next_action(out["state"], None, WINDOW_START + 20)
    assert later["action"] == {"action": "await_watchdog",
                               "events": "evidence/watchdog-events.jsonl"}


def test_create_is_refused_before_window_and_aborted_after():
    state = fresh()
    state, action, _ = drive(state, None, WINDOW_START - 1)
    assert (state["state"], action["action"]) == ("admitted", "wait")
    state, action, _ = drive(state, None, WINDOW_END)
    assert state["state"] == "aborted"
    assert "window_closed_before_create" in state["issues"]


def test_uncertain_create_never_creates_again_and_reconciles_by_label():
    now = WINDOW_START + 10
    state = to_state("create_requested")
    state, action, _ = drive(state, {"kind": "create_result", "outcome": "uncertain",
                                     "error": "cli_timeout"}, now)
    assert state["state"] == "uncertain_create"
    assert action["action"] == "reconcile_by_label" and action["label"] == LABEL
    empty = {"kind": "inventory", "status": "ok", "rows": []}
    for _ in range(5):
        state, action, _ = drive(state, empty, now)
        assert state["state"] == "uncertain_create"
        assert action["action"] == "reconcile_by_label"
    failed = {"kind": "inventory", "status": "error", "error": "cli_non_json"}
    state, action, _ = drive(state, failed, now)
    assert action["action"] == "reconcile_by_label"
    foreign = {"kind": "inventory", "status": "ok",
               "rows": [{"id": 1, "label": "someone-else"}, {"id": ID, "label": LABEL}]}
    state, action, _ = drive(state, foreign, now)
    assert state["state"] == "bound" and state["instance_ids"] == [ID]
    assert action["action"] == "arm_watchdog"


def test_uncertain_create_unresolved_after_startup_limit_escalates_without_create():
    state = to_state("create_requested")
    state, _, _ = drive(state, {"kind": "create_result", "outcome": "uncertain",
                                "error": "x"}, WINDOW_START + 10)
    state, action, _ = drive(state, {"kind": "inventory", "status": "ok", "rows": []},
                             WINDOW_START + 10 + 1200)
    assert state["state"] == "escalated"
    assert "create_unresolved" in state["issues"]
    assert action["action"] == "manual_review"


def test_duplicate_label_matches_stop_and_destroy_all_and_report():
    state = to_state("create_requested")
    state, _, _ = drive(state, None, WINDOW_START + 10)  # restart: no result recorded
    assert state["state"] == "uncertain_create"
    rows = [{"id": ID, "label": LABEL}, {"id": ID + 1, "label": LABEL},
            {"id": 7, "label": "foreign"}]
    state, action, _ = drive(state, {"kind": "inventory", "status": "ok", "rows": rows},
                             WINDOW_START + 10)
    assert state["state"] == "stopping"
    assert "duplicate_creation" in state["issues"]
    assert action == {"action": "stop", "instance_ids": [ID, ID + 1]}


def test_no_bootstrap_without_arming():
    state = to_state("bound")
    state, action, out = drive(state, {"kind": "bootstrap_complete"}, WINDOW_START + 10)
    assert state["state"] == "bound"
    assert action == {"action": "wait"}
    assert "unexpected_observation" in [i["code"] for i in out["issues"]]
    assert not ex.may_bootstrap(state)
    assert ex.may_bootstrap(to_state("bootstrapping"))


@pytest.mark.parametrize(
    "receipt",
    [
        {"kind": "watchdog_receipt", "armed": False},
        {"kind": "watchdog_receipt", "armed": True, "instance_id": ID + 1, "label": LABEL,
         "deadline_epoch": WINDOW_END},
        {"kind": "watchdog_receipt", "armed": True, "instance_id": ID, "label": "other",
         "deadline_epoch": WINDOW_END},
        {"kind": "watchdog_receipt", "armed": True, "instance_id": ID, "label": LABEL,
         "deadline_epoch": WINDOW_END + 1},
    ],
)
def test_arm_failure_or_mismatch_stops(receipt):
    state = to_state("bound")
    state, action, _ = drive(state, receipt, WINDOW_START + 10)
    assert state["state"] == "arm_failed"
    state, action, _ = drive(state, None, WINDOW_START + 10)
    assert state["state"] == "stopping"
    assert action == {"action": "stop", "instance_ids": [ID]}


def test_missing_arming_receipt_times_out_to_arm_failed():
    state = to_state("bound")
    state, action, _ = drive(state, None, WINDOW_START + 10 + 119)
    assert action["action"] == "await_watchdog"
    state, action, _ = drive(state, None, WINDOW_START + 10 + 120)
    assert state["state"] == "arm_failed"


@pytest.mark.parametrize(
    "field,value",
    [("gpu_name", "RTX 4090"), ("num_gpus", 2), ("disk_space", 100.0), ("dph_total", 2.51),
     ("image_uuid", "docker.io/other:latest"), ("label", "foreign")],
)
def test_wrong_resource_stops_before_bootstrap(field, value):
    state = to_state("armed")
    bad = {"kind": "instance_state",
           "results": {str(ID): {"status": "row", "row": row(**{field: value})}}}
    state, action, _ = drive(state, bad, WINDOW_START + 10)
    assert state["state"] == "stopping"
    assert "wrong_resource" in state["issues"]
    assert action["action"] == "stop"


def test_unready_or_incomplete_row_waits_until_startup_limit():
    state = to_state("armed")
    loading = {"kind": "instance_state",
               "results": {str(ID): {"status": "row", "row": row(status="loading")}}}
    state, action, _ = drive(state, loading, WINDOW_START + 10)
    assert (state["state"], action["action"]) == ("armed", "observe_instance")
    partial = row()
    del partial["gpu_name"]
    state, action, _ = drive(
        state, {"kind": "instance_state",
                "results": {str(ID): {"status": "row", "row": partial}}}, WINDOW_START + 20)
    assert state["state"] == "armed"
    state, action, _ = drive(state, loading, WINDOW_START + 10 + 1200)
    assert state["state"] == "stopping"
    assert "startup_limit_exceeded" in state["issues"]


def test_budget_exhaustion_moves_running_to_export():
    state = to_state("running")
    state, action, _ = drive(state, {"kind": "budget_exhausted",
                                     "decision_issue_codes": ["pilot_headroom_exhausted"]},
                             WINDOW_START + 100)
    assert state["state"] == "exporting"
    assert "budget_exhausted" in state["issues"]
    assert action["action"] == "export"


# --- A3X: failed work is its own cause, never journaled as budget exhaustion --------------

WORK_FAILED = {"kind": "work_failed", "reason": "s4_watchdog:role_dead:rollout_0",
               "decision_issue_codes": ["s4_watchdog:role_dead:rollout_0",
                                        "s4_cleanup:verified_clean"]}
BUDGET = {"kind": "budget_exhausted", "decision_issue_codes": ["pilot_headroom_exhausted"]}


def _evidence_path(state, export):
    """Drive from the cause through export, stop, destroy/retain, absence and reconcile."""
    now, trace = WINDOW_START + 100, []
    failed = {"kind": "export_failed", "error": "hash_mismatch"}
    notfound = {"kind": "absence_check", "instances": {str(ID): {"status": "not_found"}},
                "volumes": {"status": "ok", "ids": [], "label_matches": 0},
                "inventory": CLEAN_INVENTORY}
    steps = ([exported(), None] if export == "verified" else [failed, failed]) + [
        {"kind": "instance_state", "results": {str(ID): {"status": "row", "row": exited_row()}}},
        None, notfound, notfound,
        {"kind": "cost_reconciled", "ledger_status": "passed", "ledger_hash": "b" * 64,
         "settlement_status": "pending"}]
    for observation in steps:
        state, action, _ = drive(state, observation, now)
        trace.append((state["state"], action))
    return state, trace


@pytest.mark.parametrize("origin", ["bootstrapping", "running"])
@pytest.mark.parametrize("export", ["verified", "failed"])
def test_a3x_work_failed_preserves_evidence_exactly_like_budget_exhaustion(origin, export):
    failed_state, action, _ = drive(to_state(origin), WORK_FAILED, WINDOW_START + 100)
    assert (failed_state["state"], action) == ("exporting", {"action": "export"})
    assert "work_failed" in failed_state["issues"]
    assert "budget_exhausted" not in failed_state["issues"]  # the cause is not misstated
    budget_state, _, _ = drive(to_state(origin), BUDGET, WINDOW_START + 100)
    failed_end, failed_trace = _evidence_path(failed_state, export)
    budget_end, budget_trace = _evidence_path(budget_state, export)
    assert failed_trace == budget_trace
    if export == "verified":
        assert failed_trace[-1] == ("reconciled", {"action": "none"})
        assert [s for s, _ in failed_trace].count("destroying") >= 1
    else:
        # G1: no verified export -> the stopped instance is never destroyed.
        assert [s for s, _ in failed_trace][2:4] == ["stopped", "escalated"]
        assert failed_trace[3][1] == {"action": "manual_review", "label": LABEL}
        assert "destroying" not in [s for s, _ in failed_trace]
        assert "destroy_refused_export_unverified" in failed_end["issues"]
    assert {k: v for k, v in failed_end.items() if k != "issues"} == {
        k: v for k, v in budget_end.items() if k != "issues"}
    assert [i if i != "work_failed" else "budget_exhausted" for i in failed_end["issues"]] == (
        budget_end["issues"])


@pytest.mark.parametrize("bad", [
    {"kind": "work_failed", "decision_issue_codes": []},  # reason is required, no default
    {"kind": "work_failed", "reason": "", "decision_issue_codes": []},
    {"kind": "work_failed", "reason": None, "decision_issue_codes": []},
    {"kind": "work_failed", "reason": "Traceback (most recent call last)",
     "decision_issue_codes": []},  # free text is not a static cause code
    {"kind": "work_failed", "reason": "x" * 129, "decision_issue_codes": []},
    {"kind": "work_failed", "reason": "host_exited:1"},
    {"kind": "work_failed", "reason": "host_exited:1", "decision_issue_codes": [1]},
    {**WORK_FAILED, "extra": 1},
])
def test_a3x_work_failed_requires_a_static_reason(bad):
    with pytest.raises(ValueError):
        ex.validate_observation(bad)
    with pytest.raises(ValueError):
        ex.next_action(to_state("running"), bad, WINDOW_START + 100)


def test_a3x_work_failed_is_an_external_observation_only_while_work_can_run():
    assert "work_failed" in ex.EXTERNAL_OBSERVATIONS
    assert {name for name in ex.STATES if "work_failed" in ex.ALLOWED_OBSERVATIONS[name]} == {
        "bootstrapping", "running"}


def test_failed_export_retries_to_limit_then_stops_and_retains_nothing_without_authority():
    state = to_state("exporting")
    state, action, _ = drive(state, {"kind": "export_failed", "error": "hash_mismatch"},
                             WINDOW_START + 100)
    assert (state["state"], action["action"]) == ("exporting", "export")
    state, action, _ = drive(state, {"kind": "export_failed", "error": "hash_mismatch"},
                             WINDOW_START + 110)
    assert state["state"] == "stopping"
    assert "export_incomplete" in state["issues"]
    state, _, _ = drive(state, {"kind": "instance_state",
                                "results": {str(ID): {"status": "row", "row": exited_row()}}},
                        WINDOW_START + 120)
    assert state["state"] == "stopped"
    state, action, _ = drive(state, None, WINDOW_START + 130)
    # G1 (changed from the A3 behaviour, which destroyed here and only flagged
    # evidence_loss_risk): an armed instance without a verified export is never destroyed;
    # it stays stopped and the lifecycle escalates to manual review.
    assert (state["state"], action) == ("escalated", {"action": "manual_review",
                                                      "label": LABEL})
    assert {"evidence_loss_risk", "destroy_refused_export_unverified"} <= set(state["issues"])
    state, action, _ = drive(state, None, WINDOW_START + 140)
    assert (state["state"], action["action"]) == ("escalated", "manual_review")


def test_export_incomplete_with_retention_authority_holds_then_never_destroys():
    rec = record()
    rec["retention_after_stop"] = {"mode": "retain_until", "authority_ref": "D-999",
                                   "retain_until_utc": "2027-01-15T10:00:00+00:00"}
    state = ex.initial_state(rec, ex.validate_action_record(rec), INTENT,
                             watchdog_events="w.jsonl")
    state = dict(to_state("exporting"), plan=state["plan"])
    state, _, _ = drive(state, None, WINDOW_END)  # cutoff with export incomplete
    assert state["state"] == "cutoff_reached"
    state, action, _ = drive(state, None, WINDOW_END)
    assert action["action"] == "stop"
    state, _, _ = drive(state, {"kind": "instance_state",
                                "results": {str(ID): {"status": "row", "row": exited_row()}}},
                        WINDOW_END)
    state, action, _ = drive(state, None, WINDOW_END + 10)
    assert (state["state"], action["action"]) == ("stopped", "hold_for_retention")
    state, action, _ = drive(state, None, WINDOW_END + 3600)
    # G1: the retention authority lapsed without a verified export: stop-only, escalate.
    assert (state["state"], action["action"]) == ("escalated", "manual_review")
    assert "destroy_refused_export_unverified" in state["issues"]


@pytest.mark.parametrize("name", ["bound", "armed", "bootstrapping", "running", "exporting",
                                  "export_verified"])
def test_cutoff_stops_regardless_of_export(name):
    state = to_state(name)
    state, action, _ = drive(state, None, WINDOW_END)
    assert state["state"] == "cutoff_reached"
    assert "cutoff_reached" in state["issues"]
    state, action, _ = drive(state, None, WINDOW_END + 1)
    assert action == {"action": "stop", "instance_ids": [ID]}


def test_stop_requires_structured_terminal_state_and_never_trusts_errors():
    state = to_state("stopping")
    for result in ({"status": "error", "error": "cli_non_json"},
                   {"status": "row", "row": row(status="stopping")},
                   {"status": "row", "row": dict(exited_row(), next_state="running")}):
        state, action, _ = drive(state, {"kind": "instance_state",
                                         "results": {str(ID): result}}, WINDOW_START + 10)
        assert (state["state"], action["action"]) == ("stopping", "stop")
    state, _, _ = drive(state, {"kind": "instance_state",
                                "results": {str(ID): {"status": "row", "row": exited_row()}}},
                        WINDOW_START + 10)
    assert state["state"] == "stopped"


def test_absence_needs_two_consecutive_successful_not_found():
    state = to_state("stopped")
    state, _, _ = drive(state, None, WINDOW_START + 10)
    nf = {"status": "not_found"}
    volumes_ok = {"status": "ok", "ids": [], "label_matches": 0}
    sequence = [
        ({"instances": {str(ID): nf}, "volumes": volumes_ok}, "destroying"),
        ({"instances": {str(ID): {"status": "error", "error": "cli_timeout"}},
          "volumes": volumes_ok}, "destroying"),  # failure breaks consecutiveness
        ({"instances": {str(ID): nf}, "volumes": volumes_ok}, "destroying"),
        ({"instances": {str(ID): nf}, "volumes": {"status": "error", "error": "x"}},
         "destroying"),  # volume inventory failure never proves absence
        ({"instances": {str(ID): nf}, "volumes": volumes_ok}, "destroying"),
        ({"instances": {str(ID): nf}, "volumes": volumes_ok,
          "inventory": {"status": "error", "error": "inventory_failed"}},
         "destroying"),  # a failed label inventory never proves absence
        ({"instances": {str(ID): nf}, "volumes": volumes_ok}, "absent_verified"),
    ]
    for observation, expected in sequence:
        observation = {"inventory": CLEAN_INVENTORY, **observation}
        state, _, _ = drive(state, {"kind": "absence_check", **observation}, WINDOW_START + 10)
        assert state["state"] == expected


def test_present_instance_after_destroy_resets_and_redestroys():
    state = to_state("stopped")
    state, _, _ = drive(state, None, WINDOW_START + 10)
    state, action, _ = drive(state, {"kind": "absence_check",
                                     "instances": {str(ID): {"status": "row",
                                                             "row": exited_row()}},
                                     "volumes": {"status": "ok", "ids": [],
                                                 "label_matches": 0},
                                     "inventory": CLEAN_INVENTORY}, WINDOW_START + 10)
    assert action["instance_ids"] == [ID]


def test_incomplete_cleanup_is_flagged_and_keeps_retrying():
    state = to_state("stopped")
    state, _, _ = drive(state, None, WINDOW_START + 10)
    present = {"kind": "absence_check",
               "instances": {str(ID): {"status": "row", "row": exited_row()}},
               "volumes": {"status": "ok", "ids": [], "label_matches": 0},
               "inventory": {"status": "ok", "rows": [{"id": ID, "label": LABEL}]}}
    for _ in range(ex.DESTROY_ATTEMPTS_BEFORE_ESCALATION + 1):
        state, action, _ = drive(state, present, WINDOW_START + 10)
    assert state["state"] == "destroying"
    assert "cleanup_incomplete" in state["issues"]
    assert "cleanup_unverifiable" not in state["issues"]  # rows were seen: verifiable
    assert action["action"] == "destroy"


def test_unbound_label_volume_is_reported():
    state = to_state("stopped")
    state, _, _ = drive(state, None, WINDOW_START + 10)
    state, _, _ = drive(state, {"kind": "absence_check",
                                "instances": {str(ID): {"status": "not_found"}},
                                "volumes": {"status": "ok", "ids": [5], "label_matches": 1},
                                "inventory": CLEAN_INVENTORY},
                        WINDOW_START + 10)
    assert "unbound_task_volume" in state["issues"]


ALL_KINDS = sorted(ex.OBSERVATION_KINDS)


@pytest.mark.parametrize("name,kind", list(itertools.product(ex.STATES, ALL_KINDS)))
def test_every_unlisted_state_observation_pair_blocks(name, kind):
    """Cross-product: an observation not allowed in a state never changes it."""
    if kind in ex.ALLOWED_OBSERVATIONS[name]:
        return
    state = fresh()
    state["state"] = name
    state["instance_ids"] = [ID]
    state["create_requested_epoch"] = WINDOW_START
    state["cutoff_epoch"] = WINDOW_END
    before = copy.deepcopy(state)
    out = ex.next_action(state, {"kind": kind}, WINDOW_START + 10)
    assert out["state"] == before
    assert out["action"] == {"action": "wait"}
    assert "unexpected_observation" in [i["code"] for i in out["issues"]]
    assert state == before  # pure: input not mutated


def test_invalid_observation_and_clock_fail_closed():
    state = fresh()
    with pytest.raises(ValueError):
        ex.next_action(state, {"kind": "create_result"}, float("nan"))
    with pytest.raises(ValueError):
        ex.next_action(state, "not-a-dict", WINDOW_START)
    with pytest.raises(ValueError):
        ex.next_action({"state": "mystery"}, None, WINDOW_START)


def test_initial_state_requires_an_admitted_matching_report():
    rec = record()
    report = ex.validate_action_record(rec)
    with pytest.raises(ValueError):
        ex.initial_state(record(max_hourly_usd="3"), report, INTENT, watchdog_events="w")
    bad = record(max_hourly_usd="0")
    with pytest.raises(ValueError):
        ex.initial_state(bad, ex.validate_action_record(bad), INTENT, watchdog_events="w")
    with pytest.raises(ValueError):
        ex.initial_state(rec, report, "not-a-uuid", watchdog_events="w")


# ------------------------------------------------------------ journal and restart


def test_journal_is_hash_chained_and_excludes_wall_time(tmp_path):
    path = tmp_path / "journal.jsonl"
    times = iter(["2027-01-01T00:00:00+00:00", "2027-01-01T00:00:01+00:00"])
    with ex.LifecycleJournal(path, wall=lambda: next(times)) as journal:
        first = journal.append({"type": "transition", "state": fresh(), "action": None,
                                "observation": None, "issues": []})
        journal.append({"type": "observation", "observation": {"kind": "bootstrap_complete"}})
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["seq"] for line in lines] == [0, 1]
    assert lines[1]["prev_hash"] == first["record_hash"]
    assert lines[0]["prev_hash"] == "0" * 64
    assert first["record_hash"] == ex.identity_hash(
        {"seq": 0, "prev_hash": "0" * 64, "payload": lines[0]["payload"]})
    # Rewriting only the wall time keeps the chain valid; editing payload breaks it.
    lines[0]["at_utc"] = "1999-01-01T00:00:00+00:00"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with ex.LifecycleJournal(path):
        pass
    lines[0]["payload"]["state"]["label"] = "tampered"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ex.JournalError):
        ex.LifecycleJournal(path)


def test_journal_rejects_torn_tail_and_fsyncs(tmp_path, monkeypatch):
    syncs = []
    monkeypatch.setattr(ex.os, "fsync", lambda fd: syncs.append(fd))
    path = tmp_path / "journal.jsonl"
    with ex.LifecycleJournal(path) as journal:
        journal.append({"type": "observation", "observation": {"kind": "work_complete"}})
    assert syncs
    with path.open("a") as handle:
        handle.write('{"seq": 1')
    with pytest.raises(ex.JournalError):
        ex.LifecycleJournal(path)


class Clock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now


def coordinator(tmp_path, provider, now=WINDOW_START + 10):
    rec = record()
    report = ex.validate_action_record(rec)
    return ex.Coordinator(tmp_path / "journal.jsonl", provider, Clock(now),
                          record=rec, report=report, intent_id=INTENT,
                          watchdog_events=str(tmp_path / "watchdog-events.jsonl"))


def order_result():
    data = json.loads((EVIDENCE / "order-result.json").read_text())
    return {"outcome": "created", "instance_id": data["new_contract"]}


def test_coordinator_journals_intent_before_create_and_binds(tmp_path):
    foreign = [{"id": 7, "label": "foreign"}]
    provider = ex.FakeProvider({"list_instances": [foreign, foreign],
                                "create": [order_result()]})
    coord = coordinator(tmp_path, provider)
    coord.step()  # admitted -> create_intent: read-only label precheck
    assert provider.calls == [("list_instances",)]
    coord.step()  # create_intent -> create_requested, same-step label check, then create
    image = "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21"
    assert provider.calls == [("list_instances",), ("list_instances",),
                              ("create", 56500949, image, 160, LABEL)]
    records = [json.loads(x) for x in (tmp_path / "journal.jsonl").read_text().splitlines()]
    kinds = [(r["payload"]["type"], (r["payload"].get("state") or {}).get("state"))
             for r in records]
    assert kinds[-2:] == [("transition", "create_requested"), ("observation", None)]
    coord.step()
    assert coord.state["state"] == "bound"


def test_crash_after_create_request_restarts_into_label_reconciliation(tmp_path):
    provider = ex.FakeProvider({"list_instances": [[], []], "create": [ex.ProviderCrash()]})
    coord = coordinator(tmp_path, provider)
    coord.step()
    with pytest.raises(ex.ProviderCrash):
        coord.step()  # process killed after create_requested was journaled
    coord.close()  # the dead process's lock is released with its descriptor
    provider = ex.FakeProvider({"list_instances": [[{"id": ID, "label": LABEL}]]})
    restarted = coordinator(tmp_path, provider, now=WINDOW_START + 50)
    assert restarted.state["state"] == "create_requested"
    assert restarted.state["cutoff_epoch"] == WINDOW_END  # never renewed
    restarted.step()
    assert restarted.state["state"] == "uncertain_create"
    assert provider.calls == [("list_instances",)]
    restarted.step()
    assert restarted.state["state"] == "bound"
    assert all(call[0] != "create" for call in provider.calls)


def test_restart_at_every_state_replays_identically_and_never_renews(tmp_path):
    for index, name in enumerate(ex.NORMAL_PATH):
        if name in ("admitted",):
            continue
        state = to_state(name) if name not in ("absent_verified", "reconciled", "destroying") \
            else None
        if state is None:
            continue
        path = tmp_path / f"j{index}.jsonl"
        with ex.LifecycleJournal(path) as journal:
            journal.append({"type": "transition", "state": state, "action": None,
                            "observation": None, "issues": []})
        with ex.LifecycleJournal(path) as journal:
            replayed, pending = journal.replay()
        assert replayed == state
        assert pending is None
        later = ex.next_action(replayed, None, WINDOW_START + 20)
        if state.get("cutoff_epoch") is not None:
            assert later["state"]["cutoff_epoch"] == state["cutoff_epoch"]
        assert later["action"]["action"] != "create" or name == "create_intent"


def test_coordinator_full_fake_lifecycle_with_exited_shape(tmp_path):
    shown = [row(), exited_row(), ex.NOT_FOUND, ex.NOT_FOUND]
    provider = ex.FakeProvider({
        "create": [order_result()],
        "show_instance": shown,
        "stop": ["acknowledged_unverified"],
        "destroy": ["requested_unverified"],
        "list_volumes": [[], []],
        "list_instances": [[], [], [], []],
    })
    coord = coordinator(tmp_path, provider)
    coord.step()
    coord.step()
    coord.step()  # bound -> arm_watchdog (external)
    assert coord.state["state"] == "bound"
    (tmp_path / "watchdog-events.jsonl").write_text(json.dumps(
        {"at_utc": "x", "event": "armed", "instance_id": ID, "expected_label": LABEL,
         "deadline_utc_epoch": WINDOW_END}) + "\n")
    assert coord.observe_watchdog()["armed"] is True
    coord.step()  # -> armed, observe_instance
    coord.step()  # -> bootstrapping
    assert coord.state["state"] == "bootstrapping"
    export_root = export_copy(tmp_path)
    for observation in ({"kind": "bootstrap_complete"}, {"kind": "work_complete"},
                        exported()):
        coord.observe(observation, export_root=export_root)  # ignored but for export_verified
        coord.step()
    assert coord.state["state"] == "export_verified"
    coord.step()  # -> stopping: stop + exact show (exited)
    coord.step()  # -> stopped
    coord.step()  # -> destroying: destroy + NotFound + volumes []
    coord.step()  # first NotFound recorded; verify again
    coord.step()  # second consecutive NotFound -> absent_verified
    assert coord.state["state"] == "absent_verified"
    assert [c[0] for c in provider.calls] == [
        "list_instances", "list_instances", "create", "show_instance", "stop", "show_instance", "destroy",
        "show_instance", "list_volumes", "list_instances", "show_instance", "list_volumes",
        "list_instances"]
    effects = [r["payload"] for r in coord.journal.records
               if r["payload"].get("type") == "effect"]
    assert effects == [
        {"type": "effect", "operation": "stop", "resource_id": ID,
         "outcome": "acknowledged_unverified"},
        {"type": "effect", "operation": "destroy", "resource_id": ID,
         "outcome": "requested_unverified"}]
    assert all(c[1] == ID for c in provider.calls if c[0] in ("stop", "destroy",
                                                                "show_instance"))


def test_coordinator_refuses_unadmitted_record(tmp_path):
    rec = record(max_hourly_usd="0")
    with pytest.raises(ValueError):
        ex.Coordinator(tmp_path / "j.jsonl", ex.FakeProvider({}), Clock(WINDOW_START),
                       record=rec, report=ex.validate_action_record(rec), intent_id=INTENT,
                       watchdog_events="w")


def test_coordinator_rejects_journal_from_another_record(tmp_path):
    coord = coordinator(tmp_path, ex.FakeProvider({"list_instances": [[]]}))
    coord.step()
    coord.close()
    rec = record(max_hourly_usd="2.40")
    with pytest.raises(ValueError):
        ex.Coordinator(tmp_path / "journal.jsonl", ex.FakeProvider({}), Clock(WINDOW_START),
                       record=rec, report=ex.validate_action_record(rec), intent_id=INTENT,
                       watchdog_events="evidence/watchdog-events.jsonl")


def test_observe_rejects_unknown_kinds_and_raw_fields(tmp_path):
    coord = coordinator(tmp_path, ex.FakeProvider({}))
    with pytest.raises(ValueError):
        coord.observe({"kind": "watchdog_receipt", "armed": False})  # watchdog file only
    with pytest.raises(ValueError):
        coord.observe({"kind": "anything"})
    with pytest.raises(ValueError):
        coord.observe({"kind": "work_complete", "stdout": "SECRET"})
    with pytest.raises(ValueError):
        coord.observe({"kind": "instance_state", "results": {}})  # provider-only kind
    coord.observe({"kind": "work_complete"})
    with pytest.raises(ValueError):
        coord.observe({"kind": "export_failed", "error": "x"})  # never silently replaced


# ------------------------------------------------------------ watchdog receipt


def test_watchdog_receipt_reads_durable_armed_event(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"at_utc": "x", "event": "armed", "instance_id": ID,
                                "expected_label": LABEL,
                                "deadline_utc_epoch": WINDOW_END}) + "\n")
    receipt = ex.read_watchdog_receipt(path)
    assert receipt == {"kind": "watchdog_receipt", "armed": True, "instance_id": ID,
                       "label": LABEL, "deadline_epoch": WINDOW_END}
    path.write_text(json.dumps({"event": "command_error", "operation": "show"}) + "\n")
    assert ex.read_watchdog_receipt(path) == {"kind": "watchdog_receipt", "armed": False}
    assert ex.read_watchdog_receipt(tmp_path / "missing.jsonl") == {
        "kind": "watchdog_receipt", "armed": False}
    path.write_text("not json\n")
    assert ex.read_watchdog_receipt(path)["armed"] is False


# ------------------------------------------------------------ Vast CLI adapter


class Runner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, timeout):
        assert isinstance(argv, list) and timeout > 0
        self.calls.append(argv)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def completed(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_vast_create_uses_live_verified_argv_and_parses_order_result():
    data = (EVIDENCE / "order-result.json").read_text()
    runner = Runner([completed(json.dumps({k: v for k, v in json.loads(data).items()
                                           if k != "at_utc"}))])
    client = ex.VastCliClient(runner)
    result = client.create(56500949, "docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21", 160,
                           LABEL)
    assert result == {"outcome": "created", "instance_id": ID}
    order = json.loads((EVIDENCE / "order-request.json").read_text())["args"]
    # Live-verified argv plus the global --retry 1 (A3-12), label replaced.
    assert runner.calls[0] == order[:2] + ["--retry", "1"] + order[2:-1] + [LABEL]


@pytest.mark.parametrize(
    "response",
    [completed("prose SECRET"), completed('{"success": false, "msg": "SECRET"}'),
     completed('{"success": true}'), completed('{"success": true, "new_contract": true}'),
     completed("", '{"error": true}', 1), subprocess.TimeoutExpired(["x"], 1),
     OSError("missing")],
)
def test_vast_create_failures_are_uncertain_never_success(response):
    client = ex.VastCliClient(Runner([response]))
    result = client.create(1, "img", 10, LABEL)
    assert result["outcome"] == "uncertain"
    assert "SECRET" not in json.dumps(result)


def test_vast_show_instance_row_not_found_and_errors():
    shape = json.loads((EVIDENCE / "state-065200.json").read_text())
    runner = Runner([
        completed(json.dumps(shape)),
        completed("", json.dumps({"error": True, "status_code": 404, "msg": "SECRET"})),
        completed("", json.dumps({"error": True, "status_code": 500, "msg": "x"})),
        completed("Traceback SECRET", "", 1),
        completed(json.dumps({**shape, "id": 1})),
    ])
    client = ex.VastCliClient(runner)
    assert client.show_instance(ID)["actual_status"] == "loading"
    assert client.show_instance(ID) is ex.NOT_FOUND
    for _ in range(3):
        with pytest.raises(ex.ProviderError) as error:
            client.show_instance(ID)
        assert "SECRET" not in str(error.value)
    assert runner.calls[0] == ["vastai", "--raw", "show", "instance", str(ID)]


def test_vast_stop_destroy_and_volumes_argv_and_acknowledgments():
    runner = Runner([
        completed("stopping instance 58628722.\ntrue"),
        completed(""),  # installed 1.0.7 destroy success: rc 0, nothing printed
        completed("[]"),
        completed('{"success": true}'),
        completed("", '{"error": true, "status_code": 401, "msg": "x"}'),
    ])
    client = ex.VastCliClient(runner)
    assert client.stop(ID) == "acknowledged_unverified"
    assert client.destroy(ID) == "requested_unverified"
    assert client.list_volumes() == []
    assert client.destroy_volume(9) == "acknowledged_unverified"
    with pytest.raises(ex.ProviderError):
        client.stop(ID)
    assert runner.calls[:4] == [
        ["vastai", "--raw", "stop", "instance", str(ID)],
        ["vastai", "--raw", "destroy", "instance", str(ID), "--yes"],
        ["vastai", "--raw", "show", "volumes"],
        ["vastai", "--raw", "delete", "volume", "9"],
    ]


def test_vast_list_instances_requires_a_list_of_identified_rows():
    client = ex.VastCliClient(Runner([completed("[]"), completed('{"success": true}'),
                                      completed('[{"label": "x"}]')]))
    assert client.list_instances() == []
    for _ in range(2):
        with pytest.raises(ex.ProviderError):
            client.list_instances()


def test_vast_charges_paginates_and_keeps_decimal_text():
    charges = json.loads((EVIDENCE / "charges-after-delete.json").read_text())["task_rows"]
    runner = Runner([
        completed(json.dumps({"results": charges, "next_token": "tok"})),
        completed(json.dumps({"results": [], "next_token": None})),
    ])
    rows = ex.VastCliClient(runner).charges(1793491200, 1793577600)
    assert rows[0]["amount"] == "0.6"
    assert rows[0]["items"][0]["amount"] == "0.577"
    assert runner.calls == [
        ["vastai", "--raw", "show", "invoices-v1", "--charges", "--start-date", "1793491200",
         "--end-date", "1793577600", "--limit", "100"],
        ["vastai", "--raw", "show", "invoices-v1", "--charges", "--start-date", "1793491200",
         "--end-date", "1793577600", "--limit", "100", "--next-token", "tok"],
    ]


def test_default_runner_env_is_allowlisted_without_credentials(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "SECRET")
    monkeypatch.setenv("HF_TOKEN", "SECRET")
    env = ex.allowlisted_env()
    assert "VAST_API_KEY" not in env and "HF_TOKEN" not in env
    assert set(env) <= set(ex.ENV_ALLOWLIST)


def test_fake_provider_scripts_timeouts_and_records_calls():
    provider = ex.FakeProvider({"list_instances": [ex.ProviderError("cli_timeout"), []]})
    with pytest.raises(ex.ProviderError):
        provider.list_instances()
    assert provider.list_instances() == []
    with pytest.raises(AssertionError):
        provider.list_instances()  # exhausted script is a test failure, never a default


def test_lifecycle_limits_cannot_exceed_the_window():
    value = record()
    value["lifecycle_limits"]["startup_limit_seconds"] = 3601
    assert "limit_exceeds_window" in codes(ex.validate_action_record(value))


def test_charge_rows_become_one_cumulative_snapshot_per_bound_resource():
    text = (EVIDENCE / "charges-after-delete.json").read_text()
    rows = json.loads(text, parse_float=str)["task_rows"]
    events = ex.charge_snapshot_events(rows, "physloop-s2-host-20261101-v1", [ID], 1000)
    # D-019 known answer: one $0.6 row for instance 58628722.
    assert events == [{"event_id": f"charge-{ID}-1000", "kind": "provider_charge_snapshot",
                       "resource_id": str(ID), "at_epoch": 1000, "amount_usd": "0.6"}]
    second_day = dict(rows[0], amount="0.25")
    foreign = dict(rows[0], source="instance-1", metadata={"label": "someone-else"})
    events = ex.charge_snapshot_events(rows + [second_day, foreign], "nope", [ID], 1000)
    assert [e["amount_usd"] for e in events] == ["0.85"]  # 0.6 + 0.25; foreign row dropped
    with pytest.raises(ValueError):
        ex.charge_snapshot_events([dict(rows[0], amount=0.6)], "nope", [ID], 1000)


# ------------------------------------------------------------ review regressions (A3)


def plain_coordinator(path, provider, now=WINDOW_START + 10, events=None):
    rec = record()
    return ex.Coordinator(path, provider, Clock(now), record=rec,
                          report=ex.validate_action_record(rec), intent_id=INTENT,
                          watchdog_events=str(events or path.with_name("watchdog.jsonl")))


def test_a3_01_second_live_coordinator_on_one_journal_is_refused(tmp_path):
    path = tmp_path / "j.jsonl"
    provider = ex.FakeProvider({"list_instances": [[], []],
                                "create": [{"outcome": "created", "instance_id": ID}]})
    first = plain_coordinator(path, provider)
    first.step()
    with pytest.raises(ex.JournalError, match="journal_locked"):
        plain_coordinator(path, provider)
    first.step()
    first.close()
    assert [c[0] for c in provider.calls].count("create") == 1
    with ex.LifecycleJournal(path):
        pass  # chain intact


def test_a3_01_lost_journal_never_issues_a_second_create(tmp_path):
    path = tmp_path / "j.jsonl"
    provider = ex.FakeProvider({
        "list_instances": [[], [], [{"id": ID, "label": LABEL}],
                           [{"id": ID, "label": LABEL}]],
        "create": [{"outcome": "created", "instance_id": ID}]})
    first = plain_coordinator(path, provider)
    first.step()
    first.step()
    first.close()
    path.unlink()  # journal lost or rolled back
    second = plain_coordinator(path, provider)
    second.step()  # admitted -> create_intent: label precheck
    decision = second.step()  # label already present: never create
    assert decision["state"]["state"] == "uncertain_create"
    assert "label_present_before_create" in decision["state"]["issues"]
    second.step()
    assert second.state["state"] == "bound" and second.state["instance_ids"] == [ID]
    second.close()
    assert [c[0] for c in provider.calls].count("create") == 1


def test_a3_01_external_append_is_detected_before_the_next_append(tmp_path):
    path = tmp_path / "j.jsonl"
    journal = ex.LifecycleJournal(path)
    journal.append({"type": "observation", "observation": {"kind": "work_complete"}})
    with open(path, "ab") as handle:  # a writer that ignored the lock
        handle.write(b'{"forged": true}\n')
    with pytest.raises(ex.JournalError, match="journal_changed_underneath"):
        journal.append({"type": "observation", "observation": {"kind": "work_complete"}})
    journal.close()


def test_a3_02_observe_refuses_hand_written_watchdog_receipt(tmp_path):
    coord = plain_coordinator(tmp_path / "j.jsonl", ex.FakeProvider({}))
    with pytest.raises(ValueError):
        coord.observe({"kind": "watchdog_receipt", "armed": True, "instance_id": ID,
                       "label": LABEL, "deadline_epoch": WINDOW_END})
    coord.close()


def armed_line(**overrides):
    value = {"event": "armed", "instance_id": ID, "expected_label": LABEL,
             "deadline_utc_epoch": WINDOW_END}
    value.update(overrides)
    return json.dumps(value) + "\n"


@pytest.mark.parametrize("plan_file_armed", [False, True])
def test_a3_02_observe_watchdog_reads_only_the_plan_events_path(tmp_path, plan_file_armed):
    events = tmp_path / "plan-events.jsonl"
    provider = ex.FakeProvider({"list_instances": [[], []], "create": [order_result()],
                                "show_instance": [row(status="loading")]})
    coord = plain_coordinator(tmp_path / "j.jsonl", provider, events=events)
    for _ in range(3):
        coord.step()
    assert coord.state["state"] == "bound"
    (tmp_path / "other.jsonl").write_text(armed_line())  # a file the plan never named
    if plan_file_armed:
        events.write_text(armed_line())
    assert coord.observe_watchdog()["armed"] is plan_file_armed
    coord.step()
    assert coord.state["state"] == ("armed" if plan_file_armed else "arm_failed")
    coord.close()


@pytest.mark.parametrize("terminal", ["stopped_verified", "instance_absent", "scope_mismatch",
                                      "scope_mismatch_after_arm", "stop_unconfirmed",
                                      "binding_unverified_past_deadline", "deadline_reached"])
def test_a3_02_receipt_after_watchdog_exit_is_not_armed(tmp_path, terminal):
    path = tmp_path / "events.jsonl"
    path.write_text(armed_line() + json.dumps({"event": terminal, "instance_id": ID}) + "\n")
    assert ex.read_watchdog_receipt(path)["armed"] is False
    path.write_text(armed_line() + json.dumps({"event": "command_error", "operation": "show",
                                               "error": "cli_timeout"}) + "\n")
    assert ex.read_watchdog_receipt(path)["armed"] is True


def test_a3_04_stopping_needs_two_consecutive_not_found():
    state = to_state("stopping")
    nf = {"kind": "instance_state", "results": {str(ID): {"status": "not_found"}}}
    state, action, _ = drive(state, nf, WINDOW_START + 10)
    assert state["state"] == "stopping" and action["action"] == "stop"
    state, _, _ = drive(state, {"kind": "instance_state", "results": {
        str(ID): {"status": "error", "error": "cli_timeout"}}}, WINDOW_START + 11)
    state, _, _ = drive(state, nf, WINDOW_START + 12)
    assert state["state"] == "stopping"  # an error broke consecutiveness
    state, _, _ = drive(state, nf, WINDOW_START + 13)
    assert state["state"] == "stopped" and state["stopped_ids"] == [ID]
    state, action, _ = drive(state, None, WINDOW_START + 14)
    assert state["state"] == "destroying"
    assert action["instance_ids"] == [ID]  # destroying does its own absence count


def retained_state(name):
    rec = record()
    rec["retention_after_stop"] = {"mode": "retain_until", "authority_ref": "D-999",
                                   "retain_until_utc": "2027-01-22T09:00:00+00:00"}
    plan = ex.initial_state(rec, ex.validate_action_record(rec), INTENT,
                            watchdog_events="w")["plan"]
    state = to_state("exporting")
    state["plan"] = plan
    state, _, _ = drive(state, {"kind": "export_failed", "error": "x"}, WINDOW_START + 20)
    state, _, _ = drive(state, {"kind": "export_failed", "error": "x"}, WINDOW_START + 21)
    assert state["state"] == "stopping"
    if name == "stopping":
        return state
    state, _, _ = drive(state, {"kind": "instance_state", "results": {
        str(ID): {"status": "row", "row": exited_row()}}}, WINDOW_START + 22)
    assert state["state"] == name
    return state


def test_a3_04_hold_for_retention_reobserves_and_restops_a_restarted_instance():
    state = retained_state("stopped")
    state, action, _ = drive(state, None, WINDOW_START + 30)
    assert action == {"action": "hold_for_retention",
                      "retain_until_utc": "2027-01-22T09:00:00+00:00", "instance_ids": [ID]}
    state, action, _ = drive(state, {"kind": "instance_state", "results": {
        str(ID): {"status": "row", "row": exited_row()}}}, WINDOW_START + 6 * 86400)
    assert (state["state"], action["action"]) == ("stopped", "hold_for_retention")
    state, action, _ = drive(state, {"kind": "instance_state", "results": {
        str(ID): {"status": "row", "row": row()}}}, WINDOW_START + 6 * 86400 + 1)
    assert state["state"] == "stopping"
    assert "instance_running_during_retention" in state["issues"]
    assert action == {"action": "stop", "instance_ids": [ID]}


def test_a3_05_installed_cli_destroy_success_prints_nothing():
    # vastai 1.0.7 destroy__instance drops the result: rc 0, empty stdout and stderr.
    runner = Runner([completed(""), completed("", json.dumps(
        {"error": True, "status_code": 404, "msg": "SECRET"}))])
    client = ex.VastCliClient(runner)
    assert client.destroy(ID) == "requested_unverified"
    with pytest.raises(ex.ProviderError, match="destroy_rejected"):
        client.destroy(ID)


def test_a3_06_persistent_lookup_errors_escalate_cleanup_unverifiable():
    state = to_state("stopped")
    state, _, _ = drive(state, None, WINDOW_START + 10)
    obs = {"kind": "absence_check",
           "instances": {str(ID): {"status": "error", "error": "cli_nonzero"}},
           "volumes": {"status": "ok", "ids": [], "label_matches": 0},
           "inventory": {"status": "ok", "rows": []}}
    for _ in range(ex.DESTROY_ATTEMPTS_BEFORE_ESCALATION + 1):
        state, action, _ = drive(state, obs, WINDOW_START + 10)
    assert state["state"] == "destroying" and action["action"] == "destroy"
    assert "cleanup_unverifiable" in state["issues"]


def test_a3_06_stop_and_destroy_request_outcomes_are_journaled(tmp_path):
    provider = ex.FakeProvider({"stop": [ex.ProviderError("stop_rejected")],
                                "show_instance": [row()]})
    coord = plain_coordinator(tmp_path / "j.jsonl", provider)
    coord.journal.append({"type": "transition", "state": to_state("export_verified"),
                          "action": None, "observation": None, "issues": []})
    coord.step()
    coord.close()
    effects = [json.loads(line)["payload"] for line in
               (tmp_path / "j.jsonl").read_text().splitlines()]
    assert {"type": "effect", "operation": "stop", "resource_id": ID,
            "outcome": "stop_rejected"} in effects


def test_a3_08_action_record_schema_version_must_be_the_integer_one():
    for version in (True, 1.0):
        report = ex.validate_action_record(record(schema_version=version))
        assert report["admitted"] is False
        assert "unknown_schema" in codes(report)


def test_a3_11_journal_refuses_symlink_and_non_regular_files(tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(ex.JournalError, match="journal_not_regular_file"):
        ex.LifecycleJournal(link)
    with pytest.raises(ex.JournalError, match="journal_not_regular_file"):
        ex.LifecycleJournal.read(link)
    with pytest.raises(ex.JournalError, match="journal_not_regular_file"):
        ex.LifecycleJournal(tmp_path)
    assert target.read_bytes() == b""


def test_a3_11_read_only_loader_never_creates(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ex.JournalError, match="journal_missing"):
        ex.LifecycleJournal.read(missing)
    assert not missing.exists()


def test_a3_12_create_disables_the_cli_internal_retry():
    runner = Runner([completed(json.dumps({"success": True, "new_contract": ID}))])
    ex.VastCliClient(runner).create(1, "img", 10, LABEL)
    assert runner.calls[0][:4] == ["vastai", "--raw", "--retry", "1"]


def test_a3_12_post_bind_duplicate_label_is_stopped_destroyed_and_reported():
    state = to_state("stopped")
    state, _, _ = drive(state, None, WINDOW_START + 10)
    nf = {"status": "not_found"}
    dup = ID + 1
    check = {"kind": "absence_check", "instances": {str(ID): nf},
             "volumes": {"status": "ok", "ids": [], "label_matches": 0},
             "inventory": {"status": "ok", "rows": [{"id": dup, "label": LABEL}]}}
    state, action, _ = drive(state, check, WINDOW_START + 10)
    assert state["state"] == "stopping"
    assert "duplicate_creation" in state["issues"]
    assert state["instance_ids"] == [ID, dup]
    assert action == {"action": "stop", "instance_ids": [dup]}
    state, _, _ = drive(state, {"kind": "instance_state", "results": {
        str(dup): {"status": "row", "row": dict(exited_row(), id=dup)}}}, WINDOW_START + 11)
    assert state["state"] == "stopped"
    state, action, _ = drive(state, None, WINDOW_START + 12)
    assert action["action"] == "destroy" and dup in action["instance_ids"]
    clean = {"kind": "absence_check", "instances": {str(ID): nf, str(dup): nf},
             "volumes": {"status": "ok", "ids": [], "label_matches": 0},
             "inventory": {"status": "ok", "rows": []}}
    failed = dict(clean, inventory={"status": "error", "error": "inventory_failed"})
    state, _, _ = drive(state, clean, WINDOW_START + 13)
    state, _, _ = drive(state, failed, WINDOW_START + 14)
    assert state["state"] == "destroying"  # a failed inventory never proves absence
    state, _, _ = drive(state, clean, WINDOW_START + 15)
    assert state["state"] == "absent_verified"


@pytest.mark.parametrize("field", ["num_gpus", "disk_space", "image_uuid", "gpu_name",
                                   "dph_total"])
def test_a3_15_null_fields_wait_for_the_startup_limit_not_wrong_resource(field):
    state = to_state("armed")
    for status in ("loading", "running"):
        state, action, _ = drive(state, {"kind": "instance_state", "results": {
            str(ID): {"status": "row", "row": row(status=status, **{field: None})}}},
            WINDOW_START + 20)
        assert (state["state"], action["action"]) == ("armed", "observe_instance")
        assert "wrong_resource" not in state["issues"]
    assert field not in state["observed_row_keys"]
    state, _, _ = drive(state, {"kind": "instance_state", "results": {
        str(ID): {"status": "row", "row": row(status="loading", **{field: None})}}},
        WINDOW_START + 10 + 1200)
    assert state["state"] == "stopping" and "startup_limit_exceeded" in state["issues"]


# ------------------------------------------------------------ re-verification residuals (A3R)


def test_a3r_01_rolled_back_journal_before_create_never_creates_twice(tmp_path):
    # The precheck inventory (label absent) is journaled as a pending observation; a
    # journal copy taken then and restored after the create must not replay it into a
    # second create: create lists instances again in its own step.
    path = tmp_path / "j.jsonl"
    provider = ex.FakeProvider({
        "list_instances": [[], [], [{"id": ID, "label": LABEL}], [{"id": ID, "label": LABEL}]],
        "create": [{"outcome": "created", "instance_id": ID}]})
    first = plain_coordinator(path, provider)
    first.step()  # admitted -> create_intent; label-absent inventory pending
    snapshot = path.read_bytes()  # operator copies the journal between two CLI steps
    first.step()  # create_intent -> create_requested -> one create
    first.close()
    path.write_bytes(snapshot)  # rollback to the stale label-absent inventory
    second = plain_coordinator(path, provider, now=WINDOW_START + 600)
    decision = second.step()
    assert decision["state"]["state"] == "create_requested"
    decision = second.step()  # the in-step inventory saw our label: uncertain, no create
    assert decision["state"]["state"] == "uncertain_create"
    assert decision["action"] == {"action": "reconcile_by_label", "label": LABEL}
    assert "label_present_before_create" in decision["state"]["issues"]
    second.step()
    assert second.state["state"] == "bound" and second.state["instance_ids"] == [ID]
    second.close()
    assert [c[0] for c in provider.calls].count("create") == 1


def test_a3r_01_failed_inventory_in_the_create_step_never_creates(tmp_path):
    provider = ex.FakeProvider({"list_instances": [[], ex.ProviderError("cli_timeout"),
                                                   []]})
    coord = plain_coordinator(tmp_path / "j.jsonl", provider)
    coord.step()
    coord.step()
    decision = coord.step()
    assert decision["state"]["state"] == "uncertain_create"
    coord.close()
    assert [c[0] for c in provider.calls] == ["list_instances"] * 3  # never create
    observations = [json.loads(line)["payload"] for line in
                    (tmp_path / "j.jsonl").read_text().splitlines()]
    assert {"type": "observation", "observation": {
        "kind": "create_result", "outcome": "uncertain",
        "error": "inventory_failed_before_create"}} in observations


def test_a3r_02_alternating_error_and_not_found_escalates_at_the_check_limit():
    state = to_state("stopped")
    state, action, _ = drive(state, None, WINDOW_START + 10)
    assert state["state"] == "destroying"
    base = {"kind": "absence_check", "volumes": {"status": "ok", "ids": [], "label_matches": 0},
            "inventory": {"status": "ok", "rows": []}}
    error = dict(base, instances={str(ID): {"status": "error", "error": "cli_timeout"}})
    missing = dict(base, instances={str(ID): {"status": "not_found"}})
    seen = []
    for check in range(1, 2 * ex.DESTROYING_CHECK_LIMIT + 1):
        state, action, _ = drive(state, error if check % 2 else missing, WINDOW_START + 10)
        seen.append(state["state"])
        if state["state"] != "destroying":
            break
    assert seen == ["destroying"] * (ex.DESTROYING_CHECK_LIMIT - 1) + ["escalated"]
    assert "cleanup_check_limit_reached" in state["issues"]
    assert action == {"action": "manual_review", "label": LABEL}
    state, action, _ = drive(state, None, WINDOW_START + 11)
    assert (state["state"], action["action"]) == ("escalated", "manual_review")


def test_a3r_02_check_limit_counts_from_entering_destroying():
    state = to_state("stopped")
    state, _, _ = drive(state, None, WINDOW_START + 10)
    nf = {"status": "not_found"}
    clean = {"kind": "absence_check", "instances": {str(ID): nf},
             "volumes": {"status": "ok", "ids": [], "label_matches": 0},
             "inventory": {"status": "ok", "rows": []}}
    state, _, _ = drive(state, clean, WINDOW_START + 10)
    state, _, _ = drive(state, clean, WINDOW_START + 11)
    assert state["state"] == "absent_verified"  # well inside the limit
    assert "cleanup_check_limit_reached" not in state["issues"]


@pytest.mark.parametrize("result,code", [
    ({"status": "not_found"}, "retained_instance_missing"),
    ({"status": "error", "error": "cli_timeout"}, "retention_check_failed"),
])
def test_a3r_04_hold_for_retention_reports_a_missing_or_unchecked_instance(result, code):
    state = retained_state("stopped")
    state, action, _ = drive(state, None, WINDOW_START + 30)
    assert action["action"] == "hold_for_retention"
    assert code not in state["issues"]
    state, action, _ = drive(state, {"kind": "instance_state", "results": {str(ID): result}},
                             WINDOW_START + 86400)
    assert (state["state"], action["action"]) == ("stopped", "hold_for_retention")
    assert code in state["issues"]


def test_a3r_02_coordinator_with_alternating_lookups_stops_at_the_limit(tmp_path):
    limit = ex.DESTROYING_CHECK_LIMIT
    provider = ex.FakeProvider({
        "destroy": ["requested_unverified"] * limit,
        "show_instance": [ex.ProviderError("cli_timeout"), ex.NOT_FOUND] * limit,
        "list_volumes": [[]] * limit, "list_instances": [[]] * limit})
    coord = plain_coordinator(tmp_path / "j.jsonl", provider)
    coord.journal.append({"type": "transition", "state": to_state("stopped"),
                          "action": None, "observation": None, "issues": []})
    for _ in range(limit + 1):  # enter destroying, then `limit` absence checks
        coord.step()
    assert coord.state["state"] == "escalated"
    assert "cleanup_check_limit_reached" in coord.state["issues"]
    calls = len(provider.calls)
    assert coord.step()["action"] == {"action": "manual_review", "label": LABEL}
    assert len(provider.calls) == calls  # stop condition: no further provider calls
    coord.close()


# ------------------------------------------------------------ G1: verified-export destroy guard


def test_g1_receipt_hash_literal_is_the_canonical_identity():
    assert ex.identity_hash(RECEIPTS) == RECEIPTS_HASH
    assert ex.check_export_verification(envelope(), RECEIPTS_HASH) == []
    assert ex.validate_observation(exported())["manifest_hash"] == RECEIPTS_HASH


@pytest.mark.parametrize("mutate, code", [
    # failed: the report itself says the export did not verify
    (lambda v: v.update(status="blocked"), "verification_not_passed"),
    (lambda v: v["result"].update(verified=False), "verification_not_passed"),
    (lambda v: v["result"].update(issues=["export_file_set_mismatch"]),
     "verification_not_passed"),
    (lambda v: v["result"].update(file_count=2), "verification_not_passed"),
    # forged: receipts edited, hash left alone -> the bytes do not re-verify
    (lambda v: v["receipts"][0].update(size_bytes=3), "manifest_bytes_do_not_reverify"),
    (lambda v: v.update(receipts=[]), "receipts_missing"),
    (lambda v: v.update(extra=1), "verification_shape"),
    (lambda v: v.update(artifact_kind="s2_export_receipt"), "verification_kind_unknown"),
])
def test_g1_failed_or_forged_reports_are_refused(mutate, code):
    report = copy.deepcopy(envelope())
    mutate(report)
    assert code in ex.check_export_verification(report, RECEIPTS_HASH)
    with pytest.raises(ValueError, match="export verification refused|keys"):
        ex.validate_observation({"kind": "export_verified", "manifest_hash": RECEIPTS_HASH,
                                 "verification": report})


def test_g1_mismatched_typed_hash_and_bare_hash_are_refused():
    assert ex.check_export_verification(envelope(), "a" * 64) == ["manifest_hash_mismatch"]
    with pytest.raises(ValueError):  # the A3 bare-hash shape no longer exists
        ex.validate_observation({"kind": "export_verified", "manifest_hash": RECEIPTS_HASH})
    with pytest.raises(ValueError):
        ex.validate_observation(dict(exported(), manifest_hash="a" * 64))
    state = to_state("exporting")
    with pytest.raises(ValueError):
        ex.next_action(state, dict(exported(), manifest_hash="a" * 64), WINDOW_START + 100)
    assert state["export_verified"] is False


def test_g1_forged_report_at_the_hard_cutoff_is_stop_only():
    state = to_state("exporting")
    with pytest.raises(ValueError):
        ex.next_action(state, exported(status="blocked"), WINDOW_START + 100)
    state, _, _ = drive(state, None, WINDOW_END)  # the forged report never entered
    assert state["state"] == "cutoff_reached"
    assert {"cutoff_reached", "export_incomplete"} <= set(state["issues"])
    state, action, _ = drive(state, None, WINDOW_END)
    assert action == {"action": "stop", "instance_ids": [ID]}
    state, _, _ = drive(state, {"kind": "instance_state", "results": {
        str(ID): {"status": "row", "row": exited_row()}}}, WINDOW_END)
    state, action, _ = drive(state, None, WINDOW_END + 1)
    assert (state["state"], action["action"]) == ("escalated", "manual_review")
    assert "destroy_refused_export_unverified" in state["issues"]


def test_g1_never_armed_instance_is_destroyed_without_export():
    state = to_state("bound")
    state, action, _ = drive(state, None, WINDOW_START + 10 + 120)  # arm limit 120 s
    assert state["state"] == "arm_failed"
    state, action, _ = drive(state, None, WINDOW_START + 140)
    state, _, _ = drive(state, {"kind": "instance_state", "results": {
        str(ID): {"status": "row", "row": exited_row()}}}, WINDOW_START + 150)
    assert state["state"] == "stopped" and state["armed_epoch"] is None
    state, action, _ = drive(state, None, WINDOW_START + 160)
    assert state["state"] == "destroying" and action["action"] == "destroy"
    assert "destroyed_before_any_work" in state["issues"]
    assert "destroy_refused_export_unverified" not in state["issues"]


def _bundle(tmp_path):
    from gate.s2_bundle import seal_bundle, verify_bundle
    from gate.s2_identity import canonical_json_bytes

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "attempts.json").write_bytes(canonical_json_bytes({"attempts": 1}) + b"\n")
    sealed = seal_bundle(root, {
        "artifact_kind": "s2_bundle_index", "schema_version": 1, "scope": "synthetic",
        "bundle_id": "g1-bundle", "required_roles": [], "entries": [
            {"path": "attempts.json", "role": "attempts", "restriction_class": "public_safe",
             "licensed_lineage": False, "replay": None}]})
    assert sealed["status"] == "passed"
    return root, verify_bundle(root)


def test_g1_bundle_report_reverifies_from_its_manifest_bytes(tmp_path):
    root, report = _bundle(tmp_path)
    digest = report["manifest_sha256"]
    observation = {"kind": "export_verified", "manifest_hash": digest, "verification": report}
    assert ex.validate_observation(observation) is observation
    assert ex.reverify_export(observation, root) == []
    forged = copy.deepcopy(report)
    forged["manifest"]["entries"][0]["size_bytes"] += 1  # status still "passed"
    assert "manifest_bytes_do_not_reverify" in ex.check_export_verification(forged, digest)
    assert ex.check_export_verification(report, "b" * 64) == ["manifest_hash_mismatch"]
    (root / "attempts.json").write_bytes(b"{}\n")  # the copy changed after verification
    failed = __import__("gate.s2_bundle", fromlist=["verify_bundle"]).verify_bundle(root)
    assert failed["status"] == "blocked"
    assert "verification_not_passed" in ex.check_export_verification(failed, digest)
    assert ex.reverify_export(observation, root) == ["export_report_not_reproduced"]


def test_g1_reverify_binds_the_envelope_to_the_bytes_on_disk(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    (root / "attempts.json").write_bytes(b"{}")
    report = ex.export_verification_report(root, copy.deepcopy(RECEIPTS))
    assert report == envelope()  # literal known answer: the producer's exact envelope
    observation = exported()
    assert ex.reverify_export(observation, root) == []
    (root / "attempts.json").write_bytes(b"[]")  # same size, other bytes
    assert ex.reverify_export(observation, root) == ["export_report_not_reproduced"]
    assert ex.reverify_export({"kind": "work_complete"}, root) == ["not_export_verified"]


def test_g1_coordinator_refuses_a_forged_export_report(tmp_path):
    coord = plain_coordinator(tmp_path / "j.jsonl", ex.FakeProvider({}))
    coord.journal.append({"type": "transition", "state": to_state("exporting"),
                          "action": None, "observation": None, "issues": []})
    root = export_copy(tmp_path)
    with pytest.raises(ValueError):
        coord.observe(exported(status="blocked"), export_root=root)
    assert coord.journal.replay()[1] is None
    coord.observe(exported(), export_root=root)
    coord.step()
    assert coord.state["state"] == "export_verified" and coord.state["export_verified"] is True
    coord.close()


# ------------------------------------------------------------ G1 review regressions


def _exporting_coordinator(tmp_path, rec, provider=None):
    provider = provider or ex.FakeProvider({})
    coord = ex.Coordinator(tmp_path / "j.jsonl", provider, Clock(WINDOW_START + 100),
                           record=rec, report=ex.validate_action_record(rec), intent_id=INTENT,
                           watchdog_events=str(tmp_path / "w.jsonl"))
    init = ex.initial_state(rec, ex.validate_action_record(rec), INTENT,
                            watchdog_events=str(tmp_path / "w.jsonl"))
    state = to_state("exporting")
    state.update(record_hash=init["record_hash"], plan=init["plan"])
    coord.journal.append({"type": "transition", "state": state, "action": None,
                          "observation": None, "issues": []})
    return coord, provider


def test_g1_07_coordinator_observe_binds_the_report_to_bytes_on_disk(tmp_path):
    """Review G1-07: a self-consistent envelope naming no file never reaches the journal."""
    receipts = [{"path": "nowhere/evidence.json", "size_bytes": 6,
                 "sha256": "c2b4a9e1b8a8d0d0e8b5f1e4ad4a2c6fe2a0fd86b3b7c43a0fd6f0bd7b8c1d10"}]
    digest = ex.identity_hash(receipts)
    forged = {"kind": "export_verified", "manifest_hash": digest,
              "verification": envelope(receipts, manifest_sha256=digest)}
    assert ex.check_export_verification(forged["verification"], digest) == []  # self-consistent
    coord, provider = _exporting_coordinator(tmp_path, record())
    with pytest.raises(ValueError, match="export_root_required"):
        coord.observe(forged)
    with pytest.raises(ValueError, match="export_report_not_reproduced"):
        coord.observe(forged, export_root=export_copy(tmp_path))
    assert coord.journal.replay()[1] is None
    for _ in range(3):
        coord.step()
    assert coord.state["state"] == "exporting" and provider.calls == []
    coord.observe(exported(), export_root=tmp_path / "export")  # control: the real copy
    assert coord.journal.replay()[1]["kind"] == "export_verified"
    coord.close()


def _hand_sealed_empty_bundle(root, scope="synthetic"):
    """A zero-entry manifest and seal written by hand (seal_bundle refuses an empty index)."""
    import hashlib

    from gate import s2_bundle as b
    from gate.s2_identity import canonical_json_bytes

    root.mkdir(parents=True)
    manifest = {"artifact_kind": b.MANIFEST_KIND, "schema_version": 1, "scope": scope,
                "bundle_id": "empty-export", "index_sha256": "0" * 64, "entries": [],
                "publication_class": "public_safe_candidate", "publication_authorized": False,
                "authenticity": "not_verified", "units": {"size_bytes": "bytes"}}
    data = canonical_json_bytes(manifest) + b"\n"
    (root / b.MANIFEST_NAME).write_bytes(data)
    seal = {"artifact_kind": b.SEAL_KIND, "schema_version": 1, "manifest_name": b.MANIFEST_NAME,
            "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "manifest_size_bytes": len(data)}
    (root / b.SEAL_NAME).write_bytes(canonical_json_bytes(seal) + b"\n")
    return b.verify_bundle(root)


def test_g1_04_zero_entry_bundle_is_never_a_verified_export(tmp_path):
    report = _hand_sealed_empty_bundle(tmp_path / "export")
    assert report["entries_checked"] == 0 and report["manifest"]["entries"] == []
    problems = ex.check_export_verification(report, report["manifest_sha256"])
    assert "entries_not_all_checked" in problems
    with pytest.raises(ValueError, match="entries_not_all_checked"):
        ex.validate_observation({"kind": "export_verified",
                                 "manifest_hash": report["manifest_sha256"],
                                 "verification": report})
    _, good = _bundle(tmp_path)  # control: one entry, all checked
    assert ex.check_export_verification(good, good["manifest_sha256"]) == []


@pytest.mark.parametrize("change", [{"schema_version": 2}, {"schema_version": "1"},
                                    {"schema_version": True}, {"producer": None},
                                    {"producer": {}}, "drop_producer", "drop_schema_version"])
def test_g1_05_bundle_report_version_and_producer_fail_closed(tmp_path, change):
    _, report = _bundle(tmp_path)
    mutated = copy.deepcopy(report)
    if change == "drop_producer":
        del mutated["producer"]
    elif change == "drop_schema_version":
        del mutated["schema_version"]
    else:
        mutated.update(change)
    assert ex.check_export_verification(mutated, report["manifest_sha256"]) == [
        "verification_shape"]
    assert ex.check_export_verification(report, report["manifest_sha256"]) == []  # control


def _actual_bundle(tmp_path, roles):
    from gate.s2_bundle import seal_bundle, verify_bundle
    from gate.s2_identity import canonical_json_bytes

    root = tmp_path / "actual-bundle"
    root.mkdir()
    entries = []
    for role in roles:
        (root / f"{role}.json").write_bytes(canonical_json_bytes({"role": role}) + b"\n")
        entries.append({"path": f"{role}.json", "role": role, "restriction_class": "internal",
                        "licensed_lineage": False, "replay": None})
    sealed = seal_bundle(root, {"artifact_kind": "s2_bundle_index", "schema_version": 1,
                                "scope": "actual", "bundle_id": "g1-actual",
                                "required_roles": [], "entries": entries})
    return root, sealed, verify_bundle(root)


def test_g1_01_export_report_is_bound_to_the_record_scope(tmp_path):
    from gate.s2_bundle import ACTUAL_MINIMUM_ROLES

    _, synthetic = _bundle(tmp_path)
    assert ex.check_export_binding(synthetic, "synthetic") == []
    assert ex.check_export_binding(synthetic, "actual") == ["export_scope_mismatch"]
    assert ex.check_export_binding(synthetic, "public_source_fixture") == [
        "export_scope_mismatch"]
    assert ex.check_export_binding(envelope(), "synthetic") == []
    assert ex.check_export_binding(envelope(), "actual") == ["actual_export_requires_bundle"]
    assert ex.check_export_binding(envelope(), None) == ["record_scope_invalid"]
    assert ex.check_export_binding({"artifact_kind": "x"}, "synthetic") == [
        "verification_kind_unknown"]
    _root, sealed, actual = _actual_bundle(tmp_path, ACTUAL_MINIMUM_ROLES)
    assert sealed["status"] == "passed", sealed["issues"]
    assert actual["status"] == "passed" and actual["scope"] == "actual"
    assert ex.check_export_binding(actual, "actual") == []
    assert ex.check_export_binding(actual, "synthetic") == ["export_scope_mismatch"]
    short = copy.deepcopy(actual)  # binding re-checks the roles itself (independent of seal)
    short["manifest"]["entries"] = short["manifest"]["entries"][1:]
    assert ex.check_export_binding(short, "actual") == ["actual_export_roles_missing"]


def test_g1_01_coordinator_refuses_a_synthetic_export_for_an_actual_rental(tmp_path):
    rec = record(scope="actual", approval={
        "decision_id": "D-030", "quoted_instruction": "Synthetic fixture instruction."})
    root, synthetic = _bundle(tmp_path)
    coord, provider = _exporting_coordinator(tmp_path, rec)
    observation = {"kind": "export_verified", "manifest_hash": synthetic["manifest_sha256"],
                   "verification": synthetic}
    assert ex.reverify_export(observation, root) == []  # the bytes on disk do re-verify
    with pytest.raises(ValueError, match="export_scope_mismatch"):
        coord.observe(observation, export_root=root)
    with pytest.raises(ValueError, match="actual_export_requires_bundle"):
        coord.observe(exported(), export_root=export_copy(tmp_path))
    assert coord.journal.replay()[1] is None and provider.calls == []
    coord.close()
