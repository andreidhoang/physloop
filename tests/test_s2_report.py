"""Known answers and negative controls for gate.s2_report (CPU-13; build-spec B2; F4).

Evidence references, readiness and bundle reports here are invented fixtures. The bundle
fixture is the shape ``gate.s2_bundle.verify_bundle`` returns: its ``manifest`` lists every
cited row's path and sha256, and ``manifest_sha256`` is the SHA-256 of the canonical
manifest bytes plus a newline, exactly what ``seal_bundle`` writes. One test seals and
verifies a real bundle in a temporary directory, so the binding is checked against the
producer of record rather than against the fixture alone.
"""

import copy
import hashlib

import pytest

from gate.s2_bundle import seal_bundle, verify_bundle
from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.s2_report import ITEMS, MILESTONES, build_s2_report, render_markdown

PRODUCER = {"module": "gate.synthetic_producer", "source_sha256": "b" * 64}


def evidence_row(kind, status="passed", scope="synthetic", blockers=(), producer=PRODUCER):
    row = {"path": f"outputs/{kind}.json", "sha256": identity_hash(kind), "artifact_kind": kind,
           "status": status, "scope": scope, "blockers": list(blockers)}
    if producer is not None:
        row["producer"] = dict(producer)
    return row


def all_evidence(scope="synthetic"):
    keys = {key for spec in ITEMS.values() for key in spec["evidence"]}
    return {key: evidence_row(key, scope=scope) for key in sorted(keys)}


def manifest_for(rows, scope="synthetic"):
    entries = sorted(({"path": row["path"], "size_bytes": 1, "sha256": row["sha256"],
                       "role": "analysis_output", "restriction_class": "internal",
                       "licensed_lineage": False, "replay": None} for row in rows),
                     key=lambda entry: entry["path"])
    return {"artifact_kind": "s2_bundle_manifest", "schema_version": 1, "scope": scope,
            "bundle_id": "synthetic-report-test", "index_sha256": "c" * 64, "entries": entries,
            "publication_class": "internal", "publication_authorized": False,
            "authenticity": "not_verified", "units": {"size_bytes": "bytes"}}


def bundle(verified=True, rows=None, scope="synthetic"):
    manifest = manifest_for(all_evidence(scope).values() if rows is None else rows, scope)
    return {"artifact_kind": "s2_bundle_verification", "schema_version": 1,
            "status": "passed" if verified else "blocked",
            "seal_state": "sealed" if verified else "unsealed", "verified": verified,
            "manifest": manifest,
            "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest) + b"\n").hexdigest()}


def rehearsal(**changes):
    value = {"scope": "synthetic", "passed": True, "replay_identical": True,
             "negative_controls": {"missing_artifact": True, "corrupt_export": True},
             "counts": {"pilot_jobs": 10, "full_sweep_units": 2646}}
    value.update(changes)
    return value


def readiness(scope="synthetic", status="passed", checks=()):
    return {"artifact_kind": "s2_readiness_report", "schema_version": 1, "scope": scope,
            "status": status,
            "sections": {"owner_decisions": {"status": "blocked"},
                         "action_scope": {"status": "not_run"}},
            "next_blocking_checks": list(checks)}


def report(**changes):
    arguments = {"scope": "synthetic", "evidence": all_evidence(), "readiness": readiness(),
                 "bundle_verification": bundle(), "rehearsal": rehearsal()}
    arguments.update(changes)
    return build_s2_report(**arguments)


def actual_report(**changes):
    arguments = {"scope": "actual", "evidence": all_evidence("actual"),
                 "readiness": readiness(scope="actual"),
                 "bundle_verification": bundle(scope="actual"),
                 "rehearsal": rehearsal(scope="actual")}
    arguments.update(changes)
    return build_s2_report(**arguments)


def test_every_card_item_is_mapped_with_owner_inputs_unfilled():
    result = report()
    assert result["status"] == "passed"
    assert list(result["items"]) == ["S2.1", "S2.2", "S2.3", "S2.4", "S2.5", "S2.6", "S2.7"]
    for value in result["items"].values():
        assert value["engineering_status"] == "passed"
        assert value["unverified_reasons"] == []
        assert value["scientific_status"] == "blocked"
        assert value["scientific_blocker"] == "owner_inputs_unregistered"
        assert value["owner_inputs"] and value["owner_inputs_filled_by_agent"] is False
    assert result["scientific_verdict"] is None
    assert result["cloud_launch_authorized"] is False
    assert result["evidence_binding"]["bound"] == len(all_evidence())
    assert result["evidence_binding"]["unverified"] == 0


def test_cpu_rehearsal_is_the_highest_label_reachable_on_cpu():
    milestones = report()["milestones"]
    assert list(milestones) == list(MILESTONES)
    assert {name: value["status"] for name, value in milestones.items()} == {
        "CPU_REHEARSED": "passed", "PACKET_COMPLETE": "not_applicable",
        "HOST_QUALIFIED": "qualify_on_host", "ACTUAL_INPUT_QUALIFIED": "qualify_on_host",
        "S2_ACCEPTED": "not_run", "S2_STOPPED_OR_INSUFFICIENT": "not_run"}


def test_s2_accepted_is_unreachable_even_with_actual_scope_inputs():
    result = actual_report()
    statuses = {name: value["status"] for name, value in result["milestones"].items()}
    assert statuses["PACKET_COMPLETE"] == "passed"
    assert statuses["CPU_REHEARSED"] == "not_applicable"
    assert statuses["S2_ACCEPTED"] == "not_run"
    assert statuses["HOST_QUALIFIED"] == "qualify_on_host"
    assert all(value["scientific_status"] == "unverified" for value in result["items"].values())
    assert all(value["engineering_status"] == "passed" for value in result["items"].values())


@pytest.mark.parametrize("changes", [
    {"rehearsal": rehearsal(passed=False)},
    {"rehearsal": rehearsal(replay_identical=False)},
    {"rehearsal": rehearsal(negative_controls={"missing_artifact": False})},
    {"rehearsal": rehearsal(negative_controls={})},
    {"bundle_verification": bundle(verified=False)},
    {"bundle_verification": None},
])
def test_cpu_rehearsed_needs_every_control_replay_and_sealed_bundle(changes):
    assert report(**changes)["milestones"]["CPU_REHEARSED"]["status"] == "blocked"


def test_labels_integrity_and_authorization_stay_separate():
    result = report(bundle_verification=bundle(verified=False))
    assert result["evidence_integrity"]["verified"] is False
    assert result["evidence_integrity"]["authenticity"] == "not_verified"
    assert result["human_authorization"] == {
        "status": "not_authenticated", "readiness_scope": "synthetic", "owner_decisions": "blocked",
        "action_scope": "not_run", "cloud_launch_authorized": False}
    # F4: an unverified bundle verifies no evidence, so no item can read passed
    assert result["items"]["S2.5"]["engineering_status"] == "unverified"
    assert result["items"]["S2.5"]["unverified_reasons"] == [
        "population: bundle_not_verified", "validity: bundle_not_verified"]


def test_a_passed_row_listing_blockers_never_reads_passed():
    """Review F4-06: a row declared passed while listing blocker codes is self-contradictory,
    so its item reads unverified with a stable reason; the declared status is kept."""
    evidence = all_evidence()
    evidence["telemetry"] = evidence_row("telemetry",
                                         blockers=["stabilization_rule_unregistered"])
    result = report(evidence=evidence)
    item = result["items"]["S2.7"]
    assert item["engineering_status"] == "unverified"
    assert item["engineering_blockers"] == ["stabilization_rule_unregistered"]
    assert item["unverified_reasons"] == ["telemetry: declared_passed_with_blockers"]
    assert [row["status"] for row in item["evidence"]] == ["passed", "passed"]
    assert result["evidence_binding"]["bound"] == len(evidence)  # binding is unaffected
    assert result["items"]["S2.6"]["engineering_status"] == "passed"  # control
    # a blocked row with the same codes still reads blocked (precedence unchanged)
    evidence["telemetry"] = evidence_row("telemetry", "blocked",
                                         blockers=["stabilization_rule_unregistered"])
    assert report(evidence=evidence)["items"]["S2.7"]["engineering_status"] == "blocked"


def test_item_status_follows_its_own_evidence():
    evidence = all_evidence()
    evidence["repeats"] = evidence_row("repeats", "blocked", blockers=["method_unregistered"])
    del evidence["telemetry"]
    result = report(evidence=evidence)
    assert result["items"]["S2.4"]["engineering_status"] == "blocked"
    assert result["items"]["S2.4"]["engineering_blockers"] == ["method_unregistered"]
    assert result["items"]["S2.7"]["engineering_status"] == "not_run"
    assert result["items"]["S2.7"]["missing_evidence"] == ["telemetry"]
    assert result["items"]["S2.6"]["engineering_status"] == "passed"
    unverified = all_evidence()
    unverified["population"] = evidence_row("population", "unverified")
    assert report(evidence=unverified)["items"]["S2.5"]["engineering_status"] == "unverified"
    assert report(evidence={})["items"]["S2.1"]["engineering_status"] == "not_run"


# --- F4: an item reads passed only on evidence a verified sealed bundle carries ----------------


def test_auditor_probe_sha_absent_from_the_verified_bundle_never_passes_s2_7():
    """Completeness audit 'CPU-13 report': a sha256 absent from a verified bundle gave passed."""
    evidence = all_evidence()
    evidence["telemetry"] = {**evidence["telemetry"], "sha256": "d" * 64}
    result = report(evidence=evidence)
    item = result["items"]["S2.7"]
    assert item["engineering_status"] == "unverified"
    assert item["unverified_reasons"] == ["telemetry: evidence_not_in_verified_bundle"]
    row = next(row for row in item["evidence"] if row["key"] == "telemetry")
    assert row["binding"] == {"status": "unverified",
                              "reasons": ["evidence_not_in_verified_bundle"]}
    assert row["status"] == "passed"  # the declared status is kept, never promoted
    assert result["items"]["S2.2"]["engineering_status"] == "passed"  # cost row still bound
    assert result["evidence_binding"] == {
        "bound": len(evidence) - 1, "unverified": 1,
        "rule": ("a row is bound only when a passed, sealed verify_bundle manifest lists its "
                 "path with its sha256 and the row names its producer")}


def test_same_bytes_cited_under_another_path_are_not_bound():
    evidence = all_evidence()
    evidence["telemetry"] = {**evidence["telemetry"], "path": "outputs/other.json"}
    item = report(evidence=evidence)["items"]["S2.7"]
    assert item["unverified_reasons"] == ["telemetry: evidence_path_mismatch"]


def test_missing_producer_identity_reads_unverified():
    evidence = all_evidence()
    evidence["power"] = evidence_row("power", producer=None)
    item = report(evidence=evidence)["items"]["S2.6"]
    assert item["engineering_status"] == "unverified"
    assert item["unverified_reasons"] == ["power: producer_identity_missing"]


def test_without_a_bundle_verification_nothing_is_bound():
    result = report(bundle_verification=None)
    assert {value["engineering_status"] for value in result["items"].values()} == {"unverified"}
    assert result["items"]["S2.6"]["unverified_reasons"] == [
        "gate: no_bundle_verification", "power: no_bundle_verification"]
    assert result["evidence_binding"]["bound"] == 0


def test_bundle_of_another_scope_binds_nothing():
    result = actual_report(bundle_verification=bundle(scope="synthetic",
                                                      rows=all_evidence("actual").values()))
    assert result["items"]["S2.3"]["engineering_status"] == "unverified"
    assert set(result["items"]["S2.3"]["unverified_reasons"]) == {
        "actual_inspection: bundle_scope_mismatch", "capture: bundle_scope_mismatch",
        "ingestion: bundle_scope_mismatch"}


def test_blocked_and_missing_take_precedence_over_unverified():
    evidence = all_evidence()
    evidence["repeats"] = evidence_row("repeats", "blocked", blockers=["method_unregistered"],
                                       producer=None)
    del evidence["telemetry"]
    result = report(evidence=evidence, bundle_verification=None)
    assert result["items"]["S2.4"]["engineering_status"] == "blocked"
    assert result["items"]["S2.7"]["engineering_status"] == "not_run"


@pytest.mark.parametrize("producer", [
    {"module": "", "source_sha256": None},
    {"module": "gate/s2_power.py", "source_sha256": None},
    {"module": "gate.s2_power", "source_sha256": "XYZ"},
    {"module": "gate.s2_power"},
    {"module": "gate.s2_power", "source_sha256": None, "extra": 1},
    "gate.s2_power",
])
def test_malformed_producer_identity_is_invalid_evidence(producer):
    evidence = all_evidence()
    evidence["power"] = {**evidence["power"], "producer": producer}
    result = report(evidence=evidence)
    assert result["status"] == "blocked" and result["items"] == {}
    assert "invalid_evidence" in {item["code"] for item in result["issues"]}


def test_producer_without_source_hash_and_with_function_is_accepted():
    evidence = all_evidence()
    evidence["power"] = evidence_row("power", producer={
        "module": "gate.s2_power", "function": "run_sensitivity", "source_sha256": None})
    result = report(evidence=evidence)
    assert result["items"]["S2.6"]["engineering_status"] == "passed"
    row = next(row for row in result["items"]["S2.6"]["evidence"] if row["key"] == "power")
    assert row["producer"]["function"] == "run_sensitivity"


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(manifest=None),
    lambda value: value.pop("manifest"),
    lambda value: value["manifest"].update(entries="x"),
    lambda value: value["manifest"].update(artifact_kind="other"),
    lambda value: value["manifest"]["entries"][0].update(sha256="nothex"),
    # an edited manifest no longer recomputes to the sealed manifest hash
    lambda value: value["manifest"]["entries"][0].update(sha256="e" * 64),
    lambda value: value.update(manifest_sha256="f" * 64),
])
def test_verified_bundle_without_its_sealed_manifest_fails_closed(mutate):
    value = bundle()
    mutate(value)
    result = report(bundle_verification=value)
    assert result["status"] == "blocked" and result["items"] == {}
    assert "invalid_bundle_verification" in {item["code"] for item in result["issues"]}


def test_real_sealed_bundle_binds_its_rows(tmp_path):
    """Oracle: gate.s2_bundle seals and verifies real files; rows cite their file sha256."""
    root = tmp_path / "bundle"
    (root / "outputs").mkdir(parents=True)
    keys = sorted({key for spec in ITEMS.values() for key in spec["evidence"]})
    evidence, entries = {}, []
    for key in keys:
        data = canonical_json_bytes({"artifact_kind": key, "status": "passed"}) + b"\n"
        (root / "outputs" / f"{key}.json").write_bytes(data)
        entries.append({"path": f"outputs/{key}.json", "role": "analysis_output",
                        "restriction_class": "public_safe", "licensed_lineage": False,
                        "replay": None})
        evidence[key] = {**evidence_row(key), "sha256": hashlib.sha256(data).hexdigest()}
    index = {"artifact_kind": "s2_bundle_index", "schema_version": 1, "scope": "synthetic",
             "bundle_id": "report-oracle", "required_roles": [], "entries": entries}
    assert seal_bundle(root, index)["sealed"] is True
    verified = verify_bundle(root)
    assert verified["verified"] is True
    result = report(evidence=evidence, bundle_verification=verified)
    assert result["status"] == "passed", result["issues"]
    assert {value["engineering_status"] for value in result["items"].values()} == {"passed"}
    # the auditor's probe against the real bundle
    evidence["telemetry"]["sha256"] = "0" * 64
    probe = report(evidence=evidence, bundle_verification=verified)["items"]["S2.7"]
    assert probe["engineering_status"] == "unverified"
    # a file changed after sealing: verify_bundle blocks, so nothing is bound
    (root / "outputs" / "cost.json").write_bytes(b"changed\n")
    changed = verify_bundle(root)
    assert changed["verified"] is False
    blocked = report(evidence=evidence, bundle_verification=changed)
    assert {value["engineering_status"] for value in blocked["items"].values()} == {"unverified"}


def test_next_actions_group_readiness_blockers_by_resolver():
    checks = [{"section": "owner_decisions", "code": "owner_decision_missing",
               "path": "plan.owner_decisions.camera_fix", "resolver": "owner",
               "blocks_pre_rental_status": True},
              {"section": "action_scope", "code": "quote_missing", "path": "evidence_index",
               "resolver": "agent", "blocks_pre_rental_status": True}]
    actions = report(readiness=readiness(status="blocked", checks=checks))["next_actions"]
    assert actions == {"owner": [("owner_decisions: owner_decision_missing at "
                                 "plan.owner_decisions.camera_fix")],
                       "agent": ["action_scope: quote_missing at evidence_index"], "host": []}


@pytest.mark.parametrize("changes, code", [
    ({"scope": "pilot"}, "invalid_scope"),
    ({"evidence": {"unknown": evidence_row("unknown")}}, "invalid_evidence"),
    ({"evidence": {"power": {**evidence_row("power"), "status": "GO"}}}, "invalid_evidence"),
    ({"evidence": {"power": {**evidence_row("power"), "sha256": "x"}}}, "invalid_evidence"),
    ({"scope": "actual", "evidence": all_evidence(), "readiness": readiness("actual")},
     "synthetic_evidence_in_actual_report"),
    ({"rehearsal": {"passed": True}}, "invalid_rehearsal"),
    ({"readiness": {"artifact_kind": "other"}}, "invalid_readiness"),
    ({"bundle_verification": {"verified": True}}, "invalid_bundle_verification"),
    ({"evidence": {"power": float("nan")}}, "invalid_input"),
])
def test_invalid_inputs_block_without_items(changes, code):
    result = report(**changes)
    assert result["status"] == "blocked" and result["items"] == {}
    assert code in {item["code"] for item in result["issues"]}


def test_report_is_deterministic_and_renders():
    first, second = report(), report(evidence=copy.deepcopy(all_evidence()))
    assert identity_hash(first) == identity_hash(second)
    text = render_markdown(first)
    assert "| CPU_REHEARSED | passed |" in text
    assert "| S2_ACCEPTED | not_run |" in text
    assert "| S2.4 | Frozen-teacher reliability and controls | passed | blocked |" in text
    assert "full_sweep_units: 2646" in text
    assert "Huy-owned inputs (not filled by agents)" in text
    assert "- none" in text.split("## Unverified evidence")[1]
    rejected = render_markdown(report(scope="pilot"))
    assert "Report inputs were rejected" in rejected
    evidence = all_evidence()
    evidence["telemetry"] = {**evidence["telemetry"], "sha256": "d" * 64}
    probe = render_markdown(report(evidence=evidence))
    assert "| S2.7 | Cost and necessary telemetry | unverified | blocked |" in probe
    assert "- S2.7: telemetry: evidence_not_in_verified_bundle" in probe


PRE_RENTAL_CHECK = {"section": "owner_decisions", "code": "owner_decision_missing",
                    "path": "plan.owner_decisions.camera_fix", "resolver": "owner",
                    "blocks_pre_rental_status": True}


def _changed(value, **changes):
    value = copy.deepcopy(value)
    value.update(changes)
    return value


@pytest.mark.parametrize("value", [
    _changed(readiness(scope="actual"), schema_version=99),
    _changed(readiness(scope="actual"), schema_version=True),
    readiness(scope="actual", checks=[PRE_RENTAL_CHECK]),
    _changed(readiness(scope="actual"), status="GO"),
    _changed(readiness(scope="actual"), scope="ACTUAL"),
    _changed(readiness(scope="actual"), sections=[1]),
    _changed(readiness(scope="actual"), sections={"owner_decisions": "blocked"}),
    _changed(readiness(scope="actual"), next_blocking_checks=["x"]),
    _changed(readiness(scope="actual"), next_blocking_checks={"a": 1}),
])
def test_inconsistent_or_unknown_readiness_never_completes_the_packet(value):
    """Review B2-05/B2-07: unknown versions and passed-with-blockers fail closed."""
    result = actual_report(readiness=value)
    assert result["status"] == "blocked" and result["milestones"] == {}
    assert "invalid_readiness" in {item["code"] for item in result["issues"]}


@pytest.mark.parametrize("value", [
    _changed(bundle(), seal_state="unsealed", status="blocked"),
    _changed(bundle(), seal_state="unsealed"),
    _changed(bundle(), status="blocked"),
    _changed(bundle(), schema_version=99),
    _changed(bundle(verified=False), status="passed"),
    _changed(bundle(), verified="yes"),
])
def test_inconsistent_bundle_verification_never_rehearses(value):
    """Review B2-05: verified only with a sealed, passed verification of schema 1."""
    result = report(bundle_verification=value)
    assert result["status"] == "blocked" and result["milestones"] == {}
    assert "invalid_bundle_verification" in {item["code"] for item in result["issues"]}


def test_consistent_blocked_readiness_is_accepted_as_input():
    result = actual_report(readiness=readiness(scope="actual", status="blocked",
                                               checks=[PRE_RENTAL_CHECK]))
    assert result["status"] == "passed"
    assert result["milestones"]["PACKET_COMPLETE"]["status"] == "blocked"


def test_readiness_tool_rejection_output_is_accepted_as_blocked_input(tmp_path):
    """inspect_readiness's own fail-closed output (scope null) stays a usable input."""
    from gate.s2_readiness import inspect_readiness

    rejected = inspect_readiness({"not": "a plan"}, None, None, root=tmp_path)
    assert rejected["scope"] is None and rejected["status"] == "blocked"
    result = report(readiness=rejected)
    assert result["status"] == "passed"
    assert result["milestones"]["PACKET_COMPLETE"]["status"] == "not_applicable"
    assert _changed(readiness(), scope=None)["status"] == "passed"
    assert report(readiness=_changed(readiness(), scope=None))["status"] == "blocked"


# ------------------------------------------------------------ F4-02 milestone binding

def test_cpu_rehearsed_requires_every_cited_row_bound():
    assert report()["milestones"]["CPU_REHEARSED"] == {"status": "passed", "reason": None}
    evidence = all_evidence()
    evidence["power"] = evidence_row("power", producer=None)
    label = report(evidence=evidence)["milestones"]["CPU_REHEARSED"]
    assert label == {"status": "blocked", "reason": (
        "evidence not bound to the verified bundle: power (producer_identity_missing)")}


def test_cpu_rehearsed_blocks_on_bytes_absent_from_the_bundle():
    evidence = all_evidence()
    evidence["telemetry"] = {**evidence["telemetry"], "sha256": "f" * 64}
    label = report(evidence=evidence)["milestones"]["CPU_REHEARSED"]
    assert label["status"] == "blocked"
    assert "telemetry (evidence_not_in_verified_bundle)" in label["reason"]


def test_cpu_rehearsed_needs_cited_evidence():
    label = report(evidence={})["milestones"]["CPU_REHEARSED"]
    assert label == {"status": "blocked", "reason": "no evidence row is cited"}


def test_cpu_rehearsed_lists_controls_and_unbound_rows_together():
    evidence = all_evidence()
    evidence["gate"] = evidence_row("gate", producer=None)
    label = report(evidence=evidence, rehearsal=rehearsal(
        negative_controls={"missing_artifact": False}))["milestones"]["CPU_REHEARSED"]
    assert label["status"] == "blocked"
    assert label["reason"] == (
        "rehearsal/replay/bundle incomplete; controls not rejected: ['missing_artifact']; "
        "evidence not bound to the verified bundle: gate (producer_identity_missing)")
