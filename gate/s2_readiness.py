"""Evidence-aware S2 readiness report with separate sections (CPU-02).

Tickets: readiness CPU-02; build-spec B2. Serves S2.1/S2.2 [C] (G2, G7) under
D-018/D-024. Agent-authored evaluator engineering.

``inspect_readiness(plan, evidence_index, schedule, action_record=None, *, root)``
returns six independent sections plus ``next_blocking_checks``:

- ``evidence_integrity`` - every evidence-index reference is read under ``root`` with
  ``gate.s2_identity.read_artifact`` (no-follow, size and SHA-256). A match proves the
  bytes are unchanged since the index was written; it never proves authorship
  (``authenticity: not_verified``).
- ``planning_consistency`` - ``gate.s2_runtime.validate_runtime_inventory``; the
  prepared packet (pilot ``gate.s2_packet.build_pilot_packet`` or full sweep
  ``build_full_sweep_packet``) rebuilt from its own inputs and compared unit by unit;
  the supplied schedule equal to the packet's; review receipts whose subject changed.
- ``owner_decisions`` - every phase-required Huy decision is a required field with no
  default. A value without a linked, byte-verified ``owner_record`` is a declaration
  (``declared_unlinked``), not a recorded decision.
- ``scientific_admission`` - ``gate.s2_preflight.validate_s2_preflight``; its
  ``cloud_launch_authorized=False`` semantics are kept. A planning-ready manifest is
  reported ``unverified``: stage admission is Huy's recorded decision and this report
  never states one.
- ``action_scope`` - ``gate.s2_execution.validate_action_record`` (structural pre-create
  check), closed authority (D-019, closed by D-020, is never new authority), linked
  approval/licence/quote evidence, and the unit-run authority bound to the packet's
  request hash and ordered job IDs (re-checked by ``scripts/s2_run_units.py``). The
  authority's ``action_record_ref`` follows the runner's reference rule (recorded,
  non-placeholder, never credential-shaped) and must name the validated action record:
  its approval decision ID as a delimited token, or the record's identity hash. Linked
  approvals and owner records must carry the plan's phase and scope.
- ``host_qualification`` - post-provision only: before a separately approved setup it
  is ``qualify_on_host`` and does not block pre-rental status. A host-smoke receipt
  never satisfies model/renderer qualification; a receipt for another host or phase is
  rejected.

``status`` is ``passed`` only when the five pre-rental sections have no blocking
issue. It means the declared facts are consistent and their bytes unchanged; it is
never an admission, an approval or launch authority (``cloud_launch_authorized`` and
``authorization_authenticated`` are always False).

Verification level: ``inspect_readiness`` L1 (fixtures and negative controls).
Actual approvals, host fit and owner authorship remain outside CPU verification.
This module never writes, launches, rents, accepts terms or chooses an owner value.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

from gate.s2_identity import identity_hash, read_artifact
from gate.s2_runtime import is_placeholder, secret_like

SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
PHASES = ("harmonizer_pilot", "full_sweep")
STATUSES = ("passed", "blocked", "not_run", "not_estimable", "unverified", "not_applicable",
            "qualify_on_host")
SECTIONS = ("evidence_integrity", "planning_consistency", "owner_decisions",
            "scientific_admission", "action_scope", "host_qualification")
PRE_RENTAL_SECTIONS = SECTIONS[:-1]
EVIDENCE_KINDS = ("owner_record", "action_approval", "quote", "license_receipt", "host_receipt",
                  "model_renderer_qualification", "review_receipt", "source_receipt")
REVIEW_SUBJECTS = ("scientific_manifest", "runtime_inventory", "packet", "schedule")
# D-019's bounded host check was completed and closed by D-020 on 2026-11-01.
CLOSED_AUTHORITIES = {"D-019": "completed and closed by D-020 on 2026-11-01"}
# Huy-owned decisions per phase (build-spec section 6, readiness section 7, plan 02 S2.1).
OWNER_DECISIONS = {
    "harmonizer_pilot": (
        "camera_fix", "harmonizer_source", "licence_access_actions", "scene_selection",
        "execution_order", "requested_steps", "prediction_ref", "decision_rule",
        "retry_selection_rule", "missing_data_rule", "terminal_semantics", "seed_mode",
        "evaluated_path_estimand", "rental_scope",
    ),
    "full_sweep": (
        "camera_fix", "harmonizer_source", "licence_access_actions", "scene_list",
        "roads_strata", "execution_order", "replicates", "requested_steps", "prediction_ref",
        "decision_rule", "margins", "reliability_method_floor", "retry_selection_rule",
        "missing_data_rule", "terminal_semantics", "seed_mode", "evaluated_path_estimand",
        "information_boundary", "rental_scope", "remaining_cap_release",
    ),
}
_PLACEHOLDERS = {"tbd", "todo", "later", "____", "placeholder", "xxx", "n/a", "?", ""}
_PLAN_FIELDS = {"artifact_kind", "schema_version", "scope", "phase", "as_of_utc", "host_id",
                "scientific_manifest", "runtime_inventory", "packet", "unit_run_authority",
                "owner_decisions"}
_INDEX_FIELDS = {"artifact_kind", "schema_version", "scope", "entries"}
_ENTRY_FIELDS = {"id", "kind", "path", "sha256", "size_bytes", "scope", "phase", "host_id",
                 "decision_id", "subject", "subject_sha256", "valid_until_utc"}
_DECISION_FIELDS = {"value_ref", "evidence_id"}
_AUTHORITY_FIELDS = {"artifact_kind", "schema_version", "scope", "action_record_ref",
                     "packet_request_sha256", "job_ids"}
_PACKET_KINDS = {"harmonizer_pilot": "s2_pilot_packet", "full_sweep": "s2_full_sweep_packet"}
# Who resolves a blocker: Huy (owner), an agent preparing facts, or the admitted host.
_SECTION_RESOLVER = {"evidence_integrity": "agent", "planning_consistency": "agent",
                     "owner_decisions": "owner", "scientific_admission": "owner",
                     "action_scope": "owner", "host_qualification": "host"}
_CODE_RESOLVER = {"quote_missing": "agent", "quote_expired": "agent",
                  "packet_missing": "agent", "runtime_inventory_blocked": "owner"}


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _issue(issues, code, path, message, severity="blocking", evidence_refs=()):
    issues.append({"code": code, "path": path, "severity": severity,
                   "evidence_refs": list(evidence_refs), "message": message})


def _blocking(issues):
    return any(item["severity"] == "blocking" for item in issues)


def _text(value):
    return type(value) is str and value.strip().casefold() not in _PLACEHOLDERS


def _utc(value):
    if type(value) is not str:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def _hash(value):
    try:
        return identity_hash(value)
    except (TypeError, ValueError, UnicodeError):
        return None


def _section(status="passed", **fields):
    return {"status": status, "issues": [], **fields}


def _close(section, *, blocked_status="blocked"):
    if _blocking(section["issues"]):
        section["status"] = blocked_status
    return section


def _evidence(plan, index, root):
    section = _section(entries=[], authenticity="not_verified")
    issues = section["issues"]
    verified = {}
    if type(index) is not dict or set(index) != _INDEX_FIELDS or index.get(
            "artifact_kind") != "s2_evidence_index" or index.get("schema_version") != 1 or type(
            index.get("entries")) is not list:
        _issue(issues, "invalid_evidence_index", "evidence_index",
               f"Require s2_evidence_index schema 1 with exactly {sorted(_INDEX_FIELDS)}")
        return _close(section), verified
    if index["scope"] != plan["scope"]:
        _issue(issues, "scope_mismatch", "evidence_index.scope",
               "Evidence index scope differs from the plan scope")
    seen = set()
    for position, entry in enumerate(index["entries"]):
        path = f"evidence_index.entries[{position}]"
        if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
            _issue(issues, "invalid_evidence_entry", path,
                   f"Entries have exactly {sorted(_ENTRY_FIELDS)}")
            continue
        if type(entry["id"]) is not str or not entry["id"] or entry["id"] in seen:
            _issue(issues, "invalid_evidence_entry", f"{path}.id", "Unique nonempty id")
            continue
        if type(entry["kind"]) is not str or type(entry["scope"]) is not str:
            _issue(issues, "invalid_evidence_entry", path, "kind and scope are strings")
            continue
        seen.add(entry["id"])
        row = {"id": entry["id"], "kind": entry["kind"], "path": entry["path"],
               "bytes": "invalid_reference", "authenticity": "not_verified"}
        section["entries"].append(row)
        if entry["kind"] not in EVIDENCE_KINDS:
            _issue(issues, "unknown_evidence_kind", f"{path}.kind",
                   f"Kind is one of {list(EVIDENCE_KINDS)}")
            continue
        if entry["scope"] not in SCOPES:
            _issue(issues, "invalid_scope", f"{path}.scope", "Evidence scope must be declared")
            continue
        if plan["scope"] == "actual" and entry["scope"] != "actual":
            _issue(issues, "synthetic_evidence_for_actual_plan", f"{path}.scope",
                   "A synthetic or fixture record cannot satisfy an actual action")
            continue
        try:
            read_artifact(root, {key: entry[key] for key in ("path", "sha256", "size_bytes")})
        except FileNotFoundError:
            row["bytes"] = "missing"
            _issue(issues, "nonexistent_reference", f"{path}.path",
                   "Referenced evidence does not exist under the root")
            continue
        except (OSError, ValueError) as exc:
            changed = "checksum" in str(exc) or "size" in str(exc)
            row["bytes"] = "changed" if changed else "invalid_reference"
            _issue(issues, "changed_bytes" if changed else "invalid_reference", f"{path}.path",
                   f"Evidence bytes not verified: {exc}")
            continue
        if entry["kind"] == "quote" and _utc(entry["valid_until_utc"]) is None:
            _issue(issues, "invalid_evidence_entry", f"{path}.valid_until_utc",
                   "A quote needs a timezone-aware valid_until_utc")
            continue
        row["bytes"] = "verified"
        verified[entry["id"]] = entry
    return _close(section), verified


def _packet_problem(plan, packet):
    if type(packet) is not dict:
        return "packet_missing", "No prepared packet is recorded"
    if packet.get("artifact_kind") != _PACKET_KINDS[plan["phase"]]:
        return "packet_phase_mismatch", f"Phase {plan['phase']} needs a {_PACKET_KINDS[plan['phase']]}"
    if packet.get("prepared") is not True or any(
            type(item) is dict and item.get("severity") == "blocking"
            for item in packet.get("issues") or []):
        return "packet_not_prepared", "The packet is not prepared or carries blocking issues"
    if packet.get("scope") != plan["scope"]:
        return "scope_mismatch", "Packet scope differs from the plan scope"
    if _hash(packet.get("inputs")) != packet.get("request_sha256"):
        return "packet_request_hash_mismatch", "request_sha256 is not the hash of inputs"
    try:
        if plan["phase"] == "harmonizer_pilot":
            from gate.s2_packet import build_pilot_packet as build
        else:
            from gate.s2_packet import build_full_sweep_packet as build
        rebuilt = build(packet["inputs"])
    except Exception as exc:  # noqa: BLE001 - an unbuildable input set is a finding
        return "packet_rebuild_failed", f"Rebuild raised {type(exc).__name__}"
    if (rebuilt.get("prepared") is not True
            or _hash(rebuilt.get("units")) != _hash(packet.get("units"))
            or _hash(rebuilt.get("schedule")) != _hash(packet.get("schedule"))):
        return "packet_rebuild_mismatch", "Units or schedule differ from a rebuild of inputs"
    return None


def _planning(plan, schedule, verified):
    from gate.s2_runtime import validate_runtime_inventory

    section = _section(runtime_inventory=None, packet=None, schedule_sha256=_hash(schedule))
    issues = section["issues"]
    inventory = validate_runtime_inventory(plan["runtime_inventory"])
    section["runtime_inventory"] = {
        "status": inventory["status"], "valid": inventory["valid"],
        "issue_codes": sorted({item["code"] for item in inventory["issues"]})}
    if inventory["status"] != "passed":
        _issue(issues, "runtime_inventory_blocked", "plan.runtime_inventory",
               f"Runtime inventory is {inventory['status']}: "
               f"{section['runtime_inventory']['issue_codes']}")
    elif inventory["scope"] != plan["scope"]:
        _issue(issues, "scope_mismatch", "plan.runtime_inventory.scope",
               "Inventory scope differs from the plan scope")
    packet = plan["packet"]
    problem = _packet_problem(plan, packet)
    if problem:
        _issue(issues, problem[0], "plan.packet", problem[1])
    else:
        section["packet"] = {"artifact_kind": packet["artifact_kind"],
                             "request_sha256": packet["request_sha256"],
                             "unit_count": len(packet["units"]),
                             "schedule_sha256": _hash(packet["schedule"])}
        if _hash(schedule) != _hash(packet["schedule"]):
            _issue(issues, "schedule_mismatch", "schedule",
                   "The supplied schedule differs from the packet's schedule")
    if type(schedule) is not list or not schedule or any(
            type(job) is not dict or job.get("phase") != plan["phase"] for job in schedule):
        _issue(issues, "schedule_phase_mismatch", "schedule",
               f"Every scheduled job must have phase {plan['phase']}")
    subjects = {"scientific_manifest": _hash(plan["scientific_manifest"]),
                "runtime_inventory": _hash(plan["runtime_inventory"]),
                "packet": packet.get("request_sha256") if type(packet) is dict else None,
                "schedule": _hash(schedule)}
    for entry in verified.values():
        if entry["kind"] != "review_receipt":
            continue
        if entry["subject"] not in REVIEW_SUBJECTS:
            _issue(issues, "invalid_review_subject", f"evidence:{entry['id']}",
                   f"Review subject is one of {list(REVIEW_SUBJECTS)}")
        elif entry["subject_sha256"] != subjects[entry["subject"]]:
            _issue(issues, "reviewed_artifact_changed", f"evidence:{entry['id']}",
                   f"The reviewed {entry['subject']} changed after review",
                   evidence_refs=[entry["id"]])
    return _close(section)


def _owner(plan, verified):
    section = _section(decisions={}, authenticity="not_verified")
    issues = section["issues"]
    decisions = plan["owner_decisions"]
    required = OWNER_DECISIONS[plan["phase"]]
    if type(decisions) is not dict or set(decisions) != set(required):
        present = set(decisions) if type(decisions) is dict else set()
        _issue(issues, "invalid_owner_decisions", "plan.owner_decisions",
               f"Owner decisions are exactly {sorted(required)}; missing "
               f"{sorted(set(required) - present)}, unexpected {sorted(present - set(required))}")
        return _close(section)
    for name in required:
        value, path = decisions[name], f"plan.owner_decisions.{name}"
        if value is None:
            section["decisions"][name] = "missing"
            _issue(issues, "owner_decision_missing", path, f"Huy has not recorded {name}")
            continue
        if type(value) is not dict or set(value) != _DECISION_FIELDS or not _text(
                value["value_ref"]):
            section["decisions"][name] = "invalid"
            _issue(issues, "invalid_owner_decision", path,
                   "Decision is {value_ref, evidence_id} with a non-placeholder value_ref")
            continue
        evidence_id = value["evidence_id"]
        evidence = verified.get(evidence_id) if type(evidence_id) is str else None
        if evidence_id is None:
            section["decisions"][name] = "declared_unlinked"
            _issue(issues, "owner_decision_unlinked", path,
                   "A typed value is a declaration until linked to an inspected owner record")
        elif evidence is None or evidence["kind"] != "owner_record":
            section["decisions"][name] = "owner_record_unverified"
            _issue(issues, "owner_record_unverified", path,
                   "evidence_id is not a byte-verified owner_record in the evidence index")
        elif evidence["phase"] != plan["phase"] or evidence["scope"] != plan["scope"]:
            # Review B2-06: a record for another phase or scope is not this decision.
            section["decisions"][name] = "owner_record_scope_mismatch"
            _issue(issues, "owner_record_scope_mismatch", path,
                   "The linked owner_record is for another phase or scope",
                   evidence_refs=[evidence_id])
        else:
            section["decisions"][name] = "linked_bytes_verified"
    return _close(section)


def _scientific(plan):
    from gate.s2_preflight import validate_s2_preflight

    report = validate_s2_preflight(plan["scientific_manifest"])
    section = _section("unverified", phase=report["phase"], cloud_launch_authorized=False,
                       attestation_authenticity=report["attestation_authenticity"],
                       warnings=sorted({item["code"] for item in report["warnings"]}),
                       note="Planning consistency only; stage admission is Huy's recorded "
                            "decision and is never reported here")
    for item in report["blocking_reasons"]:
        _issue(section["issues"], item["code"], "plan.scientific_manifest." + item["path"],
               item["message"])
    if report["phase"] != plan["phase"]:
        _issue(section["issues"], "scientific_phase_mismatch", "plan.scientific_manifest.phase",
               "The scientific manifest is for another phase")
    return _close(section)


def _closed(value):
    return [name for name in CLOSED_AUTHORITIES if type(value) is str and name in value]


def _action(plan, action_record, verified):
    from gate.s2_execution import validate_action_record

    section = _section(action_record=None, unit_run_authority=None, quote=None,
                       authenticated=False)
    issues = section["issues"]
    if action_record is None:
        section["status"] = "not_run"
        _issue(issues, "action_record_missing", "action_record",
               "No rental action record: provider, offer, resources, window, caps and approval "
               "are recorded by Huy before any create")
    else:
        report = validate_action_record(action_record)
        section["action_record"] = {
            "status": report["status"], "record_sha256": report["input_hashes"]["record"],
            "issue_codes": sorted({item["code"] for item in report["issues"]})}
        for item in report["issues"]:
            _issue(issues, item["code"], "action_record." + item["path"], item["message"])
        record = action_record if type(action_record) is dict else {}
        if record.get("scope") != plan["scope"]:
            code = ("synthetic_action_for_actual_plan" if plan["scope"] == "actual"
                    else "scope_mismatch")
            _issue(issues, code, "action_record.scope", "Action record scope differs from plan")
        approval = record.get("approval") if type(record.get("approval")) is dict else {}
        decision = approval.get("decision_id")
        for name in _closed(decision):
            _issue(issues, "closed_authority_reused", "action_record.approval.decision_id",
                   f"{name} was {CLOSED_AUTHORITIES[name]}; it is not new authority")
        linked = [entry for entry in verified.values()
                  if entry["kind"] == "action_approval" and type(decision) is str
                  and entry["decision_id"] == decision]
        if not linked:
            _issue(issues, "approval_not_linked", "action_record.approval",
                   "No byte-verified action_approval record carries this decision ID")
        elif not any(entry["phase"] == plan["phase"] and entry["scope"] == plan["scope"]
                     for entry in linked):
            # Review B2-06: an approval for another phase or scope does not link.
            _issue(issues, "approval_scope_mismatch", "action_record.approval",
                   "The action_approval with this decision ID is for another phase or scope",
                   evidence_refs=sorted(entry["id"] for entry in linked))
        refs = record.get("license_attestation_refs")
        for ref in refs if type(refs) is list else []:
            entry = verified.get(ref)
            if entry is None or entry["kind"] != "license_receipt":
                _issue(issues, "license_attestation_unlinked", "action_record.license_"
                       "attestation_refs", f"{ref!r} is not a byte-verified license_receipt")
    as_of = _utc(plan["as_of_utc"])
    quotes = [entry for entry in verified.values() if entry["kind"] == "quote"
              and entry["phase"] == plan["phase"]]
    if not quotes:
        _issue(issues, "quote_missing", "evidence_index",
               "No byte-verified quote for this phase")
    else:
        current = [entry for entry in quotes if _utc(entry["valid_until_utc"]) > as_of]
        section["quote"] = {"current": sorted(entry["id"] for entry in current),
                            "expired": sorted(entry["id"] for entry in quotes
                                              if entry not in current)}
        if not current:
            _issue(issues, "quote_expired", "evidence_index",
                   "Every recorded quote expired before as_of_utc; re-quote before rental")
    _authority(plan, issues, section, action_record)
    return _close(section, blocked_status="not_run" if section["status"] == "not_run"
                  else "blocked")


def _names_record(ref, action_record):
    """True when ``ref`` names the action record (decision-ID token or record hash)."""
    if type(ref) is not str or type(action_record) is not dict:
        return False
    if ref == _hash(action_record):
        return True
    approval = action_record.get("approval")
    decision = approval.get("decision_id") if type(approval) is dict else None
    if type(decision) is not str or not decision.strip():
        return False
    token = r"(?<![A-Za-z0-9])" + re.escape(decision.strip()) + r"(?![A-Za-z0-9])"
    return re.search(token, ref) is not None


def _authority(plan, issues, section, action_record=None):
    authority, packet = plan["unit_run_authority"], plan["packet"]
    if authority is None:
        _issue(issues, "unit_run_authority_missing", "plan.unit_run_authority",
               "No s2_unit_run_authority binds a recorded action to this packet")
        return
    if type(authority) is not dict or set(authority) != _AUTHORITY_FIELDS or authority.get(
            "artifact_kind") != "s2_unit_run_authority" or authority.get("schema_version") != 1:
        _issue(issues, "invalid_unit_run_authority", "plan.unit_run_authority",
               f"Require s2_unit_run_authority schema 1 with {sorted(_AUTHORITY_FIELDS)}")
        return
    ref = authority["action_record_ref"]
    if not _text(ref) or is_placeholder(ref) or secret_like(ref):
        # Review B2-06: the runner's rule (scripts/s2_run_units.validate_authority).
        _issue(issues, "authority_unrecorded", "plan.unit_run_authority.action_record_ref",
               "A recorded, non-placeholder, non-credential action reference is required")
    elif action_record is not None and not _names_record(ref, action_record):
        _issue(issues, "authority_action_record_mismatch",
               "plan.unit_run_authority.action_record_ref",
               "The reference names neither the action record's approval decision ID nor "
               "its identity hash")
    for name in _closed(ref):
        _issue(issues, "closed_authority_reused", "plan.unit_run_authority.action_record_ref",
               f"{name} was {CLOSED_AUTHORITIES[name]}; it is not new authority")
    if plan["scope"] == "actual" and (authority["scope"] != "actual" or (
            type(ref) is str and "synthetic" in ref.casefold())):
        _issue(issues, "synthetic_authority_for_actual_plan", "plan.unit_run_authority",
               "A synthetic authority fixture never satisfies an actual action")
    units = packet.get("units") if type(packet) is dict and type(packet.get("units")) is list \
        else []
    ordered = [unit["job"].get("job_id") if type(unit) is dict and type(unit.get("job")) is dict
               else None for unit in units]
    binding = {
        "scope": authority["scope"] == plan["scope"],
        "packet_request_sha256": type(packet) is dict
        and authority["packet_request_sha256"] == packet.get("request_sha256"),
        "job_ids": authority["job_ids"] == ordered and bool(ordered),
    }
    section["unit_run_authority"] = binding
    for name, matches in binding.items():
        if not matches:
            _issue(issues, "authority_packet_mismatch", f"plan.unit_run_authority.{name}",
                   f"Authority {name} does not bind this packet")


def _host(plan, verified):
    section = _section("qualify_on_host", timing="post_provision",
                       blocks_pre_rental_status=False, receipts=[],
                       note="Compatibility is measured only on a delivered host after a "
                            "separately approved bounded setup")
    issues = section["issues"]
    for entry in sorted(verified.values(), key=lambda item: item["id"]):
        if entry["kind"] not in ("host_receipt", "model_renderer_qualification"):
            continue
        path = f"evidence:{entry['id']}"
        if entry["host_id"] != plan["host_id"] or entry["phase"] != plan["phase"] or (
                plan["host_id"] is None):
            _issue(issues, "host_receipt_scope_mismatch", path,
                   "Receipt names another host or phase (or no host is bound yet)",
                   evidence_refs=[entry["id"]])
            continue
        section["receipts"].append(entry["id"])
        if entry["kind"] == "host_receipt":
            _issue(issues, "host_smoke_not_model_renderer_qualification", path,
                   "A host-smoke receipt cannot satisfy teacher/renderer/Harmonizer "
                   "qualification", severity="info", evidence_refs=[entry["id"]])
        elif entry["scope"] == "actual":
            section["status"] = "unverified"
            section["note"] = ("A model/renderer qualification receipt is byte-verified; its "
                               "acceptance is a host review, not this report")
    if _blocking(issues):
        section["status"] = "blocked"
    return section


def _plan_problem(plan):
    if type(plan) is not dict or set(plan) != _PLAN_FIELDS:
        present = set(plan) if type(plan) is dict else set()
        return (f"Plan fields are exactly {sorted(_PLAN_FIELDS)}; missing "
                f"{sorted(_PLAN_FIELDS - present)}, unexpected {sorted(present - _PLAN_FIELDS)}")
    if plan["artifact_kind"] != "s2_readiness_plan" or plan["schema_version"] != 1 or type(
            plan["schema_version"]) is not int:
        return "Require s2_readiness_plan schema 1"
    if plan["scope"] not in SCOPES or type(plan["scope"]) is not str:
        return f"scope is one of {list(SCOPES)}"
    if plan["phase"] not in PHASES or type(plan["phase"]) is not str:
        return f"phase is one of {list(PHASES)}"
    if _utc(plan["as_of_utc"]) is None:
        return "as_of_utc must be a timezone-aware ISO 8601 time"
    if plan["host_id"] is not None and not _text(plan["host_id"]):
        return "host_id is null before provisioning or a recorded host identifier"
    if _hash(plan) is None:
        return "Plan must be strict JSON"
    return None


def inspect_readiness(plan, evidence_index, schedule, action_record=None, *, root):
    """Return the CPU-02 readiness report; deterministic for identical inputs and bytes."""
    report = {
        "artifact_kind": "s2_readiness_report", "schema_version": SCHEMA_VERSION,
        "scope": None,
        "producer": {"module": "gate.s2_readiness", "function": "inspect_readiness",
                     "source_sha256": _SOURCE_SHA256},
        "input_hashes": {"plan": _hash(plan), "evidence_index": _hash(evidence_index),
                         "schedule": _hash(schedule), "action_record": _hash(action_record)},
        "units": {"time": "UTC ISO 8601"}, "status": "blocked", "issues": [],
        "phase": None, "sections": {}, "next_blocking_checks": [],
        "cloud_launch_authorized": False, "authorization_authenticated": False,
        "meaning": "Declared facts consistent and referenced bytes unchanged; never an "
                   "admission, approval or launch authority",
    }
    problem = _plan_problem(plan)
    if problem:
        _issue(report["issues"], "invalid_plan", "plan", problem)
        return report
    report["scope"], report["phase"] = plan["scope"], plan["phase"]
    evidence, verified = _evidence(plan, evidence_index, Path(root))
    sections = {
        "evidence_integrity": evidence,
        "planning_consistency": _planning(plan, schedule, verified),
        "owner_decisions": _owner(plan, verified),
        "scientific_admission": _scientific(plan),
        "action_scope": _action(plan, action_record, verified),
        "host_qualification": _host(plan, verified),
    }
    report["sections"] = sections
    checks, seen = [], set()
    for name in SECTIONS:
        for item in sections[name]["issues"]:
            key = (name, item["code"], item["path"])
            if item["severity"] != "blocking" or key in seen:
                continue
            seen.add(key)
            checks.append({"section": name, "code": item["code"], "path": item["path"],
                           "resolver": _CODE_RESOLVER.get(item["code"],
                                                          _SECTION_RESOLVER[name]),
                           "blocks_pre_rental_status": name in PRE_RENTAL_SECTIONS})
    report["next_blocking_checks"] = checks
    if not any(check["blocks_pre_rental_status"] for check in checks):
        report["status"] = "passed"
    return report

