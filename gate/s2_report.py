"""S2 card report: items S2.1-S2.7 mapped to evidence, status, owner inputs and blockers.

Tickets: readiness CPU-13 (one-command rehearsal and S2 report); build-spec B2. Serves
S2.1-S2.7 [C] (G1, G2, G3-R1, G7) under D-018/D-024. Agent-authored evaluator
engineering.

``build_s2_report`` keeps four things in separate fields, never merged into one verdict:

- ``items`` - every S2 card item with its evidence references, an ``engineering_status``
  computed from those artifacts, a ``scientific_status`` (always ``blocked`` on
  synthetic or fixture scope and ``unverified`` on actual scope: interpretation and the
  registered inputs are Huy's) and the Huy-owned inputs plan 02 names for the item.
- ``milestones`` - the readiness-spec labels ``CPU_REHEARSED``, ``PACKET_COMPLETE``,
  ``HOST_QUALIFIED``, ``ACTUAL_INPUT_QUALIFIED``, ``S2_ACCEPTED`` and
  ``S2_STOPPED_OR_INSUFFICIENT``. From CPU inputs at most ``CPU_REHEARSED`` can pass;
  ``PACKET_COMPLETE`` needs actual-scope readiness; the host, actual-input, acceptance
  and stop labels are never set to ``passed`` by this module (they need a delivered host
  and Huy's evaluation of actual evidence).
- ``evidence_integrity`` - the sealed-bundle verification (bytes unchanged since sealing;
  ``authenticity: not_verified``).
- ``human_authorization`` - owner-decision and action-scope status from readiness;
  always ``not_authenticated`` and never launch authority.

Evidence binding (completeness audit "CPU-13 report"; module F4). A row's declared status
is never trusted on its own: an item's ``engineering_status`` reads ``passed`` only when every
cited row is ``bound``, i.e. ``bundle_verification`` is a passed, sealed
``gate.s2_bundle.verify_bundle`` report whose ``manifest`` lists the row's ``path`` with the
row's ``sha256``, the manifest scope equals the report scope, and the row names its producer
identity (optional row key ``producer``: ``{module, source_sha256, function?}``, module a
dotted Python name, ``source_sha256`` 64 lowercase hex or null where unavailable). Otherwise
the row's ``binding`` is ``unverified`` with stable reasons (``no_bundle_verification``,
``bundle_not_verified``, ``bundle_scope_mismatch``, ``evidence_not_in_verified_bundle``,
``evidence_path_mismatch``, ``producer_identity_missing``) and the item reads ``unverified``
listing them. A row declared ``passed`` that lists blocker codes is self-contradictory and
also reads ``unverified`` with reason ``declared_passed_with_blockers`` (review F4-06; its
codes stay in ``engineering_blockers``). Precedence: ``blocked`` (any declared blocked row) >
``not_run`` (missing evidence) > ``unverified`` > ``passed``. The declared row status is
kept, never promoted.
A verified bundle report must carry its manifest and the manifest must recompute to
``manifest_sha256`` (``sha256(canonical_json_bytes(manifest) + b"\n")``, the bytes
``seal_bundle`` writes); otherwise ``invalid_bundle_verification``. The binding proves the
cited bytes were sealed, not that the producer was honest (``authenticity: not_verified``).

Inputs fail closed (``invalid_readiness``/``invalid_bundle_verification``): a readiness
report needs schema_version 1, a known scope (null only for a blocked report whose plan
was rejected), status passed/blocked, section objects and checks with exactly
``section, code, path, resolver, blocks_pre_rental_status``; ``passed`` with a
pre-rental blocking check is inconsistent. A bundle verification needs schema_version 1
and ``verified`` exactly when ``status`` is passed and ``seal_state`` is sealed.

Milestone binding (review F4-02): ``CPU_REHEARSED`` additionally requires that evidence is
cited and that every cited row is ``bound``; one unbound row (for example a row without a
producer identity, or bytes absent from the verified bundle) blocks the label and its
reason names the row and its binding reasons.

``render_markdown`` prints the same content for people. Verification level: both
functions L1 (literal known answers, negative controls and a real seal/verify oracle in
``tests/test_s2_report.py``). This module never reads files, launches, rents, publishes or fills an
owner value, and a synthetic pilot never yields a scientific release verdict.
"""

import hashlib
import re
from pathlib import Path

from gate.s2_identity import canonical_json_bytes, identity_hash

SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
STATUSES = ("passed", "blocked", "not_run", "not_estimable", "unverified", "not_applicable",
            "qualify_on_host")
MILESTONES = ("CPU_REHEARSED", "PACKET_COMPLETE", "HOST_QUALIFIED", "ACTUAL_INPUT_QUALIFIED",
              "S2_ACCEPTED", "S2_STOPPED_OR_INSUFFICIENT")
EVIDENCE_KEYS = (
    "readiness", "runtime_inventory", "bootstrap_plan", "pilot_packet", "full_sweep_packet",
    "lifecycle", "unit_runs", "cost", "ingestion", "capture", "actual_inspection",
    "repeats", "probes", "tensor_manifest", "validity", "population", "power", "gate",
    "telemetry", "bundle",
)
# Plan 02 S2 card; owner inputs are the card's Huy-owned registrations (never filled here).
ITEMS = {
    "S2.1": {
        "title": "Rights, revisions and registration",
        "evidence": ("readiness", "runtime_inventory", "bootstrap_plan"),
        "owner_inputs": (
            "target population", "road/scene/stratum rules", "metrics", "margins",
            "validity and terminal semantics", "seed mode", "Harmonizer setting and source",
            "control horizon/frequency", "run method/topology", "reliability floors",
            "scientific predictions", "licence/access actions"),
    },
    "S2.2": {
        "title": "Bounded cost/protocol pilot",
        "evidence": ("pilot_packet", "lifecycle", "unit_runs", "cost"),
        "owner_inputs": ("five-scene registration", "execution order", "pilot prediction",
                         "camera fix", "rental action record"),
    },
    "S2.3": {
        "title": "Complete extraction and known answers",
        "evidence": ("ingestion", "capture", "actual_inspection"),
        "owner_inputs": ("retry-selection rule", "missing-data rule", "terminal semantics"),
    },
    "S2.4": {
        "title": "Frozen-teacher reliability and controls",
        "evidence": ("repeats", "probes", "tensor_manifest", "full_sweep_packet"),
        "owner_inputs": ("reliability method, estimand and floor", "seed mode",
                         "replicate count", "scene list and order"),
    },
    "S2.5": {
        "title": "Validity and contamination",
        "evidence": ("validity", "population"),
        "owner_inputs": ("validity definition", "overlap/exclusion rule",
                         "road and stratum assignment"),
    },
    "S2.6": {
        "title": "Operational power and confirmation design",
        "evidence": ("power", "gate"),
        "owner_inputs": ("margins", "strata", "superiority rule", "evaluated-path estimand",
                         "information boundary"),
    },
    "S2.7": {
        "title": "Cost and necessary telemetry",
        "evidence": ("telemetry", "cost"),
        "owner_inputs": ("stabilization criterion", "noise and cost predictions"),
    },
}
NOT_CLAIMED = (
    "no host, renderer, model or Harmonizer qualification",
    "no scientific admission, release verdict or S2 acceptance",
    "no rental, funding, licence or publication authority",
    "no per-mile rate, safety, human-parity or real-world validation claim",
    "synthetic numbers are fixtures, not measurements",
)
_EVIDENCE_FIELDS = {"path", "sha256", "artifact_kind", "status", "scope", "blockers"}
_OPTIONAL_EVIDENCE_FIELDS = {"producer"}
_PRODUCER_REQUIRED = {"module", "source_sha256"}
_PRODUCER_OPTIONAL = {"function"}
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_FUNCTION = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BINDING_RULE = ("a row is bound only when a passed, sealed verify_bundle manifest lists its "
                "path with its sha256 and the row names its producer")
# Review F4-06: a row declared passed while listing blocker codes reads unverified.
CONTRADICTORY_ROW = "declared_passed_with_blockers"
_REHEARSAL_FIELDS = {"scope", "passed", "negative_controls", "replay_identical", "counts"}
_SHA = re.compile(r"[0-9a-f]{64}")
_CHECK_FIELDS = {"section", "code", "path", "resolver", "blocks_pre_rental_status"}
SEAL_STATES = ("sealed", "unsealed", "invalid")


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _issue(issues, code, path, message, severity="blocking"):
    issues.append({"code": code, "path": path, "severity": severity, "evidence_refs": [],
                   "message": message})


def _producer_problem(key, producer):
    if type(producer) is not dict or not _PRODUCER_REQUIRED <= set(producer) or not set(
            producer) <= _PRODUCER_REQUIRED | _PRODUCER_OPTIONAL:
        return (f"{key}.producer needs {sorted(_PRODUCER_REQUIRED)} and optionally "
                f"{sorted(_PRODUCER_OPTIONAL)}")
    if type(producer["module"]) is not str or not _MODULE.fullmatch(producer["module"]):
        return f"{key}.producer.module must be a dotted Python module name"
    source = producer["source_sha256"]
    if source is not None and (type(source) is not str or not _SHA.fullmatch(source)):
        return f"{key}.producer.source_sha256 must be 64 lowercase hex or null"
    if "function" in producer and (type(producer["function"]) is not str
                                   or not _FUNCTION.fullmatch(producer["function"])):
        return f"{key}.producer.function must be a Python identifier"
    return None


def _evidence_problem(key, value):
    if type(value) is not dict or not _EVIDENCE_FIELDS <= set(value) or not set(value) <= (
            _EVIDENCE_FIELDS | _OPTIONAL_EVIDENCE_FIELDS):
        return (f"{key} needs exactly {sorted(_EVIDENCE_FIELDS)} plus optionally "
                f"{sorted(_OPTIONAL_EVIDENCE_FIELDS)}")
    if "producer" in value:
        problem = _producer_problem(key, value["producer"])
        if problem:
            return problem
    if type(value["path"]) is not str or not value["path"] or type(
            value["artifact_kind"]) is not str:
        return f"{key} needs a path and an artifact_kind"
    if type(value["sha256"]) is not str or not _SHA.fullmatch(value["sha256"]):
        return f"{key}.sha256 must be 64 lowercase hex"
    if value["status"] not in STATUSES:
        return f"{key}.status must be one of {list(STATUSES)}"
    if value["scope"] not in SCOPES:
        return f"{key}.scope must be one of {list(SCOPES)}"
    if type(value["blockers"]) is not list or not all(
            type(code) is str for code in value["blockers"]):
        return f"{key}.blockers must be a list of issue codes"
    return None


def _readiness_problem(value):
    """Why a readiness report is unusable as input (review B2-05/B2-07); None when usable."""
    if type(value) is not dict or value.get("artifact_kind") != "s2_readiness_report":
        return "Require an s2_readiness_report"
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        return "Require readiness schema_version 1; unknown versions fail closed"
    if value.get("status") not in ("passed", "blocked") or type(value.get("status")) is not str:
        return "Readiness status is passed or blocked"
    # inspect_readiness reports scope null (and blocked) when it rejects the plan itself.
    if not (value.get("scope") is None and value["status"] == "blocked") and (
            value.get("scope") not in SCOPES or type(value.get("scope")) is not str):
        return f"Readiness scope is one of {list(SCOPES)} (null only when blocked)"
    sections = value.get("sections")
    if type(sections) is not dict or not all(
            type(name) is str and type(section) is dict and type(section.get("status")) is str
            for name, section in sections.items()):
        return "Readiness sections are an object of section objects with a status"
    checks = value.get("next_blocking_checks")
    if type(checks) is not list or not all(
            type(check) is dict and set(check) == _CHECK_FIELDS
            and all(type(check[key]) is str for key in ("section", "code", "path", "resolver"))
            and type(check["blocks_pre_rental_status"]) is bool for check in checks):
        return f"next_blocking_checks are objects with exactly {sorted(_CHECK_FIELDS)}"
    if value["status"] == "passed" and any(check["blocks_pre_rental_status"]
                                           for check in checks):
        return "Readiness says passed while listing a pre-rental blocking check"
    return None


def _bundle_problem(value):
    """Why a bundle verification is unusable as input (review B2-05); None when usable."""
    if type(value) is not dict or value.get("artifact_kind") != "s2_bundle_verification":
        return "Require an s2_bundle_verification report"
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        return "Require bundle verification schema_version 1; unknown versions fail closed"
    verified, status, seal = value.get("verified"), value.get("status"), value.get("seal_state")
    if type(verified) is not bool or status not in ("passed", "blocked") or seal not in (
            SEAL_STATES) or type(status) is not str or type(seal) is not str:
        return "verified is boolean, status passed/blocked and seal_state a known state"
    if verified != (status == "passed") or (verified and seal != "sealed"):
        return "verified only with seal_state sealed and status passed, and conversely"
    if verified:
        return _manifest_problem(value.get("manifest"), value.get("manifest_sha256"))
    return None


def _manifest_problem(manifest, manifest_sha256):
    """F4: a verified bundle report carries the manifest it sealed; None when consistent."""
    if type(manifest) is not dict or manifest.get("artifact_kind") != "s2_bundle_manifest" or (
            manifest.get("scope") not in SCOPES or type(manifest.get("scope")) is not str):
        return "A verified bundle report carries its s2_bundle_manifest with a known scope"
    entries = manifest.get("entries")
    if type(entries) is not list or not all(
            type(entry) is dict and type(entry.get("path")) is str and entry["path"]
            and type(entry.get("sha256")) is str and _SHA.fullmatch(entry["sha256"])
            for entry in entries):
        return "Manifest entries need a path and a 64-hex sha256"
    if type(manifest_sha256) is not str or not _SHA.fullmatch(manifest_sha256):
        return "A verified bundle report carries its manifest_sha256"
    try:
        recomputed = hashlib.sha256(canonical_json_bytes(manifest) + b"\n").hexdigest()
    except (TypeError, ValueError, UnicodeError):
        return "Manifest is not strict JSON"
    if recomputed != manifest_sha256:
        return "The supplied manifest does not recompute to the sealed manifest_sha256"
    return None


def _binding(scope, row, bundle):
    """Why one cited row is not bound to the verified bundle (empty when bound)."""
    reasons = []
    if bundle is None:
        reasons.append("no_bundle_verification")
    elif bundle["verified"] is not True:
        reasons.append("bundle_not_verified")
    elif bundle["manifest"]["scope"] != scope:
        reasons.append("bundle_scope_mismatch")
    else:
        entries = bundle["manifest"]["entries"]
        if not any(entry["path"] == row["path"] and entry["sha256"] == row["sha256"]
                   for entry in entries):
            same_bytes = any(entry["sha256"] == row["sha256"] for entry in entries)
            reasons.append("evidence_path_mismatch" if same_bytes
                           else "evidence_not_in_verified_bundle")
    if "producer" not in row:
        reasons.append("producer_identity_missing")
    return reasons


def _engineering_status(rows):
    """blocked > not_run (missing) > unverified (declared or unbound) > passed."""
    present = [row for row in rows if row["status"] != "missing"]
    if not present:
        return "not_run"
    statuses = {row["status"] for row in present}
    if "blocked" in statuses:
        return "blocked"
    if len(present) < len(rows):
        return "not_run"
    if statuses == {"passed"} and all(row["binding"]["status"] == "bound" and not row["blockers"]
                                      for row in present):
        return "passed"
    return "unverified"


def _milestones(rehearsal, readiness, bundle, bindings):
    """Milestone labels. ``bindings`` maps every cited evidence key to its binding.

    Review F4-02: ``CPU_REHEARSED`` reads ``passed`` only when, beside a passed rehearsal,
    an identical replay, every negative control rejected and a verified sealed bundle,
    at least one evidence row is cited and every cited row is ``bound`` to that bundle
    (``_binding``: listed path and sha256, and a producer identity). A declared row that
    is not bound blocks the label and is named in its reason.
    """
    milestones = {name: {"status": "not_run", "reason": None} for name in MILESTONES}
    cpu = milestones["CPU_REHEARSED"]
    if rehearsal is None:
        cpu["reason"] = "no rehearsal result supplied"
    elif rehearsal["scope"] != "synthetic":
        cpu.update(status="not_applicable", reason="CPU rehearsal is a synthetic-scope label")
    else:
        controls = rehearsal["negative_controls"]
        unbound = sorted(f"{key} ({', '.join(value['reasons'])})"
                         for key, value in bindings.items() if value["status"] != "bound")
        ok = (rehearsal["passed"] is True and rehearsal["replay_identical"] is True
              and bool(controls) and all(value is True for value in controls.values())
              and bundle is not None and bundle.get("verified") is True)
        bound = bool(bindings) and not unbound
        failed = sorted(name for name, value in controls.items() if value is not True)
        reasons = []
        if not ok:
            reasons.append(f"rehearsal/replay/bundle incomplete; controls not rejected: "
                           f"{failed}")
        if not bindings:
            reasons.append("no evidence row is cited")
        elif unbound:
            reasons.append("evidence not bound to the verified bundle: " + "; ".join(unbound))
        cpu.update(status="passed" if ok and bound else "blocked",
                   reason="; ".join(reasons) or None)
    packet = milestones["PACKET_COMPLETE"]
    if readiness is None:
        packet["reason"] = "no readiness report supplied"
    elif readiness.get("scope") != "actual":
        packet.update(status="not_applicable",
                      reason="synthetic or fixture readiness cannot complete a real packet")
    else:
        passed = readiness.get("status") == "passed"
        packet.update(status="passed" if passed else "blocked",
                      reason=None if passed else "readiness has blocking checks")
    for name in ("HOST_QUALIFIED", "ACTUAL_INPUT_QUALIFIED"):
        milestones[name].update(
            status="qualify_on_host",
            reason="Only a named delivered host and actual artifacts establish this label")
    milestones["S2_ACCEPTED"]["reason"] = (
        "Registered controls, reliability, validity, power and measured forecast on actual "
        "evidence, judged by Huy; unreachable from CPU inputs")
    milestones["S2_STOPPED_OR_INSUFFICIENT"]["reason"] = (
        "A measured study result; no actual study has run")
    return milestones


def build_s2_report(*, scope, evidence, readiness=None, bundle_verification=None,
                    rehearsal=None):
    """Return the deterministic S2 report artifact (L1). Invalid inputs block with issues."""
    report = {
        "artifact_kind": "s2_report", "schema_version": SCHEMA_VERSION, "scope": None,
        "producer": {"module": "gate.s2_report", "function": "build_s2_report",
                     "source_sha256": _SOURCE_SHA256},
        "input_hashes": {}, "units": {}, "status": "blocked", "issues": [],
        "items": {}, "milestones": {}, "evidence_integrity": None, "evidence_binding": None,
        "human_authorization": None, "next_actions": {"owner": [], "agent": [], "host": []},
        "counts": None, "not_claimed": list(NOT_CLAIMED),
        "cloud_launch_authorized": False, "publication_authorized": False,
        "scientific_verdict": None,
    }
    issues = report["issues"]
    if scope not in SCOPES or type(scope) is not str:
        _issue(issues, "invalid_scope", "scope", f"scope is one of {list(SCOPES)}")
        return report
    report["scope"] = scope
    for name, value in (("evidence", evidence), ("readiness", readiness),
                        ("bundle_verification", bundle_verification), ("rehearsal", rehearsal)):
        try:
            report["input_hashes"][name] = identity_hash(value)
        except (TypeError, ValueError, UnicodeError):
            _issue(issues, "invalid_input", name, "Inputs must be strict JSON")
            return report
    if type(evidence) is not dict or not set(evidence) <= set(EVIDENCE_KEYS):
        _issue(issues, "invalid_evidence", "evidence",
               f"Evidence keys are a subset of {list(EVIDENCE_KEYS)}")
        return report
    for key in sorted(evidence):
        problem = _evidence_problem(key, evidence[key])
        if problem:
            _issue(issues, "invalid_evidence", f"evidence.{key}", problem)
        elif scope == "actual" and evidence[key]["scope"] != "actual":
            _issue(issues, "synthetic_evidence_in_actual_report", f"evidence.{key}",
                   "An actual-scope report cannot cite synthetic or fixture evidence")
    if rehearsal is not None and (type(rehearsal) is not dict or set(
            rehearsal) != _REHEARSAL_FIELDS or type(rehearsal["negative_controls"]) is not dict
            or type(rehearsal["counts"]) is not dict or rehearsal["scope"] not in SCOPES):
        _issue(issues, "invalid_rehearsal", "rehearsal",
               f"Rehearsal summary needs exactly {sorted(_REHEARSAL_FIELDS)}")
    problem = None if readiness is None else _readiness_problem(readiness)
    if problem:
        _issue(issues, "invalid_readiness", "readiness", problem)
    problem = None if bundle_verification is None else _bundle_problem(bundle_verification)
    if problem:
        _issue(issues, "invalid_bundle_verification", "bundle_verification", problem)
    if issues:
        return report
    scientific = "unverified" if scope == "actual" else "blocked"
    bindings = {}
    for key in sorted(evidence):
        reasons = _binding(scope, evidence[key], bundle_verification)
        bindings[key] = {"status": "unverified" if reasons else "bound", "reasons": reasons}
    report["evidence_binding"] = {
        "bound": sum(value["status"] == "bound" for value in bindings.values()),
        "unverified": sum(value["status"] != "bound" for value in bindings.values()),
        "rule": BINDING_RULE}
    for item, spec in ITEMS.items():
        rows = []
        for key in spec["evidence"]:
            value = evidence.get(key)
            if value is None:
                rows.append({"key": key, "status": "missing"})
                continue
            row = {"key": key, **{field: value[field] for field in sorted(_EVIDENCE_FIELDS)},
                   "producer": value.get("producer"), "binding": bindings[key]}
            rows.append(row)
        blockers = sorted({code for row in rows for code in row.get("blockers", [])})
        missing = [row["key"] for row in rows if row["status"] == "missing"]
        unverified = sorted(
            [f"{row['key']}: {reason}" for row in rows if "binding" in row
             for reason in row["binding"]["reasons"]]
            + [f"{row['key']}: {CONTRADICTORY_ROW}" for row in rows
               if row["status"] == "passed" and row["blockers"]])
        report["items"][item] = {
            "title": spec["title"], "evidence": rows,
            "engineering_status": _engineering_status(rows),
            "engineering_blockers": blockers, "missing_evidence": missing,
            "unverified_reasons": unverified,
            "scientific_status": scientific,
            "scientific_blocker": ("owner_inputs_unregistered" if scope != "actual"
                                   else "owner_interpretation_required"),
            "owner_inputs": list(spec["owner_inputs"]), "owner_inputs_filled_by_agent": False,
        }
    report["milestones"] = _milestones(rehearsal, readiness, bundle_verification, bindings)
    report["evidence_integrity"] = (
        {"status": "not_run", "seal_state": None, "verified": False,
         "authenticity": "not_verified"} if bundle_verification is None else
        {"status": bundle_verification.get("status"),
         "seal_state": bundle_verification.get("seal_state"),
         "verified": bundle_verification.get("verified") is True,
         "manifest_sha256": bundle_verification.get("manifest_sha256"),
         "authenticity": "not_verified"})
    sections = (readiness or {}).get("sections") or {}
    report["human_authorization"] = {
        "status": "not_authenticated",
        "readiness_scope": (readiness or {}).get("scope"),
        "owner_decisions": (sections.get("owner_decisions") or {}).get("status", "not_run"),
        "action_scope": (sections.get("action_scope") or {}).get("status", "not_run"),
        "cloud_launch_authorized": False,
    }
    for check in (readiness or {}).get("next_blocking_checks", []):
        resolver = check.get("resolver")
        if resolver in report["next_actions"]:
            entry = f"{check.get('section')}: {check.get('code')} at {check.get('path')}"
            if entry not in report["next_actions"][resolver]:
                report["next_actions"][resolver].append(entry)
    if rehearsal is not None:
        report["counts"] = rehearsal["counts"]
    report["status"] = "passed"
    return report


def render_markdown(report):
    """Human-readable rendering of ``build_s2_report`` output (pure)."""
    lines = [f"# S2 report ({report.get('scope')})", "",
             ("Engineering status, milestone labels, evidence integrity and human "
             "authorization are separate; none of them is an admission."), ""]
    if report.get("status") != "passed":
        lines += ["Report inputs were rejected:", ""]
        lines += [f"- {item['code']} at {item['path']}: {item['message']}"
                  for item in report.get("issues", [])]
        return "\n".join(lines) + "\n"
    lines += ["## Milestone labels", "", "| Label | Status | Reason |", "|---|---|---|"]
    for name in MILESTONES:
        value = report["milestones"][name]
        lines.append(f"| {name} | {value['status']} | {value['reason'] or ''} |")
    integrity = report["evidence_integrity"]
    authorization = report["human_authorization"]
    lines += ["", "## Evidence integrity", "",
              (f"Bundle: {integrity['status']} (seal {integrity['seal_state']}); authenticity "
              "not verified."), "", "## Human authorization", "",
              (f"Owner decisions: {authorization['owner_decisions']}; action scope: "
              f"{authorization['action_scope']} (readiness scope "
              f"{authorization['readiness_scope']}); not authenticated; no launch authority."), "",
              "## S2 card items", "",
              "| Item | Title | Engineering | Scientific | Missing evidence | Blockers |",
              "|---|---|---|---|---|---|"]
    for item, value in report["items"].items():
        lines.append(f"| {item} | {value['title']} | {value['engineering_status']} | "
                     f"{value['scientific_status']} | {', '.join(value['missing_evidence'])} | "
                     f"{', '.join(value['engineering_blockers'])} |")
    lines += ["", "## Unverified evidence", "",
              f"Rule: {BINDING_RULE}.", ""]
    reasons = [f"- {item}: {reason}" for item, value in report["items"].items()
               for reason in value["unverified_reasons"]]
    lines += reasons or ["- none"]
    lines += ["", "## Huy-owned inputs (not filled by agents)", ""]
    for item, value in report["items"].items():
        lines.append(f"- {item}: {'; '.join(value['owner_inputs'])}")
    lines += ["", "## Next actions", ""]
    for resolver in ("owner", "agent", "host"):
        actions = report["next_actions"][resolver]
        lines.append(f"- {resolver}: " + ("; ".join(actions) if actions else "none listed"))
    if report["counts"] is not None:
        lines += ["", "## Counts", ""]
        lines += [f"- {key}: {value}" for key, value in sorted(report["counts"].items())]
    lines += ["", "## Not claimed", ""] + [f"- {text}" for text in report["not_claimed"]]
    return "\n".join(lines) + "\n"
