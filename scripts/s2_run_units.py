"""Guest S2 unit runner and attempt recorder: execute a prepared packet's units serially.

Tickets: build-spec B1 (runbook section 4 steps 6-8; lane R1 P4 serial executor + attempt
recorder, P4b; W2 producer; lane R2 section 1 F1-F6, section 2 outputs and destructive
cleanup, section 4 P1-P3 and C1-C3); module G2 of the D-024 final round: completeness
audit "Safety" (hash-bound authority and readiness instead of free text), gap "CPU-08 to
runner" (phase-aware budget fed by the cost ledger), runbook step 6 / gap "diagnostic
qualification unit" (declared diagnostic units), review F6 (clock anchor and step times).
Agent-authored under D-018/D-024; serves S2.1-S2.7 [C] (G1, G2, G3-R1, G7).

Usage (on the admitted guest, after recorded authority)::

    python3 scripts/s2_run_units.py PACKET.json --authority AUTHORITY.json \
        --run-config RUN_CONFIG.json --budget-file BUDGET.json \
        --journal-dir /abs/new/journal --unit-selection {all,diagnostics_only} \
        --clock-uncertainty-us N --decisions docs/DECISIONS.md --decision-id D-NNN \
        [--readiness-report READINESS.json] \
        [--release-record RELEASE.json --release-decision-id D-NNN] [--execute]

Without ``--execute`` it applies exactly the checks ``run_units`` applies before any effect
(``preflight``; review G2-05), prints the plan and runs nothing. A real-runner dry run
without ``--decisions`` prints ``authority NOT verified`` and exits 2, never ``validated``.

``PACKET`` is a prepared ``s2_pilot_packet`` or ``s2_full_sweep_packet``
(``gate.s2_executor.PACKET_KINDS``; any other kind is refused before any effect). With
``--unit-selection all`` it runs the packet's declared diagnostic qualification units
(``gate.s2_packet`` ``diagnostic_units``; pilot only) first, then every unit of
``gate.s2_executor.build_execution_plan`` in the registered order; ``diagnostics_only``
runs only the declared diagnostic units. Diagnostic units have their own ledger
(``diagnostic-ledger-events.jsonl``, reconciled against the plan's ``diagnostic_schedule``),
unit records (``diagnostics/NNN``) and budget line; they are never one of the pilot jobs,
never in ``ledger`` and never an inference input. Each unit runs the plan's steps verbatim
through an injected runner:

1. Budget check on the cost ledger (CPU-08; ``BudgetMeter``): ``gate.s2_cost.reconcile_cost``
   over the starting snapshot's ``prior_events`` plus this run's own ``compute_interval``
   segments (contiguous from ``accounted_until_epoch``, the open one accrued to now), then
   ``gate.s2_cost.can_start_next_unit`` with the configured worst-case unit duration
   (``gate.s2_executor.worst_case_unit_seconds`` with ``KILL_GRACE_S``) plus cleanup, so
   headroom shrinks unit by unit within a run. D-018 nesting: pilot units (and declared
   diagnostics) are metered from the pilot tier (pilot <= stage <= project); full-sweep
   units from the stage tier (stage <= project, never the pilot cap). A diagnostic unit's
   exposure must also fit strictly under its declared budget line cap minus that line's
   cumulative spend: the snapshot's ``prior_line_spend_usd`` (the previous runs'
   ``line_spend.cumulative_spent_usd``, carried by the operator; no default) plus this run's
   segments on that line (review G2-02). The quote's scope must be the packet's and an
   actual budget refuses synthetic or placeholder prior references (G2-03); a snapshot
   with a ``credit_snapshot`` or ``settlement_final`` is refused, because ``gate.s2_cost``
   takes usage as a max over those figures and would absorb this run's segments (G2-01).
   The budget file is only that starting snapshot, read once. A refusal
   stops the run before the unit: no ledger event is written for it and every later unit
   stays ``not_run``. Closed segments go to ``cost-events.jsonl``.
2. The ledger ``started`` event (schema v2) is appended and fsynced before any side effect.
3. Run and driver-log directories must not exist (re-running the wizard in an existing run
   folder overwrites YAMLs, lane R2 section 2 item 6). Wizard prepare.
4. ``gate.s2_launch.check_generated_configs`` on ``yaml.safe_load`` of the generated files
   (blocks e.g. 6 renderer cameras vs 4 driver cameras before any GPU work); the Compose
   ``shm_size`` override (``docker-compose.override.yaml``) when the packet declares a
   non-default value (A4 never writes it); ``docker compose config`` and
   ``gate.s2_launch.check_generated_compose`` on the effective configuration.
5. The driver starts with ``gate.s2_launch.filter_environment`` (allowlisted env only) and
   is ready only when its port is listening AND the pinned start line is in its log,
   within the configured timeout; a driver that exits or never becomes ready stops the run.
6. ``docker compose up --pull never --exit-code-from runtime-0`` with stdout+stderr teed
   to the capture file. Its exit code is recorded, never used as success evidence.
7. Before teardown, on every exit path once Compose was started (including an internal
   error, KeyboardInterrupt or SystemExit): ``compose logs``/``ps``/``ps -q`` ->
   ``docker inspect``, the ``run_metadata.yaml`` bytes and a no-follow manifest of the run
   directory (``scripts/s2_guest_manifest.py``, symlinks recorded, never followed).
   An interrupt inside a capture is recorded on that step, later docker captures are
   recorded ``not_run`` and the metadata read and run-dir hash still run before teardown.
8. Teardown: ``compose down``, SIGINT to the driver (a driver that survives SIGINT,
   SIGTERM and SIGKILL is ``teardown_unverified`` and stops the run), port released
   (verified). GPU memory release is recorded ``qualify_on_host``. The driver stop and
   port check run (``finally``) even when ``compose down`` raises or is interrupted.
9. ``gate.s2_capture.reconcile_unit_attempts`` over the captured logs, rollout folders,
   run metadata and summary, each read with ``gate.s2_identity.read_artifact`` against
   the run-dir manifest hashed before teardown (a mismatch or absent manifest is an issue
   and those bytes are not reconciled), then terminal ledger events binding every upstream attempt
   (``gate.s2_executor.unit_terminal_events``): an internal retry yields two attempts; a
   pre-UUID crash yields an attempt without upstream identity; a failed rollout whose
   Compose exited 0 is recorded as failed. A failed rollout does not stop the run; a
   pre-GPU block, a driver that never becomes ready, a teardown that cannot be verified,
   a unit with no upstream attempt evidence at all (systemic, e.g. Compose rejecting its
   argv or timing out before any runtime line; agent technical default, D-018) or an
   internal error does. An interrupt is recorded (terminal event from the upstream
   evidence, cancelled when it stopped Compose; unit record; partial run summary) and
   re-raised.

Authority (shared contract 2.6; completeness audit "Safety": free-text authority no longer
suffices). ``--authority`` is a required ``s2_unit_run_authority`` record binding the
packet's ``request_sha256`` and its exact ordered job IDs; the plan binds the packet to
that hash (``gate.s2_executor`` rebuilds it from its inputs). Every runner that does not
declare ``fixture_only = True`` (``is_fixture_runner``) is real: it refuses any scope other
than ``actual`` and any ``SYNTHETIC`` reference (a fixture runner refuses scope
``actual``, review G2-06), and ``verify_run_authority`` requires
``gate.s2_authority.verify_authority`` over the supplied ``docs/DECISIONS.md`` bytes to
authorize the packet's ``artifact_kind`` with ``identity_hash(packet)`` and, as
``also_required``, ``s2_readiness_report`` with the identity hash of a passed
``gate.s2_readiness`` report for this packet (``readiness_issues``); the authority's
``action_record_ref`` must name that decision. Only an entry flagged
``engineering_only_diagnostic: true`` with ``--unit-selection diagnostics_only`` runs
without readiness, and such an entry never authorizes pilot or sweep units. A full-sweep
packet (any runner) also needs Huy's release record authorized as ``s2_full_release`` by
``--release-decision-id``; there is no default release.

Clock anchor (review F6): at run start ``run-summary.json`` records ``clock_anchor``
``{clock_id, monotonic_us, utc, uncertainty_us, anchor_ref}`` (adjacent monotonic and wall
samples; ``uncertainty_us`` is the operator's ``--clock-uncertainty-us``, no default), the
mapping ``gate.s2_telemetry_sources.allocation_from_journals`` needs. Every step note
carries monotonic ``start_us`` (when the runner moved on from the previous note) and
``end_us`` (when the note was written).

Statuses follow shared contract 2.2 (``passed``, ``blocked``, ``not_run``,
``unverified``, ``not_applicable``, ``qualify_on_host``); domain values stay in
``outcome`` (steps and unit records) and ``run_outcome`` (``completed``, ``stopped``,
``interrupted``, ``refused``). Unit records carry the contract 2.1 envelope;
unit records and the run summary carry ``status_reasons`` naming each non-passed cause.

Outputs: ``<journal>/ledger-events.jsonl`` (pilot/sweep units; canonical JSON, one fsync
per event), ``<journal>/diagnostic-ledger-events.jsonl`` (only when diagnostics are
declared), ``<journal>/cost-events.jsonl`` (this run's compute segments),
``<journal>/units/<NNN>/`` and ``<journal>/diagnostics/<NNN>/`` (wizard stdout, unit
record; a diagnostic unit's record is an ``s2_diagnostic_unit_record`` with top-level
``role: diagnostic_qualification``, ``inference_input: false`` and ``diagnostic_id``, and its
ledger job has phase ``diagnostic_qualification``; review G2-04), each ``cost-events.jsonl``
line being a ``gate.s2_cost`` ``compute_interval`` plus its ``line_id`` (strip it before
``reconcile_cost``), ``<journal>/run-summary.json`` (clock anchor, authority verification,
budget with per-line ``line_spend``,
``gate.s2_ledger.reconcile_attempts`` over the packet schedule and the pilot events, and
over the diagnostic schedule and diagnostic events) and, inside each run dir,
``physloop-capture/`` (check reports, captures, run-dir manifest, unit attempt evidence).
Existing files are never overwritten.

Verification level: ``run_units``, ``validate_authority``, ``verify_run_authority``,
``readiness_issues``, ``validate_budget`` and ``BudgetMeter`` L1 with fake runner, clock
and port probe (``tests/test_s2_run_units.py``); ``SubprocessRunner`` L1 against the local
Python interpreter only; ``port_open`` and every real tool (uv, docker, driver) are L3 -
a completed run is a host record, not ``HOST_QUALIFIED``. The authority check proves that
the supplied decision bytes contain the grant lines, not that Huy wrote them
(``supplied_not_authenticated``). Agent technical defaults (D-018): file read bounds
``MAX_CONFIG_BYTES`` = 16 MiB and ``MAX_LOG_BYTES`` = 512 MiB; ``KILL_GRACE_S`` = 10 s
between SIGINT/SIGTERM/SIGKILL; a wall clock that steps backwards stops every later
budget check (``wall_clock_regressed``). Owner inputs, required with no default: caps,
limits and reserves in the budget snapshot, diagnostic roles and budget-line caps, the
full-sweep release record, the clock uncertainty bound. Operator input with no default:
each declared line's carried-forward spend (``prior_line_spend_usd``).

This runner never creates, stops or destroys a rental, downloads gated assets itself,
accepts terms, chooses an owner value, selects among retried attempts, deletes any run
output, counts a diagnostic unit as a pilot job or infers success from an exit code.
"""

import argparse
import contextlib
import hashlib
import math
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Context, Decimal, DecimalException, localcontext
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from gate.s2_authority import verify_authority
from gate.s2_capture import reconcile_unit_attempts
from gate.s2_cost import billed_seconds, can_start_next_unit, reconcile_cost
from gate.s2_executor import (
    UNIT_STEPS,
    attempt_id,
    build_execution_plan,
    cancelled_event,
    container_ids,
    driver_ready_line,
    has_upstream_attempt_evidence,
    inspect_argv,
    started_event,
    terminal_established,
    unit_terminal_events,
    worst_case_unit_seconds,
)
from gate.s2_identity import (
    _directory_fd,
    canonical_json_bytes,
    identity_hash,
    read_artifact,
    read_json_bytes,
)
from gate.s2_launch import (
    check_generated_compose,
    check_generated_configs,
    filter_environment,
)
from gate.s2_ledger import reconcile_attempts
from gate.s2_runtime import is_placeholder, secret_like
from scripts.s2_guest_manifest import build_manifest

ARTIFACT_KIND = "s2_unit_run_summary"
SCHEMA_VERSION = 2
UNIT_RECORD_KIND = "s2_unit_record"
UNIT_RECORD_VERSION = 1
# Review G2-04: a diagnostic unit record has its own kind and top-level role markers, so no
# consumer of ``s2_unit_record`` (gate.s2_actual_extract, telemetry) can take it as a pilot
# job's record. Pilot records keep the exact v1 shape.
DIAGNOSTIC_RECORD_KIND = "s2_diagnostic_unit_record"
DIAGNOSTIC_ROLE = "diagnostic_qualification"
# Shared engineering contract 2.2; domain outcomes live in separate ``outcome`` fields.
STATUSES = ("passed", "blocked", "not_run", "not_estimable", "unverified", "not_applicable",
            "qualify_on_host")
AUTHORITY_KIND = "s2_unit_run_authority"
AUTHORITY_FIELDS = {"artifact_kind", "schema_version", "scope", "action_record_ref",
                    "packet_request_sha256", "job_ids"}
UNIT_SELECTIONS = ("all", "diagnostics_only")
BUDGET_KIND = "s2_unit_budget"
# Version 2 (review G2-02) adds ``prior_line_spend_usd``; a version 1 snapshot is refused.
BUDGET_VERSION = 2
BUDGET_FIELDS = {"artifact_kind", "schema_version", "scope", "phase", "prior_events", "quote",
                 "caps_usd", "storage_retention_until_epoch", "limits", "instance_resource_id",
                 "gpu_count", "accounted_until_epoch", "cleanup_seconds",
                 "next_unit_upper_bound_usd", "prior_line_spend_usd"}
# Review G2-01: gate.s2_cost takes usage as max(per-resource figure + transfer, observed
# credit consumption, final settlement). With either of these kinds in the starting snapshot
# this run's own segments can be absorbed into that max, so headroom would not shrink.
NON_ADDITIVE_PRIOR_KINDS = ("credit_snapshot", "settlement_final")
# Packet kind -> (budget phase, s2_cost tier the unit is metered from). D-018 nesting: a
# pilot unit is checked against pilot <= stage <= project, a full-sweep unit against
# stage <= project only (never the pilot cap).
PACKET_BUDGET = {"s2_pilot_packet": ("pilot", "pilot"),
                 "s2_full_sweep_packet": ("full_sweep", "stage")}
READINESS_PHASES = {"s2_pilot_packet": "harmonizer_pilot", "s2_full_sweep_packet": "full_sweep"}
READINESS_KIND = "s2_readiness_report"
RELEASE_KIND = "s2_full_release"
RUN_EVENT_PREFIX = "s2_run_units:"
MAX_CONFIG_BYTES = 16 * 1024 * 1024  # agent technical default (D-018)
MAX_LOG_BYTES = 512 * 1024 * 1024  # agent technical default (D-018)
KILL_GRACE_S = 10  # agent technical default (D-018)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _source_sha256():
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


_SOURCE_SHA256 = _source_sha256()


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


# ------------------------------------------------------------------ authority

def validate_authority(authority, packet, *, real_runner):
    """Issues that refuse execution; empty means the authority binds this packet (L1).

    Structural only: it never authenticates the referenced approval.
    """
    if type(authority) is not dict or set(authority) != AUTHORITY_FIELDS:
        return [_issue("invalid_authority", "authority",
                       f"Authority requires exactly {sorted(AUTHORITY_FIELDS)}")]
    issues = []
    if (authority["artifact_kind"] != AUTHORITY_KIND
            or type(authority["schema_version"]) is not int
            or authority["schema_version"] != 1):
        issues.append(_issue("invalid_authority", "authority.artifact_kind",
                             f"Expected {AUTHORITY_KIND} schema version 1"))
    ref = authority["action_record_ref"]
    if type(ref) is not str or not ref.strip() or is_placeholder(ref) or secret_like(ref):
        issues.append(_issue("authority_unrecorded", "authority.action_record_ref",
                             "A recorded, non-placeholder, non-credential reference is required"))
    if type(packet) is not dict:
        return issues + [_issue("invalid_packet", "packet", "Packet must be an object")]
    if authority["scope"] != packet.get("scope"):
        issues.append(_issue("scope_mismatch", "authority.scope",
                             "Authority scope differs from the packet scope"))
    if authority["packet_request_sha256"] != packet.get("request_sha256"):
        issues.append(_issue("authority_packet_mismatch", "authority.packet_request_sha256",
                             "Authority binds another packet request"))
    units = packet.get("units") if type(packet.get("units")) is list else []
    ordered = [unit.get("job", {}).get("job_id") for unit in units if type(unit) is dict]
    if authority["job_ids"] != ordered:
        issues.append(_issue("authority_jobs_mismatch", "authority.job_ids",
                             "Authority must list exactly the packet's ordered job IDs"))
    if not real_runner and packet.get("scope") == "actual":
        # Review G2-06: a CPU fake must never mint actual-scope unit records.
        issues.append(_issue("fixture_runner_actual_scope_refused", "scope",
                             "A fixture runner never executes an actual-scope packet"))
    if real_runner:
        if packet.get("scope") != "actual" or authority["scope"] != "actual":
            issues.append(_issue("fixture_scope_refused", "scope",
                                 "The real runner executes only scope 'actual' packets"))
        if type(ref) is not str or "synthetic" in ref.casefold():
            issues.append(_issue("synthetic_authority_refused", "authority.action_record_ref",
                                 "Synthetic references are not recorded authority"))
    return issues


def _decision_token(ref, decision_id):
    """True when ``ref`` names ``decision_id`` as a delimited token (readiness's rule)."""
    return type(ref) is str and type(decision_id) is str and re.search(
        r"(?<![A-Za-z0-9])" + re.escape(decision_id) + r"(?![A-Za-z0-9])", ref) is not None


def readiness_issues(report, packet):
    """Issues when ``report`` is not a passed readiness report for this packet (L1).

    Requires a ``gate.s2_readiness.inspect_readiness`` report of schema version 1 with
    status ``passed``, the packet's scope, the phase of the packet kind, and a
    ``planning_consistency.packet`` that names this packet's ``request_sha256``. It checks
    the report's declared content only; the decision entry binds its hash.
    """
    path = "readiness_report"
    if type(report) is not dict or report.get("artifact_kind") != READINESS_KIND or type(
            report.get("schema_version")) is not int or report["schema_version"] != 1:
        return [_issue("invalid_readiness_report", path,
                       f"Require an {READINESS_KIND} of schema version 1")]
    issues = []
    if report.get("status") != "passed":
        issues.append(_issue("readiness_not_passed", f"{path}.status",
                             "Only a passed readiness report admits a real unit run"))
    if report.get("scope") != packet.get("scope"):
        issues.append(_issue("readiness_scope_mismatch", f"{path}.scope",
                             "Readiness scope differs from the packet scope"))
    if report.get("phase") != READINESS_PHASES.get(packet.get("artifact_kind")):
        issues.append(_issue("readiness_phase_mismatch", f"{path}.phase",
                             "Readiness phase differs from the packet kind's phase"))
    sections = report.get("sections") if type(report.get("sections")) is dict else {}
    planning = sections.get("planning_consistency")
    bound = planning.get("packet") if type(planning) is dict else None
    if type(bound) is not dict or bound.get("request_sha256") != packet.get("request_sha256"):
        issues.append(_issue("readiness_packet_mismatch",
                             f"{path}.sections.planning_consistency.packet.request_sha256",
                             "Readiness was not computed for this packet's request"))
    return issues


def _authority_summary(check):
    keys = ("decision_id", "authorized", "engineering_only_diagnostic", "decisions_sha256",
            "entry_sha256", "closed_by", "authenticity")
    return {key: check.get(key) for key in keys} | {
        "issue_codes": sorted({item["code"] for item in check.get("issues", [])})}


def verify_run_authority(packet, authority, *, real_runner, unit_selection, decisions_md,
                         decision_id, readiness_report, release_record, release_decision_id):
    """Hash-bound authority for one run; returns ``(issues, record)`` (L1).

    Every runner: ``validate_authority`` (structure, packet request hash, ordered jobs). A
    full-sweep packet additionally needs a recorded release: ``decisions_md`` must
    authorize ``s2_full_release`` with the identity hash of ``release_record`` (Huy's
    record, no default) in ``release_decision_id``; a pilot packet takes no release.

    A real runner (``is_fixture_runner`` false) additionally needs ``decisions_md`` bytes
    and ``decision_id`` such that ``gate.s2_authority.verify_authority`` authorizes the
    packet's ``artifact_kind`` with ``identity_hash(packet)`` and, as ``also_required``,
    ``("s2_readiness_report", identity_hash(readiness_report))`` for a readiness report that
    passes ``readiness_issues``. The readiness requirement is waived only when the entry
    carries ``engineering_only_diagnostic: true`` and ``unit_selection`` is
    ``diagnostics_only``; such an entry never authorizes pilot or sweep units. The
    authority record's ``action_record_ref`` must name ``decision_id`` as a token, so a
    free-text reference no longer suffices.
    """
    issues = validate_authority(authority, packet, real_runner=real_runner)
    record = {"mode": "recorded_decision" if real_runner else "fixture_runner",
              "packet_sha256": _hash_or_none(packet), "decision": None,
              "readiness_report_sha256": None, "readiness_waived": False, "release": None}
    if type(packet) is not dict:
        return issues, record
    kind = packet.get("artifact_kind")
    release_inputs = (release_record, release_decision_id)
    if kind == "s2_full_sweep_packet":
        if type(release_record) is not dict or not release_record or type(
                release_decision_id) is not str or type(decisions_md) is not bytes:
            issues.append(_issue("full_release_unrecorded", "release_record",
                                 "A full sweep needs Huy's release record, its decision ID "
                                 "and the decision log; no default release exists"))
        else:
            check = verify_authority(decisions_md, decision_id=release_decision_id,
                                     artifact_kind=RELEASE_KIND,
                                     artifact_sha256=_hash_or_none(release_record) or "")
            record["release"] = {"release_record_sha256": _hash_or_none(release_record),
                                 **_authority_summary(check)}
            if not check["authorized"]:
                issues.append(_issue("full_release_unrecorded", "release_record",
                                     f"{release_decision_id} does not authorize "
                                     f"{RELEASE_KIND} for this release record: "
                                     f"{record['release']['issue_codes']}"))
    elif any(value is not None for value in release_inputs):
        issues.append(_issue("release_not_applicable", "release_record",
                             "Only a full-sweep packet takes a release record"))
    if not real_runner:
        return issues, record
    if type(decisions_md) is not bytes or type(decision_id) is not str:
        issues.append(_issue("decision_record_missing", "decisions_md",
                             "A real run needs the decision log bytes and a decision ID"))
        return issues, record
    if type(authority) is dict and not _decision_token(authority.get("action_record_ref"),
                                                       decision_id):
        issues.append(_issue("authority_decision_mismatch", "authority.action_record_ref",
                             f"The authority's action reference does not name {decision_id}"))
    packet_hash = record["packet_sha256"] or ""
    check = verify_authority(decisions_md, decision_id=decision_id, artifact_kind=str(kind),
                             artifact_sha256=packet_hash)
    record["decision"] = _authority_summary(check)
    if not check["authorized"]:
        issues.append(_issue("packet_not_authorized", "decisions_md",
                             f"{decision_id} does not authorize {kind} sha256:{packet_hash}: "
                             f"{record['decision']['issue_codes']}"))
        return issues, record
    flagged = check["engineering_only_diagnostic"] is True
    if flagged and unit_selection != "diagnostics_only":
        issues.append(_issue("engineering_only_decision_for_scientific_units", "unit_selection",
                             f"{decision_id} is engineering_only_diagnostic; it authorizes "
                             "only a run of declared diagnostic units"))
        return issues, record
    if flagged and readiness_report is None:
        record["readiness_waived"] = True
        return issues, record
    problems = readiness_issues(readiness_report, packet)
    if problems:
        return issues + problems, record
    readiness_hash = _hash_or_none(readiness_report) or ""
    record["readiness_report_sha256"] = readiness_hash
    check = verify_authority(decisions_md, decision_id=decision_id, artifact_kind=str(kind),
                             artifact_sha256=packet_hash,
                             also_required=((READINESS_KIND, readiness_hash),))
    record["decision"] = _authority_summary(check)
    if not check["authorized"]:
        issues.append(_issue("readiness_not_authorized", "decisions_md",
                             f"{decision_id} does not authorize {READINESS_KIND} "
                             f"sha256:{readiness_hash}: {record['decision']['issue_codes']}"))
    return issues, record


# ------------------------------------------------------------------ budget (CPU-08)

def _decimal(value):
    if type(value) is not str:
        raise ValueError("expected a decimal string")
    amount = Decimal(value)
    if not amount.is_finite() or amount < 0:
        raise ValueError("expected a finite nonnegative decimal")
    return amount


# gate.s2_cost precision (60 digits) with ceiling rounding, so spend is never understated.
_COST_CONTEXT = Context(prec=60, rounding=ROUND_CEILING)


def _plain(value):
    value = value.normalize(_COST_CONTEXT)
    return "0" if value == 0 else format(value, "f")


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_budget(budget, packet, *, worst_case_s):
    """Validate the operator's starting budget snapshot; return ``(issues, context)`` (L1).

    The snapshot is read once per run: ``prior_events`` and ``quote`` are
    ``gate.s2_cost`` inputs (the prior ledger snapshot), ``caps_usd``, ``limits`` and
    ``storage_retention_until_epoch`` the nested caps and limits (owner values, no
    defaults), ``instance_resource_id`` the bound instance, ``accounted_until_epoch`` the
    time up to which ``prior_events`` account for that instance's compute (none of its
    compute intervals may be open or end later, so run time is never counted twice or
    missed), ``gpu_count``, ``cleanup_seconds`` and ``next_unit_upper_bound_usd`` the
    per-unit exposure inputs. ``phase`` must be the packet kind's budget phase and
    ``quote.scope`` the packet scope (G2-03). ``prior_line_spend_usd`` maps exactly the
    packet's declared diagnostic budget lines to their spend before this run (G2-02).
    Refused: ``credit_snapshot``/``settlement_final`` prior events (G2-01, they make
    ``reconcile_cost``'s usage a max that would absorb this run's segments) and, for an
    actual packet, prior events whose ``decision_ref``/``evidence_ref`` is synthetic or a
    placeholder (G2-03).
    """
    issues = []
    if type(budget) is not dict or budget.get("artifact_kind") != BUDGET_KIND or type(
            budget.get("schema_version")) is not int or budget["schema_version"] != BUDGET_VERSION:
        return [_issue("invalid_budget", "budget.artifact_kind",
                       f"Expected {BUDGET_KIND} schema version {BUDGET_VERSION}")], None
    if set(budget) != BUDGET_FIELDS:
        return [_issue("invalid_budget", "budget",
                       f"Budget snapshot requires exactly {sorted(BUDGET_FIELDS)}")], None
    phase, tier = PACKET_BUDGET.get(packet.get("artifact_kind") if type(packet) is dict
                                    else None, (None, None))
    packet_scope = packet.get("scope") if type(packet) is dict else None
    if budget["scope"] != packet_scope:
        issues.append(_issue("scope_mismatch", "budget.scope",
                             "Budget scope differs from the packet scope"))
    quote = budget["quote"]
    if type(quote) is dict and quote.get("scope") != packet_scope:
        # Review G2-03: an actual run is never metered on a synthetic quote (or vice versa).
        issues.append(_issue("quote_scope_mismatch", "budget.quote.scope",
                             "The quote's scope differs from the packet scope"))
    caps = _declared_line_caps(packet)
    spend = budget["prior_line_spend_usd"]
    if type(spend) is not dict or set(spend) != set(caps) or not all(
            _decimal_or_none(value) is not None for value in spend.values()):
        issues.append(_issue("invalid_budget", "budget.prior_line_spend_usd",
                             "Each declared diagnostic budget line needs its spend before this "
                             f"run as a nonnegative decimal string (exactly {sorted(caps)}); "
                             "no default"))
    if phase is None or budget["phase"] != phase:
        issues.append(_issue("budget_phase_mismatch", "budget.phase",
                             f"This packet is metered as phase {phase!r}"))
    resource = budget["instance_resource_id"]
    if type(resource) is not str or not resource.strip() or resource.startswith(
            RUN_EVENT_PREFIX):
        issues.append(_issue("invalid_budget", "budget.instance_resource_id",
                             "The bound instance resource ID is required"))
    count = budget["gpu_count"]
    if type(count) is not int or count < 1:
        issues.append(_issue("invalid_budget", "budget.gpu_count", "A positive integer"))
    for key in ("accounted_until_epoch", "cleanup_seconds"):
        if not _seconds(budget[key]):
            issues.append(_issue("invalid_budget", f"budget.{key}",
                                 "A finite nonnegative number is required"))
    try:
        _decimal(budget["next_unit_upper_bound_usd"])
    except (ValueError, DecimalException):
        issues.append(_issue("invalid_budget", "budget.next_unit_upper_bound_usd",
                             "A nonnegative decimal string is required"))
    events = budget["prior_events"]
    if type(events) is not list:
        issues.append(_issue("invalid_budget", "budget.prior_events", "A list of cost events"))
        events = []
    accounted = budget["accounted_until_epoch"]
    for position, event in enumerate(events):
        if type(event) is not dict:
            continue
        if type(event.get("event_id")) is str and event["event_id"].startswith(
                RUN_EVENT_PREFIX):
            issues.append(_issue("reserved_event_id", f"budget.prior_events[{position}]",
                                 f"Event IDs starting {RUN_EVENT_PREFIX!r} are the runner's"))
        if event.get("kind") in NON_ADDITIVE_PRIOR_KINDS:
            issues.append(_issue(
                "non_additive_prior_event", f"budget.prior_events[{position}]",
                f"{event['kind']} can absorb this run's segments into gate.s2_cost's "
                "max(per-resource, credit consumption, settlement) usage; express pre-run "
                "usage as prior_usage, provider_charge_snapshot or closed compute intervals"))
        if packet_scope == "actual":
            refs = [event.get(key) for key in ("decision_ref", "evidence_ref") if key in event]
            if any(type(ref) is not str or "synthetic" in ref.casefold() or is_placeholder(ref)
                   for ref in refs):
                issues.append(_issue(
                    "synthetic_reference_in_actual_budget", f"budget.prior_events[{position}]",
                    "An actual budget's prior events need recorded, non-synthetic references"))
        if event.get("kind") == "compute_interval" and event.get("resource_id") == resource:
            end = event.get("end_epoch")
            if end is None or not _seconds(end) or not _seconds(accounted) or end > accounted:
                issues.append(_issue(
                    "prior_compute_overlaps_run", f"budget.prior_events[{position}]",
                    "The instance's prior compute must be closed at or before "
                    "accounted_until_epoch; the runner accounts for time after it"))
    if issues:
        return issues, None
    scope = {"tier": tier, "as_of_epoch": accounted, "caps_usd": budget["caps_usd"],
             "storage_retention_until_epoch": budget["storage_retention_until_epoch"]}
    snapshot = reconcile_cost(events, budget["quote"], scope)
    structural = sorted({item["code"] for item in snapshot["issues"]
                         if item["code"] in ("invalid_quote", "invalid_scope", "invalid_event")})
    if structural:
        return [_issue("budget_snapshot_invalid", "budget",
                       f"gate.s2_cost.reconcile_cost rejects the snapshot: {structural}")], None
    next_unit = {"upper_bound_usd": budget["next_unit_upper_bound_usd"],
                 "max_seconds": worst_case_s, "cleanup_seconds": budget["cleanup_seconds"],
                 "gpu_count": count}
    probe = can_start_next_unit(snapshot, budget["limits"], next_unit, now_epoch=accounted)
    if any(item["code"] == "invalid_next_unit_input" for item in probe["issues"]):
        return [_issue("invalid_budget", "budget.limits",
                       "limits must be task_allowance_usd, reserves_usd (with cleanup) and "
                       "deadline_epoch; no default is chosen")], None
    return [], {"tier": tier, "phase": phase, "events": list(events), "quote": quote,
                "scope": scope, "limits": budget["limits"], "next_unit": next_unit,
                "resource": resource, "gpu_count": count, "accounted_until": accounted,
                "rate": _decimal(quote["compute_usd_per_hour"]),
                "per_gpu": quote["rate_basis"] == "per_gpu",
                "granularity": quote["billing_granularity_seconds"],
                "snapshot_ledger_status": snapshot["status"],
                "line_caps": caps,
                "line_prior": {line_id: _decimal(value) for line_id, value in spend.items()}}


def _decimal_or_none(value):
    try:
        return _decimal(value)
    except (ValueError, DecimalException):
        return None


def _declared_line_caps(packet):
    """``{line_id: cap_usd}`` of the packet's declared diagnostic budget lines ({} if none)."""
    units = packet.get("diagnostic_units") if type(packet) is dict else None
    caps = {}
    for unit in units if type(units) is list else []:
        block = unit.get("diagnostic") if type(unit) is dict else None
        line = block.get("budget_line") if type(block) is dict else None
        if type(line) is dict and type(line.get("line_id")) is str:
            caps[line["line_id"]] = line.get("cap_usd")
    return caps


class BudgetMeter:
    """Per-unit continuation from the cost ledger; headroom shrinks unit by unit (L1).

    Each check runs ``gate.s2_cost.reconcile_cost`` over the starting snapshot's
    ``prior_events`` plus this run's own ``compute_interval`` segments for
    ``<instance_resource_id>#<run key>`` (contiguous from ``accounted_until_epoch``; the
    current one open and accrued to now), at the packet's tier, then
    ``can_start_next_unit`` with the configured worst-case unit duration. A diagnostic unit
    also needs its exposure strictly below its declared budget line cap minus the line's
    cumulative spend (carried-forward ``prior_line_spend_usd`` plus this run's segments on
    the line). Additivity holds because ``validate_budget`` refuses the snapshot kinds that
    make ``reconcile_cost``'s usage a max (G2-01). Closed segments are appended (fsync) to ``cost-events.jsonl`` for
    the operator's next reconciliation; they are runner-side accrual on the quote, which a
    later provider reconciliation of the instance supersedes (never add both).
    """

    def __init__(self, context, run_key, path):
        self.context, self.run_key = context, run_key
        self.resource = f"{context['resource']}#{run_key}"
        self.boundary = context["accounted_until"]
        self.segments, self.lines, self.clock_fault = [], [], False
        self.stream = open(path, "xb")  # noqa: SIM115 - closed by close()

    def close(self):
        self.stream.close()

    def _segment(self, end):
        return {"event_id": f"{RUN_EVENT_PREFIX}{self.run_key}:segment:{len(self.segments):04d}",
                "kind": "compute_interval", "resource_id": self.resource,
                "start_epoch": self.boundary, "end_epoch": end,
                "gpu_count": self.context["gpu_count"]}

    def _usd(self, event):
        # gate.s2_cost's own precision and conservative (ceiling) rounding.
        billed = billed_seconds(event["end_epoch"] - event["start_epoch"],
                                self.context["granularity"])
        factor = event["gpu_count"] if self.context["per_gpu"] else 1
        with localcontext(_COST_CONTEXT):
            return self.context["rate"] * factor * billed / 3600

    def check(self, now, line=None):
        """Decision for the next unit at wall time ``now`` (epoch seconds)."""
        codes, decision, ledger = [], None, None
        if self.clock_fault or not _seconds(now) or now < self.boundary:
            self.clock_fault = True
            codes.append("wall_clock_regressed")
        else:
            events = [*self.context["events"], *self.segments, self._segment(None)]
            scope = {**self.context["scope"], "as_of_epoch": now}
            ledger = reconcile_cost(events, self.context["quote"], scope)
            decision = can_start_next_unit(ledger, self.context["limits"],
                                           self.context["next_unit"], now_epoch=now)
            codes += [item["code"] for item in decision["issues"]]
        line_detail = None
        if line is not None:
            prior, run_spent = self._line_spend(line["line_id"])
            cap = _decimal(line["cap_usd"])
            exposure = None if decision is None or decision["exposure_usd"] is None else (
                Decimal(decision["exposure_usd"]))
            with localcontext(_COST_CONTEXT):
                spent = prior + run_spent
                left = cap - spent
                line_detail = {"line_id": line["line_id"], "cap_usd": line["cap_usd"],
                               "prior_spent_usd": _plain(prior),
                               "run_spent_usd": _plain(run_spent), "spent_usd": _plain(spent),
                               "remaining_usd": _plain(left)}
                if exposure is None or not exposure < left:
                    codes.append("diagnostic_budget_line_exhausted")
        return {"can_start": not codes, "issues": codes, "tier": self.context["tier"],
                "as_of_epoch": now, "budget_line": line_detail,
                "exposure_usd": None if decision is None else decision["exposure_usd"],
                "remaining_usd": None if decision is None else decision["remaining_usd"],
                "ledger_status": None if ledger is None else ledger["status"],
                "ledger_sha256": None if ledger is None else _hash_or_none(ledger),
                "run_segments": len(self.segments)}

    def _line_spend(self, line_id):
        """``(prior, this run)`` spend on one budget line (review G2-02: cumulative)."""
        with localcontext(_COST_CONTEXT):
            run_spent = sum((self._usd(event) for event, owner in zip(
                self.segments, self.lines, strict=True) if owner == line_id), Decimal(0))
        return self.context["line_prior"].get(line_id, Decimal(0)), run_spent

    def line_spend(self):
        """Per declared line: cap, carried-forward spend, this run's and the cumulative spend.

        ``cumulative_spent_usd`` is what the operator carries into the next run's
        ``prior_line_spend_usd`` for the same packet, so a line cap holds across runs.
        """
        result = {}
        for line_id, cap in sorted(self.context["line_caps"].items()):
            prior, run_spent = self._line_spend(line_id)
            with localcontext(_COST_CONTEXT):
                result[line_id] = {"cap_usd": cap, "prior_spent_usd": _plain(prior),
                                   "run_spent_usd": _plain(run_spent),
                                   "cumulative_spent_usd": _plain(prior + run_spent)}
        return result

    def close_segment(self, now, line_id=None):
        """Close the open segment at ``now`` and append it to ``cost-events.jsonl``.

        The journal line is the ``gate.s2_cost`` ``compute_interval`` plus ``line_id`` (the
        diagnostic budget line it was spent on, or null); strip ``line_id`` before passing a
        line to ``reconcile_cost``, whose event key set is exact.
        """
        if not _seconds(now) or now < self.boundary:
            self.clock_fault = True
            now = self.boundary
        event = self._segment(now)
        self.stream.write(canonical_json_bytes({**event, "line_id": line_id}) + b"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.segments.append(event)
        self.lines.append(line_id)
        self.boundary = now
        return event


# ------------------------------------------------------------------ filesystem helpers

def _read_file(path, max_bytes):
    """Read one regular file with no symlink anywhere on its path; bounded (L1)."""
    path = PurePosixPath(path)
    directory = _directory_fd(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                             | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{path} is not a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError(f"{path} exceeds the {max_bytes}-byte bound")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if len(data) > max_bytes:
        raise ValueError(f"{path} grew past the {max_bytes}-byte bound")
    return data


def _write_new(path, data):
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                         | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write_new(path, canonical_json_bytes(value) + b"\n")


def _real_directory(path):
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(metadata.st_mode)


# ------------------------------------------------------------------ real effects (L3)

class SubprocessRunner:
    """argv without a shell, closed stdin, own session; TERM/KILL escalation on timeout."""

    def __init__(self, tee=None):
        self._tee = tee if tee is not None else sys.stdout.buffer

    @staticmethod
    def _signal_group(process, sequence):
        for sig in sequence:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return process.poll()
            try:
                return process.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                continue
        return process.poll()

    def run(self, argv, *, cwd, env, stdout_path, timeout_s, merge_stderr=False):
        with contextlib.ExitStack() as stack:
            out = stack.enter_context(open(stdout_path, "xb"))
            err = (subprocess.STDOUT if merge_stderr
                   else stack.enter_context(open(f"{stdout_path}.stderr", "xb")))
            process = subprocess.Popen(
                list(argv), cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE if merge_stderr else out, stderr=err,
                start_new_session=True, close_fds=True)
            pump = None
            if merge_stderr:
                def copy():
                    for chunk in iter(lambda: process.stdout.read1(65536), b""):
                        out.write(chunk)
                        out.flush()
                        self._tee.write(chunk)
                        self._tee.flush()
                pump = threading.Thread(target=copy, daemon=True)
                pump.start()
            try:
                code = process.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                self._signal_group(process, (signal.SIGINT, signal.SIGTERM, signal.SIGKILL))
                code, timed_out = None, True
            if pump is not None:
                pump.join(KILL_GRACE_S)
            out.flush()
            os.fsync(out.fileno())
        return {"returncode": code, "timed_out": timed_out}

    def start(self, argv, *, cwd, env, stdout_path):
        # Popen duplicates the descriptor, so closing ours leaves the child's open.
        with open(stdout_path, "xb") as out:
            return subprocess.Popen(list(argv), cwd=cwd, env=dict(env),
                                    stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    close_fds=True)

    @staticmethod
    def poll(handle):
        return handle.poll()

    def stop(self, handle, *, timeout_s):
        if handle.poll() is not None:
            return {"returncode": handle.returncode, "escalated": False}
        try:
            os.killpg(handle.pid, signal.SIGINT)
        except ProcessLookupError:
            return {"returncode": handle.poll(), "escalated": False}
        try:
            return {"returncode": handle.wait(timeout=timeout_s), "escalated": False}
        except subprocess.TimeoutExpired:
            code = self._signal_group(handle, (signal.SIGTERM, signal.SIGKILL))
            return {"returncode": code, "escalated": True}


def port_open(port):
    """True when a TCP connection to 127.0.0.1:port succeeds (L3)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


class SystemClock:
    @staticmethod
    def monotonic_us():
        return time.monotonic_ns() // 1000

    @staticmethod
    def sleep(seconds):
        time.sleep(max(0.0, seconds))

    @staticmethod
    def epoch():
        return time.time()


# ------------------------------------------------------------------ the run

class _Stop(Exception):
    """A step refused; ``reason`` becomes the cancellation reason."""

    def __init__(self, reason, *, stop_run=True):
        super().__init__(reason)
        self.reason = reason
        self.stop_run = stop_run


class _Run:
    def __init__(self, packet, plan, *, journal, runner, port_open_fn, clock, meter,
                 host_env, worst_case_s, diagnostics):
        self.packet, self.plan, self.journal = packet, plan, journal
        self.runner, self.port_open, self.clock = runner, port_open_fn, clock
        self.meter, self.host_env = meter, host_env
        self.worst_case_s = worst_case_s
        self.scope = packet["scope"]
        self.input_hashes = {"packet": plan["input_hashes"]["packet"],
                             "execution_plan": _hash_or_none(plan)}
        self.events = []  # every event of this run, for per-job lookups
        self.ledger_events = {"pilot": [], "diagnostic": []}
        self.interrupt = None  # a KeyboardInterrupt/SystemExit re-raised after recording
        # Held for the whole run and closed by close(); one fsync per event. Diagnostic
        # units have their own ledger, never the pilot schedule's.
        self.ledgers = {"pilot": open(journal / "ledger-events.jsonl", "xb")}  # noqa: SIM115
        if diagnostics:
            self.ledgers["diagnostic"] = open(  # noqa: SIM115
                journal / "diagnostic-ledger-events.jsonl", "xb")
        self.target = "pilot"

    def close(self):
        for stream in self.ledgers.values():
            stream.close()

    def emit(self, event):
        stream = self.ledgers[self.target]
        stream.write(canonical_json_bytes(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        self.events.append(event)
        self.ledger_events[self.target].append(event)

    def now(self):
        return self.clock.monotonic_us()

    def env(self, command):
        try:
            return filter_environment(command, self.host_env)
        except ValueError as exc:
            raise _Stop(f"environment_refused:{exc}") from exc

    def wait_for(self, predicate, timeout_s, poll_s):
        deadline = self.now() + int(timeout_s * 1_000_000)
        while True:
            result = predicate()
            if result is not None:
                return result
            if self.now() >= deadline:
                return None
            self.clock.sleep(poll_s)

    def _interrupted(self, exc, record, state, *, stopped_compose=True):
        """Record a KeyboardInterrupt/SystemExit; the run re-raises it after recording.

        ``stopped_compose`` is False once Compose had already returned (captures,
        teardown): the interrupt then did not stop upstream, so it is not the attempt's
        cancellation reason and the terminal events follow the upstream evidence alone.
        """
        self.interrupt = self.interrupt or exc
        reason = f"interrupted:{type(exc).__name__}"
        record["stop_run"] = record["stop_run"] or reason
        record["outcome"] = "interrupted"
        if stopped_compose:
            state["interruption"] = state["interruption"] or reason

    # -------------------------------------------------------------- one unit
    def unit(self, unit_plan, unit, *, role="pilot_job"):
        """Run one unit; ``role`` is ``pilot_job`` or ``diagnostic_qualification``."""
        steps = {item["id"]: item for item in unit_plan["steps"]}
        record = {"index": unit_plan["index"], "job_id": unit_plan["job_id"],
                  "scene_id": unit_plan["scene_id"], "steps": [], "outcome": None,
                  "stop_run": None, "compose_exit_code": None, "unit_outcome": None,
                  "ledger_events": 0, "issues": []}
        job_id = unit_plan["job_id"]
        work = self.journal / steps["wizard_prepare"]["stdout_journal_path"]
        name = work.parent.name if work.parent.parent == self.journal / "units" else (
            f"{work.parent.parent.name}/{work.parent.name}")
        unit_ref = [f"unit_record:{name}"]
        diagnostic = unit_plan.get("diagnostic")
        line = diagnostic["budget_line"] if diagnostic else None
        self.target = "diagnostic" if diagnostic else "pilot"
        mark = [self.now()]

        def note(step_id, status, **detail):
            # start_us: when the runner moved on from the previous note (this step's work
            # began then); end_us: when this note is written. Both monotonic (F6).
            assert status in STATUSES, status
            end = self.now()
            record["steps"].append({"id": step_id, "status": status, "start_us": mark[0],
                                    "end_us": end, **detail})
            mark[0] = end

        budget = steps["budget_check"]
        try:
            check = self.meter.check(self.clock.epoch(), line)
        except Exception as exc:  # noqa: BLE001 - a failed budget check is a stop, not a crash
            check = {"can_start": False, "issues": [f"budget_check_failed:{type(exc).__name__}"]}
        note(budget["id"], "passed" if check["can_start"] else "blocked",
             issues=list(check["issues"]), role=role,
             decision={key: value for key, value in check.items()
                       if key not in ("can_start", "issues")},
             worst_case_unit_seconds=self.worst_case_s)
        if not check["can_start"]:
            record.update(outcome="not_started", stop_run="budget_stop")
            return self._envelope(record, diagnostic)

        started = steps["ledger_started"]
        self.emit(started_event(job_id, 1, self.now(), started["evidence_refs"]))
        note(started["id"], "passed", attempt_id=started["attempt_id"])
        state = {"driver": None, "compose_attempted": False, "terminal": False,
                 "teardown_ok": None, "interruption": None, "manifest": None}
        try:
            try:
                self._prepare(steps, note, work)
                self._gpu(steps, note, state, record)
            except _Stop as stop:
                record["stop_run"] = stop.reason if stop.stop_run else None
                if not state["compose_attempted"]:
                    self.emit(cancelled_event(job_id, 1, self.now(), stop.reason, unit_ref))
                    state["terminal"] = True
                record["outcome"] = "blocked"
            except Exception as exc:  # noqa: BLE001 - recorded, captured, torn down, run stops
                reason = f"internal_error:{type(exc).__name__}"
                record.update(stop_run=reason, outcome="internal_error")
                state["interruption"] = reason
                note("internal_error", "blocked", error=type(exc).__name__)
            except BaseException as exc:  # noqa: BLE001 - Ctrl-C: capture, tear down, re-raise
                self._interrupted(exc, record, state)
            # Every exit path after Compose started keeps logs/ps/inspect and the run-dir
            # hash before teardown destroys the containers.
            if state["compose_attempted"]:
                self._captures(steps, note, state)
        except BaseException as exc:  # noqa: BLE001 - an interrupt during the captures
            self._interrupted(exc, record, state, stopped_compose=False)
        finally:
            if state["driver"] is not None or state["compose_attempted"]:
                try:
                    state["teardown_ok"] = self._teardown(steps, note, state)
                except BaseException as exc:  # noqa: BLE001 - teardown itself interrupted
                    state["teardown_ok"] = False
                    note("teardown", "blocked", error=type(exc).__name__)
                    if not isinstance(exc, Exception):
                        self._interrupted(exc, record, state, stopped_compose=False)
                if not state["teardown_ok"]:
                    record["stop_run"] = record["stop_run"] or "teardown_unverified"
        if not state["terminal"]:
            try:
                if state["compose_attempted"]:
                    self._record_attempts(steps, note, state, record, job_id)
                else:
                    self.emit(cancelled_event(job_id, 1, self.now(),
                                              record["stop_run"] or "internal_error", unit_ref))
                    state["terminal"] = True
            except BaseException as exc:  # noqa: BLE001 - the attempt still gets a terminal
                reason = f"recorder_error:{type(exc).__name__}"
                note("reconcile", "blocked", error=type(exc).__name__)
                if not isinstance(exc, Exception):
                    self._interrupted(exc, record, state)
                if not state["terminal"]:
                    self.emit(cancelled_event(job_id, 1, self.now(),
                                              state["interruption"] or reason, unit_ref))
                    state["terminal"] = True
                record["stop_run"] = record["stop_run"] or reason
        if record["outcome"] is None:
            record["outcome"] = "executed"
        record["ledger_events"] = sum(1 for event in self.events if event["job_id"] == job_id)
        return self._envelope(record, diagnostic)

    def _envelope(self, record, diagnostic=None):
        """Contract 2.1 envelope around one unit record; ``outcome`` stays the domain value.

        ``passed`` only when the unit executed without a stop, every step other than
        ``compose_up`` is ``passed``/``not_applicable``/``qualify_on_host``, there is no
        input issue and every terminal ledger event of the job is established
        (``gate.s2_executor.terminal_established``: completed, or a log-backed upstream
        failure). ``compose_up`` stays ``unverified`` by design (its exit code is never
        success evidence; reconcile and the terminal events decide). A blocked or skipped
        step, a stop or a non-executed outcome is ``blocked``; an unverified step, an input
        issue or an unestablished terminal is ``unverified``. ``status_reasons`` names each.

        A declared diagnostic unit's record (``diagnostic`` block given) is a
        ``DIAGNOSTIC_RECORD_KIND`` with top-level ``role`` ``diagnostic_qualification``,
        ``inference_input: false`` and its ``diagnostic_id`` (review G2-04), so it is never
        read as an ``s2_unit_record`` of a pilot job.
        """
        blocking, unverified = [], []
        if record["outcome"] == "not_started":
            status = "not_run"
            blocking.append(f"stop_run:{record['stop_run']}")
        else:
            if record["outcome"] != "executed":
                blocking.append(f"outcome:{record['outcome']}")
            if record["stop_run"]:
                blocking.append(f"stop_run:{record['stop_run']}")
            for step in record["steps"]:
                if step["id"] == "compose_up":
                    continue
                if step["status"] in ("blocked", "not_run"):
                    blocking.append(f"step_{step['status']}:{step['id']}")
                elif step["status"] in ("unverified", "not_estimable"):
                    unverified.append(f"step_{step['status']}:{step['id']}")
            unverified += [f"issue:{code}"
                           for code in sorted({item["code"] for item in record["issues"]})]
            terminal = [event for event in self.events if event["job_id"] == record["job_id"]
                        and event["type"] != "started"]
            if not terminal:
                unverified.append("terminal_absent")
            unverified += [f"terminal_not_established:{event['attempt_id']}:{event['type']}:"
                           f"{event.get('reason')}"
                           for event in terminal if not terminal_established(event)]
            status = "blocked" if blocking else "unverified" if unverified else "passed"
        markers = {} if diagnostic is None else {
            "role": DIAGNOSTIC_ROLE, "inference_input": False,
            "diagnostic_id": diagnostic["diagnostic_id"]}
        return {"artifact_kind": DIAGNOSTIC_RECORD_KIND if diagnostic else UNIT_RECORD_KIND,
                "schema_version": UNIT_RECORD_VERSION, **markers, "scope": self.scope,
                "producer": {"module": "scripts.s2_run_units", "function": "_Run.unit",
                             "source_sha256": _SOURCE_SHA256},
                "input_hashes": dict(self.input_hashes),
                "units": {"timeout_s": "seconds", "timestamp_us": "monotonic microseconds"},
                "status": status, "status_reasons": blocking + unverified, **record}

    def _prepare(self, steps, note, work):
        absent = steps["run_dirs_absent"]
        existing = [path for path in absent["paths"] if os.path.lexists(path)]
        if existing:
            note(absent["id"], "blocked", existing=existing)
            raise _Stop("run_dir_exists")
        work.parent.mkdir(parents=True, exist_ok=False)
        Path(absent["paths"][1]).mkdir(parents=True, exist_ok=False)
        note(absent["id"], "passed")

        wizard = steps["wizard_prepare"]
        result = self.runner.run(wizard["command"]["argv"], cwd=wizard["command"]["cwd"],
                                 env=self.env(wizard["command"]), stdout_path=str(work),
                                 timeout_s=wizard["timeout_s"])
        if result.get("timed_out") or result.get("returncode") != 0:
            note(wizard["id"], "blocked", outcome="failed", returncode=result.get("returncode"),
                 timed_out=bool(result.get("timed_out")))
            raise _Stop("wizard_prepare_failed")
        run_dir = steps["generated_config_check"]["run_dir"]
        if not _real_directory(run_dir):
            note(wizard["id"], "blocked", outcome="failed", reason="run_dir_not_created")
            raise _Stop("wizard_prepare_failed")
        Path(run_dir, "physloop-capture").mkdir(exist_ok=False)
        note(wizard["id"], "passed", returncode=0)

        check = steps["generated_config_check"]
        try:
            generated = yaml.safe_load(_read_file(
                f"{run_dir}/{check['generated_user_config']}", MAX_CONFIG_BYTES))
            driver = yaml.safe_load(_read_file(f"{run_dir}/{check['driver_config']}",
                                               MAX_CONFIG_BYTES))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            note(check["id"], "blocked", error=type(exc).__name__)
            raise _Stop("generated_config_unreadable") from exc
        report = check_generated_configs(generated, driver, check["expected"])
        _write_json(check["report_path"], report)
        if report["status"] != "passed":
            codes = sorted({item["code"] for item in report["issues"]
                            if item["severity"] == "blocking"})
            note(check["id"], "blocked", codes=codes)
            raise _Stop("generated_config_blocked:" + ",".join(codes))
        note(check["id"], "passed")

        override = steps["compose_shm_override"]
        present = [name for name in override["conflicting_names"]
                   if os.path.lexists(f"{override['run_dir']}/{name}")]
        if present:
            note(override["id"], "blocked", present=present)
            raise _Stop("compose_override_preexisting")
        if override["content"] is None:
            note(override["id"], "not_applicable")
        else:
            _write_new(f"{override['run_dir']}/{override['name']}",
                       yaml.safe_dump(override["content"], sort_keys=True,
                                      default_flow_style=False).encode("utf-8"))
            note(override["id"], "passed", services=sorted(override["content"]["services"]))

        config = steps["compose_config_check"]
        command = config["command"]
        result = self.runner.run(command["argv"], cwd=command["cwd"], env=self.env(command),
                                 stdout_path=command["stdout_path"],
                                 timeout_s=config["timeout_s"])
        if result.get("timed_out") or result.get("returncode") != 0:
            note(config["id"], "blocked", returncode=result.get("returncode"))
            raise _Stop("compose_config_failed")
        try:
            effective = read_json_bytes(_read_file(command["stdout_path"], MAX_CONFIG_BYTES))
        except (OSError, ValueError) as exc:
            note(config["id"], "blocked", error=type(exc).__name__)
            raise _Stop("compose_config_unreadable") from exc
        report = check_generated_compose(effective, config["expected"])
        _write_json(config["report_path"], report)
        if report["status"] != "passed":
            codes = sorted({item["code"] for item in report["issues"]
                            if item["severity"] == "blocking"})
            note(config["id"], "blocked", codes=codes)
            raise _Stop("compose_config_blocked:" + ",".join(codes))
        note(config["id"], "passed",
             warnings=sorted({item["code"] for item in report["issues"]}))

        free = steps["driver_port_free"]
        if self.port_open(free["port"]):
            note(free["id"], "blocked")
            raise _Stop("driver_port_in_use")
        note(free["id"], "passed")

    def _gpu(self, steps, note, state, record):
        start = steps["driver_start"]
        command = start["command"]
        state["driver"] = self.runner.start(command["argv"], cwd=command["cwd"],
                                            env=self.env(command),
                                            stdout_path=start["stdout_path"])
        note(start["id"], "passed")

        ready = steps["driver_ready"]

        def probe():
            if self.runner.poll(state["driver"]) is not None:
                return "exited"
            if not self.port_open(ready["port"]):
                return None
            try:
                text = _read_file(ready["log_path"], MAX_LOG_BYTES).decode("utf-8", "replace")
            except (OSError, ValueError):
                return None
            return "ready" if driver_ready_line(text, ready["port"]) else None

        outcome = self.wait_for(probe, ready["timeout_s"], ready["poll_interval_s"])
        if outcome != "ready":
            reason = "driver_exited_before_ready" if outcome == "exited" else "driver_not_ready"
            note(ready["id"], "blocked", reason=reason)
            raise _Stop(reason)
        note(ready["id"], "passed")

        up = steps["compose_up"]
        command = up["command"]
        env = self.env(command)
        state["compose_attempted"] = True
        result = self.runner.run(command["argv"], cwd=command["cwd"], env=env,
                                 stdout_path=up["stdout_path"], timeout_s=up["timeout_s"],
                                 merge_stderr=True)
        record["compose_exit_code"] = result.get("returncode")
        if result.get("timed_out"):
            state["interruption"] = "compose_up_timeout"
        # Recorded, never verified here: success is established only by reconcile.
        note(up["id"], "unverified", outcome="recorded", returncode=result.get("returncode"),
             timed_out=bool(result.get("timed_out")), exit_code_is_success_evidence=False)

    def _captures(self, steps, note, state):
        """Bounded captures before teardown; each step's failure is recorded, never raised.

        A KeyboardInterrupt/SystemExit inside a docker capture is recorded on that step;
        the remaining docker captures are recorded ``not_run`` (no new subprocess after an
        interrupt), the local run-metadata read and run-dir hash still run, and the first
        interrupt is re-raised at the end, before teardown.
        """
        ids, interrupt = None, None
        for step_id in ("capture:compose_up_stdout", "capture:compose_logs",
                        "capture:compose_ps", "capture:compose_container_ids",
                        "capture:container_inspect"):
            step = steps[step_id]
            capture = step["capture"]
            if interrupt is not None:
                note(step_id, "not_run", outcome="skipped", reason="interrupted")
                continue
            try:
                ids = self._capture(step_id, step, capture, ids, note)
            except Exception as exc:  # noqa: BLE001 - a failed capture is evidence
                note(step_id, "blocked", outcome="failed", error=type(exc).__name__)
            except BaseException as exc:  # noqa: BLE001 - Ctrl-C: record, hash, re-raise
                note(step_id, "blocked", outcome="interrupted", error=type(exc).__name__)
                interrupt = exc
        metadata = steps["run_metadata_capture"]
        try:
            data = _read_file(f"{metadata['run_dir']}/{metadata['name']}", MAX_CONFIG_BYTES)
            note(metadata["id"], "passed", outcome="recorded",
                 sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        except Exception as exc:  # noqa: BLE001 - absent metadata is evidence, not a crash
            note(metadata["id"], "unverified", outcome="missing", error=type(exc).__name__)
        except BaseException as exc:  # noqa: BLE001 - Ctrl-C: the run dir is still hashed
            note(metadata["id"], "blocked", outcome="interrupted", error=type(exc).__name__)
            interrupt = interrupt or exc
        digest = steps["run_dir_hash"]
        try:
            manifest = build_manifest(digest["root"], scope=self.scope, symlink_policy="record")
            _write_json(digest["manifest_path"], manifest)
        except BaseException as exc:  # noqa: BLE001 - without a stored manifest nothing is bound
            note(digest["id"], "blocked", error=type(exc).__name__)
            if not isinstance(exc, Exception):
                interrupt = interrupt or exc
        else:
            # Reconcile consumes only bytes these receipts hashed before teardown (2.4).
            state["manifest"] = manifest
            note(digest["id"], manifest["status"], files=manifest["totals"],
                 files_sha256=manifest["files_sha256"],
                 issues=sorted({item["code"] for item in manifest["issues"]}))
        if interrupt is not None:
            raise interrupt

    def _capture(self, step_id, step, capture, ids, note):
        """One capture step; returns the container IDs known after it (L1 with fakes)."""
        if capture["id"] == "compose_up_stdout":
            present = os.path.isfile(capture["stdout_path"])
            note(step_id, "passed" if present else "unverified",
                 outcome="recorded" if present else "missing")
            return ids
        if capture["id"] == "container_inspect" and ids is None:
            note(step_id, "not_run", outcome="skipped", reason="no_container_ids")
            return ids
        try:
            argv = (inspect_argv(capture, ids) if capture["id"] == "container_inspect"
                    else capture["argv"])
            env = self.env(capture)
            result = self.runner.run(argv, cwd=capture["cwd"], env=env,
                                     stdout_path=capture["stdout_path"],
                                     timeout_s=step["timeout_s"])
        except Exception as exc:  # noqa: BLE001 - a failed capture is evidence
            note(step_id, "blocked", outcome="failed", error=type(exc).__name__)
            return ids
        ok = result.get("returncode") == 0 and not result.get("timed_out")
        note(step_id, "passed" if ok else "blocked",
             outcome="recorded" if ok else "failed", returncode=result.get("returncode"))
        if capture["id"] == "compose_container_ids" and ok:
            try:
                ids = container_ids(_read_file(capture["stdout_path"], MAX_CONFIG_BYTES)
                                    .decode("utf-8", "strict"))
            except (OSError, ValueError, UnicodeError):
                ids = None
        return ids

    def _teardown(self, steps, note, state):
        """Compose down, then (always, via finally) driver stop and port-release check.

        An exception or interrupt during compose down is recorded on that step and re-raised
        only after the driver was stopped and its port release verified and recorded.
        """
        try:
            down_ok = self._compose_down(steps, note) if state["compose_attempted"] else True
        finally:
            driver_ok = self._stop_driver(steps, note, state)
        return down_ok and driver_ok

    def _compose_down(self, steps, note):
        down = steps["compose_down"]
        command = down["command"]
        try:
            result = self.runner.run(command["argv"], cwd=command["cwd"],
                                     env=self.env(command),
                                     stdout_path=command["stdout_path"],
                                     timeout_s=down["timeout_s"])
            down_ok = result.get("returncode") == 0 and not result.get("timed_out")
        except (_Stop, OSError) as exc:
            result, down_ok = {"error": type(exc).__name__}, False
        except BaseException as exc:  # recorded; re-raised after the driver stop
            note(down["id"], "blocked", outcome="failed", error=type(exc).__name__)
            raise
        note(down["id"], "passed" if down_ok else "blocked",
             returncode=result.get("returncode"))
        return down_ok

    def _stop_driver(self, steps, note, state):
        ok = True
        if state["driver"] is not None:
            stop = steps["driver_stop"]
            result = self.runner.stop(state["driver"], timeout_s=stop["timeout_s"])
            # A process that survived SIGINT, SIGTERM and SIGKILL is not torn down.
            exited = result.get("returncode") is not None
            note(stop["id"], "passed" if exited else "blocked",
                 outcome="exited" if exited else "process_survived_signals",
                 returncode=result.get("returncode"), escalated=bool(result.get("escalated")))
            ok = ok and exited
            note("gpu_memory_released", "qualify_on_host",
                 reason="GPU memory release after driver exit is checked only on the host")
            released = steps["driver_port_released"]
            closed = self.wait_for(lambda: True if not self.port_open(released["port"]) else None,
                                   released["timeout_s"], released["poll_interval_s"])
            note(released["id"], "passed" if closed else "blocked")
            ok = ok and bool(closed)
            state["driver"] = None
        return ok

    def _verified_inputs(self, run_dir, manifest, logs_wanted, issues):
        """Bytes of the reconcile inputs, each verified against the pre-teardown receipt."""
        receipts = {} if manifest is None else {item["path"]: item for item in manifest["files"]}

        def read(relative, max_bytes):
            receipt = receipts.get(relative)
            if receipt is None:
                return None
            if receipt["size_bytes"] > max_bytes:
                issues.append(_issue("capture_exceeds_bound", relative,
                                     f"Receipted size exceeds {max_bytes} bytes"))
                return None
            try:
                return read_artifact(run_dir, receipt)
            except (OSError, ValueError) as exc:
                issues.append(_issue("capture_receipt_mismatch", relative,
                                     "Bytes differ from the receipt hashed before teardown: "
                                     f"{type(exc).__name__}"))
                return None

        logs = []
        for capture_id, relative in logs_wanted:
            data = read(relative, MAX_LOG_BYTES)
            if data is not None:
                logs.append({"source": capture_id, "text": data.decode("utf-8", "replace")})
        metadata = read("run_metadata.yaml", MAX_CONFIG_BYTES)
        if metadata is not None:
            try:
                metadata = metadata.decode("utf-8", "strict")
            except UnicodeError:
                issues.append(_issue("capture_unreadable", "run_metadata.yaml",
                                     "run_metadata.yaml is not UTF-8"))
                metadata = None
        return logs, sorted(receipts), metadata, read

    def _record_attempts(self, steps, note, state, record, job_id):
        step = steps["reconcile"]
        run_dir = step["run_dir"]
        issues = []
        manifest = state["manifest"]
        if manifest is None:
            issues.append(_issue("run_dir_manifest_absent", "physloop-capture/run-dir-manifest.json",
                                 "No run-dir manifest was stored before teardown; no captured "
                                 "byte can be verified, so none is reconciled"))
        logs_wanted = []
        for capture_id in ("capture:compose_up_stdout", "capture:compose_logs"):
            capture = steps[capture_id]["capture"]
            logs_wanted.append((capture["id"], str(PurePosixPath(capture["stdout_path"])
                                                   .relative_to(run_dir))))
        logs, files, metadata, read = self._verified_inputs(run_dir, manifest, logs_wanted,
                                                            issues)
        summary = None
        if step["summary_name"] in files:
            data = read(step["summary_name"], MAX_LOG_BYTES)
            if data is not None:
                try:
                    summary = read_json_bytes(data)
                except ValueError:
                    summary = {"unreadable": True}
        evidence = {"scope": self.scope, "scene_id": step["scene_id"],
                    "requested_steps": step["requested_steps"], "logs": logs,
                    "rollout_files": files, "run_metadata": metadata, "summary": summary,
                    "compose_exit_code": record["compose_exit_code"]}
        report = reconcile_unit_attempts(evidence)
        _write_json(step["report_path"], report)
        ref = f"unit_attempt_evidence:{identity_hash(report)}"
        record["issues"].extend(issues)
        note(step["id"], "unverified" if issues else "passed", outcome="recorded",
             unit_outcome=report["unit_outcome"], report_status=report["status"],
             evidence_ref=ref, input_issues=sorted({item["code"] for item in issues}),
             codes=sorted({item["code"] for item in report["issues"]}))
        events = unit_terminal_events(job_id, report, self.now(), ref, log_count=len(logs),
                                      interruption=state["interruption"])
        for event in events:
            self.emit(event)
        state["terminal"] = True
        note(steps["ledger_terminal"]["id"], "passed", outcome="recorded",
             attempts=[{"attempt_id": event["attempt_id"], "type": event["type"],
                        "reason": event["reason"]}
                       for event in events if event["type"] != "started"])
        record["unit_outcome"] = report["unit_outcome"]
        # A unit with no upstream attempt evidence at all (e.g. Compose rejecting its own
        # argv, or a Compose timeout before any runtime line) is systemic, not a failed
        # rollout: stop instead of repeating it on every paid unit. The ledger event keeps
        # its own reason (e.g. cancelled compose_up_timeout). Agent technical default
        # (D-018); an owner rule may relax it.
        if not has_upstream_attempt_evidence(report):
            record["stop_run"] = record["stop_run"] or "no_upstream_attempt_evidence"


def is_fixture_runner(runner):
    """True only for a runner object that declares ``fixture_only = True`` (a CPU fake).

    Anything else, including ``SubprocessRunner``, subclasses and undeclared duck types, is
    treated as a real runner, so scope and authority refusals cannot be skipped by a
    caller-supplied flag (fails closed).
    """
    return getattr(runner, "fixture_only", False) is True


def _utc(epoch):
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="microseconds")


def _selected(plan, unit_selection, packet):
    """``(role, unit_plan, unit)`` in execution order: declared diagnostics, then jobs."""
    diagnostics = list(zip(plan.get("diagnostic_unit_plans", []),
                           packet.get("diagnostic_units", []), strict=True))
    chosen = [("diagnostic_qualification", unit_plan, unit) for unit_plan, unit in diagnostics]
    if unit_selection == "all":
        chosen += [("pilot_job", unit_plan, unit) for unit_plan, unit in
                   zip(plan["unit_plans"], packet["units"], strict=True)]
    return chosen


def preflight(packet, *, authority, run_config, budget, real_runner, unit_selection,
              clock_uncertainty_us, decisions_md=None, decision_id=None, readiness_report=None,
              release_record=None, release_decision_id=None):
    """Every input check ``run_units`` applies before any effect; also the dry run (L1).

    Review G2-05: the CLI dry run calls this exact function, so it never says
    ``validated`` for inputs ``--execute`` refuses. Returns ``{plan, issues, authority,
    worst_case_s, context}``; ``issues`` empty means the run may start.
    """
    plan = build_execution_plan(packet, run_config)
    result = {"plan": plan, "issues": [], "authority": None, "worst_case_s": None,
              "context": None}
    if plan["status"] != "passed":
        result["issues"] = list(plan["issues"])
        return result
    issues = []
    if unit_selection not in UNIT_SELECTIONS:
        issues.append(_issue("invalid_unit_selection", "unit_selection",
                             f"Choose one of {list(UNIT_SELECTIONS)}; no default"))
    elif unit_selection == "diagnostics_only" and not plan.get("diagnostic_unit_plans"):
        issues.append(_issue("no_declared_diagnostics", "unit_selection",
                             "diagnostics_only needs declared diagnostic units"))
    if type(clock_uncertainty_us) is not int or clock_uncertainty_us < 0:
        issues.append(_issue("clock_uncertainty_unset", "clock_uncertainty_us",
                             "The operator's clock uncertainty bound (integer microseconds "
                             ">= 0) is required; no default"))
    found, result["authority"] = verify_run_authority(
        packet, authority, real_runner=real_runner, unit_selection=unit_selection,
        decisions_md=decisions_md, decision_id=decision_id,
        readiness_report=readiness_report, release_record=release_record,
        release_decision_id=release_decision_id)
    issues += found
    result["worst_case_s"] = worst_case_unit_seconds(run_config, kill_grace_s=KILL_GRACE_S)
    found, result["context"] = validate_budget(budget, packet,
                                               worst_case_s=result["worst_case_s"])
    issues += found
    result["issues"] = issues
    return result


def run_units(packet, *, authority, run_config, journal_dir, runner, port_open_fn, clock,
              budget, host_env, unit_selection, clock_uncertainty_us, decisions_md=None,
              decision_id=None, readiness_report=None, release_record=None,
              release_decision_id=None):
    """Execute the selected units in order; return and write the run summary (L1 with fakes).

    ``budget`` is the operator's starting ``s2_unit_budget`` snapshot (``validate_budget``;
    read once, never re-read per unit): every unit's budget check runs on the cost ledger
    (``BudgetMeter``), so headroom shrinks unit by unit. ``unit_selection`` is ``all``
    (declared diagnostic units first, then every packet unit) or ``diagnostics_only``
    (requires declared diagnostics); no default. ``clock_uncertainty_us`` is the operator's
    bound (integer microseconds, no default) recorded in the clock anchor. Authority:
    ``verify_run_authority``. Any refusal happens before the journal directory exists.

    ``status`` follows the shared vocabulary: ``passed`` only when the run completed and
    every unit record (diagnostic and pilot) is ``passed`` (``_Run._envelope``),
    ``unverified`` when it completed with only unverified units, else ``blocked``;
    ``status_reasons`` names each non-passed unit and a non-completed outcome.
    ``run_outcome`` keeps the domain value (``completed``, ``stopped``, ``interrupted``,
    ``refused``). A KeyboardInterrupt or SystemExit is recorded (captures, teardown,
    terminal event, partial summary) and re-raised.
    """
    real = not is_fixture_runner(runner)
    checked = preflight(packet, authority=authority, run_config=run_config, budget=budget,
                        real_runner=real, unit_selection=unit_selection,
                        clock_uncertainty_us=clock_uncertainty_us, decisions_md=decisions_md,
                        decision_id=decision_id, readiness_report=readiness_report,
                        release_record=release_record,
                        release_decision_id=release_decision_id)
    plan = checked["plan"]
    summary = {
        "artifact_kind": ARTIFACT_KIND, "schema_version": SCHEMA_VERSION,
        "scope": plan["scope"],
        "producer": {"module": "scripts.s2_run_units", "source_sha256": _SOURCE_SHA256},
        "input_hashes": {**plan["input_hashes"], "authority": _hash_or_none(authority),
                         "execution_plan": _hash_or_none(plan),
                         "budget": _hash_or_none(budget),
                         "decisions_md": hashlib.sha256(decisions_md).hexdigest()
                         if type(decisions_md) is bytes else None,
                         "readiness_report": _hash_or_none(readiness_report),
                         "release_record": _hash_or_none(release_record)},
        "units": {"timestamp_us": "monotonic microseconds", "worst_case_unit": "seconds",
                  "epoch": "UTC seconds (budget only)", "usd": "decimal strings"},
        "status": "blocked", "status_reasons": ["run_outcome:refused"],
        "run_outcome": "refused", "issues": [], "unit_selection": unit_selection,
        "authority": None, "clock_anchor": None, "budget": None,
        "unit_records": [], "diagnostic_unit_records": [],
        "stop_reason": None, "not_run_job_ids": [], "not_run_diagnostic_job_ids": [],
        "attempt_ids": [], "ledger": None, "diagnostic_ledger": None,
        "worst_case_unit_seconds": None, "runtime_qualified": False,
        "score_admitted": False, "cloud_launch_authorized": False,
    }
    summary["authority"] = checked["authority"]
    if checked["issues"]:
        summary["issues"] = checked["issues"]
        return summary
    worst_case_s, context = checked["worst_case_s"], checked["context"]
    summary["worst_case_unit_seconds"] = worst_case_s
    journal = Path(journal_dir).absolute()
    selected = _selected(plan, unit_selection, packet)
    run_key = identity_hash({"execution_plan": summary["input_hashes"]["execution_plan"],
                             "budget": summary["input_hashes"]["budget"],
                             "unit_selection": unit_selection, "journal": str(journal)})[:16]
    journal.mkdir(parents=True, exist_ok=False)
    monotonic_us = clock.monotonic_us()
    summary["clock_anchor"] = {"clock_id": f"s2_run_units:{run_key}",
                               "monotonic_us": monotonic_us, "utc": _utc(clock.epoch()),
                               "uncertainty_us": clock_uncertainty_us,
                               "anchor_ref": "run-summary.json#clock_anchor"}
    meter = BudgetMeter(context, run_key, journal / "cost-events.jsonl")
    summary["budget"] = {"tier": context["tier"], "phase": context["phase"],
                         "starting_snapshot_sha256": summary["input_hashes"]["budget"],
                         "snapshot_ledger_status": context["snapshot_ledger_status"],
                         "run_resource_id": meter.resource,
                         "accounted_until_epoch": context["accounted_until"],
                         "cost_events_path": "cost-events.jsonl", "run_segments": None,
                         "basis": "gate.s2_cost reconcile_cost + can_start_next_unit per unit"}
    run = _Run(packet, plan, journal=journal, runner=runner, port_open_fn=port_open_fn,
               clock=clock, meter=meter, host_env=host_env, worst_case_s=worst_case_s,
               diagnostics="diagnostic_unit_plans" in plan)
    open_line = None  # budget line of the segment still open (the unit in flight)
    try:
        for role, unit_plan, unit in selected:
            diagnostic = role == "diagnostic_qualification"
            line = unit_plan["diagnostic"]["budget_line"]["line_id"] if diagnostic else None
            # An exception escaping run.unit leaves this unit's spend on its line (G2R).
            open_line = line
            record = run.unit(unit_plan, unit, role=role)
            summary["diagnostic_unit_records" if diagnostic else "unit_records"].append(record)
            if record["outcome"] != "not_started":
                meter.close_segment(clock.epoch(), line)
            open_line = None
            if record["outcome"] != "not_started":
                work = unit_plan["steps"][UNIT_STEPS.index("wizard_prepare")][
                    "stdout_journal_path"]
                unit_dir = journal / PurePosixPath(work).parent
                unit_dir.mkdir(parents=True, exist_ok=True)
                _write_json(unit_dir / "unit-record.json", record)
            if record["stop_run"]:
                summary["stop_reason"] = {"index": unit_plan["index"],
                                          "job_id": unit_plan["job_id"], "role": role,
                                          "reason": record["stop_run"]}
                break
    finally:
        run.close()
        try:
            if open_line is not None:  # a unit's exception escaped: close its segment
                meter.close_segment(clock.epoch(), open_line)
            meter.close_segment(clock.epoch(), None)  # the tail after the last unit
        finally:
            meter.close()
        summary["budget"]["run_segments"] = len(meter.segments)
        summary["budget"]["clock_fault"] = meter.clock_fault
        summary["budget"]["line_spend"] = meter.line_spend()
        executed = {record["job_id"] for record in
                    summary["unit_records"] + summary["diagnostic_unit_records"]
                    if record["outcome"] != "not_started"}
        summary["not_run_job_ids"] = [unit["job_id"] for unit in plan["unit_plans"]
                                      if unit["job_id"] not in executed]
        summary["not_run_diagnostic_job_ids"] = [
            unit["job_id"] for unit in plan.get("diagnostic_unit_plans", [])
            if unit["job_id"] not in executed]
        summary["ledger"] = reconcile_attempts(packet["schedule"], run.ledger_events["pilot"])
        if "diagnostic_unit_plans" in plan:
            summary["diagnostic_ledger"] = reconcile_attempts(
                plan["diagnostic_schedule"], run.ledger_events["diagnostic"])
        summary["attempt_ids"] = sorted({event["attempt_id"] for event in run.events})
        records = summary["diagnostic_unit_records"] + summary["unit_records"]
        if run.interrupt is not None:
            summary["run_outcome"] = "interrupted"
        elif summary["stop_reason"] is not None:
            summary["run_outcome"] = "stopped"
        elif len(records) == len(selected):
            summary["run_outcome"] = "completed"
        else:
            summary["run_outcome"] = "interrupted"  # an unexpected error left the loop
        # Contract 2.2: passed only when the run completed and every unit record passed.
        reasons = [] if summary["run_outcome"] == "completed" else [
            f"run_outcome:{summary['run_outcome']}"]
        reasons += [f"diagnostic:{record['index']}:{record['status']}"
                    for record in summary["diagnostic_unit_records"]
                    if record["status"] != "passed"]
        reasons += [f"unit:{record['index']}:{record['status']}"
                    for record in summary["unit_records"] if record["status"] != "passed"]
        summary["status_reasons"] = reasons
        if not reasons:
            summary["status"] = "passed"
        elif summary["run_outcome"] == "completed" and all(
                record["status"] in ("passed", "unverified") for record in records):
            summary["status"] = "unverified"
        else:
            summary["status"] = "blocked"
        _write_json(journal / "run-summary.json", summary)
    if run.interrupt is not None:
        raise run.interrupt
    return summary


def _hash_or_none(value):
    try:
        return identity_hash(value)
    except (TypeError, ValueError, UnicodeError):
        return None


def _read_json_file(path):
    return None if path is None else read_json_bytes(Path(path).read_bytes())


def main(argv=None, *, runner=None, port_open_fn=None, clock=None, environ=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packet")
    parser.add_argument("--authority", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--budget-file", required=True,
                        help="s2_unit_budget starting snapshot, read once")
    parser.add_argument("--journal-dir", required=True)
    parser.add_argument("--unit-selection", required=True, choices=UNIT_SELECTIONS)
    parser.add_argument("--clock-uncertainty-us", required=True, type=int)
    parser.add_argument("--decisions", help="docs/DECISIONS.md bytes (required to execute)")
    parser.add_argument("--decision-id")
    parser.add_argument("--readiness-report")
    parser.add_argument("--release-record")
    parser.add_argument("--release-decision-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    packet = read_json_bytes(Path(args.packet).read_bytes())
    authority = read_json_bytes(Path(args.authority).read_bytes())
    run_config = read_json_bytes(Path(args.run_config).read_bytes())
    budget = read_json_bytes(Path(args.budget_file).read_bytes())
    inputs = {"decisions_md": None if args.decisions is None
              else Path(args.decisions).read_bytes(),
              "decision_id": args.decision_id,
              "readiness_report": _read_json_file(args.readiness_report),
              "release_record": _read_json_file(args.release_record),
              "release_decision_id": args.release_decision_id}
    # Real unless the supplied runner declares itself a fixture (B1-F14; fails closed).
    real = runner is None or not is_fixture_runner(runner)
    # Review G2-05: the dry run applies exactly run_units' input checks (``preflight``).
    checked = preflight(packet, authority=authority, run_config=run_config, budget=budget,
                        real_runner=real, unit_selection=args.unit_selection,
                        clock_uncertainty_us=args.clock_uncertainty_us, **inputs)
    issues = checked["issues"]
    if issues or not args.execute:
        # Without --decisions a real-runner dry run cannot verify authority; it says so and
        # never reports "validated". Every other refusal is printed as such.
        unverified = not args.execute and real and any(
            item["code"] == "decision_record_missing" for item in issues)
        refusals = [item for item in issues
                    if not (unverified and item["code"] == "decision_record_missing")]
        for item in refusals:
            print(f"refused: {item['code']} at {item['path']}: {item['message']}")
        if not refusals:
            for role, unit, _unit in _selected(checked["plan"], args.unit_selection, packet):
                print(f"{role} {unit['index']}: {unit['job_id']} {unit['scene_id']} "
                      f"({len(unit['steps'])} steps); first attempt "
                      f"{attempt_id(unit['job_id'], 1)}")
            if unverified:
                print("authority NOT verified (supply --decisions and --decision-id); "
                      "not validated; nothing executed")
            else:
                if not real:
                    print("authority mode fixture_runner: no recorded decision verified")
                print("validated; nothing executed (pass --execute)")
        return 2 if issues else 0
    clock = clock or SystemClock()
    summary = run_units(packet, authority=authority, run_config=run_config,
                        journal_dir=args.journal_dir, runner=runner or SubprocessRunner(),
                        port_open_fn=port_open_fn or port_open, clock=clock, budget=budget,
                        host_env=dict(os.environ if environ is None else environ),
                        unit_selection=args.unit_selection,
                        clock_uncertainty_us=args.clock_uncertainty_us, **inputs)
    print(f"{summary['status']} ({summary['run_outcome']}): "
          f"{len(summary['unit_records'])} unit records, "
          f"{len(summary['diagnostic_unit_records'])} diagnostic; stop {summary['stop_reason']}")
    if summary["run_outcome"] == "refused":
        return 2
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
