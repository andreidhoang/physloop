"""S2 bounded rental lifecycle: admission, journal, state machine and Vast adapter.

Readiness CPU-10 (lifecycle and recovery) and runbook section 4 steps 1-3 and 9-10;
build-spec A3 (P0-P2 of lane R1). Serves S2.2/S2.3/S2.7 [C] (G2, G7) under
D-018/D-024. Agent-authored evaluator engineering.

Pieces and verification ceiling (build-spec section 1):

- ``validate_action_record`` - L1. Pre-create admission (P0): every field required,
  no defaults, placeholders rejected, nested caps and the window's worst-case compute
  exposure checked against the recorded headroom. Structural only: it never
  authenticates an approval (``authenticated`` is always ``False``) and never grants
  rental authority.
- ``LifecycleJournal`` - L1. Append-only JSONL, one fsync per record, hash-chained
  with ``gate.s2_identity.identity_hash``; wall time is stored beside, never inside,
  the hashed content. Reopening verifies the chain and fails closed on tampering or a
  torn tail. Single writer: the writable journal holds an exclusive ``flock`` for its
  lifetime (a second writer gets ``journal_locked``), is opened ``O_NOFOLLOW`` and must
  be a regular file, and re-checks the file's size, tail and inode before every append
  (``journal_changed_underneath``). ``LifecycleJournal.read`` is the read-only loader
  (no write descriptor, never creates).
- ``next_action`` - L1 pure transition function over ``STATES``. Before any create the
  machine lists instances and refuses to create when a row already carries the unique
  label (lost or rolled-back journal): it enters ``uncertain_create`` instead. The
  ``Coordinator`` lists instances again in the same step, immediately before
  ``provider.create``, so a stale journaled label-absent inventory (a journal copy
  restored after the create) can never be replayed into a second create; a present
  label or a failed inventory there yields an ``uncertain`` create result. The
  intent is journaled before every side effect, so a restart in ``create_requested``
  without a recorded result becomes ``uncertain_create`` and reconciles by the unique
  label; a second create is never issued. The absolute cutoff is fixed when the create
  request is journaled and is never renewed. Before ``absent_verified`` an inventory
  must show no row carrying the label; an unbound labelled row is a duplicate that is
  stopped and destroyed (``duplicate_creation``). Unlisted (state, observation) pairs
  never transition. Failed or watchdog-terminated work is ``work_failed`` (a required
  static ``reason`` code plus ``decision_issue_codes``), a cause distinct from
  ``budget_exhausted`` that follows the same evidence path (export, verify, stop,
  destroy/retain, absence, reconcile).
- Verified-export destroy guard (G1; completeness-audit Safety "destroy follows a typed
  export hash") - L1. An ``export_verified`` observation carries the export
  verification report itself (``verification``), never a bare hash:
  ``validate_observation`` accepts it only when the report re-verifies from its own
  bytes (``check_export_verification``), its status is ``passed`` and its manifest
  sha256 equals the typed ``manifest_hash``. Two report forms: a
  ``gate.s2_bundle.verify_bundle`` report (``sha256(canonical_json_bytes(manifest) +
  b"\\n")`` must equal its ``manifest_sha256``, the exact bytes ``seal_bundle`` writes)
  or an ``s2_export_verification`` envelope built by ``export_verification_report``
  around ``gate.s2_pilot.verify_export`` (``identity_hash(receipts)`` is the manifest
  hash). ``reverify_export`` (L1 over local files) re-runs the producer on the export
  copy and requires the identical report; the lifecycle CLI calls it before journaling.
  ``check_export_binding`` (review G1-01) binds the report to the rental: a bundle report's
  scope must equal the record's, an ``actual`` rental needs an ``actual`` bundle covering
  ``gate.s2_bundle.ACTUAL_MINIMUM_ROLES``, and the scope-less envelope is accepted only for
  synthetic/fixture records. A zero-entry bundle, or a bundle report with an unknown
  ``schema_version`` or no ``producer``, is refused (G1-04/G1-05). ``Coordinator.observe``
  requires ``export_root`` for ``export_verified`` and runs both the binding and
  ``reverify_export`` itself (G1-07), so the library path cannot journal a report that names
  no bytes on disk. A forged, failed or mismatched report is refused, so
  ``export_verified`` is never set from it. Without a verified export the machine never destroys an instance that was
  armed (bootstrap/work could have left evidence on it): storage is held under the
  record's ``retain_until`` authority, and otherwise (``destroy_after_stop``, or the
  retention lapsed) the instance stays stopped and the machine escalates to
  ``manual_review`` (``destroy_refused_export_unverified``). An instance that was never
  armed (no work was ever permitted, ``may_bootstrap``) holds no evidence and is still
  destroyed. Label-proven duplicates found while destroying are destroyed as before.
- ``Coordinator`` - L1 with ``FakeProvider``; executing it against a real provider is
  L3 (``qualify_on_host``).
- ``VastCliClient`` - L2 against the installed Vast CLI 1.0.7 source and the live
  D-019 evidence; real provider state shapes are L3.
- ``charge_snapshot_events`` - L2 against the live D-019 charge rows; feeds
  ``gate.s2_cost.reconcile_cost``. Snapshots are rounded and settlement stays pending.

Agent technical defaults (D-018), visible to Huy:

- ``MAX_WINDOW_SECONDS`` = 12 h hard bound on any approved window (the action record
  supplies the actual window, which the watchdog must equal).
- ``ABSENCE_CONFIRMATIONS`` = 2: compute absence (in ``stopping`` and in
  ``destroying``) needs two consecutive successful exact-ID ``NotFound`` results; a
  failed call or a present row resets the count. Volume absence needs two consecutive
  successful volume listings (the CLI has no exact-ID volume lookup; L3).
- ``DESTROY_ATTEMPTS_BEFORE_ESCALATION`` = 5: after that many present rows,
  ``cleanup_incomplete``; after more than that many consecutive checks with a failed
  lookup or inventory, ``cleanup_unverifiable``. Exact-ID destroy keeps being retried
  until ``DESTROYING_CHECK_LIMIT``.
  Stop/destroy request outcomes are journaled as static codes (``effect`` records).
- ``DESTROYING_CHECK_LIMIT`` = 20: total absence checks allowed since entering
  ``destroying``. Results that never settle (for example a lookup alternating error and
  ``NotFound``, which resets both counters above) reach it; the machine then records
  ``cleanup_check_limit_reached`` and stops in ``escalated`` (``manual_review``).
- Readiness: a row bootstraps only when every resource field is present and non-null;
  a null field is "not yet known" (never ``wrong_resource``) until the startup limit.
  The non-null keys of the last observed row are kept in ``observed_row_keys``.
- ``NOT_FOUND_STATUS_CODES`` = {404}: the HTTP status the CLI's ``--raw`` error JSON
  must carry for an exact-ID lookup to count as ``NotFound``. Which status Vast
  returns for a destroyed ID is unverified (L3); any other shape is an error.

Vast CLI flag provenance (installed ``vastai`` 1.0.7, the version recorded in
``evidence/s2-vast-host-20261101-v1/plan.json``):

========================================  =================================================
argv after ``vastai --raw``                source
========================================  =================================================
``--retry 1 create instance <offer>       live-verified: order-request.json (D-019);
--image I --ssh --direct --disk N          skill references/cli_commands.md. Global
--cancel-unavail --label L``               ``--retry 1`` from installed source
                                          cli/main.py:67 (parent global, mirrored on
                                          subparsers by cli/parser.py:118-134): one HTTP
                                          attempt, so a timed-out PUT /asks/ is never
                                          resent by the CLI (api/client.py:102-139)
``show instances``                        live-verified: vast_stop_watchdog.py (D-019/D-020)
``stop instance <id>``                    live-verified: vast_stop_watchdog.py; prints prose,
                                          so the response is never an acknowledgment
``show instance <id>``                    skill references/cli_commands.md; NotFound shape
                                          from installed source cli/main.py ``_emit_error``
``destroy instance <id> --yes``           skill references; ``--yes`` from installed source
                                          (without it the CLI prompts, prints "Aborted." and
                                          exits 0). Consent is the admitted action record.
                                          Success prints nothing (destroy__instance drops
                                          the result): ``requested_unverified``.
``show volumes``, ``delete volume <id>``  installed source cli/commands/storage.py only
``show invoices-v1 --charges ...``        live-verified command (host-validation report);
                                          date/limit/token flags from installed source
========================================  =================================================

The CLI exits 0 after an HTTP error and prints ``{"error": true, "status_code": N}``
to stderr (installed source); exit status alone is therefore never success. Raw
stdout/stderr are parsed but never logged or journaled; credentials are never passed
in argv (no ``--api-key``) and the subprocess environment is an allowlist that
excludes ``VAST_API_KEY`` (the CLI reads its key file).

This module never selects a provider or offer, never retries a rental, never extends
a window, never destroys a resource it did not bind, and never treats an API error,
an empty inventory or a stop response as success.
"""

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from gate.s2_identity import canonical_json_bytes, identity_hash, read_json_bytes

SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
TIERS = ("pilot", "stage", "project")
MAX_WINDOW_SECONDS = 12 * 3600  # agent technical default (D-018)
ABSENCE_CONFIRMATIONS = 2  # agent technical default (D-018)
DESTROY_ATTEMPTS_BEFORE_ESCALATION = 5  # agent technical default (D-018)
DESTROYING_CHECK_LIMIT = 20  # agent technical default (D-018)
NOT_FOUND_STATUS_CODES = frozenset({404})  # agent technical default (D-018); L3
WATCHDOG_SCRIPT = "scripts/vast_stop_watchdog.py"
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "XDG_CONFIG_HOME")
ROW_KEYS = ("id", "label", "actual_status", "intended_status", "next_state", "gpu_name",
            "num_gpus", "disk_space", "dph_total", "image_uuid")
# Readiness needs these non-null; intended_status/next_state must be present but may be null.
READY_KEYS = ("id", "label", "actual_status", "gpu_name", "num_gpus", "disk_space",
              "dph_total", "image_uuid")
_PLACEHOLDER = re.compile(r"\b(tbd|todo|placeholder|xxx)\b|____", re.IGNORECASE)
_PLACEHOLDER_EXACT = {"tbd", "todo", "later", "placeholder", "xxx", "n/a", "?", ""}
_LABEL = re.compile(r"[A-Za-z0-9_.-]{1,128}")
# A ``work_failed`` reason is a static cause code, never raw tool or provider text.
_CAUSE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")

NORMAL_PATH = (
    "admitted", "create_intent", "create_requested", "bound", "armed", "bootstrapping",
    "running", "exporting", "export_verified", "stopping", "stopped", "destroying",
    "absent_verified", "reconciled",
)
BRANCH_STATES = ("uncertain_create", "arm_failed", "cutoff_reached", "escalated", "aborted")
STATES = NORMAL_PATH + BRANCH_STATES
PRE_STOP = ("bound", "armed", "bootstrapping", "running", "exporting", "export_verified")
ALLOWED_OBSERVATIONS = {
    "admitted": frozenset(),
    "create_intent": frozenset({"inventory"}),
    "create_requested": frozenset({"create_result"}),
    "uncertain_create": frozenset({"inventory"}),
    "bound": frozenset({"watchdog_receipt"}),
    "arm_failed": frozenset(),
    "armed": frozenset({"instance_state"}),
    "bootstrapping": frozenset({"bootstrap_complete", "bootstrap_failed", "budget_exhausted",
                                "work_failed"}),
    "running": frozenset({"work_complete", "budget_exhausted", "work_failed"}),
    "exporting": frozenset({"export_verified", "export_failed"}),
    "export_verified": frozenset(),
    "cutoff_reached": frozenset(),
    "stopping": frozenset({"instance_state"}),
    "stopped": frozenset({"instance_state"}),
    "destroying": frozenset({"absence_check"}),
    "absent_verified": frozenset({"cost_reconciled"}),
    "reconciled": frozenset(),
    "escalated": frozenset(),
    "aborted": frozenset(),
}
_OBSERVATION_KEYS = {
    "create_result": None,  # variant-checked below
    "inventory": None,
    "watchdog_receipt": None,
    "instance_state": {"results"},
    "absence_check": {"instances", "volumes", "inventory"},
    "bootstrap_complete": set(),
    "bootstrap_failed": {"error"},
    "work_complete": set(),
    "budget_exhausted": {"decision_issue_codes"},
    # Failed or watchdog-terminated work (A3X): a cause distinct from budget exhaustion.
    "work_failed": {"reason", "decision_issue_codes"},
    "export_verified": {"manifest_hash", "verification"},
    "export_failed": {"error"},
    "cost_reconciled": {"ledger_status", "ledger_hash", "settlement_status"},
}
OBSERVATION_KINDS = frozenset(_OBSERVATION_KEYS)
# The watchdog receipt is not external: only ``Coordinator.observe_watchdog`` produces
# it, from the plan's own watchdog event file.
EXTERNAL_OBSERVATIONS = frozenset({
    "bootstrap_complete", "bootstrap_failed", "work_complete", "work_failed",
    "budget_exhausted", "export_verified", "export_failed", "cost_reconciled",
})
# Watchdog events that may follow ``armed`` while it is still watching.
WATCHDOG_LIVE_EVENTS = frozenset({"command_error", "deadline_clamped_to_prior_arm"})
NOT_FOUND = "not_found"


def _issue(code, path, message, severity="blocking"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


# ---------------------------------------------------------------- action record (P0)


class _FieldError(ValueError):
    def __init__(self, code, path, message):
        super().__init__(message)
        self.code, self.path = code, path


def _text(value, path):
    if type(value) is not str or not value.strip():
        raise _FieldError("invalid_field", path, "non-empty text required")
    if value.strip().lower() in _PLACEHOLDER_EXACT or _PLACEHOLDER.search(value):
        raise _FieldError("placeholder_value", path, "placeholder text is not a value")
    return value


def _positive_int(value, path):
    if type(value) is not int or value <= 0:
        raise _FieldError("invalid_field", path, "positive integer required")
    return value


def _amount(value, path, *, positive=False):
    if type(value) is not str:
        raise _FieldError("invalid_field", path, "decimal string required")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise _FieldError("invalid_field", path, "decimal string required") from None
    if not amount.is_finite() or amount < 0 or (positive and amount <= 0):
        raise _FieldError("invalid_field", path, "finite nonnegative amount required")
    return amount


def _utc(value, path):
    if type(value) is not str:
        raise _FieldError("invalid_field", path, "ISO 8601 time with timezone required")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise _FieldError("invalid_field", path, "ISO 8601 time required") from None
    if parsed.tzinfo is None:
        raise _FieldError("invalid_field", path, "timezone required")
    return parsed.timestamp()


def _keys(value, expected, path):
    if type(value) is not dict:
        raise _FieldError("invalid_field", path, "object required")
    missing, extra = expected - set(value), set(value) - expected
    if missing:
        raise _FieldError("missing_field", path, f"missing {sorted(missing)}")
    if extra:
        raise _FieldError("unknown_field", path, f"unknown {sorted(extra)}")
    return value


_RECORD_KEYS = {
    "artifact_kind", "schema_version", "scope", "provider", "offer", "region", "resources",
    "max_hourly_usd", "window", "lifecycle_limits", "retention_after_stop",
    "export_destination", "cap_headroom_snapshot", "approval", "license_attestation_refs",
    "scientific_registration_ref", "engineering_only",
}


def _check_record(record, issues):
    """Return the parsed plan or None; append every finding to ``issues``."""
    def attempt(function):
        try:
            return function()
        except _FieldError as error:
            issues.append(_issue(error.code, error.path, str(error)))
            return None

    if type(record) is not dict:
        issues.append(_issue("invalid_field", "record", "object required"))
        return None
    version = record.get("schema_version")
    if record.get("artifact_kind") != "s2_action_record" or type(
        version
    ) is not int or version != SCHEMA_VERSION:
        issues.append(_issue("unknown_schema", "record", "unknown kind or schema version"))
        return None
    if attempt(lambda: _keys(record, _RECORD_KEYS, "record")) is None:
        return None
    plan = {}
    if record["scope"] not in SCOPES:
        issues.append(_issue("invalid_field", "scope", "scope must be declared"))
    if record["provider"] != "vast":
        issues.append(_issue("unsupported_provider", "provider",
                             "only the tested Vast adapter exists; another provider needs its "
                             "own tested cutoff before create"))
    offer = attempt(lambda: _keys(record["offer"], {"offer_id", "sku"}, "offer"))
    if offer is not None:
        plan["offer_id"] = attempt(lambda: _positive_int(offer["offer_id"], "offer.offer_id"))
        attempt(lambda: _text(offer["sku"], "offer.sku"))
    attempt(lambda: _text(record["region"], "region"))
    resources = attempt(lambda: _keys(
        record["resources"], {"gpu_count", "gpu_edition", "disk_gb", "image", "volumes"},
        "resources"))
    if resources is not None:
        plan["gpu_count"] = attempt(lambda: _positive_int(resources["gpu_count"],
                                                          "resources.gpu_count"))
        plan["gpu_edition"] = attempt(lambda: _text(resources["gpu_edition"],
                                                    "resources.gpu_edition"))
        plan["disk_gb"] = attempt(lambda: _positive_int(resources["disk_gb"],
                                                        "resources.disk_gb"))
        image = attempt(lambda: _text(resources["image"], "resources.image"))
        if image is not None and re.search(r"\s", image):
            issues.append(_issue("invalid_field", "resources.image", "image has whitespace"))
        plan["image"] = image
        if type(resources["volumes"]) is not list:
            issues.append(_issue("invalid_field", "resources.volumes", "list required"))
        elif resources["volumes"]:
            issues.append(_issue("volumes_not_supported", "resources.volumes",
                                 "the lifecycle creates no separate volume"))
    rate = attempt(lambda: _amount(record["max_hourly_usd"], "max_hourly_usd", positive=True))
    plan["max_hourly_usd"] = None if rate is None else record["max_hourly_usd"]
    window = attempt(lambda: _keys(record["window"], {"start_utc", "end_utc"}, "window"))
    start = end = None
    if window is not None:
        start = attempt(lambda: _utc(window["start_utc"], "window.start_utc"))
        end = attempt(lambda: _utc(window["end_utc"], "window.end_utc"))
    if start is not None and end is not None:
        if end <= start:
            issues.append(_issue("window_not_positive", "window", "end must follow start"))
        elif end - start > MAX_WINDOW_SECONDS:
            issues.append(_issue("window_exceeds_hard_bound", "window",
                                 "window exceeds the 12 h agent technical default (D-018)"))
        plan.update(window_start_epoch=start, window_end_epoch=end,
                    window_seconds=end - start)
    limits = attempt(lambda: _keys(record["lifecycle_limits"], {
        "startup_limit_seconds", "arm_limit_seconds", "export_attempt_limit"},
        "lifecycle_limits"))
    if limits is not None:
        for key in sorted(limits):
            plan[key] = attempt(lambda key=key: _positive_int(limits[key],
                                                              f"lifecycle_limits.{key}"))
    if limits is not None and start is not None and end is not None and end > start:
        for key in ("startup_limit_seconds", "arm_limit_seconds"):
            if type(plan.get(key)) is int and plan[key] > end - start:
                issues.append(_issue("limit_exceeds_window", f"lifecycle_limits.{key}",
                                     "a lifecycle limit cannot exceed the approved window"))
    retention = attempt(lambda: _keys(record["retention_after_stop"], {
        "mode", "authority_ref", "retain_until_utc"}, "retention_after_stop"))
    if retention is not None:
        attempt(lambda: _text(retention["authority_ref"], "retention_after_stop.authority_ref"))
        mode = retention["mode"]
        plan["retention_mode"] = mode
        plan["retain_until_epoch"] = None
        if mode == "destroy_after_stop":
            if retention["retain_until_utc"] is not None:
                issues.append(_issue("invalid_field", "retention_after_stop.retain_until_utc",
                                     "destroy_after_stop has no retention time"))
        elif mode == "retain_until":
            until = attempt(lambda: _utc(retention["retain_until_utc"],
                                         "retention_after_stop.retain_until_utc"))
            plan["retain_until_epoch"] = until
            if until is not None and end is not None and until < end:
                issues.append(_issue("retention_before_window_end", "retention_after_stop",
                                     "retention must last at least until the window end"))
        else:
            issues.append(_issue("invalid_field", "retention_after_stop.mode",
                                 "mode must be destroy_after_stop or retain_until"))
    attempt(lambda: _text(record["export_destination"], "export_destination"))
    _check_caps(record, rate, start, end, issues, attempt)
    approval = attempt(lambda: _keys(record["approval"], {"decision_id", "quoted_instruction"},
                                     "approval"))
    if approval is not None:
        if type(approval["decision_id"]) is not str or not re.fullmatch(
            r"D-\d{3}", approval["decision_id"]
        ):
            issues.append(_issue("invalid_field", "approval.decision_id",
                                 "decision ID like D-019 required"))
        attempt(lambda: _text(approval["quoted_instruction"], "approval.quoted_instruction"))
    refs = record["license_attestation_refs"]
    if type(refs) is not list or len(set(map(str, refs))) != len(refs) or any(
        type(ref) is not str or not ref.strip() for ref in refs
    ):
        issues.append(_issue("invalid_field", "license_attestation_refs",
                             "unique non-empty reference strings required"))
    engineering, registration = record["engineering_only"], record["scientific_registration_ref"]
    if type(engineering) is not bool or not (
        (engineering is True and registration is None)
        or (engineering is False and type(registration) is str and registration.strip())
    ):
        issues.append(_issue("registration_or_engineering_only", "scientific_registration_ref",
                             "exactly one of a scientific registration or engineering_only"))
    elif engineering is False and type(refs) is list and not refs:
        issues.append(_issue("license_attestation_required", "license_attestation_refs",
                             "scientific work needs licence/access attestation references"))
    return plan


def _check_caps(record, rate, start, end, issues, attempt):
    snapshot = attempt(lambda: _keys(record["cap_headroom_snapshot"], {
        "as_of_utc", "tier", "caps_usd", "used_usd", "liability_usd"}, "cap_headroom_snapshot"))
    if snapshot is None:
        return
    attempt(lambda: _utc(snapshot["as_of_utc"], "cap_headroom_snapshot.as_of_utc"))
    if snapshot["tier"] not in TIERS:
        issues.append(_issue("invalid_field", "cap_headroom_snapshot.tier", "unknown tier"))
        return
    values = {}
    for name in ("caps_usd", "used_usd", "liability_usd"):
        table = attempt(lambda name=name: _keys(snapshot[name], set(TIERS),
                                                f"cap_headroom_snapshot.{name}"))
        if table is None:
            return
        parsed = [attempt(lambda tier=tier, name=name, table=table: _amount(
            table[tier], f"cap_headroom_snapshot.{name}.{tier}")) for tier in TIERS]
        if None in parsed:
            return
        values[name] = parsed
    caps, used, owed = values["caps_usd"], values["used_usd"], values["liability_usd"]
    if not caps[0] <= caps[1] <= caps[2]:
        issues.append(_issue("caps_not_nested", "cap_headroom_snapshot.caps_usd",
                             "caps must satisfy pilot <= stage <= project"))
    if not (used[0] <= used[1] <= used[2] and owed[0] <= owed[1] <= owed[2]):
        issues.append(_issue("budget_usage_not_nested", "cap_headroom_snapshot",
                             "used and liability must be nested pilot <= stage <= project"))
    if rate is None or start is None or end is None or end <= start:
        return
    exposure = rate * Decimal(str(end - start)) / Decimal(3600)
    for index in range(TIERS.index(snapshot["tier"]), len(TIERS)):
        headroom = caps[index] - used[index] - owed[index]
        if not exposure < headroom:
            issues.append(_issue("window_exposure_exceeds_headroom",
                                 f"cap_headroom_snapshot.{TIERS[index]}",
                                 "max hourly rate x window must fit strictly in the headroom"))


def validate_action_record(record):
    """Pre-create admission of a rental action record (P0). L1; never authenticates."""
    issues = []
    try:
        record_hash = identity_hash(record)
    except ValueError:
        record_hash = None
        issues.append(_issue("invalid_field", "record", "record is not strict JSON"))
    plan = _check_record(record, issues) if record_hash is not None else None
    admitted = not issues and plan is not None
    scope = record.get("scope") if type(record) is dict else None
    return {
        "artifact_kind": "s2_action_record_report",
        "schema_version": SCHEMA_VERSION,
        "scope": scope if scope in SCOPES else None,
        "producer": {"module": "gate.s2_execution"},
        "input_hashes": {"record": record_hash},
        "units": {"time": "seconds", "currency": "USD"},
        "status": "passed" if admitted else "blocked",
        "admitted": admitted,
        "issues": issues,
        "window_seconds": plan.get("window_seconds") if admitted else None,
        "cutoff_epoch": plan.get("window_end_epoch") if admitted else None,
        "authenticated": False,
        "cloud_launch_authorized": False,
        "note": "structural admission only; approval authenticity and execution remain "
                "the operator's recorded responsibility",
    }


# ---------------------------------------------------------------- journal


class JournalError(ValueError):
    """The lifecycle journal is unreadable, torn or its hash chain is broken."""


_GENESIS = "0" * 64


def _read_fd(fd):
    chunks, offset = [], 0
    while True:
        chunk = os.pread(fd, 1 << 20, offset)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def _parse_journal(data):
    if data and not data.endswith(b"\n"):
        raise JournalError("torn_journal_tail")
    records, previous = [], _GENESIS
    for index, line in enumerate(data.splitlines()):
        try:
            item = read_json_bytes(line)
        except ValueError:
            raise JournalError("journal_line_not_strict_json") from None
        if type(item) is not dict or set(item) != {
            "seq", "prev_hash", "payload", "record_hash", "at_utc"
        }:
            raise JournalError("journal_record_shape")
        body = {"seq": item["seq"], "prev_hash": item["prev_hash"],
                "payload": item["payload"]}
        if item["seq"] != index or item["prev_hash"] != previous or item[
            "record_hash"
        ] != identity_hash(body):
            raise JournalError("journal_chain_broken")
        previous = item["record_hash"]
        records.append(item)
    return records


def replay_records(records):
    """Return (latest state, observation recorded after it or None). L1."""
    state, pending = None, None
    for item in records:
        payload = item["payload"]
        if type(payload) is not dict:
            continue
        if payload.get("type") == "transition":
            state, pending = payload["state"], None
        elif payload.get("type") == "observation":
            pending = payload["observation"]
    return state, pending  # ``effect`` records are evidence only


class LifecycleJournal:
    """Append-only, fsynced, hash-chained JSONL with one locked writer. L1.

    The writable journal is opened ``O_NOFOLLOW`` (regular files only) and holds an
    exclusive ``flock`` until ``close``; reopening verifies the chain. Every append
    first re-checks that the file is still the one loaded, at the size and tail last
    written, so an out-of-band writer is detected instead of forking the chain.
    """

    def __init__(self, path, *, wall=None):
        self.path = path
        self._wall = wall or (lambda: datetime.now(UTC).isoformat())
        self.fd = None
        try:
            fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.EISDIR, errno.ENXIO, errno.EOPNOTSUPP):
                raise JournalError("journal_not_regular_file") from None
            if error.errno in (errno.EACCES, errno.EPERM) and not os.access(path, os.R_OK):
                raise JournalError("journal_unreadable") from None
            raise JournalError("journal_unwritable") from None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise JournalError("journal_not_regular_file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise JournalError("journal_locked") from None
            try:
                data = _read_fd(fd)
            except OSError:
                raise JournalError("journal_unreadable") from None
            self.records = _parse_journal(data)
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        self._identity = (info.st_dev, info.st_ino)
        self._size = len(data)
        self._tail = data.splitlines(keepends=True)[-1] if data else b""

    @staticmethod
    def read(path):
        """Read-only load: no write descriptor, never creates, never follows a link."""
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            raise JournalError("journal_missing") from None
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise JournalError("journal_not_regular_file") from None
            raise JournalError("journal_unreadable") from None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise JournalError("journal_not_regular_file")
            try:
                data = _read_fd(fd)
            except OSError:
                raise JournalError("journal_unreadable") from None
            return _parse_journal(data)
        finally:
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)  # releases the flock
            self.fd = None

    def _verify_unchanged(self):
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError:
            raise JournalError("journal_changed_underneath") from None
        size = os.fstat(self.fd).st_size
        tail = os.pread(self.fd, len(self._tail), self._size - len(self._tail)) if (
            self._tail) else b""
        if (current.st_dev, current.st_ino) != self._identity or size != self._size or (
            tail != self._tail
        ):
            raise JournalError("journal_changed_underneath")

    def append(self, payload):
        if self.fd is None:
            raise JournalError("journal_closed")
        self._verify_unchanged()
        previous = self.records[-1]["record_hash"] if self.records else _GENESIS
        body = {"seq": len(self.records), "prev_hash": previous, "payload": payload}
        item = {**body, "record_hash": identity_hash(body), "at_utc": self._wall()}
        line = (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode()
        data = line
        try:
            while data:
                written = os.write(self.fd, data)
                if not written:
                    raise OSError("journal_write_failed")
                data = data[written:]
            os.fsync(self.fd)
        except OSError:
            raise JournalError("journal_unwritable") from None
        self._size += len(line)
        self._tail = line
        self.records.append(item)
        return item

    def replay(self):
        """Return (latest state, observation recorded after it or None)."""
        return replay_records(self.records)


# ---------------------------------------------------------------- state machine


def initial_state(record, report, intent_id, *, watchdog_events):
    """Build the ``admitted`` state from an admitted, hash-matching record. L1."""
    if type(report) is not dict or report.get("admitted") is not True:
        raise ValueError("action record is not admitted")
    if report["input_hashes"]["record"] != identity_hash(record):
        raise ValueError("report does not belong to this record")
    try:
        canonical = str(uuid.UUID(intent_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("intent_id must be a canonical UUID") from None
    if canonical != intent_id:
        raise ValueError("intent_id must be a canonical lowercase UUID")
    if type(watchdog_events) is not str or not watchdog_events.strip():
        raise ValueError("watchdog events path required")
    plan = _check_record(record, [])
    plan["watchdog_events"] = watchdog_events
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "admitted",
        "intent_id": intent_id,
        "label": f"physloop-s2-{intent_id}",
        "record_hash": report["input_hashes"]["record"],
        "plan": plan,
        "create_requested_epoch": None,
        "cutoff_epoch": None,
        "bound_epoch": None,
        "armed_epoch": None,
        "instance_ids": [],
        "volume_ids": [],
        "stopped_ids": [],
        "absence_streak": {},
        "volume_absence_streak": 0,
        "destroy_attempts": 0,
        "cleanup_unverified_checks": 0,
        "destroying_checks": 0,
        "observed_row_keys": None,
        "export_attempts": 0,
        "export_verified": False,
        "settlement_status": None,
        "issues": [],
    }


def may_bootstrap(state):
    """Bootstrap/work is permitted only after a durable watchdog arm receipt. L1."""
    return state.get("state") in ("bootstrapping", "running") and state.get(
        "armed_epoch") is not None


def _is_stopped(row):
    # D-020: Vast reports a real stop as actual_status "exited" with explicit stop intent.
    return (row.get("actual_status") in ("stopped", "exited")
            and row.get("intended_status") == "stopped"
            and row.get("next_state") in (None, "stopped"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_row(row):
    _require(type(row) is dict and set(row) <= set(ROW_KEYS), "row keys not allowlisted")
    _require(type(row.get("id")) is int and row["id"] > 0, "row id required")


def _validate_result(result):
    _require(type(result) is dict and result.get("status") in ("row", "not_found", "error"),
             "result status")
    if result["status"] == "row":
        _require(set(result) == {"status", "row"}, "row result keys")
        _validate_row(result["row"])
    elif result["status"] == "error":
        _require(set(result) == {"status", "error"} and type(result["error"]) is str,
                 "error result keys")
    else:
        _require(set(result) == {"status"}, "not_found result keys")


def _validate_inventory(inventory):
    _require(type(inventory) is dict, "inventory")
    if inventory.get("status") == "ok":
        _require(set(inventory) == {"status", "rows"} and type(inventory["rows"]) is list,
                 "inventory keys")
        for item in inventory["rows"]:
            _require(type(item) is dict and set(item) == {"id", "label"}, "inventory row")
            _require(type(item["id"]) is int and item["id"] > 0, "inventory id")
            _require(item["label"] is None or type(item["label"]) is str, "inventory label")
    else:
        _require(set(inventory) == {"status", "error"} and inventory["status"] == "error"
                 and type(inventory["error"]) is str, "inventory keys")


def _labelled(inventory, label):
    return sorted({row["id"] for row in inventory["rows"] if row["label"] == label})


BUNDLE_REPORT_KIND = "s2_bundle_verification"
EXPORT_REPORT_KIND = "s2_export_verification"
_EXPORT_REPORT_KEYS = {"artifact_kind", "schema_version", "producer", "receipts",
                       "manifest_sha256", "result", "status"}


def check_export_verification(report, manifest_hash):
    """Static problem codes for an export verification report; empty means it holds. L1.

    Pure: the report must re-verify from its own bytes, be ``passed`` and name
    ``manifest_hash``. It proves self-consistency only; ``reverify_export`` binds it to
    the export copy on disk.
    """
    if type(report) is not dict:
        return ["verification_not_object"]
    kind = report.get("artifact_kind")
    if kind == BUNDLE_REPORT_KIND:
        return _bundle_report_problems(report, manifest_hash)
    if kind == EXPORT_REPORT_KIND:
        return _export_report_problems(report, manifest_hash)
    return ["verification_kind_unknown"]


def _bundle_report_problems(report, manifest_hash):
    # Contract 2.1: an unknown version or a missing producer fails closed (review G1-05).
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or type(report.get("producer")) is not dict or not report["producer"]):
        return ["verification_shape"]
    problems = []
    if report.get("status") != "passed" or report.get("verified") is not True:
        problems.append("verification_not_passed")
    if report.get("seal_state") != "sealed":
        problems.append("bundle_not_sealed")
    issues = report.get("issues")
    if type(issues) is not list or any(
            type(i) is not dict or i.get("severity") == "blocking" for i in issues):
        problems.append("verification_has_blocking_issues")
    manifest, digest = report.get("manifest"), report.get("manifest_sha256")
    if type(manifest) is not dict or type(digest) is not str:
        return problems + ["manifest_missing"]
    try:
        data = canonical_json_bytes(manifest) + b"\n"  # the bytes seal_bundle writes
    except ValueError:
        return problems + ["manifest_not_canonical_json"]
    if hashlib.sha256(data).hexdigest() != digest:
        problems.append("manifest_bytes_do_not_reverify")
    entries = manifest.get("entries")
    # Contract 2.2: absence is never a pass, so a zero-entry manifest verifies nothing (G1-04).
    if type(entries) is not list or not entries or report.get("entries_checked") != len(entries):
        problems.append("entries_not_all_checked")
    if report.get("scope") != manifest.get("scope"):
        problems.append("scope_mismatch")
    if digest != manifest_hash:
        problems.append("manifest_hash_mismatch")
    return problems


def _export_report_problems(report, manifest_hash):
    if set(report) != _EXPORT_REPORT_KEYS or report["schema_version"] != 1 or type(
            report["schema_version"]) is not int:
        return ["verification_shape"]
    problems = []
    receipts, result = report["receipts"], report["result"]
    if type(receipts) is not list or not receipts:
        return ["receipts_missing"]
    try:
        digest = identity_hash(receipts)
    except ValueError:
        return ["receipts_not_json"]
    if digest != report["manifest_sha256"]:
        problems.append("manifest_bytes_do_not_reverify")
    if (report["status"] != "passed" or type(result) is not dict
            or result.get("verified") is not True or result.get("issues") != []
            or result.get("file_count") != len(receipts)):
        problems.append("verification_not_passed")
    if report["manifest_sha256"] != manifest_hash:
        problems.append("manifest_hash_mismatch")
    return problems


def check_export_binding(report, scope):
    """Static codes binding an export report to the rental's record ``scope``; empty holds. L1.

    Review G1-01: a self-consistent report proves nothing about *which* rental it exports.
    A bundle report's ``scope`` must equal the record's, and an ``actual`` rental needs an
    ``actual`` sealed bundle whose manifest covers every ``gate.s2_bundle.ACTUAL_MINIMUM_ROLES``
    role. The scope-less ``s2_export_verification`` envelope (``verify_export`` over receipts,
    no roles) is accepted only for ``synthetic``/``public_source_fixture`` records.
    """
    if scope not in SCOPES:
        return ["record_scope_invalid"]
    kind = report.get("artifact_kind") if type(report) is dict else None
    if kind == EXPORT_REPORT_KIND:
        return ["actual_export_requires_bundle"] if scope == "actual" else []
    if kind != BUNDLE_REPORT_KIND:
        return ["verification_kind_unknown"]
    manifest = report.get("manifest")
    if report.get("scope") != scope or type(manifest) is not dict or manifest.get(
            "scope") != scope:
        return ["export_scope_mismatch"]
    if scope == "actual":
        from gate.s2_bundle import ACTUAL_MINIMUM_ROLES

        entries = manifest.get("entries")
        roles = {entry.get("role") for entry in entries if type(entry) is dict} if type(
            entries) is list else set()
        if not set(ACTUAL_MINIMUM_ROLES) <= roles:
            return ["actual_export_roles_missing"]
    return []


def export_verification_report(root, receipts):
    """Wrap ``gate.s2_pilot.verify_export`` in a hash-bearing envelope. L1 (local files)."""
    from gate.s2_pilot import verify_export

    result = verify_export(root, receipts)
    try:
        digest = identity_hash(receipts)
    except ValueError:
        digest = None
    return {"artifact_kind": EXPORT_REPORT_KIND, "schema_version": 1,
            "producer": {"module": "gate.s2_execution", "function": "verify_export"},
            "receipts": receipts, "manifest_sha256": digest, "result": result,
            "status": "passed" if result["verified"] and digest is not None else "blocked"}


def reverify_export(observation, root):
    """Re-run the report's producer on the export copy at ``root``; codes, empty if same. L1.

    The recomputed report must be identical (``identity_hash``) to the one carried by the
    observation, so a report that was written by hand, or describes other bytes, is refused.
    """
    problems = []
    try:
        validate_observation(observation)
    except ValueError:
        problems.append("observation_invalid")
    if type(observation) is not dict or observation.get("kind") != "export_verified":
        return problems or ["not_export_verified"]
    report = observation.get("verification")
    kind = report.get("artifact_kind") if type(report) is dict else None
    try:
        if kind == BUNDLE_REPORT_KIND:
            from gate.s2_bundle import verify_bundle

            again = verify_bundle(root)
        elif kind == EXPORT_REPORT_KIND:
            again = export_verification_report(root, report["receipts"])
        else:
            return problems + ["verification_kind_unknown"]
        same = identity_hash(again) == identity_hash(report)
    except (OSError, ValueError, TypeError, KeyError):
        return problems + ["export_unreadable"]
    if not same:
        problems.append("export_report_not_reproduced")
    return problems


def validate_observation(observation):
    """Strict structural check of an observation (no raw provider text). L1."""
    _require(type(observation) is dict, "observation must be an object")
    kind = observation.get("kind")
    _require(kind in OBSERVATION_KINDS, "unknown observation kind")
    fields = set(observation) - {"kind"}
    if kind == "create_result":
        if observation.get("outcome") == "created":
            _require(fields == {"outcome", "instance_id"}, "create_result keys")
            _require(type(observation["instance_id"]) is int and observation["instance_id"] > 0,
                     "instance id")
        else:
            _require(fields == {"outcome", "error"} and observation["outcome"] == "uncertain"
                     and type(observation["error"]) is str, "create_result keys")
    elif kind == "inventory":
        _validate_inventory({key: observation[key] for key in fields})
    elif kind == "watchdog_receipt":
        if observation.get("armed") is True:
            _require(fields == {"armed", "instance_id", "label", "deadline_epoch"},
                     "receipt keys")
            _require(type(observation["instance_id"]) is int, "receipt id")
            _require(type(observation["label"]) is str, "receipt label")
            _require(type(observation["deadline_epoch"]) in (int, float) and math.isfinite(
                observation["deadline_epoch"]), "receipt deadline")
        else:
            _require(fields == {"armed"} and observation["armed"] is False, "receipt keys")
    else:
        _require(fields == _OBSERVATION_KEYS[kind], f"{kind} keys")
        if kind == "instance_state":
            _require(type(observation["results"]) is dict, "results")
            for key, result in observation["results"].items():
                _require(key.isdigit(), "result key")
                _validate_result(result)
        elif kind == "absence_check":
            _require(type(observation["instances"]) is dict, "instances")
            for key, result in observation["instances"].items():
                _require(key.isdigit(), "result key")
                _validate_result(result)
            volumes = observation["volumes"]
            _require(type(volumes) is dict, "volumes")
            if volumes.get("status") == "ok":
                _require(set(volumes) == {"status", "ids", "label_matches"}, "volume keys")
                _require(type(volumes["ids"]) is list and all(
                    type(i) is int for i in volumes["ids"]), "volume ids")
                _require(type(volumes["label_matches"]) is int, "volume label matches")
            else:
                _require(set(volumes) == {"status", "error"} and volumes["status"] == "error",
                         "volume keys")
            _validate_inventory(observation["inventory"])
        elif kind in ("bootstrap_failed", "export_failed"):
            _require(type(observation["error"]) is str, "error text")
        elif kind in ("budget_exhausted", "work_failed"):
            _require(type(observation["decision_issue_codes"]) is list and all(
                type(c) is str for c in observation["decision_issue_codes"]), "issue codes")
            if kind == "work_failed":
                _require(type(observation["reason"]) is str
                         and _CAUSE.fullmatch(observation["reason"]) is not None,
                         "work_failed reason must be a static cause code")
        elif kind == "export_verified":
            _require(type(observation["manifest_hash"]) is str and re.fullmatch(
                r"[0-9a-f]{64}", observation["manifest_hash"]), "manifest hash")
            problems = check_export_verification(observation["verification"],
                                                 observation["manifest_hash"])
            _require(not problems, "export verification refused: " + ",".join(problems))
        elif kind == "cost_reconciled":
            _require(observation["ledger_status"] in ("passed", "blocked", "not_estimable"),
                     "ledger status")
            _require(type(observation["ledger_hash"]) is str and re.fullmatch(
                r"[0-9a-f]{64}", observation["ledger_hash"]), "ledger hash")
            _require(observation["settlement_status"] in ("pending", "final"), "settlement")
    return observation


def _utc_text(epoch):
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _add(state, code):
    if code not in state["issues"]:
        state["issues"].append(code)


def _go(state, name, action, issues=()):
    state["state"] = name
    return {"state": state, "action": action, "issues": list(issues)}


def _stop(state, code=None):
    if code:
        _add(state, code)
    ids = [i for i in state["instance_ids"] if i not in state["stopped_ids"]]
    return _go(state, "stopping", {"action": "stop", "instance_ids": ids})


def _enter_destroying(state):
    state["destroying_checks"] = 0  # the check limit counts from entering destroying
    return _go(state, "destroying", _destroy_action(state))


def _destroy_action(state):
    pending = [i for i in state["instance_ids"]
               if state["absence_streak"].get(str(i), 0) == 0]
    return {"action": "destroy", "instance_ids": pending,
            "volume_ids": list(state["volume_ids"]),
            "verify_instance_ids": list(state["instance_ids"])}


def _resource_mismatch(plan, row, label):
    """A known value that differs from the plan. ``None`` is not yet known (A3-15)."""
    if row.get("label") != label:
        return True
    checks = {"gpu_name": plan["gpu_edition"], "num_gpus": plan["gpu_count"],
              "image_uuid": plan["image"]}
    for key, expected in checks.items():
        if row.get(key) is not None and row[key] != expected:
            return True
    if row.get("disk_space") is not None and (type(row["disk_space"]) not in (int, float)
                                              or row["disk_space"] != plan["disk_gb"]):
        return True
    if row.get("dph_total") is not None:
        if type(row["dph_total"]) not in (int, float) or not math.isfinite(row["dph_total"]):
            return True
        if Decimal(str(row["dph_total"])) > Decimal(plan["max_hourly_usd"]):
            return True
    return False


def next_action(state, observation, clock):
    """Pure transition: (state, observation, now epoch) -> new state and one action. L1.

    The input state is never mutated. An observation not allowed in the current
    state produces no transition and a ``wait`` action with ``unexpected_observation``.
    """
    _require(type(clock) in (int, float) and math.isfinite(clock), "clock must be finite")
    _require(type(state) is dict and state.get("state") in STATES, "unknown state")
    if observation is not None:
        _require(type(observation) is dict and observation.get("kind") in OBSERVATION_KINDS,
                 "unknown observation")
    now = clock
    state = json.loads(json.dumps(state))  # deep copy; JSON-native by construction
    name = state["state"]
    if observation is not None:
        if observation["kind"] not in ALLOWED_OBSERVATIONS[name]:
            return {"state": state, "action": {"action": "wait"}, "issues": [
                _issue("unexpected_observation", "observation",
                       f"{observation['kind']} is not accepted in state {name}")]}
        validate_observation(observation)
    plan = state["plan"]
    kind = observation["kind"] if observation else None

    if name in PRE_STOP:
        if kind == "export_verified":
            state["export_verified"] = True
        if now >= state["cutoff_epoch"]:
            _add(state, "cutoff_reached")
            if not state["export_verified"]:
                _add(state, "export_incomplete")
            return _go(state, "cutoff_reached", {"action": "advance"})

    if name == "admitted":
        if now >= plan["window_end_epoch"]:
            _add(state, "window_closed_before_create")
            return _go(state, "aborted", {"action": "none"})
        if now < plan["window_start_epoch"]:
            return _go(state, "admitted", {"action": "wait"})
        return _go(state, "create_intent", {"action": "check_label_absent",
                                            "label": state["label"]})
    if name == "create_intent":
        if now >= plan["window_end_epoch"]:
            _add(state, "window_closed_before_create")
            return _go(state, "aborted", {"action": "none"})
        if kind != "inventory" or observation["status"] != "ok":
            # No create without a successful inventory showing the label unused.
            return _go(state, "create_intent", {"action": "check_label_absent",
                                                "label": state["label"]})
        state["create_requested_epoch"] = now
        state["cutoff_epoch"] = plan["window_end_epoch"]  # absolute; never renewed
        if _labelled(observation, state["label"]):
            # Lost/rolled-back journal or a second driver: never create again.
            _add(state, "label_present_before_create")
            return _go(state, "uncertain_create", {"action": "reconcile_by_label",
                                                   "label": state["label"]})
        return _go(state, "create_requested", {
            "action": "create", "offer_id": plan["offer_id"], "label": state["label"],
            "image": plan["image"], "disk_gb": plan["disk_gb"]})
    if name == "create_requested":
        if kind == "create_result" and observation["outcome"] == "created":
            return _bind(state, [observation["instance_id"]], now)
        if kind == "create_result":
            _add(state, "uncertain_create")
            if observation["error"] in ("label_present_before_create",
                                        "inventory_failed_before_create"):
                _add(state, observation["error"])
        else:
            _add(state, "create_result_missing_after_restart")
        return _go(state, "uncertain_create", {"action": "reconcile_by_label",
                                               "label": state["label"]})
    if name == "uncertain_create":
        if kind == "inventory" and observation["status"] == "ok":
            ours = _labelled(observation, state["label"])
            if len(ours) == 1:
                return _bind(state, ours, now)
            if len(ours) > 1:
                state["instance_ids"] = ours
                state["bound_epoch"] = now
                return _stop(state, "duplicate_creation")
        limit = min(state["create_requested_epoch"] + plan["startup_limit_seconds"],
                    state["cutoff_epoch"])
        if now >= limit:
            _add(state, "create_unresolved")
            return _go(state, "escalated", {"action": "manual_review", "label": state["label"]})
        return _go(state, "uncertain_create", {"action": "reconcile_by_label",
                                               "label": state["label"]})
    if name == "bound":
        if kind == "watchdog_receipt":
            receipt = observation
            if (receipt["armed"] is True and receipt["instance_id"] == state["instance_ids"][0]
                    and receipt["label"] == state["label"]
                    and receipt["deadline_epoch"] <= state["cutoff_epoch"]):
                state["armed_epoch"] = now
                return _go(state, "armed", {"action": "observe_instance",
                                            "instance_id": state["instance_ids"][0]})
            _add(state, "arm_failed")
            return _go(state, "arm_failed", {"action": "advance"})
        if now >= state["bound_epoch"] + plan["arm_limit_seconds"]:
            _add(state, "arm_failed")
            return _go(state, "arm_failed", {"action": "advance"})
        return _go(state, "bound", {"action": "await_watchdog",
                                    "events": plan["watchdog_events"]})
    if name in ("arm_failed", "cutoff_reached"):
        return _stop(state)
    if name == "armed":
        return _armed(state, observation, now)
    if name == "bootstrapping":
        if kind == "bootstrap_complete":
            return _go(state, "running", {"action": "run_units"})
        if kind == "bootstrap_failed":
            _add(state, "bootstrap_failed")
            return _go(state, "exporting", {"action": "export"})
        if kind in ("budget_exhausted", "work_failed"):
            _add(state, kind)  # the cause; the evidence path is the same
            return _go(state, "exporting", {"action": "export"})
        return _go(state, "bootstrapping", {"action": "bootstrap"})
    if name == "running":
        if kind == "work_complete":
            return _go(state, "exporting", {"action": "export"})
        if kind in ("budget_exhausted", "work_failed"):
            _add(state, kind)  # the cause; the evidence path is the same
            return _go(state, "exporting", {"action": "export"})
        return _go(state, "running", {"action": "run_units"})
    if name == "exporting":
        if kind == "export_verified":
            return _go(state, "export_verified", {"action": "advance"})
        if kind == "export_failed":
            state["export_attempts"] += 1
            if state["export_attempts"] >= plan["export_attempt_limit"]:
                return _stop(state, "export_incomplete")
        return _go(state, "exporting", {"action": "export"})
    if name == "export_verified":
        return _stop(state)
    if name == "stopping":
        if kind == "instance_state":
            streak = state["absence_streak"]
            for key, result in sorted(observation["results"].items()):
                instance = int(key)
                if instance not in state["instance_ids"] or instance in state["stopped_ids"]:
                    continue
                if result["status"] == "not_found":
                    streak[key] = streak.get(key, 0) + 1
                    if streak[key] >= ABSENCE_CONFIRMATIONS:
                        state["stopped_ids"].append(instance)
                else:
                    streak[key] = 0
                    if (result["status"] == "row" and result["row"]["id"] == instance
                            and _is_stopped(result["row"])):
                        state["stopped_ids"].append(instance)
            state["stopped_ids"].sort()
            if set(state["stopped_ids"]) == set(state["instance_ids"]):
                state["absence_streak"] = {}  # destroying counts its own absence
                return _go(state, "stopped", {"action": "advance"})
        return _stop(state)
    if name == "stopped":
        if state["export_verified"]:
            return _enter_destroying(state)
        until = plan["retain_until_epoch"]
        if plan["retention_mode"] == "retain_until" and until is not None and now < until:
            if kind == "instance_state":
                for instance in state["instance_ids"]:
                    result = observation["results"].get(str(instance))
                    if result is not None and result["status"] == "not_found":
                        _add(state, "retained_instance_missing")  # evidence loss (A3R-04)
                    elif result is not None and result["status"] == "error":
                        _add(state, "retention_check_failed")
                running = sorted(
                    int(key) for key, result in observation["results"].items()
                    if result["status"] == "row" and not _is_stopped(result["row"]))
                if running:
                    state["stopped_ids"] = [i for i in state["stopped_ids"]
                                            if i not in running]
                    return _stop(state, "instance_running_during_retention")
            _add(state, "export_incomplete_retained")
            return _go(state, "stopped", {"action": "hold_for_retention",
                                          "retain_until_utc": _utc_text(until),
                                          "instance_ids": list(state["instance_ids"])})
        if state["armed_epoch"] is None:
            # Never armed: bootstrap and work were never permitted (may_bootstrap), so the
            # instance holds no evidence and destroying it loses nothing.
            _add(state, "destroyed_before_any_work")
            return _enter_destroying(state)
        # G1: no verified export -> never destroy; the stopped instance keeps its storage.
        _add(state, "evidence_loss_risk")
        _add(state, "destroy_refused_export_unverified")
        return _go(state, "escalated", {"action": "manual_review", "label": state["label"]})
    if name == "destroying":
        if kind == "absence_check":
            return _destroying(state, observation)
        return _go(state, "destroying", _destroy_action(state))
    if name == "absent_verified":
        if kind == "cost_reconciled":
            state["settlement_status"] = observation["settlement_status"]
            if observation["ledger_status"] != "passed":
                _add(state, f"cost_ledger_{observation['ledger_status']}")
            return _go(state, "reconciled", {"action": "none"})
        return _go(state, "absent_verified", {"action": "reconcile_cost"})
    if name == "escalated":
        return _go(state, "escalated", {"action": "manual_review", "label": state["label"]})
    return _go(state, name, {"action": "none"})  # reconciled, aborted: terminal


def _destroying(state, observation):
    inventory, volumes = observation["inventory"], observation["volumes"]
    labelled = _labelled(inventory, state["label"]) if inventory["status"] == "ok" else []
    duplicates = [i for i in labelled if i not in state["instance_ids"]]
    if duplicates:
        # Provably ours by the unique label, never bound: stop, then destroy, all of them.
        state["instance_ids"] = state["instance_ids"] + duplicates
        return _stop(state, "duplicate_creation")
    present = unverified = False
    for instance in state["instance_ids"]:
        key = str(instance)
        result = observation["instances"].get(key)
        if result is not None and result["status"] == "not_found" and instance not in labelled:
            state["absence_streak"][key] = state["absence_streak"].get(key, 0) + 1
            continue
        state["absence_streak"][key] = 0
        if (result is not None and result["status"] == "row") or instance in labelled:
            present = True
        else:
            unverified = True
    if volumes["status"] == "ok" and not set(volumes["ids"]) & set(state["volume_ids"]):
        if volumes["label_matches"]:
            _add(state, "unbound_task_volume")
            state["volume_absence_streak"] = 0
        else:
            state["volume_absence_streak"] += 1
    else:
        state["volume_absence_streak"] = 0
    if volumes["status"] != "ok" or inventory["status"] != "ok":
        unverified = True
    if present:
        state["destroy_attempts"] += 1
        if state["destroy_attempts"] > DESTROY_ATTEMPTS_BEFORE_ESCALATION:
            _add(state, "cleanup_incomplete")
    state["cleanup_unverified_checks"] = state["cleanup_unverified_checks"] + 1 if (
        unverified) else 0
    if state["cleanup_unverified_checks"] > DESTROY_ATTEMPTS_BEFORE_ESCALATION:
        _add(state, "cleanup_unverifiable")
    if inventory["status"] == "ok" and not labelled and (
        state["volume_absence_streak"] >= ABSENCE_CONFIRMATIONS
    ) and all(state["absence_streak"].get(str(i), 0) >= ABSENCE_CONFIRMATIONS
              for i in state["instance_ids"]):
        return _go(state, "absent_verified", {"action": "reconcile_cost"})
    state["destroying_checks"] = state.get("destroying_checks", 0) + 1
    if state["destroying_checks"] >= DESTROYING_CHECK_LIMIT:
        # Stop condition (A3R-02): unsettled results never loop unbounded.
        _add(state, "cleanup_check_limit_reached")
        return _go(state, "escalated", {"action": "manual_review", "label": state["label"]})
    return _go(state, "destroying", _destroy_action(state))


def _bind(state, ids, now):
    state["instance_ids"] = list(ids)
    state["bound_epoch"] = now
    plan = state["plan"]
    argv = [
        WATCHDOG_SCRIPT, "--instance-id", str(ids[0]), "--expected-label", state["label"],
        "--deadline-utc", _utc_text(state["cutoff_epoch"]),
        "--max-window-seconds", str(int(plan["window_seconds"])),
        "--events", plan["watchdog_events"], "--first-arm",
    ]
    return _go(state, "bound", {"action": "arm_watchdog", "argv": argv,
                                "deadline_epoch": state["cutoff_epoch"]})


def _armed(state, observation, now):
    plan = state["plan"]
    primary = state["instance_ids"][0]
    limit_reached = now >= state["create_requested_epoch"] + plan["startup_limit_seconds"]
    if observation is not None:
        result = observation["results"].get(str(primary))
        if result is not None and result["status"] == "not_found":
            return _stop(state, "instance_vanished")
        if result is not None and result["status"] == "row":
            row = result["row"]
            state["observed_row_keys"] = sorted(k for k in row if row[k] is not None)
            if row["id"] != primary or _resource_mismatch(plan, row, state["label"]):
                return _stop(state, "wrong_resource")
            complete = all(key in row for key in ROW_KEYS) and all(
                row[key] is not None for key in READY_KEYS)
            if complete and row.get("actual_status") == "running":
                return _go(state, "bootstrapping", {"action": "bootstrap"})
    if limit_reached:
        return _stop(state, "startup_limit_exceeded")
    return _go(state, "armed", {"action": "observe_instance", "instance_id": primary})


# ---------------------------------------------------------------- watchdog receipt


def read_watchdog_receipt(path):
    """Read the watchdog's durable event file; ``armed`` only while it still watches. L1.

    ``armed`` needs exactly one valid ``armed`` event and, after it, only
    ``WATCHDOG_LIVE_EVENTS``. Any other later event (``stopped_verified``,
    ``instance_absent``, ``scope_mismatch*``, ``stop_unconfirmed``, ``deadline_reached``
    ...) means the watchdog has exited or is stopping, so it is not armed. The file is
    opened without following a symlink and must be a regular file.
    """
    not_armed = {"kind": "watchdog_receipt", "armed": False}
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, TypeError, ValueError):
        return not_armed
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return not_armed
        lines = _read_fd(fd).splitlines()
    except OSError:
        return not_armed
    finally:
        os.close(fd)
    receipt = None
    for line in lines:
        try:
            event = read_json_bytes(line)
        except ValueError:
            return not_armed
        if type(event) is not dict:
            return not_armed
        name = event.get("event")
        if name == "armed":
            identity, label = event.get("instance_id"), event.get("expected_label")
            deadline = event.get("deadline_utc_epoch")
            if receipt is not None or not (
                type(identity) is int and type(label) is str
                and type(deadline) in (int, float) and math.isfinite(deadline)
            ):
                return not_armed
            receipt = {"kind": "watchdog_receipt", "armed": True, "instance_id": identity,
                       "label": label, "deadline_epoch": deadline}
        elif receipt is not None and name not in WATCHDOG_LIVE_EVENTS:
            return not_armed
    return receipt or not_armed


# ---------------------------------------------------------------- provider clients


class ProviderError(Exception):
    """A provider call failed; the message is a static code, never provider output."""


class ProviderCrash(BaseException):
    """Simulates the coordinator process dying mid-call (FakeProvider only)."""


def allowlisted_env():
    """Environment for provider subprocesses: allowlist only, no credential variables."""
    return {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}


def subprocess_runner(argv, timeout):
    """Default runner: no shell, closed stdin, allowlisted env. Never used by tests."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False,
                          stdin=subprocess.DEVNULL, env=allowlisted_env())


def _decimal_text_hook(text):
    return text


class VastCliClient:
    """``vastai --raw`` adapter with an injected runner. L2 (see module provenance)."""

    def __init__(self, runner=None, *, cli="vastai", timeout=30, max_charge_pages=50):
        self.runner = runner or subprocess_runner
        self.cli = cli
        self.timeout = timeout
        self.max_charge_pages = max_charge_pages

    def _run(self, args):
        try:
            result = self.runner([self.cli, "--raw", *args], self.timeout)
        except subprocess.TimeoutExpired:
            raise ProviderError("cli_timeout") from None
        except UnicodeError:
            raise ProviderError("cli_undecodable_output") from None
        except OSError:
            raise ProviderError("cli_unavailable") from None
        if type(result.stdout) is not str or type(result.stderr) is not str:
            raise ProviderError("cli_undecodable_output")
        return result

    @staticmethod
    def _error_status(stderr):
        """Status code of the CLI's --raw error JSON on stderr, else None."""
        for line in stderr.splitlines():
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if type(payload) is dict and payload.get("error") is True:
                code = payload.get("status_code")
                return code if type(code) is int else 0
        return None

    @staticmethod
    def _json(text, *, decimals_as_text=False):
        try:
            if decimals_as_text:
                return json.loads(text, parse_float=_decimal_text_hook,
                                  parse_constant=_reject_constant)
            return json.loads(text, parse_constant=_reject_constant)
        except ValueError:
            raise ProviderError("cli_non_json") from None

    def create(self, offer_id, image, disk_gb, label):
        """Never raises: anything but an explicit new_contract is ``uncertain``."""
        # Global --retry 1: the CLI must never resend a create on its own (A3-12).
        args = ["--retry", "1", "create", "instance", str(offer_id), "--image", image,
                "--ssh", "--direct", "--disk", str(disk_gb), "--cancel-unavail",
                "--label", label]
        try:
            result = self._run(args)
            if result.returncode != 0:
                raise ProviderError("cli_nonzero")
            payload = self._json(result.stdout)
        except ProviderError as error:
            return {"outcome": "uncertain", "error": str(error)}
        contract = payload.get("new_contract") if type(payload) is dict else None
        if type(payload) is dict and payload.get("success") is True and type(
            contract
        ) is int and contract > 0:
            return {"outcome": "created", "instance_id": contract}
        return {"outcome": "uncertain", "error": "create_not_acknowledged"}

    def show_instance(self, instance_id):
        result = self._run(["show", "instance", str(instance_id)])
        if result.returncode != 0:
            raise ProviderError("cli_nonzero")
        if not result.stdout.strip():
            status = self._error_status(result.stderr)
            if status in NOT_FOUND_STATUS_CODES:
                return NOT_FOUND
            raise ProviderError("cli_error_status")
        payload = self._json(result.stdout)
        if type(payload) is not dict or payload.get("id") != instance_id:
            raise ProviderError("instance_identity_mismatch")
        return payload

    def list_instances(self):
        result = self._run(["show", "instances"])
        if result.returncode != 0:
            raise ProviderError("cli_nonzero")
        payload = self._json(result.stdout)
        if type(payload) is not list or any(
            type(row) is not dict or type(row.get("id")) is not int or row["id"] <= 0
            for row in payload
        ):
            raise ProviderError("inventory_not_full_list")
        return payload

    def stop(self, instance_id):
        """Prose on success (live D-020); only a later state read confirms a stop."""
        result = self._run(["stop", "instance", str(instance_id)])
        if result.returncode != 0 or self._error_status(result.stderr) is not None:
            raise ProviderError("stop_rejected")
        return "acknowledged_unverified"

    def destroy(self, instance_id):
        """Installed 1.0.7 prints nothing on success (destroy__instance drops the result).

        Never success: only later exact-ID absence confirms a destroy.
        """
        result = self._run(["destroy", "instance", str(instance_id), "--yes"])
        if result.returncode != 0:
            raise ProviderError("cli_nonzero")
        if self._error_status(result.stderr) is not None:
            raise ProviderError("destroy_rejected")
        if not result.stdout.strip():
            return "requested_unverified"
        payload = self._json(result.stdout)
        if type(payload) is not dict or payload.get("success") is not True:
            raise ProviderError("destroy_not_acknowledged")
        return "acknowledged_unverified"

    def list_volumes(self):
        result = self._run(["show", "volumes"])
        if result.returncode != 0:
            raise ProviderError("cli_nonzero")
        payload = self._json(result.stdout)
        if type(payload) is not list or any(type(row) is not dict for row in payload):
            raise ProviderError("volume_inventory_not_list")
        return payload

    def destroy_volume(self, volume_id):
        result = self._run(["delete", "volume", str(volume_id)])
        if result.returncode != 0:
            raise ProviderError("cli_nonzero")
        payload = self._json(result.stdout) if result.stdout.strip() else None
        if type(payload) is not dict or payload.get("success") is not True:
            raise ProviderError("volume_delete_not_acknowledged")
        return "acknowledged_unverified"

    def charges(self, start_epoch, end_epoch):
        """All charge rows in [start, end], decimals kept as their JSON text."""
        base = ["show", "invoices-v1", "--charges", "--start-date", str(int(start_epoch)),
                "--end-date", str(int(end_epoch)), "--limit", "100"]
        rows, token = [], None
        for _ in range(self.max_charge_pages):
            result = self._run(base + (["--next-token", token] if token else []))
            if result.returncode != 0:
                raise ProviderError("cli_nonzero")
            payload = self._json(result.stdout, decimals_as_text=True)
            if type(payload) is not dict or type(payload.get("results")) is not list:
                raise ProviderError("charges_not_paginated_object")
            rows.extend(payload["results"])
            token = payload.get("next_token")
            if not token:
                return rows
            if type(token) is not str:
                raise ProviderError("charges_invalid_token")
        raise ProviderError("charges_page_limit")


def charge_snapshot_events(rows, label, instance_ids, at_epoch):
    """Turn provider charge rows into cumulative ``provider_charge_snapshot`` events. L2/L3.

    Keeps only rows provably ours: ``source == "instance-<bound id>"`` or
    ``metadata.label`` equal to our unique label with an ``instance-<id>`` source.
    Per-day rows of one resource are summed into one cumulative snapshot. Amounts
    must be decimal text (``VastCliClient.charges`` preserves the JSON text) or ints.
    The row envelope is the live D-019 shape (charges-after-delete.json); the real
    provider's ``--raw`` envelope is L3.
    """
    _require(type(at_epoch) in (int, float) and math.isfinite(at_epoch) and at_epoch >= 0,
             "at_epoch must be finite")
    bound = {str(i) for i in instance_ids}
    totals = {}
    for row in rows:
        _require(type(row) is dict, "charge row must be an object")
        match = re.fullmatch(r"instance-(\d+)", str(row.get("source") or ""))
        metadata = row.get("metadata") if type(row.get("metadata")) is dict else {}
        if match is None or (match.group(1) not in bound and metadata.get("label") != label):
            continue
        amount = row.get("amount")
        if type(amount) is int:
            amount = str(amount)
        _require(type(amount) is str, "charge amount must be decimal text")
        try:
            value = Decimal(amount)
        except InvalidOperation:
            raise ValueError("charge amount must be decimal text") from None
        _require(value.is_finite() and value >= 0, "charge amount must be nonnegative")
        resource = match.group(1)
        totals[resource] = totals.get(resource, Decimal(0)) + value
    return [{"event_id": f"charge-{resource}-{int(at_epoch)}", "kind": "provider_charge_snapshot",
             "resource_id": resource, "at_epoch": at_epoch,
             "amount_usd": format(total.normalize(), "f") if total else "0"}
            for resource, total in sorted(totals.items())]


def _reject_constant(value):
    raise ValueError(value)


class FakeProvider:
    """Scripted provider for CPU tests: each method pops its next scripted result.

    A scripted exception instance is raised (``ProviderError`` for timeouts or prose,
    ``ProviderCrash`` for a killed process). An exhausted script fails the test.
    """

    def __init__(self, script):
        self.script = {key: list(value) for key, value in script.items()}
        self.calls = []

    def _next(self, method, *args):
        self.calls.append((method, *args))
        queue = self.script.get(method)
        assert queue, f"FakeProvider script exhausted for {method}"
        value = queue.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def create(self, offer_id, image, disk_gb, label):
        return self._next("create", offer_id, image, disk_gb, label)

    def show_instance(self, instance_id):
        return self._next("show_instance", instance_id)

    def list_instances(self):
        return self._next("list_instances")

    def stop(self, instance_id):
        return self._next("stop", instance_id)

    def destroy(self, instance_id):
        return self._next("destroy", instance_id)

    def list_volumes(self):
        return self._next("list_volumes")

    def destroy_volume(self, volume_id):
        return self._next("destroy_volume", volume_id)

    def charges(self, start_epoch, end_epoch):
        return self._next("charges", start_epoch, end_epoch)


# ---------------------------------------------------------------- coordinator


def _filter_row(row):
    return {key: row[key] for key in ROW_KEYS if key in row and (
        row[key] is None or type(row[key]) in (str, int, bool)
        or (type(row[key]) is float and math.isfinite(row[key])))}


_OUTCOME = re.compile(r"[a-z_]{1,64}")


def verify_journal_owner(state, record, report, intent_id, *, watchdog_events):
    """Raise ValueError unless a journal state belongs to this record and intent. L1."""
    expected = initial_state(record, report, intent_id, watchdog_events=watchdog_events)
    if type(state) is not dict or (
        state.get("record_hash"), state.get("intent_id"), state.get("plan")
    ) != (expected["record_hash"], expected["intent_id"], expected["plan"]):
        raise ValueError("journal belongs to a different action record or intent")
    return state


class Coordinator:
    """Journal-backed driver: decide, journal the intent, act, journal the result.

    L1 with ``FakeProvider``; L3 against a real provider. External actions (arming
    the watchdog, bootstrap, unit runs, export, cost reconciliation) are performed by
    the operator's tools and reported back through ``observe``; the watchdog receipt
    only through ``observe_watchdog`` from the plan's event file. The journal's
    exclusive lock is held for the coordinator's lifetime.
    """

    def __init__(self, journal_path, provider, clock, *, record, report, intent_id,
                 watchdog_events, wall=None):
        expected = initial_state(record, report, intent_id, watchdog_events=watchdog_events)
        self.provider, self.clock = provider, clock
        self.scope = record["scope"]  # binds export_verified reports to this rental (G1-01)
        self.journal = LifecycleJournal(journal_path, wall=wall)
        state, _ = self.journal.replay()
        if state is None:
            self.journal.append({"type": "transition", "state": expected, "action": None,
                                 "observation": None, "issues": []})
            return
        try:
            verify_journal_owner(state, record, report, intent_id,
                                 watchdog_events=watchdog_events)
        except ValueError:
            self.journal.close()
            raise

    @property
    def state(self):
        return self.journal.replay()[0]

    def close(self):
        self.journal.close()

    def _append_observation(self, observation):
        validate_observation(observation)
        if self.journal.replay()[1] is not None:
            raise ValueError("an observation is already pending; step before observing again")
        self.journal.append({"type": "observation", "observation": observation})
        return observation

    def observe(self, observation, *, export_root=None):
        """Journal an external observation (operator tools); strict schema.

        An ``export_verified`` observation also needs ``export_root``, the local export copy:
        its report must be reproduced there (``reverify_export``) and be bound to this
        rental's scope (``check_export_binding``), so a self-consistent report that names
        no bytes on disk, or another rental's export, never licenses destroy (G1-01/G1-07).
        """
        validate_observation(observation)
        if observation["kind"] not in EXTERNAL_OBSERVATIONS:
            raise ValueError("this observation is produced by the coordinator only")
        if observation["kind"] == "export_verified":
            if export_root is None:
                raise ValueError("export_root_required")
            problems = (check_export_binding(observation["verification"], self.scope)
                        or reverify_export(observation, export_root))
            if problems:
                raise ValueError("export verification refused: " + ",".join(problems))
        return self._append_observation(observation)

    def observe_watchdog(self):
        """Journal the receipt read from the plan's own watchdog event file (A3-02)."""
        state = self.state
        return self._append_observation(read_watchdog_receipt(state["plan"]["watchdog_events"]))

    def step(self):
        state, pending = self.journal.replay()
        decision = next_action(state, pending, self.clock.time())
        self.journal.append({"type": "transition", "state": decision["state"],
                             "action": decision["action"], "observation": pending,
                             "issues": decision["issues"]})
        result = self._execute(decision["state"], decision["action"])
        if result is not None:
            self.journal.append({"type": "observation", "observation": result})
        return decision

    def _effect(self, operation, resource_id, call):
        """Request a side effect; journal its outcome as a static code, never trust it."""
        try:
            outcome = call(resource_id)
        except ProviderError as error:
            outcome = str(error)
        if type(outcome) is not str or not _OUTCOME.fullmatch(outcome):
            outcome = "unrecognised"
        self.journal.append({"type": "effect", "operation": operation,
                             "resource_id": resource_id, "outcome": outcome})

    def _show(self, instance_id):
        try:
            value = self.provider.show_instance(instance_id)
        except ProviderError as error:
            return {"status": "error", "error": str(error)}
        if value == NOT_FOUND:
            return {"status": "not_found"}
        if type(value) is not dict or value.get("id") != instance_id:
            return {"status": "error", "error": "instance_identity_mismatch"}
        return {"status": "row", "row": _filter_row(value)}

    def _inventory(self):
        try:
            rows = self.provider.list_instances()
            return {"status": "ok", "rows": [
                {"id": row["id"], "label": row.get("label") if type(
                    row.get("label")) is str else None} for row in rows]}
        except (ProviderError, KeyError, TypeError, AttributeError):
            return {"status": "error", "error": "inventory_failed"}

    def _execute(self, state, action):
        verb = action["action"]
        if verb == "create":
            # Same-step label check (A3R-01): a journaled inventory may be stale (a journal
            # restored from a copy taken before the create), so it never licenses a create.
            inventory = self._inventory()
            if inventory["status"] != "ok":
                return {"kind": "create_result", "outcome": "uncertain",
                        "error": "inventory_failed_before_create"}
            if _labelled(inventory, action["label"]):
                return {"kind": "create_result", "outcome": "uncertain",
                        "error": "label_present_before_create"}
            try:
                result = self.provider.create(action["offer_id"], action["image"],
                                              action["disk_gb"], action["label"])
            except ProviderError as error:
                result = {"outcome": "uncertain", "error": str(error)}
            if type(result) is not dict or result.get("outcome") not in ("created", "uncertain"):
                result = {"outcome": "uncertain", "error": "create_result_unrecognised"}
            return {"kind": "create_result", **result}
        if verb in ("check_label_absent", "reconcile_by_label"):
            return {"kind": "inventory", **self._inventory()}
        if verb == "observe_instance":
            identity = action["instance_id"]
            return {"kind": "instance_state", "results": {str(identity): self._show(identity)}}
        if verb == "hold_for_retention":
            return {"kind": "instance_state",
                    "results": {str(i): self._show(i) for i in action["instance_ids"]}}
        if verb == "stop":
            results = {}
            for identity in action["instance_ids"]:
                self._effect("stop", identity, self.provider.stop)
                results[str(identity)] = self._show(identity)
            return {"kind": "instance_state", "results": results}
        if verb == "destroy":
            for identity in action["instance_ids"]:
                self._effect("destroy", identity, self.provider.destroy)
            for identity in action["volume_ids"]:
                self._effect("destroy_volume", identity, self.provider.destroy_volume)
            instances = {str(i): self._show(i) for i in action["verify_instance_ids"]}
            try:
                rows = self.provider.list_volumes()
                volumes = {"status": "ok", "ids": sorted(
                    row["id"] for row in rows if type(row.get("id")) is int),
                    "label_matches": sum(state["label"] in (row.get("label"), row.get("name"))
                                         for row in rows)}
            except (ProviderError, TypeError, AttributeError):
                volumes = {"status": "error", "error": "volume_inventory_failed"}
            return {"kind": "absence_check", "instances": instances, "volumes": volumes,
                    "inventory": self._inventory()}
        return None  # wait, advance, arm/await watchdog, external work
