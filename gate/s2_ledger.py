"""Pure reconciliation of synthetic/local S2 launch receipts.

``timestamp_us`` is recorder event time in one monotonic clock domain, not a
wall-clock timestamp taken independently from remote services. Unique events
must arrive in nondecreasing order; the reconciler never sorts the evidence.
Repeated event IDs are idempotent only for identical canonical JSON values.

This v0 retains retries but cannot select them for inference: admission requires
one completed first attempt per job. It verifies receipt consistency, not their
authenticity or physical validity. It performs no I/O and grants no launch right.

Event schema versions (build-spec B1; lane R1 P4b). Versioning is per event, so
v1 and v2 events may share one batch:

- v1 (no ``schema_version`` key): exactly the original fields; ``completed``
  requires ``reason == "exact_horizon"`` and a non-null upstream identity.
- v2 (``schema_version: 2``, an int): the v1 fields plus ``evidence_refs`` (a
  nonempty list of unique nonempty texts naming the captured evidence the recorder
  bound; a v2 event that binds no evidence, above all a completion, is invalid).
  ``completed`` requires the neutral ``reason == "process_completed"``
  (``PROCESS_COMPLETED``) and a non-null upstream identity. It states only that
  the pinned upstream process evidence shows the attempt complete; it never
  claims an exact or a clipped horizon. The horizon is evidence-derived
  downstream (adapter/diagnostics), and clipped-horizon acceptance is Huy's.

Any other ``schema_version`` value (including an explicit ``1`` or a boolean) is
an invalid event. The report, job and attempt key sets are unchanged from v1
(``gate.s2_diagnostics`` validates them exactly); the version and the neutral
reason stay visible in each attempt's ``events`` and ``terminal_reason``.

Public API: ``reconcile_attempts`` and ``valid_event`` (the exact per-event schema check
the reconciler applies; review F6 shared change, so consumers such as
``gate.s2_telemetry_sources`` need not import the private ``_valid_event``, which is
kept as the same function object for existing importers).

Verification level: L1 for ``reconcile_attempts`` and ``valid_event`` (pure, literal
known answers).
"""

import json
import math
import re

_JOB_FIELDS = {
    "job_id",
    "phase",
    "arm",
    "policy_id",
    "scene_id",
    "road_id",
    "stratum",
    "replicate_id",
    "protocol_hash",
    "requested_steps",
}
# ``diagnostic_qualification`` is the phase of a declared diagnostic unit's job in its own
# ledger (gate.s2_packet ``DIAGNOSTIC_PHASE``; review G2-04), never a pilot or sweep job.
PHASES = ("harmonizer_pilot", "full_sweep", "synthetic", "diagnostic_qualification")
_EVENT_FIELDS = {"event_id", "job_id", "attempt_id", "type", "timestamp_us"}
_IDENTITY_FIELDS = {"run_uuid", "clipgt_id", "rollout_id"}
_TERMINAL = {"completed", "failed", "cancelled"}
EVENT_SCHEMA_VERSIONS = (1, 2)
PROCESS_COMPLETED = "process_completed"
_V2_FIELDS = {"schema_version", "evidence_refs"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _identity(value):
    return (
        isinstance(value, dict)
        and set(value) == _IDENTITY_FIELDS
        and all(_text(value[key]) for key in _IDENTITY_FIELDS)
    )


def _valid_job(value):
    return (
        isinstance(value, dict)
        and set(value) == _JOB_FIELDS
        and all(
            _text(value[key]) for key in ("job_id", "policy_id", "scene_id", "road_id", "stratum")
        )
        and value["phase"] in PHASES
        and value["arm"] in ("A", "B")
        and _integer(value["replicate_id"])
        and _integer(value["requested_steps"], 1)
        and isinstance(value["protocol_hash"], str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", value["protocol_hash"]) is not None
    )


def _event_version(value):
    """1 for an unversioned event, 2 for ``schema_version: 2``; None otherwise."""
    if "schema_version" not in value:
        return 1
    version = value["schema_version"]
    return 2 if type(version) is int and version == 2 else None


def _valid_refs(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(_text(item) for item in value)
        and len(set(value)) == len(value)
    )


def _valid_event(value):
    if not isinstance(value, dict):
        return False
    version = _event_version(value)
    if version is None:
        return False
    kind = value.get("type")
    extra = (
        {"parent_attempt_id", "retry_reason", "upstream_identity"}
        if kind == "started"
        else {"reason", "upstream_identity"}
    )
    if version == 2:
        extra = extra | _V2_FIELDS
        if not _valid_refs(value.get("evidence_refs")):
            return False
    if (
        set(value) != _EVENT_FIELDS | extra
        or kind not in ("started", "completed", "failed", "cancelled")
        or not all(_text(value[key]) for key in ("event_id", "job_id", "attempt_id"))
        or not _integer(value["timestamp_us"])
    ):
        return False
    upstream = value["upstream_identity"]
    if upstream is not None and not _identity(upstream):
        return False
    if kind == "started":
        parent, reason = value["parent_attempt_id"], value["retry_reason"]
        return (parent is None and reason is None) or (_text(parent) and _text(reason))
    completion_reason = "exact_horizon" if version == 1 else PROCESS_COMPLETED
    return (
        (value["reason"] == completion_reason and upstream is not None)
        if kind == "completed"
        else _text(value["reason"])
    )


# Public alias (review F6); ``_valid_event`` stays importable for existing callers.
valid_event = _valid_event


def reconcile_attempts(schedule, events):
    """Return all receipts, reconciled jobs/attempts, issues and conservative admission.

    Inputs are JSON-native lists. Malformed rows remain in ``schedule``/``events``;
    non-JSON values become null with an explicit issue, rather than being coerced
    into an invented observation. No job is selected when any batch issue exists.
    """
    issues = []

    def issue(code, path, message):
        issues.append({"code": code, "path": path, "message": message})

    def snapshot(value, path, parents=None):
        parents = set() if parents is None else parents
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if isinstance(value, (dict, list)) and id(value) not in parents:
            branch = parents | {id(value)}
            if isinstance(value, list):
                return [snapshot(item, f"{path}[{i}]", branch) for i, item in enumerate(value)]
            result = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    issue("invalid_json_value", path, "Non-string JSON object key omitted.")
                    continue
                result[key] = snapshot(item, f"{path}.{key}", branch)
            return result
        issue("invalid_json_value", path, "Non-JSON or cyclic value retained as null.")
        return None

    raw_schedule, raw_events = snapshot(schedule, "schedule"), snapshot(events, "events")
    if not isinstance(raw_schedule, list) or not raw_schedule:
        issue("invalid_schedule", "schedule", "Require a nonempty explicit job list.")
    if not isinstance(raw_events, list):
        issue("invalid_events", "events", "Require an event list.")
    schedule_rows = raw_schedule if isinstance(raw_schedule, list) else []
    event_rows = raw_events if isinstance(raw_events, list) else []
    jobs, job_by_id, logical_jobs, scenes, roads = [], {}, set(), {}, {}
    for index, entry in enumerate(schedule_rows):
        path = f"schedule[{index}]"
        if not _valid_job(entry):
            issue("invalid_schedule_entry", path, "Require the exact typed job schema.")
            continue
        job_id = entry["job_id"]
        logical = (
            entry["phase"],
            entry["arm"],
            entry["scene_id"],
            entry["replicate_id"],
            entry["protocol_hash"].lower(),
        )
        if job_id in job_by_id:
            issue("duplicate_job_id", path, "Job ID occurs more than once.")
        if logical in logical_jobs:
            issue("duplicate_logical_job", path, "Logical job occurs more than once.")
        scene = entry["scene_id"]
        pair = (entry["road_id"], entry["stratum"])
        if scene in scenes and scenes[scene] != pair:
            issue("scene_identity_conflict", path, "Scene maps to conflicting road/stratum.")
        if entry["road_id"] in roads and roads[entry["road_id"]] != entry["stratum"]:
            issue("road_stratum_conflict", path, "A road belongs to conflicting strata.")
        scenes.setdefault(scene, pair)
        roads.setdefault(entry["road_id"], entry["stratum"])
        logical_jobs.add(logical)
        record = {**entry, "attempt_ids": [], "status": "missing", "selected_attempt_id": None}
        jobs.append(record)
        job_by_id.setdefault(job_id, record)

    attempts, attempt_by_key, attempt_owner, seen_events, upstream_owner = [], {}, {}, {}, {}
    attempt_started = set()
    last_timestamp = None
    idempotent_events = 0
    for index, event in enumerate(event_rows):
        path = f"events[{index}]"
        event_id = event.get("event_id") if isinstance(event, dict) else None
        fingerprint = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if _text(event_id) and event_id in seen_events:
            if seen_events[event_id] == fingerprint:
                idempotent_events += 1
            else:
                issue("conflicting_event_id", path, "Repeated event ID has different content.")
            if isinstance(event, dict):
                job_id, attempt_id = event.get("job_id"), event.get("attempt_id")
                if _text(job_id) and _text(attempt_id):
                    previous = attempt_by_key.get((job_id, attempt_id))
                    if previous is not None:
                        previous["events"].append(event)
            continue
        if _text(event_id):
            seen_events[event_id] = fingerprint
        if not _valid_event(event):
            issue("invalid_event", path, "Require the exact typed event and terminal schema.")
            continue
        job_id, attempt_id, kind = event["job_id"], event["attempt_id"], event["type"]
        key = (job_id, attempt_id)
        timestamp = event["timestamp_us"]
        if last_timestamp is not None and timestamp < last_timestamp:
            issue("timestamp_regression", path, "Recorder timestamps regress in recorded order.")
        last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
        if job_id not in job_by_id:
            issue("unknown_job", path, "Event is not bound to a scheduled job.")
        if attempt_id in attempt_owner and attempt_owner[attempt_id] != job_id:
            issue("reused_attempt_id", path, "Attempt ID belongs to more than one job.")
        attempt_owner.setdefault(attempt_id, job_id)
        if key not in attempt_by_key:
            record = {
                "job_id": job_id,
                "attempt_id": attempt_id,
                "parent_attempt_id": None,
                "retry_reason": None,
                "upstream_identity": None,
                "state": "incomplete",
                "started_timestamp_us": None,
                "terminal_timestamp_us": None,
                "terminal_reason": None,
                "events": [],
            }
            attempt_by_key[key] = record
            attempts.append(record)
            if job_id in job_by_id:
                job_by_id[job_id]["attempt_ids"].append(attempt_id)
        attempt = attempt_by_key[key]
        attempt["events"].append(event)
        upstream = event["upstream_identity"]
        if upstream is not None:
            identity_key = tuple(upstream[field] for field in sorted(_IDENTITY_FIELDS))
            if identity_key in upstream_owner and upstream_owner[identity_key] != key:
                issue(
                    "reused_upstream_identity", path, "Upstream identity belongs to two attempts."
                )
            upstream_owner.setdefault(identity_key, key)
            if attempt["upstream_identity"] is None:
                attempt["upstream_identity"] = dict(upstream)
            elif attempt["upstream_identity"] != upstream:
                issue("upstream_identity_changed", path, "Attempt upstream identity changed.")
        if kind == "started":
            if key in attempt_started or len(attempt["events"]) != 1:
                issue("invalid_transition", path, "Attempt starts more than once or after its end.")
                continue
            previous_ids = job_by_id[job_id]["attempt_ids"][:-1] if job_id in job_by_id else []
            parent = event["parent_attempt_id"]
            if previous_ids and parent is None:
                issue(
                    "retry_parent_missing", path, "Additional attempt requires an explicit parent."
                )
            if parent is not None:
                parent_record = attempt_by_key.get((job_id, parent))
                if (
                    not previous_ids
                    or parent != previous_ids[-1]
                    or parent_record is None
                    or parent_record["state"] not in _TERMINAL
                    or timestamp < parent_record["terminal_timestamp_us"]
                ):
                    issue(
                        "retry_parent_invalid", path, "Parent must be the latest terminal attempt."
                    )
            attempt_started.add(key)
            attempt["parent_attempt_id"] = parent
            attempt["retry_reason"] = event["retry_reason"]
            attempt["started_timestamp_us"] = timestamp
        elif key not in attempt_started or attempt["state"] in _TERMINAL:
            issue("invalid_transition", path, "Terminal event requires one open started attempt.")
        else:
            attempt["state"] = kind
            attempt["terminal_timestamp_us"] = timestamp
            attempt["terminal_reason"] = event["reason"]

    for attempt in attempts:
        if attempt["state"] == "incomplete":
            issue(
                "incomplete_attempt", f"attempt:{attempt['attempt_id']}", "No valid terminal event."
            )
    for job_record in jobs:
        job_id = job_record["job_id"]
        # Duplicate schedule rows remain visible; only the first receives bindings.
        members = [attempt_by_key[(job_id, value)] for value in job_record["attempt_ids"]]
        path = f"job:{job_id}"
        completed = [attempt for attempt in members if attempt["state"] == "completed"]
        if not members:
            issue("missing_job", path, "Scheduled job has no attempt.")
        elif len(members) > 1:
            job_record["status"] = "invalid"
            issue("retry_selection_unregistered", path, "v0 cannot select among multiple attempts.")
            if len(completed) > 1:
                issue(
                    "multiple_completed_attempts",
                    path,
                    "Multiple completed attempts are ambiguous.",
                )
        else:
            job_record["status"] = members[0]["state"]
            if not completed:
                issue("job_not_completed", path, "Job has no completed first attempt.")
    admitted = not issues
    if admitted:
        for job_record in jobs:
            job_record["selected_attempt_id"] = job_record["attempt_ids"][0]
    counts = {
        "scheduled_jobs": len(schedule_rows),
        "attempts": len(attempts),
        "missing_jobs": sum(not row["attempt_ids"] for row in jobs),
        "idempotent_events": idempotent_events,
        **{
            f"{state}_attempts": sum(row["state"] == state for row in attempts)
            for state in ("completed", "failed", "cancelled", "incomplete")
        },
    }
    return {
        "admitted": admitted,
        "issues": issues,
        "jobs": jobs,
        "attempts": attempts,
        "counts": counts,
        "schedule": raw_schedule,
        "events": raw_events,
    }
