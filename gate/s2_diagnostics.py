"""S2 validity and complete-denominator report (readiness CPU-04; build-spec A5).

Serves S2.3/S2.5 [C] (G2). Agent-authored evaluator engineering under D-018/D-024.

``describe_validity`` is a descriptive diagnostic over the outputs of the existing
W2 consumer (``gate.s2_ledger.reconcile_attempts``) and W3 synthetic reader
(``gate.s2_extract.extract_bundle`` observations, plus optionally the adapter's
normalized ``records``). It never parses raw Parquet/ASL bytes (no second
extractor), never selects a retry, never substitutes a missing value with zero and
never emits an admitted verdict: ``gate_eligible`` is always ``False`` and no rows
are forwarded to ``gate.api``/``gate.verdict``. A report can be produced for a
rejected batch; it remains diagnostic only.

Verification level per function (build-spec section 1):

- ``deviation_bin`` - L1: pure half-open binning with literal known answers.
- ``describe_validity`` - L1 over synthetic producer output. Its use on an actual
  capture is data-gated (L3: ``qualify_on_host`` with admitted actual inputs).

Interface reconciliation (recorded, not silently chosen): build-spec A5 names
``describe_validity(ledger_report, observations, records=None)``; readiness CPU-04
names ``describe_validity(schedule, attempts, observations, validity_definition=None)``.
The ledger report already carries the schedule, jobs and attempts, so the build-spec
form is used, with the owner-supplied ``validity_definition`` added as a keyword.

Owner inputs (no defaults, never chosen here):

- ``validity_definition`` - Huy's validity semantics. ``None`` blocks the A-valid x
  B-valid table with ``validity_definition_unregistered`` while all other facts are
  still computed. A supplied definition is hashed and labelled
  ``supplied_not_authenticated``; the hash proves bytes, not that Huy approved it.
  Schema (exact keys): ``schema_version`` (1), ``definition_id`` (text),
  ``source_ref`` (text) and ``invalid_when`` (nonempty list of distinct names from
  ``INDICATORS``). A unit is invalid if any named indicator is true, valid if all are
  known false, otherwise unknown (three-valued logic; unknown never becomes valid).

Observation consumer contract. Required binding keys: ``attempt_id``, ``job_id``,
``run_uuid``, ``clipgt_id``, ``rollout_id``. The remaining W3 keys are read when
present; an absent or ``None`` value is *unknown*, never zero. Two optional
full-trajectory extension booleans, ``corridor_departure`` and
``endpoint_overshoot``, are accepted for a later actual-capture producer; the current
synthetic reader does not emit them, so those indicators are unknown. Summary
aggregates (``score_metrics``) are never read: they may be truncated and cannot
establish full-trajectory validity. Any other key fails closed.

Deviation bins are half-open in metres: ``[0,1)``, ``[1,2)``, ``[2,4)``, ``[4,inf)``
plus ``unknown`` (missing, ``None``, boolean, negative, nonfinite or non-numeric).
``out_of_envelope`` is ``full_max_deviation_m >= 4`` and is listed separately.

Agent technical defaults (D-018): units are described from the ledger's selected
attempt when the ledger admitted the batch, otherwise from a job's sole attempt;
a job with several attempts is described at attempt level only (no selection).
A selection is trusted only when it is one of the job's own attempts and, under W2
v0 admission, the job's sole completed attempt; any other selection (or one without
admission) is ``ledger_inconsistent`` and the unit stays undescribed with reason
``selection_inconsistent``. ``out_of_envelope`` lists described jobs and, separately,
every bound >= 4 m attempt; one outside any described unit is also warned. Rows
sharing a job ID carry ``duplicate_job_id`` and their own row status only, because
W2's job-keyed issues cannot be attributed to one of them.
Black frames are known-false only when the count is zero *and* render coverage is
complete. ``records`` elements are bare adapter records, joined to jobs on
``(arm, scene_id, replicate_id)``, or ``extract_bundle`` batch entries
(``arm, protocol_hash, policy_id, normalization``), joined with their protocol. A
batch entry's normalization must carry a boolean ``admitted`` and a ``records`` list
(other adapter keys are tolerated, not read); a non-admitted normalization yields
``records_on_rejected_batch`` and its records are unused. A batch entry is hashed as
its arm, protocol, policy and the normalization keys read here: ``admitted``,
``records`` and ``issues`` (each issue projected to its ``code``, ``message`` and
``identity``, a tuple identity as a JSON list); other adapter metadata is not hashed.
``records.count`` is every supplied record whether or not it is used (one per bare
element; a batch entry's ``records`` list whenever it is a list, admitted or not;
``null`` when any batch entry's ``records`` is not a list, since missing is never
zero); ``records.bound`` counts only records joined to a job. ``upstream_acceptance`` states
the W2 and W3 outcomes the inputs carry; ``not_stated`` means bare records carry no W3 flag. A
join that is not unique (bare records over the pilot's two Harmonizer protocols)
is reported as ``record_binding_ambiguous`` rather than guessed. The A-valid x
B-valid table couples arms on ``(phase, scene_id, replicate_id, protocol_hash)``;
for unseeded replicates this replicate-index coupling is arbitrary, so the table
labels it an agent default with the owner pairing convention unregistered.

This module never reads files other than its own source (for producer identity),
never performs network or subprocess calls, never authenticates sidecars, never
computes a missingness-sensitivity artifact and never imports protected inference.
"""

import hashlib
import math
from pathlib import Path

from gate.s2_identity import canonical_json_bytes, identity_hash

ARTIFACT_KIND = "s2_validity_report"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
DEVIATION_BINS = ("[0,1)", "[1,2)", "[2,4)", "[4,inf)")
ENVELOPE_M = 4.0
INDICATORS = (
    "out_of_envelope",
    "corridor_departure",
    "endpoint_overshoot",
    "terminal_not_exact_horizon",
    "render_incomplete",
    "black_frame",
    "nonfinite_pose",
    "map_incomplete",
)
TABLE_CELLS = (
    "A_valid_B_valid",
    "A_valid_B_invalid",
    "A_invalid_B_valid",
    "A_invalid_B_invalid",
)

_LEDGER_KEYS = {"admitted", "issues", "jobs", "attempts", "counts", "schedule", "events"}
_JOB_KEYS = {
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
    "attempt_ids",
    "status",
    "selected_attempt_id",
}
_ATTEMPT_KEYS = {
    "job_id",
    "attempt_id",
    "parent_attempt_id",
    "retry_reason",
    "upstream_identity",
    "state",
    "started_timestamp_us",
    "terminal_timestamp_us",
    "terminal_reason",
    "events",
}
_JOB_TEXT = (
    "job_id",
    "phase",
    "arm",
    "policy_id",
    "scene_id",
    "road_id",
    "stratum",
    "protocol_hash",
    "status",
)
_RECOMPUTED_COUNTS = (
    "scheduled_jobs",
    "attempts",
    "missing_jobs",
    "completed_attempts",
    "failed_attempts",
    "cancelled_attempts",
    "incomplete_attempts",
)
_IDENTITY_KEYS = ("run_uuid", "clipgt_id", "rollout_id")
_BINDING_KEYS = {"attempt_id", "job_id", *_IDENTITY_KEYS}
_W3_KEYS = {
    "requested_steps",
    "actual_steps",
    "completion_validated",
    "required_metrics_verified",
    "provenance_ref",
    "full_max_deviation_m",
    "metric_names",
    "run_name",
    "raw_metric_rows",
    "render_coverage",
    "black_frame_count",
    "first_black_timestamp",
    "pose_nonfinite_count",
    "first_nonfinite_timestamp",
    "map_coverage",
}
_EXTENSION_KEYS = {"corridor_departure", "endpoint_overshoot"}
_OBSERVATION_KEYS = _BINDING_KEYS | _W3_KEYS | _EXTENSION_KEYS
_RECORD_KEYS = {
    "arm",
    "scene_id",
    "road_id",
    "stratum",
    "replicate_id",
    "value",
    "completed",
    "deviation_m",
    "seed",
}
_RECORD_TEXT = ("arm", "scene_id", "road_id", "stratum")
_BATCH_KEYS = {"arm", "protocol_hash", "policy_id", "normalization"}
_DEFINITION_KEYS = {"schema_version", "definition_id", "source_ref", "invalid_when"}
_PAIR_KEY = ("phase", "scene_id", "replicate_id", "protocol_hash")


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _text(value):
    return type(value) is str and bool(value.strip())


def _count(value):
    return type(value) is int and value >= 0


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int beyond float range is not a supported metre value
        return False


def _issue_view(item):
    """Return the JSON-safe projection of one adapter normalization issue.

    An adapter issue is ``{code, identity, message}`` with ``identity`` a tuple or
    ``None``; the projection keeps exactly those keys that are present, with a tuple
    identity as a list. Anything else is returned unchanged, so a non-JSON value still
    fails closed in ``canonical_json_bytes``.
    """
    if type(item) is not dict:
        return item
    view = {key: item[key] for key in ("code", "message", "identity") if key in item}
    if type(view.get("identity")) is tuple:
        view["identity"] = list(view["identity"])
    return view


def _entry_view(entry):
    """Return the hashable view of one ``records`` element.

    An ``extract_bundle`` batch entry is reduced to its identity and to the
    normalization keys this module reads: ``admitted``, ``records`` and ``issues``
    (each issue projected by ``_issue_view``). Keys absent from the normalization stay
    absent, so absent and empty ``issues`` hash differently. Other adapter metadata
    (diagnostics, counts) is neither consumed nor hashed. Anything else is returned
    unchanged (a bare adapter record, or a malformed element).
    """
    if type(entry) is dict and set(entry) == _BATCH_KEYS and type(entry["normalization"]) is dict:
        normalization = entry["normalization"]
        view = {key: normalization[key] for key in ("admitted", "records") if key in normalization}
        if "issues" in normalization:
            issues = normalization["issues"]
            view["issues"] = (
                [_issue_view(item) for item in issues] if type(issues) is list else issues
            )
        return {
            "arm": entry["arm"],
            "protocol_hash": entry["protocol_hash"],
            "policy_id": entry["policy_id"],
            "normalization": view,
        }
    return entry


def deviation_bin(value):
    """Return the half-open bin of a full-trajectory deviation in metres (L1).

    Anything that is not a finite nonnegative int/float (including ``None`` and
    booleans) is ``unknown``; a missing deviation is never treated as zero.
    """
    if not _finite_number(value) or value < 0:
        return "unknown"
    if value < 1:
        return DEVIATION_BINS[0]
    if value < 2:
        return DEVIATION_BINS[1]
    if value < ENVELOPE_M:
        return DEVIATION_BINS[2]
    return DEVIATION_BINS[3]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_ledger(ledger_report):
    _require(type(ledger_report) is dict, "ledger_report must be a reconcile_attempts report")
    _require(set(ledger_report) == _LEDGER_KEYS, "ledger_report keys differ from the W2 report")
    _require(type(ledger_report["admitted"]) is bool, "ledger_report.admitted must be boolean")
    for name in ("issues", "jobs", "attempts"):
        _require(type(ledger_report[name]) is list, f"ledger_report.{name} must be a list")
    _require(type(ledger_report["counts"]) is dict, "ledger_report.counts must be an object")
    for item in ledger_report["issues"]:
        _require(
            type(item) is dict
            and {"code", "path", "message"} <= set(item)
            and _text(item["code"])
            and type(item["path"]) is str,
            "ledger issue requires text code and path, and a message",
        )
    for job in ledger_report["jobs"]:
        _require(type(job) is dict and set(job) == _JOB_KEYS, "ledger job keys differ from W2")
        _require(
            all(type(job[key]) is str for key in _JOB_TEXT)
            and _text(job["job_id"])
            and type(job["replicate_id"]) is int
            and type(job["requested_steps"]) is int
            and type(job["attempt_ids"]) is list
            and all(_text(value) for value in job["attempt_ids"])
            and (job["selected_attempt_id"] is None or _text(job["selected_attempt_id"])),
            "ledger job fields have unexpected types",
        )
    for attempt in ledger_report["attempts"]:
        _require(
            type(attempt) is dict and set(attempt) == _ATTEMPT_KEYS,
            "ledger attempt keys differ from W2",
        )
        _require(
            _text(attempt["attempt_id"])
            and _text(attempt["job_id"])
            and type(attempt["state"]) is str
            and (attempt["parent_attempt_id"] is None or _text(attempt["parent_attempt_id"]))
            and (
                attempt["upstream_identity"] is None or type(attempt["upstream_identity"]) is dict
            ),
            "ledger attempt fields have unexpected types",
        )


def _check_definition(definition):
    if (
        type(definition) is not dict
        or set(definition) != _DEFINITION_KEYS
        or type(definition["schema_version"]) is not int
        or definition["schema_version"] != 1
        or not _text(definition["definition_id"])
        or not _text(definition["source_ref"])
        or type(definition["invalid_when"]) is not list
        or not definition["invalid_when"]
    ):
        return (
            "Definition requires exactly schema_version 1, definition_id, source_ref and a "
            "nonempty invalid_when list"
        )
    criteria = definition["invalid_when"]
    if any(type(name) is not str or name not in INDICATORS for name in criteria):
        return "invalid_when names an indicator outside the supported vocabulary"
    if len(set(criteria)) != len(criteria):
        return "invalid_when repeats an indicator"
    return None


def _validity(indicators, criteria):
    values = [indicators[name] for name in criteria]
    if any(value is True for value in values):
        return "invalid"
    if all(value is False for value in values):
        return "valid"
    return "unknown"


def describe_validity(
    ledger_report, observations, records=None, *, validity_definition=None, scope=None
):
    """Return a deterministic CPU-04 validity/denominator artifact (diagnostic only).

    ``ledger_report`` is the unmodified ``reconcile_attempts`` output (or the
    ``ledger`` member of an ``extract_bundle`` report); ``observations`` are W3
    observation rows; ``records`` are optional adapter normalized records. ``scope``
    is derived as ``synthetic`` only when every scheduled job has phase
    ``synthetic``; otherwise it must be declared. Structural input errors raise
    ``ValueError``; data defects become issues and set ``status`` to ``blocked``.
    ``status == "passed"`` means the description is internally consistent, never that
    any unit or batch is admitted.
    """
    _check_ledger(ledger_report)
    _require(type(observations) is list, "observations must be a list")
    _require(records is None or type(records) is list, "records must be a list or None")
    # canonical_json_bytes rejects non-JSON values (NaN, cycles, non-string keys).
    ledger_hash = identity_hash(ledger_report)
    observations_hash = hashlib.sha256(
        b"\n".join(sorted(canonical_json_bytes(row) for row in observations))
    ).hexdigest()
    records_hash = (
        None
        if records is None
        else hashlib.sha256(
            b"\n".join(sorted(canonical_json_bytes(_entry_view(row)) for row in records))
        ).hexdigest()
    )
    definition_hash = None if validity_definition is None else identity_hash(validity_definition)

    jobs = ledger_report["jobs"]
    phases = {job["phase"] for job in jobs}
    if scope is None:
        _require(
            bool(jobs) and phases == {"synthetic"},
            "scope must be declared unless every scheduled job is synthetic",
        )
        scope = "synthetic"
    _require(scope in SCOPES, f"scope must be one of {SCOPES}")

    issues = []

    def issue(code, path, message, severity="error", refs=()):
        issues.append(
            {
                "code": code,
                "path": path,
                "severity": severity,
                "evidence_refs": sorted(refs),
                "message": message,
            }
        )

    if scope != "synthetic" and "synthetic" in phases:
        issue("scope_phase_mismatch", "scope", f"Scope {scope} declared over synthetic jobs.")

    # --- W2 reconciliation --------------------------------------------------------
    admitted = ledger_report["admitted"]
    attempts = ledger_report["attempts"]
    attempt_by_id = {}
    for attempt in attempts:
        if attempt["attempt_id"] in attempt_by_id:
            issue("ledger_inconsistent", f"attempt:{attempt['attempt_id']}", "Repeated attempt.")
        attempt_by_id.setdefault(attempt["attempt_id"], attempt)
    job_ids = [job["job_id"] for job in jobs]
    seen_job_ids, duplicate_jobs = set(), set()
    for job_id in job_ids:
        (duplicate_jobs if job_id in seen_job_ids else seen_job_ids).add(job_id)
    schedule = ledger_report["schedule"]
    recomputed = {
        "scheduled_jobs": len(schedule) if type(schedule) is list else 0,
        "attempts": len(attempts),
        "missing_jobs": sum(not job["attempt_ids"] for job in jobs),
        **{
            f"{state}_attempts": sum(attempt["state"] == state for attempt in attempts)
            for state in ("completed", "failed", "cancelled", "incomplete")
        },
    }
    for name in _RECOMPUTED_COUNTS:
        if ledger_report["counts"].get(name) != recomputed[name]:
            issue("ledger_counts_mismatch", f"counts.{name}", "W2 count does not reconcile.")
    if admitted != (not ledger_report["issues"]):
        issue("ledger_inconsistent", "admitted", "Admission disagrees with the W2 issue list.")
    for job in jobs:
        for attempt_id in job["attempt_ids"]:
            owner = attempt_by_id.get(attempt_id)
            if owner is None or owner["job_id"] != job["job_id"]:
                issue("ledger_inconsistent", f"job:{job['job_id']}", "Attempt binding differs.")
    job_by_id = {}
    for job in jobs:
        job_by_id.setdefault(job["job_id"], job)
    unknown_job_attempts = sum(attempt["job_id"] not in job_by_id for attempt in attempts)
    ledger_reasons = {}
    for item in ledger_report["issues"]:
        path = item["path"]
        if type(path) is str and path.startswith("attempt:"):
            attempt = attempt_by_id.get(path[len("attempt:") :])
            path = None if attempt is None else f"job:{attempt['job_id']}"
        if type(path) is str and path.startswith("job:"):
            ledger_reasons.setdefault(path[len("job:") :], set()).add(item["code"])

    # --- bind observations --------------------------------------------------------
    bound, conflicted_attempts, rejected_jobs = {}, set(), set()
    for index, row in enumerate(observations):
        # Paths name the attempt, not the list position, so input order cannot change them.
        path = (
            f"observation:{row['attempt_id']}"
            if type(row) is dict and _text(row.get("attempt_id"))
            else f"observation:sha256:{identity_hash(row)}"
        )
        if type(row) is not dict or not _BINDING_KEYS <= set(row):
            issue("invalid_observation", path, "Observation lacks binding keys.")
            continue
        if not all(_text(row[key]) for key in _BINDING_KEYS):
            issue("invalid_observation", path, "Observation binding keys must be text.")
            continue
        attempt = attempt_by_id.get(row["attempt_id"])
        if attempt is None:
            issue("observation_unbound", path, "Observation names no ledger attempt.")
            continue
        unknown = sorted(set(row) - _OBSERVATION_KEYS)
        reason = None
        if unknown:
            reason = ("unknown_observation_field", f"Unsupported fields: {unknown}")
        elif row["job_id"] != attempt["job_id"]:
            reason = ("observation_job_mismatch", "Observation job differs from its attempt.")
        elif attempt["upstream_identity"] is None or any(
            attempt["upstream_identity"].get(key) != row[key] for key in _IDENTITY_KEYS
        ):
            reason = ("observation_identity_mismatch", "Upstream identity is not the attempt's.")
        else:
            job = job_by_id.get(attempt["job_id"])
            if (
                job is not None
                and "requested_steps" in row
                and row["requested_steps"] != job["requested_steps"]
            ):
                reason = (
                    "observation_requested_steps_mismatch",
                    "Observation horizon request differs from the schedule.",
                )
        if reason is not None:
            issue(reason[0], path, reason[1], refs=[row["attempt_id"]])
            rejected_jobs.add(attempt["job_id"])
            continue
        if row["attempt_id"] in bound:
            issue("duplicate_observation", path, "Attempt has several observations.")
            conflicted_attempts.add(row["attempt_id"])
            continue
        bound[row["attempt_id"]] = (index, row)
    for attempt_id in conflicted_attempts:
        bound.pop(attempt_id, None)
        rejected_jobs.add(attempt_by_id[attempt_id]["job_id"])

    def describe(row, path):
        """Return (deviation, bin, indicators, terminal) for one bound observation."""

        def present(key):
            return row.get(key)

        def bad(key):
            issue("invalid_observation_value", f"{path}.{key}", f"Unsupported value for {key}.")

        deviation = present("full_max_deviation_m")
        bin_name = deviation_bin(deviation)
        if deviation is None:
            issue("deviation_unknown", path, "Full deviation absent; binned unknown.", "warning")
        elif bin_name == "unknown":
            issue("invalid_deviation", path, "Deviation is not a finite nonnegative number.")
        known_deviation = deviation if bin_name != "unknown" else None
        indicators = dict.fromkeys(INDICATORS)
        if known_deviation is not None:
            indicators["out_of_envelope"] = known_deviation >= ENVELOPE_M
        for key in sorted(_EXTENSION_KEYS):
            value = present(key)
            if type(value) is bool:
                indicators[key] = value
            elif value is not None:
                bad(key)
        render = present("render_coverage")
        if render in ("complete", "partial"):
            indicators["render_incomplete"] = render == "partial"
        elif render is not None:
            bad("render_coverage")
        coverage = present("map_coverage")
        if coverage in ("complete", "partial"):
            indicators["map_incomplete"] = coverage == "partial"
        elif coverage is not None:
            bad("map_coverage")
        black = present("black_frame_count")
        if _count(black):
            if black > 0:
                indicators["black_frame"] = True
            elif render == "complete":
                indicators["black_frame"] = False
        elif black is not None:
            bad("black_frame_count")
        nonfinite = present("pose_nonfinite_count")
        if _count(nonfinite):
            indicators["nonfinite_pose"] = nonfinite > 0
        elif nonfinite is not None:
            bad("pose_nonfinite_count")
        requested, actual = present("requested_steps"), present("actual_steps")
        terminal = "unknown"
        if _count(requested) and _count(actual) and requested >= 1:
            if actual < requested:
                terminal = "short_horizon"
            elif actual > requested:
                bad("actual_steps")
            elif present("completion_validated") is True:
                terminal = "exact_horizon"
        elif requested is not None or actual is not None:
            bad("actual_steps")
        if terminal != "unknown":
            indicators["terminal_not_exact_horizon"] = terminal != "exact_horizon"
        return known_deviation, bin_name, indicators, terminal

    attempt_rows = []
    described_by_attempt = {}
    for attempt_id in sorted(bound):
        index, row = bound[attempt_id]
        attempt = attempt_by_id[attempt_id]
        deviation, bin_name, indicators, terminal = describe(row, f"observation:{attempt_id}")
        described_by_attempt[attempt_id] = (deviation, bin_name, indicators, terminal)
        attempt_rows.append(
            {
                "attempt_id": attempt_id,
                "job_id": attempt["job_id"],
                "attempt_state": attempt["state"],
                "parent_attempt_id": attempt["parent_attempt_id"],
                "deviation_m": deviation,
                "deviation_bin": bin_name,
                "terminal": terminal,
                "indicators": indicators,
            }
        )

    # --- validity definition ------------------------------------------------------
    criteria = None
    definition_problem = None
    if validity_definition is not None:
        definition_problem = _check_definition(validity_definition)
        if definition_problem is None:
            criteria = sorted(validity_definition["invalid_when"])
        else:
            issue("invalid_validity_definition", "validity_definition", definition_problem)

    # --- units (one per scheduled job) --------------------------------------------
    units = []
    for position, job in enumerate(jobs):
        job_id = job["job_id"]
        duplicate = job_id in duplicate_jobs
        # W2 job issues are keyed by job ID; for a repeated ID they cannot be attributed
        # to one schedule row, so both rows carry duplicate_job_id plus their own status.
        reasons = {"duplicate_job_id"} if duplicate else set(ledger_reasons.get(job_id, ()))
        if not admitted:
            reasons.add("batch_not_admitted")
        attempt_ids = job["attempt_ids"]
        selected = job["selected_attempt_id"]
        selection_problem = None
        if selected is not None and selected not in attempt_ids:
            selection_problem = "Selected attempt is not one of the job's attempts."
        elif admitted and (
            selected is None
            or len(attempt_ids) != 1
            or selected != attempt_ids[0]
            or job["status"] != "completed"
            or attempt_by_id.get(selected, {}).get("state") != "completed"
        ):
            selection_problem = "W2 v0 admission selects exactly the job's sole completed attempt."
        elif not admitted and selected is not None:
            selection_problem = "A selection exists without W2 admission."
        if selection_problem is not None:
            issue(
                "ledger_inconsistent",
                f"job:{job_id}",
                selection_problem,
                refs=[] if selected is None else [selected],
            )
            reasons.add("selection_inconsistent")
        described_attempt = None
        if not duplicate and selection_problem is None:
            candidate = (
                selected if admitted else (attempt_ids[0] if len(attempt_ids) == 1 else None)
            )
            if candidate is not None and attempt_by_id.get(candidate, {}).get("job_id") == job_id:
                described_attempt = candidate
        description = described_by_attempt.get(described_attempt)
        if description is None:
            described_attempt = None
            if not duplicate and selection_problem is None:
                reasons.add(
                    "observation_conflict" if job_id in rejected_jobs else "observation_missing"
                )
            deviation, bin_name = None, "unknown"
            indicators, terminal = dict.fromkeys(INDICATORS), "unknown"
        else:
            deviation, bin_name, indicators, terminal = description
            if job_id in rejected_jobs:
                reasons.add("observation_conflict")
        if job["status"] != "completed":
            reasons.add("job_not_completed" if attempt_ids else "missing_job")
        validity, validity_reason = None, None
        if criteria is not None:
            if job["status"] != "completed" or description is None:
                validity, validity_reason = "unknown", "no_completed_described_observation"
            else:
                validity = _validity(indicators, criteria)
                if validity == "unknown":
                    validity_reason = "criterion_unknown"
        units.append(
            {
                "position": position,
                "job_id": job_id,
                "phase": job["phase"],
                "arm": job["arm"],
                "scene_id": job["scene_id"],
                "road_id": job["road_id"],
                "stratum": job["stratum"],
                "replicate_id": job["replicate_id"],
                "protocol_hash": job["protocol_hash"],
                "ledger_status": job["status"],
                "attempt_ids": list(attempt_ids),
                "retry_count": max(0, len(attempt_ids) - 1),
                "selected_attempt_id": job["selected_attempt_id"],
                "described_attempt_id": described_attempt,
                "deviation_m": deviation,
                "deviation_bin": bin_name,
                "terminal": terminal,
                "indicators": indicators,
                "validity": validity,
                "validity_reason": validity_reason,
                "reasons": reasons,
            }
        )

    # --- records cross-check (optional) -------------------------------------------
    # Elements are either bare adapter records, joined on (arm, scene_id, replicate_id),
    # or extract_bundle batch entries, whose protocol_hash disambiguates the join.
    # records.count is every supplied record, used or not: a bare element counts one and
    # a batch entry contributes the length of its normalization's records list whenever
    # that is a list (admitted, rejected or otherwise malformed). A batch entry whose
    # records value is not a list cannot be counted, so the count is null (missing is
    # never zero). bound/out_of_envelope count only records actually joined to a job.
    flat_records, malformed_entries, rejected_entries, bare_entries, batch_entries = [], 0, 0, 0, 0
    supplied_records, uncountable_entries = 0, 0
    views = [] if records is None else [(_entry_view(entry), entry) for entry in records]
    for view, entry in sorted(views, key=lambda pair: canonical_json_bytes(pair[0])):
        path = f"record:sha256:{identity_hash(view)}"
        if type(entry) is dict and set(entry) == _BATCH_KEYS:
            batch_entries += 1
            normalization = entry["normalization"]
            if type(normalization) is dict and type(normalization.get("records")) is list:
                supplied_records += len(normalization["records"])
            else:
                uncountable_entries += 1
            if (
                not (_text(entry["arm"]) and _text(entry["protocol_hash"]))
                or type(normalization) is not dict
                or type(normalization.get("records")) is not list
                or type(normalization.get("admitted")) is not bool
                or ("issues" in normalization and type(normalization["issues"]) is not list)
            ):
                malformed_entries += 1
                issue(
                    "invalid_record",
                    path,
                    "Batch entry lacks arm, protocol, records or a boolean admitted flag.",
                )
                continue
            if normalization["admitted"] and normalization.get("issues"):
                malformed_entries += 1
                issue("invalid_record", path, "Normalization admitted despite its issues.")
                continue
            if not normalization["admitted"]:
                rejected_entries += 1
                issue(
                    "records_on_rejected_batch",
                    path,
                    "Adapter normalization did not admit this batch; its records are not used.",
                )
                continue
            for record in sorted(normalization["records"], key=canonical_json_bytes):
                flat_records.append((record, entry["arm"], entry["protocol_hash"]))
        else:
            bare_entries += 1
            supplied_records += 1
            flat_records.append((entry, None, None))
    if records is None:
        w3_acceptance = "not_supplied"
    elif malformed_entries:
        w3_acceptance = "invalid"
    elif rejected_entries:
        w3_acceptance = "rejected"
    elif bare_entries or not batch_entries:
        w3_acceptance = "not_stated"
    else:
        w3_acceptance = "admitted"
    records_section = {
        "supplied": records is not None,
        "count": None if uncountable_entries else supplied_records,
        "bound": 0,
        "out_of_envelope": 0,
    }
    if records is not None:
        usable = not (malformed_entries or rejected_entries)
        if flat_records and not admitted:
            usable = False
            issue(
                "records_on_rejected_batch",
                "records",
                "Normalized records exist for a batch W2 did not admit; they are not used.",
            )
        if not usable:
            for unit in units:
                unit["reasons"].add("normalized_records_unusable")
        else:
            by_key, by_protocol = {}, {}
            for unit in units:
                key = (unit["arm"], unit["scene_id"], unit["replicate_id"])
                by_key.setdefault(key, []).append(unit)
                by_protocol.setdefault((unit["protocol_hash"], *key), []).append(unit)
            recorded_units = set()
            for record, batch_arm, batch_protocol in flat_records:
                path = f"record:sha256:{identity_hash(record)}"
                if (
                    type(record) is not dict
                    or set(record) != _RECORD_KEYS
                    or not all(type(record[k]) is str for k in _RECORD_TEXT)
                    or type(record["replicate_id"]) is not int
                ):
                    issue("invalid_record", path, "Record differs from the adapter record shape.")
                    continue
                if batch_arm is not None and record["arm"] != batch_arm:
                    issue("invalid_record", path, "Record arm differs from its batch arm.")
                    continue
                key = (record["arm"], record["scene_id"], record["replicate_id"])
                matches = (
                    by_key.get(key)
                    if batch_protocol is None
                    else by_protocol.get((batch_protocol, *key))
                )
                if not matches:
                    issue("record_unbound", path, "Record matches no scheduled job.")
                    continue
                if len(matches) > 1:
                    issue("record_binding_ambiguous", path, "Record matches several jobs.")
                    continue
                unit = matches[0]
                if unit["position"] in recorded_units:
                    issue("duplicate_record", path, "Job has several normalized records.")
                    continue
                recorded_units.add(unit["position"])
                records_section["bound"] += 1
                if (record["road_id"], record["stratum"]) != (unit["road_id"], unit["stratum"]):
                    issue("record_observation_mismatch", path, "Road/stratum differ from job.")
                deviation = record["deviation_m"]
                if unit["deviation_m"] is None or deviation != unit["deviation_m"]:
                    issue(
                        "record_observation_mismatch",
                        path,
                        "Record deviation differs from the full-stream observation.",
                        refs=[unit["job_id"]],
                    )
                if _finite_number(deviation) and deviation >= ENVELOPE_M:
                    records_section["out_of_envelope"] += 1
            for unit in units:
                if unit["position"] not in recorded_units:
                    unit["reasons"].add("not_in_normalized_records")

    # --- A-valid x B-valid table ----------------------------------------------------
    arms = {unit["arm"] for unit in units}
    groups = {}
    for unit in units:
        key = tuple(unit[name] for name in _PAIR_KEY)
        groups.setdefault(key, []).append(unit)
    pair_rows, unpaired = [], []
    for key in sorted(groups):
        members = groups[key]
        by_arm = {}
        for unit in members:
            by_arm.setdefault(unit["arm"], []).append(unit)
        if sorted(by_arm) == ["A", "B"] and all(len(v) == 1 for v in by_arm.values()):
            pair_rows.append((key, by_arm["A"][0], by_arm["B"][0]))
        else:
            unpaired.extend(unit["job_id"] for unit in members)
    table_section = {
        "status": None,
        "code": None,
        "definition": None,
        "pairs_total": 0,
        "table": None,
        "outside_table": None,
        "unpaired_job_ids": sorted(unpaired),
        "pairing": {
            "key": list(_PAIR_KEY),
            "status": "agent_technical_default",
            "owner_convention": None,
            "note": (
                "A and B jobs are coupled by replicate index within (phase, scene, protocol); "
                "for unseeded replicates this coupling is arbitrary and the joint cells "
                "depend on it. The owner pairing/seed convention is not registered."
            ),
        },
        "pairs": [],
    }
    if validity_definition is not None:
        table_section["definition"] = {
            "definition_id": validity_definition.get("definition_id")
            if type(validity_definition) is dict
            else None,
            "sha256": definition_hash,
            "status": "supplied_not_authenticated",
        }
    if arms != {"A", "B"}:
        table_section.update(status="not_applicable", code="single_arm_schedule")
        table_section["unpaired_job_ids"] = []
    else:
        table_section["pairs_total"] = len(pair_rows)
        if validity_definition is None:
            table_section.update(status="blocked", code="validity_definition_unregistered")
        elif criteria is None:
            table_section.update(status="blocked", code="invalid_validity_definition")
        else:
            cells = dict.fromkeys(TABLE_CELLS, 0)
            outside = {"unknown_pairs": 0, "unpaired_jobs": len(unpaired)}
            for key, unit_a, unit_b in pair_rows:
                states = {"A": unit_a["validity"], "B": unit_b["validity"]}
                unknown_arms = sorted(arm for arm, state in states.items() if state == "unknown")
                if unknown_arms:
                    cell = "unknown"
                    outside["unknown_pairs"] += 1
                else:
                    cell = f"A_{states['A']}_B_{states['B']}"
                    cells[cell] += 1
                table_section["pairs"].append(
                    {
                        "phase": key[0],
                        "scene_id": key[1],
                        "replicate_id": key[2],
                        "protocol_hash": key[3],
                        "job_ids": {"A": unit_a["job_id"], "B": unit_b["job_id"]},
                        "cell": cell,
                        "unknown_arms": unknown_arms,
                    }
                )
            table_section.update(status="passed", table=cells, outside_table=outside)

    # --- aggregate counts ----------------------------------------------------------
    bins = dict.fromkeys((*DEVIATION_BINS, "unknown"), 0)
    indicator_counts = {name: {"true": 0, "false": 0, "unknown": 0} for name in INDICATORS}
    terminal_counts = {"exact_horizon": 0, "short_horizon": 0, "unknown": 0}
    for unit in units:
        bins[unit["deviation_bin"]] += 1
        terminal_counts[unit["terminal"]] += 1
        for name, value in unit["indicators"].items():
            label = "unknown" if value is None else ("true" if value else "false")
            indicator_counts[name][label] += 1
    out_of_envelope = sorted(
        unit["job_id"] for unit in units if unit["indicators"]["out_of_envelope"] is True
    )
    # Attempt level: a >= 4 m rollout of an undescribed unit (retry, multi-attempt or
    # inconsistent selection) is still listed here and warned, never silently dropped.
    out_of_envelope_attempts = sorted(
        row["attempt_id"] for row in attempt_rows if row["indicators"]["out_of_envelope"] is True
    )
    described_attempts = {unit["described_attempt_id"] for unit in units}
    undescribed_out_of_envelope = [
        attempt_id
        for attempt_id in out_of_envelope_attempts
        if attempt_id not in described_attempts
    ]
    for attempt_id in undescribed_out_of_envelope:
        issue(
            "out_of_envelope_attempt_not_described",
            f"observation:{attempt_id}",
            "A >= 4 m attempt belongs to no described unit; listed at attempt level.",
            "warning",
            refs=[attempt_id],
        )
    excluded = []
    for unit in sorted(units, key=lambda u: (u["job_id"], u["position"])):
        unit["reasons"] = sorted(unit["reasons"])
        if unit["reasons"]:
            excluded.append(
                {"job_id": unit["job_id"], "position": unit["position"], "reasons": unit["reasons"]}
            )
    selected = sum(
        unit["selected_attempt_id"] is not None
        and unit["described_attempt_id"] == unit["selected_attempt_id"]
        for unit in units
    )
    denominators = {
        "planned_jobs": recomputed["scheduled_jobs"],
        "valid_schedule_rows": len(jobs),
        "invalid_schedule_rows": max(0, recomputed["scheduled_jobs"] - len(jobs)),
        "launched_attempts": len(attempts),
        "first_attempts": sum(min(1, len(job["attempt_ids"])) for job in jobs),
        "retries": sum(unit["retry_count"] for unit in units),
        "jobs_with_retries": sum(unit["retry_count"] > 0 for unit in units),
        "completed_attempts": recomputed["completed_attempts"],
        "failed_attempts": recomputed["failed_attempts"],
        "cancelled_attempts": recomputed["cancelled_attempts"],
        "unresolved_attempts": recomputed["incomplete_attempts"],
        "jobs_without_attempt": recomputed["missing_jobs"],
        "attempts_without_scheduled_job": unknown_job_attempts,
        "bound_observations": len(bound),
        "unbound_observations": len(observations) - len(bound),
        "selected_observations": selected,
        "described_units": sum(unit["described_attempt_id"] is not None for unit in units),
        "excluded_units": len(excluded),
    }
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    errors = any(item["severity"] == "error" for item in issues)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": {"module": "gate.s2_diagnostics", "source_sha256": _source_sha256()},
        "input_hashes": {
            "ledger_report": ledger_hash,
            "observations": observations_hash,
            "records": records_hash,
            "validity_definition": definition_hash,
        },
        "units_of_measure": {
            "deviation": "m",
            "deviation_bins": "jobs (one unit per scheduled job)",
            "validity_table": (
                "cells and outside_table.unknown_pairs: A/B comparison pairs; "
                "outside_table.unpaired_jobs: jobs"
            ),
            "out_of_envelope": "count/job_ids: described jobs; attempt_*: bound attempts",
            "records": (
                "count: normalized records supplied, whether or not used (one per bare "
                "element; each batch entry's records list, admitted or not; null when a "
                "batch entry's records is not a list); bound and out_of_envelope: records "
                "joined to a scheduled job"
            ),
            "denominators": "jobs, attempts or observations as named",
        },
        "status": "blocked" if errors else "passed",
        "status_meaning": "description consistency only; never admission",
        "diagnostic_only": True,
        "gate_eligible": False,
        "admitted_verdict": None,
        "ledger_admitted": admitted,
        "upstream_acceptance": {
            "w2_ledger": "admitted" if admitted else "rejected",
            "w3_normalization": w3_acceptance,
        },
        "ledger_issues": [
            {"code": item["code"], "path": item["path"], "message": item["message"]}
            for item in ledger_report["issues"]
        ],
        "denominators": denominators,
        "deviation_bins": {"counts": bins, "denominator": len(units)},
        "out_of_envelope": {
            "threshold_m": ENVELOPE_M,
            "count": len(out_of_envelope),
            "job_ids": out_of_envelope,
            "attempt_count": len(out_of_envelope_attempts),
            "attempt_ids": out_of_envelope_attempts,
            "undescribed_attempt_ids": undescribed_out_of_envelope,
        },
        "terminal": terminal_counts,
        "indicators": indicator_counts,
        "validity_table": table_section,
        "records": records_section,
        "excluded_units": excluded,
        "units": sorted(units, key=lambda u: (u["job_id"], u["position"])),
        "observations": attempt_rows,
        "issues": issues,
    }
