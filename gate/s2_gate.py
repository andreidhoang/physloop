"""S2 registered-gate glue: adapter records -> ``gate.api`` request -> verdict (lane-r1 F8).

Serves S2.3/S2.6 [C] (G1, G3-R1). Agent-authored evaluator engineering under D-018/D-024
(build-spec A7). The statistical decision is made only by the protected path
``gate.api.parse_request`` -> ``gate.api.evaluate`` -> ``gate.verdict.verdict``; this
module never re-implements a bound, an average or a pairing rule.

Verification level per function (build-spec section 1):

- ``gate_request_from_records`` - L1: pure translation with literal known answers and
  negative controls (missing/unknown/invalid registration fields, malformed records).
- ``evaluate_registered`` - L1 over synthetic and adapter-shaped records. Its use on an
  actual capture is data-gated: ``qualify_on_host`` with admitted actual inputs and
  Huy's recorded registration. A ``passed`` status says the protected gate ran on the
  supplied inputs; it never says the registration is Huy's or that a capture qualified.

Inputs. ``records`` are the adapter-normalized records of both arms
(``gate.alpasim_adapter.normalize_summary(...)["records"]``), exactly the keys of
``gate.contracts.RolloutRecord``. ``adapter_metric`` is the same adapter output's own metric
declaration (``normalize_summary(...)["metric"]``); it is required (keyword-only, no default)
and must equal ``registration.metric`` in name, kind, higher_is_worse and unit, otherwise the
evaluation blocks with ``metric_mismatch`` (an inverted direction would otherwise turn a
NO-GO into a GO). ``registration`` is Huy's registration; every field in
``REGISTRATION_FIELDS`` is required, with no default (owner inputs: margin, alpha,
required strata, estimand, score definition, deviation convention, seed mode, metric,
reference/candidate, protocol identity and the analysis mode). An unregistered or invalid
field yields ``status: blocked`` naming the field. A supplied registration is hashed and
labelled ``supplied_not_authenticated``: the hash proves bytes, not Huy's approval.

Conventions enforced here, not chosen here:

- ``estimand`` must equal the protected gate's estimand (``VerdictResult.estimand``);
  any other estimand is not computable by the gate and blocks.
- ``seed_mode`` ``synthetic_seeded`` is accepted only under ``scope="synthetic"``; any
  other scope blocks with ``seed_mode_scope_mismatch``.
- Every record ``value`` is a finite real (never bool, never an overflowing integer) in
  [0, 1] for a continuous metric or exactly 0/1 for a binary one; otherwise
  ``record_invalid`` names it before the protected gate is called.
- ``analysis_mode`` must be ``fixed_sample``: ``gate.api`` has no sequential entry point,
  so registered interim looks are not evaluable through this glue (no look is taken).
- Out-of-envelope rollouts (full-trajectory ``deviation_m >= 4`` m) are scored by the
  protected gate exactly like every other admitted rollout and are reported separately
  (identities, per-arm and per-stratum counts, cross-checked against the gate's own
  counts). They are never dropped, never re-scored and never silently mixed in. A record
  without a full deviation blocks (``deviation_missing``): missing is never "inside".
- INSUFFICIENT and NO-GO are copied from the gate unchanged. A blocked input yields no
  verdict at all (``verdict: null``), never a substitute INSUFFICIENT.
- Superiority is not evaluated (``superiority: not_evaluated``); a non-inferiority GO is
  not an improvement claim.

Agent technical defaults (D-018), inherited by Huy unless changed: the registration shape
(``seed_mode``, ``score_definition`` and ``deviation_convention`` top-level, joined into
``gate.api``'s protocol object); ``analysis_mode`` limited to ``fixed_sample``; the estimand
equality rule; out-of-envelope handling ``scored_by_protected_gate_and_reported_separately``
with ``>=`` at 4 m; ``deviation_missing`` blocking; canonical record order (scene, replicate,
arm); the required ``adapter_metric`` keyword. Not cross-checked here: which policy each arm
is (records carry the arm, not the policy).

Interface reconciliation (recorded, not silently chosen): build-spec A7 names
``gate_request_from_records(records, registration)``; review finding A7-01 requires the
adapter's metric declaration, so both entry points take a required keyword-only
``adapter_metric``. A build-spec amendment is requested from the coordinator.

This module never reads files other than its own source (producer identity), never
performs network or subprocess calls, never selects retries or fills missing values,
never edits or copies ``gate/verdict.py``/``gate/stats/``, and never fills an owner value.
"""

import hashlib
import math
from pathlib import Path

from gate import api
from gate.contracts import PairingError
from gate.s2_identity import canonical_json_bytes, identity_hash
from gate.verdict import VerdictResult

ARTIFACT_KIND = "s2_registered_gate_evaluation"
SCHEMA_VERSION = 1
REGISTRATION_SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
ENVELOPE_M = 4.0
SEED_MODES = ("synthetic_seeded", "unseeded", "seeded")
METRIC_KINDS = ("binary", "continuous")
ANALYSIS_MODES = ("fixed_sample",)
RECORD_KEYS = (
    "arm",
    "scene_id",
    "road_id",
    "stratum",
    "replicate_id",
    "value",
    "completed",
    "deviation_m",
    "seed",
)
# Field -> requirement. Literal values are the only accepted value; None means "Huy's value".
REGISTRATION_FIELDS = {
    "schema_version": REGISTRATION_SCHEMA_VERSION,
    "registration_id": None,
    "reference": None,
    "candidate": None,
    "protocol": None,
    "seed_mode": None,
    "score_definition": None,
    "deviation_convention": None,
    "estimand": VerdictResult.estimand,
    "required_strata": None,
    "margin": None,
    "alpha": None,
    "metric": None,
    "analysis_mode": None,
}
PROTOCOL_KEYS = ("identity", "config_hash", "evaluator")
METRIC_KEYS = ("name", "kind", "higher_is_worse", "unit")


class GateInputError(ValueError):
    """Inputs cannot become a gate request; ``issues`` names every defect found."""

    def __init__(self, issues):
        super().__init__("; ".join(f"{item['path']}: {item['code']}" for item in issues))
        self.issues = issues


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _issue(code, path, message, severity="error"):
    return {
        "code": code,
        "path": path,
        "severity": severity,
        "evidence_refs": [],
        "message": message,
    }


def _text(value):
    return type(value) is str and bool(value.strip())


def _real(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_object(value, keys, path, issues):
    """Exact-key object check; returns False (with issues) when unusable."""
    if type(value) is not dict:
        issues.append(_issue("registration_field_invalid", path, "Expected a JSON object."))
        return False
    for key in sorted(set(keys) - set(value)):
        issues.append(_issue("registration_field_missing", f"{path}.{key}", "Required."))
    for key in sorted(set(value) - set(keys), key=str):
        issues.append(_issue("registration_field_unknown", f"{path}.{key}", "Not accepted."))
    return set(value) == set(keys)


def _metric_value_issues(metric, path, issues):
    """Value checks for an exact-key metric object (shared by registration and adapter)."""
    for key in ("name", "unit"):
        if not _text(metric[key]):
            issues.append(
                _issue(_invalid_code(path), f"{path}.{key}", "Expected a nonempty string.")
            )
    if metric["kind"] not in METRIC_KINDS:
        issues.append(
            _issue(_invalid_code(path), f"{path}.kind", f"Expected one of {METRIC_KINDS}.")
        )
    if type(metric["higher_is_worse"]) is not bool:
        issues.append(
            _issue(_invalid_code(path), f"{path}.higher_is_worse", "Expected a boolean.")
        )


def _invalid_code(path):
    return "adapter_metric_invalid" if path == "adapter_metric" else "registration_field_invalid"


def _usable_metric(value):
    """True when ``value`` is an exact-key metric object with valid field values."""
    issues = []
    if type(value) is not dict or set(value) != set(METRIC_KEYS):
        return False
    _metric_value_issues(value, "metric", issues)
    return not issues


def _adapter_metric_issues(adapter_metric, registration):
    """Validate the adapter's metric declaration and compare it with ``registration.metric``."""
    if type(adapter_metric) is not dict:
        return [
            _issue(
                "adapter_metric_invalid",
                "adapter_metric",
                "Expected the adapter's metric object (normalize_summary(...)['metric']).",
            )
        ]
    issues = []
    for key in sorted(set(METRIC_KEYS) - set(adapter_metric)):
        issues.append(_issue("adapter_metric_invalid", f"adapter_metric.{key}", "Required."))
    for key in sorted(set(adapter_metric) - set(METRIC_KEYS), key=str):
        issues.append(_issue("adapter_metric_invalid", f"adapter_metric.{key}", "Not accepted."))
    if issues:
        return issues
    _metric_value_issues(adapter_metric, "adapter_metric", issues)
    registered = registration.get("metric") if type(registration) is dict else None
    if issues or not _usable_metric(registered):
        return issues
    for key in METRIC_KEYS:
        if registered[key] != adapter_metric[key] or type(registered[key]) is not type(
            adapter_metric[key]
        ):
            issues.append(
                _issue(
                    "metric_mismatch",
                    f"registration.metric.{key}",
                    f"Registered {key} {registered[key]!r} differs from the adapter's "
                    f"declaration {adapter_metric[key]!r}.",
                )
            )
    return issues


def _registration_issues(registration):
    issues = []
    if type(registration) is not dict:
        return [_issue("registration_invalid", "registration", "Expected a JSON object.")]
    for field in REGISTRATION_FIELDS:
        if field not in registration:
            issues.append(
                _issue(
                    "registration_field_missing",
                    f"registration.{field}",
                    "Owner registration field is required and has no default.",
                )
            )
    for field in sorted(set(registration) - set(REGISTRATION_FIELDS), key=str):
        issues.append(
            _issue("registration_field_unknown", f"registration.{field}", "Not accepted.")
        )

    def invalid(field, message, code="registration_field_invalid"):
        issues.append(_issue(code, f"registration.{field}", message))

    get = registration.get
    if "schema_version" in registration and (
        type(get("schema_version")) is not int
        or get("schema_version") != REGISTRATION_SCHEMA_VERSION
    ):
        invalid("schema_version", "Registration schema_version must be integer 1.")
    for field in (
        "registration_id",
        "reference",
        "candidate",
        "score_definition",
        "deviation_convention",
    ):
        if field in registration and not _text(get(field)):
            invalid(field, "Expected a nonempty string.")
    if "seed_mode" in registration and get("seed_mode") not in SEED_MODES:
        invalid("seed_mode", f"Expected one of {SEED_MODES}.")
    if "estimand" in registration and get("estimand") != VerdictResult.estimand:
        invalid(
            "estimand",
            f"The protected gate computes only '{VerdictResult.estimand}'.",
            "estimand_unsupported",
        )
    if "analysis_mode" in registration and get("analysis_mode") not in ANALYSIS_MODES:
        invalid(
            "analysis_mode",
            "gate.api evaluates a fixed-sample analysis only; registered looks are not "
            "evaluable through this glue.",
            "analysis_mode_unsupported",
        )
    for field in ("margin", "alpha"):
        if field in registration:
            value = get(field)
            if not _real(value) or not 0 < value < 1:
                invalid(field, "Expected a finite number strictly between 0 and 1.")
    if "required_strata" in registration:
        strata = get("required_strata")
        if (
            type(strata) is not list
            or not strata
            or not all(_text(name) for name in strata)
            or len(set(strata)) != len(strata)
        ):
            invalid("required_strata", "Expected a nonempty list of distinct nonempty strings.")
    if "protocol" in registration and _check_object(
        get("protocol"), PROTOCOL_KEYS, "registration.protocol", issues
    ):
        for key in PROTOCOL_KEYS:
            if not _text(get("protocol")[key]):
                invalid(f"protocol.{key}", "Expected a nonempty string.")
    if "metric" in registration and _check_object(
        get("metric"), METRIC_KEYS, "registration.metric", issues
    ):
        _metric_value_issues(get("metric"), "registration.metric", issues)
    return issues


def _value_issue(value, kind):
    """Message for an invalid record value, or None. ``kind`` None checks finiteness only."""
    if not _real(value):
        return "Expected a finite real number (not bool, not an overflowing integer)."
    if kind == "continuous" and not 0 <= value <= 1:
        return "A continuous metric value must lie in [0, 1]."
    if kind == "binary" and value not in (0, 1):
        return "A binary metric value must be exactly 0 or 1."
    return None


def _record_issues(records, kind=None):
    if type(records) is not list:
        return [_issue("records_invalid", "records", "Expected a list of adapter records.")]
    issues = []
    for index, row in enumerate(records):
        path = f"records[{index}]"
        if type(row) is not dict or set(row) != set(RECORD_KEYS):
            issues.append(
                _issue("record_invalid", path, f"Expected exactly the keys {RECORD_KEYS}.")
            )
            continue
        message = _value_issue(row["value"], kind)
        if message is not None:
            issues.append(_issue("record_invalid", f"{path}.value", message))
        deviation = row["deviation_m"]
        if deviation is None:
            issues.append(
                _issue(
                    "deviation_missing",
                    f"{path}.deviation_m",
                    "A full-trajectory deviation is required to report the 4 m envelope.",
                )
            )
        elif not _real(deviation) or deviation < 0:
            issues.append(
                _issue("record_invalid", f"{path}.deviation_m", "Expected a finite value >= 0.")
            )
        try:
            canonical_json_bytes(row)
        except ValueError as exc:
            issues.append(_issue("record_invalid", path, str(exc)))
    return issues


def _record_sort_key(row):
    replicate = row["replicate_id"]
    return (
        str(row["scene_id"]),
        type(replicate).__name__,
        str(replicate),
        str(row["arm"]),
    )


def gate_request_from_records(records, registration, *, adapter_metric):
    """Return the ``gate.api`` schema-v1 request for registered records (L1).

    ``adapter_metric`` is the adapter's own ``normalize_summary(...)["metric"]``; it must
    equal ``registration.metric`` field by field. Raises ``GateInputError`` (with
    shared-shape issues) when the registration, the adapter metric or the records are
    unusable. Records are copied in a canonical order so the request does not depend on
    input order; no record is filtered, re-scored or modified.
    """
    registration_issues = _registration_issues(registration)
    metric = registration.get("metric") if type(registration) is dict else None
    kind = metric["kind"] if _usable_metric(metric) else None
    issues = (
        registration_issues
        + _adapter_metric_issues(adapter_metric, registration)
        + _record_issues(records, kind)
    )
    if issues:
        raise GateInputError(issues)
    protocol = registration["protocol"]
    return {
        "schema_version": api.SCHEMA_VERSION,
        "spec": {
            "reference": registration["reference"],
            "candidate": registration["candidate"],
            "protocol": {
                "identity": protocol["identity"],
                "seed_mode": registration["seed_mode"],
                "config_hash": protocol["config_hash"],
                "evaluator": protocol["evaluator"],
                "score_definition": registration["score_definition"],
                "deviation_convention": registration["deviation_convention"],
            },
            "required_strata": list(registration["required_strata"]),
            "margin": registration["margin"],
            "alpha": registration["alpha"],
            "metric": dict(registration["metric"]),
        },
        "records": [dict(row) for row in sorted(records, key=_record_sort_key)],
    }


def _records_hash(records):
    """Order-insensitive hash of JSON-native rows; None when rows are not hashable."""
    if type(records) is not list:
        return None
    try:
        lines = sorted(canonical_json_bytes(row) for row in records)
    except (ValueError, TypeError):
        return None
    return hashlib.sha256(b"\n".join(lines)).hexdigest()


def _registration_hash(registration):
    try:
        return identity_hash(registration)
    except (ValueError, TypeError):
        return None


def _out_of_envelope(records, required_strata, gate_response):
    """Count >= 4 m rollouts independently and cross-check the protected gate counts."""
    rows = sorted(
        (row for row in records if row["deviation_m"] >= ENVELOPE_M),
        key=_record_sort_key,
    )
    by_arm = {"A": 0, "B": 0}
    by_stratum = {name: {"A": 0, "B": 0} for name in required_strata}
    for row in rows:
        by_arm[row["arm"]] += 1
        by_stratum[row["stratum"]][row["arm"]] += 1
    details = gate_response["verdict"]["details"]
    agree = all(
        (details[name]["out_of_envelope_a"], details[name]["out_of_envelope_b"])
        == (by_stratum[name]["A"], by_stratum[name]["B"])
        for name in required_strata
    )
    return {
        "threshold_m": ENVELOPE_M,
        "comparison": "deviation_m >= threshold_m (full trajectory, from the adapter)",
        "handling": "scored_by_protected_gate_and_reported_separately",
        "count": len(rows),
        "by_arm": by_arm,
        "by_stratum": by_stratum,
        "rollouts": [
            {
                key: row[key]
                for key in ("arm", "scene_id", "road_id", "stratum", "replicate_id",
                            "deviation_m")
            }
            for row in rows
        ],
        "gate_counts_agree": agree,
    }


def _metric_hash(adapter_metric):
    try:
        return identity_hash(adapter_metric)
    except (ValueError, TypeError):
        return None


def evaluate_registered(records, registration, *, scope, adapter_metric):
    """Evaluate adapter records under a registration through ``gate.api`` (L1).

    ``scope`` is the caller's declaration from ``SCOPES``; ``adapter_metric`` is the
    adapter's own metric declaration (required, no default). Structural misuse of ``scope``
    raises ``ValueError``; every data or registration defect yields a deterministic
    ``blocked`` artifact naming the defect, with no verdict.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    issues = []
    request = gate_response = verdict = envelope = None
    if (
        scope != "synthetic"
        and type(registration) is dict
        and registration.get("seed_mode") == "synthetic_seeded"
    ):
        issues.append(
            _issue(
                "seed_mode_scope_mismatch",
                "registration.seed_mode",
                f"seed_mode 'synthetic_seeded' contradicts scope '{scope}'; it is accepted "
                "only for synthetic inputs.",
            )
        )
    try:
        candidate_request = gate_request_from_records(
            records, registration, adapter_metric=adapter_metric
        )
    except GateInputError as exc:
        issues.extend(exc.issues)
    else:
        request = candidate_request if not issues else None
    if request is not None:
        try:
            gate_response = api.evaluate(*api.parse_request(request))
        except (PairingError, ValueError, ArithmeticError) as exc:
            issues.append(_issue("gate_rejected", "records", f"{type(exc).__name__}: {exc}"))
            gate_response = None
    if gate_response is not None:
        verdict = gate_response["verdict"]["overall"]
        envelope = _out_of_envelope(
            request["records"], registration["required_strata"], gate_response
        )
        if not envelope["gate_counts_agree"]:
            issues.append(
                _issue(
                    "out_of_envelope_count_mismatch",
                    "out_of_envelope",
                    "Independent >= 4 m counts disagree with the protected gate counts.",
                )
            )
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    blocked = any(item["severity"] == "error" for item in issues)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": {"module": "gate.s2_gate", "source_sha256": _source_sha256()},
        "input_hashes": {
            "records": _records_hash(records),
            "registration": _registration_hash(registration),
            "adapter_metric": _metric_hash(adapter_metric),
        },
        "units": {
            "deviation": "m",
            "margin": "metric difference units (positive means candidate worse)",
            "counts": "rollouts (out_of_envelope) or pairs/roads as named by gate.api",
        },
        "status": "blocked" if blocked else "passed",
        "status_meaning": (
            "passed: the protected gate evaluated the supplied inputs; the verdict field "
            "carries GO/NO-GO/INSUFFICIENT unchanged. Not a qualification or admission."
        ),
        "registration_id": (
            registration.get("registration_id") if type(registration) is dict else None
        ),
        "registration_authenticity": "supplied_not_authenticated",
        "request_hash": None if request is None else identity_hash(request),
        "verdict": None if blocked else verdict,
        "gate_response": None if blocked else gate_response,
        "out_of_envelope": None if blocked else envelope,
        "superiority": "not_evaluated",
        "issues": issues,
    }
