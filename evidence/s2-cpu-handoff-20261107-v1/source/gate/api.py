"""Agent-authored public admission boundary for the fixed-sample S1 gate.

JSON requests declare raw A/B arms and protocol metadata. Parsing never guesses
missing identities or statistical choices. Admission and pairing precede inference.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, fields

from gate.contracts import ComparisonSpec, MetricSpec, Protocol, RolloutRecord, pair_records
from gate.verdict import verdict

SCHEMA_VERSION = 1


def _object(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return dict(value)


def _construct(contract, data, label):
    try:
        return contract(**_object(data, label))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _parse_request(payload: dict) -> tuple[list[RolloutRecord], ComparisonSpec, Protocol | None]:
    """Parse schema version 1 without silently accepting unknown fields.

    Returns ``(records, spec, candidate_protocol)``. An omitted/null candidate protocol
    declares the same protocol as the reference; it is not a compatibility inference.
    """
    request = _object(payload, "request")
    permitted = {"schema_version", "spec", "records", "candidate_protocol"}
    unknown = set(request) - permitted
    if unknown:
        raise ValueError(f"unknown request fields: {sorted(unknown)}")
    version = request.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError("schema_version must be integer 1")
    missing = {"spec", "records"} - set(request)
    if missing:
        raise ValueError(f"missing request fields: {sorted(missing)}")
    raw_spec = _object(request["spec"], "spec")
    try:
        raw_spec["protocol"] = _construct(Protocol, raw_spec["protocol"], "spec.protocol")
        raw_spec["metric"] = _construct(MetricSpec, raw_spec["metric"], "spec.metric")
        strata = raw_spec["required_strata"]
    except KeyError as exc:
        raise ValueError(f"missing spec field: {exc.args[0]}") from exc
    if not isinstance(strata, list):
        raise TypeError("required_strata must be a JSON array")
    raw_spec["required_strata"] = tuple(strata)
    spec = _construct(ComparisonSpec, raw_spec, "spec")
    if not isinstance(request["records"], list):
        raise TypeError("records must be a JSON array")
    records = [
        _construct(RolloutRecord, raw_record, f"records[{i}]")
        for i, raw_record in enumerate(request["records"])
    ]
    candidate = request.get("candidate_protocol")
    candidate_protocol = (
        None if candidate is None else _construct(Protocol, candidate, "candidate_protocol")
    )
    return records, spec, candidate_protocol


def parse_request(payload: dict) -> tuple[list[RolloutRecord], ComparisonSpec, Protocol | None]:
    """Parse JSON version 1 into (records, spec, candidate_protocol), or raise ValueError.

    Type errors from JSON structure and dataclass construction are normalized at this
    boundary so command-line consumers have one invalid-request exception to handle.
    """
    try:
        return _parse_request(payload)
    except TypeError as exc:
        raise ValueError(f"invalid request: {exc}") from exc


def evaluate(
    records: Iterable[RolloutRecord],
    spec: ComparisonSpec,
    candidate_protocol: Protocol | None = None,
) -> dict:
    """Validate raw arms, pair them, and return a versioned JSON-native verdict.

    The result contains admission counts and the declared comparison, but no raw rows.
    Binary values are normalized to failure direction once by ``pair_records``;
    continuous [0,1] values retain their original direction until the statistical core.
    """
    paired = pair_records(records, spec, candidate_protocol=candidate_protocol)
    rows = []
    for paired_row in paired.rows:
        row = dict(paired_row)
        if spec.metric.kind == "binary":
            del row["a_value"]
            del row["b_value"]
        elif not all(0 <= row[name] <= 1 for name in ("a_value", "b_value")):
            raise ValueError("continuous outcomes must be in [0,1]")
        rows.append(row)
    result = verdict(
        rows,
        margin=spec.margin,
        alpha=spec.alpha,
        required_strata=spec.required_strata,
        higher_is_worse=spec.metric.higher_is_worse,
    )
    response = {
        "schema_version": SCHEMA_VERSION,
        "analysis_mode": "fixed_sample",
        "comparison": asdict(spec),
        "pairing": {
            field.name: getattr(paired, field.name)
            for field in fields(paired)
            if field.name != "rows"
        },
        "verdict": asdict(result),
        "difference_convention": "positive means candidate worse",
        "interval_convention": "each endpoint individually one-sided; not a joint interval",
    }
    # Canonical JSON-native containers make the returned object round-trip unchanged.
    return json.loads(json.dumps(response, allow_nan=False))
