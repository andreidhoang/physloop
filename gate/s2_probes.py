"""S2 repeatability diagnostics and probe comparators (readiness CPU-05; build-spec A6).

Serves S2.4 [C] (G3·R1, G7). Agent-authored evaluator engineering under D-018/D-024.

``describe_repeats`` is descriptive: counts, within-scene dispersion (sample and
population variance reported by name), between-scene variance of scene means,
binary replicate discordance and, on request, a paired two-policy event
contingency. Inferential reliability (an ICC variant, an interval method, a
floor, PASS/KILL) belongs to Huy's registered contract; no estimator is
qualified here, so a missing method is ``blocked: method_unregistered`` and any
registered method is ``unsupported_method`` until a reviewed implementation is
added to ``QUALIFIED_METHODS`` (empty by design). Descriptives are emitted in
every case. No Kendall W: a two-arm same-policy control is not a policy panel.

``compare_probe_artifacts`` compares two fixed-input policy or fixed-pose
renderer probe receipts whose tensors travel as safetensors payload bytes. It
checks receipt integrity, input identity, units, dtype and shape, then exact
byte equality per tensor, the first differing element, a descriptive maximum
absolute difference and nonfinite mismatches; file-level identity comes from
``gate.tensor_manifest``. Without a registered tolerance there is no
tolerance verdict (``tolerance_unregistered``); the comparator never decides
that a difference is acceptable on its own.

Verification level per function (build-spec section 1):

- ``describe_repeats`` - L1 with hand-derived rational known answers. Use on
  actual repeated rollouts is data-gated (``qualify_on_host``).
- ``compare_probe_artifacts`` - L1 comparator over hand-built payloads. CPU
  byte comparison qualifies only the comparator: fixed-input GPU behaviour,
  renderer noise and seed consumption remain ``qualify_on_host``/``unverified``;
  non-actual scopes report ``host_qualification: not_run`` for real probes.

Interface reconciliation (recorded, not silently chosen): build-spec A6 names
``describe_repeats(observations, method=None)``; readiness CPU-05 names
``describe_repeats(records, grouping)``. The build-spec form is used and the
grouping is fixed by the explicit observation axes below (policy/checkpoint,
training run, road, scene, simulator replicate), so it is not a free parameter.
CPU-05's ``compare_probe_artifacts(reference, repeat, probe_spec)`` becomes
``compare_probe_artifacts(a, b, kind)``; the per-tensor spec (semantic step,
role and units) travels in each probe receipt.

Observation contract (exact keys, fail closed): ``policy_id``, ``road_id``,
``scene_id``, ``replicate_id``, ``metric_id`` (nonempty text);
``training_run_id`` (nonempty text or ``None``); ``metric_kind``
(``continuous`` | ``binary``); ``value`` (finite int/float for continuous, ``0``,
``1`` or a boolean for binary, ``None`` when missing - never read as zero).
A cell is (metric, policy, training run, road, scene); a panel is (metric,
policy, training run). A duplicated replicate within a cell rejects every row
of that replicate; a scene seen under two roads rejects every row of the scene;
a metric seen with two kinds rejects every row of the metric.

Method registration (owner contract, exact keys, no defaults): ``schema_version``
(1), ``method_id``, ``estimand``, ``hierarchy``, ``interval_method``,
``missing_data``, ``degenerate_case``, ``source_ref`` (nonempty, non-placeholder
text) and ``floor`` (finite number). It is hashed, labelled
``supplied_not_authenticated`` and never treated as Huy's approval.

Probe receipt contract (exact keys): ``artifact_kind`` ``s2_probe_receipt``,
``schema_version`` 1, ``scope``, ``probe_kind``, ``input_sha256`` (hash of the
exact fixed input), ``payload_sha256``, ``payload_size_bytes``, ``seed_declared``
(int or ``None``) and ``tensors`` (name -> ``step`` (int >= 0, tick or frame),
``role`` and ``units`` text). Divergence order is the semantic key
``(step, role, name)``, never the payload's byte or key order.

Agent technical defaults (D-018): the paired contingency couples the two
policies on ``(metric, training run, road, scene, replicate_id)``; for unseeded
replicates this coupling is arbitrary, so every paired entry and its
discordance ratio carry status ``unverified`` with code
``pairing_convention_unregistered`` (a warning issue), never ``passed``. Maximum absolute differences are computed in float64
for F64/F32/F16/BF16, exactly for integer and BOOL dtypes, and not at all for
the F8 dtypes (byte comparison only). Differences whose element bytes differ
with at least one nonfinite side are counted as nonfinite mismatches and
excluded from the maximum.

This module never reads files, never runs subprocesses or network calls, never
chooses a reliability estimator, floor, tolerance or pairing for Huy, and never
imports protected inference.
"""

import hashlib
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

from gate import tensor_manifest
from gate.s2_identity import identity_hash

REPEATS_KIND = "s2_repeatability_diagnostics"
PROBE_RECEIPT_KIND = "s2_probe_receipt"
PROBE_COMPARISON_KIND = "s2_probe_comparison"
SCHEMA_VERSION = 1
SCOPES = tensor_manifest.SCOPES
PROBE_KINDS = ("fixed_input_policy", "fixed_pose_renderer")
METRIC_KINDS = ("continuous", "binary")
PAIRING_UNREGISTERED = "pairing_convention_unregistered"
QUALIFIED_METHODS = {}  # method_id -> implementation; empty until a reviewed estimator exists
_HEX64 = re.compile(r"[0-9a-f]{64}")
PLACEHOLDERS = {"tbd", "todo", "later", "____", "placeholder", "xxx", "n/a", "?"}

_OBS_KEYS = {
    "policy_id",
    "training_run_id",
    "road_id",
    "scene_id",
    "replicate_id",
    "metric_id",
    "metric_kind",
    "value",
}
_METHOD_TEXT = (
    "method_id",
    "estimand",
    "hierarchy",
    "interval_method",
    "missing_data",
    "degenerate_case",
    "source_ref",
)
_METHOD_KEYS = {"schema_version", "floor", *_METHOD_TEXT}
_RECEIPT_KEYS = {
    "artifact_kind",
    "schema_version",
    "scope",
    "probe_kind",
    "input_sha256",
    "payload_sha256",
    "payload_size_bytes",
    "seed_declared",
    "tensors",
}
_TOLERANCE_KEYS = {"schema_version", "tolerance_id", "max_abs", "source_ref"}
_NUMPY_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "u1",
}
_INTEGER_DTYPES = {"I64", "I32", "I16", "I8", "U8", "BOOL"}
OVERFLOW_REASON = "difference_overflows_float64"


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
    return type(value) is str and bool(value.strip()) and "\x00" not in value


def _owner_text(value):
    return _text(value) and value.strip().casefold() not in PLACEHOLDERS


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _quantity(value=None, reason=None, status=None):
    """Exact rational quantity: ``{status, value (float), exact (str), reason}``."""
    if value is None:
        return {"status": status or "not_estimable", "value": None, "exact": None, "reason": reason}
    return {"status": "passed", "value": float(value), "exact": str(value), "reason": None}


def _mean(values):
    return sum(values, Fraction(0)) / len(values)


def _sum_squares(values):
    centre = _mean(values)
    return sum(((v - centre) ** 2 for v in values), Fraction(0))


def _supplied_hash(rows):
    """Order-invariant hash of every supplied row, accepted or not.

    Each row contributes its canonical-JSON identity hash; a row that is not
    JSON-native contributes a marker naming its position (its content cannot be
    hashed canonically). Returns ``(hash, number of non-JSON rows)``.
    """
    digests, non_json = [], 0
    for position, row in enumerate(rows):
        try:
            digests.append(identity_hash(row))
        except (ValueError, TypeError):
            non_json += 1
            digests.append(identity_hash({"non_json_row_position": position}))
    return identity_hash(sorted(digests)), non_json


def _sample_variance(values):
    if len(values) < 2:
        return None
    return _sum_squares(values) / (len(values) - 1)


# --- describe_repeats ----------------------------------------------------------


def _check_observation(row):
    if type(row) is not dict or set(row) != _OBS_KEYS:
        return "observation keys must be exactly " + ", ".join(sorted(_OBS_KEYS))
    for key in ("policy_id", "road_id", "scene_id", "replicate_id", "metric_id"):
        if not _text(row[key]):
            return f"{key} must be nonempty text"
    if row["training_run_id"] is not None and not _text(row["training_run_id"]):
        return "training_run_id must be nonempty text or None"
    kind, value = row["metric_kind"], row["value"]
    if kind not in METRIC_KINDS:
        return f"metric_kind must be one of {METRIC_KINDS}"
    if value is None:
        return None
    if kind == "binary":
        if type(value) is bool or (type(value) is int and value in (0, 1)):
            return None
        return "binary values must be 0, 1 or a boolean"
    if type(value) is bool or not _finite(value):
        return "continuous values must be finite numbers"
    return None


def _check_method(method):
    if type(method) is not dict or set(method) != _METHOD_KEYS:
        return "registration keys must be exactly " + ", ".join(sorted(_METHOD_KEYS))
    if type(method["schema_version"]) is not int or method["schema_version"] != 1:
        return "registration schema_version must be 1"
    for key in _METHOD_TEXT:
        if not _owner_text(method[key]):
            return f"{key} must be nonempty, non-placeholder text"
    if type(method["floor"]) is bool or not _finite(method["floor"]):
        return "floor must be a finite number"
    return None


def _method_section(method, issues):
    if method is None:
        issues.append(
            _issue(
                "method_unregistered",
                "method",
                "No registered reliability method; descriptive diagnostics only",
            )
        )
        return {
            "status": "blocked",
            "code": "method_unregistered",
            "method_id": None,
            "registration_sha256": None,
            "authentication": None,
            "result": None,
        }
    problem = _check_method(method)
    if problem:
        issues.append(_issue("method_registration_invalid", "method", problem))
        return {
            "status": "blocked",
            "code": "method_registration_invalid",
            "method_id": None,
            "registration_sha256": None,
            "authentication": None,
            "result": None,
        }
    digest = identity_hash(method)
    if method["method_id"] not in QUALIFIED_METHODS:
        issues.append(
            _issue(
                "unsupported_method",
                "method",
                f"Method {method['method_id']!r} has no qualified implementation",
            )
        )
        return {
            "status": "blocked",
            "code": "unsupported_method",
            "method_id": method["method_id"],
            "registration_sha256": digest,
            "authentication": "supplied_not_authenticated",
            "result": None,
        }
    raise NotImplementedError("QUALIFIED_METHODS must stay empty until a reviewed estimator exists")


def describe_repeats(observations, method=None, *, scope, paired_policies=None):
    """Describe within-scene repeatability without choosing an inferential method (L1).

    ``paired_policies=(a, b)`` adds a paired event contingency for every binary
    metric observed under both policies. Overall ``status`` is ``blocked`` until a
    qualified registered method exists; ``descriptive_status`` reports whether the
    observations themselves were well formed.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if type(observations) is not list:
        raise ValueError("observations must be a list")
    if paired_policies is not None and (
        type(paired_policies) not in (tuple, list)
        or len(paired_policies) != 2
        or not all(_text(p) for p in paired_policies)
        or paired_policies[0] == paired_policies[1]
    ):
        raise ValueError("paired_policies must be two distinct policy IDs")

    issues = []
    valid = []
    for position, row in enumerate(observations):
        problem = _check_observation(row)
        if problem:
            issues.append(_issue("observation_invalid", f"observations/{position}", problem))
        else:
            valid.append(row)
    rejected = len(observations) - len(valid)

    def drop(predicate, code, path, message):
        nonlocal valid, rejected
        kept = [row for row in valid if not predicate(row)]
        if len(kept) != len(valid):
            issues.append(_issue(code, path, message))
            rejected += len(valid) - len(kept)
            valid = kept

    kinds, roads = {}, {}
    for row in valid:
        kinds.setdefault(row["metric_id"], set()).add(row["metric_kind"])
        roads.setdefault(row["scene_id"], set()).add(row["road_id"])
    for metric in sorted(m for m, k in kinds.items() if len(k) > 1):
        drop(lambda r, m=metric: r["metric_id"] == m, "metric_kind_conflict",
             f"metrics/{metric}", "Metric observed with more than one kind")
    for scene in sorted(s for s, r in roads.items() if len(r) > 1):
        drop(lambda r, s=scene: r["scene_id"] == s, "scene_road_conflict",
             f"scenes/{scene}", "Scene observed under more than one road")

    def cell_key(row):
        return (row["metric_id"], row["policy_id"], row["training_run_id"] or "",
                row["road_id"], row["scene_id"])

    seen = {}
    for row in valid:
        key = (*cell_key(row), row["replicate_id"])
        seen[key] = seen.get(key, 0) + 1
    for key in sorted(k for k, n in seen.items() if n > 1):
        drop(lambda r, k=key: (*cell_key(r), r["replicate_id"]) == k, "duplicate_replicate",
             "cells/" + "/".join(key), "Replicate observed more than once in one cell")

    grouped = {}
    for row in valid:
        grouped.setdefault(cell_key(row), []).append(row)
    cells = []
    panels = {}
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda r: r["replicate_id"])
        metric, policy, run, road, scene = key
        kind = rows[0]["metric_kind"]
        values = [Fraction(int(r["value"]) if kind == "binary" else r["value"])
                  for r in rows if r["value"] is not None]
        missing = [r["replicate_id"] for r in rows if r["value"] is None]
        cell = {
            "metric_id": metric,
            "metric_kind": kind,
            "policy_id": policy,
            "training_run_id": run or None,
            "road_id": road,
            "scene_id": scene,
            "replicates": [r["replicate_id"] for r in rows],
            "n_observed": len(values),
            "n_missing": len(missing),
            "missing_replicates": missing,
            "mean": _quantity(_mean(values)) if values else _quantity(reason="no_observed_values"),
            "sample_variance": _quantity(_sample_variance(values))
            if len(values) >= 2
            else _quantity(reason="fewer_than_two_observed_replicates"),
            "population_variance": _quantity(_sum_squares(values) / len(values))
            if values
            else _quantity(reason="no_observed_values"),
        }
        if kind == "binary":
            events = sum(1 for v in values if v == 1)
            cell.update(
                events=events,
                non_events=len(values) - events,
                replicate_pairs=len(values) * (len(values) - 1) // 2,
                discordant_replicate_pairs=events * (len(values) - events),
            )
        cells.append(cell)
        panels.setdefault((metric, policy, run), []).append((cell, values))

    panel_rows = [_panel(key, entries) for key, entries in sorted(panels.items())]
    paired = []
    if paired_policies is not None:
        paired = _paired(valid, tuple(paired_policies), issues)

    method_section = _method_section(method, issues)
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    input_errors = any(
        item["severity"] == "error" and item["path"] != "method" for item in issues
    )
    observations_hash, non_json = _supplied_hash(observations)
    accepted_hash, _ = _supplied_hash(valid)
    return {
        "artifact_kind": REPEATS_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": {"module": "gate.s2_probes", "source_sha256": _source_sha256()},
        "input_hashes": {
            "observations": observations_hash,
            "accepted_observations": accepted_hash,
            "method": method_section["registration_sha256"],
        },
        "units": {
            "values": "metric units as supplied (binary: event indicator 0/1)",
            "variances": "squared metric units",
            "counts": "observations, replicates, scenes or replicate pairs as named",
        },
        "status": "blocked",
        "status_meaning": "inferential reliability is blocked until a qualified registered "
        "method exists; descriptives are never an admission",
        "descriptive_status": "blocked" if input_errors else "passed",
        "method": method_section,
        "counts": {
            "observations": len(observations),
            "accepted_observations": len(valid),
            "rejected_observations": rejected,
            "non_json_observations": non_json,
            "cells": len(cells),
        },
        "cells": cells,
        "panels": panel_rows,
        "paired": paired,
        "host_qualification": "qualify_on_host" if scope == "actual" else "not_run",
        "issues": issues,
    }


def _panel(key, entries):
    metric, policy, run = key
    kind = entries[0][0]["metric_kind"]
    within = [_sample_variance(values) for _, values in entries if len(values) >= 2]
    means = [_mean(values) for _, values in entries if values]
    between = _sample_variance(means) if len(means) >= 2 else None
    if between is None:
        reliability = {"status": "not_estimable", "reason": "fewer_than_two_scenes", "value": None}
    elif between == 0:
        reliability = {
            "status": "not_estimable",
            "reason": "zero_between_scene_variance",
            "value": None,
        }
    else:
        reliability = {"status": "blocked", "reason": "method_unregistered", "value": None}
    panel = {
        "metric_id": metric,
        "metric_kind": kind,
        "policy_id": policy,
        "training_run_id": run or None,
        "n_scenes": len(entries),
        "n_roads": len({cell["road_id"] for cell, _ in entries}),
        "n_observed": sum(len(values) for _, values in entries),
        "n_missing": sum(cell["n_missing"] for cell, _ in entries),
        "cells_with_two_or_more": len(within),
        "within_scene_sample_variance": {
            "cells_estimable": len(within),
            "min": _quantity(min(within)) if within else _quantity(reason="no_estimable_cell"),
            "max": _quantity(max(within)) if within else _quantity(reason="no_estimable_cell"),
            "unweighted_mean": _quantity(_mean(within))
            if within
            else _quantity(reason="no_estimable_cell"),
            "label": "descriptive unweighted mean of cell sample variances; not a pooled "
            "estimator",
        },
        "between_scene_sample_variance_of_means": _quantity(between)
        if between is not None
        else _quantity(reason="fewer_than_two_scenes"),
        "reliability_coefficients": reliability,
    }
    if kind == "binary":
        pairs = sum(cell["replicate_pairs"] for cell, _ in entries)
        discordant = sum(cell["discordant_replicate_pairs"] for cell, _ in entries)
        panel.update(
            scenes_discordant=sum(
                1 for cell, _ in entries if cell["events"] and cell["non_events"]
            ),
            replicate_pairs=pairs,
            discordant_replicate_pairs=discordant,
            discordant_pair_share=_quantity(Fraction(discordant, pairs))
            if pairs
            else _quantity(reason="no_replicate_pairs"),
        )
    return panel


def _paired(rows, policies, issues):
    policy_a, policy_b = policies
    present = {row["policy_id"] for row in rows}
    for policy in policies:
        if policy not in present:
            issues.append(
                _issue("paired_policy_absent", f"paired/{policy}", "Paired policy has no rows")
            )
    by_metric = {}
    for row in rows:
        if row["metric_kind"] != "binary" or row["policy_id"] not in policies:
            continue
        key = (row["training_run_id"] or "", row["road_id"], row["scene_id"], row["replicate_id"])
        by_metric.setdefault(row["metric_id"], {}).setdefault(key, {})[row["policy_id"]] = row
    result = []
    for metric in sorted(by_metric):
        table = {"a1_b1": 0, "a1_b0": 0, "a0_b1": 0, "a0_b0": 0}
        unpaired = {"a_only": 0, "b_only": 0, "missing": 0}
        for key in sorted(by_metric[metric]):
            sides = by_metric[metric][key]
            if policy_b not in sides:
                unpaired["a_only"] += 1
                continue
            if policy_a not in sides:
                unpaired["b_only"] += 1
                continue
            a, b = sides[policy_a]["value"], sides[policy_b]["value"]
            if a is None or b is None:
                unpaired["missing"] += 1
                continue
            table[f"a{int(a)}_b{int(b)}"] += 1
        discordant = table["a1_b0"] + table["a0_b1"]
        if discordant:
            ratio = _quantity(Fraction(table["a1_b0"], discordant))
            ratio.update(status="unverified", reason=PAIRING_UNREGISTERED)
        else:
            ratio = _quantity(reason="zero_total_discordance")
        result.append(
            {
                "metric_id": metric,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "status": "unverified",
                "code": PAIRING_UNREGISTERED,
                "pairing_convention": "(training run, road, scene, replicate_id); agent "
                "technical default (D-018), arbitrary for unseeded replicates, owner pairing "
                "convention unregistered",
                "table": table,
                "pairs": sum(table.values()),
                "unpaired": unpaired,
                "discordant_pairs": discordant,
                "discordance_ratio_a1_b0": ratio,
            }
        )
    if result:
        issues.append(
            _issue(
                PAIRING_UNREGISTERED,
                "paired",
                "Paired quantities couple replicates by an agent default; no owner pairing "
                "convention is registered, so they are unverified",
                severity="warning",
            )
        )
    return result


# --- compare_probe_artifacts ----------------------------------------------------


def _check_receipt(receipt):
    if type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS:
        return "receipt keys must be exactly " + ", ".join(sorted(_RECEIPT_KEYS))
    if receipt["artifact_kind"] != PROBE_RECEIPT_KIND:
        return f"artifact_kind must be {PROBE_RECEIPT_KIND}"
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        return "schema_version must be 1"
    if receipt["scope"] not in SCOPES:
        return f"scope must be one of {SCOPES}"
    if receipt["probe_kind"] not in PROBE_KINDS:
        return f"probe_kind must be one of {PROBE_KINDS}"
    for key in ("input_sha256", "payload_sha256"):
        value = receipt[key]
        if type(value) is not str or _HEX64.fullmatch(value) is None:
            return f"{key} must be 64 lowercase hexadecimal characters"
    if type(receipt["payload_size_bytes"]) is not int or receipt["payload_size_bytes"] < 0:
        return "payload_size_bytes must be a non-negative integer"
    seed = receipt["seed_declared"]
    if seed is not None and type(seed) is not int:
        return "seed_declared must be an integer or None"
    tensors = receipt["tensors"]
    if type(tensors) is not dict or not tensors:
        return "tensors must be a nonempty object"
    keys = set()
    for name, spec in tensors.items():
        if (
            not _text(name)
            or type(spec) is not dict
            or set(spec) != {"step", "role", "units"}
            or type(spec["step"]) is not int
            or spec["step"] < 0
            or not _text(spec["role"])
            or not _text(spec["units"])
        ):
            return f"tensor {name!r} needs step (int >= 0), role and units text"
        if (spec["step"], spec["role"]) in keys:
            return f"tensor {name!r} repeats a (step, role) key"
        keys.add((spec["step"], spec["role"]))
    return None


def _check_tolerance(tolerance):
    if type(tolerance) is not dict or set(tolerance) != _TOLERANCE_KEYS:
        return "tolerance keys must be exactly " + ", ".join(sorted(_TOLERANCE_KEYS))
    if type(tolerance["schema_version"]) is not int or tolerance["schema_version"] != 1:
        return "tolerance schema_version must be 1"
    for key in ("tolerance_id", "source_ref"):
        if not _owner_text(tolerance[key]):
            return f"{key} must be nonempty, non-placeholder text"
    value = tolerance["max_abs"]
    if type(value) is bool or not _finite(value) or value < 0:
        return "max_abs must be a finite non-negative number"
    return None


def _decode(dtype, raw):
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).astype(np.float64)
    array = np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype])
    if dtype in _INTEGER_DTYPES:
        return array
    return array.astype(np.float64)


def _element_comparison(dtype, shape, raw_a, raw_b):
    """Per-element comparison of equal-dtype, equal-shape raw tensors."""
    itemsize = tensor_manifest.DTYPE_ITEMSIZE[dtype]
    rows_a = np.frombuffer(raw_a, dtype=np.uint8).reshape(-1, itemsize)
    rows_b = np.frombuffer(raw_b, dtype=np.uint8).reshape(-1, itemsize)
    differs = np.any(rows_a != rows_b, axis=1)
    count = int(differs.sum())
    first = None
    if count:
        flat = int(np.argmax(differs))
        first = [int(i) for i in np.unravel_index(flat, shape)] if shape else []
    if dtype not in _NUMPY_DTYPES and dtype != "BF16":
        return count, first, {"status": "not_estimable", "value": None,
                              "reason": "dtype_not_decoded"}, None
    xa, xb = _decode(dtype, raw_a), _decode(dtype, raw_b)
    if dtype in _INTEGER_DTYPES:
        if dtype == "I64":  # exact Python integers: int64 subtraction could overflow
            pairs = zip(xa[differs].tolist(), xb[differs].tolist(), strict=True)
            value = max((abs(x - y) for x, y in pairs), default=0)
        else:
            delta = np.abs(xa[differs].astype(np.int64) - xb[differs].astype(np.int64))
            value = int(delta.max()) if delta.size else 0
        return count, first, {"status": "passed", "value": value, "reason": None}, 0
    finite = np.isfinite(xa) & np.isfinite(xb)
    nonfinite_mismatch = int((differs & ~finite).sum())
    selected = differs & finite
    if not selected.any():
        return count, first, {"status": "passed", "value": 0.0, "reason": None}, nonfinite_mismatch
    # Only F64 can overflow here (F32/F16/BF16 differences fit in float64). An
    # overflow is detected explicitly, never reported as inf or left as a warning.
    with np.errstate(over="ignore"):
        value = float(np.max(np.abs(xa[selected] - xb[selected])))
    if not math.isfinite(value):
        return count, first, {"status": "not_estimable", "value": None,
                              "reason": OVERFLOW_REASON}, nonfinite_mismatch
    return count, first, {"status": "passed", "value": value, "reason": None}, nonfinite_mismatch


def compare_probe_artifacts(a, b, kind, *, tolerance=None):
    """Compare two probe artifacts ``{"receipt": ..., "payload": bytes}`` (L1).

    ``outcome`` is ``identical`` (every tensor byte-identical with equal dtype,
    shape, units and semantic key), ``different`` or ``not_comparable`` (status
    ``blocked``: invalid receipt or payload, kind/scope mismatch, or different
    fixed inputs - the same seed on different inputs is not a determinism test).
    ``status`` is ``blocked`` whenever any error-severity issue exists; an
    invalid tolerance registration blocks the report while ``outcome`` and the
    per-tensor rows are still reported (the tolerance verdict stays ``None``).
    """
    if kind not in PROBE_KINDS:
        raise ValueError(f"kind must be one of {PROBE_KINDS}")
    issues = []
    manifests = {}
    receipts = {}
    input_hashes = {}
    for label, artifact in (("a", a), ("b", b)):
        input_hashes[label] = {"receipt": None, "payload": None}
        if type(artifact) is not dict or set(artifact) != {"receipt", "payload"}:
            issues.append(_issue("receipt_invalid", label, "Artifact needs receipt and payload"))
            continue
        receipt, body = artifact["receipt"], artifact["payload"]
        problem = _check_receipt(receipt)
        if problem or type(body) is not bytes:
            issues.append(_issue("receipt_invalid", label, problem or "payload must be bytes"))
            continue
        receipts[label] = receipt
        input_hashes[label] = {
            "receipt": identity_hash(receipt),
            "payload": hashlib.sha256(body).hexdigest(),
        }
        if receipt["probe_kind"] != kind:
            issues.append(_issue("probe_kind_mismatch", label, "Receipt is for another probe"))
        if (
            len(body) != receipt["payload_size_bytes"]
            or input_hashes[label]["payload"] != receipt["payload_sha256"]
        ):
            issues.append(_issue("payload_hash_mismatch", label, "Payload bytes differ from receipt"))
            continue
        manifest = tensor_manifest.build_manifest(
            {"payload.safetensors": body}, scope=receipt["scope"]
        )
        if manifest["status"] != "passed":
            issues.append(_issue("payload_invalid", label, "Payload is not valid safetensors"))
            continue
        if set(manifest["tensors"]) != set(receipt["tensors"]):
            issues.append(
                _issue("receipt_payload_mismatch", label, "Receipt and payload tensor names differ")
            )
            continue
        manifests[label] = (manifest, body, tensor_manifest.parse_safetensors(body))
    if len(receipts) == 2:
        if receipts["a"]["scope"] != receipts["b"]["scope"]:
            issues.append(_issue("scope_mismatch", "scope", "Receipts come from different scopes"))
        if receipts["a"]["input_sha256"] != receipts["b"]["input_sha256"]:
            issues.append(
                _issue(
                    "inputs_differ",
                    "input_sha256",
                    "Different fixed inputs: not a repeat or determinism comparison",
                )
            )

    tolerance_section = {"status": "blocked", "code": "tolerance_unregistered", "verdict": None}
    if tolerance is not None:
        problem = _check_tolerance(tolerance)
        if problem:
            tolerance_section = {
                "status": "blocked",
                "code": "tolerance_registration_invalid",
                "verdict": None,
            }
            issues.append(_issue("tolerance_registration_invalid", "tolerance", problem))
    report = {
        "artifact_kind": PROBE_COMPARISON_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": receipts["a"]["scope"] if "a" in receipts else None,
        "producer": {"module": "gate.s2_probes", "source_sha256": _source_sha256()},
        "input_hashes": input_hashes,
        "units": {
            "max_abs_diff": "tensor units as declared in the receipts",
            "differing_elements": "elements",
            "first_differing_element": "row-major index per dimension",
        },
        "probe_kind": kind,
        "status": "blocked",
        "outcome": "not_comparable",
        "tensors": [],
        "first_divergence": None,
        "file_level": None,
        "tolerance": tolerance_section,
        "seed": {
            "declared_a": receipts["a"]["seed_declared"] if "a" in receipts else None,
            "declared_b": receipts["b"]["seed_declared"] if "b" in receipts else None,
            "consumption": "unverified",
        },
        "host_qualification": "qualify_on_host"
        if "a" in receipts and receipts["a"]["scope"] == "actual"
        else "not_run",
        "issues": issues,
    }
    comparable_errors = [i for i in issues if i["path"] != "tolerance"]
    if comparable_errors or len(manifests) != 2:
        if tolerance is not None and tolerance_section["code"] != "tolerance_registration_invalid":
            report["tolerance"] = {
                "status": "not_run",
                "code": "comparison_not_performed",
                "verdict": None,
            }
        issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
        return report

    (ma, body_a, parsed_a), (mb, body_b, parsed_b) = manifests["a"], manifests["b"]
    spec_a, spec_b = receipts["a"]["tensors"], receipts["b"]["tensors"]
    file_level = tensor_manifest.compare_manifests(ma, mb)
    rows = []
    max_values = []
    for name in sorted(set(spec_a) | set(spec_b)):
        spec = spec_a.get(name) or spec_b[name]
        row = {
            "name": name,
            "step": spec["step"],
            "role": spec["role"],
            "units_a": spec_a[name]["units"] if name in spec_a else None,
            "units_b": spec_b[name]["units"] if name in spec_b else None,
            "dtype_a": ma["tensors"][name]["dtype"] if name in spec_a else None,
            "dtype_b": mb["tensors"][name]["dtype"] if name in spec_b else None,
            "shape_a": ma["tensors"][name]["shape"] if name in spec_a else None,
            "shape_b": mb["tensors"][name]["shape"] if name in spec_b else None,
            "relation": None,
            "differing_elements": None,
            "first_differing_element": None,
            "max_abs_diff": {"status": "not_estimable", "value": None,
                             "reason": "not_element_comparable"},
            "nonfinite_mismatch_count": None,
        }
        rows.append(row)
        if name not in spec_b:
            row["relation"] = "only_in_a"
            continue
        if name not in spec_a:
            row["relation"] = "only_in_b"
            continue
        ta, tb = ma["tensors"][name], mb["tensors"][name]
        if (spec_a[name]["step"], spec_a[name]["role"]) != (
            spec_b[name]["step"],
            spec_b[name]["role"],
        ):
            row["relation"] = "order_key_mismatch"
        elif row["units_a"] != row["units_b"]:
            row["relation"] = "units_mismatch"
        elif ta["dtype"] != tb["dtype"]:
            row["relation"] = "dtype_mismatch"
        elif ta["shape"] != tb["shape"]:
            row["relation"] = "shape_mismatch"
        if row["relation"] not in (None, "units_mismatch"):
            continue
        raw_a = _tensor_bytes(parsed_a, body_a, name)
        raw_b = _tensor_bytes(parsed_b, body_b, name)
        count, first, max_abs, nonfinite = _element_comparison(
            ta["dtype"], ta["shape"], raw_a, raw_b
        )
        row.update(
            differing_elements=count,
            first_differing_element=first,
            max_abs_diff=max_abs,
            nonfinite_mismatch_count=nonfinite,
        )
        if row["relation"] is None:
            row["relation"] = "identical" if ta["sha256"] == tb["sha256"] else "bytes_differ"
        if row["relation"] == "bytes_differ":
            max_values.append((max_abs, nonfinite))
    rows.sort(key=lambda r: (r["step"], r["role"], r["name"]))
    divergent = [r for r in rows if r["relation"] != "identical"]
    first_divergence = None
    if divergent:
        head = divergent[0]
        first_divergence = {
            "name": head["name"],
            "step": head["step"],
            "role": head["role"],
            "relation": head["relation"],
            "element": head["first_differing_element"],
        }
    outcome = "different" if divergent else "identical"
    if tolerance is not None and tolerance_section["code"] != "tolerance_registration_invalid":
        tolerance_section = _tolerance_verdict(tolerance, rows, max_values, outcome)
    issues.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    blocked = any(item["severity"] == "error" for item in issues)
    report.update(
        status="blocked" if blocked else "passed",
        outcome=outcome,
        tensors=rows,
        first_divergence=first_divergence,
        file_level={
            "relation": file_level["relation"],
            "files_identical": file_level["files_identical"],
            "tensors_identical": file_level["tensors_identical"],
        },
        tolerance=tolerance_section,
    )
    return report


def _tensor_bytes(parsed, body, name):
    begin, end = parsed["tensors"][name]["data_offsets"]
    base = 8 + parsed["header_bytes"]
    return body[base + begin : base + end]


def _tolerance_verdict(tolerance, rows, max_values, outcome):
    section = {
        "status": "passed",
        "code": None,
        "tolerance_id": tolerance["tolerance_id"],
        "registration_sha256": identity_hash(tolerance),
        "authentication": "supplied_not_authenticated",
        "verdict": None,
    }
    if outcome == "identical":
        section["verdict"] = "identical"
        return section
    structural = any(r["relation"] not in ("identical", "bytes_differ") for r in rows)
    # An F64 difference overflows only when its exact magnitude reaches the float64
    # overflow threshold; a registered max_abs is accepted only when it converts
    # to a finite float, i.e. lies strictly below that threshold: it exceeds.
    overflow = any(m["reason"] == OVERFLOW_REASON for m, _ in max_values)
    undecoded = any(m["status"] != "passed" for m, _ in max_values)
    nonfinite = any(n for _, n in max_values)
    if structural or nonfinite or overflow:
        section["verdict"] = "exceeds"
    elif undecoded:
        section.update(status="not_estimable", verdict=None, code="dtype_not_decoded")
    elif all(m["value"] <= tolerance["max_abs"] for m, _ in max_values):
        section["verdict"] = "within_registered_tolerance"
    else:
        section["verdict"] = "exceeds"
    return section
