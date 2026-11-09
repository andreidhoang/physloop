"""Numerical parity ladder for S4 checkpoints, exports and actions (A10; E02, E04, E16).

Serves S4 engineering preparation (S4.1-S4.4, S4.6 [C]; G9, G7) under D-024. Agent-authored
evaluator engineering (D-018). The ladder follows protocol-design section 4 ("a single loose
``allclose`` is inadequate") and stops at the first failing rung:

1. ``structure`` - key sets, dtypes and shapes must match; any mismatch is verdict
   ``exceeds`` with status ``blocked`` (an error issue never leaves ``passed``).
2. ``bitwise`` - rule-11 identity per tensor (build-spec section 2 rule 11); all identical is
   ``identical`` and no numeric rung runs.
3. ``elementwise`` - for differing tensors only: per-element absolute and relative error and
   dtype-aware ULP distance (IEEE bit patterns in sign-magnitude order; ``+0``/``-0`` are
   0 ULP apart). Integer and boolean tensors admit no tolerance. An error that overflows
   float64 is reported as ``None`` with an overflow count (missing is never zero) and
   always exceeds a registered tolerance.
4. ``action_level`` - trajectory comparison (``compare_trajectories``); its real inputs come
   from a host (L3), so the tensor ladder always reports it as ``qualify_on_host``.

Report status (build-spec section 2): ``blocked`` when any error issue exists (structure
mismatch, bytes changed since the manifest); ``qualify_on_host`` when the elementwise rung is
deferred (no ``load``); ``passed`` otherwise. The verdict is data beside the status.

Cross-dtype mode (opt-in, ``cross_dtype="reference_cast_to_candidate"``; agent technical
default, D-018): a reference/candidate dtype difference between two float dtypes is then a
declared cast (``structure.dtype_cast``), not a structure failure. The reference tensor is
rounded to the candidate's stored dtype with ``Tensor.to`` (round-to-nearest-even, the cast
the pinned Cosmos export applies; ``teacher_rl.export_audit``, L2), and the bitwise and
elementwise rungs - including the ULP distance - are measured in the candidate's stored
dtype (``measured_in_dtype``). A zero-update export is therefore ``identical`` to its
release. Integer/boolean dtype differences stay structure failures. Without ``load`` a cast
tensor's bitwise identity cannot be decided from manifests (``cast_undetermined``,
``qualify_on_host``).

Tolerances are required, registered owner inputs (Huy; ``registration_ref`` names where
they are registered) and are never defaulted or widened here. Without one, a difference is
``tolerance_unregistered`` while every measured fact is still emitted. Tolerance semantics
(stated so Huy registers values knowing them): every non-null criterion must hold for each
differing element - ``|c - r| <= atol + rtol * |r|`` when ``atol`` or ``rtol`` is given (a
missing one counts as 0) and ``ulp(c, r) <= max_ulp`` when ``max_ulp`` is given. A non-finite
element that is not bitwise equal always exceeds. Relative error is ``|c - r| / |r|`` and is
undefined (``None``, counted in ``rel_error_reference_zero``) where ``r == 0``.

Verdicts: ``identical | within_registered_tolerance | exceeds | tolerance_unregistered``.

Agent technical defaults (D-018), inherited by Huy when he registers values: the conjunction
rule above (every non-null criterion must hold), the combined ``atol + rtol * |r|`` bound,
"a non-finite mismatch always exceeds", "integer and boolean tensors admit no tolerance",
the ``registration_ref`` form ``experiments/<file>.md#<anchor>`` (``REGISTRATION_REF``) with
the placeholder words ``PLACEHOLDERS`` rejected (as in R4 C11), "an overflowed error always
exceeds", the cross-dtype convention above, and stopping the ladder at the first failing
rung. None of them is a tolerance value.

Verification level per function (build-spec section 1):

- ``validate_tolerance``, ``validate_trajectory_tolerance`` - L1 (torch-free).
- ``ladder_from_manifests`` - L1 for rungs 1-2 on manifests; rung 3 runs only through an
  injected ``load`` callable whose tensors are re-hashed against the manifest (bytes verified
  at consumption); on an actual checkpoint it is L3 (``qualify_on_host``).
- ``compare_tensors``, ``compare_trajectories`` - L1 on toy CPU tensors with literal ULP and
  error known answers; actual E02/E04/E16 inputs are L3.

Torch is imported lazily. This module never loads model code, never downloads, never
writes files, never selects or widens a tolerance and never interprets a verdict as a
learning or validation claim.
"""

import hashlib
import math
import re
from pathlib import Path

from gate.s2_identity import identity_hash
from teacher_rl.export_audit import tensor_table

LADDER_KIND = "s4_parity_ladder"
ACTION_KIND = "s4_action_parity"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
VERDICTS = ("identical", "within_registered_tolerance", "exceeds", "tolerance_unregistered")
TOLERANCE_KEYS = ("registration_ref", "atol", "rtol", "max_ulp")
TRAJECTORY_TOLERANCE_KEYS = ("registration_ref", "max_displacement", "units")
PLACEHOLDERS = ("tbd", "tba", "todo", "later", "pending", "placeholder", "xxx", "fixme",
                "unknown", "unset", "none", "null")
REGISTRATION_REF = re.compile(
    r"experiments/[A-Za-z0-9][A-Za-z0-9._-]*\.md#[A-Za-z0-9][A-Za-z0-9._-]*"
)
CROSS_DTYPE_MODES = ("reference_cast_to_candidate",)
FLOAT_BITS = {"F64": 64, "F32": 32, "F16": 16, "BF16": 16, "F8_E4M3": 8, "F8_E5M2": 8}
UNITS = {
    "max_abs_error": "units of the tensor values",
    "max_rel_error": "|c - r| / |r| (dimensionless)",
    "max_ulp": "units in the last place of measured_in_dtype",
    "abs_error_overflow": "elements whose |c - r| exceeds the float64 range",
    "rel_error_overflow": "elements whose finite |c - r| / |r| exceeds the float64 range",
    "first_exceeding_index": "row-major flat element index",
}
_ENTRY_FIELDS = ("dtype", "shape", "sha256")


def _torch():
    import torch

    return torch


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _producer():
    return {"module": "teacher_rl.parity", "source_sha256": _source_sha256()}


def _issue(code, path, message, severity="error"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _check_scope(scope):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")


def _check_ref(ref):
    if type(ref) is not str or REGISTRATION_REF.fullmatch(ref) is None:
        raise ValueError("registration_ref must be experiments/<file>.md#<anchor>, naming "
                         "where the tolerance is registered")
    words = set(re.split(r"[^a-z0-9]+", ref.lower()))
    if words & set(PLACEHOLDERS) or "____" in ref or ".." in ref:
        raise ValueError("registration_ref is a placeholder")


def _nonnegative_finite(value, key):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be a finite non-negative number or null")
    return float(value)


def validate_tolerance(tolerance):
    """Validate a registered tensor tolerance; return its normalized form (L1).

    Exactly ``registration_ref``, ``atol``, ``rtol``, ``max_ulp``; at least one criterion is
    non-null; ``atol``/``rtol`` finite and non-negative; ``max_ulp`` a non-negative int.
    """
    if type(tolerance) is not dict or set(tolerance) != set(TOLERANCE_KEYS):
        raise ValueError(f"tolerance must have exactly the keys {TOLERANCE_KEYS}")
    _check_ref(tolerance["registration_ref"])
    atol = _nonnegative_finite(tolerance["atol"], "atol")
    rtol = _nonnegative_finite(tolerance["rtol"], "rtol")
    max_ulp = tolerance["max_ulp"]
    if max_ulp is not None and (type(max_ulp) is not int or max_ulp < 0):
        raise ValueError("max_ulp must be a non-negative integer or null")
    if atol is None and rtol is None and max_ulp is None:
        raise ValueError("a registered tolerance needs at least one criterion")
    return {"registration_ref": tolerance["registration_ref"], "atol": atol, "rtol": rtol,
            "max_ulp": max_ulp}


def validate_trajectory_tolerance(tolerance):
    """Validate a registered action-level tolerance (L1)."""
    if type(tolerance) is not dict or set(tolerance) != set(TRAJECTORY_TOLERANCE_KEYS):
        raise ValueError(f"trajectory tolerance needs exactly {TRAJECTORY_TOLERANCE_KEYS} "
                         "(units are required)")
    _check_ref(tolerance["registration_ref"])
    value = _nonnegative_finite(tolerance["max_displacement"], "max_displacement")
    if value is None:
        raise ValueError("max_displacement is required")
    if type(tolerance["units"]) is not str or not tolerance["units"].strip():
        raise ValueError("units must be a nonempty string")
    return {"registration_ref": tolerance["registration_ref"], "max_displacement": value,
            "units": tolerance["units"]}


# ---------------------------------------------------------------------------
# Rungs 1-2 over entry tables (torch-free)
# ---------------------------------------------------------------------------


def _check_cross_dtype(cross_dtype):
    if cross_dtype is not None and cross_dtype not in CROSS_DTYPE_MODES:
        raise ValueError(f"cross_dtype must be None or one of {CROSS_DTYPE_MODES}")


def _castable(a, b, cross_dtype):
    return cross_dtype is not None and a in FLOAT_BITS and b in FLOAT_BITS


def _structure(ref, cand, cross_dtype, declared_absent=()):
    common = sorted(set(ref) & set(cand))
    differ = [n for n in common if ref[n]["dtype"] != cand[n]["dtype"]]
    rung = {
        "status": "passed",
        "declared_absent": sorted(declared_absent),
        "missing_in_candidate": sorted(set(ref) - set(cand)),
        "unexpected_in_candidate": sorted(set(cand) - set(ref)),
        "dtype_mismatch": [n for n in differ
                           if not _castable(ref[n]["dtype"], cand[n]["dtype"], cross_dtype)],
        "shape_mismatch": [n for n in common if ref[n]["shape"] != cand[n]["shape"]],
    }
    if any(rung[k] for k in rung if k != "status"):
        rung["status"] = "blocked"
    rung["dtype_cast"] = [
        {"name": n, "reference_dtype": ref[n]["dtype"], "candidate_dtype": cand[n]["dtype"]}
        for n in differ if _castable(ref[n]["dtype"], cand[n]["dtype"], cross_dtype)
    ]
    return rung


def _finish(report):
    """Derive the status from the issues and deferred rungs (build-spec section 2)."""
    if any(i["severity"] == "error" for i in report["issues"]):
        report["status"] = "blocked"
    elif report["rungs"]["elementwise"]["status"] == "qualify_on_host":
        report["status"] = "qualify_on_host"
    else:
        report["status"] = "passed"
    return report


def _base_report(scope, tolerance, hashes, cross_dtype=None, names=None):
    return {
        "artifact_kind": LADDER_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": hashes,
        "units": dict(UNITS),
        "level": "L1",
        "status": "passed",
        "verdict": None,
        "tolerance": tolerance,
        "cross_dtype": cross_dtype,
        "names": names,
        "rungs": {
            "structure": None,
            "bitwise": {"status": "not_run", "identical": [], "differing": [],
                        "cast_undetermined": []},
            "elementwise": {"status": "not_run", "tensors": {}},
            "action_level": {"status": "qualify_on_host",
                             "note": "compare_trajectories on host-captured actions (L3)"},
        },
        "issues": [],
    }


def _run_rungs(report, ref, cand, measure, declared_absent=()):
    """Fill rungs 1-3. ``measure(name)`` returns an elementwise entry or raises."""
    tolerance = report["tolerance"]
    structure = _structure(ref, cand, report["cross_dtype"], declared_absent)
    report["rungs"]["structure"] = structure
    if structure["status"] != "passed":
        report["issues"].append(_issue("structure_mismatch", "structure",
                                       "key sets, dtypes or shapes differ"))
        report["verdict"] = "exceeds"
        return _finish(report)
    casts = {c["name"] for c in structure["dtype_cast"]}
    identical, differing, undetermined, entries = [], [], [], {}
    for name in sorted(ref):
        if name not in casts:
            (identical if ref[name]["sha256"] == cand[name]["sha256"] else differing).append(name)
        elif measure is None:
            undetermined.append(name)  # the cast reference's bytes are not in a manifest
        else:
            try:
                entry = measure(name)
            except _ConsumptionError as exc:
                report["issues"].append(_issue(exc.code, f"tensors/{name}", str(exc)))
                continue
            if entry["differing_elements"] == 0:
                identical.append(name)
            else:
                differing.append(name)
                entries[name] = entry
    report["rungs"]["bitwise"] = {"status": "passed", "identical": identical,
                                  "differing": differing, "cast_undetermined": undetermined}
    if report["issues"]:
        report["rungs"]["bitwise"]["status"] = "blocked"
        report["rungs"]["elementwise"] = {"status": "blocked", "tensors": entries}
        return _finish(report)
    if not differing and not undetermined:
        report["rungs"]["elementwise"] = {"status": "not_applicable", "tensors": {}}
        report["verdict"] = "identical"
        return _finish(report)
    if measure is None:
        report["rungs"]["elementwise"] = {"status": "qualify_on_host", "tensors": {}}
        return _finish(report)
    for name in differing:
        if name in entries:
            continue
        try:
            entries[name] = measure(name)
        except _ConsumptionError as exc:
            report["issues"].append(_issue(exc.code, f"tensors/{name}", str(exc)))
    entries = dict(sorted(entries.items()))
    if report["issues"]:
        report["rungs"]["elementwise"] = {"status": "blocked", "tensors": entries}
        return _finish(report)
    report["rungs"]["elementwise"] = {"status": "passed", "tensors": entries}
    if tolerance is None:
        report["verdict"] = "tolerance_unregistered"
    elif any(e["verdict"] == "exceeds" for e in entries.values()):
        report["verdict"] = "exceeds"
    else:
        report["verdict"] = "within_registered_tolerance"
    return _finish(report)


class _ConsumptionError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _check_names(names):
    if names is None:
        return None
    if (
        type(names) not in (list, tuple)
        or not names
        or not all(type(n) is str and n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("names must be None or a nonempty list of distinct nonempty strings")
    return sorted(names)


def _measure_pair(reference, candidate, ref_dtype, cand_dtype, tolerance):
    """Elementwise entry in the candidate's stored dtype (a declared cast rounds the
    reference with ``Tensor.to`` first)."""
    if ref_dtype != cand_dtype:
        reference = reference.to(candidate.dtype)
    entry = _elementwise(reference, candidate, cand_dtype, tolerance)
    entry["reference_dtype"] = ref_dtype
    entry["measured_in_dtype"] = cand_dtype
    return entry


def ladder_from_manifests(reference, candidate, *, tolerance, load=None, cross_dtype=None,
                          names=None):
    """Run the ladder on two tensor manifests (L1 rungs 1-2; rung 3 via ``load``).

    ``load(side, name)`` (``side`` is ``"reference"`` or ``"candidate"``) returns the torch
    tensor; each loaded tensor is re-hashed and must equal its manifest entry
    (``bytes_changed_since_manifest`` otherwise, blocking with no verdict) before any cast
    or measurement. Without ``load`` the elementwise rung is ``qualify_on_host`` and no
    verdict is given for differing or cast tensors. ``names`` restricts the comparison to
    declared tensor names (e.g. E02's ``requires_parity_ladder``); a declared name absent
    from both manifests is a structure failure (``declared_absent``).
    """
    norm = None if tolerance is None else validate_tolerance(tolerance)
    _check_cross_dtype(cross_dtype)
    declared = _check_names(names)
    tables, issues = {}, []
    for label, artifact in (("reference", reference), ("candidate", candidate)):
        table, problem = tensor_table(artifact)
        if problem:
            issues.append(_issue("input_invalid", label, problem))
        tables[label] = table
    if not issues and reference["scope"] != candidate["scope"]:
        issues.append(_issue("scope_mismatch", "scope", "manifests come from different scopes"))
    hashes = {label: (artifact.get("tensor_set_sha256") if type(artifact) is dict else None)
              for label, artifact in (("reference", reference), ("candidate", candidate))}
    report = _base_report(None if issues else reference["scope"], norm, hashes, cross_dtype,
                          declared)
    if issues:
        report["status"] = "blocked"
        report["issues"] = issues
        return report
    ref, cand = tables["reference"], tables["candidate"]
    absent = ()
    if declared is not None:
        absent = [n for n in declared if n not in ref and n not in cand]
        ref = {n: ref[n] for n in declared if n in ref}
        cand = {n: cand[n] for n in declared if n in cand}
    if load is None:
        return _run_rungs(report, ref, cand, None, absent)
    from teacher_rl.params import tensor_identity

    def measure(name):
        tensors = []
        for side, table in (("reference", ref), ("candidate", cand)):
            tensor = load(side, name)
            ident = tensor_identity(tensor)
            if any(ident[k] != table[name][k] for k in _ENTRY_FIELDS):
                raise _ConsumptionError("bytes_changed_since_manifest",
                                        f"{side} tensor differs from its manifest entry")
            tensors.append(tensor)
        return _measure_pair(tensors[0], tensors[1], ref[name]["dtype"], cand[name]["dtype"],
                             norm)

    return _run_rungs(report, ref, cand, measure, absent)


def compare_tensors(reference, candidate, *, tolerance, scope, cross_dtype=None):
    """Run the ladder on two mappings of name -> torch tensor (L1 on CPU toys)."""
    from teacher_rl.params import tensor_identity

    _check_scope(scope)
    _check_cross_dtype(cross_dtype)
    norm = None if tolerance is None else validate_tolerance(tolerance)
    if not reference or not candidate:
        raise ValueError("compare_tensors needs nonempty mappings")
    ref = {n: tensor_identity(t) for n, t in sorted(reference.items())}
    cand = {n: tensor_identity(t) for n, t in sorted(candidate.items())}

    def set_hash(table):
        return identity_hash({n: {k: e[k] for k in ("dtype", "shape", "byte_length", "sha256")}
                              for n, e in table.items()})

    report = _base_report(scope, norm, {"reference": set_hash(ref), "candidate": set_hash(cand)},
                          cross_dtype)

    def measure(name):
        return _measure_pair(reference[name], candidate[name], ref[name]["dtype"],
                             cand[name]["dtype"], norm)

    return _run_rungs(report, ref, cand, measure)


# ---------------------------------------------------------------------------
# Rung 3: elementwise (torch)
# ---------------------------------------------------------------------------


def _plain(tensor):
    """CPU, contiguous, with lazy conjugate/negative bits materialized (logical values)."""
    return tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()


def _ulp_keys(tensor, bits):
    """Map float bit patterns to integers ordered like the values (sign-magnitude)."""
    import numpy as np

    torch = _torch()
    view = {64: torch.int64, 32: torch.int32, 16: torch.int16, 8: torch.int8}[bits]
    raw = _plain(tensor).reshape(-1).view(view).numpy().astype(np.int64)
    magnitude = raw & np.int64((1 << (bits - 1)) - 1)
    return np.where(raw < 0, -magnitude, magnitude)


def _ulp_distance(a, b, bits):
    import numpy as np

    ka, kb = _ulp_keys(a, bits), _ulp_keys(b, bits)
    ua, ub = ka.astype(np.uint64), kb.astype(np.uint64)
    return np.where(ka >= kb, ua - ub, ub - ua)  # exact modulo 2**64; max 2**64 - 2


def _elementwise(reference, candidate, dtype, tolerance):
    import numpy as np

    torch = _torch()
    r = _plain(reference).reshape(-1)
    c = _plain(candidate).reshape(-1)
    itemsize = r.element_size()
    equal_bits = (
        r.view(torch.uint8).reshape(-1, itemsize) == c.view(torch.uint8).reshape(-1, itemsize)
    ).all(dim=1).numpy() if r.numel() else np.zeros(0, dtype=bool)
    differ = ~equal_bits
    entry = {
        "dtype": dtype,
        "shape": [int(d) for d in reference.shape],
        "numel": int(r.numel()),
        "differing_elements": int(differ.sum()),
        "max_abs_error": None,
        "max_rel_error": None,
        "rel_error_reference_zero": 0,
        "max_ulp": None,
        "abs_error_overflow": 0,
        "rel_error_overflow": 0,
        "nonfinite_mismatch": 0,
        "elements_exceeding": None,
        "first_exceeding_index": None,
        "verdict": None,
    }
    if dtype not in FLOAT_BITS:
        # Python ints over the differing elements only: no int64 wrap-around.
        index = np.flatnonzero(differ).tolist()
        rl, cl = r.tolist(), c.tolist()
        entry["max_abs_error"] = max((abs(int(cl[i]) - int(rl[i])) for i in index), default=0)
        exceeding = differ
        entry["verdict"] = "exceeds" if tolerance is not None else "tolerance_unregistered"
    else:
        rv, cv = r.to(torch.float64).numpy(), c.to(torch.float64).numpy()
        finite = np.isfinite(rv) & np.isfinite(cv)
        nonfinite = differ & ~finite
        entry["nonfinite_mismatch"] = int(nonfinite.sum())
        measurable = differ & finite
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            absolute = np.abs(cv - rv)
            relative = np.where(rv != 0, absolute / np.where(rv != 0, np.abs(rv), 1.0), 0.0)
        overflow = measurable & ~np.isfinite(absolute)
        entry["abs_error_overflow"] = int(overflow.sum())
        ulp = _ulp_distance(r, c, FLOAT_BITS[dtype])
        if measurable.any():
            if not overflow.any():
                entry["max_abs_error"] = float(absolute[measurable].max())
            entry["max_ulp"] = int(ulp[measurable].max())
            nonzero = measurable & (rv != 0)
            zero_ref = measurable & (rv == 0) & (absolute != 0)
            entry["rel_error_reference_zero"] = int(zero_ref.sum())
            rel_overflow = nonzero & ~overflow & ~np.isfinite(relative)
            entry["rel_error_overflow"] = int(rel_overflow.sum())
            if nonzero.any() and np.isfinite(relative[nonzero]).all():
                entry["max_rel_error"] = float(relative[nonzero].max())
        exceeding = nonfinite | overflow
        if tolerance is not None:
            if tolerance["atol"] is not None or tolerance["rtol"] is not None:
                scale = np.where(finite, np.abs(rv), 0.0)
                bound = (tolerance["atol"] or 0.0) + (tolerance["rtol"] or 0.0) * scale
                exceeding |= measurable & ~(absolute <= bound)
            if tolerance["max_ulp"] is not None:
                exceeding |= measurable & (ulp > np.uint64(tolerance["max_ulp"]))
    if tolerance is None:
        entry["verdict"] = "tolerance_unregistered"
        return entry
    entry["elements_exceeding"] = int(exceeding.sum())
    if entry["elements_exceeding"]:
        entry["first_exceeding_index"] = int(np.flatnonzero(exceeding)[0])
        entry["verdict"] = "exceeds"
    else:
        entry["verdict"] = "within_registered_tolerance"
    return entry


# ---------------------------------------------------------------------------
# Rung 4: action-level trajectories
# ---------------------------------------------------------------------------


def compare_trajectories(reference, candidate, *, tolerance, scope):
    """Compare trajectories ``id -> [T, D]`` by per-waypoint Euclidean displacement (L1).

    Displacement is the L2 norm over the last dimension, computed in float64. Structure
    (ids, shapes, dtypes) must match, else verdict ``exceeds`` with status ``blocked``; a
    non-finite waypoint that is not bitwise equal exceeds; a displacement that overflows
    float64 is counted in ``displacement_overflow``, makes max/mean ``None`` (not
    estimable) and exceeds. The trajectory units come from the registered tolerance. Real
    inputs are host-captured (L3).
    """
    import numpy as np

    torch = _torch()
    _check_scope(scope)
    norm = None if tolerance is None else validate_trajectory_tolerance(tolerance)
    for mapping in (reference, candidate):
        if not mapping:
            raise ValueError("compare_trajectories needs nonempty mappings")
        for value in mapping.values():
            if not isinstance(value, torch.Tensor) or value.dim() != 2:
                raise ValueError("each trajectory must be a 2-D [waypoints, dims] tensor")
    common = sorted(set(reference) & set(candidate))
    structure = {
        "missing_in_candidate": sorted(set(reference) - set(candidate)),
        "unexpected_in_candidate": sorted(set(candidate) - set(reference)),
        "shape_mismatch": [n for n in common
                           if list(reference[n].shape) != list(candidate[n].shape)],
        "dtype_mismatch": [n for n in common if reference[n].dtype != candidate[n].dtype],
    }
    report = {
        "artifact_kind": ACTION_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": {},
        "units": {"displacement": None if norm is None else norm["units"],
                  "waypoints": "rows of the trajectory tensor",
                  "displacement_overflow": "waypoints"},
        "level": "L1",
        "status": "passed",
        "verdict": None,
        "tolerance": norm,
        "structure": structure,
        "trajectories": {},
        "issues": [],
    }
    if any(structure.values()):
        report["issues"].append(_issue("structure_mismatch", "structure",
                                       "trajectory ids, shapes or dtypes differ"))
        report["verdict"] = "exceeds"
        report["status"] = "blocked"
        return report
    verdicts = []
    for name in common:
        r = reference[name].detach().cpu().to(torch.float64).numpy()
        c = candidate[name].detach().cpu().to(torch.float64).numpy()
        same_bits = torch.equal(_plain(reference[name]).view(torch.uint8),
                                _plain(candidate[name]).view(torch.uint8))
        finite = np.isfinite(r).all(axis=1) & np.isfinite(c).all(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            disp = np.where(finite, np.sqrt(((c - r) ** 2).sum(axis=1)), np.inf)
        overflow = finite & ~np.isfinite(disp)
        measured = finite & ~overflow
        bad = int((~finite).sum()) if not same_bits else 0
        finite_disp = disp[measured]
        estimable = finite_disp.size and not overflow.any()
        entry = {
            "waypoints": int(r.shape[0]),
            "max_displacement": float(finite_disp.max()) if estimable else None,
            "mean_displacement": float(finite_disp.mean()) if estimable else None,
            "final_displacement": float(disp[-1]) if r.shape[0] and measured[-1] else None,
            "argmax_waypoint": int(np.argmax(disp)) if r.shape[0] else None,
            "nonfinite_waypoints": bad,
            "displacement_overflow": int(overflow.sum()),
            "verdict": None,
        }
        if same_bits:
            entry["verdict"] = "identical"
        elif norm is None:
            entry["verdict"] = "tolerance_unregistered"
        elif (bad or overflow.any()
              or (entry["max_displacement"] or 0.0) > norm["max_displacement"]):
            entry["verdict"] = "exceeds"
        else:
            entry["verdict"] = "within_registered_tolerance"
        report["trajectories"][name] = entry
        verdicts.append(entry["verdict"])
    if all(v == "identical" for v in verdicts):
        report["verdict"] = "identical"
    elif "exceeds" in verdicts:
        report["verdict"] = "exceeds"
    elif "tolerance_unregistered" in verdicts:
        report["verdict"] = "tolerance_unregistered"
    else:
        report["verdict"] = "within_registered_tolerance"
    return report
