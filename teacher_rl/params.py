"""Parameter/buffer snapshots, diffs, trainable-set and optimizer-state checks (A10; E03, E07).

Serves S4 engineering preparation (S4.1-S4.4, S4.6 [C]; G9, G7) under D-024. Agent-authored
evaluator engineering (D-018). Tickets: E03 (snapshot every named parameter and buffer before
and after one trainer step; frozen VLM unchanged; only declared trainable tensors change;
``vlm.lm_head.weight`` explicit; trainable dtypes recorded, R3 finding 4), the torch part of
E07 (a skipped optimizer step at exactly-zero reduced gradient norm leaves parameters and
optimizer moments/step counters unchanged while the scheduler advances), and E02 inputs
(policy snapshot that classifies tensors absent from an export as parameters or buffers).

Tensor identity follows build-spec section 2 rule 11, shared with torch-free
``gate.tensor_manifest``: SHA-256 over the tensor's raw bytes in its stored dtype, row-major
contiguous, little-endian (exactly the byte range safetensors stores), with the safetensors
dtype name and shape recorded beside it. Non-contiguous, negated or conjugated views are
materialized first, so the hash is of the logical values, never of the storage.

Verification level per function (build-spec section 1):

- ``safetensors_dtype``, ``tensor_bytes``, ``tensor_identity``, ``snapshot``, ``diff``,
  ``check_trainable_set``, ``optimizer_state_snapshot``, ``compare_optimizer_state`` - L1
  (toy torch modules on CPU with literal known answers and negative controls).
- The same functions applied in process to the pinned AlpaGym trainer (H4 capture) are L3
  (``qualify_on_host``): a toy pass does not show the 10B policy's mask or dtypes.

Agent technical defaults (D-018): ``DEFAULT_PREFIXES`` (the pinned Alpamayo top-level module
names used to group changed fractions; names outside them fall in ``(other)``) and
``EXPLICIT_NAMES`` (``vlm.lm_head.weight``, reported explicitly for E03). The trainable and
frozen prefix lists are required inputs of ``check_trainable_set`` (engineering facts from the
pinned freeze sites, R3 section 5), never defaulted.

Torch is imported lazily (only where a tensor or module is handled), so ``diff`` and
``check_trainable_set`` run torch-free on JSON artifacts. This module never trains, never
calls ``optimizer.step``, never loads model code or checkpoints, never downloads, never
writes files and never decides whether a change is scientifically acceptable.
"""

import hashlib
import math
import sys
from collections.abc import Mapping
from pathlib import Path

from gate.s2_identity import identity_hash
from gate.tensor_manifest import DTYPE_ITEMSIZE, HASH_CONVENTION, tensor_record
from teacher_rl.export_audit import policy_name_to_release_key, tensor_table

if sys.byteorder != "little":  # rule 11 is little-endian; never byte-swap silently
    raise ImportError("teacher_rl.params requires a little-endian host")

SNAPSHOT_KIND = "s4_param_snapshot"
DIFF_KIND = "s4_param_diff"
TRAINABLE_KIND = "s4_trainable_set_check"
OPTIMIZER_KIND = "s4_optimizer_state_snapshot"
OPTIMIZER_COMPARISON_KIND = "s4_optimizer_state_comparison"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
DEFAULT_PREFIXES = ("vlm.", "expert.", "action_in_proj.", "action_out_proj.")
OTHER_BUCKET = "(other)"
EXPLICIT_NAMES = ("vlm.lm_head.weight",)
UNITS = {
    "numel": "elements",
    "byte_length": "bytes",
    "shape": "elements per dimension, row-major",
    "changed_tensor_fraction": "changed tensors / tensors in the bucket",
    "changed_numel_fraction": "elements of changed tensors / elements in the bucket",
    "uncompared_tensors": "tensors of the bucket present in only one snapshot",
}
_ENTRY_FIELDS = ("dtype", "shape", "byte_length", "sha256")
_DIFF_LISTS = ("changed", "only_in_before", "only_in_after", "requires_grad_changed",
               "kind_changed")
_DRIFT_ERRORS = (
    ("only_in_before", "tensor_only_in_before",
     "present before the step but absent after it (vanished or renamed)"),
    ("only_in_after", "tensor_only_in_after",
     "present after the step but absent before it (appeared or renamed)"),
    ("requires_grad_changed", "requires_grad_changed",
     "requires_grad differs between the two snapshots"),
    ("kind_changed", "kind_changed", "parameter/buffer kind differs between the snapshots"),
)
_DTYPE_NAMES = {
    "float64": "F64", "float32": "F32", "float16": "F16", "bfloat16": "BF16",
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8", "uint8": "U8",
    "bool": "BOOL", "float8_e4m3fn": "F8_E4M3", "float8_e5m2": "F8_E5M2",
}
if set(_DTYPE_NAMES.values()) != set(DTYPE_ITEMSIZE):  # one rule-11 dtype table
    raise ImportError("teacher_rl.params dtype names differ from gate.tensor_manifest")


class SnapshotError(ValueError):
    """A tensor, module or optimizer that cannot be snapshotted (fails closed), with a code."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _torch():
    import torch

    return torch


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _producer():
    return {"module": "teacher_rl.params", "source_sha256": _source_sha256()}


def _issue(code, path, message, severity="error"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _sorted_issues(issues):
    return sorted(issues, key=lambda i: (i["path"], i["code"], i["message"]))


def _status(issues):
    return "blocked" if any(i["severity"] == "error" for i in issues) else "passed"


def _check_scope(scope):
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")


def _try_hash(value):
    try:
        return identity_hash(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Rule-11 tensor identity
# ---------------------------------------------------------------------------


def safetensors_dtype(tensor):
    """Return the rule-11 safetensors dtype name of a torch tensor or raise (L1)."""
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise SnapshotError("not_a_tensor", f"expected a torch.Tensor, got {type(tensor)}")
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in _DTYPE_NAMES:
        raise SnapshotError("unsupported_dtype", f"{tensor.dtype} is outside rule 11")
    return _DTYPE_NAMES[name]


def tensor_bytes(tensor):
    """Return the rule-11 raw bytes of a tensor as a read-only memoryview (L1).

    Rejects tensor subclasses other than ``nn.Parameter`` (call ``full_tensor()`` on a
    DTensor first), sparse, quantized and meta tensors. Copies to CPU and materializes
    non-contiguous, negated or conjugated views.
    """
    torch = _torch()
    dtype = safetensors_dtype(tensor)
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        raise SnapshotError("tensor_subclass_unsupported",
                            f"{type(tensor).__name__} must be materialized to a plain tensor")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise SnapshotError("tensor_layout_unsupported", f"layout {tensor.layout}")
    if tensor.device.type == "meta":
        raise SnapshotError("tensor_has_no_data", "meta tensors carry no bytes")
    plain = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
    raw = plain.reshape(-1).view(torch.uint8).numpy()
    view = memoryview(raw).toreadonly()
    del dtype
    return view


def tensor_identity(tensor):
    """Return ``{dtype, shape, numel, byte_length, sha256}`` under rule 11 (L1).

    The hash, byte-length check and dtype table are ``gate.tensor_manifest``'s
    (``tensor_record``), so a snapshot and a safetensors manifest share one implementation.
    """
    record = tensor_record(safetensors_dtype(tensor), [int(d) for d in tensor.shape],
                           tensor_bytes(tensor))
    return {
        "dtype": record["dtype"],
        "shape": record["shape"],
        "numel": int(tensor.numel()),
        "byte_length": record["byte_length"],
        "sha256": record["sha256"],
    }


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _check_tensor_name(name):
    if type(name) is not str or not name or "\x00" in name:
        raise ValueError("Tensor names must be nonempty strings without NUL")


def _buffer_persistent(module, name):
    owner, _, leaf = name.rpartition(".")
    holder = module.get_submodule(owner) if owner else module
    return leaf not in holder._non_persistent_buffers_set


def snapshot(module_or_state, *, scope, buffers=True):
    """Snapshot every named parameter (and buffer) with rule-11 identities (L1).

    An ``nn.Module`` yields ``named_parameters()`` (shared tensors once, as the pinned Cosmos
    export iterates them) with ``requires_grad``, ``named_buffers()`` with ``persistent``,
    and ``.training`` for every named module. A mapping (e.g. a ``state_dict``) yields kind
    ``state`` entries whose ``requires_grad``/``persistent`` are ``None`` (unknown, never
    ``False``) and ``modules`` ``None``.
    """
    torch = _torch()
    _check_scope(scope)
    if type(buffers) is not bool:
        raise ValueError("buffers must be True or False")
    tensors = {}
    if isinstance(module_or_state, torch.nn.Module):
        module = module_or_state
        source = "module"
        for name, param in module.named_parameters():
            tensors[name] = {"kind": "parameter", **tensor_identity(param),
                             "requires_grad": bool(param.requires_grad), "persistent": None}
        if buffers:
            for name, buf in module.named_buffers():
                if name in tensors:
                    raise SnapshotError("duplicate_name", f"{name!r} is parameter and buffer")
                tensors[name] = {"kind": "buffer", **tensor_identity(buf),
                                 "requires_grad": bool(buf.requires_grad),
                                 "persistent": _buffer_persistent(module, name)}
        modules = {name: bool(sub.training) for name, sub in module.named_modules()}
    elif isinstance(module_or_state, Mapping):
        source = "state_dict"
        modules = None
        for name, tensor in module_or_state.items():
            _check_tensor_name(name)
            tensors[name] = {"kind": "state", **tensor_identity(tensor),
                             "requires_grad": None, "persistent": None}
    else:
        raise TypeError("snapshot needs an nn.Module or a mapping of name -> tensor")
    for name in tensors:
        _check_tensor_name(name)
    if not tensors:
        raise ValueError("Nothing to snapshot: the module or mapping is empty")
    tensors = {name: _order_entry(tensors[name]) for name in sorted(tensors)}
    kinds = [e["kind"] for e in tensors.values()]
    return {
        "artifact_kind": SNAPSHOT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": {},
        "units": dict(UNITS),
        "hash_convention": HASH_CONVENTION,
        "status": "passed",
        "source": source,
        "tensors": tensors,
        "modules": None if modules is None else dict(sorted(modules.items())),
        "tensor_set_sha256": identity_hash(
            {n: {k: e[k] for k in _ENTRY_FIELDS} for n, e in tensors.items()}
        ),
        "totals": {
            "tensor_count": len(tensors),
            "parameter_count": kinds.count("parameter"),
            "buffer_count": kinds.count("buffer"),
            "numel": sum(e["numel"] for e in tensors.values()),
            "byte_length": sum(e["byte_length"] for e in tensors.values()),
        },
        "issues": [],
    }


def _order_entry(entry):
    keys = ("kind", "dtype", "shape", "numel", "byte_length", "sha256", "requires_grad",
            "persistent")
    return {k: entry[k] for k in keys}


# ---------------------------------------------------------------------------
# Diff (torch-free)
# ---------------------------------------------------------------------------


def _validated(artifact, label):
    """Return (tensors, problem) for a snapshot or tensor manifest artifact."""
    table, problem = tensor_table(artifact)
    if problem:
        return None, _issue("input_invalid", label, problem)
    full = artifact["tensors"]
    return {
        name: {
            **{k: entry[k] for k in _ENTRY_FIELDS},
            "kind": entry["kind"],
            "requires_grad": full[name].get("requires_grad"),
            "numel": math.prod(entry["shape"]),
        }
        for name, entry in table.items()
    }, None


def _bucket(name, prefixes):
    release_name = policy_name_to_release_key(name)
    for prefix in prefixes:
        if release_name.startswith(prefix):
            return prefix
    return OTHER_BUCKET


def _check_prefixes(prefixes, label):
    if (
        type(prefixes) not in (tuple, list)
        or not prefixes
        or not all(type(p) is str and p for p in prefixes)
        or len(set(prefixes)) != len(prefixes)
    ):
        raise ValueError(f"{label} prefixes must be a nonempty list of distinct nonempty strings")
    return tuple(prefixes)


def diff(before, after, *, prefixes=DEFAULT_PREFIXES):
    """Compare two snapshots (or tensor manifests) tensor by tensor (L1, torch-free).

    A tensor is ``changed`` when its dtype, shape or byte hash differs. Tensors are matched
    by their raw names; a name present in only one snapshot is listed in ``only_in_before``
    / ``only_in_after`` and never counted as unchanged. Changed fractions are reported per
    prefix both as tensor counts and as element counts of changed tensors over the
    ``before`` tensors (``UNITS``); a bucket with no tensors, or with any tensor present in
    only one snapshot (``uncompared_tensors`` > 0), has fractions ``None`` (not estimable:
    missing is never zero). Names are grouped after stripping compile/checkpoint wrappers
    and one policy-wrapper prefix (``expert_model.``), so policy snapshots and release
    manifests group alike. Key-set drift is a fact here (status stays ``passed``, with a
    ``tensor_set_changed`` warning); ``check_trainable_set`` turns it into a blocking error.
    """
    prefixes = _check_prefixes(prefixes, "diff")
    issues = []
    tables = {}
    for label, artifact in (("before", before), ("after", after)):
        table, problem = _validated(artifact, label)
        if problem:
            issues.append(problem)
        tables[label] = table
    if not issues and before["scope"] != after["scope"]:
        issues.append(_issue("scope_mismatch", "scope", "snapshots come from different scopes"))
    report = {
        "artifact_kind": DIFF_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": None,
        "producer": _producer(),
        "input_hashes": {"before": _try_hash(before), "after": _try_hash(after)},
        "units": dict(UNITS),
        "status": "blocked",
        "prefixes": list(prefixes),
        "changed": None,
        "unchanged": None,
        "only_in_before": None,
        "only_in_after": None,
        "dtype_changed": None,
        "shape_changed": None,
        "requires_grad_changed": None,
        "kind_changed": None,
        "training_changed": None,
        "modules_only_in_before": None,
        "modules_only_in_after": None,
        "by_prefix": None,
        "issues": issues,
    }
    if issues:
        return report
    a, b = tables["before"], tables["after"]
    common = sorted(set(a) & set(b))
    changed = [n for n in common if any(a[n][k] != b[n][k] for k in ("dtype", "shape", "sha256"))]
    changed_set = set(changed)
    only_before, only_after = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    buckets = {p: [0, 0, 0, 0, 0] for p in (*prefixes, OTHER_BUCKET)}
    for name in sorted(a):
        counts = buckets[_bucket(name, prefixes)]
        counts[0] += 1
        counts[2] += a[name]["numel"]
        if name in changed_set:
            counts[1] += 1
            counts[3] += a[name]["numel"]
    for name in (*only_before, *only_after):
        buckets[_bucket(name, prefixes)][4] += 1

    def fraction(part, whole, uncompared):
        return None if whole == 0 or uncompared else part / whole

    by_prefix = {
        p: {"tensors": t, "changed_tensors": c, "changed_tensor_fraction": fraction(c, t, u),
            "numel": n, "changed_numel": cn, "changed_numel_fraction": fraction(cn, n, u),
            "uncompared_tensors": u}
        for p, (t, c, n, cn, u) in buckets.items()
    }
    for label, names in (("only_in_before", only_before), ("only_in_after", only_after)):
        if names:
            issues.append(_issue("tensor_set_changed", label,
                                 f"{len(names)} tensors have no counterpart (never compared)",
                                 severity="warning"))
    ma, mb = before.get("modules"), after.get("modules")
    has_modules = isinstance(ma, dict) and isinstance(mb, dict)
    report.update(
        scope=before["scope"],
        status="passed",
        changed=changed,
        unchanged=[n for n in common if n not in changed_set],
        only_in_before=only_before,
        only_in_after=only_after,
        dtype_changed=[n for n in common if a[n]["dtype"] != b[n]["dtype"]],
        shape_changed=[n for n in common if a[n]["shape"] != b[n]["shape"]],
        requires_grad_changed=[n for n in common
                               if a[n]["requires_grad"] != b[n]["requires_grad"]],
        kind_changed=[n for n in common if a[n]["kind"] != b[n]["kind"]],
        training_changed=sorted(m for m in set(ma) & set(mb) if ma[m] != mb[m])
        if has_modules else None,
        modules_only_in_before=sorted(set(ma) - set(mb)) if has_modules else None,
        modules_only_in_after=sorted(set(mb) - set(ma)) if has_modules else None,
        by_prefix=by_prefix,
        issues=_sorted_issues(issues),
    )
    return report


# ---------------------------------------------------------------------------
# E03 trainable set (torch-free)
# ---------------------------------------------------------------------------


def check_trainable_set(manifest, trainable_prefixes, frozen_prefixes, *, diff=None,
                        explicit_names=EXPLICIT_NAMES):
    """Check a snapshot's trainable mask against declared prefixes (E03; L1, torch-free).

    Every parameter must match exactly one declared prefix (``tensor_unclassified`` /
    ``prefix_overlap`` otherwise); trainable ones must have ``requires_grad`` true and frozen
    ones false; unknown ``requires_grad`` (state dicts, manifests) blocks. With ``diff`` (a
    ``diff`` whose ``before`` or ``after`` is this manifest), a changed frozen parameter or
    any changed buffer blocks; so does any key-set drift (``tensor_only_in_before`` /
    ``tensor_only_in_after``: a vanished or renamed tensor is never a pass), any
    ``requires_grad_changed`` and any ``kind_changed`` tensor; an unchanged trainable tensor
    is reported (``info``). An explicit name absent from either side is ``changed``.
    Trainable dtypes are recorded; non-F32 trainables are a warning (R3 finding 4: casting
    masters is Huy's training-method decision).
    """
    trainable = _check_prefixes(trainable_prefixes, "trainable")
    frozen = _check_prefixes(frozen_prefixes, "frozen")
    issues = []
    table, problem = _validated(manifest, "manifest")
    report = {
        "artifact_kind": TRAINABLE_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": None,
        "producer": _producer(),
        "input_hashes": {"manifest": _try_hash(manifest),
                         "diff": None if diff is None else _try_hash(diff)},
        "units": {"dtype_counts": "tensors"},
        "status": "blocked",
        "trainable_prefixes": list(trainable),
        "frozen_prefixes": list(frozen),
        "trainable": None,
        "frozen": None,
        "unclassified": None,
        "buffers": None,
        "explicit": None,
        "trainable_dtypes": None,
        "frozen_dtypes": None,
        "changed_frozen": None,
        "changed_buffers": None,
        "unchanged_trainable": None,
        "issues": issues,
    }
    if problem:
        issues.append(problem)
        return report
    changed = drift = None
    if diff is not None:
        if (
            type(diff) is not dict
            or diff.get("artifact_kind") != DIFF_KIND
            or diff.get("schema_version") != SCHEMA_VERSION
            or diff.get("status") != "passed"
            or type(diff.get("input_hashes")) is not dict
            or any(type(diff.get(key)) is not list for key in _DIFF_LISTS)
        ):
            issues.append(_issue("diff_invalid", "diff", "not a passed s4_param_diff"))
        elif report["input_hashes"]["manifest"] not in (
            diff["input_hashes"].get("before"), diff["input_hashes"].get("after")
        ):
            issues.append(_issue("diff_not_for_manifest", "diff",
                                 "the diff was computed from other snapshots"))
        else:
            changed = set(diff["changed"])
            drift = set(diff["only_in_before"]) | set(diff["only_in_after"])
            for key, code, message in _DRIFT_ERRORS:
                for name in sorted(diff[key]):
                    issues.append(_issue(code, name, message))

    groups = {"trainable": [], "frozen": [], "unclassified": [], "buffers": []}
    dtypes = {"trainable": {}, "frozen": {}}
    classification = {}
    for name, entry in table.items():
        if entry["kind"] == "buffer":
            groups["buffers"].append(name)
            classification[name] = "buffer"
            continue
        release_name = policy_name_to_release_key(name)
        is_trainable = release_name.startswith(trainable)
        is_frozen = release_name.startswith(frozen)
        if is_trainable and is_frozen:
            issues.append(_issue("prefix_overlap", name, "matches trainable and frozen prefixes"))
            classification[name] = "ambiguous"
            continue
        if not (is_trainable or is_frozen):
            issues.append(_issue("tensor_unclassified", name, "matches no declared prefix"))
            groups["unclassified"].append(name)
            classification[name] = "unclassified"
            continue
        group = "trainable" if is_trainable else "frozen"
        classification[name] = group
        groups[group].append(name)
        dtypes[group][entry["dtype"]] = dtypes[group].get(entry["dtype"], 0) + 1
        grad = entry["requires_grad"]
        if grad is None:
            issues.append(_issue("requires_grad_unknown", name,
                                 "snapshot an nn.Module to record requires_grad"))
        elif group == "trainable" and grad is not True:
            issues.append(_issue("trainable_parameter_not_requires_grad", name,
                                 "declared trainable but requires_grad is False"))
        elif group == "frozen" and grad is not False:
            issues.append(_issue("frozen_parameter_requires_grad", name,
                                 "declared frozen but requires_grad is True"))
        if group == "trainable" and entry["dtype"] != "F32":
            issues.append(_issue("trainable_not_float32", name,
                                 f"trainable tensor stored as {entry['dtype']}",
                                 severity="warning"))

    changed_frozen = changed_buffers = unchanged_trainable = None
    if changed is not None:
        changed_frozen = sorted(n for n in groups["frozen"] if n in changed)
        changed_buffers = sorted(n for n in groups["buffers"] if n in changed)
        unchanged_trainable = sorted(n for n in groups["trainable"] if n not in changed)
        for name in changed_frozen:
            issues.append(_issue("frozen_parameter_changed", name, "frozen parameter changed"))
        for name in changed_buffers:
            issues.append(_issue("buffer_changed", name,
                                 "buffer changed: runtime behaviour is not frozen"))
        for name in unchanged_trainable:
            issues.append(_issue("trainable_unchanged", name,
                                 "declared trainable but unchanged", severity="info"))

    explicit = {}
    for name in explicit_names:
        matches = [n for n in table if policy_name_to_release_key(n) == name]
        drifted = drift is not None and any(
            policy_name_to_release_key(n) == name for n in drift
        )
        if not matches:
            issues.append(_issue("explicit_name_absent", name, "explicit tensor not present",
                                 severity="warning"))
            explicit[name] = {"present": False, "classification": None,
                              "requires_grad": None, "changed": True if drifted else None}
            continue
        actual = matches[0]
        explicit[name] = {
            "present": True,
            "classification": classification[actual],
            "requires_grad": table[actual]["requires_grad"],
            "changed": None if changed is None else (actual in changed or actual in drift),
        }
    report.update(
        scope=manifest["scope"],
        trainable=sorted(groups["trainable"]),
        frozen=sorted(groups["frozen"]),
        unclassified=sorted(groups["unclassified"]),
        buffers=sorted(groups["buffers"]),
        explicit=explicit,
        trainable_dtypes=dict(sorted(dtypes["trainable"].items())),
        frozen_dtypes=dict(sorted(dtypes["frozen"].items())),
        changed_frozen=changed_frozen,
        changed_buffers=changed_buffers,
        unchanged_trainable=unchanged_trainable,
        issues=_sorted_issues(issues),
        status=_status(issues),
    )
    return report


# ---------------------------------------------------------------------------
# E07 optimizer state
# ---------------------------------------------------------------------------


def _json_scalar(value, path):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        entry = tensor_identity(value)
        if value.numel() == 1:
            item = value.detach().cpu().reshape(-1)[0].item()
            entry["value"] = item if not isinstance(item, float) or math.isfinite(item) else None
        return entry
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SnapshotError("hyperparameter_not_finite", path)
        return value
    if type(value) in (tuple, list):
        return [_json_scalar(v, f"{path}[{i}]") for i, v in enumerate(value)]
    raise SnapshotError("hyperparameter_not_serializable", f"{path} has {type(value).__name__}")


def _parameter_names(named_parameters):
    torch = _torch()
    if isinstance(named_parameters, torch.nn.Module):
        items = named_parameters.named_parameters(remove_duplicate=True)
    elif isinstance(named_parameters, Mapping):
        items = named_parameters.items()
    else:
        raise TypeError("named_parameters must be an nn.Module or a mapping name -> parameter")
    return {id(param): name for name, param in items}


def optimizer_state_snapshot(optimizer, named_parameters, *, scope):
    """Snapshot an optimizer's param groups and per-parameter state by name (L1).

    Every optimizer parameter must have a name in ``named_parameters`` (a module or a
    mapping); hyperparameters stay per group (the scheduler may change ``lr``); state
    tensors (``exp_avg``, ``exp_avg_sq``, ``step``...) get rule-11 identities and, for
    one-element tensors, their ``value``.
    """
    _check_scope(scope)
    names = _parameter_names(named_parameters)
    groups, state = [], {}
    for index, group in enumerate(optimizer.param_groups):
        members = []
        for param in group["params"]:
            if id(param) not in names:
                raise SnapshotError("optimizer_param_unnamed",
                                    f"group {index} holds a parameter with no name")
            members.append(names[id(param)])
            raw = optimizer.state.get(param, {})
            state[names[id(param)]] = {
                key: _json_scalar(value, f"state/{names[id(param)]}/{key}")
                for key, value in sorted(raw.items())
            }
        hyper = {key: _json_scalar(value, f"param_groups/{index}/{key}")
                 for key, value in sorted(group.items()) if key != "params"}
        groups.append({"index": index, "params": members, "hyperparameters": hyper})
    state = {name: entry for name, entry in sorted(state.items()) if entry}
    return {
        "artifact_kind": OPTIMIZER_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": {},
        "units": {"step": "optimizer steps recorded by the optimizer"},
        "status": "passed",
        "optimizer_class": type(optimizer).__name__,
        "param_groups": groups,
        "state": state,
        "issues": [],
    }


def _optimizer_ok(snap):
    return (
        type(snap) is dict
        and snap.get("artifact_kind") == OPTIMIZER_KIND
        and snap.get("schema_version") == SCHEMA_VERSION
        and snap.get("status") == "passed"
        and snap.get("scope") in SCOPES
        and type(snap.get("param_groups")) is list
        and type(snap.get("state")) is dict
    )


def compare_optimizer_state(before, after):
    """Compare two optimizer snapshots: per-parameter state vs group hyperparameters (L1).

    ``state_identical`` covers every per-parameter state entry (moments and step counters);
    hyperparameter changes (e.g. ``lr`` advanced by the scheduler) are listed separately,
    because E07 allows them while forbidding state changes at a skipped step.
    """
    issues = []
    for label, snap in (("before", before), ("after", after)):
        if not _optimizer_ok(snap):
            issues.append(_issue("input_invalid", label, "not a passed optimizer snapshot"))
    if not issues and before["scope"] != after["scope"]:
        issues.append(_issue("scope_mismatch", "scope", "snapshots come from different scopes"))
    report = {
        "artifact_kind": OPTIMIZER_COMPARISON_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": None,
        "producer": _producer(),
        "input_hashes": {"before": _try_hash(before), "after": _try_hash(after)},
        "units": {},
        "status": "blocked",
        "state_identical": None,
        "changed_state": None,
        "state_only_in_before": None,
        "state_only_in_after": None,
        "param_membership_identical": None,
        "hyperparameters_changed": None,
        "issues": issues,
    }
    if issues:
        return report
    sa, sb = before["state"], after["state"]
    common = sorted(set(sa) & set(sb))
    changed = [n for n in common if sa[n] != sb[n]]
    membership = [g["params"] for g in before["param_groups"]] == [
        g["params"] for g in after["param_groups"]
    ]
    hyper = []
    for ga, gb in zip(before["param_groups"], after["param_groups"], strict=False):
        ha, hb = ga["hyperparameters"], gb["hyperparameters"]
        for key in sorted(set(ha) | set(hb)):
            if ha.get(key) != hb.get(key):
                hyper.append({"group": ga["index"], "key": key, "before": ha.get(key),
                              "after": hb.get(key)})
    only_before, only_after = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
    report.update(
        scope=before["scope"],
        status="passed",
        state_identical=not (changed or only_before or only_after),
        changed_state=changed,
        state_only_in_before=only_before,
        state_only_in_after=only_after,
        param_membership_identical=membership
        and len(before["param_groups"]) == len(after["param_groups"]),
        hyperparameters_changed=hyper,
    )
    return report
