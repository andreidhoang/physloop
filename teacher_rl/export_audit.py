"""Torch-free key-set and byte audit of S4 checkpoint exports (build-spec A10; E02, E03).

Serves S4 engineering preparation (S4.1-S4.4, S4.6 [C]; G9, G7) under D-024. Agent-authored
evaluator engineering (D-018): it reports facts about release, AlpaGym/Cosmos-RL export and
inference-converted checkpoints so that a zero-update or post-training export can be audited
before any parity or learning claim. Tickets: E02 (conversion/export parity, zero-update arm),
E03 (buffers and key sets), R3 findings 4, 9 and 10.

Upstream facts re-derived here (L2: checked against pinned source text, untested on a host):

- Export key mapping = recipes ``ExpertModelWeightMapper.policy_map_local_key_to_hf_key``
  (alpamayo-recipes 1d99fc5, ``recipes/alpamayo1_x_rl/models/expert_model/weight_mapper.py``
  :40-59) followed by Cosmos-RL ``HFModelWeightMapper.policy_map_local_key_to_hf_key``
  (cosmos-rl d2a2c57, ``cosmos_rl/policy/model/hf_models/weight_mapper.py``:353-369) with
  ``clear_weight_name`` (``cosmos_rl/utils/util.py``:369-370). The Cosmos tail depends on
  ``is_vlm`` (whether the mapper config carries ``vision_config``). On the AlpaGym path the
  config is ``AutoConfig.from_pretrained(vlm_name_or_path)`` (recipes
  ``reasoning_vla/config.py``:85-125), INFERRED to be a VLM config, so ``is_vlm`` is true;
  the value is therefore a required input (``hf_config_is_vlm``), never assumed.
- Inference key mapping = AlpaGym ``to_inference_weight_key`` (alpagym 972d160,
  ``packages/policies/alpamayo_r1/scripts/convert_alpagym_checkpoint_to_inference.py``:89-101).
  The converter renames tensors shard by shard and saves with ``{"format": "pt"}`` metadata.
- The Cosmos export iterates ``named_parameters`` only (no buffers), casts to the export dtype,
  and writes greedy chunks of at most ``4 * 1024**3`` bytes named ``{idx:05d}.safetensors`` plus
  ``model.safetensors.index.json`` (cosmos-rl ``llm_trainer.py``:488-501, 539-573, 734-742);
  a chunk exceeds the limit only when it holds one tensor larger than the limit.
- AlpaGym converts a release checkpoint by symlinking its shards (R3 section 2), so the Gym
  bundle's tensors equal the release's. The no-follow manifest reader refuses symlinked
  shards by design, so the bundle is evidenced by an ``s4_bundle_links`` record
  (``scripts/s4_snapshot_params.py links``: link targets read without following, their
  physical resolution, plus the release file hashes) that ``check_zero_update_export``
  re-binds to its ``release``; a byte-copied bundle may instead be given as an ordinary
  tensor manifest. A link target must be canonical (``link_target_problem``): a lexical
  ``..`` after a named component could pass through a symlinked directory.

The vendored converter and mapper (hash-checked, with their Apache-2.0 LICENSE) are executed
as oracles in ``tests/test_s4_export_audit.py``; Cosmos-RL lines are recorded with SHA-256 in
``tests/fixtures/s4_upstream_export/cosmos_rl_references.json``.

Verification level per function (build-spec section 1):

- ``clear_weight_name``, ``policy_name_to_export_key``, ``export_key_to_inference_key`` - L2
  (pure; equal to executed pinned upstream source on enumerated names).
- ``tensor_table``, ``check_export_shards``, ``audit_export``, ``check_zero_update_export`` -
  L1 logic over manifests with literal known answers; their application to an actual export
  is L3 (``qualify_on_host``). A synthetic pass qualifies the auditor, not an export.

Inputs are ``gate.tensor_manifest`` artifacts (``tensor_manifest``, schema 1) or
``teacher_rl.params`` snapshots (``s4_param_snapshot``, schema 1); both carry per-tensor
``dtype``/``shape``/``byte_length``/``sha256`` under build-spec section 2 rule 11, and their
``tensor_set_sha256`` is recomputed here, so a tampered input is rejected.

Agent technical defaults (D-018): ``EXPORT_SHARD_MAX_BYTES`` mirrors the pinned upstream
constant; the policy-wrapper prefixes ``expert_model.``/``reasoning_vla.`` mirror the pinned
mapper. Neither is an owner value.

This module never imports torch or safetensors, never reads or writes files, never executes
model or upstream code, never downloads, never decides a parity tolerance and never reports
E02 or any learning result as passed: action-level parity (and cast-tensor parity when any
tensor is cast) stays ``qualify_on_host``, so the best E02 status is ``qualify_on_host``.
"""

import hashlib
import posixpath
import re
from pathlib import Path

from gate.s2_identity import identity_hash

ARTIFACT_KIND = "s4_export_audit"
E02_KIND = "s4_e02_zero_update_check"
SHARDS_KIND = "s4_export_shard_check"
SCHEMA_VERSION = 1
SCOPES = ("synthetic", "public_source_fixture", "actual")
INPUT_KINDS = {"tensor_manifest": 1, "s4_param_snapshot": 1}
DTYPE_ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}
EXPERT_PREFIXES = (
    "expert.",
    "action_in_proj.",
    "action_out_proj.",
    "expert_hist_traj_tokenizer.",
    "diffusion.",
)
POLICY_WRAPPER_PREFIXES = ("expert_model.", "reasoning_vla.")
EXPORT_SHARD_MAX_BYTES = 4 * 1024**3
EXPORT_SHARD_NAME = re.compile(r"(\d{5})\.safetensors")
EXPORT_INDEX_NAME = "model.safetensors.index.json"
CONVERTER_METADATA = {"format": "pt"}
UNITS = {"byte_length": "bytes", "tensor_bytes": "bytes", "max_shard_bytes": "bytes"}
IS_VLM_SOURCE = (
    "cosmos-rl d2a2c57 cosmos_rl/policy/model/hf_models/weight_mapper.py:64,353-369; "
    "alpamayo-recipes 1d99fc5 recipes/alpamayo1_x_rl/models/reasoning_vla/config.py:85-125"
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ENTRY_FIELDS = ("dtype", "shape", "byte_length", "sha256")
_SNAPSHOT_KINDS = ("parameter", "buffer", "state")
LINKS_KIND = "s4_bundle_links"
LINKS_SCHEMA_VERSION = 1
_LINK_KINDS = ("symlink", "regular_file", "missing", "other")
_DEFERRED = ("qualify_on_host", "not_run")


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _producer():
    return {"module": "teacher_rl.export_audit", "source_sha256": _source_sha256()}


def _issue(code, path, message, severity="error"):
    return {"code": code, "path": path, "severity": severity, "evidence_refs": [],
            "message": message}


def _sorted_issues(issues):
    return sorted(issues, key=lambda i: (i["path"], i["code"], i["message"]))


def _status(issues):
    return "blocked" if any(i["severity"] == "error" for i in issues) else "passed"


def _check_flag(value):
    if type(value) is not bool:
        raise ValueError("hf_config_is_vlm must be True or False (an L2 fact recorded on host)")


def _check_name(name):
    if type(name) is not str or not name:
        raise ValueError("Tensor names must be nonempty strings")


# ---------------------------------------------------------------------------
# Key mappings (L2)
# ---------------------------------------------------------------------------


def _clear(name):
    return name.replace("._orig_mod", "").replace("._checkpoint_wrapped_module", "")


def clear_weight_name(name):
    """Cosmos-RL ``clear_weight_name``: drop compile/checkpoint wrapper segments (L2)."""
    _check_name(name)
    return _clear(name)


def policy_name_to_export_key(name, *, hf_config_is_vlm):
    """Map a policy-side parameter name to its Cosmos export key (L2).

    Reproduces ``ExpertModelWeightMapper.policy_map_local_key_to_hf_key`` then the
    ``HFModelWeightMapper`` tail with no reverse conversion mapping. ``str.replace(..., 1)``
    replaces the first occurrence anywhere, exactly as upstream does.
    """
    _check_flag(hf_config_is_vlm)
    name = clear_weight_name(name)
    for wrapper in POLICY_WRAPPER_PREFIXES:
        if name.startswith(wrapper):
            name = name[len(wrapper):]
            break
    if name.startswith(EXPERT_PREFIXES):
        return name
    name = name.removeprefix("vlm.")
    name = name.replace("model.language_model.", "model.", 1)
    name = name.replace("model.visual.", "visual.", 1)
    name = _clear(name)  # the HFModelWeightMapper tail clears again
    if not hf_config_is_vlm and name != "lm_head.weight" and not name.startswith("model."):
        name = "model." + name
    return name


def export_key_to_inference_key(key):
    """Map a Cosmos export key to its ``alpamayo1_5`` inference key (L2)."""
    _check_name(key)
    if key == "lm_head.weight":
        return "vlm.lm_head.weight"
    if key.startswith("visual."):
        return "vlm.model.visual." + key[len("visual."):]
    if key.startswith("model."):
        return "vlm.model.language_model." + key[len("model."):]
    return key


def policy_name_to_release_key(name):
    """Strip compile/checkpoint wrappers and one policy-wrapper prefix (L2)."""
    name = clear_weight_name(name)
    for wrapper in POLICY_WRAPPER_PREFIXES:
        if name.startswith(wrapper):
            return name[len(wrapper):]
    return name


# ---------------------------------------------------------------------------
# Input validation (L1)
# ---------------------------------------------------------------------------


def _entry_ok(entry, kind):
    if type(entry) is not dict or not set(_ENTRY_FIELDS) <= set(entry):
        return False
    dtype, shape, length, digest = (entry[k] for k in _ENTRY_FIELDS)
    if dtype not in DTYPE_ITEMSIZE or type(shape) is not list:
        return False
    if not all(type(d) is int and d >= 0 for d in shape):
        return False
    numel = 1
    for dim in shape:
        numel *= dim
    if type(length) is not int or length != numel * DTYPE_ITEMSIZE[dtype]:
        return False
    if type(digest) is not str or _HEX64.fullmatch(digest) is None:
        return False
    return kind != "s4_param_snapshot" or entry.get("kind") in _SNAPSHOT_KINDS


def tensor_table(artifact):
    """Return ``(table, problem)`` for a manifest or snapshot artifact (L1).

    ``table`` maps tensor name -> ``{dtype, shape, byte_length, sha256, kind, file}``
    (``kind`` and ``file`` are ``None`` when the artifact does not record them). ``problem``
    is ``None`` or a message; any problem leaves ``table`` ``None`` (fail closed).
    """
    if type(artifact) is not dict:
        return None, "artifact is not an object"
    kind = artifact.get("artifact_kind")
    if kind not in INPUT_KINDS or artifact.get("schema_version") != INPUT_KINDS[kind]:
        return None, f"unsupported artifact {kind!r} schema {artifact.get('schema_version')!r}"
    if type(artifact.get("schema_version")) is not int:
        return None, "schema_version must be an integer"
    if artifact.get("scope") not in SCOPES:
        return None, "scope is missing or unknown"
    if artifact.get("status") != "passed":
        return None, "artifact status is not passed"
    tensors = artifact.get("tensors")
    if type(tensors) is not dict or not tensors:
        return None, "tensors must be a nonempty object"
    for name, entry in tensors.items():
        if type(name) is not str or not name or not _entry_ok(entry, kind):
            return None, f"tensor entry {name!r} is malformed"
    try:
        recomputed = identity_hash(
            {n: {k: e[k] for k in _ENTRY_FIELDS} for n, e in tensors.items()}
        )
    except ValueError:
        return None, "tensor entries are not JSON-native"
    if recomputed != artifact.get("tensor_set_sha256"):
        return None, "tensor_set_sha256 does not match the tensor entries"
    table = {
        name: {
            **{k: entry[k] for k in _ENTRY_FIELDS},
            "kind": entry.get("kind") if kind == "s4_param_snapshot" else None,
            "file": entry.get("file") if kind == "tensor_manifest" else None,
        }
        for name, entry in tensors.items()
    }
    return table, None


def _load_inputs(named, required=()):
    """Validate named artifacts; return (tables, scope, issues, input_hashes).

    A ``required`` label given as None is ``input_invalid`` (never a later KeyError).
    """
    tables, issues, hashes, scopes = {}, [], {}, set()
    for label, artifact in named.items():
        if artifact is None:
            hashes[label] = None
            if label in required:
                issues.append(_issue("input_invalid", label, f"the {label} artifact is required"))
            continue
        try:
            hashes[label] = identity_hash(artifact)
        except (ValueError, TypeError):
            hashes[label] = None
        table, problem = tensor_table(artifact)
        if problem:
            issues.append(_issue("input_invalid", label, problem))
            continue
        tables[label] = table
        scopes.add(artifact["scope"])
    if not issues and len(scopes) > 1:
        issues.append(_issue("scope_mismatch", "scope", f"inputs mix scopes {sorted(scopes)}"))
    scope = scopes.pop() if len(scopes) == 1 and not issues else None
    return tables, scope, issues, hashes


# ---------------------------------------------------------------------------
# Shard layout (L1 logic over L2 facts)
# ---------------------------------------------------------------------------


def check_export_shards(export_manifest, *, max_shard_bytes=EXPORT_SHARD_MAX_BYTES):
    """Check a Cosmos export's shard layout against the pinned writer (L1).

    Requires a sharded ``tensor_manifest`` with an index; shard names must be
    ``00000.safetensors`` ... without gaps; a shard holding more than one tensor may not
    exceed ``max_shard_bytes``; a single tensor larger than the limit alone in its shard is
    upstream behaviour (warning). ``max_shard_bytes`` defaults to the pinned 4 GiB.
    """
    if type(max_shard_bytes) is not int or max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be a positive integer")
    issues = []
    table, problem = tensor_table(export_manifest)
    result = {
        "artifact_kind": SHARDS_KIND,
        "schema_version": SCHEMA_VERSION,
        "producer": _producer(),
        "units": dict(UNITS),
        "max_shard_bytes": max_shard_bytes,
        "status": "blocked",
        "files": [],
        "issues": issues,
    }
    if problem or export_manifest.get("artifact_kind") != "tensor_manifest":
        issues.append(_issue("input_invalid", "export", problem or "export must be a manifest"))
        return result
    if export_manifest.get("index") is None:
        issues.append(_issue("export_index_missing", "index",
                             f"Cosmos exports write {EXPORT_INDEX_NAME}"))
        return result
    index_name = export_manifest["index"].get("name")
    if index_name != EXPORT_INDEX_NAME:
        issues.append(_issue("export_index_name_unexpected", "index",
                             f"index is {index_name!r}, expected {EXPORT_INDEX_NAME!r}"))
    shards = {}
    for name, entry in table.items():
        shards.setdefault(entry["file"], []).append((name, entry["byte_length"]))
    numbers = []
    for shard in sorted(shards):
        match = EXPORT_SHARD_NAME.fullmatch(shard)
        if match is None:
            issues.append(_issue("export_shard_name_unexpected", f"files/{shard}",
                                 "Cosmos export shards are named {idx:05d}.safetensors"))
        else:
            numbers.append(int(match.group(1)))
        size = sum(length for _, length in shards[shard])
        count = len(shards[shard])
        result["files"].append({"name": shard, "tensor_bytes": size, "tensor_count": count})
        if size > max_shard_bytes:
            if count == 1:
                issues.append(_issue("export_single_tensor_over_limit", f"files/{shard}",
                                     f"one tensor of {size} bytes exceeds {max_shard_bytes}",
                                     severity="warning"))
            else:
                issues.append(_issue("export_shard_over_limit", f"files/{shard}",
                                     f"{count} tensors total {size} bytes > {max_shard_bytes}"))
    if numbers and sorted(numbers) != list(range(len(numbers))):
        issues.append(_issue("export_shard_numbering_gap", "files",
                             f"shard numbers {sorted(numbers)} are not 0..{len(numbers) - 1}"))
    result["issues"] = _sorted_issues(issues)
    result["status"] = _status(issues)
    return result


# ---------------------------------------------------------------------------
# Export audit
# ---------------------------------------------------------------------------


def _classify_missing(release_name, snapshot):
    if snapshot is None:
        return "unclassified"
    entry = snapshot.get(release_name)
    return entry["kind"] if entry is not None else "not_in_snapshot"


def _snapshot_by_release_name(table):
    mapped, collisions = {}, []
    for name, entry in sorted(table.items()):
        release_name = policy_name_to_release_key(name)
        if release_name in mapped:
            collisions.append(release_name)
        mapped[release_name] = entry
    return mapped, collisions


def _converter_section(export_table, inference_table, export_manifest, inference_manifest):
    issues = []
    section = {"status": "not_run", "missing": [], "unexpected": [], "changed": []}
    if inference_table is None:
        return section, issues
    expected = {export_key_to_inference_key(k): k for k in export_table}
    section["missing"] = sorted(set(expected) - set(inference_table))
    section["unexpected"] = sorted(set(inference_table) - set(expected))
    if section["missing"] or section["unexpected"]:
        issues.append(_issue(
            "converter_key_mismatch", "converter",
            f"{len(section['missing'])} missing, {len(section['unexpected'])} unexpected keys",
        ))
    shards_moved = []
    for inf_key in sorted(set(expected) & set(inference_table)):
        source = export_table[expected[inf_key]]
        target = inference_table[inf_key]
        if any(source[k] != target[k] for k in _ENTRY_FIELDS):
            section["changed"].append(inf_key)
        if source["file"] != target["file"]:
            shards_moved.append(inf_key)
    if section["changed"]:
        issues.append(_issue("converter_changed_tensor", "converter",
                             "the converter must only rename: dtype, shape or bytes changed "
                             f"for {len(section['changed'])} tensors"))
    if shards_moved:
        issues.append(_issue("converter_shard_layout_changed", "converter",
                             f"{len(shards_moved)} tensors are in a different shard file",
                             severity="warning"))
    for entry in inference_manifest.get("files", []):
        if type(entry) is dict and entry.get("metadata") != CONVERTER_METADATA:
            issues.append(_issue("converter_metadata_unexpected", f"files/{entry.get('name')}",
                                 f"pinned converter writes {CONVERTER_METADATA}",
                                 severity="warning"))
    section["status"] = _status(issues)
    return section, issues


def audit_export(*, release, export, hf_config_is_vlm, inference=None, policy_snapshot=None,
                 max_shard_bytes=EXPORT_SHARD_MAX_BYTES):
    """Audit release -> Cosmos export -> inference key sets, dtypes, bytes and shards (L1/L2).

    ``release`` and ``export`` are required artifacts; ``inference`` (the converter output)
    and ``policy_snapshot`` (a ``teacher_rl.params`` snapshot of the policy model, used to
    classify release tensors absent from the export as parameters or buffers) are optional
    but their absence is reported: without a snapshot every absent tensor is
    ``absent_from_export_unclassified`` (blocking). ``hf_config_is_vlm`` is required.

    Facts, not verdicts: same-dtype byte changes are listed (expected after training,
    forbidden only by ``check_zero_update_export``); casts are listed for the parity ladder.
    """
    _check_flag(hf_config_is_vlm)
    named = {"release": release, "export": export, "inference": inference,
             "policy_snapshot": policy_snapshot}
    tables, scope, issues, hashes = _load_inputs(named, required=("release", "export"))
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "producer": _producer(),
        "input_hashes": hashes,
        "units": dict(UNITS),
        "level": "L2",
        "status": "blocked",
        "assumptions": {"hf_config_is_vlm": {"value": hf_config_is_vlm, "level": "L2",
                                             "source": IS_VLM_SOURCE}},
        "key_sets": None,
        "tensors": None,
        "buffers_absent_from_export": None,
        "converter": None,
        "shards": None,
        "issues": issues,
    }
    if issues:
        report["issues"] = _sorted_issues(issues)
        return report

    rel, exp = tables["release"], tables["export"]
    snapshot = None
    if "policy_snapshot" in tables:
        snapshot, collisions = _snapshot_by_release_name(tables["policy_snapshot"])
        for name in collisions:
            issues.append(_issue("snapshot_name_collision", f"policy_snapshot/{name}",
                                 "two snapshot names map to one release name"))

    to_export, owners = {}, {}
    for name in sorted(rel):
        key = policy_name_to_export_key(name, hf_config_is_vlm=hf_config_is_vlm)
        if key in owners:
            issues.append(_issue("export_key_collision", f"release/{name}",
                                 f"{owners[key]!r} and {name!r} map to export key {key!r}"))
        owners[key] = name
        to_export[name] = key
        back = export_key_to_inference_key(key)
        if back != name:
            issues.append(_issue("mapping_not_invertible", f"release/{name}",
                                 f"export key {key!r} converts back to {back!r}"))

    matched, missing = [], []
    for name in sorted(rel):
        key = to_export[name]
        if key in exp and export_key_to_inference_key(key) == name:
            matched.append((name, key))
        else:
            missing.append({"name": name, "classification": _classify_missing(name, snapshot)})
    paired_exports = {key for _, key in matched}
    unexpected = sorted(set(exp) - paired_exports)
    if unexpected:
        issues.append(_issue("unexpected_in_export", "export",
                             f"{len(unexpected)} export keys have no release counterpart"))
    buffers_absent = None if snapshot is None else []
    for item in missing:
        cls, path = item["classification"], f"release/{item['name']}"
        if cls == "buffer":
            buffers_absent.append(item["name"])
            issues.append(_issue("buffer_absent_from_export", path,
                                 "export iterates named_parameters only (R3 finding 9)",
                                 severity="warning"))
        elif cls == "unclassified":
            issues.append(_issue("absent_from_export_unclassified", path,
                                 "supply a policy snapshot to classify parameter vs buffer"))
        elif cls == "not_in_snapshot":
            issues.append(_issue("release_tensor_absent_from_export", path,
                                 "absent from the export and from the policy snapshot"))
        else:
            issues.append(_issue("parameter_absent_from_export", path,
                                 f"{cls} is in the policy but not exported"))
    if snapshot is not None:
        for name, entry in sorted(snapshot.items()):
            if entry["kind"] == "parameter" and name not in rel:
                issues.append(_issue("snapshot_parameter_not_in_release",
                                     f"policy_snapshot/{name}",
                                     "policy parameter has no release tensor"))

    identical, casts, changed, reshaped = [], [], [], []
    for name, key in matched:
        a, b = rel[name], exp[key]
        if a["shape"] != b["shape"]:
            reshaped.append(name)
        elif a["dtype"] != b["dtype"]:
            casts.append({"name": name, "export_key": key, "release_dtype": a["dtype"],
                          "export_dtype": b["dtype"]})
        elif a["sha256"] != b["sha256"]:
            changed.append(name)
        else:
            identical.append(name)
    if reshaped:
        issues.append(_issue("shape_changed", "export",
                             f"{len(reshaped)} tensors changed shape between release and export"))
    if casts:
        issues.append(_issue("export_dtype_cast", "export",
                             f"{len(casts)} tensors were cast (R3 findings 4 and 9)",
                             severity="warning"))

    converter, converter_issues = _converter_section(
        exp, tables.get("inference"), export, inference
    )
    issues.extend(converter_issues)
    if inference is None:
        issues.append(_issue("inference_manifest_not_supplied", "inference",
                             "converter output not audited (converter.status not_run)",
                             severity="warning"))
    shards = check_export_shards(export, max_shard_bytes=max_shard_bytes)
    issues.extend(dict(i, path=f"shards/{i['path']}") for i in shards["issues"])

    report.update(
        key_sets={
            "release_count": len(rel),
            "export_count": len(exp),
            "inference_count": len(tables["inference"]) if "inference" in tables else None,
            "matched": len(matched),
            "missing_from_export": missing,
            "unexpected_in_export": unexpected,
        },
        tensors={"identical": identical, "dtype_casts": casts, "bytes_changed": changed,
                 "shape_changed": reshaped},
        buffers_absent_from_export=buffers_absent,
        converter=converter,
        shards={"status": shards["status"], "files": shards["files"]},
        issues=_sorted_issues(issues),
        status=_status(issues),
    )
    return report


# ---------------------------------------------------------------------------
# E02 zero-update export requirements
# ---------------------------------------------------------------------------


def _identities(table):
    return {name: {k: entry[k] for k in _ENTRY_FIELDS} for name, entry in table.items()}


def _links_record_problem(record):
    """Return a message when an ``s4_bundle_links`` record is malformed, else ``None``."""
    keys = ("scope", "status", "bundle_root", "release_root", "release_tensor_set_sha256",
            "release_files", "entries")
    if type(record) is not dict or not set(keys) <= set(record):
        return "link record is missing required fields"
    if record.get("schema_version") != LINKS_SCHEMA_VERSION:
        return f"unsupported link record schema {record.get('schema_version')!r}"
    if record["status"] != "passed":
        codes = sorted({i.get("code") for i in record.get("issues") or [] if type(i) is dict
                        and type(i.get("code")) is str})
        return f"link record status is not passed (issues: {', '.join(codes) or 'none'})"
    for root in ("bundle_root", "release_root"):
        value = record[root]
        if type(value) is not str or not posixpath.isabs(value) or ".." in value.split("/"):
            return f"{root} must be an absolute path without traversal"
    files, entries = record["release_files"], record["entries"]
    if type(files) is not dict or not files or type(entries) is not dict:
        return "release_files and entries must be objects"
    if set(entries) != set(files):
        return "entries must cover exactly the release files"
    for name, entry in entries.items():
        if (
            type(name) is not str or not name or "/" in name
            or type(files[name]) is not str or _HEX64.fullmatch(files[name]) is None
            or type(entry) is not dict or entry.get("kind") not in _LINK_KINDS
        ):
            return f"link entry {name!r} is malformed"
        if entry["kind"] == "symlink" and (type(entry.get("target")) is not str
                                            or not entry["target"]):
            return f"symlink entry {name!r} has no target"
    return None


def link_target_problem(target):
    """Return why a symlink target is not canonical, else ``None`` (L1; A10R).

    Allowed: an absolute path, or a relative path whose ``..`` components form only a
    leading ascent run (as ``os.path.relpath`` writes). Refused: empty or ``.`` components
    and any ``..`` after a named component, which lexical normalization would resolve
    without seeing whether that component is a symlinked directory.
    """
    if type(target) is not str or not target or "\x00" in target:
        return "link target must be a nonempty string without NUL"
    absolute = target.startswith("/")
    ascending = not absolute
    for part in (target[1:] if absolute else target).split("/"):
        if part in ("", "."):
            return f"link target {target!r} has an empty or '.' component"
        if part != "..":
            ascending = False
        elif not ascending:
            return f"link target {target!r} has '..' after a named component"
    return None


def _check_bundle_links(record, release, scope):
    """Bind a link record to ``release``; return (requirement status, issues)."""
    problem = _links_record_problem(record)
    if problem is None and record["scope"] != scope:
        problem = f"link record scope {record['scope']!r} differs from the inputs"
    if problem:
        return "blocked", [_issue("input_invalid", "gym_bundle", problem)]
    hashes = release.get("input_hashes") if release.get("artifact_kind") == "tensor_manifest" \
        else None
    files = hashes.get("files") if type(hashes) is dict else None
    if (
        files != record["release_files"]
        or record["release_tensor_set_sha256"] != release.get("tensor_set_sha256")
    ):
        return "blocked", [_issue("gym_bundle_links_not_for_release", "gym_bundle",
                                  "the link record was made against other release bytes "
                                  "(release must be the tensor manifest it hashed)")]
    issues = []
    for name, entry in sorted(record["entries"].items()):
        expected = posixpath.join(record["release_root"], name)
        if entry["kind"] == "symlink":
            target = entry["target"]
            joined = target if posixpath.isabs(target) else posixpath.join(
                record["bundle_root"], target)
            ok = (link_target_problem(target) is None
                  and posixpath.normpath(joined) == expected
                  and entry.get("resolved_target") == expected)
        else:
            ok = entry["kind"] == "regular_file" and entry.get("sha256") == files[name]
        if not ok:
            issues.append(_issue("gym_bundle_differs_from_release", f"gym_bundle/{name}",
                                 f"{entry['kind']} is neither a canonical link physically "
                                 f"resolving to {expected!r} nor a byte copy of the release "
                                 "file"))
    return ("blocked" if issues else "passed"), issues


def check_zero_update_export(*, release, export, inference, optimizer_steps_applied,
                             hf_config_is_vlm, policy_snapshot=None, gym_bundle=None,
                             max_shard_bytes=EXPORT_SHARD_MAX_BYTES):
    """Check the engineering requirements of an E02 zero-update export (L1/L2).

    ``optimizer_steps_applied`` is the exact in-process count (H4 capture); ``None`` blocks
    (``optimizer_step_count_unrecorded``) and any positive count blocks (``not_zero_update``).
    Every same-dtype tensor must be bitwise identical to the release; with no same-dtype
    tensor compared the requirement is ``not_applicable`` (never ``passed``). Cast tensors
    (``requires_parity_ladder``; ``teacher_rl.parity`` cross-dtype mode) and action-level
    behaviour need the parity ladder on a host (``qualify_on_host``). ``gym_bundle`` is an
    ``s4_bundle_links`` record (symlinked bundle) or a tensor manifest (byte copy). Missing
    inference or Gym-bundle inputs block by name. Status: ``blocked`` on any error issue, else
    ``qualify_on_host`` when any requirement is ``qualify_on_host``/``not_run`` - always, since
    action-level parity is host-only. This function never returns ``passed`` and is never a
    scientific E02 pass.
    """
    if optimizer_steps_applied is not None and (
        type(optimizer_steps_applied) is not int or optimizer_steps_applied < 0
    ):
        raise ValueError("optimizer_steps_applied must be None or a non-negative integer")
    audit = audit_export(release=release, export=export, inference=inference,
                         policy_snapshot=policy_snapshot, hf_config_is_vlm=hf_config_is_vlm,
                         max_shard_bytes=max_shard_bytes)
    issues, req = [], {}
    if optimizer_steps_applied is None:
        req["optimizer_steps_applied_zero"] = "blocked"
        issues.append(_issue("optimizer_step_count_unrecorded", "optimizer_steps_applied",
                             "the exact optimizer.step count must be captured in process"))
    elif optimizer_steps_applied > 0:
        req["optimizer_steps_applied_zero"] = "blocked"
        issues.append(_issue("not_zero_update", "optimizer_steps_applied",
                             f"{optimizer_steps_applied} optimizer steps were applied"))
    else:
        req["optimizer_steps_applied_zero"] = "passed"

    if gym_bundle is None:
        req["gym_bundle_matches_release"] = "not_run"
        issues.append(_issue("gym_bundle_manifest_missing", "gym_bundle",
                             "hash the converted Gym bundle (its shards are release symlinks)"))
    elif type(gym_bundle) is dict and gym_bundle.get("artifact_kind") == LINKS_KIND:
        if audit["scope"] is None:
            req["gym_bundle_matches_release"] = "blocked"
        else:
            req["gym_bundle_matches_release"], link_issues = _check_bundle_links(
                gym_bundle, release, audit["scope"])
            issues.extend(link_issues)
    else:
        tables, _, problems, _ = _load_inputs({"release": release, "gym_bundle": gym_bundle},
                                              required=("release", "gym_bundle"))
        if problems:
            req["gym_bundle_matches_release"] = "blocked"
            issues.extend(problems)
        elif _identities(tables["gym_bundle"]) != _identities(tables["release"]):
            req["gym_bundle_matches_release"] = "blocked"
            issues.append(_issue("gym_bundle_differs_from_release", "gym_bundle",
                                 "converted bundle tensors must equal the release tensors"))
        else:
            req["gym_bundle_matches_release"] = "passed"

    req["export_audit"] = audit["status"]
    if audit["status"] != "passed":
        issues.append(_issue("export_audit_blocked", "export_audit",
                             "the export audit has blocking issues"))
    tensors = audit["tensors"]
    compared = None
    if tensors is None:
        req["same_dtype_tensors_bitwise_identical"] = "blocked"
    elif tensors["bytes_changed"] or tensors["shape_changed"]:
        compared = len(tensors["identical"]) + len(tensors["bytes_changed"])
        req["same_dtype_tensors_bitwise_identical"] = "blocked"
        for name in tensors["bytes_changed"]:
            issues.append(_issue("zero_update_bytes_changed", f"export/{name}",
                                 "same dtype but different bytes after zero updates"))
    else:
        compared = len(tensors["identical"])
        # Absence is never a pass: with every tensor cast nothing was compared bitwise.
        req["same_dtype_tensors_bitwise_identical"] = "passed" if compared else "not_applicable"

    if inference is None:
        req["inference_conversion"] = "not_run"
        issues.append(_issue("inference_manifest_missing", "inference",
                             "hash the converter output for the zero-update export"))
    else:
        req["inference_conversion"] = (audit["converter"] or {}).get("status", "blocked")
    casts = [] if tensors is None else [c["name"] for c in tensors["dtype_casts"]]
    req["cast_tensor_parity"] = "qualify_on_host" if casts else "not_applicable"
    req["action_level_parity"] = "qualify_on_host"
    if any(i["severity"] == "error" for i in issues) or audit["status"] != "passed":
        status = "blocked"
    elif any(value in _DEFERRED for value in req.values()):
        status = "qualify_on_host"
    else:  # unreachable while action_level_parity is host-only; kept fail-closed
        status = "passed"
    return {
        "artifact_kind": E02_KIND,
        "schema_version": SCHEMA_VERSION,
        "scope": audit["scope"],
        "producer": _producer(),
        "input_hashes": dict(audit["input_hashes"],
                             gym_bundle=None if gym_bundle is None else identity_hash(gym_bundle)),
        "units": {"optimizer_steps_applied": "optimizer.step calls",
                  "same_dtype_tensors_compared": "tensors"},
        "level": "L2",
        "status": status,
        "optimizer_steps_applied": optimizer_steps_applied,
        "requirements": req,
        "same_dtype_tensors_compared": compared,
        "requires_parity_ladder": casts,
        "buffers_absent_from_export": audit["buffers_absent_from_export"],
        "export_audit": audit,
        "issues": _sorted_issues(issues),
    }
