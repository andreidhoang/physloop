"""Known answers and negative controls for the torch-free S4 export audit (A10; E02, E03).

Oracles are independent of ``teacher_rl.export_audit``:

- ``to_inference_weight_key`` is AST-extracted from the hash-checked vendored AlpaGym
  converter (pinned 972d160) and executed in a builtins-only namespace.
- ``ExpertModelWeightMapper.policy_map_local_key_to_hf_key`` and its ``_EXPERT_PREFIXES``
  are AST-extracted from the vendored recipes mapper (pinned 1d99fc5) and executed on a
  stub parent chain. The Cosmos-RL tail (``clear_weight_name`` and the
  ``HFModelWeightMapper`` branch) is re-derived here from the cited lines recorded with
  their SHA-256 in ``tests/fixtures/s4_upstream_export/cosmos_rl_references.json``.
- Manifests are built from hand-written safetensors bytes by ``gate.tensor_manifest``.

Coverage: mapping known answers (both ``is_vlm`` branches), round trip, missing and
unexpected keys, buffer classification, dtype casts, converter renaming, shard layout,
the E02 zero-update requirements and torch-freedom of the module and both host scripts.
"""

import ast
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from gate.s2_identity import identity_hash
from gate.tensor_manifest import build_manifest
from teacher_rl import export_audit as ea

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "s4_upstream_export"
CONVERTER = ("alpagym", "convert_alpagym_checkpoint_to_inference_972d160e.py.txt")
MAPPER = ("alpamayo-recipes", "expert_weight_mapper_1d99fc50.py.txt")
LICENSES = {
    "alpagym": "85d0bf7032265ecf3647a293362c7ac55930d253d6a78dfcbce245c98eba2a72",
    "alpamayo-recipes": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}


# ---------------------------------------------------------------------------
# Vendored upstream oracles
# ---------------------------------------------------------------------------


def _vendored(repo, name):
    provenance = json.loads((FIXTURES / repo / "provenance.json").read_text())
    (source,) = [s for s in provenance["sources"] if s["file"] == name]
    content = (FIXTURES / repo / name).read_bytes()
    assert hashlib.sha256(content).hexdigest() == source["sha256"], "upstream bytes changed"
    return content, source, provenance


@pytest.mark.parametrize(("repo", "name"), [CONVERTER, MAPPER])
def test_vendored_sources_match_provenance_git_blob_and_license(repo, name):
    content, source, provenance = _vendored(repo, name)
    blob = b"blob " + str(len(content)).encode() + b"\0" + content
    assert hashlib.sha1(blob).hexdigest() == source["git_blob_sha"]
    assert source["commit"] == provenance["commit"]
    assert source["commit"] in source["source_url"] and source["path"] in source["source_url"]
    assert content.startswith(b"# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA")
    assert b"SPDX-License-Identifier: Apache-2.0" in content.splitlines()[1]
    license_bytes = (FIXTURES / repo / provenance["license_file"]).read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == LICENSES[repo]
    assert provenance["license_file_sha256"] == LICENSES[repo]
    assert b"Apache License" in license_bytes and b"Version 2.0, January 2004" in license_bytes


def test_cosmos_references_record_hashes_not_bytes():
    refs = json.loads((FIXTURES / "cosmos_rl_references.json").read_text())
    assert refs["vendored"] is False
    assert refs["commit"] == "d2a2c57c4bd6496482bc42da19a59b4432705eda"
    paths = {item["path"]: item for item in refs["references"]}
    assert set(paths) == {
        "cosmos_rl/utils/util.py",
        "cosmos_rl/policy/model/hf_models/weight_mapper.py",
        "cosmos_rl/policy/model/hf_models/__init__.py",
        "cosmos_rl/policy/trainer/llm_trainer/llm_trainer.py",
    }
    for item in refs["references"] + refs["related"]:
        assert len(item["sha256"]) == 64 and item["lines"]


def _upstream_to_inference_weight_key():
    content, _, _ = _vendored(*CONVERTER)
    tree = ast.parse(content)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    (node,) = [n for n in nodes if n.name == "to_inference_weight_key"]
    namespace = {"__builtins__": {"len": len, "str": str}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), CONVERTER[1], "exec"), namespace)  # noqa: S102
    return namespace["to_inference_weight_key"]


def _cosmos_clear_weight_name(name):
    # cosmos_rl/utils/util.py:369-370 (sha256 6ee7aed2...), re-derived.
    return name.replace("._orig_mod", "").replace("._checkpoint_wrapped_module", "")


class _CosmosHFTail:
    """cosmos_rl/policy/model/hf_models/weight_mapper.py:353-369 (sha256 e52339fe...)."""

    reverse_hf_conversion_mapping = None

    def __init__(self, is_vlm):
        self.is_vlm = is_vlm

    def policy_map_local_key_to_hf_key(self, name):
        name = _cosmos_clear_weight_name(name)
        if not self.is_vlm and name != "lm_head.weight" and not name.startswith("model."):
            name = "model." + name
        return name


def _upstream_expert_mapper(is_vlm):
    content, _, _ = _vendored(*MAPPER)
    tree = ast.parse(content)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert cls.name == "ExpertModelWeightMapper"
    (prefixes,) = [
        n for n in cls.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "_EXPERT_PREFIXES"
    ]
    (method,) = [
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "policy_map_local_key_to_hf_key"
    ]
    method.decorator_list = []

    class ReasoningVLAWeightMapper(_CosmosHFTail):
        pass

    namespace = {
        "__builtins__": {"super": super, "str": str, "len": len},
        "util": SimpleNamespace(clear_weight_name=_cosmos_clear_weight_name),
        "ReasoningVLAWeightMapper": ReasoningVLAWeightMapper,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), MAPPER[1], "exec"), namespace)  # noqa: S102
    expert_mapper = type(
        "ExpertModelWeightMapper",
        (ReasoningVLAWeightMapper,),
        {
            "_EXPERT_PREFIXES": ast.literal_eval(prefixes.value),
            "policy_map_local_key_to_hf_key": namespace["policy_map_local_key_to_hf_key"],
        },
    )
    return expert_mapper(is_vlm), ast.literal_eval(prefixes.value)


POLICY_NAMES = [
    "expert_model.vlm.model.language_model.layers.0.mlp.up_proj.weight",
    "expert_model.vlm.model.language_model.embed_tokens.weight",
    "expert_model.vlm.model.language_model.norm.weight",
    "expert_model.vlm.model.visual.blocks.0.attn.qkv.weight",
    "expert_model.vlm.model.visual.merger.mlp.0.bias",
    "expert_model.vlm.lm_head.weight",
    "expert_model.expert.layers.0.self_attn.q_proj.weight",
    "expert_model.expert.norm.weight",
    "expert_model.action_in_proj.proj.weight",
    "expert_model.action_out_proj.bias",
    "expert_model.diffusion.sigma",
    "expert_model._orig_mod.expert.norm.weight",
    "expert_model.expert.layers.0._checkpoint_wrapped_module.mlp.weight",
    "reasoning_vla.vlm.model.visual.patch_embed.proj.weight",
    "vlm.model.language_model.layers.1.self_attn.k_proj.weight",
    "vlm.model.visual.blocks.1.norm1.weight",
    "vlm.lm_head.weight",
    "expert.layers.1.mlp.down_proj.weight",
    "action_space.accel_mean",
    "action_in_proj.encoder.freqs",
    "model.language_model.layers.0.x",
    "lm_head.weight",
    "other.vlm.thing",
]


@pytest.mark.parametrize("is_vlm", [True, False])
def test_policy_mapping_equals_executed_upstream_mapper(is_vlm):
    upstream, _ = _upstream_expert_mapper(is_vlm)
    for name in POLICY_NAMES:
        expected = upstream.policy_map_local_key_to_hf_key(name)
        assert ea.policy_name_to_export_key(name, hf_config_is_vlm=is_vlm) == expected, name


def test_expert_prefixes_equal_vendored_tuple():
    _, prefixes = _upstream_expert_mapper(True)
    assert ea.EXPERT_PREFIXES == prefixes


def test_inference_mapping_equals_executed_upstream_converter():
    upstream = _upstream_to_inference_weight_key()
    keys = [
        "lm_head.weight",
        "visual.blocks.0.attn.qkv.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.embed_tokens.weight",
        "expert.norm.weight",
        "action_in_proj.proj.weight",
        "action_out_proj.bias",
        "lm_head.bias",
        "vlm.lm_head.weight",
        "visualx.y",
        "modelx.y",
    ]
    for key in keys:
        assert ea.export_key_to_inference_key(key) == upstream(key), key


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("expert_model.vlm.model.language_model.layers.0.mlp.up_proj.weight",
         "model.layers.0.mlp.up_proj.weight"),
        ("expert_model.vlm.model.visual.blocks.0.attn.qkv.weight",
         "visual.blocks.0.attn.qkv.weight"),
        ("expert_model.vlm.lm_head.weight", "lm_head.weight"),
        ("expert_model.expert.layers.0.self_attn.q_proj.weight",
         "expert.layers.0.self_attn.q_proj.weight"),
        ("expert_model._orig_mod.expert.norm.weight", "expert.norm.weight"),
        ("action_space.accel_mean", "action_space.accel_mean"),
    ],
)
def test_policy_mapping_literals(name, expected):
    assert ea.policy_name_to_export_key(name, hf_config_is_vlm=True) == expected


def test_non_vlm_branch_literal_breaks_round_trip():
    exported = ea.policy_name_to_export_key("vlm.model.visual.blocks.0.x", hf_config_is_vlm=False)
    assert exported == "model.visual.blocks.0.x"
    assert ea.export_key_to_inference_key(exported) == "vlm.model.language_model.visual.blocks.0.x"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("lm_head.weight", "vlm.lm_head.weight"),
        ("visual.x", "vlm.model.visual.x"),
        ("model.x", "vlm.model.language_model.x"),
        ("expert.x", "expert.x"),
    ],
)
def test_inference_mapping_literals(key, expected):
    assert ea.export_key_to_inference_key(key) == expected


def test_mapping_rejects_non_bool_flag_and_bad_names():
    with pytest.raises(ValueError, match="hf_config_is_vlm"):
        ea.policy_name_to_export_key("vlm.lm_head.weight", hf_config_is_vlm=None)
    with pytest.raises(ValueError, match="hf_config_is_vlm"):
        ea.policy_name_to_export_key("vlm.lm_head.weight", hf_config_is_vlm=1)
    with pytest.raises(ValueError, match="nonempty"):
        ea.export_key_to_inference_key("")


# ---------------------------------------------------------------------------
# Hand-written safetensors fixtures
# ---------------------------------------------------------------------------

ITEMSIZE = {"F32": 4, "BF16": 2, "I64": 8}


def _st(tensors, metadata=None):
    """Serialize ``{name: (dtype, shape, raw)}`` to safetensors bytes in name order."""
    header, data, offset = {}, b"", 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        data += raw
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(blob)) + blob + data


def _bf16(*values):
    return b"".join(struct.pack("<f", v)[2:] for v in values)


def _f32(*values):
    return struct.pack(f"<{len(values)}f", *values)


# Release (inference naming). Two persistent buffers, one F32 parameter.
RELEASE = {
    "vlm.model.visual.proj.weight": ("BF16", [2], _bf16(1.0, 2.0)),
    "vlm.model.language_model.layers.0.weight": ("BF16", [2], _bf16(3.0, 4.0)),
    "vlm.lm_head.weight": ("BF16", [2], _bf16(5.0, 6.0)),
    "expert.layers.0.weight": ("BF16", [2], _bf16(7.0, 8.0)),
    "action_in_proj.proj.weight": ("BF16", [2], _bf16(9.0, 10.0)),
    "action_in_proj.encoder.freqs": ("F32", [1, 2], _f32(1.0, 100.0)),
    "action_out_proj.bias": ("F32", [2], _f32(0.5, -0.5)),
    "action_space.accel_mean": ("F32", [], _f32(0.25)),
}
BUFFERS = {"action_in_proj.encoder.freqs", "action_space.accel_mean"}
PARAMS = sorted(set(RELEASE) - BUFFERS)


def _export_tensors(release=RELEASE, **changes):
    out = {}
    for name in PARAMS:
        key = ea.policy_name_to_export_key(name, hf_config_is_vlm=True)
        dtype, shape, raw = release[name]
        if dtype == "F32":  # Cosmos casts to param_dtype bfloat16
            raw = b"".join(raw[i + 2 : i + 4] for i in range(0, len(raw), 4))
            dtype = "BF16"
        out[key] = (dtype, shape, raw)
    out.update(changes)
    return {k: v for k, v in out.items() if v is not None}


def _sharded(tensors, shard_of, *, scope="synthetic", metadata=None, total=None):
    shards = {}
    for name, value in tensors.items():
        shards.setdefault(shard_of(name), {})[name] = value
    files = {shard: _st(content, metadata) for shard, content in shards.items()}
    total_size = sum(len(raw) for _, _, raw in tensors.values()) if total is None else total
    index = {"metadata": {"total_size": total_size},
             "weight_map": {n: shard_of(n) for n in sorted(tensors)}}
    files[ea.EXPORT_INDEX_NAME] = json.dumps(index).encode()
    return build_manifest(files, scope=scope, index_name=ea.EXPORT_INDEX_NAME)


def _two_shards(name):
    return "00000.safetensors" if name.startswith(("model.", "visual.", "lm_head")) else (
        "00001.safetensors"
    )


def _release_manifest(release=RELEASE, scope="synthetic"):
    return build_manifest({"model.safetensors": _st(release)}, scope=scope)


def _export_manifest(tensors=None, shard_of=_two_shards, **kw):
    return _sharded(_export_tensors() if tensors is None else tensors, shard_of, **kw)


def _inference_manifest(export_tensors=None, shard_of=_two_shards, **kw):
    tensors = _export_tensors() if export_tensors is None else export_tensors
    renamed = {ea.export_key_to_inference_key(k): v for k, v in tensors.items()}

    def shard(name):
        (original,) = [k for k in tensors if ea.export_key_to_inference_key(k) == name]
        return shard_of(original)

    kw.setdefault("metadata", {"format": "pt"})
    return _sharded(renamed, shard, **kw)


def _snapshot(names_kinds, scope="synthetic"):
    """A hand-built ``s4_param_snapshot`` artifact (schema per teacher_rl.params)."""
    tensors = {}
    for name, kind in names_kinds.items():
        local = name.removeprefix("expert_model.")
        dtype, shape, raw = RELEASE[local]
        tensors[name] = {
            "kind": kind,
            "dtype": dtype,
            "shape": shape,
            "numel": len(raw) // ITEMSIZE[dtype],
            "byte_length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "requires_grad": (not local.startswith("vlm.")) if kind == "parameter" else False,
            "persistent": True if kind == "buffer" else None,
        }
    tensors = dict(sorted(tensors.items()))
    return {
        "artifact_kind": "s4_param_snapshot",
        "schema_version": 1,
        "scope": scope,
        "producer": {"module": "teacher_rl.params", "source_sha256": "0" * 64},
        "input_hashes": {},
        "units": {},
        "hash_convention": "rule 11",
        "status": "passed",
        "source": "module",
        "tensors": tensors,
        "modules": {"": True},
        "tensor_set_sha256": identity_hash(
            {n: {k: e[k] for k in ("dtype", "shape", "byte_length", "sha256")}
             for n, e in tensors.items()}
        ),
        "totals": {},
        "issues": [],
    }


def _full_snapshot():
    kinds = {f"expert_model.{n}": "parameter" for n in PARAMS}
    kinds.update({f"expert_model.{n}": "buffer" for n in BUFFERS})
    return _snapshot(kinds)


def _codes(report, severity=None):
    return sorted(
        {i["code"] for i in report["issues"] if severity is None or i["severity"] == severity}
    )


def _audit(**kw):
    args = {
        "release": _release_manifest(),
        "export": _export_manifest(),
        "inference": _inference_manifest(),
        "policy_snapshot": _full_snapshot(),
        "hf_config_is_vlm": True,
    }
    args.update(kw)
    return ea.audit_export(**args)


# ---------------------------------------------------------------------------
# audit_export
# ---------------------------------------------------------------------------


def test_zero_update_export_audit_known_answer():
    report = _audit()
    assert report["artifact_kind"] == "s4_export_audit" and report["schema_version"] == 1
    assert report["scope"] == "synthetic"
    assert report["status"] == "passed", report["issues"]
    assert _codes(report, "error") == []
    assert _codes(report) == ["buffer_absent_from_export", "export_dtype_cast"]
    keys = report["key_sets"]
    assert (keys["release_count"], keys["export_count"], keys["inference_count"]) == (8, 6, 6)
    assert keys["matched"] == 6
    assert keys["unexpected_in_export"] == []
    assert keys["missing_from_export"] == [
        {"name": "action_in_proj.encoder.freqs", "classification": "buffer"},
        {"name": "action_space.accel_mean", "classification": "buffer"},
    ]
    assert report["buffers_absent_from_export"] == sorted(BUFFERS)
    t = report["tensors"]
    assert t["identical"] == [
        "action_in_proj.proj.weight",
        "expert.layers.0.weight",
        "vlm.lm_head.weight",
        "vlm.model.language_model.layers.0.weight",
        "vlm.model.visual.proj.weight",
    ]
    assert t["dtype_casts"] == [{
        "name": "action_out_proj.bias",
        "export_key": "action_out_proj.bias",
        "release_dtype": "F32",
        "export_dtype": "BF16",
    }]
    assert t["bytes_changed"] == [] and t["shape_changed"] == []
    assert report["converter"]["status"] == "passed"
    assert report["shards"]["status"] == "passed"
    assert report["shards"]["files"] == [
        {"name": "00000.safetensors", "tensor_bytes": 12, "tensor_count": 3},
        {"name": "00001.safetensors", "tensor_bytes": 12, "tensor_count": 3},
    ]
    assert report["assumptions"]["hf_config_is_vlm"]["value"] is True
    assert report["assumptions"]["hf_config_is_vlm"]["level"] == "L2"
    assert set(report["input_hashes"]) == {"release", "export", "inference", "policy_snapshot"}


def test_audit_is_deterministic_and_json_native():
    a, b = _audit(), _audit()
    assert identity_hash(a) == identity_hash(b)


def test_parameter_missing_from_export_is_blocking():
    export = _export_tensors(**{"lm_head.weight": None})
    report = _audit(export=_export_manifest(export), inference=_inference_manifest(export))
    assert report["status"] == "blocked"
    assert "parameter_absent_from_export" in _codes(report, "error")
    assert {"name": "vlm.lm_head.weight", "classification": "parameter"} in (
        report["key_sets"]["missing_from_export"]
    )


def test_unclassified_absence_without_snapshot_fails_closed():
    report = _audit(policy_snapshot=None)
    assert report["status"] == "blocked"
    assert _codes(report, "error") == ["absent_from_export_unclassified"]
    assert report["buffers_absent_from_export"] is None
    assert [m["classification"] for m in report["key_sets"]["missing_from_export"]] == [
        "unclassified", "unclassified",
    ]


def test_unexpected_export_key_is_blocking():
    export = _export_tensors(**{"expert.extra.weight": ("BF16", [1], _bf16(1.0))})
    report = _audit(export=_export_manifest(export), inference=_inference_manifest(export))
    assert "unexpected_in_export" in _codes(report, "error")
    assert report["key_sets"]["unexpected_in_export"] == ["expert.extra.weight"]


def test_non_vlm_mapping_assumption_breaks_round_trip_and_blocks():
    report = _audit(hf_config_is_vlm=False)
    assert report["status"] == "blocked"
    assert "mapping_not_invertible" in _codes(report, "error")
    assert report["assumptions"]["hf_config_is_vlm"]["value"] is False


def test_shape_change_is_blocking_and_same_dtype_byte_change_is_reported():
    export = _export_tensors(**{
        "expert.layers.0.weight": ("BF16", [1, 2], _bf16(7.0, 8.0)),
        "action_in_proj.proj.weight": ("BF16", [2], _bf16(9.0, 11.0)),
    })
    report = _audit(export=_export_manifest(export), inference=_inference_manifest(export))
    assert "shape_changed" in _codes(report, "error")
    assert report["tensors"]["shape_changed"] == ["expert.layers.0.weight"]
    assert report["tensors"]["bytes_changed"] == ["action_in_proj.proj.weight"]


def test_snapshot_parameter_not_in_release_is_blocking():
    snap = _full_snapshot()
    extra = dict(snap["tensors"]["expert_model.expert.layers.0.weight"])
    snap["tensors"]["expert_model.expert.embed_tokens.weight"] = extra
    snap["tensors"] = dict(sorted(snap["tensors"].items()))
    snap["tensor_set_sha256"] = identity_hash(
        {n: {k: e[k] for k in ("dtype", "shape", "byte_length", "sha256")}
         for n, e in snap["tensors"].items()}
    )
    report = _audit(policy_snapshot=snap)
    assert "snapshot_parameter_not_in_release" in _codes(report, "error")


def test_converter_rename_mismatch_and_changed_bytes_block():
    export = _export_tensors()
    bad = dict(export)
    bad["expert.layers.0.weight"] = ("BF16", [2], _bf16(7.0, 9.0))
    report = _audit(inference=_inference_manifest(bad))
    assert report["converter"]["status"] == "blocked"
    assert report["converter"]["changed"] == ["expert.layers.0.weight"]
    assert "converter_changed_tensor" in _codes(report, "error")

    wrong = {ea.export_key_to_inference_key(k): v for k, v in export.items()}
    wrong["lm_head.weight"] = wrong.pop("vlm.lm_head.weight")  # left unrenamed
    files = {"00000.safetensors": _st(wrong, {"format": "pt"})}
    report = _audit(inference=build_manifest(files, scope="synthetic"))
    assert "converter_key_mismatch" in _codes(report, "error")
    assert report["converter"]["missing"] == ["vlm.lm_head.weight"]
    assert report["converter"]["unexpected"] == ["lm_head.weight"]


def test_converter_metadata_and_shard_names_are_checked():
    report = _audit(inference=_inference_manifest(metadata={"format": "np"}))
    assert "converter_metadata_unexpected" in _codes(report, "warning")
    report = _audit(inference=_inference_manifest(shard_of=lambda n: "00000.safetensors"))
    assert "converter_shard_layout_changed" in _codes(report, "warning")


def test_inference_not_supplied_is_not_run_not_pass():
    report = _audit(inference=None)
    assert report["converter"]["status"] == "not_run"
    assert report["status"] == "passed"
    assert "inference_manifest_not_supplied" in _codes(report, "warning")


def test_shard_layout_rules():
    # Non-Cosmos names, gap in numbering and over-limit shards are each reported.
    named = _export_manifest(shard_of=lambda n: "model-00001-of-00002.safetensors")
    report = _audit(export=named, inference=None)
    assert "export_shard_name_unexpected" in _codes(report, "error")

    gap = _export_manifest(shard_of=lambda n: "00000.safetensors" if n.startswith("model")
                           else "00002.safetensors")
    report = _audit(export=gap, inference=None)
    assert "export_shard_numbering_gap" in _codes(report, "error")

    shards = ea.check_export_shards(_export_manifest(), max_shard_bytes=11)
    assert shards["status"] == "blocked"
    assert [i["code"] for i in shards["issues"]] == ["export_shard_over_limit"] * 2
    # Upstream writes a tensor larger than the limit alone in its own shard: legitimate.
    one_each = _export_manifest(shard_of=lambda n: f"{PARAMS_EXPORT.index(n):05d}.safetensors")
    single = ea.check_export_shards(one_each, max_shard_bytes=2)
    assert single["status"] == "passed"
    assert {i["code"] for i in single["issues"]} == {"export_single_tensor_over_limit"}
    assert {i["severity"] for i in single["issues"]} == {"warning"}

    unindexed = build_manifest({"00000.safetensors": _st(_export_tensors())}, scope="synthetic")
    report = ea.check_export_shards(unindexed)
    assert [i["code"] for i in report["issues"]] == ["export_index_missing"]


PARAMS_EXPORT = sorted(_export_tensors())


def test_upstream_shard_limit_constant():
    assert ea.EXPORT_SHARD_MAX_BYTES == 4 * 1024**3 == 4294967296
    assert ea.EXPORT_SHARD_NAME.fullmatch("00000.safetensors")
    assert not ea.EXPORT_SHARD_NAME.fullmatch("model-00001-of-00002.safetensors")


def test_invalid_or_mismatched_inputs_block():
    tampered = _release_manifest()
    tampered["tensors"]["expert.layers.0.weight"]["sha256"] = "f" * 64
    report = _audit(release=tampered)
    assert report["status"] == "blocked" and _codes(report) == ["input_invalid"]
    assert report["tensors"] is None and report["key_sets"] is None

    report = _audit(release=_release_manifest(scope="actual"))
    assert report["status"] == "blocked" and _codes(report) == ["scope_mismatch"]

    unknown = _release_manifest()
    unknown["schema_version"] = 2
    assert _codes(_audit(release=unknown)) == ["input_invalid"]

    blocked = _release_manifest()
    blocked["status"] = "blocked"
    assert _codes(_audit(release=blocked)) == ["input_invalid"]

    assert _codes(_audit(policy_snapshot={"artifact_kind": "s4_param_snapshot"})) == [
        "input_invalid"
    ]
    with pytest.raises(ValueError, match="hf_config_is_vlm"):
        _audit(hf_config_is_vlm="yes")


@pytest.mark.parametrize("label", ["release", "export"])
def test_missing_required_artifact_is_input_invalid_not_a_key_error(label):
    # B3 re-verification: a None release or export used to raise KeyError in audit_export.
    report = _audit(**{label: None})
    assert report["status"] == "blocked"
    assert {(i["code"], i["path"]) for i in report["issues"]} == {("input_invalid", label)}
    assert report["input_hashes"][label] is None
    assert report["tensors"] is None and report["key_sets"] is None


# ---------------------------------------------------------------------------
# E02 zero-update export requirements
# ---------------------------------------------------------------------------


def _e02(**kw):
    args = {
        "release": _release_manifest(),
        "export": _export_manifest(),
        "inference": _inference_manifest(),
        "policy_snapshot": _full_snapshot(),
        "hf_config_is_vlm": True,
        "optimizer_steps_applied": 0,
        "gym_bundle": _release_manifest(),
    }
    args.update(kw)
    return ea.check_zero_update_export(**args)


def test_e02_zero_update_known_answer():
    report = _e02()
    assert report["artifact_kind"] == "s4_e02_zero_update_check"
    # A10-F2: cast and action-level parity are deferred to a host, so E02 is never passed.
    assert report["status"] == "qualify_on_host", report["issues"]
    assert _codes(report, "error") == []
    assert report["same_dtype_tensors_compared"] == 5
    assert report["requirements"] == {
        "optimizer_steps_applied_zero": "passed",
        "gym_bundle_matches_release": "passed",
        "export_audit": "passed",
        "same_dtype_tensors_bitwise_identical": "passed",
        "inference_conversion": "passed",
        "cast_tensor_parity": "qualify_on_host",
        "action_level_parity": "qualify_on_host",
    }
    assert report["requires_parity_ladder"] == ["action_out_proj.bias"]
    assert report["buffers_absent_from_export"] == sorted(BUFFERS)
    assert report["export_audit"]["status"] == "passed"


def test_e02_step_count_must_be_captured_zero():
    report = _e02(optimizer_steps_applied=None)
    assert report["status"] == "blocked"
    assert "optimizer_step_count_unrecorded" in _codes(report)
    assert _codes(_e02(optimizer_steps_applied=1)) == ["not_zero_update"]
    with pytest.raises(ValueError, match="optimizer_steps_applied"):
        _e02(optimizer_steps_applied=True)
    with pytest.raises(ValueError, match="optimizer_steps_applied"):
        _e02(optimizer_steps_applied=-1)


def test_e02_byte_change_without_dtype_change_blocks():
    export = _export_tensors(**{"expert.layers.0.weight": ("BF16", [2], _bf16(7.0, 8.5))})
    report = _e02(export=_export_manifest(export), inference=_inference_manifest(export))
    assert report["status"] == "blocked"
    assert "zero_update_bytes_changed" in _codes(report)
    assert report["requirements"]["same_dtype_tensors_bitwise_identical"] == "blocked"


def test_e02_missing_inputs_block_by_name():
    report = _e02(inference=None, gym_bundle=None)
    assert report["status"] == "blocked"
    assert report["requirements"]["inference_conversion"] == "not_run"
    assert report["requirements"]["gym_bundle_matches_release"] == "not_run"
    assert {"inference_manifest_missing", "gym_bundle_manifest_missing"} <= set(_codes(report))


@pytest.mark.parametrize("label", ["release", "export"])
def test_e02_missing_required_artifact_blocks_not_a_key_error(label):
    # A None release with a manifest Gym bundle used to reach tables["release"] (KeyError).
    report = _e02(**{label: None, "gym_bundle": _release_manifest()})
    assert report["status"] == "blocked"
    assert report["requirements"]["export_audit"] == "blocked"
    assert {(i["code"], i["path"]) for i in report["export_audit"]["issues"]} == {
        ("input_invalid", label)}
    if label == "release":  # the Gym-bundle comparison names the missing release too
        assert report["requirements"]["gym_bundle_matches_release"] == "blocked"
        assert ("input_invalid", "release") in {(i["code"], i["path"])
                                                for i in report["issues"]}


def test_e02_gym_bundle_must_equal_release():
    other = dict(RELEASE)
    other["expert.layers.0.weight"] = ("BF16", [2], _bf16(7.0, 7.5))
    report = _e02(gym_bundle=_release_manifest(other))
    assert report["requirements"]["gym_bundle_matches_release"] == "blocked"
    assert "gym_bundle_differs_from_release" in _codes(report)


# ---------------------------------------------------------------------------
# Torch freedom and effect boundaries (static)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["teacher_rl/export_audit.py", "scripts/s4_snapshot_params.py"],
)
def test_module_never_imports_torch_or_effectful_modules(path):
    tree = ast.parse((ROOT / path).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"torch", "safetensors", "subprocess", "socket", "urllib",
                           "requests", "huggingface_hub", "http"}


def test_parity_script_imports_torch_only_lazily():
    tree = ast.parse((ROOT / "scripts/s4_parity_check.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "torch" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"torch", "safetensors"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"subprocess", "socket", "urllib", "requests", "http",
                           "huggingface_hub"}


# ---------------------------------------------------------------------------
# Review regressions (A10-F2, A10-F7)
# ---------------------------------------------------------------------------

ALL_F32 = {
    "vlm.lm_head.weight": ("F32", [2], _f32(5.0, 6.0)),
    "expert.layers.0.weight": ("F32", [2], _f32(7.0, 8.0)),
    "action_out_proj.bias": ("F32", [2], _f32(0.5, -0.5)),
}


def test_e02_all_cast_garbage_export_is_never_passed():
    # A10-F2: every tensor cast -> zero same-dtype comparisons -> not_applicable, and the
    # deferred cast parity keeps the whole check qualify_on_host (values are unchecked).
    garbage = {
        "lm_head.weight": ("BF16", [2], _bf16(-5000.0, -6000.0)),
        "expert.layers.0.weight": ("BF16", [2], _bf16(-7000.0, -8000.0)),
        "action_out_proj.bias": ("BF16", [2], _bf16(123.0, 456.0)),
    }
    release = _release_manifest(ALL_F32)
    report = ea.check_zero_update_export(
        release=release, export=_sharded(garbage, lambda n: "00000.safetensors"),
        inference=_sharded({ea.export_key_to_inference_key(k): v for k, v in garbage.items()},
                           lambda n: "00000.safetensors", metadata={"format": "pt"}),
        optimizer_steps_applied=0, hf_config_is_vlm=True,
        gym_bundle=_links_record(release),
    )
    assert report["status"] == "qualify_on_host"
    assert report["requirements"]["same_dtype_tensors_bitwise_identical"] == "not_applicable"
    assert report["requirements"]["cast_tensor_parity"] == "qualify_on_host"
    assert report["same_dtype_tensors_compared"] == 0
    assert report["requires_parity_ladder"] == sorted(ALL_F32)


def test_e02_status_is_blocked_by_any_error_and_never_passed():
    assert _e02(optimizer_steps_applied=1)["status"] == "blocked"
    assert _e02(gym_bundle=None)["status"] == "blocked"
    for report in (_e02(), _e02(gym_bundle=_links_record(_release_manifest()))):
        assert report["status"] != "passed"


RELEASE_ROOT = "/ckpt/release"
BUNDLE_ROOT = "/ckpt/bundle"


def _links_record(release, entries=None, **changes):
    """A hand-built ``s4_bundle_links`` record (schema per scripts/s4_snapshot_params.py)."""
    files = release["input_hashes"]["files"]
    if entries is None:
        entries = {name: {"kind": "symlink", "target": f"{RELEASE_ROOT}/{name}",
                          "normalized_target": f"{RELEASE_ROOT}/{name}",
                          "resolved_target": f"{RELEASE_ROOT}/{name}", "sha256": None}
                   for name in files}
    record = {
        "artifact_kind": "s4_bundle_links",
        "schema_version": 1,
        "scope": release["scope"],
        "producer": {"module": "scripts.s4_snapshot_params", "source_sha256": "0" * 64},
        "input_hashes": {"release_manifest": identity_hash(release)},
        "units": {},
        "status": "passed",
        "bundle_root": BUNDLE_ROOT,
        "release_root": RELEASE_ROOT,
        "release_tensor_set_sha256": release["tensor_set_sha256"],
        "release_files": dict(files),
        "entries": entries,
        "issues": [],
    }
    record.update(changes)
    return record


def test_e02_gym_bundle_link_record_binds_to_release():
    # A10-F7: the converted bundle symlinks the release shards; the no-follow manifest reader
    # refuses symlinks, so the bundle is evidenced by its link targets plus release bytes.
    release = _release_manifest()
    report = _e02(gym_bundle=_links_record(release))
    assert report["requirements"]["gym_bundle_matches_release"] == "passed"
    relative = _links_record(release, entries={
        "model.safetensors": {"kind": "symlink", "target": "../release/model.safetensors",
                              "normalized_target": f"{RELEASE_ROOT}/model.safetensors",
                              "resolved_target": f"{RELEASE_ROOT}/model.safetensors",
                              "sha256": None}})
    assert _e02(gym_bundle=relative)["requirements"]["gym_bundle_matches_release"] == "passed"
    copy = _links_record(release, entries={
        "model.safetensors": {"kind": "regular_file", "target": None, "normalized_target": None,
                              "sha256": release["input_hashes"]["files"]["model.safetensors"]}})
    assert _e02(gym_bundle=copy)["requirements"]["gym_bundle_matches_release"] == "passed"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"entries": {"model.safetensors": {
            "kind": "symlink", "target": "/elsewhere/model.safetensors",
            "normalized_target": "/elsewhere/model.safetensors", "sha256": None}}},
         "gym_bundle_differs_from_release"),
        ({"entries": {"model.safetensors": {
            "kind": "regular_file", "target": None, "normalized_target": None,
            "sha256": "0" * 64}}}, "gym_bundle_differs_from_release"),
        ({"entries": {"model.safetensors": {
            "kind": "missing", "target": None, "normalized_target": None, "sha256": None}}},
         "gym_bundle_differs_from_release"),
        ({"entries": {}}, "input_invalid"),
        ({"release_files": {"model.safetensors": "f" * 64}}, "gym_bundle_links_not_for_release"),
        ({"release_tensor_set_sha256": "f" * 64}, "gym_bundle_links_not_for_release"),
        ({"status": "blocked"}, "input_invalid"),
        ({"schema_version": 2}, "input_invalid"),
        ({"scope": "actual"}, "input_invalid"),
        ({"release_root": "relative/release"}, "input_invalid"),
    ],
)
def test_e02_gym_bundle_link_record_negative_controls(change, code):
    release = _release_manifest()
    record = _links_record(release)
    record.update(change)
    report = _e02(gym_bundle=record)
    assert report["requirements"]["gym_bundle_matches_release"] == "blocked"
    assert code in _codes(report, "error")
    assert report["status"] == "blocked"


def _symlink_entry(target, resolved=f"{RELEASE_ROOT}/model.safetensors"):
    return {"model.safetensors": {"kind": "symlink", "target": target,
                                  "normalized_target": f"{RELEASE_ROOT}/model.safetensors",
                                  "resolved_target": resolved, "sha256": None}}


@pytest.mark.parametrize(
    ("target", "resolved"),
    [
        # A10R (re-verify probe case C): lexically the release, physically through a
        # symlinked directory; the record's own resolved_target claim cannot rescue it.
        ("../evil/../release/model.safetensors", f"{RELEASE_ROOT}/model.safetensors"),
        ("/ckpt/evil/../release/model.safetensors", f"{RELEASE_ROOT}/model.safetensors"),
        (f"{RELEASE_ROOT}/./model.safetensors", f"{RELEASE_ROOT}/model.safetensors"),
        ("./../release/model.safetensors", f"{RELEASE_ROOT}/model.safetensors"),
        (f"{RELEASE_ROOT}//model.safetensors", f"{RELEASE_ROOT}/model.safetensors"),
        # A clean target whose recorded physical resolution is elsewhere or absent.
        (f"{RELEASE_ROOT}/model.safetensors", "/elsewhere/release/model.safetensors"),
        (f"{RELEASE_ROOT}/model.safetensors", None),
    ],
)
def test_e02_gym_bundle_link_target_must_be_canonical_and_physically_the_release(target,
                                                                                  resolved):
    release = _release_manifest()
    record = _links_record(release, entries=_symlink_entry(target, resolved))
    report = _e02(gym_bundle=record)
    assert report["requirements"]["gym_bundle_matches_release"] == "blocked"
    assert "gym_bundle_differs_from_release" in _codes(report, "error")


def test_e02_gym_bundle_link_record_without_resolved_target_is_blocked():
    release = _release_manifest()
    entries = _symlink_entry(f"{RELEASE_ROOT}/model.safetensors")
    del entries["model.safetensors"]["resolved_target"]
    report = _e02(gym_bundle=_links_record(release, entries=entries))
    assert report["requirements"]["gym_bundle_matches_release"] == "blocked"


def test_e02_gym_bundle_link_record_needs_a_release_manifest():
    snapshot_release = _full_snapshot()
    record = _links_record(_release_manifest())
    report = ea.check_zero_update_export(
        release=snapshot_release, export=_export_manifest(), inference=_inference_manifest(),
        optimizer_steps_applied=0, hf_config_is_vlm=True, gym_bundle=record,
    )
    assert report["requirements"]["gym_bundle_matches_release"] == "blocked"
