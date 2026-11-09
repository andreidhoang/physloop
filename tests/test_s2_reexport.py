"""Known answers and negative controls for the S2.4 policy B re-export control.

No test runs a converter, uv, hf, docker or a network call. Converter facts come from the
pinned AlpaGym 972d160 / alpamayo-recipes 1d99fc5 sources vendored byte-identically in
tests/fixtures/s2_reexport (hash- and git-blob-checked against provenance.json); only named
pure functions are executed from them, in a restricted namespace.

Hand-built safetensors (no torch/safetensors import). Release tensors and literal SHA-256 of
their stored bytes (computed independently with ``shasum -a 256`` when authored):

- ``vlm.model.language_model.embed_tokens.weight`` BF16 [2, 2] = 803f004080bf0000
  (1.0, 2.0, -1.0, 0.0) -> f8b1d024...; changed last element 0x0001 -> 803f004080bf0100
  -> 9a0bf914...
- ``expert.layers.0.mlp.down_proj.weight`` F32 [2] = 0000803f000000c0 (1.0, -2.0) ->
  ee4ac73c...; cast to BF16 803f00c0 -> 7b429b1e...
- ``action_out_proj.bias`` F32 [] = 0000003f (0.5) -> d99e5843...
"""

import ast
import copy
import hashlib
import inspect
import json
import os
import struct
from pathlib import Path

import pytest

from gate import s2_bootstrap as boot
from gate import s2_reexport as rx
from gate.s2_identity import identity_hash
from gate.tensor_manifest import build_manifest
from scripts import s2_guest_bootstrap as guest
from scripts import s2_reexport as cli
from scripts.s2_reexport import main

FIXTURES = Path(__file__).parent / "fixtures" / "s2_reexport"
EMBED = "vlm.model.language_model.embed_tokens.weight"
DOWN = "expert.layers.0.mlp.down_proj.weight"
BIAS = "action_out_proj.bias"
SHA_EMBED = "f8b1d02420e892bef2abc97c49b1e019d99df77a396425e55e0f356ce1f1fcb8"
SHA_EMBED_CHANGED = "9a0bf9140e7bdb4a54c190431e2d37728dcb84ad75b7f03b96f9d2ca14a8b592"
SHA_DOWN = "ee4ac73c2bd27756ab82780f27c73a7bc4d3f0bb6acb37e008bc27eccd7e588b"
SHA_DOWN_BF16 = "7b429b1e3fd37fd03505ae4982471ea2c830392213b48a4e69976b5ebebce8e4"
SHA_BIAS = "d99e58435243d9fef9c88273b8d553b4fba4d0baf8009d29eae74fa99e0d9f57"
SHARD1 = "model-00001-of-00002.safetensors"
SHARD2 = "model-00002-of-00002.safetensors"
RELEASE = {
    SHARD1: [(EMBED, "BF16", [2, 2], bytes.fromhex("803f004080bf0000"))],
    SHARD2: [(DOWN, "F32", [2], bytes.fromhex("0000803f000000c0")),
             (BIAS, "F32", [], bytes.fromhex("0000003f"))],
}
COSMOS_REVISION = "b" * 40
RECORD = {"artifact_kind": "s2_reexport_action_record", "schema_version": 1,
          "action_record_ref": "docs/DECISIONS.md#synthetic-reexport-test",
          "authorized_phases": ["verify", "convert", "control"],
          "authorizations": {"host_setup": "docs/DECISIONS.md#synthetic-host-setup"}}
GUEST = list(boot.GUEST_PYTHON)


# --- hand-built safetensors ---------------------------------------------------------------


def st_file(entries, *, metadata=None, reverse=False):
    """Safetensors bytes: header (insertion order, optionally reversed) + contiguous data."""
    header, data = {}, b""
    for name, dtype, shape, raw in entries:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    items = list(header.items())
    if reverse:
        items.reverse()
    if metadata is not None:
        items.insert(0, ("__metadata__", metadata))
    raw_header = json.dumps(dict(items), separators=(",", ":")).encode()
    return struct.pack("<Q", len(raw_header)) + raw_header + data


def index_bytes(layout, indent=None):
    weight_map = {name: shard for shard, entries in layout.items() for name, *_ in entries}
    total = sum(len(raw) for entries in layout.values() for *_, raw in entries)
    return json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map},
                      indent=indent).encode()


def checkpoint(layout, *, metadata=None, reverse=False, indent=None):
    files = {shard: st_file(entries, metadata=metadata, reverse=reverse)
             for shard, entries in layout.items()}
    files[rx.INDEX_NAME] = index_bytes(layout, indent)
    return files


def reexport_like(layout):
    """What C2 writes: same tensors, metadata {"format": "pt"}, other header order/index."""
    return checkpoint(layout, metadata={"format": "pt"}, reverse=True, indent=2)


def manifest(files, scope="synthetic"):
    return build_manifest(files, scope=scope, index_name=rx.INDEX_NAME)


def write_dir(root, files):
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name).write_bytes(data)


def codes(report, severity=None):
    return {i["code"] for i in report["issues"] if severity is None or i["severity"] == severity}


# --- vendored upstream sources ------------------------------------------------------------


def provenance():
    return json.loads((FIXTURES / "provenance.json").read_text())


def vendored(name):
    entry = next(e for repo in provenance()["repositories"] for e in repo["sources"]
                 if e["file"] == name)
    data = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    assert hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"]
    return data, entry


def test_vendored_sources_and_licenses_match_provenance():
    record = provenance()
    assert [r["repository"] for r in record["repositories"]] == [
        "https://github.com/NVlabs/alpagym", "https://github.com/NVlabs/alpamayo-recipes"]
    for repo in record["repositories"]:
        assert repo["license"] == "Apache-2.0"
        data = (FIXTURES / repo["license_file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == repo["license_file_sha256"]
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        assert hashlib.sha1(blob).hexdigest() == repo["license_file_git_blob_sha"]
        assert b"Apache License" in data
        for entry in repo["sources"]:
            data, _ = vendored(entry["file"])
            assert repo["commit"] in entry["source_url"]
            assert b"SPDX-License-Identifier: Apache-2.0" in data
    listed = {"provenance.json"} | {r["license_file"] for r in record["repositories"]} | {
        e["file"] for r in record["repositories"] for e in r["sources"]}
    present = {str(p.relative_to(FIXTURES)) for p in FIXTURES.rglob("*") if p.is_file()}
    assert present == listed  # nothing unlisted, nothing missing
    # The module's pinned converter facts are the vendored files' own.
    alpagym = record["repositories"][0]
    assert alpagym["commit"] == rx.ALPAGYM_COMMIT
    for key, item in rx.CONVERTERS.items():
        entry = next(e for e in alpagym["sources"] if e["path"] == item["path"])
        data, _ = vendored(entry["file"])
        assert (len(data), entry["sha256"], entry["git_blob_sha"]) == (
            item["size_bytes"], item["sha256"], item["git_blob_sha"]), key


def _tree(name):
    data, _ = vendored(name)
    return ast.parse(data, filename=name)


def _function(tree, name):
    (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return node


RELEASE_SRC = "alpagym/convert_release_to_alpagym_checkpoint_972d160e.py.txt"
INFERENCE_SRC = "alpagym/convert_alpagym_checkpoint_to_inference_972d160e.py.txt"
UTILS_SRC = "alpamayo-recipes/checkpoint_utils_1d99fc50.py.txt"


def _exec(name, function, namespace):
    path = FIXTURES / name
    tree = _tree(name)
    code = compile(ast.Module(body=[_function(tree, function)], type_ignores=[]), str(path),
                   "exec")
    exec(code, namespace)  # noqa: S102 - one named pure function from pinned, hashed source
    return namespace[function]


def test_inference_key_map_is_the_identity_on_release_names():
    to_inference = _exec(INFERENCE_SRC, "to_inference_weight_key",
                         {"__builtins__": {"len": len}, "str": str})
    for key in (EMBED, "vlm.model.visual.blocks.0.attn.qkv.weight", "vlm.lm_head.weight",
                DOWN, BIAS, "action_in_proj.mlp.0.weight", "expert.norm.weight"):
        assert to_inference(key) == key
    # Flattened Cosmos-export names (the converter's real input) are renamed under vlm.
    assert to_inference("lm_head.weight") == "vlm.lm_head.weight"
    assert to_inference("visual.blocks.0.attn.qkv.weight") == (
        "vlm.model.visual.blocks.0.attn.qkv.weight")
    assert to_inference("model.embed_tokens.weight") == EMBED


def test_output_directory_policy_refuses_a_non_empty_destination(tmp_path):
    prepare = _exec(UTILS_SRC, "prepare_output_dir", {
        "__builtins__": {"any": any, "bool": bool, "FileExistsError": FileExistsError,
                         "NotADirectoryError": NotADirectoryError},
        "Path": Path, "shutil": None})
    fresh = tmp_path / "fresh"
    prepare(fresh, overwrite=False)
    assert fresh.is_dir()
    prepare(fresh, overwrite=False)  # empty: reused
    (fresh / "model-00001-of-00002.safetensors").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        prepare(fresh, overwrite=False)
    (tmp_path / "file").write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        prepare(tmp_path / "file", overwrite=False)


def test_bundle_symlinks_shards_and_index_so_c2_reads_release_bytes():
    tree = _tree(UTILS_SRC)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            (target,) = node.targets
            constants[target.id] = ast.literal_eval(node.value.args[0])
    assert constants == {"DEFAULT_SYMLINK_EXTENSIONS": {".safetensors"},
                         "DEFAULT_SYMLINK_NAMES": {rx.INDEX_NAME}}


def _arguments(name):
    flags = []
    for node in ast.walk(_tree(name)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            flags.append(node.args[0].value)
    return flags


def test_plan_argv_uses_only_the_converters_own_flags_and_never_overwrite():
    assert _arguments(RELEASE_SRC) == ["--input", "--output", "--vlm-name-or-path",
                                       "--overwrite"]
    assert _arguments(INFERENCE_SRC) == ["--input", "--output", "--overwrite"]
    plan = compiled_plan()
    stages = {s["id"]: s for s in plan["stages"]}
    for stage_id, source in (("C1_release_to_gym", RELEASE_SRC),
                             ("C2_gym_to_inference", INFERENCE_SRC)):
        script_args = stages[stage_id]["argv"][5:]  # after uv run --no-sync python <script>
        used = [a for a in script_args if a.startswith("--")]
        assert "--overwrite" not in stages[stage_id]["argv"]
        assert set(used) <= set(_arguments(source)) - {"--overwrite"}


def test_converter_literals_the_plan_relies_on():
    release, inference = _tree(RELEASE_SRC), _tree(INFERENCE_SRC)
    (default,) = [n.value.value for n in release.body if isinstance(n, ast.Assign)
                  and n.targets[0].id == "DEFAULT_VLM"]
    assert default == rx.COSMOS_REASON2_REPOSITORY

    def assigned(tree, function, key):
        for node in ast.walk(_function(tree, function)):
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                    and node.targets[0].slice.value == key):
                return ast.literal_eval(node.value)
        raise AssertionError(key)

    assert assigned(release, "build_expert_config", "model_type") == rx.GYM_MODEL_TYPE
    assert assigned(inference, "build_inference_config", "model_type") == (
        rx.INFERENCE_MODEL_TYPE)
    for tree in (release, inference):
        loops = [n for n in ast.walk(_function(tree, "validate_checkpoint"))
                 if isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple)
                 and all(isinstance(e, ast.Constant) for e in n.iter.elts)]
        assert [tuple(e.value for e in loop.iter.elts) for loop in loops][-1] == (
            rx.TOKENIZER_FILES)
    # The inference converter re-saves shards with this metadata (header bytes differ).
    text = vendored(INFERENCE_SRC)[0].decode()
    assert 'metadata={"format": "pt"}' in text
    assert 'index = json.loads((input_dir / "model.safetensors.index.json").read_text())' in text


def test_pins_equal_teacher_rl_pins_and_gate_never_imports_teacher_rl():
    from teacher_rl import pins

    assert rx.ALPAGYM_REPOSITORY == pins.UPSTREAM_PINS["alpagym"]["repository"]
    assert rx.ALPAGYM_COMMIT == pins.UPSTREAM_PINS["alpagym"]["commit"]
    assert rx.ALPAGYM_UV_LOCK_SHA256 == pins.ALPAGYM_LOCK["uv_lock_sha256"]
    assert rx.TEACHER_REPOSITORY == pins.TEACHER["repository"]
    assert rx.TEACHER_REVISION == pins.TEACHER["revision"]
    assert rx.COSMOS_REASON2_REPOSITORY == pins.COSMOS_REASON2["repository"]
    tree = ast.parse(Path(rx.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(name.startswith(("teacher_rl", "torch", "safetensors")) for name in imported)


def test_private_s2_bootstrap_names_this_compiler_reuses_are_pinned():
    params = list(inspect.signature(boot._Builder.add).parameters)
    assert params[:6] == ["self", "stage_id", "phase", "purpose", "anchor", "argv"]
    for name in ("cwd", "requires", "expected", "idempotency", "timeout", "level", "env_extra"):
        assert name in params
    assert list(inspect.signature(boot._file_rule).parameters) == ["root", "path", "size", "sha"]
    assert list(inspect.signature(boot._hf_cache_path).parameters) == [
        "hf_home", "repository", "revision", "name"]
    assert {"verify", "import", "build", "sync"} <= set(boot.TIMEOUTS)
    assert boot.PLAN_KINDS["s4_runtime"].keys() == rx.PLAN_KIND_SPEC.keys()


# --- inventory and plan -------------------------------------------------------------------


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def inventory(base="/data", *, teacher_files=None, scope="synthetic"):
    files = teacher_files if teacher_files is not None else [
        {"path": "config.json", "size_bytes": 2, "sha256": "1" * 64},
        {"path": rx.INDEX_NAME, "size_bytes": 3, "sha256": "2" * 64},
        {"path": SHARD1, "size_bytes": 4, "sha256": "3" * 64},
        {"path": SHARD2, "size_bytes": 5, "sha256": "4" * 64},
    ]
    return {
        "artifact_kind": "s2_reexport_inventory", "schema_version": 1, "scope": scope,
        "host_paths": {"physloop_root": f"{base}/physloop", "home": f"{base}/home",
                       "alpagym_checkout": f"{base}/alpagym",
                       "teacher_snapshot": f"{base}/teacher",
                       "gym_bundle": f"{base}/teacher-gym", "reexport_dir": f"{base}/teacher-b",
                       "hf_home": f"{base}/hf", "evidence_dir": f"{base}/physloop/evidence/b"},
        "images": [],
        "assets": [
            {"role": "teacher", "repository": rx.TEACHER_REPOSITORY, "repo_type": "model",
             "revision": rx.TEACHER_REVISION,
             "authorization_ref": "docs/DECISIONS.md#synthetic-teacher-terms", "files": files},
            {"role": "cosmos_reason2", "repository": rx.COSMOS_REASON2_REPOSITORY,
             "repo_type": "model", "revision": COSMOS_REVISION,
             "authorization_ref": "docs/DECISIONS.md#synthetic-cosmos-reason2-terms",
             "files": [{"path": "config.json", "size_bytes": "unverified",
                        "sha256": "unverified"},
                       {"path": "tokenizer.json", "size_bytes": 6, "sha256": "5" * 64}]},
        ],
    }


def compiled_plan(inv=None, record=None, phases=("verify", "convert", "control")):
    return rx.build_reexport_plan(inv or inventory(), record or RECORD, phases=list(phases))


def test_compiled_plan_known_answer():
    plan = compiled_plan()
    assert plan["compiled"] is True, plan["issues"]
    assert plan["artifact_kind"] == "s2_reexport_plan"
    assert plan["compiler"] == {"inventory_kind": "s2_reexport",
                                "function": "gate.s2_reexport.build_reexport_plan"}
    assert plan["producer"]["module"] == "gate.s2_reexport"
    assert plan["phases"] == ["verify", "convert", "control"]
    assert plan["execution_status"] == "qualify_on_host"
    assert plan["secrets_required"] == []
    assert plan["authorizations"] == {
        "host_setup": "docs/DECISIONS.md#synthetic-host-setup",
        "terms:cosmos_reason2": "docs/DECISIONS.md#synthetic-cosmos-reason2-terms",
        "terms:teacher": "docs/DECISIONS.md#synthetic-teacher-terms"}
    assert [s["id"] for s in plan["stages"]] == [
        "V1_alpagym_pin", "V2_converter_sources", "V3_teacher_snapshot",
        "V4_cosmos_reason2_cache", "C1_release_to_gym", "C2_gym_to_inference",
        "X1_reexport_control"]
    stages = {s["id"]: s for s in plan["stages"]}
    assert stages["C1_release_to_gym"]["argv"] == [
        "uv", "run", "--no-sync", "python",
        ("/data/alpagym/packages/policies/alpamayo_r1/scripts/"
         "convert_release_to_alpagym_checkpoint.py"),
        "--input", "/data/teacher", "--output", "/data/teacher-gym",
        "--vlm-name-or-path", "nvidia/Cosmos-Reason2-8B"]
    assert stages["C2_gym_to_inference"]["argv"] == [
        "uv", "run", "--no-sync", "python",
        ("/data/alpagym/packages/policies/alpamayo_r1/scripts/"
         "convert_alpagym_checkpoint_to_inference.py"),
        "--input", "/data/teacher-gym", "--output", "/data/teacher-b"]
    assert stages["X1_reexport_control"]["argv"] == [
        *GUEST, "scripts/s2_reexport.py", "compare", "--release-root", "/data/teacher",
        "--reexport-root", "/data/teacher-b", "--index", "model.safetensors.index.json",
        "--scope", "synthetic", "--output-dir", "/data/physloop/evidence/b/s2-reexport-control"]
    for stage_id in ("C1_release_to_gym", "C2_gym_to_inference"):
        assert stages[stage_id]["cwd"] == "/data/alpagym"
        assert stages[stage_id]["env_set"]["HF_HUB_OFFLINE"] == "1"
        assert stages[stage_id]["env_set"]["HF_HOME"] == "/data/hf"
        assert stages[stage_id]["idempotency_check"] is None
        assert stages[stage_id]["level"] == "L3"
    assert stages["X1_reexport_control"]["cwd"] == "/data/physloop"
    assert stages["C1_release_to_gym"]["requires"] == [
        "host_setup", "terms:cosmos_reason2", "terms:teacher"]
    assert stages["C2_gym_to_inference"]["requires"] == ["host_setup", "terms:teacher"]
    assert stages["V4_cosmos_reason2_cache"]["requires"] == [
        "host_setup", "terms:cosmos_reason2"]
    assert [s["timeout_s"] for s in plan["stages"]] == [600, 600, 600, 600, 900, 7200, 3600]
    assert all(s["argv"] is None for s in plan["stages"][:4])
    # Verify rules: pinned HEAD and lock, converter bytes, teacher files, the HF cache layout.
    v1, v2 = stages["V1_alpagym_pin"], stages["V2_converter_sources"]
    assert v1["expected_outputs"] == [
        {"kind": "file_regex", "root": "alpagym_checkout", "path": "/data/alpagym/.git/HEAD",
         "pattern": r"\A972d160eed0e23d388497851504a3a233fec5879\n\Z"},
        {"kind": "file", "root": "alpagym_checkout", "path": "/data/alpagym/uv.lock",
         "size_bytes": None, "sha256": rx.ALPAGYM_UV_LOCK_SHA256}]
    assert [(r["size_bytes"], r["sha256"]) for r in v2["expected_outputs"]] == [
        (7213, "936a290377955fe28ec8db4ed28e7622fe3d52450c8bce439255daf0b5f8b38f"),
        (7472, "ec926d0dd345711663db2493753de492b1a7367cb640a108a87439da776e25c8")]
    assert [r["path"] for r in stages["V4_cosmos_reason2_cache"]["expected_outputs"]] == [
        f"/data/hf/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/{COSMOS_REVISION}/{name}"
        for name in ("config.json", "tokenizer.json")]
    assert stages["V4_cosmos_reason2_cache"]["expected_outputs"][0]["sha256"] is None
    assert [r["pattern"] for r in stages["C2_gym_to_inference"]["expected_outputs"][:2]] == [
        r'"model_type":\s*"alpamayo1_5"', r'"vlm_name_or_path":\s*"nvidia/Cosmos\-Reason2\-8B"']
    assert stages["X1_reexport_control"]["expected_outputs"][-1] == {
        "kind": "file_regex", "root": "evidence_dir",
        "path": "/data/physloop/evidence/b/s2-reexport-control/control.json",
        "pattern": '"control":"passed"'}
    assert stages["X1_reexport_control"]["idempotency_check"] == (
        stages["X1_reexport_control"]["expected_outputs"])
    assert {i["code"] for i in plan["issues"]} == {
        "prerequisites_produced_elsewhere", "hf_offline_cache_resolution_unverified"}
    assert all(i["severity"] == "info" for i in plan["issues"])


def test_control_stage_rechecks_the_pinned_teacher_bytes(tmp_path):
    # G4-R6: X1 certifies B against A only while A is still the inventory's pinned bytes,
    # also in a control-only plan (no V3).
    for phases in (("verify", "convert", "control"), ("control",)):
        plan = compiled_plan(phases=phases)
        assert plan["compiled"] is True, plan["issues"]
        x1 = next(s for s in plan["stages"] if s["id"] == "X1_reexport_control")
        teacher = [r for r in x1["expected_outputs"] if r["root"] == "teacher_snapshot"]
        assert teacher == [
            {"kind": "file", "root": "teacher_snapshot", "path": f"/data/teacher/{name}",
             "size_bytes": size, "sha256": sha * 64}
            for name, size, sha in (("config.json", 2, "1"), (rx.INDEX_NAME, 3, "2"),
                                    (SHARD1, 4, "3"), (SHARD2, 5, "4"))]
        assert x1["idempotency_check"] == x1["expected_outputs"]
    base = tmp_path / "host"
    data = {"config.json": b"{}\n", rx.INDEX_NAME: b"{}\n", SHARD1: b"1", SHARD2: b"2"}
    write_dir(base / "teacher", data)
    files = [{"path": n, "size_bytes": len(d), "sha256": _hash(d)} for n, d in data.items()]
    plan = compiled_plan(inventory(str(base), teacher_files=files), phases=["control"])
    x1 = next(s for s in plan["stages"] if s["id"] == "X1_reexport_control")
    rules = [r for r in x1["expected_outputs"] if r["root"] == "teacher_snapshot"]
    assert all(boot.evaluate_rule(r, roots=plan["roots"])["ok"] for r in rules)
    (base / "teacher" / SHARD1).write_bytes(b"9")
    assert [boot.evaluate_rule(r, roots=plan["roots"])["reason"] for r in rules] == [
        None, None, "sha256_mismatch", None]


def test_plan_is_sealed_deterministic_and_content_valid():
    plan = compiled_plan()
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    assert plan["plan_sha256"] == identity_hash(body)
    assert compiled_plan() == plan
    assert boot._plan_content_issues(plan) == []
    assert plan["units"] == boot.UNITS
    assert set(plan) == boot.plan_fields("s2_runtime")
    json.dumps(plan, allow_nan=False)


def test_kind_is_registered_exactly_as_declared():
    # G4R item 3: gate.s2_bootstrap.PLAN_KINDS carries the spec this compiler declares
    # (tuples, not lists), so the single guest runner validates and recompiles the plan.
    spec = boot.PLAN_KINDS["s2_reexport"]
    assert spec == rx.PLAN_KIND_SPEC
    assert type(spec["inputs"]) is tuple and type(spec["extra_fields"]) is tuple
    assert boot.validate_plan(compiled_plan()) == []


def test_unregistered_kind_is_refused_by_the_runner(monkeypatch):
    monkeypatch.delitem(boot.PLAN_KINDS, "s2_reexport")
    plan = compiled_plan()
    assert [i["code"] for i in boot.validate_plan(plan)] == ["unknown_plan_kind"]


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setitem(boot.PLAN_KINDS, "s2_reexport", dict(rx.PLAN_KIND_SPEC))


def test_registered_kind_validates_and_recompiles(registered):
    plan = compiled_plan()
    assert boot.compiler_identity("s2_reexport") == rx.COMPILER
    assert boot.validate_plan(plan) == []
    assert boot.resolve_compiler("s2_reexport") is rx.build_reexport_plan
    assert boot.plan_kind(plan) == ("s2_reexport", [])
    tampered = copy.deepcopy(plan)
    tampered["stages"][4]["argv"].append("--overwrite")
    assert "plan_hash_mismatch" in codes({"issues": boot.validate_plan(tampered)})


@pytest.mark.parametrize("mutate, code", [
    (lambda inv, rec: inv["assets"][1].update(revision=None), "revision_unresolved"),
    (lambda inv, rec: inv["assets"][1].update(authorization_ref=None), "authorization_missing"),
    (lambda inv, rec: inv["assets"][0].update(authorization_ref=None), "authorization_missing"),
    (lambda inv, rec: inv["host_paths"].update(reexport_dir="/data/teacher/b"),
     "inventory_invalid"),
    (lambda inv, rec: inv["host_paths"].update(gym_bundle=None), "host_path_unset"),
    (lambda inv, rec: inv["assets"][0].update(revision="c" * 40), "inventory_invalid"),
    (lambda inv, rec: inv["assets"][0]["files"][2].update(sha256="unverified"),
     "inventory_invalid"),
    (lambda inv, rec: inv["assets"][0]["files"].pop(1), "inventory_invalid"),
    (lambda inv, rec: inv.update(images=[{"role": "renderer"}]), "inventory_invalid"),
    (lambda inv, rec: inv["assets"][0].update(authorization_ref="ref-hf_" + "Z" * 32),
     "inventory_invalid"),  # credential-like text, built at run time (no literal token)
    (lambda inv, rec: rec.update(authorized_phases=["verify"]), "phase_not_authorized"),
    (lambda inv, rec: rec["authorizations"].pop("host_setup"), "authorization_missing"),
    (lambda inv, rec: rec["authorizations"].update({"acquire:teacher": "x"}),
     "invalid_action_record"),
    (lambda inv, rec: rec.update(action_record_ref="TODO"), "invalid_action_record"),
    (lambda inv, rec: inv.update(scope="real"), "inventory_invalid"),
])
def test_plan_refusals(mutate, code):
    inv, rec = inventory(), copy.deepcopy(RECORD)
    mutate(inv, rec)
    plan = compiled_plan(inv, rec)
    assert plan["compiled"] is False and plan["stages"] == []
    assert code in codes(plan), plan["issues"]


def test_inventory_validation_names_each_defect():
    inv = inventory()
    inv["host_paths"]["reexport_dir"] = "/data/teacher/b"
    inv["assets"][0]["files"][2]["sha256"] = "unverified"
    inv["assets"][0]["files"] = [f for f in inv["assets"][0]["files"]
                                 if f["path"] != rx.INDEX_NAME]
    report = rx.validate_reexport_inventory(inv)
    assert report["valid"] is False and report["status"] == "blocked"
    assert {"overlapping_paths", "teacher_hash_unverified", "teacher_file_missing"} <= codes(
        report)
    assert rx.validate_reexport_inventory(inventory())["status"] == "passed"
    secret = inventory()
    secret["assets"][1]["authorization_ref"] = "hf_" + "A" * 34
    assert "secret_like_value" in codes(rx.validate_reexport_inventory(secret))


@pytest.mark.parametrize("phases", [[], ["verify", "verify"], ["download"], "verify", None])
def test_bad_phases_are_refused(phases):
    plan = rx.build_reexport_plan(inventory(), RECORD, phases=phases)
    assert plan["compiled"] is False and "invalid_phases" in codes(plan)


@pytest.mark.parametrize("bad", [None, [], "inventory", {"artifact_kind": "x"},
                                 {**inventory(), "host_paths": None}])
def test_malformed_inventory_never_raises(bad):
    plan = rx.build_reexport_plan(bad, RECORD, phases=["verify"])
    assert plan["compiled"] is False and plan["stages"] == []
    assert plan["plan_sha256"] == identity_hash(
        {k: v for k, v in plan.items() if k != "plan_sha256"})


def test_phase_subset_compiles_only_its_stages():
    plan = compiled_plan(phases=["control", "verify"])
    assert plan["phases"] == ["verify", "control"]
    assert [s["phase"] for s in plan["stages"]] == ["verify"] * 4 + ["control"]
    assert "hf_offline_cache_resolution_unverified" not in codes(plan)
    control_only = compiled_plan(inventory(), phases=["control"])
    assert control_only["authorizations"] == {
        "host_setup": RECORD["authorizations"]["host_setup"],
        "terms:teacher": "docs/DECISIONS.md#synthetic-teacher-terms"}


# --- control verdict ----------------------------------------------------------------------


def test_release_manifest_literal_hashes():
    release = manifest(checkpoint(RELEASE))
    assert release["status"] == "passed"
    assert {name: (e["dtype"], e["shape"], e["sha256"])
            for name, e in release["tensors"].items()} == {
        BIAS: ("F32", [], SHA_BIAS), DOWN: ("F32", [2], SHA_DOWN),
        EMBED: ("BF16", [2, 2], SHA_EMBED)}


def test_identical_tensors_with_reserialized_files_pass():
    control = rx.reexport_control(manifest(checkpoint(RELEASE)),
                                  manifest(reexport_like(RELEASE)))
    assert control["status"] == "passed" and control["control"] == "passed"
    assert control["issues"] == []
    assert control["tensors"]["identical_count"] == 3
    assert control["tensors"]["release_count"] == control["tensors"]["reexport_count"] == 3
    assert control["tensors"]["release_tensor_set_sha256"] == (
        control["tensors"]["reexport_tensor_set_sha256"])
    assert all(value == [] for value in control["differences"].values())
    assert control["files"]["files_identical"] is False
    assert control["files"]["metadata_identical"] is False
    assert control["files"]["relation"] == "tensors_identical_files_differ"
    assert control["role"].startswith("policy B admission precondition")
    assert control["host_qualification"] == "qualify_on_host"


def test_reshardered_reexport_is_still_tensor_identical():
    one_shard = {"model-00001-of-00001.safetensors": RELEASE[SHARD1] + RELEASE[SHARD2]}
    control = rx.reexport_control(manifest(checkpoint(RELEASE)),
                                  manifest(reexport_like(one_shard)))
    assert control["status"] == "passed"
    assert [f["name"] for f in control["files"]["reexport"]] == [
        "model-00001-of-00001.safetensors", rx.INDEX_NAME]


def test_byte_identical_files_pass_with_a_warning():
    control = rx.reexport_control(manifest(checkpoint(RELEASE)), manifest(checkpoint(RELEASE)))
    assert control["status"] == "passed"
    assert codes(control, "warning") == {"reexport_files_identical_to_release"}
    assert control["files"]["relation"] == "tensors_and_files_identical"


def _variant(shard, index, entry):
    layout = copy.deepcopy(RELEASE)
    layout[shard][index] = entry
    return layout


def test_one_changed_tensor_is_blocked_and_listed():
    changed = _variant(SHARD1, 0, (EMBED, "BF16", [2, 2], bytes.fromhex("803f004080bf0100")))
    control = rx.reexport_control(manifest(checkpoint(RELEASE)),
                                  manifest(reexport_like(changed)))
    assert control["status"] == "blocked" and control["control"] == "blocked"
    assert control["differences"]["bytes_changed"] == [EMBED]
    assert control["differences"]["dtype_changed"] == []
    assert control["differences"]["changed"] == [{
        "name": EMBED, "fields": ["sha256"],
        "release": {"dtype": "BF16", "shape": [2, 2], "sha256": SHA_EMBED},
        "reexport": {"dtype": "BF16", "shape": [2, 2], "sha256": SHA_EMBED_CHANGED}}]
    assert codes(control, "error") == {"tensors_differ"}
    assert control["tensors"]["identical_count"] == 2


def test_dtype_cast_is_blocked_with_dtype_and_bytes_listed():
    cast = _variant(SHARD2, 0, (DOWN, "BF16", [2], bytes.fromhex("803f00c0")))
    control = rx.reexport_control(manifest(checkpoint(RELEASE)), manifest(reexport_like(cast)))
    assert control["status"] == "blocked"
    assert control["differences"]["dtype_changed"] == [DOWN]
    assert control["differences"]["bytes_changed"] == [DOWN]
    assert control["differences"]["changed"][0]["reexport"] == {
        "dtype": "BF16", "shape": [2], "sha256": SHA_DOWN_BF16}


def test_same_bytes_under_another_dtype_or_shape_are_not_identical():
    as_int = _variant(SHARD2, 0, (DOWN, "I32", [2], bytes.fromhex("0000803f000000c0")))
    control = rx.reexport_control(manifest(checkpoint(RELEASE)),
                                  manifest(reexport_like(as_int)))
    assert control["status"] == "blocked"
    assert control["differences"]["dtype_changed"] == [DOWN]
    assert control["differences"]["bytes_changed"] == []
    flat = _variant(SHARD1, 0, (EMBED, "BF16", [4], bytes.fromhex("803f004080bf0000")))
    control = rx.reexport_control(manifest(checkpoint(RELEASE)), manifest(reexport_like(flat)))
    assert control["differences"]["shape_changed"] == [EMBED]
    assert control["status"] == "blocked"


def test_missing_and_extra_tensors_are_blocked_and_listed():
    missing = copy.deepcopy(RELEASE)
    missing[SHARD2] = missing[SHARD2][:1]
    control = rx.reexport_control(manifest(checkpoint(RELEASE)),
                                  manifest(reexport_like(missing)))
    assert control["status"] == "blocked"
    assert control["differences"]["only_in_release"] == [BIAS]
    assert control["differences"]["only_in_reexport"] == []
    extra = copy.deepcopy(RELEASE)
    extra[SHARD2].append(("vlm.lm_head.weight", "U8", [1], b"\x07"))
    control = rx.reexport_control(manifest(checkpoint(RELEASE)), manifest(reexport_like(extra)))
    assert control["differences"]["only_in_reexport"] == ["vlm.lm_head.weight"]
    assert control["status"] == "blocked"


def test_blocked_or_mismatched_manifests_give_no_verdict():
    release = manifest(checkpoint(RELEASE))
    broken = checkpoint(RELEASE)
    broken[SHARD2] = broken[SHARD2][:-1]  # truncated data
    for other in (manifest(broken), manifest(reexport_like(RELEASE), scope="actual"),
                  {"artifact_kind": "tensor_manifest"}):
        control = rx.reexport_control(release, other)
        assert control["status"] == "blocked"
        assert "comparison_blocked" in codes(control, "error")
        assert control["tensors"]["identical_count"] is None
        assert control["files"]["files_identical"] is None
    tampered = copy.deepcopy(manifest(reexport_like(RELEASE)))
    tampered["tensors"][EMBED]["sha256"] = SHA_EMBED_CHANGED  # not resealed
    control = rx.reexport_control(release, tampered)
    assert control["status"] == "blocked"
    assert "manifest_inconsistent" in codes(control)


def test_two_empty_manifests_are_blocked_not_passed():
    # G4-R3: absence is never a pass. Two self-consistent manifests that hold no tensors
    # (build_manifest itself refuses them; reexport_control re-checks) give no verdict.
    real = manifest(checkpoint(RELEASE))
    empty = copy.deepcopy(real)
    empty["tensors"] = {}
    empty["tensor_set_sha256"] = identity_hash({})
    empty["totals"] = dict(empty["totals"], tensor_count=0, tensor_bytes=0)
    control = rx.reexport_control(empty, copy.deepcopy(empty))
    assert control["status"] == "blocked" and control["control"] == "blocked"
    assert "empty_tensor_set" in codes(control, "error")
    text = json.dumps(control, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert rx.CONTROL_PASSED_PATTERN not in text
    # One empty side against a real release stays blocked as well.
    assert rx.reexport_control(real, empty)["status"] == "blocked"


def test_control_envelope_and_determinism():
    args = (manifest(checkpoint(RELEASE)), manifest(reexport_like(RELEASE)))
    first, second = rx.reexport_control(*args), rx.reexport_control(*copy.deepcopy(args))
    assert first == second
    for key in ("artifact_kind", "schema_version", "scope", "producer", "input_hashes",
                "units", "status", "issues"):
        assert key in first
    assert first["artifact_kind"] == "s2_reexport_control"
    assert first["input_hashes"] == {"release_manifest": identity_hash(args[0]),
                                     "reexport_manifest": identity_hash(args[1])}
    text = json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert text.count(rx.CONTROL_PASSED_PATTERN) == 1


# --- CLI ----------------------------------------------------------------------------------


def test_cli_compare_writes_three_files_and_exit_codes(tmp_path, capsys):
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    write_dir(tmp_path / "b", reexport_like(RELEASE))
    args = ["compare", "--release-root", str(tmp_path / "release"), "--reexport-root",
            str(tmp_path / "b"), "--index", rx.INDEX_NAME, "--scope", "synthetic",
            "--output-dir", str(tmp_path / "evidence" / "control")]
    assert main(args) == 0
    out = tmp_path / "evidence" / "control"
    assert sorted(p.name for p in out.iterdir()) == sorted(rx.CONTROL_FILES)
    control = json.loads((out / "control.json").read_text())
    assert control["status"] == "passed"
    assert rx.CONTROL_PASSED_PATTERN in (out / "control.json").read_text()
    release = json.loads((out / "release.manifest.json").read_text())
    assert release["tensors"][EMBED]["sha256"] == SHA_EMBED
    assert main(args) == 2  # existing output is never overwritten
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["error"] == "output_exists"

    write_dir(tmp_path / "b2", reexport_like(_variant(
        SHARD1, 0, (EMBED, "BF16", [2, 2], bytes.fromhex("803f004080bf0100")))))
    blocked = list(args)
    blocked[blocked.index("--reexport-root") + 1] = str(tmp_path / "b2")
    blocked[-1] = str(tmp_path / "evidence" / "control2")
    assert main(blocked) == 1
    text = (tmp_path / "evidence" / "control2" / "control.json").read_text()
    assert rx.CONTROL_PASSED_PATTERN not in text


@pytest.mark.parametrize("case, error", [
    ("same_root", "roots_overlap"), ("nested_root", "roots_overlap"),
    ("output_inside", "output_inside_checkpoint"), ("relative", "invalid_path"),
    ("symlink_shard", "unreadable_checkpoint"), ("missing_index", "unreadable_checkpoint"),
])
def test_cli_compare_refusals(tmp_path, capsys, case, error):
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    write_dir(tmp_path / "b", reexport_like(RELEASE))
    release, b, out = str(tmp_path / "release"), str(tmp_path / "b"), str(tmp_path / "out")
    if case == "same_root":
        b = release
    elif case == "nested_root":
        b = str(tmp_path / "release" / "inner")
    elif case == "output_inside":
        out = str(tmp_path / "b" / "control")
    elif case == "relative":
        out = "out"
    elif case == "symlink_shard":
        (tmp_path / "b" / SHARD1).unlink()
        (tmp_path / "b" / SHARD1).symlink_to(tmp_path / "release" / SHARD1)
    elif case == "missing_index":
        (tmp_path / "b" / rx.INDEX_NAME).unlink()
    code = main(["compare", "--release-root", release, "--reexport-root", b, "--index",
                 rx.INDEX_NAME, "--scope", "synthetic", "--output-dir", out])
    assert code == 2
    assert json.loads(capsys.readouterr().err.splitlines()[-1])["error"] == error
    assert not (tmp_path / "out").exists()


def _compare(release, b, out):
    return main(["compare", "--release-root", str(release), "--reexport-root", str(b),
                 "--index", rx.INDEX_NAME, "--scope", "synthetic", "--output-dir", str(out)])


def _error(capsys):
    return json.loads(capsys.readouterr().err.splitlines()[-1])["error"]


def _case_insensitive(tmp_path):
    probe = tmp_path / "case-probe"
    probe.mkdir()
    return (tmp_path / "CASE-PROBE").is_dir()


def test_cli_compare_refuses_a_double_slash_alias_of_one_checkpoint(tmp_path, capsys):
    # G4-R1: '//p' names the same directory as '/p' on Linux and macOS; the self-compare
    # would pass vacuously. Non-canonical spellings are refused before any check.
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    release = str(tmp_path / "release")
    assert _compare(release, "/" + release, tmp_path / "ev") == 2
    assert _error(capsys) == "invalid_path"
    assert not (tmp_path / "ev").exists()


def test_cli_compare_refuses_a_case_variant_alias_of_one_checkpoint(tmp_path, capsys):
    # G4-R1: directory identity (st_dev, st_ino), not strings, decides disjointness.
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem: no case-variant alias exists")
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    assert _compare(tmp_path / "release", tmp_path / "RELEASE", tmp_path / "ev") == 2
    assert _error(capsys) == "roots_overlap"
    assert not (tmp_path / "ev").exists()


def test_cli_compare_refuses_a_symlinked_root(tmp_path, capsys):
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    (tmp_path / "alias").symlink_to(tmp_path / "release")
    assert _compare(tmp_path / "release", tmp_path / "alias", tmp_path / "ev") == 2
    assert _error(capsys) == "unreadable_checkpoint"
    assert not (tmp_path / "ev").exists()


def test_cli_compare_never_writes_through_a_double_slash_output(tmp_path, capsys):
    # G4-R2: '//<b>/ctl' lies inside the re-export.
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    write_dir(tmp_path / "b", reexport_like(RELEASE))
    before = sorted(p.name for p in (tmp_path / "b").iterdir())
    assert _compare(tmp_path / "release", tmp_path / "b", "/" + str(tmp_path / "b" / "ctl")) == 2
    assert _error(capsys) == "invalid_path"
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == before


@pytest.mark.parametrize("target", ["b", "release", "outside"])
def test_cli_compare_never_writes_through_a_symlinked_output_parent(tmp_path, capsys, target):
    # G4-R2: mkdir(parents=True) follows symlinks; an evidence directory that is a symlink
    # into a checkpoint must not receive the control files (nor any symlinked parent at all).
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    write_dir(tmp_path / "b", reexport_like(RELEASE))
    (tmp_path / "outside").mkdir()
    before = {name: sorted(p.name for p in (tmp_path / name).iterdir())
              for name in ("release", "b", "outside")}
    (tmp_path / "evidence").symlink_to(tmp_path / target)
    code = _compare(tmp_path / "release", tmp_path / "b", tmp_path / "evidence" / "control")
    assert code == 2
    assert _error(capsys) in ("output_inside_checkpoint", "invalid_output_path")
    assert {name: sorted(p.name for p in (tmp_path / name).iterdir())
            for name in ("release", "b", "outside")} == before


def test_cli_compare_refuses_a_case_variant_output_inside_a_checkpoint(tmp_path, capsys):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem: no case-variant alias exists")
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    write_dir(tmp_path / "b", reexport_like(RELEASE))
    before = sorted(p.name for p in (tmp_path / "b").iterdir())
    assert _compare(tmp_path / "release", tmp_path / "b", tmp_path / "B" / "control") == 2
    assert _error(capsys) == "output_inside_checkpoint"
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == before


# --- G4R: error contract and portable identity logic (injected (st_dev, st_ino) chains) ----


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory search permission")
def test_cli_compare_unsearchable_checkpoint_root_is_a_json_error(tmp_path, capsys):
    # G4R item 1: openat(root_fd, '..') needs search permission on the root; a 0444 root
    # must give the documented JSON error and exit 2, not an uncaught PermissionError.
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    write_dir(tmp_path / "b", reexport_like(RELEASE))
    os.chmod(tmp_path / "release", 0o444)
    try:
        assert _compare(tmp_path / "release", tmp_path / "b", tmp_path / "ev") == 2
    finally:
        os.chmod(tmp_path / "release", 0o755)
    assert _error(capsys) == "unreadable_checkpoint"
    assert not (tmp_path / "ev").exists()


def _two_checkpoints(tmp_path):
    write_dir(tmp_path / "release", checkpoint(RELEASE))
    write_dir(tmp_path / "b", reexport_like(RELEASE))
    return tmp_path / "release", tmp_path / "b"


def test_cli_compare_output_ancestry_os_error_is_a_json_error(tmp_path, capsys, monkeypatch):
    # G4R item 1 (output side): an OSError while walking the output's ancestry through '..'
    # is the CLI's invalid_output_path error (exit 2), never a traceback.
    release, b = _two_checkpoints(tmp_path)
    roots = {str(release): cli._root(release), str(b): cli._root(b)}
    monkeypatch.setattr(cli, "_root", lambda path: roots[str(path)])

    def unsearchable(fd):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "_ancestry", unsearchable)
    assert _compare(release, b, tmp_path / "ev") == 2
    assert _error(capsys) == "invalid_output_path"
    assert not (tmp_path / "ev").exists()


def test_ancestry_is_the_physical_identity_chain_up_to_the_filesystem_root(tmp_path):
    # Portable: chain[0] is the directory's own (st_dev, st_ino); the chain passes through
    # every real ancestor and ends at '/' without a repeat.
    (tmp_path / "x" / "y").mkdir(parents=True)
    fd = os.open(tmp_path / "x" / "y", os.O_RDONLY | os.O_DIRECTORY)
    try:
        chain = cli._ancestry(fd)
    finally:
        os.close(fd)
    ident = [(st.st_dev, st.st_ino) for st in (os.stat(p) for p in (
        tmp_path / "x" / "y", tmp_path / "x", tmp_path, "/"))]
    assert chain[0] == ident[0] and chain[1] == ident[1]
    assert ident[2] in chain and chain[-1] == ident[3]
    assert len(set(chain)) == len(chain)


# Injected identities: the real directories are distinct, so every string check passes and
# only the (st_dev, st_ino) rule at the identity check decides. ANC is a shared ancestor.
ANC, REL, REX, OUT = (7, 1), (7, 10), (7, 20), (7, 30)


def _inject_roots(monkeypatch, release, b, release_root, reexport_root):
    table = {str(release): release_root, str(b): reexport_root}
    monkeypatch.setattr(cli, "_root", lambda path: table[str(path)])


@pytest.mark.parametrize("release_root, reexport_root", [
    ((REL, [REL, ANC]), (REX, [REX, (7, 11), REL, ANC])),  # re-export nested in the release
    ((REL, [REL, (7, 21), REX, ANC]), (REX, [REX, ANC])),  # release nested in the re-export
    ((REL, [REL, ANC]), (REL, [REL, ANC])),  # one directory under two names
], ids=["reexport_inside_release", "release_inside_reexport", "same_directory"])
def test_r1_identity_overlapping_roots_are_refused(tmp_path, capsys, monkeypatch,
                                                   release_root, reexport_root):
    # G4R item 2 / mutant R1_identity: each disjunct of the identity rule is needed.
    release, b = _two_checkpoints(tmp_path)
    _inject_roots(monkeypatch, release, b, release_root, reexport_root)
    assert _compare(release, b, tmp_path / "ev") == 2
    assert _error(capsys) == "roots_overlap"
    assert not (tmp_path / "ev").exists()


def test_r1_identity_disjoint_roots_with_a_shared_ancestor_proceed(tmp_path, monkeypatch):
    release, b = _two_checkpoints(tmp_path)
    _inject_roots(monkeypatch, release, b, (REL, [REL, ANC]), (REX, [REX, ANC]))
    monkeypatch.setattr(cli, "_ancestry", lambda fd: [OUT, ANC])
    assert _compare(release, b, tmp_path / "ev") == 0
    assert sorted(p.name for p in (tmp_path / "ev").iterdir()) == sorted(rx.CONTROL_FILES)


@pytest.mark.parametrize("chain", [
    [OUT, (7, 12), REL, ANC],  # output's existing ancestor lies deep inside the release
    [OUT, REX, ANC],  # inside the re-export
    [REL, ANC],  # the deepest existing ancestor is the release root itself
    [REX, ANC],  # ... or the re-export root itself
], ids=["inside_release", "inside_reexport", "is_release", "is_reexport"])
def test_r2_chain_output_under_either_root_is_refused(tmp_path, capsys, monkeypatch, chain):
    # G4R item 2 / mutant R2_chain: any root anywhere in the output's chain refuses.
    release, b = _two_checkpoints(tmp_path)
    _inject_roots(monkeypatch, release, b, (REL, [REL, ANC]), (REX, [REX, ANC]))
    monkeypatch.setattr(cli, "_ancestry", lambda fd: list(chain))
    assert _compare(release, b, tmp_path / "ev") == 2
    assert _error(capsys) == "output_inside_checkpoint"
    assert not (tmp_path / "ev").exists()


def test_r2_chain_output_outside_both_roots_is_written(tmp_path, monkeypatch):
    release, b = _two_checkpoints(tmp_path)
    _inject_roots(monkeypatch, release, b, (REL, [REL, ANC]), (REX, [REX, ANC]))
    monkeypatch.setattr(cli, "_ancestry", lambda fd: [OUT, (7, 31), ANC])
    assert _compare(release, b, tmp_path / "ev" / "control") == 0
    assert sorted(p.name for p in (tmp_path / "ev" / "control").iterdir()) == sorted(
        rx.CONTROL_FILES)


def test_cli_compile_is_a_dry_run(tmp_path, capsys):
    (tmp_path / "inv.json").write_text(json.dumps(inventory()))
    (tmp_path / "rec.json").write_text(json.dumps(RECORD))
    base = ["compile", "--inventory", str(tmp_path / "inv.json"), "--action-record",
            str(tmp_path / "rec.json")]
    assert main([*base, "--phase", "verify", "--phase", "convert", "--phase", "control"]) == 0
    assert json.loads(capsys.readouterr().out) == compiled_plan()
    assert main([*base, "--phase", "convert", "--output", str(tmp_path / "plan.json")]) == 0
    assert json.loads((tmp_path / "plan.json").read_text())["phases"] == ["convert"]
    refused = inventory()
    refused["assets"][1]["revision"] = None
    (tmp_path / "inv.json").write_text(json.dumps(refused))
    assert main([*base, "--phase", "convert"]) == 1
    (tmp_path / "inv.json").write_text("{not json")
    assert main([*base, "--phase", "convert"]) == 2


# --- end to end through the single guest runner (fake runner; nothing executes) ----------


class ConverterFake:
    """Plays C1/C2 by writing what the pinned converters write; X1 runs the real CLI."""

    def __init__(self, paths, reexport_layout):
        self.paths, self.layout, self.calls = paths, reexport_layout, []

    def __call__(self, argv, *, cwd, env, stdin, timeout_s):
        self.calls.append(list(argv))
        script = argv[4] if argv[:4] == ["uv", "run", "--no-sync", "python"] else None
        if script and script.endswith("convert_release_to_alpagym_checkpoint.py"):
            bundle = Path(self.paths["gym_bundle"])
            bundle.mkdir()
            (bundle / "config.json").write_text(json.dumps(
                {"model_type": rx.GYM_MODEL_TYPE,
                 "vlm_name_or_path": rx.COSMOS_REASON2_REPOSITORY}, indent=2) + "\n")
            for name in rx.TOKENIZER_FILES:
                (bundle / name).write_text("{}\n")
            return guest.RunResult(0, b"Checkpoint ready\n", b"", False)
        if script and script.endswith("convert_alpagym_checkpoint_to_inference.py"):
            out = Path(self.paths["reexport_dir"])
            write_dir(out, reexport_like(self.layout))
            (out / "config.json").write_text(json.dumps(
                {"model_type": rx.INFERENCE_MODEL_TYPE,
                 "vlm_name_or_path": rx.COSMOS_REASON2_REPOSITORY}, indent=2) + "\n")
            for name in rx.TOKENIZER_FILES:
                (out / name).write_text("{}\n")
            return guest.RunResult(0, b"Inference-format checkpoint ready\n", b"", False)
        if argv[:len(GUEST) + 2] == [*GUEST, "scripts/s2_reexport.py", "compare"]:
            code = main(argv[len(GUEST) + 1:])
            return guest.RunResult(code, b"", b"", False)
        return guest.RunResult(127, b"", b"unexpected argv\n", False)


def e2e(tmp_path, reexport_layout, registered_kind=True, runner_cls=None):
    base = tmp_path / "host"
    files = checkpoint(RELEASE)
    write_dir(base / "teacher", files)
    (base / "teacher" / "config.json").write_text("{}\n")
    teacher_files = [{"path": name, "size_bytes": len(data), "sha256": _hash(data)}
                     for name, data in sorted({**files, "config.json": b"{}\n"}.items())]
    inv = inventory(str(base), teacher_files=teacher_files)
    record = copy.deepcopy(RECORD)
    plan = rx.build_reexport_plan(inv, record, phases=["convert", "control"])
    assert plan["compiled"] is True, plan["issues"]
    for name in ("physloop", "alpagym", "hf", "home"):
        (base / name).mkdir(exist_ok=True)
    runner = (runner_cls or ConverterFake)(inv["host_paths"], reexport_layout)
    result = guest.run_plan(plan, action_record=record, inventory=inv, runner=runner,
                            journal_dir=str(tmp_path / "journal"), secret_files={},
                            environ={"LANG": "C.UTF-8"}, clock=lambda: "2026-11-09T00:00:00Z")
    return plan, inv, runner, result


def test_end_to_end_plan_runs_through_the_guest_runner(tmp_path, registered):
    _, inv, runner, result = e2e(tmp_path, RELEASE)
    assert result["status"] == "completed", result
    assert [call[4].rsplit("/", 1)[-1] if call[:4] == ["uv", "run", "--no-sync", "python"]
            else call[len(GUEST) + 1] for call in runner.calls] == [
        "convert_release_to_alpagym_checkpoint.py", "convert_alpagym_checkpoint_to_inference.py",
        "compare"]
    control = json.loads((Path(rx.control_dir(inv["host_paths"]["evidence_dir"]))
                          / "control.json").read_text())
    assert control["status"] == "passed"
    assert control["files"]["relation"] == "tensors_identical_files_differ"


def test_end_to_end_changed_reexport_stops_at_the_control(tmp_path, registered):
    changed = _variant(SHARD2, 1, (BIAS, "F32", [], bytes.fromhex("0000803f")))
    _, inv, _, result = e2e(tmp_path, changed)
    assert result["status"] != "completed"
    assert result["failed_stage"] == "X1_reexport_control"
    control = json.loads((Path(rx.control_dir(inv["host_paths"]["evidence_dir"]))
                          / "control.json").read_text())
    assert control["differences"]["bytes_changed"] == [BIAS]


class DriftingTeacherFake(ConverterFake):
    """C2 also rewrites the release config.json (a non-tensor file): A drifts before X1."""

    def __call__(self, argv, *, cwd, env, stdin, timeout_s):
        result = super().__call__(argv, cwd=cwd, env=env, stdin=stdin, timeout_s=timeout_s)
        if argv[4:5] and argv[4].endswith("convert_alpagym_checkpoint_to_inference.py"):
            (Path(self.paths["teacher_snapshot"]) / "config.json").write_text('{"x": 1}\n')
        return result


def test_end_to_end_drifted_release_snapshot_fails_the_control_stage(tmp_path, registered):
    # G4-R6: the tensors still match (the control itself passes), but the release snapshot
    # is no longer the pinned teacher, so X1's expected outputs fail and B is not admitted.
    _, inv, _, result = e2e(tmp_path, RELEASE, runner_cls=DriftingTeacherFake)
    assert result["status"] != "completed"
    assert result["failed_stage"] == "X1_reexport_control"
    control = json.loads((Path(rx.control_dir(inv["host_paths"]["evidence_dir"]))
                          / "control.json").read_text())
    assert control["status"] == "passed"


def test_end_to_end_runs_through_the_registered_kind_without_patching(tmp_path):
    # G4R item 3: no monkeypatched registration; the runner resolves the compiler from
    # gate.s2_bootstrap.PLAN_KINDS itself, recompiles, and executes C1, C2 and X1.
    _, inv, runner, result = e2e(tmp_path, RELEASE)
    assert result["status"] == "completed", result
    assert len(runner.calls) == 3
    control = json.loads((Path(rx.control_dir(inv["host_paths"]["evidence_dir"]))
                          / "control.json").read_text())
    assert control["status"] == "passed"


def test_end_to_end_is_refused_while_the_kind_is_unregistered(tmp_path, monkeypatch):
    monkeypatch.delitem(boot.PLAN_KINDS, "s2_reexport")
    _, _, runner, result = e2e(tmp_path, RELEASE)
    assert result["status"] == "refused" and result["reason"] == "unknown_plan_kind"
    assert runner.calls == []


def test_verify_rules_evaluate_against_vendored_converter_bytes(tmp_path):
    base = tmp_path / "host"
    plan = compiled_plan(inventory(str(base)), phases=["verify"])
    co = base / "alpagym"
    for key, item in rx.CONVERTERS.items():
        name = {"release_to_gym": RELEASE_SRC, "gym_to_inference": INFERENCE_SRC}[key]
        target = co / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((FIXTURES / name).read_bytes())
    stage = next(s for s in plan["stages"] if s["id"] == "V2_converter_sources")
    results = [boot.evaluate_rule(rule, roots=plan["roots"]) for rule in stage["expected_outputs"]]
    assert [r["ok"] for r in results] == [True, True]
    target = co / rx.CONVERTERS["gym_to_inference"]["path"]
    target.write_bytes(target.read_bytes().replace(b"--overwrite", b"--overwritE"))
    results = [boot.evaluate_rule(rule, roots=plan["roots"]) for rule in stage["expected_outputs"]]
    assert [r["reason"] for r in results] == ["sha256_mismatch", None]
