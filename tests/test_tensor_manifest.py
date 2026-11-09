"""Known answers and negative controls for gate.tensor_manifest (A6; CPU-05 F2).

Every safetensors fixture is written by hand: an 8-byte little-endian header
length, a literal JSON header and literal data bytes. Expected tensor hashes are
SHA-256 values of the literal byte slices, computed independently with
``shasum -a 256`` when the test was authored. No torch or safetensors import.
"""

import ast
import fnmatch
import gc
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from gate import tensor_manifest as tm

FIXTURES = Path(__file__).parent / "fixtures" / "s2_probes"
GITIGNORE = Path(__file__).resolve().parent.parent / ".gitignore"

# Literal fixture: tensor "a" F32 [1.0, -2.0], tensor "b" U8 scalar 7.
HEADER = (
    b'{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
    b'"b":{"dtype":"U8","shape":[],"data_offsets":[8,9]}}'
)
DATA = bytes.fromhex("0000803f000000c0") + b"\x07"
FILE = struct.pack("<Q", 105) + HEADER + DATA
SHA_A = "ee4ac73c2bd27756ab82780f27c73a7bc4d3f0bb6acb37e008bc27eccd7e588b"
SHA_B = "ca358758f6d27e6cf45272937977a748fd88391db679ceda7dc7bf1f005ee879"
SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA_FILE = "8b34f1f8081fdbdb29ecac426b4c36eccdbb9cf16f1fcff7f07957128d167726"
SHA_HEADER = "aff22c45667e10e73aa3786342bc4d826c40cf047f2fcf6ed724e2a7ca76b3cf"


def st(header, data=b""):
    """Assemble safetensors bytes from a raw header (bytes) or a dict."""
    raw = header if type(header) is bytes else json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + data


def codes(report):
    return {item["code"] for item in report["issues"]}


def single(data, name="model.safetensors", **kwargs):
    return tm.build_manifest({name: data}, scope="synthetic", **kwargs)


# --- literal known answers -------------------------------------------------


def test_literal_fixture_bytes_and_hashes():
    assert len(HEADER) == 105 and len(FILE) == 122
    assert hashlib.sha256(FILE).hexdigest() == SHA_FILE


def test_parse_literal_fixture():
    parsed = tm.parse_safetensors(FILE)
    assert parsed["header_bytes"] == 105
    assert parsed["header_sha256"] == SHA_HEADER
    assert parsed["metadata"] is None
    assert parsed["tensors"] == {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8], "byte_length": 8},
        "b": {"dtype": "U8", "shape": [], "data_offsets": [8, 9], "byte_length": 1},
    }


def test_manifest_literal_hashes():
    report = single(FILE)
    assert report["artifact_kind"] == "tensor_manifest"
    assert report["schema_version"] == 1
    assert report["scope"] == "synthetic"
    assert report["status"] == "passed", report["issues"]
    assert report["layout"] == "single_file"
    assert report["tensors"] == {
        "a": {
            "file": "model.safetensors",
            "dtype": "F32",
            "shape": [2],
            "byte_length": 8,
            "sha256": SHA_A,
        },
        "b": {
            "file": "model.safetensors",
            "dtype": "U8",
            "shape": [],
            "byte_length": 1,
            "sha256": SHA_B,
        },
    }
    (entry,) = report["files"]
    assert entry["sha256"] == SHA_FILE
    assert entry["size_bytes"] == 122
    assert entry["header_sha256"] == SHA_HEADER
    assert report["input_hashes"]["files"] == {"model.safetensors": SHA_FILE}
    assert report["totals"] == {"tensor_count": 2, "tensor_bytes": 9, "file_count": 1}
    assert report["producer"]["module"] == "gate.tensor_manifest"


def test_tensor_hash_is_raw_byte_range_without_dtype_or_shape_prefix():
    assert tm.tensor_sha256(bytes.fromhex("0000803f000000c0")) == SHA_A
    assert tm.tensor_sha256(memoryview(b"\x07")) == SHA_B
    record = tm.tensor_record("F32", [2], bytes.fromhex("0000803f000000c0"))
    assert record == {"dtype": "F32", "shape": [2], "byte_length": 8, "sha256": SHA_A}
    with pytest.raises(tm.TensorManifestError) as err:
        tm.tensor_record("F32", [3], bytes(8))
    assert err.value.code == "tensor_size_mismatch"
    with pytest.raises(tm.TensorManifestError) as err:
        tm.tensor_record("U16", [1], bytes(2))
    assert err.value.code == "unsupported_dtype"


def test_dtype_table_is_exactly_rule_11():
    assert tm.DTYPE_ITEMSIZE == {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
        "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    }


def test_header_order_need_not_match_offset_order_and_empty_tensor():
    header = {
        "z": {"dtype": "I16", "shape": [1], "data_offsets": [2, 4]},
        "e": {"dtype": "F32", "shape": [0, 3], "data_offsets": [2, 2]},
        "y": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
    }
    report = single(st(header, b"\x80\x3f\x05\x00"))
    assert report["status"] == "passed", report["issues"]
    assert report["tensors"]["e"]["sha256"] == SHA_EMPTY
    assert report["tensors"]["e"]["byte_length"] == 0
    assert report["tensors"]["y"]["sha256"] == hashlib.sha256(b"\x80\x3f").hexdigest()
    assert list(report["tensors"]) == ["e", "y", "z"]


def test_trailing_space_padding_allowed_and_metadata_recorded():
    header = (
        b'{"__metadata__":{"format":"pt"},"a":{"dtype":"U8","shape":[1],'
        b'"data_offsets":[0,1]}}    '
    )
    report = single(st(header, b"\x07"))
    assert report["status"] == "passed", report["issues"]
    assert report["files"][0]["metadata"] == {"format": "pt"}
    assert report["tensors"]["a"]["sha256"] == SHA_B


# --- malformed headers fail closed -------------------------------------------


@pytest.mark.parametrize(
    "data, code",
    [
        (b"\x01\x00\x00", "file_too_short"),
        (struct.pack("<Q", 0), "header_empty"),
        (struct.pack("<Q", 50) + b"{}", "header_length_out_of_bounds"),
        (st(b" {}"), "header_start_invalid"),
        (st(b'{"a":1} \x00'), "header_json_invalid"),
        (st(b"\xff{}"), "header_start_invalid"),
        (st(b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
            b'"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}', bytes(4)),
         "header_json_invalid"),
        (st(b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,NaN]}}', bytes(4)),
         "header_json_invalid"),
        (st(b"[]"), "header_start_invalid"),
        (st({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "x": 1}}, bytes(4)),
         "tensor_entry_invalid"),
        (st({"a": {"dtype": "U16", "shape": [1], "data_offsets": [0, 2]}}, bytes(2)),
         "unsupported_dtype"),
        (st({"a": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}, bytes(4)),
         "tensor_shape_invalid"),
        (st({"a": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, bytes(4)),
         "tensor_shape_invalid"),
        (st({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}, bytes(4)),
         "tensor_offsets_invalid"),
        (st({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}, bytes(4)),
         "tensor_offsets_invalid"),
        (st({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, bytes(4)),
         "tensor_size_mismatch"),
        (st({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, bytes(8)),
         "data_not_contiguous"),
        (st({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
             "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]}}, bytes(6)),
         "data_not_contiguous"),
        (st({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, bytes(8)),
         "data_not_fully_covered"),
        (st({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, bytes(2)),
         "tensor_offsets_out_of_bounds"),
        (st({"__metadata__": {"k": 1}, "a": {"dtype": "U8", "shape": [1],
                                             "data_offsets": [0, 1]}}, b"\x00"),
         "metadata_invalid"),
        (st({"": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, b"\x00"),
         "tensor_name_invalid"),
    ],
)
def test_malformed_files_rejected(data, code):
    with pytest.raises(tm.TensorManifestError) as err:
        tm.parse_safetensors(data)
    assert err.value.code == code
    report = single(data)
    assert report["status"] == "blocked"
    assert code in codes(report)
    assert report["tensors"] == {}


def test_header_cap_checked_before_decoding():
    oversized = struct.pack("<Q", tm.MAX_HEADER_BYTES + 1) + b"{}"
    with pytest.raises(tm.TensorManifestError) as err:
        tm.parse_safetensors(oversized)
    assert err.value.code == "header_too_large"
    with pytest.raises(tm.TensorManifestError) as err:
        tm.parse_safetensors(FILE, max_header_bytes=104)
    assert err.value.code == "header_too_large"
    assert tm.parse_safetensors(FILE, max_header_bytes=105)["header_bytes"] == 105
    with pytest.raises(ValueError):
        tm.parse_safetensors(FILE, max_header_bytes=0)


def test_tensor_in_two_unindexed_files_rejected():
    report = tm.build_manifest(
        {"one.safetensors": FILE, "two.safetensors": FILE}, scope="synthetic"
    )
    assert report["status"] == "blocked"
    assert "tensor_in_multiple_files" in codes(report)


@pytest.mark.parametrize(
    "files",
    [
        {"../x.safetensors": FILE},
        {"/abs.safetensors": FILE},
        {"a//b.safetensors": FILE},
        {"A.safetensors": FILE, "a.safetensors": FILE},
        {},
    ],
)
def test_bad_file_names_rejected(files):
    with pytest.raises(ValueError):
        tm.build_manifest(files, scope="synthetic")


def test_scope_required_and_validated():
    with pytest.raises(ValueError):
        tm.build_manifest({"m.safetensors": FILE}, scope="real")


# --- sharded index -----------------------------------------------------------


def shards():
    one = st({"w1": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}, b"\x01\x02")
    two = st({"w2": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}}, b"\xff")
    return one, two


def index(weight_map, total_size=3, extra=None):
    body = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    if extra:
        body.update(extra)
    return json.dumps(body).encode()


def test_sharded_manifest_known_answer():
    one, two = shards()
    idx = index({"w1": "s-1.safetensors", "w2": "s-2.safetensors"})
    report = tm.build_manifest(
        {"s-1.safetensors": one, "s-2.safetensors": two, "model.safetensors.index.json": idx},
        scope="synthetic",
        index_name="model.safetensors.index.json",
    )
    assert report["status"] == "passed", report["issues"]
    assert report["layout"] == "sharded"
    assert report["tensors"]["w1"]["file"] == "s-1.safetensors"
    assert report["tensors"]["w1"]["sha256"] == hashlib.sha256(b"\x01\x02").hexdigest()
    assert report["tensors"]["w2"]["file"] == "s-2.safetensors"
    assert report["index"]["total_size_declared"] == 3
    assert report["index"]["total_size_observed"] == 3
    assert report["index"]["sha256"] == hashlib.sha256(idx).hexdigest()


@pytest.mark.parametrize(
    "weight_map, total, code",
    [
        ({"w1": "s-1.safetensors"}, 3, "tensor_not_in_index"),
        ({"w1": "s-1.safetensors", "w2": "s-1.safetensors"}, 3, "index_shard_mismatch"),
        ({"w1": "s-1.safetensors", "w2": "s-2.safetensors", "w3": "s-2.safetensors"}, 3,
         "indexed_tensor_missing"),
        ({"w1": "s-1.safetensors", "w2": "../s-2.safetensors"}, 3, "index_invalid"),
        ({"w1": "s-1.safetensors", "w2": "s-3.safetensors"}, 3, "index_shard_missing"),
    ],
)
def test_sharded_index_negative_controls(weight_map, total, code):
    one, two = shards()
    report = tm.build_manifest(
        {
            "s-1.safetensors": one,
            "s-2.safetensors": two,
            "model.safetensors.index.json": index(weight_map, total),
        },
        scope="synthetic",
        index_name="model.safetensors.index.json",
    )
    assert report["status"] == "blocked"
    assert code in codes(report)


def test_total_size_mismatch_is_a_warning_not_identity_failure():
    one, two = shards()
    report = tm.build_manifest(
        {
            "s-1.safetensors": one,
            "s-2.safetensors": two,
            "i.json": index({"w1": "s-1.safetensors", "w2": "s-2.safetensors"}, 4),
        },
        scope="synthetic",
        index_name="i.json",
    )
    assert report["status"] == "passed"
    (issue,) = report["issues"]
    assert issue["code"] == "index_total_size_mismatch" and issue["severity"] == "warning"
    assert report["index"]["total_size_declared"] == 4
    assert report["index"]["total_size_observed"] == 3


def test_index_without_total_size_and_unknown_key():
    one, two = shards()
    files = {"s-1.safetensors": one, "s-2.safetensors": two}
    files["i.json"] = json.dumps(
        {"weight_map": {"w1": "s-1.safetensors", "w2": "s-2.safetensors"}}
    ).encode()
    report = tm.build_manifest(files, scope="synthetic", index_name="i.json")
    assert report["status"] == "passed" and report["index"]["total_size_declared"] is None
    files["i.json"] = index({"w1": "s-1.safetensors", "w2": "s-2.safetensors"}, 3, {"x": 1})
    report = tm.build_manifest(files, scope="synthetic", index_name="i.json")
    assert "index_invalid" in codes(report)


def test_tensor_in_two_shards_rejected():
    one, _ = shards()
    report = tm.build_manifest(
        {
            "s-1.safetensors": one,
            "s-2.safetensors": one,
            "model.safetensors.index.json": index({"w1": "s-1.safetensors"}, 2),
        },
        scope="synthetic",
        index_name="model.safetensors.index.json",
    )
    assert report["status"] == "blocked"
    assert "tensor_in_multiple_files" in codes(report)


def test_index_duplicate_key_and_unlisted_file_rejected():
    one, two = shards()
    dup = (
        b'{"metadata":{"total_size":3},"weight_map":{"w1":"s-1.safetensors",'
        b'"w1":"s-1.safetensors","w2":"s-2.safetensors"}}'
    )
    report = tm.build_manifest(
        {"s-1.safetensors": one, "s-2.safetensors": two, "i.json": dup},
        scope="synthetic",
        index_name="i.json",
    )
    assert "index_invalid" in codes(report)
    report = tm.build_manifest(
        {
            "s-1.safetensors": one,
            "s-2.safetensors": two,
            "extra.safetensors": FILE,
            "i.json": index({"w1": "s-1.safetensors", "w2": "s-2.safetensors"}),
        },
        scope="synthetic",
        index_name="i.json",
    )
    assert "file_not_in_index" in codes(report)
    with pytest.raises(ValueError):
        tm.build_manifest({"s-1.safetensors": one}, scope="synthetic", index_name="i.json")


# --- comparison ----------------------------------------------------------------


def test_reexport_control_tensor_identical_files_differ():
    reordered = (
        b'{"__metadata__":{"format":"pt"},'
        b'"b":{"dtype":"U8","shape":[],"data_offsets":[0,1]},'
        b'"a":{"dtype":"F32","shape":[2],"data_offsets":[1,9]}}'
    )
    other = st(reordered, b"\x07" + bytes.fromhex("0000803f000000c0"))
    a = single(FILE)
    b = single(other, name="exported.safetensors")
    result = tm.compare_manifests(a, b)
    assert result["status"] == "passed", result["issues"]
    assert result["tensors_identical"] is True
    assert result["files_identical"] is False
    assert result["metadata_identical"] is False
    assert result["relation"] == "tensors_identical_files_differ"
    assert result["identical"] == ["a", "b"]
    assert result["only_in_a"] == [] and result["only_in_b"] == [] and result["changed"] == []
    assert a["tensor_set_sha256"] == b["tensor_set_sha256"]


def test_identical_files():
    result = tm.compare_manifests(single(FILE), single(FILE))
    assert result["relation"] == "tensors_and_files_identical"
    assert result["files_identical"] is True


def test_same_bytes_different_dtype_or_shape_is_not_identical():
    base = {"t": {"dtype": "I32", "shape": [2], "data_offsets": [0, 8]}}
    as_f32 = {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    as_2x1 = {"t": {"dtype": "I32", "shape": [2, 1], "data_offsets": [0, 8]}}
    raw = bytes(range(8))
    a = single(st(base, raw))
    for other, field in ((as_f32, "dtype"), (as_2x1, "shape")):
        result = tm.compare_manifests(a, single(st(other, raw)))
        assert result["tensors_identical"] is False
        assert result["relation"] == "tensors_differ"
        (change,) = result["changed"]
        assert change["name"] == "t" and change["fields"] == [field]
        assert result[f"{field}_changed"] == ["t"]
        assert result["bytes_changed"] == []


def test_only_in_and_bytes_changed():
    a = single(st({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                   "y": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}, b"\x01\x02"))
    b = single(st({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                   "z": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}, b"\x09\x02"))
    result = tm.compare_manifests(a, b)
    assert result["only_in_a"] == ["y"]
    assert result["only_in_b"] == ["z"]
    assert result["bytes_changed"] == ["x"]
    assert result["changed"][0]["fields"] == ["sha256"]
    assert result["tensors_identical"] is False


def test_compare_refuses_blocked_tampered_or_unknown_manifests():
    good = single(FILE)
    blocked = single(b"\x00")
    result = tm.compare_manifests(good, blocked)
    assert result["status"] == "blocked" and "manifest_blocked" in codes(result)
    assert result["tensors_identical"] is None and result["relation"] is None

    tampered = json.loads(json.dumps(good))
    tampered["tensors"]["a"]["sha256"] = SHA_B
    result = tm.compare_manifests(good, tampered)
    assert "manifest_inconsistent" in codes(result)
    assert result["tensors_identical"] is None

    future = json.loads(json.dumps(good))
    future["schema_version"] = 2
    assert "manifest_invalid" in codes(tm.compare_manifests(good, future))

    other_scope = tm.build_manifest({"m.safetensors": FILE}, scope="actual")
    assert "scope_mismatch" in codes(tm.compare_manifests(good, other_scope))
    assert "manifest_invalid" in codes(tm.compare_manifests(good, {"artifact_kind": "x"}))


def test_manifest_is_deterministic_and_order_invariant():
    one, two = shards()
    files = {"s-2.safetensors": two, "s-1.safetensors": one}
    first = tm.build_manifest(files, scope="synthetic")
    second = tm.build_manifest(dict(reversed(list(files.items()))), scope="synthetic")
    assert first == second
    assert [f["name"] for f in first["files"]] == ["s-1.safetensors", "s-2.safetensors"]


# --- directory reader ------------------------------------------------------------


def test_directory_reader_matches_bytes(tmp_path):
    (tmp_path / "m.safetensors").write_bytes(FILE)
    report = tm.build_manifest_from_directory(tmp_path, ["m.safetensors"], scope="synthetic")
    assert report == single(FILE, name="m.safetensors")


def test_directory_reader_fixture(tmp_path):
    (tmp_path / "tiny.safetensors").write_bytes(FILE)
    report = tm.build_manifest_from_directory(tmp_path, ["tiny.safetensors"], scope="synthetic")
    assert report["status"] == "passed"
    assert report["files"][0]["sha256"] == SHA_FILE


def test_directory_reader_index_only(tmp_path):
    one, two = shards()
    (tmp_path / "s-1.safetensors").write_bytes(one)
    (tmp_path / "s-2.safetensors").write_bytes(two)
    (tmp_path / "i.json").write_bytes(index({"w1": "s-1.safetensors", "w2": "s-2.safetensors"}))
    report = tm.build_manifest_from_directory(tmp_path, None, scope="synthetic", index_name="i.json")
    assert report["status"] == "passed", report["issues"]
    assert sorted(report["tensors"]) == ["w1", "w2"]


def test_directory_reader_rejects_symlink_escape_and_non_regular(tmp_path):
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(FILE)
    root = tmp_path / "ckpt"
    root.mkdir()
    os.symlink(outside, root / "m.safetensors")
    (root / "sub").mkdir()
    os.symlink(tmp_path, root / "link")
    for name in ("m.safetensors", "sub", "link/outside.safetensors", "missing.safetensors"):
        with pytest.raises(ValueError):
            tm.build_manifest_from_directory(root, [name], scope="synthetic")
    with pytest.raises(ValueError):
        tm.build_manifest_from_directory(root, ["../outside.safetensors"], scope="synthetic")
    with pytest.raises(ValueError):
        tm.build_manifest_from_directory(root, None, scope="synthetic")


def test_directory_reader_size_cap(tmp_path):
    (tmp_path / "m.safetensors").write_bytes(FILE)
    with pytest.raises(ValueError):
        tm.build_manifest_from_directory(
            tmp_path, ["m.safetensors"], scope="synthetic", max_file_bytes=121
        )


BANNED_IMPORTS = {"torch", "safetensors"}


def _imported_roots(tree):
    """Root package of every import statement or literal dynamic import, at any depth."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            arg = node.args[0]
            if name in ("__import__", "import_module") and isinstance(arg, ast.Constant) \
                    and isinstance(arg.value, str):
                roots.add(arg.value.split(".")[0])
    return roots


def test_import_scan_sees_nested_and_dynamic_imports():
    source = (
        "def f():\n    import torch.nn\n"
        "class C:\n    def g(self):\n        from safetensors.torch import load\n"
        "x = __import__('torch')\nimport importlib\ny = importlib.import_module('safetensors')\n"
    )
    assert _imported_roots(ast.parse(source)) >= BANNED_IMPORTS


def test_module_never_imports_torch_or_safetensors():
    """Order-independent: an AST scan plus a fresh isolated interpreter (A6R-2)."""
    tree = ast.parse(Path(tm.__file__).read_text(), filename=tm.__file__)
    assert not _imported_roots(tree) & BANNED_IMPORTS
    repo = str(Path(tm.__file__).resolve().parents[1])
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import gate.tensor_manifest as m; "
        "print(m.__file__); print(sorted(n for n in ('torch', 'safetensors') if n in sys.modules)); "
        "print(sys.dont_write_bytecode)"
    )
    gate_dir = Path(repo) / "gate"
    before = _pycache_entries(gate_dir)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, repo],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded_from, loaded, no_bytecode = result.stdout.splitlines()
    assert Path(loaded_from).resolve() == Path(tm.__file__).resolve()
    assert loaded == "[]"
    # A6R2: -I implies -E, so PYTHONDONTWRITEBYTECODE is ignored; the probe must pass -B
    # itself and leave no new bytecode in the checkout.
    assert no_bytecode == "True"
    assert _pycache_entries(gate_dir) - before == set()


def _pycache_entries(root):
    return {str(p.relative_to(root)) for d in root.rglob("__pycache__") for p in d.rglob("*")}


# --- review regressions (A6-F1, F4, F5, F11) ---------------------------------------


def _ignore_patterns():
    patterns = []
    for line in GITIGNORE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!")) or line.endswith("/"):
            continue
        patterns.append(line)
    return patterns


def test_no_module_fixture_is_git_ignored():
    """A6-F1: a fixture matching .gitignore never reaches CI or a fresh clone."""
    patterns = _ignore_patterns()
    files = [p for p in FIXTURES.rglob("*") if p.is_file()] if FIXTURES.exists() else []
    ignored = [
        str(p.relative_to(FIXTURES))
        for p in files
        if any(fnmatch.fnmatch(p.name, pat) for pat in patterns)
    ]
    assert ignored == []


def _open_fds():
    gc.collect()  # free any traceback that still references a mapping
    return len(os.listdir("/dev/fd"))


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd")
@pytest.mark.parametrize("bad", [0, True, -1, 1.5])
def test_directory_reader_bad_header_cap_is_value_error_without_fd_leak(tmp_path, bad):
    """A6-F4: caller misuse raises ValueError and leaks no descriptor."""
    (tmp_path / "m.safetensors").write_bytes(FILE)
    before = _open_fds()
    for _ in range(3):
        with pytest.raises(ValueError):
            tm.build_manifest_from_directory(
                tmp_path, ["m.safetensors"], scope="synthetic", max_header_bytes=bad
            )
    assert _open_fds() == before
    with pytest.raises(ValueError):
        tm.build_manifest({"m.safetensors": FILE}, scope="synthetic", max_header_bytes=bad)


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd")
def test_directory_reader_closes_descriptors_when_a_view_is_still_held(tmp_path, monkeypatch):
    """A6-F4: an exception whose traceback still holds a buffer view closes every fd."""
    (tmp_path / "m.safetensors").write_bytes(FILE)
    (tmp_path / "n.safetensors").write_bytes(FILE)

    def exploding(files, **kwargs):
        held = [memoryview(buffer) for buffer in files.values()]  # noqa: F841
        raise RuntimeError("boom")

    monkeypatch.setattr(tm, "build_manifest", exploding)
    before = _open_fds()
    for _ in range(3):
        with pytest.raises(RuntimeError, match="boom"):
            tm.build_manifest_from_directory(
                tmp_path, ["m.safetensors", "n.safetensors"], scope="synthetic"
            )
    assert _open_fds() == before


def test_index_bytes_are_part_of_file_identity():
    """A6-F5: identical shards with a different index are not identical files."""
    one, two = shards()
    weight_map = {"w1": "s-1.safetensors", "w2": "s-2.safetensors"}
    a = tm.build_manifest(
        {"s-1.safetensors": one, "s-2.safetensors": two, "i.json": index(weight_map, 3)},
        scope="synthetic",
        index_name="i.json",
    )
    b = tm.build_manifest(
        {"s-1.safetensors": one, "s-2.safetensors": two, "i.json": index(weight_map, 4)},
        scope="synthetic",
        index_name="i.json",
    )
    assert a["index"]["sha256"] != b["index"]["sha256"]
    result = tm.compare_manifests(a, b)
    assert result["status"] == "passed", result["issues"]
    assert result["tensors_identical"] is True
    assert result["files_identical"] is False
    assert result["relation"] == "tensors_identical_files_differ"
    assert {"name": "i.json", "sha256": a["index"]["sha256"]} in result["files"]["a"]
    same = tm.compare_manifests(a, a)
    assert same["files_identical"] is True
    assert same["relation"] == "tensors_and_files_identical"


@pytest.mark.parametrize(
    "header", [b"{}", b'{"__metadata__":{"format":"pt"}}']
)
def test_zero_tensor_manifest_is_not_a_pass(header):
    """A6-F11: an empty tensor set is absence, never a passed or identical manifest."""
    report = single(st(header))
    assert report["status"] == "blocked"
    assert "empty_tensor_set" in codes(report)
    assert report["tensor_set_sha256"] is None
    result = tm.compare_manifests(report, single(st(b"{}")))
    assert result["status"] == "blocked"
    assert result["tensors_identical"] is None


def test_comparison_refuses_a_passed_looking_manifest_with_no_tensors():
    """G4R item 4: an empty tensor set is refused at comparison even when the manifest
    claims ``passed`` with a self-consistent ``tensor_set_sha256`` (forged or foreign)."""
    forged = json.loads(json.dumps(single(FILE)))
    forged["tensors"] = {}
    forged["totals"].update(tensor_count=0, tensor_bytes=0)
    forged["tensor_set_sha256"] = tm._tensor_set_sha256({})
    assert forged["status"] == "passed"
    result = tm.compare_manifests(forged, forged)
    assert result["status"] == "blocked"
    assert [(i["path"], i["code"]) for i in result["issues"]] == [
        ("a", "empty_tensor_set"), ("b", "empty_tensor_set")]
    assert result["tensors_identical"] is None and result["relation"] is None
    good = single(FILE)
    assert tm.compare_manifests(good, good)["status"] == "passed"  # negative control
