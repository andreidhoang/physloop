"""Independent serialization and local artifact integrity controls for S2."""

import hashlib
from pathlib import Path

import pytest

from gate.s2_identity import (
    canonical_json_bytes,
    identity_hash,
    read_artifact,
    read_json_bytes,
)


def receipt(path="source.json", data=b"{}"):
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def test_canonical_encoding_has_literal_golden_bytes():
    value = {"z": [True, None, 3], "a": {"name": "đường", "x": 1.5}}
    assert canonical_json_bytes(value) == (
        b'{"a":{"name":"\xc4\x91\xc6\xb0\xe1\xbb\x9dng","x":1.5},"z":[true,null,3]}'
    )
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_identity_hash_has_independent_sha256_golden():
    assert identity_hash({}) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert identity_hash({"b": 2, "a": 1}) == identity_hash({"a": 1, "b": 2})


def test_configuration_rope_and_content_changes_change_identity():
    baseline = {"steps": 5, "rope_offset": 16, "source": "abc"}
    for key, value in (("steps", 10), ("rope_offset", 17), ("source", "abd")):
        assert identity_hash({**baseline, key: value}) != identity_hash(baseline)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"), float("inf"), float("-inf"), {1: "bad"}, (1, 2), {1, 2},
        b"bytes", Path("source"), {"nested": [float("nan")]}, {"x": "\ud800"},
    ],
)
def test_canonical_encoding_rejects_non_json_native_values(value):
    with pytest.raises(ValueError):
        canonical_json_bytes(value)


def test_recursive_container_is_rejected_as_value_error():
    value = []
    value.append(value)
    with pytest.raises(ValueError):
        canonical_json_bytes(value)


def test_parser_reads_strict_utf8_and_preserves_json_native_types():
    assert read_json_bytes('{"name":"đường","flag":true,"count":2}'.encode()) == {
        "name": "đường", "flag": True, "count": 2,
    }


@pytest.mark.parametrize(
    "data",
    [
        b'{"a":1,"a":2}', b'{"a":1,"\\u0061":2}', b'{"x":{"a":1,"a":2}}',
        b'NaN', b'Infinity', b'-Infinity', b'1e9999', b'{"x":[1e9999]}',
        b'"\xff"', b'{} trailing', b'{', b'"\\ud800"',
    ],
)
def test_parser_rejects_ambiguous_nonfinite_or_invalid_data(data):
    with pytest.raises(ValueError):
        read_json_bytes(data)


@pytest.mark.parametrize("data", ["{}", bytearray(b"{}"), None])
def test_parser_requires_bytes(data):
    with pytest.raises(ValueError):
        read_json_bytes(data)


def test_artifact_returns_exact_verified_bytes(tmp_path):
    data = b' {"value": 1} \n'
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "source.json").write_bytes(data)
    assert read_artifact(tmp_path, receipt("nested/source.json", data)) == data


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "0" * 64}, {"sha256": "A" * 64}, {"sha256": "0" * 63},
        {"sha256": None}, {"size_bytes": 1}, {"size_bytes": -1},
        {"size_bytes": True}, {"size_bytes": 2.0}, {"extra": True},
        {"path": ""}, {"path": None}, {"path": ["source.json"]},
        {"path": "/source.json"}, {"path": "../source.json"},
        {"path": "nested/../source.json"}, {"path": "source.json\x00"},
    ],
)
def test_artifact_rejects_malformed_or_mismatched_receipts(tmp_path, change):
    (tmp_path / "source.json").write_bytes(b"{}")
    with pytest.raises((ValueError, OSError)):
        read_artifact(tmp_path, {**receipt(), **change})


@pytest.mark.parametrize("missing", ["path", "sha256", "size_bytes"])
def test_artifact_requires_each_receipt_field(tmp_path, missing):
    value = receipt()
    del value[missing]
    with pytest.raises(ValueError):
        read_artifact(tmp_path, value)


@pytest.mark.parametrize("value", [None, [], "source.json"])
def test_artifact_requires_receipt_object(tmp_path, value):
    with pytest.raises(ValueError):
        read_artifact(tmp_path, value)


def test_artifact_detects_same_size_content_tampering(tmp_path):
    (tmp_path / "source.json").write_bytes(b"[]")
    with pytest.raises(ValueError):
        read_artifact(tmp_path, receipt())


def test_artifact_refuses_symlink_file_even_with_matching_bytes(tmp_path):
    (tmp_path / "real.json").write_bytes(b"{}")
    (tmp_path / "source.json").symlink_to("real.json")
    with pytest.raises((ValueError, OSError)):
        read_artifact(tmp_path, receipt())


def test_artifact_refuses_symlink_directory_even_inside_root(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "source.json").write_bytes(b"{}")
    (tmp_path / "alias").symlink_to("real", target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        read_artifact(tmp_path, receipt("alias/source.json"))


def test_artifact_refuses_symlink_root(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "source.json").write_bytes(b"{}")
    (tmp_path / "alias").symlink_to("real", target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        read_artifact(tmp_path / "alias", receipt())


def test_artifact_refuses_outside_symlink(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "source.json").symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        read_artifact(root, receipt())


def test_artifact_refuses_missing_or_nonregular_file(tmp_path):
    with pytest.raises(OSError):
        read_artifact(tmp_path, receipt())
    (tmp_path / "source.json").mkdir()
    with pytest.raises((ValueError, OSError)):
        read_artifact(tmp_path, receipt())
