"""Independent pinned-source receipts and binary32 consumer known answers."""

import hashlib
import json
import re
import tomllib
from pathlib import Path

import numpy as np

from gate.s2_asl import inspect_asl
from tests.s2_asl_fixture import actor_event, framed, metadata

ROOT = Path(__file__).parent / "fixtures" / "alpasim_asl" / "quaternion_compat"


def test_full_quaternion_consumer_and_dependency_source_receipts():
    for folder in (ROOT, ROOT / "glam"):
        manifest = json.loads((folder / "provenance.json").read_text())
        entries = [*manifest["sources"]]
        if "license_file" in manifest:
            entries.append(manifest["license_file"])
        for entry in entries:
            data = (folder / entry["file"]).read_bytes()
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            assert hashlib.sha1(blob).hexdigest() == entry["git_blob_sha"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    lock = tomllib.loads((ROOT / "glam" / "alpasim_Cargo.lock.txt").read_text())
    package = next(item for item in lock["package"] if item["name"] == "glam")
    manifest = json.loads((ROOT / "glam" / "provenance.json").read_text())
    assert package["version"] == "0.29.3"
    assert package["checksum"] == manifest["crate_checksum"]
    assert manifest["dependency_commit"] == "d36e7eeff05338c56c4aa8d59fc2615e7963b1b7"


def _pinned_constants():
    source = (ROOT / "array_utils.rs.txt").read_text()
    unit = re.search(r"const UNIT_QUAT_LEN_SQ_TOL: f32 = ([^;]+);", source)
    near_zero = re.search(r"if len_sq < ([^ ]+) \{", source)
    assert unit and near_zero
    return np.float32(unit.group(1)), np.float32(near_zero.group(1))


def test_binary32_reference_catches_backend_boundary_without_normalizing_values():
    tolerance, near_zero = _pinned_constants()
    rng = np.random.default_rng(1729)
    directions = rng.normal(size=(64, 4))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    boundary = directions * np.sqrt(1 + 1e-4)
    values = np.vstack([
        [[1, 0, 0, 0], [0, 0, 0, 0], [1e-7, 0, 0, 0], [2, 0.5, -0.25, 0],
         [1.0000499486923218, 0, 0, 0], [1.0000500679016113, 0, 0, 0],
         [0.009820855222642422, -0.6184602975845337,
          -0.5168955326080322, -0.5918863415718079],
         [0.07397221773862839, -0.6074876189231873,
          0.4539276957511902, -0.6477164030075073], [3e38, 3e38, 3e38, 3e38]],
        boundary,
    ]).astype(np.float32)
    # Independent NumPy binary32 arithmetic; source uses xyzw internally.
    with np.errstate(over="ignore"):
        xyzw = values[:, [1, 2, 3, 0]]
        squares = xyzw * xyzw
        x, y, z, w = squares.T
        norms = np.vstack([(x + z) + (y + w), ((x + y) + z) + w, (x + y) + (z + w)])
        accepted = (norms >= near_zero) & (np.abs(norms - np.float32(1)) <= tolerance)
    assert accepted[:, 6].tolist() == [True, False, False]
    assert accepted[:, 7].tolist() == [False, False, True]
    seen = set()
    for index, quaternion in enumerate(values):
        result = inspect_asl(
            framed(metadata()) + framed(actor_event(1000, quaternion=quaternion)),
            expected_session_uuid="rollout-1", expected_scene_id="scene-1", requested_steps=2,
        )
        decisions = accepted[:, index]
        expected = bool(decisions.all())
        assert result["inspection_valid"] is expected
        expected_code = (
            None if expected else "quaternion_rounding_ambiguous" if decisions.any()
            else "upstream_quaternion_incompatible"
        )
        assert {item["code"] for item in result["issues"]} == (
            set() if expected_code is None else {expected_code})
        assert result["pose_events"][0]["actors"][0]["pose"]["quaternion_wxyz"] == (
            quaternion.tolist())
        seen.add(expected_code)
    assert seen == {None, "quaternion_rounding_ambiguous", "upstream_quaternion_incompatible"}
