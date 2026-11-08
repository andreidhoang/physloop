"""Known answers for exact public AlpaSim geometry helpers; no simulator proof.

The checked-in source is unmodified. Only its two geometry function AST nodes are
compiled, so importing the full evaluator, datasets and GPU libraries is unnecessary.
No local geometry implementation substitutes for either upstream version.
"""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point

FIXTURES = Path(__file__).parent / "fixtures" / "alpasim"
SOURCES = {
    "fixed": (
        "ground_truth_e121e591.py.txt",
        "5bd239cbe30fee33aeff44c86cf43a05da0ec5095068c86f2ef14f4c102c76c4",
    ),
    "before_fix": (
        "ground_truth_20235345.py.txt",
        "cdc7c9e2a3101b74335ef381f98088fc901f8e264e44f2bf8e89faa27fc45b11",
    ),
}
HELPERS = {"_end_direction", "_lateral_distance_to_gt"}


def _load_helpers(version):
    filename, digest = SOURCES[version]
    path = FIXTURES / filename
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == digest, "upstream source bytes changed"
    tree = ast.parse(source, filename=str(path))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]
    assert {node.name for node in functions} == HELPERS
    assert len(functions) == len(HELPERS)
    namespace = {
        "__builtins__": {"abs": abs, "float": float, "len": len, "min": min},
        "np": np,
        "LineString": LineString,
        "Point": Point,
    }
    # Execute only named helpers from locally pinned, hash-checked public source.
    code = compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec")
    exec(code, namespace)  # noqa: S102
    return namespace["_end_direction"], namespace["_lateral_distance_to_gt"]


@pytest.fixture(scope="module")
def fixed():
    return _load_helpers("fixed")


@pytest.fixture(scope="module")
def before_fix():
    return _load_helpers("before_fix")


@pytest.mark.parametrize("version", SOURCES)
def test_source_snapshot_matches_provenance_and_git_blob(version):
    filename, digest = SOURCES[version]
    content = (FIXTURES / filename).read_bytes()
    metadata = json.loads((FIXTURES / "provenance.json").read_text())[version]
    assert metadata["file"] == filename
    assert metadata["sha256"] == digest == hashlib.sha256(content).hexdigest()
    git_blob = b"blob " + str(len(content)).encode() + b"\0" + content
    assert hashlib.sha1(git_blob).hexdigest() == metadata["git_blob_sha"]
    assert metadata["commit"] in metadata["source_url"]
    assert content.startswith(b"# SPDX-License-Identifier: Apache-2.0\n")
    assert b"Copyright (c) 2025-2026 NVIDIA Corporation" in content.splitlines()[1]


def test_upstream_license_is_preserved():
    assert hashlib.sha256((FIXTURES / "LICENSE").read_bytes()).hexdigest() == (
        "50201750cc0542d4656d8d95f002f78357021b7c3c975b53924000b7ddc5a24e"
    )


@pytest.mark.parametrize(("coords", "xy"), [
    ([(0, 0), (10, 0), (9.99, 0), (9.99, 0.000002)], (15, 0)),
    ([(0, 0), (10, 0), (10, 0.000002)], (15, 0.000002)),
])
def test_issue_198_original_cases_reproduce_bug_then_fix(fixed, before_fix, coords, xy):
    gt, point = LineString(coords), Point(xy)
    old_direction, old_distance = before_fix
    direction, distance = fixed
    assert old_distance(gt, point, gt.length, old_direction(gt)) == pytest.approx(5.0)
    # Tiny residual from the meaningful direction is retained; do not round to zero.
    assert distance(gt, point, direction(gt)) == pytest.approx(0.0, abs=1e-5)


def test_direction_requires_at_least_one_metre_displacement(fixed):
    direction, _ = fixed
    assert direction(LineString([(0, 0), (0.999, 0)])) is None
    np.testing.assert_array_equal(direction(LineString([(0, 0), (1, 0)])), [1.0, 0.0])
    assert direction(LineString([(2, 3), (2, 3)])) is None


def test_duplicate_stationary_tail_uses_earlier_supported_motion(fixed):
    direction, distance = fixed
    gt = LineString([(0, 0), (10, 0), (10, 0)])
    np.testing.assert_array_equal(direction(gt), [1.0, 0.0])
    assert distance(gt, Point(15, 0), direction(gt)) == 0.0


@pytest.mark.parametrize(("xy", "expected"), [
    ((15, 0), 0.0),       # Forward overshoot along the endpoint ray.
    ((10, 4), 4.0),       # Exactly perpendicular at the endpoint.
    ((5, 3), 3.0),        # Along the interior of the recorded path.
    ((-5, 0), 5.0),       # Behind the start; never extend the endpoint ray backwards.
    ((15, 3.999), 3.999),
    ((15, 4.0), 4.0),
    ((15, 4.001), 4.001),
])
def test_endpoint_ray_preserves_genuine_lateral_distances(fixed, xy, expected):
    direction, distance = fixed
    gt = LineString([(0, 0), (10, 0)])
    assert distance(gt, Point(xy), direction(gt)) == pytest.approx(expected)


def test_short_stationary_path_has_no_invented_endpoint_ray(fixed):
    direction, distance = fixed
    gt = LineString([(0, 0), (0.5, 0)])
    assert distance(gt, Point(5, 0), direction(gt)) == 4.5


def test_path_union_keeps_an_earlier_recorded_position(fixed):
    direction, distance = fixed
    gt = LineString([(20, 5), (0, 0), (10, 0)])
    assert distance(gt, Point(20, 5), direction(gt)) == 0.0
