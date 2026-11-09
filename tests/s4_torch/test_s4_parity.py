"""Known answers for the S4 parity ladder (A10; E02, E04 numerics, E16 action level).

Oracles are independent of ``teacher_rl.parity``: ULP distances are counted from IEEE bit
patterns unpacked with ``struct`` (sign-magnitude order), absolute/relative errors are
hand-computed literals, and the rung order is the build-spec ladder (structure -> bitwise ->
elementwise -> action level). Tolerances are always supplied by the test as registered
inputs; the module never chooses one.
"""

import ast
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gate.s2_identity import identity_hash
from scripts import s4_parity_check
from teacher_rl import parity

ROOT = Path(__file__).resolve().parents[2]


def _tol(**kw):
    base = {"registration_ref": "experiments/2026-11-09-s4-parity.md#tolerances",
            "atol": None, "rtol": None, "max_ulp": None}
    base.update(kw)
    return base


def _ordered(bits, width):
    sign = 1 << (width - 1)
    return -(bits & (sign - 1)) if bits & sign else bits


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _ladder(ref, cand, tolerance=None):
    return parity.compare_tensors(ref, cand, tolerance=tolerance, scope="synthetic")


# ---------------------------------------------------------------------------
# Rungs 1-2: structure and bitwise identity
# ---------------------------------------------------------------------------


def test_identical_tensors_stop_at_bitwise_rung():
    ref = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3], dtype=torch.int64)}
    report = _ladder(ref, {k: v.clone() for k, v in ref.items()})
    assert report["artifact_kind"] == "s4_parity_ladder" and report["schema_version"] == 1
    assert report["verdict"] == "identical" and report["status"] == "passed"
    assert report["rungs"]["structure"]["status"] == "passed"
    assert report["rungs"]["bitwise"] == {"status": "passed", "identical": ["a", "b"],
                                          "differing": [], "cast_undetermined": []}
    assert report["rungs"]["elementwise"] == {"status": "not_applicable", "tensors": {}}
    assert report["rungs"]["action_level"]["status"] == "qualify_on_host"
    assert report["tolerance"] is None


@pytest.mark.parametrize(
    ("cand", "field"),
    [
        ({"a": torch.tensor([1.0, 2.0])}, "missing_in_candidate"),
        ({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([1.0]), "c": torch.tensor([0.0])},
         "unexpected_in_candidate"),
        ({"a": torch.tensor([1.0, 2.0], dtype=torch.bfloat16), "b": torch.tensor([1.0])},
         "dtype_mismatch"),
        ({"a": torch.tensor([[1.0, 2.0]]), "b": torch.tensor([1.0])}, "shape_mismatch"),
    ],
)
def test_structure_mismatch_exceeds_without_numeric_rungs(cand, field):
    ref = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([1.0])}
    report = _ladder(ref, cand, _tol(atol=1e9))
    assert report["verdict"] == "exceeds"
    structure = report["rungs"]["structure"]
    assert structure["status"] == "blocked" and structure[field]
    assert report["rungs"]["bitwise"]["status"] == "not_run"
    assert report["rungs"]["elementwise"]["status"] == "not_run"
    assert "structure_mismatch" in {i["code"] for i in report["issues"]}
    assert report["status"] == "blocked"  # A10-F5: an error issue never leaves "passed"


def test_difference_without_tolerance_is_unregistered_but_facts_are_emitted():
    report = _ladder({"a": torch.tensor([1.0])},
                     {"a": torch.nextafter(torch.tensor([1.0]), torch.tensor([2.0]))})
    assert report["verdict"] == "tolerance_unregistered"
    entry = report["rungs"]["elementwise"]["tensors"]["a"]
    assert entry["max_ulp"] == 1 and entry["max_abs_error"] == 2.0**-23
    assert entry["verdict"] == "tolerance_unregistered" and entry["elements_exceeding"] is None


# ---------------------------------------------------------------------------
# Rung 3: per-element absolute/relative error and dtype-aware ULP distance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "a", "b", "ulp"),
    [
        (torch.float32, 1.0, float(torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))), 1),
        (torch.bfloat16, 1.0, 1.0078125, 1),
        (torch.float16, 1.0, 1.0 + 2.0**-10, 1),
        (torch.float32, 0.0, -0.0, 0),
        (torch.float32, 1.0, -1.0, 2 * _f32_bits(1.0)),
        (torch.float32, 2.0**-149, -(2.0**-149), 2),
        (torch.float64, 1e300, -1e300,
         2 * struct.unpack("<q", struct.pack("<d", 1e300))[0]),
    ],
)
def test_ulp_distance_literals(dtype, a, b, ulp):
    x, y = torch.tensor([a], dtype=dtype), torch.tensor([b], dtype=dtype)
    report = _ladder({"t": x}, {"t": y})
    entry = report["rungs"]["elementwise"]["tensors"]["t"]
    assert entry["max_ulp"] == ulp
    assert entry["max_abs_error"] == abs(a - b)


def test_ulp_oracle_cross_check_on_random_f32():
    gen = torch.Generator().manual_seed(3)
    a = torch.randn(64, generator=gen)
    b = a + torch.randn(64, generator=gen) * 1e-6
    entry = _ladder({"t": a}, {"t": b})["rungs"]["elementwise"]["tensors"]["t"]
    expected = max(
        abs(_ordered(_f32_bits(float(x)), 32) - _ordered(_f32_bits(float(y)), 32))
        for x, y in zip(a.tolist(), b.tolist(), strict=True)
    )
    assert entry["max_ulp"] == expected
    assert entry["max_abs_error"] == max(
        abs(float(x) - float(y)) for x, y in zip(a.tolist(), b.tolist(), strict=True)
    )


def test_ulp_tolerance_boundary():
    one = torch.tensor([1.0, 4.0])
    two = torch.tensor([float(torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))), 4.0])
    assert _ladder({"t": one}, {"t": two}, _tol(max_ulp=1))["verdict"] == (
        "within_registered_tolerance"
    )
    report = _ladder({"t": one}, {"t": two}, _tol(max_ulp=0))
    assert report["verdict"] == "exceeds"
    entry = report["rungs"]["elementwise"]["tensors"]["t"]
    assert entry["elements_exceeding"] == 1 and entry["first_exceeding_index"] == 0
    assert entry["differing_elements"] == 1


def test_relative_and_absolute_criteria():
    ref = {"t": torch.tensor([2.0, 0.0, 5.0])}
    cand = {"t": torch.tensor([2.5, 0.0, 5.0])}
    entry = _ladder(ref, cand)["rungs"]["elementwise"]["tensors"]["t"]
    assert entry["max_abs_error"] == 0.5 and entry["max_rel_error"] == 0.25
    assert _ladder(ref, cand, _tol(rtol=0.25))["verdict"] == "within_registered_tolerance"
    assert _ladder(ref, cand, _tol(rtol=0.2))["verdict"] == "exceeds"
    assert _ladder(ref, cand, _tol(atol=0.5))["verdict"] == "within_registered_tolerance"
    assert _ladder(ref, cand, _tol(atol=0.4, rtol=0.05))["verdict"] == (
        "within_registered_tolerance"  # 0.5 <= 0.4 + 0.05 * 2
    )
    # Every registered criterion must hold: abs passes, ULP fails.
    assert _ladder(ref, cand, _tol(atol=1.0, max_ulp=1))["verdict"] == "exceeds"


def test_zero_reference_relative_error_is_undefined_not_zero():
    report = _ladder({"t": torch.tensor([0.0])}, {"t": torch.tensor([1e-9])})
    entry = report["rungs"]["elementwise"]["tensors"]["t"]
    assert entry["max_rel_error"] is None and entry["rel_error_reference_zero"] == 1
    assert _ladder({"t": torch.tensor([0.0])}, {"t": torch.tensor([1e-9])},
                   _tol(atol=1e-8))["verdict"] == "within_registered_tolerance"
    assert _ladder({"t": torch.tensor([0.0])}, {"t": torch.tensor([1e-9])},
                   _tol(rtol=10.0))["verdict"] == "exceeds"


def test_nonfinite_mismatch_never_within_tolerance():
    report = _ladder({"t": torch.tensor([1.0, 2.0])},
                     {"t": torch.tensor([float("nan"), 2.0])}, _tol(atol=1e30, rtol=1e30))
    entry = report["rungs"]["elementwise"]["tensors"]["t"]
    assert report["verdict"] == "exceeds" and entry["nonfinite_mismatch"] == 1
    assert entry["max_abs_error"] is None and entry["max_ulp"] is None
    inf = _ladder({"t": torch.tensor([float("inf")])}, {"t": torch.tensor([1.0])},
                  _tol(atol=1e30))
    assert inf["verdict"] == "exceeds"
    same_nan = torch.tensor([float("nan")])
    assert _ladder({"t": same_nan}, {"t": same_nan.clone()})["verdict"] == "identical"
    json.dumps(report, allow_nan=False)


def test_integer_extremes_do_not_wrap():
    low, high = -(2**63), 2**63 - 1
    report = _ladder({"t": torch.tensor([low, 0])}, {"t": torch.tensor([high, 0])})
    assert report["rungs"]["elementwise"]["tensors"]["t"]["max_abs_error"] == 2**64 - 1


def test_integer_tensors_admit_no_tolerance():
    report = _ladder({"t": torch.tensor([3, 4])}, {"t": torch.tensor([3, 5])},
                     _tol(atol=10.0, max_ulp=10))
    entry = report["rungs"]["elementwise"]["tensors"]["t"]
    assert report["verdict"] == "exceeds"
    assert entry["max_abs_error"] == 1 and entry["max_ulp"] is None


def test_mixed_tensors_overall_verdict_and_determinism():
    ref = {"same": torch.tensor([1.0]), "close": torch.tensor([1.0, 2.0])}
    cand = {"same": torch.tensor([1.0]),
            "close": torch.tensor([1.0, float(torch.nextafter(torch.tensor(2.0),
                                                              torch.tensor(3.0)))])}
    report = _ladder(ref, cand, _tol(max_ulp=2))
    assert report["verdict"] == "within_registered_tolerance"
    assert report["rungs"]["bitwise"]["identical"] == ["same"]
    assert report["rungs"]["bitwise"]["differing"] == ["close"]
    assert report == _ladder(ref, cand, _tol(max_ulp=2))
    assert report["tolerance"]["registration_ref"].startswith("experiments/")


@pytest.mark.parametrize(
    "tolerance",
    [
        {"atol": 1e-3, "rtol": None, "max_ulp": None},
        _tol(registration_ref="TBD", atol=1e-3),
        _tol(registration_ref="  ", atol=1e-3),
        _tol(),
        _tol(atol=-1.0),
        _tol(atol=float("nan")),
        _tol(rtol=float("inf")),
        _tol(max_ulp=True),
        _tol(max_ulp=1.5),
        _tol(max_ulp=-1),
        dict(_tol(atol=1e-3), extra=1),
        [1e-3],
    ],
)
def test_tolerance_validation_fails_closed(tolerance):
    with pytest.raises(ValueError):
        parity.validate_tolerance(tolerance)


def test_scope_and_input_validation():
    with pytest.raises(ValueError, match="scope"):
        parity.compare_tensors({"a": torch.tensor([1.0])}, {"a": torch.tensor([1.0])},
                               tolerance=None, scope="real")
    with pytest.raises(ValueError, match="empty"):
        parity.compare_tensors({}, {}, tolerance=None, scope="synthetic")


# ---------------------------------------------------------------------------
# Rung 4: action-level trajectories (L1 comparator; real inputs are L3)
# ---------------------------------------------------------------------------


def _traj_tol(value):
    return {"registration_ref": "experiments/2026-11-09-s4-parity.md#action",
            "max_displacement": value, "units": "m"}


def test_trajectory_known_answers():
    ref = {"rollout-1": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])}
    cand = {"rollout-1": torch.tensor([[0.0, 0.0], [1.0, 0.3], [2.4, 0.0]])}
    # float32(2.4) - 2 = 0.40000009...: a 0.4 m bound would (correctly) be exceeded.
    report = parity.compare_trajectories(ref, cand, tolerance=_traj_tol(0.41), scope="synthetic")
    assert report["artifact_kind"] == "s4_action_parity"
    entry = report["trajectories"]["rollout-1"]
    assert entry["waypoints"] == 3
    assert entry["max_displacement"] == pytest.approx(0.4, abs=1e-7)
    assert entry["final_displacement"] == pytest.approx(0.4, abs=1e-7)
    assert entry["argmax_waypoint"] == 2
    assert entry["mean_displacement"] == pytest.approx((0.3 + 0.4) / 3, abs=1e-7)
    assert report["verdict"] == "within_registered_tolerance"
    assert report["units"]["displacement"] == "m"
    tight = parity.compare_trajectories(ref, cand, tolerance=_traj_tol(0.35), scope="synthetic")
    assert tight["verdict"] == "exceeds"
    none = parity.compare_trajectories(ref, cand, tolerance=None, scope="synthetic")
    assert none["verdict"] == "tolerance_unregistered"
    same = parity.compare_trajectories(ref, {"rollout-1": ref["rollout-1"].clone()},
                                       tolerance=None, scope="synthetic")
    assert same["verdict"] == "identical"


def test_trajectory_structure_and_nonfinite_fail_closed():
    ref = {"r": torch.tensor([[0.0, 0.0], [1.0, 0.0]])}
    short = parity.compare_trajectories(ref, {"r": torch.tensor([[0.0, 0.0]])},
                                        tolerance=_traj_tol(1.0), scope="synthetic")
    assert short["verdict"] == "exceeds" and short["structure"]["shape_mismatch"] == ["r"]
    missing = parity.compare_trajectories(ref, {"q": ref["r"]}, tolerance=_traj_tol(1.0),
                                          scope="synthetic")
    assert missing["verdict"] == "exceeds"
    nan = parity.compare_trajectories(ref, {"r": torch.tensor([[0.0, 0.0], [float("nan"), 0]])},
                                      tolerance=_traj_tol(1e9), scope="synthetic")
    assert nan["verdict"] == "exceeds"
    assert nan["trajectories"]["r"]["nonfinite_waypoints"] == 1
    with pytest.raises(ValueError, match="units"):
        parity.compare_trajectories(ref, ref, tolerance={"registration_ref": "x.md",
                                                         "max_displacement": 1.0},
                                    scope="synthetic")
    with pytest.raises(ValueError, match="2-D"):
        parity.compare_trajectories({"r": torch.tensor([1.0])}, {"r": torch.tensor([1.0])},
                                    tolerance=None, scope="synthetic")


# ---------------------------------------------------------------------------
# Host script: manifest rungs torch-free, elementwise on loaded bytes verified by hash
# ---------------------------------------------------------------------------


def _write(directory, tensors):
    directory.mkdir()
    save_file(tensors, str(directory / "model.safetensors"))


def test_parity_script_end_to_end(tmp_path, capsys):
    _write(tmp_path / "ref", {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([5.0])})
    _write(tmp_path / "cand", {"a": torch.tensor([1.0, 2.0]),
                               "b": torch.nextafter(torch.tensor([5.0]), torch.tensor([6.0]))})
    (tmp_path / "tol.json").write_text(json.dumps(_tol(max_ulp=1)))
    argv = ["--reference-root", str(tmp_path / "ref"), "--reference-file", "model.safetensors",
            "--candidate-root", str(tmp_path / "cand"), "--candidate-file", "model.safetensors",
            "--scope", "synthetic"]
    assert s4_parity_check.main([*argv, "--tolerance", str(tmp_path / "tol.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "within_registered_tolerance"
    assert out["rungs"]["bitwise"]["identical"] == ["a"]
    assert out["rungs"]["elementwise"]["tensors"]["b"]["max_ulp"] == 1
    assert out["input_hashes"]["reference"] and out["input_hashes"]["candidate"]

    # A10-F5: a produced report whose verdict is not identical/within tolerance exits 3.
    assert s4_parity_check.main(argv) == 3
    assert json.loads(capsys.readouterr().out)["verdict"] == "tolerance_unregistered"

    (tmp_path / "bad.json").write_text(json.dumps(_tol()))
    assert s4_parity_check.main([*argv, "--tolerance", str(tmp_path / "bad.json")]) == 2


def test_parity_script_loader_rejects_bytes_changed_after_manifest(tmp_path):
    _write(tmp_path / "ref", {"a": torch.tensor([1.0])})
    _write(tmp_path / "cand", {"a": torch.tensor([2.0])})
    ref_m = s4_parity_check.manifest(tmp_path / "ref", files=["model.safetensors"], index=None,
                                     scope="synthetic")
    cand_m = s4_parity_check.manifest(tmp_path / "cand", files=["model.safetensors"],
                                      index=None, scope="synthetic")
    save_file({"a": torch.tensor([3.0])}, str(tmp_path / "cand" / "model.safetensors"))
    report = parity.ladder_from_manifests(
        ref_m, cand_m, tolerance=None,
        load=s4_parity_check.loader(tmp_path / "ref", tmp_path / "cand", ref_m, cand_m),
    )
    assert report["status"] == "blocked"
    assert "bytes_changed_since_manifest" in {i["code"] for i in report["issues"]}
    assert report["verdict"] is None


def test_manifest_ladder_without_loader_defers_elementwise_to_host(tmp_path):
    _write(tmp_path / "ref", {"a": torch.tensor([1.0])})
    _write(tmp_path / "cand", {"a": torch.tensor([2.0])})
    ref_m = s4_parity_check.manifest(tmp_path / "ref", files=["model.safetensors"], index=None,
                                     scope="synthetic")
    cand_m = s4_parity_check.manifest(tmp_path / "cand", files=["model.safetensors"],
                                      index=None, scope="synthetic")
    report = parity.ladder_from_manifests(ref_m, cand_m, tolerance=None, load=None)
    assert report["rungs"]["elementwise"]["status"] == "qualify_on_host"
    # A10-F5: a deferred rung is qualify_on_host, never passed.
    assert report["verdict"] is None and report["status"] == "qualify_on_host"


def test_parity_and_params_modules_import_torch_lazily():
    for path in ("teacher_rl/parity.py", "teacher_rl/params.py", "scripts/s4_parity_check.py"):
        tree = ast.parse((ROOT / path).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] not in {"torch", "safetensors"}
                           for a in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"torch", "safetensors"}, path


# ---------------------------------------------------------------------------
# Review regressions (A10-F3 .. A10-F6, A10-F8)
# ---------------------------------------------------------------------------


def test_trajectory_structure_mismatch_is_blocked():
    report = parity.compare_trajectories({"r": torch.zeros(2, 2)}, {"q": torch.zeros(2, 2)},
                                         tolerance=None, scope="synthetic")
    assert report["verdict"] == "exceeds" and report["status"] == "blocked"


def test_f64_overflow_is_reported_as_null_with_count_and_exceeds():
    # A10-F6: |c - r| overflows float64; the report stays strict JSON.
    ref = {"t": torch.tensor([1.7e308, 1.0], dtype=torch.float64)}
    cand = {"t": torch.tensor([-1.7e308, 2.0], dtype=torch.float64)}
    report = _ladder(ref, cand)
    entry = report["rungs"]["elementwise"]["tensors"]["t"]
    assert entry["max_abs_error"] is None and entry["abs_error_overflow"] == 1
    assert entry["max_rel_error"] is None and entry["rel_error_overflow"] == 0
    json.dumps(report, allow_nan=False)
    identity_hash(report)
    bounded = _ladder(ref, cand, _tol(atol=1e300, rtol=1.0))
    assert bounded["verdict"] == "exceeds"
    assert bounded["rungs"]["elementwise"]["tensors"]["t"]["elements_exceeding"] == 1
    tiny = _ladder({"t": torch.tensor([1e-320], dtype=torch.float64)},
                   {"t": torch.tensor([1e300], dtype=torch.float64)})
    tiny_entry = tiny["rungs"]["elementwise"]["tensors"]["t"]
    assert tiny_entry["max_rel_error"] is None and tiny_entry["rel_error_overflow"] == 1
    json.dumps(tiny, allow_nan=False)


def test_trajectory_displacement_overflow_is_strict_json():
    ref = {"r": torch.tensor([[0.0, 0.0], [1.7e308, 0.0]], dtype=torch.float64)}
    cand = {"r": torch.tensor([[0.0, 0.0], [-1.7e308, 0.0]], dtype=torch.float64)}
    report = parity.compare_trajectories(ref, cand, tolerance=_traj_tol(1e9), scope="synthetic")
    entry = report["trajectories"]["r"]
    assert entry["displacement_overflow"] == 1 and entry["final_displacement"] is None
    assert report["verdict"] == "exceeds"
    json.dumps(report, allow_nan=False)


CAST = "reference_cast_to_candidate"


def test_cross_dtype_cast_reference_to_candidate_known_answers():
    # A10-F3: zero-update export = release rounded to BF16 (round-to-nearest-even).
    ref = {"w": torch.tensor([1.0, 1.00390625, 2.0 + 2.0**-8, 3.0])}
    exact = {"w": ref["w"].to(torch.bfloat16)}
    # 1.00390625 = 1 + 2^-8 is a BF16 tie: rounds to even (1.0); 2 + 2^-8 rounds to 2.0.
    assert exact["w"].view(torch.int16).tolist() == [0x3F80, 0x3F80, 0x4000, 0x4040]
    report = parity.compare_tensors(ref, exact, tolerance=None, scope="synthetic",
                                    cross_dtype=CAST)
    assert report["verdict"] == "identical" and report["status"] == "passed"
    assert report["cross_dtype"] == CAST
    assert report["rungs"]["structure"]["dtype_cast"] == [
        {"name": "w", "reference_dtype": "F32", "candidate_dtype": "BF16"}]
    assert report["rungs"]["structure"]["dtype_mismatch"] == []
    # Negative control: one BF16 element one ULP off is measured in BF16.
    off = exact["w"].clone()
    off.view(torch.int16)[3] += 1  # 3.0 -> 3.015625
    bad = parity.compare_tensors(ref, {"w": off}, tolerance=_tol(max_ulp=0),
                                 scope="synthetic", cross_dtype=CAST)
    entry = bad["rungs"]["elementwise"]["tensors"]["w"]
    assert entry["dtype"] == "BF16" and entry["reference_dtype"] == "F32"
    assert entry["measured_in_dtype"] == "BF16"
    assert entry["max_ulp"] == 1 and entry["max_abs_error"] == 0.015625
    assert entry["differing_elements"] == 1 and bad["verdict"] == "exceeds"
    ok = parity.compare_tensors(ref, {"w": off}, tolerance=_tol(max_ulp=1),
                                scope="synthetic", cross_dtype=CAST)
    assert ok["verdict"] == "within_registered_tolerance"


def test_cross_dtype_is_opt_in_and_float_only():
    ref = {"w": torch.tensor([1.0])}
    plain = parity.compare_tensors(ref, {"w": torch.tensor([1.0], dtype=torch.bfloat16)},
                                   tolerance=None, scope="synthetic")
    assert plain["rungs"]["structure"]["dtype_mismatch"] == ["w"]
    ints = parity.compare_tensors({"i": torch.tensor([1], dtype=torch.int64)},
                                  {"i": torch.tensor([1], dtype=torch.int32)},
                                  tolerance=None, scope="synthetic", cross_dtype=CAST)
    assert ints["rungs"]["structure"]["dtype_mismatch"] == ["i"]
    assert ints["status"] == "blocked"
    with pytest.raises(ValueError, match="cross_dtype"):
        parity.compare_tensors(ref, ref, tolerance=None, scope="synthetic", cross_dtype="up")


def _manifest(directory, tensors):
    _write(directory, tensors)
    return s4_parity_check.manifest(directory, files=["model.safetensors"], index=None,
                                    scope="synthetic")


def test_cross_dtype_manifest_ladder_defers_without_load_and_verifies_with_load(tmp_path):
    ref_t = {"w": torch.tensor([1.0, 2.0 + 2.0**-8]), "same": torch.tensor([5.0])}
    cand_t = {"w": ref_t["w"].to(torch.bfloat16), "same": torch.tensor([5.0])}
    ref_m = _manifest(tmp_path / "ref", ref_t)
    cand_m = _manifest(tmp_path / "cand", cand_t)
    deferred = parity.ladder_from_manifests(ref_m, cand_m, tolerance=None, cross_dtype=CAST)
    assert deferred["status"] == "qualify_on_host" and deferred["verdict"] is None
    assert deferred["rungs"]["bitwise"]["identical"] == ["same"]
    assert deferred["rungs"]["bitwise"]["cast_undetermined"] == ["w"]
    load = s4_parity_check.loader(tmp_path / "ref", tmp_path / "cand", ref_m, cand_m)
    report = parity.ladder_from_manifests(ref_m, cand_m, tolerance=None, load=load,
                                          cross_dtype=CAST)
    assert report["verdict"] == "identical" and report["status"] == "passed"
    assert report["rungs"]["bitwise"]["identical"] == ["same", "w"]


def test_manifest_ladder_restricted_to_declared_names(tmp_path):
    ref_m = _manifest(tmp_path / "ref", {"w": torch.tensor([1.0]), "buf": torch.tensor([2.0])})
    cand_m = _manifest(tmp_path / "cand", {"w": torch.tensor([1.0])})
    assert parity.ladder_from_manifests(ref_m, cand_m, tolerance=None)["status"] == "blocked"
    report = parity.ladder_from_manifests(ref_m, cand_m, tolerance=None, names=["w"])
    assert report["verdict"] == "identical" and report["names"] == ["w"]
    missing = parity.ladder_from_manifests(ref_m, cand_m, tolerance=None, names=["buf"])
    assert missing["status"] == "blocked"
    assert missing["rungs"]["structure"]["missing_in_candidate"] == ["buf"]
    with pytest.raises(ValueError, match="names"):
        parity.ladder_from_manifests(ref_m, cand_m, tolerance=None, names=[])


@pytest.mark.parametrize("ref", ["TBD later", "pending", "x", "x.md", "experiments/a.md",
                                 "experiments/../x.md#a", "experiments/TBD.md#tol",
                                 "experiments/2026-11-09-s4.md#", "docs/x.md#tol",
                                 "experiments/2026-11-09-s4.md#to do"])
def test_registration_ref_must_be_a_structured_pointer(ref):
    # A10-F8: experiments/<file>.md#<anchor>, not a free-text placeholder.
    with pytest.raises(ValueError, match="registration_ref"):
        parity.validate_tolerance(_tol(registration_ref=ref, atol=1e-3))


def test_registration_ref_structured_pointer_accepted():
    for ref in ("experiments/2026-11-09-s4-parity.md#tolerances",
                "experiments/2026-11-09-s4-parity.md#e02-cast"):
        assert parity.validate_tolerance(_tol(registration_ref=ref, atol=0.0))["atol"] == 0.0


def _cli(tmp_path, *extra, ref="ref", cand="cand"):
    return ["--reference-root", str(tmp_path / ref), "--reference-file", "model.safetensors",
            "--candidate-root", str(tmp_path / cand), "--candidate-file", "model.safetensors",
            "--scope", "synthetic", *extra]


def test_parity_script_never_writes_inside_either_checkpoint(tmp_path, capsys):
    # A10-F4: the resolved output may not lie inside the reference or candidate root.
    tmp_path = tmp_path.resolve()
    _write(tmp_path / "ref", {"a": torch.tensor([1.0, 2.0])})
    _write(tmp_path / "cand", {"a": torch.tensor([1.0, 2.5])})
    for side in ("ref", "cand"):
        target = tmp_path / side / "parity.json"
        assert s4_parity_check.main(_cli(tmp_path, "--output", str(target))) == 2
        assert not target.exists()
        err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert err["issues"][0]["code"] == "output_inside_checkpoint"
    sneaky = tmp_path / "ref" / "sub" / ".." / "p.json"
    assert s4_parity_check.main(_cli(tmp_path, "--output", str(sneaky))) == 2


def test_parity_script_exit_codes(tmp_path, capsys):
    # A10-F5/F6: 0 identical/within, 1 blocked or deferred, 2 refused, 3 exceeds/unregistered.
    tmp_path = tmp_path.resolve()
    _write(tmp_path / "ref", {"a": torch.tensor([1.0, 2.0])})
    _write(tmp_path / "same", {"a": torch.tensor([1.0, 2.0])})
    _write(tmp_path / "other", {"zzz": torch.tensor([1.0, 2.0])})
    assert s4_parity_check.main(_cli(tmp_path, cand="same")) == 0
    capsys.readouterr()
    assert s4_parity_check.main(_cli(tmp_path, cand="other")) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked" and report["verdict"] == "exceeds"
    missing_dir = tmp_path / "nodir" / "p.json"
    assert s4_parity_check.main(_cli(tmp_path, "--output", str(missing_dir), cand="same")) == 2
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert err["issues"][0]["code"] == "output_write_failed"
    _write(tmp_path / "r64", {"a": torch.tensor([1.7e308], dtype=torch.float64)})
    _write(tmp_path / "c64", {"a": torch.tensor([-1.7e308], dtype=torch.float64)})
    assert s4_parity_check.main(_cli(tmp_path, ref="r64", cand="c64")) == 3
    out = json.loads(capsys.readouterr().out)
    assert out["rungs"]["elementwise"]["tensors"]["a"]["abs_error_overflow"] == 1


def test_parity_script_cross_dtype_and_names(tmp_path, capsys):
    tmp_path = tmp_path.resolve()
    _write(tmp_path / "ref", {"w": torch.tensor([1.0, 2.0 + 2.0**-8]),
                              "buf": torch.tensor([3.0])})
    _write(tmp_path / "cand", {"w": torch.tensor([1.0, 2.0 + 2.0**-8]).to(torch.bfloat16)})
    assert s4_parity_check.main(_cli(tmp_path, "--name", "w")) == 1  # dtype differs
    capsys.readouterr()
    assert s4_parity_check.main(_cli(tmp_path, "--name", "w", "--cross-dtype", CAST)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "identical" and report["names"] == ["w"]


def test_negative_bit_views_are_measured_by_logical_values():
    # Review probe: a lazily negated view is materialized before byte views (rule 11).
    report = _ladder({"t": torch.tensor([1.0, 2.0])},
                     {"t": torch.tensor([1.0, 2.0])._neg_view()})
    entry = report["rungs"]["elementwise"]["tensors"]["t"]
    assert entry["differing_elements"] == 2 and entry["max_abs_error"] == 4.0
