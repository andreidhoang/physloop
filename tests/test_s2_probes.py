"""Known answers and negative controls for gate.s2_probes (A6; readiness CPU-05).

Oracles are hand-derived: [0,1,0] has mean 1/3, sample variance 1/3 and
population variance 2/9; [1,1,1] has both variances 0; a 0/1 cell with one event
among three replicates has 1*2 = 2 discordant replicate pairs of C(3,2) = 3.
Probe payloads are hand-built safetensors bytes with literal little-endian values.
"""

import copy
import hashlib
import json
import random
import struct
import warnings

import pytest

from gate import s2_probes as sp
from gate.s2_identity import identity_hash

# --- describe_repeats --------------------------------------------------------


def obs(value, *, rep, scene="s1", road="r1", policy="teacher", metric="collision",
        kind="binary", run=None):
    return {
        "policy_id": policy,
        "training_run_id": run,
        "road_id": road,
        "scene_id": scene,
        "replicate_id": rep,
        "metric_id": metric,
        "metric_kind": kind,
        "value": value,
    }


def cells(report):
    return {(c["policy_id"], c["scene_id"], c["metric_id"]): c for c in report["cells"]}


def codes(report):
    return {item["code"] for item in report["issues"]}


REGISTRATION = {
    "schema_version": 1,
    "method_id": "one_way_icc_candidate",
    "estimand": "scene-level repeatability of the registered score",
    "hierarchy": "replicate within scene within road",
    "interval_method": "registered interval method",
    "floor": 0.8,
    "missing_data": "complete cases",
    "degenerate_case": "not_estimable",
    "source_ref": "experiments/example.md",
}


def test_binary_known_answers_0_1_0_and_1_1_1():
    rows = [obs(v, rep=f"r{i}") for i, v in enumerate([0, 1, 0])]
    rows += [obs(v, rep=f"r{i}", scene="s2") for i, v in enumerate([1, 1, 1])]
    report = sp.describe_repeats(rows, scope="synthetic")
    c1 = cells(report)[("teacher", "s1", "collision")]
    assert c1["n_observed"] == 3 and c1["n_missing"] == 0
    assert c1["mean"] == {"status": "passed", "value": 1 / 3, "exact": "1/3", "reason": None}
    assert c1["sample_variance"]["exact"] == "1/3"
    assert c1["sample_variance"]["value"] == 1 / 3
    assert c1["population_variance"]["exact"] == "2/9"
    assert c1["events"] == 1 and c1["non_events"] == 2
    assert c1["replicate_pairs"] == 3 and c1["discordant_replicate_pairs"] == 2
    c2 = cells(report)[("teacher", "s2", "collision")]
    assert c2["sample_variance"]["exact"] == "0" and c2["sample_variance"]["value"] == 0.0
    assert c2["population_variance"]["exact"] == "0"
    assert c2["discordant_replicate_pairs"] == 0
    (panel,) = report["panels"]
    assert panel["n_scenes"] == 2 and panel["n_roads"] == 1
    assert panel["scenes_discordant"] == 1
    assert panel["discordant_replicate_pairs"] == 2 and panel["replicate_pairs"] == 6
    assert panel["discordant_pair_share"]["exact"] == "1/3"
    # means 1/3 and 1 -> sample variance of means = (1/3 - 2/3)^2 * 2 / 1 = 2/9
    assert panel["between_scene_sample_variance_of_means"]["exact"] == "2/9"
    assert panel["within_scene_sample_variance"]["unweighted_mean"]["exact"] == "1/6"


def test_method_unregistered_blocks_but_descriptives_remain():
    rows = [obs(v, rep=f"r{i}") for i, v in enumerate([0, 1, 0])]
    report = sp.describe_repeats(rows, scope="synthetic")
    assert report["status"] == "blocked"
    assert report["descriptive_status"] == "passed"
    assert report["method"]["status"] == "blocked"
    assert report["method"]["code"] == "method_unregistered"
    assert report["method"]["result"] is None
    assert "method_unregistered" in codes(report)
    assert report["host_qualification"] == "not_run"
    assert report["cells"] and report["panels"]


def test_registered_but_unqualified_method_is_unsupported_never_pass():
    rows = [obs(v, rep=f"r{i}") for i, v in enumerate([0, 1, 0])]
    report = sp.describe_repeats(rows, method=REGISTRATION, scope="synthetic")
    assert report["status"] == "blocked"
    assert report["method"]["code"] == "unsupported_method"
    assert report["method"]["method_id"] == "one_way_icc_candidate"
    assert report["method"]["registration_sha256"]
    assert report["method"]["result"] is None
    assert "passed" not in json.dumps(report["method"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("floor"),
        lambda r: r.update(estimand="TBD"),
        lambda r: r.update(schema_version=2),
        lambda r: r.update(floor=float("nan")),
        lambda r: r.update(extra="x"),
        lambda r: r.update(interval_method="  "),
    ],
)
def test_method_registration_fails_closed(mutate):
    method = copy.deepcopy(REGISTRATION)
    mutate(method)
    report = sp.describe_repeats([obs(1, rep="r0")], method=method, scope="synthetic")
    assert report["method"]["code"] == "method_registration_invalid"
    assert report["status"] == "blocked"


def test_not_estimable_cases():
    rows = [obs(1, rep="r0", kind="continuous", metric="score")]
    rows += [obs(None, rep="r0", scene="s2", kind="continuous", metric="score")]
    report = sp.describe_repeats(rows, scope="synthetic")
    one = cells(report)[("teacher", "s1", "score")]
    assert one["sample_variance"] == {
        "status": "not_estimable", "value": None, "exact": None,
        "reason": "fewer_than_two_observed_replicates",
    }
    assert one["population_variance"]["exact"] == "0"
    empty = cells(report)[("teacher", "s2", "score")]
    assert empty["n_observed"] == 0 and empty["n_missing"] == 1
    assert empty["mean"]["status"] == "not_estimable"
    assert empty["mean"]["reason"] == "no_observed_values"
    (panel,) = report["panels"]
    assert panel["between_scene_sample_variance_of_means"]["reason"] == "fewer_than_two_scenes"
    assert panel["reliability_coefficients"]["status"] == "not_estimable"


def test_zero_between_scene_variation_is_undefined_not_perfect():
    rows = []
    for scene in ("s1", "s2", "s3"):
        rows += [obs(v, rep=f"r{i}", scene=scene, kind="continuous", metric="score")
                 for i, v in enumerate([0.25, 0.75])]
    (panel,) = sp.describe_repeats(rows, scope="synthetic")["panels"]
    assert panel["between_scene_sample_variance_of_means"]["exact"] == "0"
    assert panel["reliability_coefficients"]["status"] == "not_estimable"
    assert panel["reliability_coefficients"]["reason"] == "zero_between_scene_variance"
    assert panel["within_scene_sample_variance"]["unweighted_mean"]["exact"] == "1/8"


def test_positive_between_scene_variation_still_needs_method():
    rows = [obs(v, rep=f"r{i}", scene="s1", kind="continuous", metric="m")
            for i, v in enumerate([0, 1])]
    rows += [obs(v, rep=f"r{i}", scene="s2", kind="continuous", metric="m")
             for i, v in enumerate([2, 3])]
    (panel,) = sp.describe_repeats(rows, scope="synthetic")["panels"]
    assert panel["between_scene_sample_variance_of_means"]["exact"] == "2"
    assert panel["reliability_coefficients"] == {
        "status": "blocked", "reason": "method_unregistered", "value": None,
    }


def test_missing_is_not_zero():
    rows = [obs(v, rep=f"r{i}", kind="continuous", metric="m")
            for i, v in enumerate([1, None, 0])]
    cell = sp.describe_repeats(rows, scope="synthetic")["cells"][0]
    assert cell["n_observed"] == 2 and cell["n_missing"] == 1
    assert cell["missing_replicates"] == ["r1"]
    assert cell["mean"]["exact"] == "1/2"
    assert cell["sample_variance"]["exact"] == "1/2"


def test_paired_contingency_and_discordance_ratio():
    pattern = [(1, 0), (1, 0), (0, 1), (1, 1), (0, 0)]
    rows = []
    for i, (a, b) in enumerate(pattern):
        rows.append(obs(a, rep=f"r{i}", policy="teacher"))
        rows.append(obs(b, rep=f"r{i}", policy="reexport"))
    rows.append(obs(1, rep="r9", policy="teacher"))
    report = sp.describe_repeats(rows, scope="synthetic", paired_policies=("teacher", "reexport"))
    (pair,) = report["paired"]
    assert pair["metric_id"] == "collision"
    assert pair["table"] == {"a1_b1": 1, "a1_b0": 2, "a0_b1": 1, "a0_b0": 1}
    assert pair["pairs"] == 5 and pair["unpaired"] == {"a_only": 1, "b_only": 0, "missing": 0}
    assert pair["discordant_pairs"] == 3
    assert pair["discordance_ratio_a1_b0"]["exact"] == "2/3"
    assert "arbitrary" in pair["pairing_convention"]


def test_zero_total_discordance_ratio_not_estimable():
    rows = []
    for i, v in enumerate([1, 0, 1]):
        rows.append(obs(v, rep=f"r{i}", policy="teacher"))
        rows.append(obs(v, rep=f"r{i}", policy="reexport"))
    report = sp.describe_repeats(rows, scope="synthetic", paired_policies=("teacher", "reexport"))
    ratio = report["paired"][0]["discordance_ratio_a1_b0"]
    assert ratio["status"] == "not_estimable" and ratio["reason"] == "zero_total_discordance"


@pytest.mark.parametrize(
    "row, code",
    [
        ({**obs(1, rep="r0"), "seed": 3}, "observation_invalid"),
        ({k: v for k, v in obs(1, rep="r0").items() if k != "road_id"}, "observation_invalid"),
        (obs(2, rep="r0"), "observation_invalid"),
        (obs(True, rep="r0", kind="continuous"), "observation_invalid"),
        (obs(float("nan"), rep="r0", kind="continuous"), "observation_invalid"),
        (obs(1, rep="", kind="binary"), "observation_invalid"),
        (obs(1, rep="r0", kind="ordinal"), "observation_invalid"),
        ("not a dict", "observation_invalid"),
    ],
)
def test_malformed_observations_fail_closed(row, code):
    report = sp.describe_repeats([obs(0, rep="r1"), row], scope="synthetic")
    assert code in codes(report)
    assert report["descriptive_status"] == "blocked"
    assert report["counts"]["rejected_observations"] == 1


def test_duplicate_replicate_scene_road_and_kind_conflicts():
    report = sp.describe_repeats([obs(0, rep="r0"), obs(1, rep="r0")], scope="synthetic")
    assert "duplicate_replicate" in codes(report)
    assert report["cells"] == []
    report = sp.describe_repeats(
        [obs(0, rep="r0"), obs(1, rep="r1", road="r2")], scope="synthetic"
    )
    assert "scene_road_conflict" in codes(report)
    report = sp.describe_repeats(
        [obs(0, rep="r0"), obs(0.5, rep="r1", scene="s2", kind="continuous")], scope="synthetic"
    )
    assert "metric_kind_conflict" in codes(report)


def test_paired_policies_must_be_distinct_and_present():
    with pytest.raises(ValueError):
        sp.describe_repeats([obs(0, rep="r0")], scope="synthetic", paired_policies=("a", "a"))
    report = sp.describe_repeats(
        [obs(0, rep="r0")], scope="synthetic", paired_policies=("teacher", "absent")
    )
    assert "paired_policy_absent" in codes(report)


def test_describe_repeats_reorder_invariant_and_deterministic():
    rows = []
    for scene in ("s1", "s2"):
        for i, v in enumerate([0, 1, 1]):
            rows.append(obs(v, rep=f"r{i}", scene=scene, road=f"road-{scene}"))
    first = sp.describe_repeats(rows, scope="synthetic")
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    assert sp.describe_repeats(shuffled, scope="synthetic") == first
    assert first["artifact_kind"] == "s2_repeatability_diagnostics"
    assert first["input_hashes"]["observations"] == sp.describe_repeats(
        list(reversed(rows)), scope="synthetic")["input_hashes"]["observations"]
    with pytest.raises(ValueError):
        sp.describe_repeats(rows, scope="unknown")


# --- compare_probe_artifacts -------------------------------------------------


def payload(entries):
    """entries: list of (name, dtype, shape, raw bytes) in file order."""
    header, data, offset = {}, b"", 0
    for name, dtype, shape, raw in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        data += raw
        offset += len(raw)
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + data


def probe(entries, layout, *, kind="fixed_input_policy", input_sha="a" * 64, seed=None,
          scope="synthetic"):
    body = payload(entries)
    receipt = {
        "artifact_kind": "s2_probe_receipt",
        "schema_version": 1,
        "scope": scope,
        "probe_kind": kind,
        "input_sha256": input_sha,
        "payload_sha256": hashlib.sha256(body).hexdigest(),
        "payload_size_bytes": len(body),
        "seed_declared": seed,
        "tensors": layout,
    }
    return {"receipt": receipt, "payload": body}


def F32(*values):
    return struct.pack(f"<{len(values)}f", *values)



LAYOUT = {
    "tick2/traj": {"step": 2, "role": "trajectory", "units": "m"},
    "tick5/traj": {"step": 5, "role": "trajectory", "units": "m"},
}


def test_identical_probe():
    entries = [("tick2/traj", "F32", [2], F32(1.0, 2.0)), ("tick5/traj", "F32", [2], F32(3, 4))]
    report = sp.compare_probe_artifacts(probe(entries, LAYOUT), probe(entries, LAYOUT),
                                        "fixed_input_policy")
    assert report["status"] == "passed", report["issues"]
    assert report["outcome"] == "identical"
    assert report["first_divergence"] is None
    assert report["file_level"]["relation"] == "tensors_and_files_identical"
    assert report["tolerance"] == {"status": "blocked", "code": "tolerance_unregistered",
                                   "verdict": None}
    assert report["seed"]["consumption"] == "unverified"
    assert report["host_qualification"] == "not_run"
    actual = sp.compare_probe_artifacts(probe(entries, LAYOUT, scope="actual"),
                                        probe(entries, LAYOUT, scope="actual"),
                                        "fixed_input_policy")
    assert actual["host_qualification"] == "qualify_on_host"


def test_max_abs_difference_and_first_divergence_by_semantic_order():
    a_entries = [("tick5/traj", "F32", [2], F32(3.0, 4.0)),
                 ("tick2/traj", "F32", [2], F32(1.0, 2.0))]
    b_entries = [("tick5/traj", "F32", [2], F32(3.0, 4.5)),
                 ("tick2/traj", "F32", [2], F32(1.0, 2.25))]
    report = sp.compare_probe_artifacts(probe(a_entries, LAYOUT), probe(b_entries, LAYOUT),
                                        "fixed_input_policy")
    assert report["outcome"] == "different"
    assert report["first_divergence"] == {
        "name": "tick2/traj", "step": 2, "role": "trajectory", "relation": "bytes_differ",
        "element": [1],
    }
    rows = {row["name"]: row for row in report["tensors"]}
    assert rows["tick2/traj"]["max_abs_diff"] == {"status": "passed", "value": 0.25,
                                                  "reason": None}
    assert rows["tick5/traj"]["max_abs_diff"]["value"] == 0.5
    assert rows["tick2/traj"]["differing_elements"] == 1
    assert [row["name"] for row in report["tensors"]] == ["tick2/traj", "tick5/traj"]


def test_negative_zero_differs_bytewise_with_zero_max_diff():
    a = probe([("t", "F32", [1], F32(0.0))], {"t": {"step": 0, "role": "x", "units": "m"}})
    b = probe([("t", "F32", [1], F32(-0.0))], {"t": {"step": 0, "role": "x", "units": "m"}})
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy")
    assert report["outcome"] == "different"
    row = report["tensors"][0]
    assert row["relation"] == "bytes_differ" and row["max_abs_diff"]["value"] == 0.0


def test_same_bytes_with_mismatched_dtype_or_shape_is_not_parity():
    raw = bytes(range(8))
    layout = {"t": {"step": 0, "role": "x", "units": "m"}}
    base = probe([("t", "I32", [2], raw)], layout)
    for entries, relation in (
        ([("t", "F32", [2], raw)], "dtype_mismatch"),
        ([("t", "I32", [2, 1], raw)], "shape_mismatch"),
    ):
        report = sp.compare_probe_artifacts(base, probe(entries, layout), "fixed_input_policy")
        assert report["outcome"] == "different"
        assert report["tensors"][0]["relation"] == relation
        assert report["tensors"][0]["max_abs_diff"]["status"] == "not_estimable"


def test_units_mismatch_is_not_identical():
    entries = [("t", "F32", [1], F32(1.0))]
    a = probe(entries, {"t": {"step": 0, "role": "x", "units": "m"}})
    b = probe(entries, {"t": {"step": 0, "role": "x", "units": "cm"}})
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy")
    assert report["outcome"] == "different"
    assert report["tensors"][0]["relation"] == "units_mismatch"


def test_same_seed_different_inputs_is_not_a_determinism_test():
    entries = [("t", "F32", [1], F32(1.0))]
    layout = {"t": {"step": 0, "role": "x", "units": "m"}}
    a = probe(entries, layout, seed=11, input_sha="a" * 64)
    b = probe(entries, layout, seed=11, input_sha="b" * 64)
    tol = {"schema_version": 1, "tolerance_id": "probe-tol", "max_abs": 0.5,
           "source_ref": "experiments/x.md"}
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy", tolerance=tol)
    assert report["status"] == "blocked"
    assert report["outcome"] == "not_comparable"
    assert "inputs_differ" in codes(report)
    assert report["tensors"] == []
    assert report["tolerance"] == {"status": "not_run", "code": "comparison_not_performed",
                                   "verdict": None}


def test_renderer_probe_u8_and_nonfinite_handling():
    layout = {"f0/rgb": {"step": 0, "role": "camera_front", "units": "uint8_intensity"}}
    a = probe([("f0/rgb", "U8", [2, 2], bytes([0, 10, 200, 255]))], layout,
              kind="fixed_pose_renderer")
    b = probe([("f0/rgb", "U8", [2, 2], bytes([0, 13, 200, 250]))], layout,
              kind="fixed_pose_renderer")
    report = sp.compare_probe_artifacts(a, b, "fixed_pose_renderer")
    row = report["tensors"][0]
    assert row["max_abs_diff"] == {"status": "passed", "value": 5, "reason": None}
    assert row["differing_elements"] == 2 and row["first_differing_element"] == [0, 1]

    nan = struct.pack("<f", float("nan"))
    lay = {"t": {"step": 0, "role": "x", "units": "m"}}
    same_nan = sp.compare_probe_artifacts(probe([("t", "F32", [1], nan)], lay),
                                          probe([("t", "F32", [1], nan)], lay),
                                          "fixed_input_policy")
    assert same_nan["outcome"] == "identical"
    mixed = sp.compare_probe_artifacts(probe([("t", "F32", [2], nan + F32(1.0))], lay),
                                       probe([("t", "F32", [2], F32(0.0, 1.5))], lay),
                                       "fixed_input_policy")
    row = mixed["tensors"][0]
    assert row["nonfinite_mismatch_count"] == 1
    assert row["max_abs_diff"]["value"] == 0.5


def test_bf16_decoded_and_f8_byte_only():
    lay = {"t": {"step": 0, "role": "x", "units": "m"}}
    # BF16 0x3f80 = 1.0, 0x3fc0 = 1.5
    report = sp.compare_probe_artifacts(probe([("t", "BF16", [1], b"\x80\x3f")], lay),
                                        probe([("t", "BF16", [1], b"\xc0\x3f")], lay),
                                        "fixed_input_policy")
    assert report["tensors"][0]["max_abs_diff"]["value"] == 0.5
    report = sp.compare_probe_artifacts(probe([("t", "F8_E4M3", [1], b"\x38")], lay),
                                        probe([("t", "F8_E4M3", [1], b"\x39")], lay),
                                        "fixed_input_policy")
    row = report["tensors"][0]
    assert row["relation"] == "bytes_differ"
    assert row["max_abs_diff"] == {"status": "not_estimable", "value": None,
                                   "reason": "dtype_not_decoded"}


def test_tensor_set_mismatch_and_order_key():
    lay_a = {"t": {"step": 0, "role": "x", "units": "m"}, "u": {"step": 1, "role": "x",
                                                              "units": "m"}}
    lay_b = {"t": {"step": 0, "role": "x", "units": "m"}}
    a = probe([("t", "F32", [1], F32(1.0)), ("u", "F32", [1], F32(2.0))], lay_a)
    b = probe([("t", "F32", [1], F32(1.0))], lay_b)
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy")
    assert report["outcome"] == "different"
    assert {row["name"]: row["relation"] for row in report["tensors"]} == {
        "t": "identical", "u": "only_in_a"}
    assert report["first_divergence"]["name"] == "u"

    lay_c = {"t": {"step": 3, "role": "x", "units": "m"}}
    report = sp.compare_probe_artifacts(probe([("t", "F32", [1], F32(1.0))], lay_b),
                                        probe([("t", "F32", [1], F32(1.0))], lay_c),
                                        "fixed_input_policy")
    assert report["tensors"][0]["relation"] == "order_key_mismatch"


def test_registered_tolerance_verdicts():
    lay = {"t": {"step": 0, "role": "x", "units": "m"}}
    a = probe([("t", "F32", [2], F32(1.0, 2.0))], lay)
    b = probe([("t", "F32", [2], F32(1.0, 2.25))], lay)
    tol = {"schema_version": 1, "tolerance_id": "probe-tol", "max_abs": 0.5,
           "source_ref": "experiments/x.md"}
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy", tolerance=tol)
    assert report["tolerance"]["verdict"] == "within_registered_tolerance"
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy",
                                        tolerance={**tol, "max_abs": 0.125})
    assert report["tolerance"]["verdict"] == "exceeds"
    report = sp.compare_probe_artifacts(a, a, "fixed_input_policy", tolerance=tol)
    assert report["tolerance"]["verdict"] == "identical"
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy",
                                        tolerance={**tol, "max_abs": -1})
    assert report["tolerance"]["code"] == "tolerance_registration_invalid"


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda p: p["receipt"].update(payload_sha256="0" * 64), "payload_hash_mismatch"),
        (lambda p: p.update(payload=p["payload"] + b" "), "payload_hash_mismatch"),
        (lambda p: p["receipt"].update(schema_version=2), "receipt_invalid"),
        (lambda p: p["receipt"].update(probe_kind="fixed_pose_renderer"), "probe_kind_mismatch"),
        (lambda p: p["receipt"].update(scope="actual"), "scope_mismatch"),
        (lambda p: p["receipt"]["tensors"].pop("tick5/traj"), "receipt_payload_mismatch"),
        (lambda p: p["receipt"]["tensors"]["tick5/traj"].update(step=-1), "receipt_invalid"),
        (lambda p: p["receipt"].update(extra=1), "receipt_invalid"),
    ],
)
def test_probe_receipt_negative_controls(mutate, code):
    entries = [("tick2/traj", "F32", [2], F32(1.0, 2.0)), ("tick5/traj", "F32", [2], F32(3, 4))]
    a = probe(entries, copy.deepcopy(LAYOUT))
    b = probe(entries, copy.deepcopy(LAYOUT))
    mutate(b)
    report = sp.compare_probe_artifacts(a, b, "fixed_input_policy")
    assert report["status"] == "blocked"
    assert code in codes(report)
    assert report["outcome"] == "not_comparable"


def test_malformed_payload_and_unknown_kind():
    lay = {"t": {"step": 0, "role": "x", "units": "m"}}
    good = probe([("t", "F32", [1], F32(1.0))], lay)
    bad_body = b"\x00" * 4
    bad = copy.deepcopy(good)
    bad["payload"] = bad_body
    bad["receipt"].update(payload_sha256=hashlib.sha256(bad_body).hexdigest(),
                          payload_size_bytes=4)
    report = sp.compare_probe_artifacts(good, bad, "fixed_input_policy")
    assert "payload_invalid" in codes(report) and report["status"] == "blocked"
    with pytest.raises(ValueError):
        sp.compare_probe_artifacts(good, good, "renderer")


# --- review regressions (A6-F3, F6, F8, F10) ---------------------------------------

TOL = {"schema_version": 1, "tolerance_id": "probe-tol", "max_abs": 1.0,
       "source_ref": "experiments/x.md"}


def test_f64_difference_overflow_is_explicit_and_hashable():
    """A6-F3: 1e308 - (-1e308) overflows float64; no inf, no warning, verdict exceeds."""
    lay = {"t": {"step": 0, "role": "x", "units": "m"}}
    a = probe([("t", "F64", [1], struct.pack("<d", 1e308))], lay)
    b = probe([("t", "F64", [1], struct.pack("<d", -1e308))], lay)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = sp.compare_probe_artifacts(a, b, "fixed_input_policy", tolerance=TOL)
    row = report["tensors"][0]
    assert row["relation"] == "bytes_differ"
    assert row["max_abs_diff"] == {"status": "not_estimable", "value": None,
                                   "reason": "difference_overflows_float64"}
    assert report["tolerance"]["verdict"] == "exceeds"
    identity_hash(report)  # canonical JSON: every number finite


def test_observation_input_hash_covers_rejected_rows():
    """A6-F6: invalid supplied rows change the input hash; accepted rows hashed apart."""
    base = [obs(v, rep=f"r{i}") for i, v in enumerate([0, 1, 0])]
    clean = sp.describe_repeats(base, scope="synthetic")
    noisy = sp.describe_repeats(base + ["garbage", {"x": 1}], scope="synthetic")
    assert noisy["counts"]["observations"] == 5
    assert noisy["input_hashes"]["observations"] != clean["input_hashes"]["observations"]
    assert (noisy["input_hashes"]["accepted_observations"]
            == clean["input_hashes"]["accepted_observations"])
    nonjson = sp.describe_repeats(base + [{"x": float("nan")}], scope="synthetic")
    assert nonjson["input_hashes"]["observations"] not in (
        None, clean["input_hashes"]["observations"])
    assert nonjson["counts"]["non_json_observations"] == 1


def test_invalid_tolerance_blocks_status_but_keeps_descriptives():
    """A6-F8: an error-severity issue never coexists with status passed."""
    lay = {"t": {"step": 0, "role": "x", "units": "m"}}
    a = probe([("t", "F32", [1], F32(1.0))], lay)
    report = sp.compare_probe_artifacts(a, a, "fixed_input_policy",
                                        tolerance={**TOL, "tolerance_id": "TBD"})
    assert report["tolerance"]["code"] == "tolerance_registration_invalid"
    assert report["tolerance"]["verdict"] is None
    assert report["status"] == "blocked"
    assert report["outcome"] == "identical"
    assert report["tensors"][0]["relation"] == "identical"


def test_paired_contingency_is_unverified_without_owner_pairing():
    """A6-F10: the agent coupling of unseeded replicates is never a passed quantity."""
    rows = []
    for i, (a, b) in enumerate([(1, 0), (0, 1), (1, 0)]):
        rows.append(obs(a, rep=f"r{i}", policy="teacher"))
        rows.append(obs(b, rep=f"r{i}", policy="reexport"))
    report = sp.describe_repeats(rows, scope="synthetic", paired_policies=("teacher", "reexport"))
    (pair,) = report["paired"]
    assert pair["status"] == "unverified"
    assert pair["code"] == "pairing_convention_unregistered"
    ratio = pair["discordance_ratio_a1_b0"]
    assert ratio["status"] == "unverified" and ratio["exact"] == "2/3"
    assert ratio["reason"] == "pairing_convention_unregistered"
    issue = [i for i in report["issues"] if i["code"] == "pairing_convention_unregistered"]
    assert len(issue) == 1 and issue[0]["severity"] == "warning"
    assert report["descriptive_status"] == "passed"
