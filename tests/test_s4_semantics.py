"""Known answers for E10-E14 metric semantics (build-spec A9).

Oracles: hand-derived literals for the offline modifier chain (re-derived from the
pinned AlpaSim source text, whose relevant lines are asserted below); the pinned
``MetricReturn.aggregate`` and AlpaGym ``compute_distance_to_gt_reward`` executed from
hash-checked vendored source (selected AST nodes, restricted namespace).
"""

import dataclasses
import importlib.util
import math
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teacher_rl.checks import semantics

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "s4_upstream_loader", ROOT / "tests/fixtures/s4_upstream_checks/upstream_loader.py"
)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)

TS = [0, 100_000, 200_000, 300_000, 400_000]


@pytest.fixture(scope="module")
def pinned_aggregate():
    ns = upstream.load(
        "alpasim/data.py.txt",
        ["AggregationType", "MetricReturn"],
        {"np": np, "dataclasses": dataclasses, "StrEnum": StrEnum},
    )

    def aggregate(metric):
        m = ns["MetricReturn"](
            name=metric["name"],
            timestamps_us=metric["timestamps_us"],
            values=[semantics.TOKENS.get(v, v) for v in metric["values"]],
            valid=metric["valid"],
            time_aggregation=ns["AggregationType"](metric["time_aggregation"]),
        )
        return m.aggregate()

    return aggregate


def metric(name, values, kind="max", valid=None):
    return {
        "name": name,
        "timestamps_us": list(TS),
        "values": list(values),
        "valid": [True] * len(values) if valid is None else list(valid),
        "time_aggregation": kind,
    }


def rollout(overrides=(), drop=(), aggregated=None, aggregate_fn=None):
    base = {
        "eval_relevant": metric("eval_relevant", [0.0, 0.0, 1.0, 1.0, 1.0]),
        "collision_any": metric("collision_any", [0.0] * 5),
        "collision_front": metric("collision_front", [0.0] * 5),
        "collision_rear": metric("collision_rear", [0.0] * 5),
        "collision_lateral": metric("collision_lateral", [0.0] * 5),
        "offroad": metric("offroad", [0.0] * 5),
        "img_is_black": metric("img_is_black", [0.0] * 5),
        "progress": metric("progress", [0.0, 0.1, 0.2, 0.3, 0.4], "last"),
    }
    for m in overrides:
        base[m["name"]] = m
    for name in drop:
        base.pop(name)
    metrics = [base[k] for k in sorted(base)]
    if aggregated is None:
        fn = aggregate_fn or semantics.online_aggregate
        aggregated = {}
        for m in metrics:
            if m["name"] != "eval_relevant":
                value = fn(m)
                aggregated[m["name"]] = (
                    "nan" if isinstance(value, float) and math.isnan(value) else value
                )
    return {"session_uuid": "s-1", "timestep_metrics": metrics, "aggregated_metrics": aggregated}


def by_metric(report):
    return {row["metric"]: row for row in report["metrics"]}


# --- pinned source facts the re-derivation relies on ---------------------------------


def test_pinned_source_facts():
    evaluator = upstream.source_bytes("alpasim/scenario_evaluator.py.txt").decode()
    aggregated = evaluator.index(
        "aggregated_metrics = {m.name: m.aggregate() for m in timestep_metrics}"
    )
    appended = evaluator.index('name="eval_relevant"')
    assert aggregated < appended  # the online aggregate never sees the warm-up mask
    processing = upstream.source_bytes("alpasim/processing.py.txt").decode()
    order = [
        'RemoveTimestepsBeforeEvent(pl.col("eval_relevant") > 0.0)',
        'name="offroad_or_collision"',
        'name="collision_at_fault"',
        'name="offroad_or_collision_at_fault"',
        'name="duration_frac_20s"',
        'RemoveTrajectoryWithEvent(pl.col("img_is_black") > 0.0)',
        'RemoveTimestepsAfterEvent(pl.col("offroad_or_collision") > 0.0)',
    ]
    positions = [processing.index(text) for text in order]
    assert positions == sorted(positions)
    assert 'values="values",' in processing  # the offline pivot never reads `valid`
    assert 'df_wide = df_wide.with_columns(pl.lit(0.0).alias("offroad"))' in processing
    modifiers = upstream.source_bytes("alpasim/modifiers.py.txt").decode()
    assert 'filter_event = pl.col("timestamps_us") <= pl.col("first_event_timestamp")' in modifiers
    assert 'filter_event = pl.col("timestamps_us") >= pl.col("first_event_timestamp")' in modifiers


@pytest.mark.parametrize("kind", semantics.AGGREGATIONS)
def test_online_aggregate_matches_pinned_metric_return(kind, pinned_aggregate):
    cases = [
        metric("m", [0.5, 3.0, -1.0, 2.0, 0.25], kind),
        metric("m", [0.5, 3.0, -1.0, 2.0, 0.25], kind, valid=[True, False, True, False, True]),
        metric("m", [0.5, 3.0, -1.0, 2.0, 0.25], kind, valid=[False] * 5),
        metric("m", [1.0, "nan", 2.0, 0.0, 0.0], kind),
    ]
    for case in cases:
        ours = semantics.online_aggregate(case)
        theirs = pinned_aggregate(case)
        if isinstance(theirs, float) and math.isnan(theirs):
            assert ours == "nan"
        else:
            assert ours == theirs


# --- E10 ---------------------------------------------------------------------------


def test_e10_contact_before_warm_up_is_online_only(pinned_aggregate):
    r = rollout(
        [metric("collision_any", [0, 1.0, 0, 0, 0]), metric("collision_front", [0, 1.0, 0, 0, 0])],
        aggregate_fn=pinned_aggregate,
    )
    report = semantics.reconcile_metric_domains(r, scope="synthetic")
    assert report["status"] == "passed"
    row = by_metric(report)["collision_any"]
    assert row["online_recomputed"] == row["captured_online"] == 1.0
    assert row["online_matches_capture"] is True
    assert row["offline"] == 0.0
    assert row["first_differing_stage"] == "pre_engagement_removed"
    assert "online_offline_domain_differs" in {i["code"] for i in report["issues"]}


def test_e10_contact_after_first_incident_is_truncated_offline():
    r = rollout(
        [
            metric("offroad", [0, 0, 0, 1.0, 0]),
            metric("collision_any", [0, 0, 0, 0, 1.0]),
            metric("collision_lateral", [0, 0, 0, 0, 1.0]),
        ]
    )
    rows = by_metric(semantics.reconcile_metric_domains(r, scope="synthetic"))
    assert (rows["collision_any"]["online_recomputed"], rows["collision_any"]["offline"]) == (
        1.0,
        0.0,
    )
    assert rows["collision_any"]["first_differing_stage"] == "post_incident_removed"
    assert (rows["offroad"]["online_recomputed"], rows["offroad"]["offline"]) == (1.0, 1.0)
    assert (rows["progress"]["online_recomputed"], rows["progress"]["offline"]) == (0.4, 0.3)
    assert rows["progress"]["first_differing_stage"] == "post_incident_removed"


def test_e10_contact_at_final_timestamp_counts_in_both_domains():
    r = rollout(
        [metric("collision_any", [0, 0, 0, 0, 1.0]), metric("collision_rear", [0, 0, 0, 0, 1.0])]
    )
    rows = by_metric(semantics.reconcile_metric_domains(r, scope="synthetic"))
    assert rows["collision_any"]["online_recomputed"] == rows["collision_any"]["offline"] == 1.0
    assert rows["collision_any"]["differs"] is False
    assert rows["progress"]["offline"] == 0.4
    view = semantics.offline_view(r)
    assert view["values"]["offroad_or_collision"] == 1.0
    assert view["values"]["collision_at_fault"] == 0.0  # rear contact is not at fault
    assert view["values"]["duration_frac_20s"] == 0.01  # (400000 - 200000) / 20e6
    assert view["rows_kept"] == 3


def test_e10_invalid_values_and_black_image_and_missing_offroad():
    invalid = rollout(
        [
            metric(
                "dist_to_gt_trajectory",
                [0.5, 0.6, 0.7, 9.0, 0.8],
                valid=[True, True, True, False, True],
            )
        ]
    )
    row = by_metric(semantics.reconcile_metric_domains(invalid, scope="synthetic"))[
        "dist_to_gt_trajectory"
    ]
    assert (row["online_recomputed"], row["offline"]) == (0.8, 9.0)
    assert row["first_differing_stage"] == "invalid_values_included"
    black = semantics.reconcile_metric_domains(
        rollout([metric("img_is_black", [0, 0, 0, 1.0, 0])]), scope="synthetic"
    )
    assert black["offline_state"] == "removed"
    assert by_metric(black)["progress"]["first_differing_stage"] == "black_image_trajectory_removed"
    no_offroad = semantics.reconcile_metric_domains(rollout(drop=["offroad"]), scope="synthetic")
    assert no_offroad["status"] == "unverified"
    assert no_offroad["offline_reason"] == "offroad_missing_upstream_zero_fills"
    assert all(row["offline"] is None for row in no_offroad["metrics"])  # never zero-filled


def test_e10_negative_controls():
    tampered = rollout()
    tampered["aggregated_metrics"]["collision_any"] = 1.0
    report = semantics.reconcile_metric_domains(tampered, scope="synthetic")
    assert report["status"] == "blocked"
    assert "online_aggregate_mismatch" in {i["code"] for i in report["issues"]}
    none = semantics.reconcile_metric_domains(
        rollout(aggregated=None) | {"aggregated_metrics": None}, scope="synthetic"
    )
    assert "aggregated_metrics_none" in {i["code"] for i in none["issues"]}
    bad = rollout()
    bad["timestep_metrics"][0]["values"][0] = None
    with pytest.raises(ValueError):
        semantics.reconcile_metric_domains(bad, scope="synthetic")
    with pytest.raises(ValueError):
        semantics.reconcile_metric_domains(rollout(), scope="real")
    duplicate = rollout(
        [metric("progress", [0.0, 0.1, 0.2, 0.3, 0.4], "last") | {"timestamps_us": [0, 0, 1, 2, 3]}]
    )
    assert semantics.offline_view(duplicate)["state"] == "unknown"


# --- E11 ---------------------------------------------------------------------------


def test_e11_any_contact_is_one_while_additive_is_two():
    r = rollout(
        [
            metric("collision_any", [0, 0, 1.0, 1.0, 0]),
            metric("collision_front", [0, 0, 1.0, 0, 0]),
            metric("collision_lateral", [0, 0, 0, 1.0, 0]),
        ]
    )
    report = semantics.contact_predicates(r, scope="synthetic")
    assert report["status"] == "passed"
    assert report["any_contact"] == 1.0
    assert report["additive_front_lateral"] == 2.0
    assert report["additive_differs_from_any"] is True


def test_e11_negative_controls():
    inconsistent = rollout([metric("collision_front", [0, 0, 1.0, 0, 0])])
    report = semantics.contact_predicates(inconsistent, scope="synthetic")
    assert report["status"] == "blocked"
    assert report["inconsistent_timestamps_us"] == [200_000]
    missing = semantics.contact_predicates(rollout(drop=["collision_rear"]), scope="synthetic")
    assert "collision_metric_missing" in {i["code"] for i in missing["issues"]}


# --- E12 ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pinned_distance():
    ns = upstream.load(
        "alpagym/distance_to_gt.py.txt",
        [
            "compute_distance_to_gt_reward",
            "_executed_arrays",
            "_gt_arrays",
            "_interp_pose_at",
            "_yaw_from_pose",
        ],
        {
            "np": np,
            "math": math,
            "RewardResult": lambda total, report_metrics: SimpleNamespace(
                total=total, report_metrics=report_metrics
            ),
        },
    )

    def run(executed, ground_truth):
        def pose(p, yaw=0.0):
            return SimpleNamespace(
                timestamp_us=p["timestamp_us"],
                pose=SimpleNamespace(
                    vec=SimpleNamespace(x=p["x"], y=p["y"]),
                    quat=SimpleNamespace(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2)),
                ),
            )

        episode = SimpleNamespace(
            executed_ego_trajectory=SimpleNamespace(
                poses=tuple(pose(p, p["yaw"]) for p in executed)
            )
        )
        gt = None
        if ground_truth is not None:
            gt = SimpleNamespace(
                timestamp_us=ground_truth["anchor_timestamp_us"],
                ego_trajectory=SimpleNamespace(poses=tuple(pose(p) for p in ground_truth["poses"])),
            )
        return ns["compute_distance_to_gt_reward"](episode, gt, distance_scale=-0.01)

    return run


def _exec(points):
    return [{"timestamp_us": t, "x": x, "y": y, "yaw": yaw} for t, x, y, yaw in points]


def _gt(anchor, points):
    return {
        "anchor_timestamp_us": anchor,
        "poses": [{"timestamp_us": t, "x": x, "y": y} for t, x, y in points],
    }


def test_e12_identical_path_at_different_timestamps_has_zero_deviation(pinned_distance):
    executed = _exec([(t, 10.0 * t / 1e6, 0.0, 0.0) for t in range(0, 1_000_001, 100_000)])
    gt = _gt(
        250_000, [(t, 10.0 * (t - 250_000) / 1e6, 0.0) for t in range(50_000, 950_001, 150_000)]
    )
    report = semantics.check_reference_alignment(executed, gt, scope="synthetic")
    assert report["upstream_outcome"] == "computed"
    assert report["rmse_xy"] < 1e-12
    assert report["num_paired"] == 9  # executed 100k..900k inside [50k, 950k]
    upstream_result = pinned_distance(executed, gt)
    assert upstream_result.report_metrics["gt_num_paired"] == 9.0
    assert abs(upstream_result.report_metrics["gt_rmse_xy"] - report["rmse_xy"]) < 1e-12


def test_e12_oracle_matches_pinned_reward_on_a_turning_path(pinned_distance):
    executed = _exec(
        [
            (t, 5.0 * math.sin(t / 4e5), 2.0 * (1 - math.cos(t / 4e5)), t / 8e5)
            for t in range(0, 2_000_001, 100_000)
        ]
    )
    gt = _gt(
        300_000,
        [
            (
                t,
                0.9 * (t - 300_000) / 1e5,
                0.05 * ((t - 300_000) / 1e5) ** 2,
            )
            for t in range(100_000, 1_800_001, 200_000)
        ],
    )
    ours = semantics.check_reference_alignment(executed, gt, scope="synthetic")
    theirs = pinned_distance(executed, gt).report_metrics
    assert ours["num_paired"] == theirs["gt_num_paired"]
    assert abs(ours["rmse_xy"] - theirs["gt_rmse_xy"]) < 1e-9
    assert ours["rmse_xy"] > 0.1


@pytest.mark.parametrize(
    ("code", "executed", "gt"),
    [
        ("ground_truth_missing", _exec([(0, 0, 0, 0), (1, 1, 0, 0)]), None),
        ("too_few_executed_poses", _exec([(0, 0, 0, 0)]), _gt(0, [(0, 0, 0), (1, 1, 0)])),
        ("too_few_gt_poses", _exec([(0, 0, 0, 0), (1, 1, 0, 0)]), _gt(0, [(0, 0, 0)])),
        (
            "executed_timestamps_not_increasing",
            _exec([(0, 0, 0, 0), (0, 1, 0, 0)]),
            _gt(0, [(0, 0, 0), (1, 1, 0)]),
        ),
        (
            "gt_anchor_outside_executed_range",
            _exec([(0, 0, 0, 0), (10, 1, 0, 0)]),
            _gt(11, [(0, 0, 0), (1, 1, 0)]),
        ),
        (
            "no_executed_points_in_gt_span",
            _exec([(0, 0, 0, 0), (10, 1, 0, 0)]),
            _gt(5, [(20, 0, 0), (30, 1, 0)]),
        ),
    ],
)
def test_e12_raise_paths_become_silent_retries(code, executed, gt, pinned_distance):
    report = semantics.check_reference_alignment(executed, gt, scope="synthetic")
    assert report["upstream_outcome"] == "raises"
    assert report["raise_code"] == code
    assert report["consequence"] == "silent_retry_new_session"
    with pytest.raises(ValueError):
        pinned_distance(executed, gt)


def test_e12_captured_comparison_needs_owner_tolerance():
    executed = _exec([(t, t / 1e5, 0.0, 0.0) for t in range(0, 500_001, 100_000)])
    gt = _gt(0, [(t, t / 1e5, 0.0) for t in range(0, 500_001, 250_000)])
    captured = {"gt_rmse_xy": 0.0, "gt_num_paired": 6}
    blocked = semantics.check_reference_alignment(
        executed, gt, captured=captured, scope="synthetic"
    )
    assert blocked["status"] == "blocked"
    assert "tolerance_unregistered" in {i["code"] for i in blocked["issues"]}
    ok = semantics.check_reference_alignment(
        executed, gt, captured=captured, tolerance={"absolute": 1e-9}, scope="synthetic"
    )
    assert ok["status"] == "passed" and ok["captured_matches"] is True
    wrong = semantics.check_reference_alignment(
        executed,
        gt,
        captured={"gt_rmse_xy": 0.5, "gt_num_paired": 6},
        tolerance={"absolute": 1e-9},
        scope="synthetic",
    )
    assert "captured_reference_mismatch" in {i["code"] for i in wrong["issues"]}


# --- E13 ---------------------------------------------------------------------------

THRESHOLDS = {
    "stationary_max_path_m": 0.5,
    "oscillation_min_heading_sign_changes": 3,
    "emergency_min_deceleration_mps2": 6.0,
    "detour_min_length_ratio": 1.5,
}


def test_e13_known_answers():
    still = semantics.diagnose_trajectory(
        _exec([(0, 1.0, 1.0, 0.0), (100_000, 1.0, 1.0, 0.0), (200_000, 1.0, 1.0, 0.0)]),
        thresholds=THRESHOLDS,
        scope="synthetic",
    )
    assert still["facts"]["path_length_m"] == 0.0 and still["flags"]["stationary"] is True
    wobble = semantics.diagnose_trajectory(
        _exec([(i * 100_000, float(i), 0.0, 0.1 * (i % 2)) for i in range(5)]),
        thresholds=THRESHOLDS,
        scope="synthetic",
    )
    assert wobble["facts"]["heading_sign_changes"] == 3 and wobble["flags"]["oscillation"] is True
    stop = semantics.diagnose_trajectory(
        _exec([(0, 0.0, 0, 0), (100_000, 1.0, 0, 0), (200_000, 2.0, 0, 0), (300_000, 2.0, 0, 0)]),
        thresholds=THRESHOLDS,
        scope="synthetic",
    )
    assert stop["facts"]["max_deceleration_mps2"] == pytest.approx(100.0)  # 10 m/s -> 0 in 0.1 s
    assert (
        stop["flags"]["hard_stop"] is True and stop["flags"]["hard_stop_legitimacy"] == "unverified"
    )
    detour = semantics.diagnose_trajectory(
        _exec([(0, 0.0, 0, 0), (1_000_000, 0.0, 10.0, 0), (2_000_000, 10.0, 10.0, 0)]),
        reference_xy=[[0.0, 0.0], [10.0, 0.0]],
        thresholds=THRESHOLDS,
        scope="synthetic",
    )
    assert detour["facts"]["length_ratio"] == 2.0 and detour["flags"]["long_detour"] is True


def test_e13_blocks_without_owner_thresholds_but_reports_facts():
    report = semantics.diagnose_trajectory(
        _exec([(0, 0.0, 0, 0), (100_000, 1.0, 0, 0)]), thresholds=None, scope="synthetic"
    )
    assert report["status"] == "blocked" and report["flags"] is None
    assert report["facts"]["max_speed_mps"] == 10.0
    with pytest.raises(ValueError):
        semantics.diagnose_trajectory(
            _exec([(0, 0.0, 0, 0), (100_000, 1.0, 0, 0)]),
            thresholds={"stationary_max_path_m": 1},
            scope="synthetic",
        )
    with pytest.raises(ValueError):
        semantics.diagnose_trajectory(
            _exec([(0, 0.0, 0, 0), (0, 1.0, 0, 0)]), thresholds=THRESHOLDS, scope="synthetic"
        )


# --- E14 ---------------------------------------------------------------------------


def attempt(aid, **fields):
    base = {
        "attempt_id": aid,
        "payload_id": "p0",
        "session_uuid": "u-" + aid,
        "outcome": "succeeded",
        "failure_cause": None,
        "aggregated_metrics": {"progress": 0.4, "collision_any": 0.0, "offroad": 0.0},
        "img_is_black_max": 0.0,
        "max_dist_to_gt_trajectory_m": 1.0,
        "old_logprobs_nonfinite": False,
        "nonfinite_resamples": 0,
    }
    base.update(fields)
    return base


REQUIRED = ["progress", "collision_any", "offroad"]


def test_e14_indicator_counts_and_retry_causes():
    attempts = [
        attempt("a1"),
        attempt(
            "a2",
            outcome="failed",
            failure_cause="KeyError: 'offroad'",
            aggregated_metrics={"progress": 0.1, "collision_any": 0.0},
        ),
        attempt("a3", aggregated_metrics={"progress": "nan", "collision_any": 0.0, "offroad": 0.0}),
        attempt(
            "a4",
            max_dist_to_gt_trajectory_m=4.0,
            img_is_black_max=1.0,
            old_logprobs_nonfinite=True,
            nonfinite_resamples=2,
        ),
        attempt(
            "a5",
            payload_id="p1",
            outcome="failed",
            failure_cause="ValueError: GT anchor timestamp 9 is outside",
        ),
    ]
    report = semantics.summarize_validity(attempts, required_metrics=REQUIRED, scope="synthetic")
    assert report["status"] == "passed"
    counts = report["indicator_counts"]
    assert counts["missing_metric"] == {"true": 1, "false": 4, "unknown": 0}
    assert counts["nonfinite_metric"] == {"true": 1, "false": 4, "unknown": 0}
    assert counts["out_of_envelope"] == {"true": 1, "false": 4, "unknown": 0}  # 4.0 m is out
    assert counts["black_frame"]["true"] == 1 and counts["nonfinite_log_density"]["true"] == 1
    assert counts["nonfinite_resampled"]["true"] == 1
    assert report["retry_causes"] == {"missing_metric": 1, "reference_alignment": 1}
    assert report["attempts_per_payload"] == {"p0": 4, "p1": 1}


def test_e14_unknown_is_never_zero_and_owner_input_required():
    unknown = attempt(
        "u1",
        img_is_black_max=None,
        max_dist_to_gt_trajectory_m=None,
        old_logprobs_nonfinite=None,
        nonfinite_resamples=None,
    )
    report = semantics.summarize_validity([unknown], required_metrics=REQUIRED, scope="synthetic")
    assert report["status"] == "unverified"
    assert report["indicator_counts"]["black_frame"] == {"true": 0, "false": 0, "unknown": 1}
    blocked = semantics.summarize_validity(
        [attempt("a1")], required_metrics=None, scope="synthetic"
    )
    assert blocked["status"] == "blocked"
    assert blocked["indicator_counts"]["missing_metric"]["unknown"] == 1
    none_metrics = semantics.summarize_validity(
        [attempt("n", outcome="failed", failure_cause=None, aggregated_metrics=None)],
        required_metrics=REQUIRED,
        scope="synthetic",
    )
    assert none_metrics["attempts"][0]["indicators"]["missing_metric"] is True
    assert none_metrics["retry_causes"] == {"unknown_cause": 1}
    with pytest.raises(ValueError):
        semantics.summarize_validity(
            [attempt("a"), attempt("a")], required_metrics=REQUIRED, scope="synthetic"
        )
    with pytest.raises(ValueError):
        semantics.summarize_validity(
            [attempt("a", img_is_black_max=float("nan"))],
            required_metrics=REQUIRED,
            scope="synthetic",
        )


def test_e10_reports_offline_only_combined_events():
    report = semantics.reconcile_metric_domains(
        rollout(
            [
                metric("collision_any", [0, 0, 0, 0, 1.0]),
                metric("collision_front", [0, 0, 0, 0, 1.0]),
            ]
        ),
        scope="synthetic",
    )
    offline = report["offline_values"]
    assert offline["offroad_or_collision"] == offline["collision_at_fault"] == 1.0
    assert offline["offroad_or_collision_at_fault"] == 1.0
    assert "offroad_or_collision" not in {row["metric"] for row in report["metrics"]}


# --- independent-review regressions (A9-02, A9-08, A9-09, A9-14) ------------------------


def test_e10_missing_captured_aggregates_are_unverified_not_passed():
    empty = semantics.reconcile_metric_domains(rollout(aggregated={}), scope="synthetic")
    assert empty["status"] == "unverified"
    assert empty["status_meaning"] != "reconciliation complete; differences are facts, not failures"
    assert all(
        row["online_matches_capture"] is None for row in empty["metrics"]
    )  # nothing was compared
    partial = rollout()
    partial["aggregated_metrics"].pop("progress")
    one_missing = semantics.reconcile_metric_domains(partial, scope="synthetic")
    assert one_missing["status"] == "unverified"
    assert "online_aggregate_missing" in {i["code"] for i in one_missing["issues"]}
    extra = rollout()
    extra["aggregated_metrics"]["dist_to_gt"] = 1.0
    orphan = semantics.reconcile_metric_domains(extra, scope="synthetic")
    assert orphan["status"] == "unverified"
    assert "dense_metric_missing" in {i["code"] for i in orphan["issues"]}
    complete = semantics.reconcile_metric_domains(rollout(), scope="synthetic")
    assert complete["status"] == "passed"  # eval_relevant needs no captured aggregate


def test_e12_duplicate_gt_timestamp_is_unverified_without_an_rmse(pinned_distance):
    executed = _exec([(0, 0.0, 0.0, 0.0), (10, 10.0, 0.0, 0.0), (20, 20.0, 0.0, 0.0)])
    gt = _gt(0, [(0, 0.0, 0.0), (0, 5.0, 0.0), (20, 20.0, 0.0)])
    pinned = pinned_distance(executed, gt).report_metrics["gt_rmse_xy"]
    report = semantics.check_reference_alignment(executed, gt, scope="synthetic")
    assert report["status"] == "unverified"
    assert report["upstream_outcome"] == "not_reproduced"
    assert report["rmse_xy"] is None and report["num_paired"] is None
    assert "gt_duplicate_timestamps" in {i["code"] for i in report["issues"]}
    assert semantics.reference_rmse(executed, gt) == ("not_reproduced", "gt_duplicate_timestamps")
    assert pinned > 0  # the pinned np.interp path does compute a number
    compared = semantics.check_reference_alignment(
        executed,
        gt,
        captured={"gt_rmse_xy": pinned, "gt_num_paired": 3},
        tolerance={"absolute": 1e-9},
        scope="synthetic",
    )
    assert compared["status"] == "unverified" and compared["captured_matches"] is None
    raising = semantics.check_reference_alignment(
        _exec([(0, 0.0, 0.0, 0.0)]), gt, scope="synthetic"
    )
    assert raising["upstream_outcome"] == "raises"  # upstream raises before interpolating
    assert raising["raise_code"] == "too_few_executed_poses"


def test_e14_null_metric_value_is_missing_not_a_crash():
    report = semantics.summarize_validity(
        [attempt("a", outcome="failed", aggregated_metrics={"progress": None})],
        required_metrics=["progress"],
        scope="synthetic",
    )
    indicators = report["attempts"][0]["indicators"]
    assert indicators["missing_metric"] is True and indicators["nonfinite_metric"] is False


def test_e14_status_meaning_and_succeeded_attempt_missing_a_required_metric():
    invalid = attempt(
        "a",
        aggregated_metrics={"progress": 0.4, "collision_any": 0.0, "offroad": 0.0},
        img_is_black_max=1.0,
        max_dist_to_gt_trajectory_m=7.0,
        old_logprobs_nonfinite=True,
        nonfinite_resamples=3,
    )
    report = semantics.summarize_validity([invalid], required_metrics=REQUIRED, scope="synthetic")
    assert report["status"] == "passed"
    assert "not a validity pass" in report["status_meaning"]
    inconsistent = semantics.summarize_validity(
        [attempt("b", aggregated_metrics={"progress": 0.4})],
        required_metrics=REQUIRED,
        scope="synthetic",
    )
    assert inconsistent["status"] == "blocked"
    assert "succeeded_attempt_missing_metric" in {i["code"] for i in inconsistent["issues"]}
    none_metrics = semantics.summarize_validity(
        [attempt("c", aggregated_metrics=None)], required_metrics=REQUIRED, scope="synthetic"
    )
    assert "succeeded_attempt_missing_metric" in {i["code"] for i in none_metrics["issues"]}
