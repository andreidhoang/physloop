"""Toy known answers for the protocol-design section 6 diagnostics (build-spec A9)."""

import pytest

from teacher_rl import diagnostics as d


def test_group_reward_stats_known_answers():
    stats = d.group_reward_stats([1.0, 3.0])
    assert (stats["mean"], stats["variance"], stats["tied"]) == (2.0, 1.0, False)
    assert d.group_reward_stats([0.5, 0.5, 0.5])["variance"] == 0.0
    assert d.group_reward_stats([2.0])["variance"] == "not_estimable"
    collapsed = d.group_reward_stats([0.1, 0.1 + 1e-12])
    assert collapsed["tied"] is False and collapsed["tied_float32"] is True


def _group(gid, members):
    return {
        "group_id": gid,
        "rewards": [m["progress"] - 10 * m["collision_any"] for m in members],
        "components": [
            {"progress": m["progress"], "collision_any": -10 * m["collision_any"]} for m in members
        ],
    }


GROUPS = [
    _group("tied", [{"progress": 0.4, "collision_any": 0}, {"progress": 0.4, "collision_any": 0}]),
    _group("fail", [{"progress": 0.2, "collision_any": 1}, {"progress": 0.3, "collision_any": 1}]),
    _group("mixed", [{"progress": 0.3, "collision_any": 0}, {"progress": 0.4, "collision_any": 1}]),
]


def test_classification_and_deciding_component():
    result = d.classify_groups(GROUPS, failure_components=["collision_any"])
    assert (result["tied_groups"], result["mixed_groups"], result["all_fail_groups"]) == (1, 2, 1)
    rows = {r["group_id"]: r for r in result["groups"]}
    assert rows["tied"]["deciding_component"] is None
    assert rows["fail"]["deciding_component"] == ["progress"]  # both collided: progress decides
    assert rows["mixed"]["deciding_component"] == ["collision_any"]  # 10 vs 0.1
    blocked = d.classify_groups(GROUPS, failure_components=None)
    assert blocked["all_fail_groups"] is None and blocked["tied_groups"] == 1


def test_negative_controls_for_classification():
    with pytest.raises(ValueError):
        d.classify_groups(GROUPS + [GROUPS[0]], failure_components=["collision_any"])
    with pytest.raises(ValueError):
        d.classify_groups(GROUPS, failure_components=["offroad"])
    with pytest.raises(ValueError):
        d.deciding_component([{"a": 1.0}, {"b": 1.0}], [1.0, 0.0])


def test_flip_share():
    pairs = [
        {"group_id": "a", "original": [1.0, 0.0], "resimulated": [0.9, 0.1]},
        {"group_id": "b", "original": [1.0, 0.0], "resimulated": [0.0, 1.0]},
        {"group_id": "c", "original": [1.0, 0.0, 0.5], "resimulated": [1.0, 0.0, 0.5]},
    ]
    result = d.flip_share(pairs)
    assert result["flipped"] == ["b"] and result["share"] == pytest.approx(1 / 3)
    assert d.flip_share([])["share"] == "not_estimable"
    tie_change = d.flip_share(
        [{"group_id": "t", "original": [1.0, 1.0], "resimulated": [1.0, 0.9]}]
    )
    assert tie_change["share"] == 1.0


def test_action_diversity():
    result = d.action_diversity(
        [
            {"group_id": "g", "actions": [[[0.0, 0.0]], [[3.0, 4.0]]]},
            {"group_id": "h", "actions": [[0.0], [1.0], [2.0]]},
            {"group_id": "s", "actions": [[1.0]]},
        ]
    )
    rows = {r["group_id"]: r["mean_pairwise_distance"] for r in result["groups"]}
    assert rows == {"g": 5.0, "h": pytest.approx(4 / 3), "s": "not_estimable"}
    with pytest.raises(ValueError):
        d.action_diversity([{"group_id": "x", "actions": [[1.0], [1.0, 2.0]]}])


def test_rows_after_first_hard_event():
    result = d.rows_after_first_hard_event(
        [
            {
                "rollout_id": "a",
                "row_timestamps_us": [0, 100, 200, 300],
                "first_hard_event_us": 100,
            },
            {"rollout_id": "b", "row_timestamps_us": [0, 100], "first_hard_event_us": None},
            {
                "rollout_id": "c",
                "row_timestamps_us": [0, 100, 200],
                "first_hard_event_us": "unknown",
            },
        ]
    )
    assert result == {
        "rows_total": 9,
        "rows_after_first_hard_event": 2,
        "rows_unknown": 3,
        "share": 2 / 6,
    }
    only_unknown = d.rows_after_first_hard_event(
        [{"rollout_id": "c", "row_timestamps_us": [0], "first_hard_event_us": "unknown"}]
    )
    assert only_unknown["share"] == "not_estimable"


def test_grad_norm_and_parameter_norms():
    stats = d.grad_norm_stats([0.0, 2.0, 4.0, "nan"], source="in_process_capture")
    assert stats == {
        "count": 4,
        "source": "in_process_capture",
        "exact_zero": 1,
        "printed_zero": None,
        "nonfinite": 1,
        "min": 0.0,
        "max": 4.0,
        "mean": 2.0,
    }
    assert d.grad_norm_stats([], source="in_process_capture")["mean"] == "not_estimable"
    norms = d.changed_parameter_norms(
        {"expert.layers.0.w": 3.0, "expert.norm.w": 4.0, "vlm.lm_head.weight": 0.0, "other.x": 1.0},
        prefixes=["vlm.", "expert.", "action_in_proj.", "action_out_proj."],
    )
    assert norms["modules"]["expert."] == {"tensors": 2, "l2_change": 5.0}
    assert norms["modules"]["vlm."] == {"tensors": 1, "l2_change": 0.0}
    assert norms["unmatched_tensors"] == ["other.x"]
    with pytest.raises(ValueError):
        d.changed_parameter_norms({"a": -1.0}, prefixes=["a"])


def test_clip_fraction_matches_pinned_definition():
    result = d.clip_fraction(
        [0.7, 0.8, 1.0, 1.2, 1.3, 5.0], [False] * 5 + [True], clip_low=0.2, clip_high=0.2
    )
    assert result == {"valid_rows": 5, "clipped_rows": 2, "fraction": 0.4}  # bounds are inclusive
    assert (
        d.clip_fraction([1.0], [True], clip_low=0.2, clip_high=0.2)["fraction"] == "not_estimable"
    )


def test_lag_and_throughput():
    lag = d.rollout_lag(
        [
            {"trainer_step": 5, "behaviour_version": 5},
            {"trainer_step": 5, "behaviour_version": 3},
            {"trainer_step": 6, "behaviour_version": None},
        ]
    )
    assert lag == {
        "n": 3,
        "unknown": 1,
        "max_lag": 2,
        "mean_lag": 1.0,
        "histogram": {"0": 1, "2": 1},
    }
    assert d.useful_samples_per_gpu_hour(720, 1800, 2) == {"gpu_hours": 1.0, "value": 720.0}
    assert d.useful_samples_per_gpu_hour(720, None, 2)["value"] == "not_estimable"


def test_summary_artifact_sections_and_owner_input():
    report = d.summarize_training_diagnostics(
        {
            "groups": {"groups": GROUPS, "failure_components": None},
            "kl": {"kl_beta": 0.0, "logged": [0.0], "source": "in_process_capture"},
            "throughput": {"valid_samples": 720, "paid_seconds": 1800, "gpu_count": 2},
        },
        scope="synthetic",
    )
    assert report["status"] == "blocked"
    assert "failure_components_unregistered" in {i["code"] for i in report["issues"]}
    assert report["sections"]["kl"]["kl_status"] == "not_computed"
    assert report["sections"]["clip"] == {"status": "not_run"}
    assert (
        report["artifact_kind"] == "s4_training_diagnostics" and report["diagnostic_only"] is True
    )
    ok = d.summarize_training_diagnostics(
        {"groups": {"groups": GROUPS, "failure_components": ["collision_any"]}}, scope="synthetic"
    )
    assert ok["status"] == "passed"
    with pytest.raises(ValueError):
        d.summarize_training_diagnostics({"unknown": []}, scope="synthetic")
    with pytest.raises(ValueError):
        d.summarize_training_diagnostics({}, scope=None)


# --- independent-review regressions (A9-04, A9-05, A9-06, A9-12, A9-13) -----------------


def test_parameter_norms_without_tensors_are_not_estimable():
    norms = d.changed_parameter_norms({"expert.layers.0.w": 3.0}, prefixes=["vlm.", "expert."])
    assert norms["modules"]["vlm."] == {"tensors": 0, "l2_change": "not_estimable"}
    assert norms["modules"]["expert."] == {"tensors": 1, "l2_change": 3.0}


def test_empty_diagnostics_are_not_run_not_passed():
    report = d.summarize_training_diagnostics({}, scope="synthetic")
    assert report["status"] == "not_run"
    assert {s["status"] for s in report["sections"].values()} == {"not_run"}


def test_kl_section_status_follows_kl_status():
    off = d.summarize_training_diagnostics(
        {"kl": {"kl_beta": 0.0, "logged": ["0.000000"], "source": "log_inferred"}},
        scope="synthetic",
    )
    assert off["sections"]["kl"]["status"] == "not_computed"
    printed = d.summarize_training_diagnostics(
        {"kl": {"kl_beta": 0.01, "logged": ["0.000000"], "source": "log_inferred"}},
        scope="synthetic",
    )
    assert printed["sections"]["kl"]["status"] == "unverified"
    assert printed["status"] == "unverified"


def test_rollout_lag_rejects_a_behaviour_version_from_the_future():
    with pytest.raises(ValueError):
        d.rollout_lag([{"trainer_step": 3, "behaviour_version": 5}])


def test_grad_norm_stats_logged_zero_is_not_an_exact_zero():
    logged = d.grad_norm_stats(["0.000000", "0.500000"], source="log_inferred")
    assert logged["exact_zero"] is None and logged["printed_zero"] == 1
    assert logged["min"] == 0.0 and logged["max"] == 0.5
    exact = d.grad_norm_stats([0.0, 0.5], source="in_process_capture")
    assert exact["exact_zero"] == 1 and exact["printed_zero"] is None
    with pytest.raises(ValueError):
        d.grad_norm_stats([0.0], source=None)
    with pytest.raises(ValueError):
        d.grad_norm_stats([0.0], source="log_inferred")  # parsed float, not the printed text
