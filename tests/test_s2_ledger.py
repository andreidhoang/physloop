"""Synthetic known answers for exact job and attempt reconciliation."""

import json
from copy import deepcopy

import pytest

from gate.s2_ledger import reconcile_attempts


def job(job_id="job-1", **changes):
    return {
        "job_id": job_id,
        "phase": "synthetic",
        "arm": "A",
        "policy_id": "teacher",
        "scene_id": "scene-1",
        "road_id": "road-1",
        "stratum": "urban",
        "replicate_id": 0,
        "protocol_hash": "a" * 64,
        "requested_steps": 3,
        **changes,
    }


def identity(number="1"):
    return {"run_uuid": "run-1", "clipgt_id": "clip-1", "rollout_id": number}


def started(attempt="attempt-1", time=0, **changes):
    return {
        "event_id": f"start-{attempt}",
        "job_id": "job-1",
        "attempt_id": attempt,
        "type": "started",
        "timestamp_us": time,
        "parent_attempt_id": None,
        "retry_reason": None,
        "upstream_identity": identity(),
        **changes,
    }


def terminal(attempt="attempt-1", time=1, kind="completed", **changes):
    return {
        "event_id": f"end-{attempt}",
        "job_id": "job-1",
        "attempt_id": attempt,
        "type": kind,
        "timestamp_us": time,
        "reason": "exact_horizon" if kind == "completed" else "synthetic failure",
        "upstream_identity": identity(),
        **changes,
    }


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_one_exact_attempt_admits_without_mutating_inputs():
    schedule, events = [job()], [started(), terminal()]
    original = deepcopy((schedule, events))
    report = reconcile_attempts(schedule, events)
    assert report["admitted"]
    assert report["issues"] == []
    assert report["jobs"][0]["selected_attempt_id"] == "attempt-1"
    assert report["jobs"][0]["status"] == "completed"
    assert report["attempts"][0]["events"] == events
    assert report["counts"]["scheduled_jobs"] == 1
    assert report["counts"]["completed_attempts"] == 1
    assert (schedule, events) == original
    json.dumps(report, allow_nan=False)


def test_two_arms_are_distinct_jobs():
    schedule = [job(), job("job-2", arm="B", policy_id="reexport")]
    events = [
        started(),
        terminal(),
        started("attempt-2", 2, job_id="job-2", upstream_identity=identity("2")),
        terminal("attempt-2", 3, job_id="job-2", upstream_identity=identity("2")),
    ]
    report = reconcile_attempts(schedule, events)
    assert report["admitted"]
    assert [x["selected_attempt_id"] for x in report["jobs"]] == ["attempt-1", "attempt-2"]


def test_failure_before_uuid_is_preserved():
    events = [started(upstream_identity=None), terminal(kind="failed", upstream_identity=None)]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert report["attempts"][0]["state"] == "failed"
    assert report["attempts"][0]["upstream_identity"] is None
    assert report["events"] == events
    assert report["counts"]["failed_attempts"] == 1


def test_uuid_can_be_bound_at_completion():
    report = reconcile_attempts([job()], [started(upstream_identity=None), terminal()])
    assert report["admitted"]
    assert report["attempts"][0]["upstream_identity"] == identity()


def test_successful_retry_keeps_failure_and_requires_selection_rule():
    events = [
        started(),
        terminal(kind="failed"),
        started(
            "attempt-2",
            2,
            parent_attempt_id="attempt-1",
            retry_reason="host restart",
            upstream_identity=identity("2"),
        ),
        terminal("attempt-2", 3, upstream_identity=identity("2")),
    ]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert "retry_selection_unregistered" in codes(report)
    assert [a["state"] for a in report["attempts"]] == ["failed", "completed"]
    assert report["jobs"][0]["selected_attempt_id"] is None
    assert report["events"] == events


def test_same_event_is_idempotent_even_after_later_events():
    first = started()
    events = [first, terminal(), deepcopy(first)]
    report = reconcile_attempts([job()], events)
    assert report["admitted"]
    assert report["counts"]["idempotent_events"] == 1
    assert report["counts"]["attempts"] == 1
    assert report["events"] == events


def test_conflicting_event_id_is_retained_and_blocks_entire_batch():
    bad = terminal(reason="different")
    events = [started(), terminal(), bad]
    report = reconcile_attempts([job()], events)
    assert "conflicting_event_id" in codes(report)
    assert report["events"] == events
    assert report["jobs"][0]["selected_attempt_id"] is None


@pytest.mark.parametrize(
    "events,expected",
    [
        ([], "missing_job"),
        ([started()], "incomplete_attempt"),
        ([terminal()], "invalid_transition"),
        ([started(), started(event_id="another-start")], "invalid_transition"),
        ([started(), terminal(), terminal(event_id="another-end")], "invalid_transition"),
        ([started(time=10), terminal(time=9)], "timestamp_regression"),
        ([started(), terminal(upstream_identity=identity("changed"))], "upstream_identity_changed"),
        ([started(), terminal(reason="clipped")], "invalid_event"),
        ([started(), terminal(upstream_identity=None)], "invalid_event"),
        ([started(), terminal(kind="cancelled", reason="")], "invalid_event"),
    ],
)
def test_missing_and_invalid_events_fail_closed(events, expected):
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert expected in codes(report)
    assert report["events"] == events


def test_timestamp_regression_across_attempts_is_not_silently_sorted():
    schedule = [job(), job("job-2", scene_id="scene-2")]
    events = [
        started(time=10),
        started("attempt-2", 9, job_id="job-2", upstream_identity=identity("2")),
    ]
    assert "timestamp_regression" in codes(reconcile_attempts(schedule, events))


@pytest.mark.parametrize(
    "parent,reason,expected",
    [
        (None, None, "retry_parent_missing"),
        ("unknown", "retry", "retry_parent_invalid"),
        ("attempt-1", "", "invalid_event"),
    ],
)
def test_retry_parent_and_reason_are_required(parent, reason, expected):
    events = [
        started(),
        terminal(kind="failed"),
        started(
            "attempt-2",
            2,
            parent_attempt_id=parent,
            retry_reason=reason,
            upstream_identity=identity("2"),
        ),
    ]
    assert expected in codes(reconcile_attempts([job()], events))


def test_retry_parent_must_be_terminal_and_latest_in_same_job():
    first = [
        started(),
        started(
            "attempt-2",
            1,
            parent_attempt_id="attempt-1",
            retry_reason="retry",
            upstream_identity=identity("2"),
        ),
    ]
    assert "retry_parent_invalid" in codes(reconcile_attempts([job()], first))
    branched = [
        started(),
        terminal(kind="failed"),
        started(
            "attempt-2",
            2,
            parent_attempt_id="attempt-1",
            retry_reason="retry",
            upstream_identity=identity("2"),
        ),
        terminal("attempt-2", 3, "failed", upstream_identity=identity("2")),
        started(
            "attempt-3",
            4,
            parent_attempt_id="attempt-1",
            retry_reason="branch",
            upstream_identity=identity("3"),
        ),
    ]
    assert "retry_parent_invalid" in codes(reconcile_attempts([job()], branched))


def test_multiple_successes_and_reused_upstream_ids_are_visible():
    events = [
        started(),
        terminal(),
        started("attempt-2", 2, parent_attempt_id="attempt-1", retry_reason="retry"),
        terminal("attempt-2", 3),
    ]
    report = reconcile_attempts([job()], events)
    assert {"multiple_completed_attempts", "reused_upstream_identity"} <= codes(report)
    assert report["counts"]["completed_attempts"] == 2
    assert len(report["attempts"]) == 2


@pytest.mark.parametrize(
    "second,expected",
    [
        (job(), "duplicate_job_id"),
        (job("job-2"), "duplicate_logical_job"),
        (job("job-2", arm="B", road_id="different"), "scene_identity_conflict"),
        (job("job-2", scene_id="scene-2", stratum="other"), "road_stratum_conflict"),
    ],
)
def test_schedule_conflicts_preserve_both_rows(second, expected):
    schedule = [job(), second]
    report = reconcile_attempts(schedule, [])
    assert expected in codes(report)
    assert report["schedule"] == schedule
    assert report["counts"]["scheduled_jobs"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"extra": True},
        {"replicate_id": True},
        {"requested_steps": 0},
        {"protocol_hash": "pending"},
        {"arm": "C"},
        {"phase": "S4"},
        {"scene_id": ""},
        {"policy_id": None},
    ],
)
def test_invalid_schedule_rows_are_rejected(change):
    report = reconcile_attempts([job(**change)], [])
    assert not report["admitted"]
    assert "invalid_schedule_entry" in codes(report)


def test_unknown_job_and_attempt_id_reuse_block():
    events = [started(job_id="unknown")]
    assert "unknown_job" in codes(reconcile_attempts([job()], events))
    schedule = [job(), job("job-2", arm="B")]
    events = [started(), started(time=1, event_id="another-start", job_id="job-2")]
    assert "reused_attempt_id" in codes(reconcile_attempts(schedule, events))


def test_invalid_batch_never_selects_success_from_other_job():
    report = reconcile_attempts([job(), job("job-2", arm="B")], [started(), terminal()])
    assert not report["admitted"]
    assert all(j["selected_attempt_id"] is None for j in report["jobs"])


@pytest.mark.parametrize("schedule,events", [(None, []), ([], []), ([job()], None)])
def test_invalid_top_level_shapes(schedule, events):
    report = reconcile_attempts(schedule, events)
    assert not report["admitted"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"timestamp_us": True},
        {"timestamp_us": -1},
        {"timestamp_us": 1.0},
        {"upstream_identity": {"run_uuid": "partial"}},
        {"type": []},
        {"job_id": []},
    ],
)
def test_malformed_events_stay_visible(change):
    events = [started(**change)]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert "invalid_event" in codes(report)
    assert report["events"] == events
    json.dumps(report, allow_nan=False)


def test_nonjson_value_is_explicit_and_output_stays_serializable():
    events = [started(timestamp_us=float("nan"))]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert "invalid_json_value" in codes(report)
    assert report["events"][0]["timestamp_us"] is None
    assert events[0]["timestamp_us"] != events[0]["timestamp_us"]
    json.dumps(report, allow_nan=False)


def test_cancelled_attempt_does_not_erase_known_upstream_identity():
    events = [started(), terminal(kind="cancelled", upstream_identity=None)]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert report["attempts"][0]["state"] == "cancelled"
    assert report["attempts"][0]["upstream_identity"] == identity()
    assert report["counts"]["cancelled_attempts"] == 1


def test_retry_cannot_use_parent_from_another_job():
    schedule = [job(), job("job-2", arm="B")]
    events = [
        started(),
        terminal(kind="failed"),
        started(
            "attempt-2",
            2,
            job_id="job-2",
            parent_attempt_id="attempt-1",
            retry_reason="wrong parent",
            upstream_identity=identity("2"),
        ),
    ]
    assert "retry_parent_invalid" in codes(reconcile_attempts(schedule, events))
