"""Synthetic known answers for exact job and attempt reconciliation."""

import json
from copy import deepcopy

import pytest

from gate.s2_diagnostics import _ATTEMPT_KEYS, _JOB_KEYS, _LEDGER_KEYS, _check_ledger
from gate.s2_ledger import (
    EVENT_SCHEMA_VERSIONS,
    PROCESS_COMPLETED,
    _valid_event,
    reconcile_attempts,
    valid_event,
)


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


def test_diagnostic_qualification_phase_is_a_schedule_phase_of_its_own():
    """Review G2-04: a declared diagnostic unit's ledger job carries its own phase, never
    the source pilot job's ``harmonizer_pilot``."""
    report = reconcile_attempts([job(phase="diagnostic_qualification")], [])
    assert codes(report) == {"missing_job"}
    assert report["jobs"][0]["phase"] == "diagnostic_qualification"
    assert "invalid_schedule_entry" in codes(reconcile_attempts([job(phase="diagnostic")], []))


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


# --- Event schema v2 (build-spec B1; R1 P4b): neutral process-completion reason ---------



def v2(event, refs=("capture:unit-attempt-evidence.json",)):
    return {**event, "schema_version": 2, "evidence_refs": list(refs)}


def v2_completed(**changes):
    return v2(terminal(reason="process_completed", **changes))


def test_v2_constants_are_literal():
    assert EVENT_SCHEMA_VERSIONS == (1, 2)
    assert PROCESS_COMPLETED == "process_completed"


def test_v2_process_completed_admits_and_records_the_neutral_reason():
    events = [v2(started(upstream_identity=None)), v2_completed()]
    report = reconcile_attempts([job()], events)
    assert report["admitted"], report["issues"]
    attempt = report["attempts"][0]
    assert attempt["state"] == "completed"
    assert attempt["terminal_reason"] == "process_completed"
    assert attempt["upstream_identity"] == identity()
    assert attempt["events"] == events


def test_report_and_record_key_sets_stay_identical_to_the_diagnostics_contract():
    for events in ([started(), terminal()], [v2(started()), v2_completed()]):
        report = reconcile_attempts([job()], events)
        assert set(report) == _LEDGER_KEYS
        assert set(report["jobs"][0]) == _JOB_KEYS
        assert set(report["attempts"][0]) == _ATTEMPT_KEYS
        assert set(report["counts"]) == {
            "scheduled_jobs", "attempts", "missing_jobs", "idempotent_events",
            "completed_attempts", "failed_attempts", "cancelled_attempts",
            "incomplete_attempts"}
        _check_ledger(report)


def test_v1_events_are_byte_identical_in_behaviour():
    events = [started(), terminal()]
    report = reconcile_attempts([job()], events)
    assert report["admitted"] and report["attempts"][0]["terminal_reason"] == "exact_horizon"


@pytest.mark.parametrize("event, reason", [
    # v1 keeps its exact-horizon contract; the neutral reason is a v2-only value.
    (lambda: terminal(reason="process_completed"), "v1 neutral reason"),
    # A v2 recorder never claims an exact horizon: that is evidence-derived downstream.
    (lambda: v2(terminal(reason="exact_horizon")), "v2 exact horizon"),
    (lambda: {k: v for k, v in v2_completed().items() if k != "evidence_refs"}, "no refs"),
    (lambda: {**v2_completed(), "schema_version": 3}, "unknown version"),
    (lambda: {**v2_completed(), "schema_version": True}, "bool version"),
    (lambda: {**terminal(), "schema_version": 1}, "explicit v1 marker"),
    (lambda: {**v2_completed(), "evidence_refs": "ref"}, "refs not a list"),
    (lambda: {**v2_completed(), "evidence_refs": ["a", "a"]}, "duplicate refs"),
    (lambda: {**v2_completed(), "evidence_refs": [" "]}, "blank ref"),
    (lambda: v2_completed(upstream_identity=None), "completion without identity"),
])
def test_v2_negative_controls_are_invalid_events(event, reason):
    events = [v2(started(upstream_identity=None)), event()]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"], reason
    assert "invalid_event" in codes(report), reason
    assert report["events"] == events


def test_v2_failure_cancellation_and_retry_chain_reconcile_like_v1():
    a2 = identity("2")
    events = [
        v2(started(upstream_identity=None)),
        v2(terminal(kind="failed", reason="upstream_rollout_failed")),
        v2(started("attempt-2", 1, parent_attempt_id="attempt-1",
                   retry_reason="runtime_internal_retry", upstream_identity=None)),
        v2_completed(attempt="attempt-2", time=1, upstream_identity=a2),
    ]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert codes(report) == {"retry_selection_unregistered"}
    assert [(a["state"], a["terminal_reason"]) for a in report["attempts"]] == [
        ("failed", "upstream_rollout_failed"), ("completed", "process_completed")]
    assert report["attempts"][1]["parent_attempt_id"] == "attempt-1"


def test_mixed_v1_and_v2_jobs_reconcile_in_one_batch():
    schedule = [job(), job("job-2", arm="B")]
    events = [
        started(), terminal(),
        v2(started("attempt-2", 2, job_id="job-2", upstream_identity=None)),
        v2_completed(attempt="attempt-2", time=3, job_id="job-2", upstream_identity=identity("2")),
    ]
    report = reconcile_attempts(schedule, events)
    assert report["admitted"], report["issues"]
    assert [a["terminal_reason"] for a in report["attempts"]] == [
        "exact_horizon", "process_completed"]


def test_v2_idempotent_repeat_and_conflict():
    first = v2(started(upstream_identity=None))
    report = reconcile_attempts([job()], [first, v2_completed(), deepcopy(first)])
    assert report["admitted"] and report["counts"]["idempotent_events"] == 1
    changed = {**first, "evidence_refs": ["other"]}
    report = reconcile_attempts([job()], [first, v2_completed(), changed])
    assert "conflicting_event_id" in codes(report)


@pytest.mark.parametrize("kind", ["started", "completed", "failed", "cancelled"])
def test_v2_event_without_evidence_refs_is_invalid(kind):
    """B1-F9: a v2 event (above all a completion) must bind at least one evidence ref."""
    first = v2(started(upstream_identity=None))
    terminal_event = (v2_completed() if kind == "completed"
                      else v2(terminal(kind=kind, reason="x")) if kind != "started" else None)
    if kind == "started":
        events = [v2(started(upstream_identity=None), refs=()), v2_completed()]
    else:
        events = [first, {**terminal_event, "evidence_refs": []}]
    report = reconcile_attempts([job()], events)
    assert not report["admitted"]
    assert "invalid_event" in codes(report)


def test_public_valid_event_alias_is_the_private_check():
    """Review F6: ``valid_event`` is public and identical to the kept ``_valid_event``."""
    assert valid_event is _valid_event
    assert valid_event(started()) is True
    assert valid_event(v2(started(upstream_identity=None))) is True
    assert valid_event(v2_completed()) is True
    # Negative controls: a v1 completion needs exact_horizon; v2 needs evidence refs.
    assert valid_event(terminal(reason="process_completed")) is False
    assert valid_event({**v2_completed(), "evidence_refs": []}) is False
    assert valid_event({**started(), "schema_version": True}) is False
    assert valid_event("not an event") is False
