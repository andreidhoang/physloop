"""Agent-authored watchdog known answers; all Vast commands use a fake CLI."""

import json
import subprocess

import pytest

from scripts import vast_stop_watchdog as watchdog


class Clock:
    def __init__(self):
        self.wall = 100.0
        self.mono = 0.0
        self.rollback = False

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono

    def sleep(self, seconds):
        self.mono += seconds
        self.wall += -seconds if self.rollback else seconds


def row(instance_id=123, label="physloop-test-unique", status="running"):
    return {
        "id": instance_id,
        "label": label,
        "actual_status": status,
        "intended_status": "stopped" if status == "stopped" else "running",
        "next_state": "stopped" if status == "stopped" else "running",
        "api_key": "SECRET",
    }


def fake_cli(monkeypatch, responses):
    calls = []
    pending = iter(responses)

    def run(args, **kwargs):
        assert isinstance(args, list)
        assert kwargs["timeout"] > 0
        assert not kwargs.get("shell")
        calls.append(args)
        value = next(pending)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, subprocess.CompletedProcess):
            return value
        return subprocess.CompletedProcess(args, 0, json.dumps(value), "SECRET stderr")

    monkeypatch.setattr(watchdog.subprocess, "run", run)
    return calls


def exercise(tmp_path, *, clock=None, deadline=100):
    path = tmp_path / "events.jsonl"
    with watchdog.EventLog(path) as events:
        result = watchdog.watch(
            123,
            "physloop-test-unique",
            deadline,
            events,
            clock=clock or Clock(),
            poll_seconds=1,
            command_timeout=2,
        )
    return result, [json.loads(line) for line in path.read_text().splitlines()]


def test_exact_binding_stop_then_verify_and_no_raw_secrets(tmp_path, monkeypatch, capsys):
    calls = fake_cli(
        monkeypatch,
        [[row(), row(999, "other")], {"success": True}, [row(status="stopped"), row(999, "other")]],
    )
    result, events = exercise(tmp_path)
    assert result == 0
    assert calls == [
        ["vastai", "--raw", "show", "instances"],
        ["vastai", "--raw", "stop", "instance", "123"],
        ["vastai", "--raw", "show", "instances"],
    ]
    assert events[-1]["event"] == "stopped_verified"
    assert "SECRET" not in json.dumps(events) + capsys.readouterr().out


@pytest.mark.parametrize(
    "rows", [[row(label="wrong")], [row(), row(999)], [row(), row()], [row(instance_id=True)]]
)
def test_wrong_label_duplicate_id_or_nonunique_label_never_stops(tmp_path, monkeypatch, rows):
    calls = fake_cli(monkeypatch, [rows])
    result, events = exercise(tmp_path)
    assert result == 2
    assert not any("stop" in call for call in calls)
    assert events[-1]["event"] == "scope_mismatch"


def test_missing_bound_id_is_closed_and_never_stops_other_instance(tmp_path, monkeypatch):
    calls = fake_cli(monkeypatch, [[row(999, "other")]])
    result, events = exercise(tmp_path)
    assert result == 0
    assert len(calls) == 1
    assert events[-1]["event"] == "instance_absent"


@pytest.mark.parametrize(
    "failure",
    [
        {"success": False, "api_key": "SECRET"},
        subprocess.CompletedProcess([], 1, '{"success":true}', "SECRET"),
        subprocess.CompletedProcess([], 0, "stopping instance 123. SECRET", "SECRET"),
        subprocess.TimeoutExpired(["fake"], 2, output="SECRET", stderr="SECRET"),
    ],
)
def test_stop_error_is_recorded_and_retried_with_fresh_scope_check(
    tmp_path,
    monkeypatch,
    failure,
    capsys,
):
    calls = fake_cli(
        monkeypatch, [[row()], failure, [row()], {"success": True}, [row(status="stopped")]]
    )
    result, events = exercise(tmp_path)
    assert result == 0
    assert sum("stop" in call for call in calls) == 2
    assert any(event["event"] == "command_error" for event in events)
    assert events[-1]["event"] == "stopped_verified"
    assert "SECRET" not in json.dumps(events) + capsys.readouterr().out


def test_success_response_without_stopped_state_does_not_complete(tmp_path, monkeypatch):
    calls = fake_cli(
        monkeypatch,
        [
            [row()],
            {"success": True},
            [row(status="stopping")],
            {"success": True},
            [row(status="stopped")],
        ],
    )
    result, events = exercise(tmp_path)
    assert result == 0
    assert len(calls) == 5
    assert [e["event"] for e in events].count("stopped_verified") == 1


@pytest.mark.parametrize("missing", [False, True])
def test_provisioning_with_unknown_status_still_arms_and_stops(tmp_path, monkeypatch, missing):
    pending = row(status=None)
    if missing:
        pending.pop("actual_status")
    calls = fake_cli(
        monkeypatch, [[pending], [pending], {"success": True}, [row(status="stopped")]]
    )
    result, events = exercise(tmp_path, deadline=101)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert "stop" in calls[2]
    assert events[-1]["event"] == "stopped_verified"


@pytest.mark.parametrize(
    "intended,next_state", [("running", None), ("stopped", "running"), (None, None)]
)
def test_stopped_snapshot_with_pending_start_or_unknown_intent_remains_monitored(
    tmp_path,
    monkeypatch,
    intended,
    next_state,
):
    pending = row(status="stopped")
    pending.update(intended_status=intended, next_state=next_state)
    calls = fake_cli(
        monkeypatch, [[pending], [pending], {"success": True}, [row(status="stopped")]]
    )
    result, events = exercise(tmp_path, deadline=101)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert "stop" in calls[2]
    assert events[-1]["event"] == "stopped_verified"


def test_show_timeout_retries_without_blind_stop(tmp_path, monkeypatch):
    calls = fake_cli(
        monkeypatch,
        [
            subprocess.TimeoutExpired(["fake"], 2),
            [row()],
            {"success": True},
            [row(status="stopped")],
        ],
    )
    result, events = exercise(tmp_path)
    assert result == 0
    assert "stop" not in calls[1]
    assert events[0]["event"] == "command_error"


def test_observed_provider_exited_shape_verifies_stop_despite_stale_message(tmp_path, monkeypatch):
    # Observed after the real stop; identity is replaced with the synthetic test binding.
    stopped = row(status="exited")
    stopped.update(
        intended_status="stopped", next_state="stopped", status_msg="success, running ..."
    )
    calls = fake_cli(monkeypatch, [[stopped]])
    result, events = exercise(tmp_path)
    assert result == 0
    assert len(calls) == 1
    assert events[-1]["event"] == "stopped_verified"


@pytest.mark.parametrize("intended,next_state", [("running", "stopped"), ("stopped", "running")])
def test_exited_with_queued_start_remains_monitored(tmp_path, monkeypatch, intended, next_state):
    pending = row(status="exited")
    pending.update(intended_status=intended, next_state=next_state)
    calls = fake_cli(
        monkeypatch, [[pending], [pending], {"success": True}, [row(status="stopped")]]
    )
    result, events = exercise(tmp_path, deadline=101)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert "stop" in calls[2]
    assert events[-1]["event"] == "stopped_verified"


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CompletedProcess([], 0, "SECRET malformed output", "SECRET"),
        UnicodeDecodeError("utf8", b"\xffSECRET", 0, 1, "invalid"),
        {"success": False, "api_key": "SECRET"},
    ],
)
def test_invalid_inventory_is_never_interpreted_as_closed(tmp_path, monkeypatch, failure, capsys):
    calls = fake_cli(monkeypatch, [failure, [row(status="stopped")]])
    result, events = exercise(tmp_path)
    assert result == 0
    assert len(calls) == 2
    assert events[0]["event"] == "command_error"
    assert events[-1]["event"] == "stopped_verified"
    assert "SECRET" not in json.dumps(events) + capsys.readouterr().out


def test_monotonic_deadline_survives_wall_clock_rollback(tmp_path, monkeypatch):
    clock = Clock()
    clock.rollback = True
    calls = fake_cli(
        monkeypatch, [[row()], [row()], [row()], {"success": True}, [row(status="stopped")]]
    )
    result, events = exercise(tmp_path, clock=clock, deadline=102)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert events[0]["deadline_utc_epoch"] == 102
    assert clock.mono == 3
    assert "stop" in calls[3]


def test_deadline_over_one_hour_is_rejected_without_cli(tmp_path, monkeypatch):
    calls = fake_cli(monkeypatch, [])
    with pytest.raises(ValueError):
        exercise(tmp_path, deadline=3701)
    assert calls == []


def test_event_log_is_exclusive_append_only_and_fsynced(tmp_path, monkeypatch):
    syncs = []
    monkeypatch.setattr(watchdog.os, "fsync", lambda fd: syncs.append(fd))
    path = tmp_path / "events.jsonl"
    with watchdog.EventLog(path) as events:
        events.emit("first")
        first = path.read_bytes()
        events.emit("second")
        assert path.read_bytes().startswith(first)
    assert len(syncs) == 2
    with pytest.raises(FileExistsError):
        watchdog.EventLog(path)
