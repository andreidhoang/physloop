"""Agent-authored watchdog known answers; every Vast call goes to an injected fake runner.

Rewritten for build-spec A3 (R4 C5, P2-B, P2-C; parameterized window; review A3-14:
``--first-arm``/``--prior-events`` required, a missing prior file fails closed). Tests
that encoded the old defects changed:
- ``test_missing_bound_id_is_closed...`` used to accept one inventory without the ID
  as absence (C5). Absence now needs two consecutive exact-ID NotFound results.
- The old stop loop re-required a clean inventory before every stop, so a failing or
  ambiguous inventory past the deadline never stopped (P2-C) and a duplicate label
  after arming exited without stopping (P2-B). Stop is now exact-ID and confirmed by
  exact-ID state.
No test calls a real ``vastai`` or the network.
"""

import json
import subprocess
from pathlib import Path

import pytest

from scripts import vast_stop_watchdog as watchdog

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "s2-vast-host-20261101-v1"
LABEL = "physloop-test-unique"
NOT_FOUND = subprocess.CompletedProcess(
    [], 0, "", json.dumps({"error": True, "status_code": 404, "msg": "SECRET"}))
PROSE = subprocess.CompletedProcess([], 0, "stopping instance 123. SECRET\ntrue", "")


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


def row(instance_id=123, label=LABEL, status="running"):
    return {
        "id": instance_id,
        "label": label,
        "actual_status": status,
        "intended_status": "stopped" if status == "stopped" else "running",
        "next_state": "stopped" if status == "stopped" else "running",
        "api_key": "SECRET",
    }


class Runner:
    """Fake ``vastai`` runner keyed by subcommand; records argv lists."""

    def __init__(self, show_instances=(), show_instance=(), stop=()):
        self.queues = {"instances": list(show_instances), "instance": list(show_instance),
                       "stop": list(stop)}
        self.calls = []

    def __call__(self, argv, timeout):
        assert isinstance(argv, list) and timeout > 0
        self.calls.append(argv)
        key = "stop" if argv[2] == "stop" else argv[3]
        queue = self.queues[key]
        assert queue, f"unscripted call {argv}"
        value = queue.pop(0)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, subprocess.CompletedProcess):
            return value
        return subprocess.CompletedProcess(argv, 0, json.dumps(value), "SECRET stderr")


def exercise(tmp_path, runner, *, clock=None, deadline=100, window=3600, **kwargs):
    path = tmp_path / "events.jsonl"
    with watchdog.EventLog(path) as events:
        result = watchdog.watch(
            123, LABEL, deadline, events, max_window_seconds=window,
            clock=clock or Clock(), runner=runner, poll_seconds=1, command_timeout=2,
            **kwargs,
        )
    return result, [json.loads(line) for line in path.read_text().splitlines()]


LIST = ["vastai", "--raw", "show", "instances"]
SHOW = ["vastai", "--raw", "show", "instance", "123"]
STOP = ["vastai", "--raw", "stop", "instance", "123"]


def test_exact_binding_stop_then_exact_verify_and_no_raw_secrets(tmp_path, capsys):
    runner = Runner([[row(), row(999, "other")]], [row(), row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner)
    assert result == 0
    assert runner.calls == [LIST, SHOW, STOP, SHOW]
    assert [e["event"] for e in events] == ["deadline_reached", "stop_requested",
                                            "stopped_verified"]
    assert "SECRET" not in json.dumps(events) + capsys.readouterr().out


@pytest.mark.parametrize(
    "rows", [[row(label="wrong")], [row(), row(999)], [row(), row()], [row(instance_id=True)]]
)
def test_unverified_binding_with_wrong_label_or_duplicates_never_stops(tmp_path, rows):
    runner = Runner([rows])
    result, events = exercise(tmp_path, runner)
    assert result == 2
    assert STOP not in runner.calls
    assert events[-1]["event"] == "scope_mismatch"


def test_missing_bound_id_requires_two_consecutive_exact_not_found(tmp_path):
    runner = Runner([[row(999, "other")]], [NOT_FOUND, NOT_FOUND])
    result, events = exercise(tmp_path, runner, deadline=200)
    assert result == 0
    assert runner.calls == [LIST, SHOW, SHOW]
    assert events[-1]["event"] == "instance_absent"
    assert STOP not in runner.calls


@pytest.mark.parametrize(
    "second",
    [subprocess.CompletedProcess([], 0, "", '{"error": true, "status_code": 500}'),
     subprocess.CompletedProcess([], 1, "Traceback SECRET", ""),
     subprocess.TimeoutExpired(["x"], 2)],
)
def test_c5_empty_inventory_after_arming_is_not_absence(tmp_path, second, capsys):
    # R4 repro_04 A: armed, then one successful-but-empty inventory before the deadline.
    runner = Runner([[row()], [], [row()]], [NOT_FOUND, second, row(),
                                              row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=103)
    names = [e["event"] for e in events]
    assert names[0] == "armed"
    assert "instance_absent" not in names
    assert result == 0 and names[-1] == "stopped_verified"
    assert STOP in runner.calls
    assert "SECRET" not in json.dumps(events) + capsys.readouterr().out


def test_p2b_duplicate_label_after_arming_stops_bound_instance_by_exact_id(tmp_path):
    # R4 repro_04 B: a second instance takes the same label after arming.
    runner = Runner([[row()], [row(), row(456)]], [row(), row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=110)
    names = [e["event"] for e in events]
    assert names[:2] == ["armed", "scope_mismatch_after_arm"]
    assert names[-1] == "stopped_verified"
    assert result == 2  # money safe, anomaly still surfaced
    assert ["vastai", "--raw", "stop", "instance", "456"] not in runner.calls
    assert STOP in runner.calls


def test_label_on_another_instance_before_binding_exits_without_stop(tmp_path):
    runner = Runner([[row(999)]])
    result, events = exercise(tmp_path, runner, deadline=110)
    assert result == 2
    assert events[-1] == {**events[-1], "event": "scope_mismatch",
                          "error": "label_on_other_instance"}
    assert STOP not in runner.calls and len(runner.calls) == 1


def test_label_moving_to_another_instance_after_arming_stops_bound_id(tmp_path):
    runner = Runner([[row()], [row(999)]], [row(), row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=110)
    names = [e["event"] for e in events]
    assert names[:2] == ["armed", "scope_mismatch_after_arm"]
    assert events[1]["error"] == "label_on_other_instance"
    assert names[-1] == "stopped_verified" and result == 2
    assert STOP in runner.calls
    assert ["vastai", "--raw", "stop", "instance", "999"] not in runner.calls


def test_p2c_inventory_failing_past_deadline_still_stops_by_exact_id(tmp_path):
    # R4 repro_04 C: armed, then the inventory endpoint returns prose until past deadline.
    prose = subprocess.CompletedProcess([], 0, "Error: please try again SECRET", "")
    runner = Runner([[row()]] + [prose] * 10,
                    [prose, row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=105)
    assert result == 0
    assert STOP in runner.calls
    assert events[-1]["event"] == "stopped_verified"


def test_p2c_stop_never_confirmed_exits_stop_unconfirmed(tmp_path):
    failing = subprocess.CompletedProcess([], 0, "", '{"error": true, "status_code": 503}')
    runner = Runner([[row()]], [failing] * 50, [failing] * 50)
    clock = Clock()
    result, events = exercise(tmp_path, runner, clock=clock, deadline=101,
                              stop_retry_limit_seconds=100)
    assert result == 3
    assert events[-1]["event"] == "stop_unconfirmed"
    stops = [c for c in runner.calls if c == STOP]
    assert len(stops) >= 3
    # Exponential backoff capped at 60 s: 1, 2, 4, 8, 16, 32, 60 ...
    assert clock.mono < 1 + 100 + 60 + 1


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CompletedProcess([], 1, '{"success":true}', "SECRET"),
        subprocess.CompletedProcess([], 0, "", '{"error": true, "status_code": 401}'),
        subprocess.TimeoutExpired(["fake"], 2, output="SECRET", stderr="SECRET"),
        OSError("missing cli"),
    ],
)
def test_stop_error_is_recorded_and_retried_with_exact_verification(tmp_path, failure, capsys):
    runner = Runner([[row()]], [row(), row(), row(status="stopped")], [failure, PROSE])
    result, events = exercise(tmp_path, runner)
    assert result == 0
    assert runner.calls.count(STOP) == 2
    assert any(event["event"] == "command_error" for event in events)
    assert events[-1]["event"] == "stopped_verified"
    assert "SECRET" not in json.dumps(events) + capsys.readouterr().out


def test_stop_acknowledgment_without_stopped_state_does_not_complete(tmp_path):
    runner = Runner([[row()]], [row(), row(status="stopping"), row(status="stopped")],
                    [PROSE, PROSE])
    result, events = exercise(tmp_path, runner)
    assert result == 0
    assert runner.calls == [LIST, SHOW, STOP, SHOW, STOP, SHOW]
    assert [e["event"] for e in events].count("stopped_verified") == 1


@pytest.mark.parametrize("missing", [False, True])
def test_provisioning_with_unknown_status_still_arms_and_stops(tmp_path, missing):
    pending = row(status=None)
    if missing:
        pending.pop("actual_status")
    runner = Runner([[pending]], [pending, row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=101)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert runner.calls == [LIST, SHOW, STOP, SHOW]
    assert events[-1]["event"] == "stopped_verified"


@pytest.mark.parametrize(
    "intended,next_state", [("running", None), ("stopped", "running"), (None, None)]
)
def test_stopped_snapshot_with_pending_start_or_unknown_intent_remains_monitored(
    tmp_path, intended, next_state,
):
    pending = row(status="stopped")
    pending.update(intended_status=intended, next_state=next_state)
    runner = Runner([[pending]], [pending, row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=101)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert STOP in runner.calls
    assert events[-1]["event"] == "stopped_verified"


def test_show_timeout_retries_without_blind_stop_before_binding(tmp_path):
    runner = Runner([subprocess.TimeoutExpired(["fake"], 2), [row()]],
                    [row(), row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=101)
    assert result == 0
    assert runner.calls[:2] == [LIST, LIST]
    assert events[0]["event"] == "command_error"


def test_observed_provider_exited_shape_verifies_stop_despite_stale_message(tmp_path):
    shape = json.loads((EVIDENCE / "stop-state-072043.json").read_text())["instance"]
    stopped = row(status="exited")
    stopped.update({key: shape[key] for key in ("actual_status", "intended_status",
                                                "next_state", "status_msg")})
    runner = Runner([[stopped]])
    result, events = exercise(tmp_path, runner)
    assert result == 0
    assert runner.calls == [LIST]
    assert events[-1]["event"] == "stopped_verified"


@pytest.mark.parametrize("intended,next_state", [("running", "stopped"), ("stopped", "running")])
def test_exited_with_queued_start_remains_monitored(tmp_path, intended, next_state):
    pending = row(status="exited")
    pending.update(intended_status=intended, next_state=next_state)
    runner = Runner([[pending]], [pending, row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=101)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert STOP in runner.calls


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CompletedProcess([], 0, "SECRET malformed output", "SECRET"),
        UnicodeDecodeError("utf8", b"\xffSECRET", 0, 1, "invalid"),
        {"success": False, "api_key": "SECRET"},
    ],
)
def test_invalid_inventory_is_never_interpreted_as_closed(tmp_path, failure, capsys):
    runner = Runner([failure, [row(status="stopped")]])
    result, events = exercise(tmp_path, runner)
    assert result == 0
    assert len(runner.calls) == 2
    assert events[0]["event"] == "command_error"
    assert events[-1]["event"] == "stopped_verified"
    assert "SECRET" not in json.dumps(events) + capsys.readouterr().out


def test_monotonic_deadline_survives_wall_clock_rollback(tmp_path):
    clock = Clock()
    clock.rollback = True
    runner = Runner([[row()], [row()]], [row(), row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, clock=clock, deadline=102)
    assert result == 0
    assert events[0]["event"] == "armed"
    assert events[0]["deadline_utc_epoch"] == 102
    assert STOP in runner.calls


@pytest.mark.parametrize("window,deadline", [(3600, 3701), (43201, 200), (0, 100),
                                             (True, 100), (60, 161)])
def test_window_is_required_bounded_and_matches_deadline(tmp_path, window, deadline):
    runner = Runner()
    with pytest.raises(ValueError):
        exercise(tmp_path, runner, deadline=deadline, window=window)
    assert runner.calls == []


def test_twelve_hour_window_is_accepted_as_hard_bound(tmp_path):
    runner = Runner([[row(status="stopped")]])
    result, _ = exercise(tmp_path, runner, deadline=100 + 43200, window=43200)
    assert result == 0
    assert watchdog.MAX_WINDOW_SECONDS == 12 * 3600


def test_restart_never_renews_the_persisted_deadline(tmp_path):
    prior = tmp_path / "prior.jsonl"
    prior.write_text(json.dumps({"at_utc": "x", "event": "armed", "instance_id": 123,
                                 "expected_label": LABEL, "deadline_utc_epoch": 101}) + "\n")
    runner = Runner([[row()]], [row(), row(status="stopped")], [PROSE])
    result, events = exercise(tmp_path, runner, deadline=500,
                              prior_deadline_epoch=watchdog.prior_armed_deadline(
                                  prior, 123, LABEL))
    assert result == 0
    assert events[0] == {**events[0], "event": "deadline_clamped_to_prior_arm",
                         "deadline_utc_epoch": 101}
    assert STOP in runner.calls


def test_prior_deadline_reader_fails_closed(tmp_path):
    prior = tmp_path / "prior.jsonl"
    with pytest.raises(FileNotFoundError):  # a restart must name a real prior file
        watchdog.prior_armed_deadline(prior, 123, LABEL)
    prior.write_text(json.dumps({"event": "command_error"}) + "\n")
    assert watchdog.prior_armed_deadline(prior, 123, LABEL) is None  # never armed
    prior.write_text("not json\n")
    with pytest.raises(ValueError):
        watchdog.prior_armed_deadline(prior, 123, LABEL)
    prior.write_text("[1]\n")
    with pytest.raises(TypeError):
        watchdog.prior_armed_deadline(prior, 123, LABEL)
    prior.write_text(json.dumps({"event": "armed", "instance_id": 999,
                                 "expected_label": LABEL, "deadline_utc_epoch": 5}) + "\n")
    with pytest.raises(ValueError):
        watchdog.prior_armed_deadline(prior, 123, LABEL)  # foreign binding


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


def test_default_runner_uses_allowlisted_env_and_closed_stdin(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs, argv=argv)
        return subprocess.CompletedProcess(argv, 0, "[]", "")

    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    monkeypatch.setenv("VAST_API_KEY", "SECRET")
    watchdog.subprocess_runner(["vastai", "--raw", "show", "instances"], 2)
    assert captured["stdin"] is subprocess.DEVNULL
    assert "VAST_API_KEY" not in captured["env"]
    assert set(captured["env"]) <= set(watchdog.ENV_ALLOWLIST)
    assert not captured.get("shell")


def test_main_requires_max_window_and_rejects_naive_deadline(tmp_path, capsys):
    with pytest.raises(SystemExit):
        watchdog.main(["--instance-id", "123", "--expected-label", LABEL,
                       "--deadline-utc", "2027-01-15T09:00:00+00:00",
                       "--events", str(tmp_path / "e.jsonl")])
    code = watchdog.main(["--instance-id", "123", "--expected-label", LABEL,
                          "--deadline-utc", "2027-01-15T09:00:00",
                          "--max-window-seconds", "3600", "--first-arm",
                          "--events", str(tmp_path / "e.jsonl")])
    assert code == 2


# ---------------------------------------------------------------- review regressions (A3-14)


def main_args(tmp_path, *extra, deadline="2027-01-15T09:00:00+00:00"):
    return ["--instance-id", "123", "--expected-label", LABEL, "--deadline-utc", deadline,
            "--max-window-seconds", "3600", "--events", str(tmp_path / "e.jsonl"), *extra]


def test_a3_14_main_requires_first_arm_or_prior_events(tmp_path):
    with pytest.raises(SystemExit):
        watchdog.main(main_args(tmp_path))
    with pytest.raises(SystemExit):
        watchdog.main(main_args(tmp_path, "--first-arm", "--prior-events",
                                str(tmp_path / "p.jsonl")))
    assert not (tmp_path / "e.jsonl").exists()


def test_a3_14_restart_with_missing_prior_events_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        watchdog.prior_armed_deadline(tmp_path / "missing.jsonl", 123, LABEL)
    code = watchdog.main(main_args(tmp_path, "--prior-events", str(tmp_path / "missing.jsonl")))
    assert code == 2
    assert not (tmp_path / "e.jsonl").exists()


def test_a3_14_restart_refuses_a_deadline_later_than_the_prior_arm(tmp_path):
    prior = tmp_path / "prior.jsonl"
    prior.write_text(json.dumps({"event": "armed", "instance_id": 123,
                                 "expected_label": LABEL,
                                 "deadline_utc_epoch": 1_800_000_000}) + "\n")
    code = watchdog.main(main_args(tmp_path, "--prior-events", str(prior),
                                   deadline="2027-01-15T08:00:01+00:00"))
    assert code == 2
    assert not (tmp_path / "e.jsonl").exists()
