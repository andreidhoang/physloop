"""Known answers and negative controls for teacher_rl.watchdog (E18; lane-r3 finding 14).

Processes, clock, progress counter, command runner and lifecycle coordinator are fakes: no
test sends a signal, runs docker/nvidia-smi or calls a provider. The hand-off is checked
against the real pure ``gate.s2_execution.next_action`` on a two-GPU action record.
"""

import copy
import signal

import pytest

from gate import s2_execution as ex
from gate.s2_bootstrap import RunResult
from scripts.rehearse_s4 import synthetic_rental_record
from teacher_rl import watchdog

T0 = 1_800_000_000
OK_EMPTY = RunResult(0, b"", b"", False)
LAUNCH_HASH = "e" * 64
PROJECT = "wizard_0"
RECORDED = "a" * 64  # recorded at launch, may carry no Compose label
OURS = "b" * 64      # labelled with this run's Compose project
OTHER = "c" * 64     # another run's wizard container: never touched


def spec(**overrides):
    value = {"artifact_kind": "s4_watch_spec", "schema_version": 2, "scope": "synthetic",
             "run_id": "run-a", "launch_hash": LAUNCH_HASH, "compose_project": PROJECT,
             "container_ids": [RECORDED],
             "roles": [{"name": n, "role": r, "pid": p, "pgid": p} for n, r, p in (
                 ("host", "host", 101), ("wizard", "wizard", 102),
                 ("controller", "controller", 103), ("rollout_0", "rollout", 104),
                 ("policy_0", "policy", 105))],
             "stall_seconds": 600, "poll_seconds": 30, "deadline_epoch": T0 + 30_000,
             "provider_cutoff_epoch": T0 + 36_000, "export_reserve_seconds": 3600,
             "term_grace_seconds": 10}
    value.update(overrides)
    return value


class Processes:
    def __init__(self, dead=(), stubborn=(), immortal=(), exit_code=None):
        self.groups = {101, 102, 103, 104, 105}
        self.dead = set(dead)
        self.stubborn, self.immortal = set(stubborn), set(immortal)
        self.code = exit_code
        self.signals = []

    def alive(self, pid):
        return pid not in self.dead

    def exit_code(self, pid):
        return self.code

    def group_alive(self, pgid):
        return pgid in self.groups

    def signal_group(self, pgid, sig):
        self.signals.append((pgid, sig))
        if pgid in self.immortal or (pgid in self.stubborn and sig == signal.SIGTERM):
            return "sent"
        self.groups.discard(pgid)
        self.dead.add(pgid)
        return "sent"


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def time(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds


class Runner:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def __call__(self, argv, timeout_s):
        self.calls.append(list(argv))
        return self.results.pop(0)


class Counter:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def dog(processes=None, runner=None, progress=None, clock=None, **spec_overrides):
    return watchdog.TrainingWatchdog(
        spec(**spec_overrides), processes=processes or Processes(),
        progress=progress or Counter([1, 2, 3, 4, 5]),
        runner=runner or Runner([OK_EMPTY, OK_EMPTY]), clock=clock or Clock())


def obs(alive_names=("host", "wizard", "controller", "rollout_0", "policy_0"), code=None,
        progress=1):
    names = ("host", "wizard", "controller", "rollout_0", "policy_0")
    return {"alive": {n: n in alive_names for n in names}, "host_exit_code": code,
            "progress": progress}


def test_assess_known_answers():
    s = spec()
    st = watchdog.initial_state(s, T0)
    assert watchdog.assess(s, st, obs(), T0)["decision"] == "continue"
    out = watchdog.assess(s, st, obs(alive_names=("host", "wizard", "controller", "policy_0")),
                          T0)
    # lane-r3 finding 14: a dead rollout worker with a live host/controller ends the run
    assert out["decision"] == "terminate" and out["reasons"] == ["role_dead:rollout_0"]
    assert watchdog.assess(s, st, obs(), T0 + 30_000)["reasons"] == [
        "training_deadline_reached"]
    done = watchdog.assess(s, st, obs(alive_names=(), code=0), T0 + 10)
    assert done["decision"] == "completed" and done["reasons"] == ["completed"]
    assert watchdog.assess(s, st, obs(alive_names=(), code=None), T0)["reasons"] == [
        "host_exited"]
    assert watchdog.assess(s, st, obs(alive_names=("wizard",), code=0), T0)["reasons"] == [
        "host_exited:0"]
    assert watchdog.assess(s, st, obs(alive_names=(), code=3), T0)["reasons"] == [
        "host_exited:3"]


def test_assess_progress_stall_regression_and_unreadable_counter():
    s = spec()
    st = watchdog.assess(s, watchdog.initial_state(s, T0), obs(progress=5), T0)["state"]
    assert watchdog.assess(s, st, obs(progress=5), T0 + 599)["decision"] == "continue"
    assert watchdog.assess(s, st, obs(progress=5), T0 + 600)["reasons"] == ["progress_stalled"]
    assert watchdog.assess(s, st, obs(progress=6), T0 + 600)["decision"] == "continue"
    assert watchdog.assess(s, st, obs(progress=4), T0 + 10)["reasons"] == ["progress_regressed"]
    # an unreadable counter is not progress: the stall clock keeps running
    assert watchdog.assess(s, st, obs(progress=None), T0 + 600)["reasons"] == [
        "progress_stalled"]


@pytest.mark.parametrize("change", [
    {"stall_seconds": 0}, {"term_grace_seconds": None}, {"poll_seconds": 1.5},
    {"deadline_epoch": T0 + 33_000},  # + 3600 export reserve passes the provider cutoff
    {"export_reserve_seconds": -1}, {"scope": "real"}, {"schema_version": 1},
    {"schema_version": 3}, {"roles": []}, {"run_id": "../x"},
    # G1 container scope and launch binding
    {"compose_project": None, "container_ids": []},  # no scope: never by name pattern
    {"compose_project": "Wizard 0"}, {"compose_project": ""},
    {"container_ids": ["0123456789ab"]},  # short IDs are not exact
    {"container_ids": [RECORDED, RECORDED]}, {"container_ids": "a" * 64},
    {"launch_hash": None}, {"launch_hash": "E" * 64},
])
def test_spec_negative_controls(change):
    with pytest.raises(watchdog.SpecError):
        watchdog.validate_watch_spec(spec(**change))


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop(),  # no policy role
    lambda r: r[0].update(role="controller"),  # two controllers, no host
    lambda r: r[1].update(pgid=1),  # init's group
    lambda r: r[1].update(pid=0),
    lambda r: r[2].update(name="host"),  # duplicate name
    lambda r: r[2].update(role="trainer"),
])
def test_role_negative_controls(mutate):
    roles = copy.deepcopy(spec()["roles"])
    mutate(roles)
    with pytest.raises(watchdog.SpecError):
        watchdog.validate_watch_spec(spec(roles=roles))


def test_dead_rollout_worker_tears_down_every_group_and_verifies_clean():
    processes = Processes()
    runner = Runner([OK_EMPTY, OK_EMPTY])
    clock = Clock()
    w = dog(processes=processes, runner=runner, clock=clock)
    assert w.poll()["decision"] == "continue"
    processes.dead.add(104)  # rollout_0 crashes; upstream keeps running
    report = w.run(max_polls=5)
    assert report["decision"] == "terminate" and report["reasons"] == ["role_dead:rollout_0"]
    # SIGTERM in role order host, controller, policy, rollout, wizard; nothing needs SIGKILL
    assert processes.signals == [(101, signal.SIGTERM), (103, signal.SIGTERM),
                                 (105, signal.SIGTERM), (104, signal.SIGTERM),
                                 (102, signal.SIGTERM)]
    assert runner.calls == [list(watchdog.CONTAINER_LIST), list(watchdog.GPU_LIST)]
    assert report["cleanup_status"] == "verified_clean" and report["status"] == "passed"
    assert report["handoff_observation"] == {
        "kind": "work_failed", "reason": "s4_watchdog:role_dead:rollout_0",
        "decision_issue_codes": ["s4_watchdog:role_dead:rollout_0",
                                 "s4_cleanup:verified_clean"]}
    assert report["polls"] == 2


def test_stubborn_group_gets_sigkill_after_the_grace_period():
    processes = Processes(dead={104}, stubborn={102})
    clock = Clock()
    report = dog(processes=processes, clock=clock).run(max_polls=1)
    assert processes.signals[-1] == (102, signal.SIGKILL)
    assert clock.now == T0 + 10  # exactly the 10 s grace on the injected clock
    assert report["cleanup_status"] == "verified_clean"


def test_survivor_after_sigkill_is_dirty():
    processes = Processes(dead={104}, immortal={102})
    report = dog(processes=processes).run(max_polls=1)
    assert report["cleanup_status"] == "dirty" and report["survivor_groups"] == [102]
    assert "process_group_survived" in {i["code"] for i in report["issues"]}
    assert report["handoff_observation"]["decision_issue_codes"][-1] == "s4_cleanup:dirty"
    assert report["handoff_observation"]["kind"] == "work_failed"


@pytest.mark.parametrize("decision, reasons, cleanup, reason", [
    ("terminate", ["training_deadline_reached", "progress_stalled"], "verified_clean",
     "s4_watchdog:training_deadline_reached"),
    ("terminate", ["host_exited:1"], "unverified", "s4_watchdog:host_exited:1"),
    ("completed", ["completed"], "dirty", "s4_cleanup:dirty"),
    ("completed", ["completed"], "unverified", "s4_cleanup:unverified"),
])
def test_a3x_every_non_clean_end_hands_off_work_failed_with_its_cause(decision, reasons,
                                                                      cleanup, reason):
    observation = watchdog.handoff_observation(
        {"decision": decision, "reasons": reasons, "cleanup_status": cleanup})
    assert observation == {
        "kind": "work_failed", "reason": reason,
        "decision_issue_codes": [*(f"s4_watchdog:{r}" for r in reasons),
                                 f"s4_cleanup:{cleanup}"]}


def listing(*rows):
    return RunResult(0, "".join(f"{i} {p}\n" for i, p in rows).encode(), b"", False)


def test_only_this_runs_containers_are_removed_by_exact_id_and_relisted():
    first = listing((OTHER, "wizard_1"), (RECORDED, ""), (OURS, PROJECT), ("d" * 64, ""))
    after = listing((OTHER, "wizard_1"), ("d" * 64, ""))
    runner = Runner([first, OK_EMPTY, after, OK_EMPTY])
    report = dog(processes=Processes(dead={104}), runner=runner).run(max_polls=1)
    assert runner.calls == [list(watchdog.CONTAINER_LIST), ["docker", "rm", "-f", RECORDED, OURS],
                            list(watchdog.CONTAINER_LIST), list(watchdog.GPU_LIST)]
    assert report["cleanup_status"] == "verified_clean"
    assert report["remaining_containers"] == []


def test_g1_container_listing_command_and_parser_known_answers():
    assert watchdog.CONTAINER_LIST == (
        "docker", "ps", "-a", "--no-trunc", "--format",
        '{{.ID}} {{.Label "com.docker.compose.project"}}')
    assert not any("name=" in item for item in watchdog.CONTAINER_LIST)
    s = spec()
    text = f"{OTHER} wizard_1\n{RECORDED} \n{OURS} {PROJECT}\n{'d' * 64}\n\n"
    assert watchdog.parse_container_listing(text, s) == [RECORDED, OURS]
    # a container named like the wizard but in another project is never selected
    assert watchdog.parse_container_listing(f"{OTHER} wizard_0x\n", s) == []
    assert watchdog.parse_container_listing(f"{OTHER} {PROJECT}\n", dict(
        s, compose_project=None)) == []
    for bad in ("0123456789ab wizard_0\n", "not-an-id\n", f"{OURS.upper()} {PROJECT}\n"):
        assert watchdog.parse_container_listing(bad, s) is None


@pytest.mark.parametrize("results, status, code", [
    ([RunResult(1, b"", b"", False), OK_EMPTY], "unverified", "container_check_failed"),
    ([OK_EMPTY, RunResult(None, b"", b"", True)], "unverified", "gpu_check_failed"),
    ([RunResult(0, b"not-an-id\n", b"", False), OK_EMPTY], "unverified",
     "container_check_failed"),
    ([OK_EMPTY, RunResult(0, b"4242, python\n", b"", False)], "dirty", "gpu_processes_remain"),
    ([listing((OURS, PROJECT)), OK_EMPTY, listing((OURS, PROJECT)), OK_EMPTY], "dirty",
     "run_containers_remain"),
    ([listing((OURS, PROJECT)), OK_EMPTY, RunResult(1, b"", b"", False), OK_EMPTY],
     "unverified", "container_check_failed"),
])
def test_cleanup_is_never_clean_without_successful_empty_listings(results, status, code):
    report = dog(processes=Processes(dead={104}), runner=Runner(results)).run(max_polls=1)
    assert report["cleanup_status"] == status
    assert code in {i["code"] for i in report["issues"]}
    assert report["status"] == "blocked"


def test_runner_exception_is_unverified_not_clean():
    def boom(argv, timeout_s):
        raise OSError("no docker")

    report = dog(processes=Processes(dead={104}), runner=boom).run(max_polls=1)
    assert report["cleanup_status"] == "unverified"


def test_completed_run_hands_off_work_complete_and_continue_reports_nothing():
    processes = Processes(dead={101, 102, 103, 104, 105}, exit_code=0)
    processes.groups = set()
    report = dog(processes=processes).run(max_polls=1)
    assert report["decision"] == "completed" and processes.signals == []
    assert report["handoff_observation"] == {"kind": "work_complete"}
    still = dog().run(max_polls=2)
    assert still["decision"] == "continue" and still["handoff_observation"] is None
    assert still["status"] == "not_run"


def test_stall_with_a_frozen_counter_terminates_after_stall_seconds():
    clock = Clock()
    report = dog(progress=Counter([9]), clock=clock, poll_seconds=300).run(max_polls=10)
    assert report["reasons"] == ["progress_stalled"]
    assert clock.now == T0 + 600  # two sleeps of 300 s


def walk_to_running(now):
    record = synthetic_rental_record()
    report = ex.validate_action_record(record)
    assert report["admitted"], report["issues"]
    state = ex.initial_state(record, report, "0f8e2c1a-5b7d-4c3e-9a10-2b3c4d5e6f70",
                             watchdog_events="evidence/watchdog-events.jsonl")
    label = state["label"]
    row = {"id": 7, "label": label, "actual_status": "running", "intended_status": "running",
           "next_state": "running", "gpu_name": "RTX PRO 6000 WS", "num_gpus": 2,
           "disk_space": 400.0, "dph_total": 3.2, "image_uuid": "docker.io/vastai/kvm:ubuntu"}
    path = [None, {"kind": "inventory", "status": "ok", "rows": []},
            {"kind": "create_result", "outcome": "created", "instance_id": 7},
            {"kind": "watchdog_receipt", "armed": True, "instance_id": 7, "label": label,
             "deadline_epoch": T0 + 36_000},
            {"kind": "instance_state", "results": {"7": {"status": "row", "row": row}}},
            {"kind": "bootstrap_complete"}]
    for observation in path:
        state = ex.next_action(state, observation, now)["state"]
    assert state["state"] == "running" and state["plan"]["gpu_count"] == 2
    return state


class Coordinator:
    """Duck-typed stand-in for gate.s2_execution.Coordinator over the real next_action."""

    def __init__(self, state, clock):
        self.state, self.clock, self.pending, self.observed = state, clock, None, []

    def observe(self, observation):
        ex.validate_observation(observation)
        self.pending = observation
        self.observed.append(observation)

    def step(self):
        decision = ex.next_action(self.state, self.pending, self.clock.time())
        self.state, self.pending = decision["state"], None
        return decision


def test_hand_off_moves_the_lifecycle_to_export_then_stop():
    clock = Clock(T0 + 100)
    coordinator = Coordinator(walk_to_running(T0 + 10), clock)
    report = dog(processes=Processes(dead={104})).run(max_polls=1)
    decision = watchdog.hand_off(coordinator, report)
    assert coordinator.observed == [report["handoff_observation"]]
    assert decision["state"]["state"] == "exporting"
    assert decision["action"] == {"action": "export"}
    # A3X: a terminated run is journaled as failed work, never as budget exhaustion
    assert coordinator.observed[0]["kind"] == "work_failed"
    assert "work_failed" in decision["state"]["issues"]
    assert "budget_exhausted" not in decision["state"]["issues"]


def test_hand_off_at_the_provider_cutoff_stops_regardless_of_export():
    clock = Clock(T0 + 36_000)
    coordinator = Coordinator(walk_to_running(T0 + 10), clock)
    report = dog(processes=Processes(dead={104})).run(max_polls=1)
    decision = watchdog.hand_off(coordinator, report)
    assert decision["state"]["state"] == "cutoff_reached"
    assert "export_incomplete" in decision["state"]["issues"]


# --- review B3-05: the deadline teardown never consumes the export reserve ------------------


@pytest.mark.parametrize("change", [
    {"poll_seconds": 7200},  # longer than the 3600 s export reserve
    {"export_reserve_seconds": 0, "deadline_epoch": T0 + 36_000},
    # 30 poll + 2 x 10 grace + 4 x 60 command timeouts = 290 > 289
    {"export_reserve_seconds": 289, "deadline_epoch": T0 + 35_000},
    {"term_grace_seconds": 1700},  # 30 + 3400 + 240 > 3600
])
def test_spec_bounds_teardown_time_by_the_export_reserve(change):
    with pytest.raises(watchdog.SpecError):
        watchdog.validate_watch_spec(spec(**change))


def test_spec_at_the_teardown_bound_is_accepted():
    watchdog.validate_watch_spec(spec(export_reserve_seconds=290,
                                      deadline_epoch=T0 + 35_000))


def test_sleep_is_clamped_to_the_training_deadline():
    clock = Clock(T0 + 29_990)
    report = dog(clock=clock, progress=Counter(list(range(1, 50)))).run(max_polls=5)
    assert report["decision"] == "terminate"
    assert report["reasons"] == ["training_deadline_reached"]
    # one 10 s sleep up to the deadline, not the 30 s poll interval; the fake groups die on
    # SIGTERM so teardown spends no grace time
    assert clock.now == T0 + 30_000
    assert report["polls"] == 2


def test_long_poll_interval_never_oversleeps_the_deadline():
    clock = Clock(T0 + 28_800)
    report = dog(clock=clock, progress=Counter(list(range(1, 50))), poll_seconds=3000,
                 export_reserve_seconds=3600, stall_seconds=10**6).run(max_polls=5)
    assert report["reasons"] == ["training_deadline_reached"]
    assert clock.now == T0 + 30_000 < T0 + 36_000 - 3600 + 1


# --- review B3-06: hand_off never leaves a finished run billed in bootstrapping -------------


def walk_to_bootstrapping(now):
    record = synthetic_rental_record()
    report = ex.validate_action_record(record)
    state = ex.initial_state(record, report, "0f8e2c1a-5b7d-4c3e-9a10-2b3c4d5e6f70",
                             watchdog_events="evidence/watchdog-events.jsonl")
    label = state["label"]
    row = {"id": 7, "label": label, "actual_status": "running", "intended_status": "running",
           "next_state": "running", "gpu_name": "RTX PRO 6000 WS", "num_gpus": 2,
           "disk_space": 400.0, "dph_total": 3.2, "image_uuid": "docker.io/vastai/kvm:ubuntu"}
    for observation in [None, {"kind": "inventory", "status": "ok", "rows": []},
                        {"kind": "create_result", "outcome": "created", "instance_id": 7},
                        {"kind": "watchdog_receipt", "armed": True, "instance_id": 7,
                         "label": label, "deadline_epoch": T0 + 36_000},
                        {"kind": "instance_state", "results": {"7": {"status": "row",
                                                                     "row": row}}}]:
        state = ex.next_action(state, observation, now)["state"]
    assert state["state"] == "bootstrapping"
    return state


def completed_report():
    processes = Processes(dead={101, 102, 103, 104, 105}, exit_code=0)
    processes.groups = set()
    report = dog(processes=processes, clock=Clock(T0 + 100)).run(max_polls=1)
    assert report["handoff_observation"] == {"kind": "work_complete"}
    return report


def test_hand_off_in_bootstrapping_substitutes_work_failed():
    coordinator = Coordinator(walk_to_bootstrapping(T0 + 10), Clock(T0 + 200))
    decision = watchdog.hand_off(coordinator, completed_report())
    assert coordinator.observed == [{
        "kind": "work_failed", "reason": "s4_watchdog:lifecycle_state:bootstrapping",
        "decision_issue_codes": ["s4_watchdog:completed",
                                 "s4_watchdog:lifecycle_state:bootstrapping",
                                 "s4_cleanup:verified_clean"]}]
    assert decision["state"]["state"] == "exporting"
    assert "work_failed" in decision["state"]["issues"]
    assert "budget_exhausted" not in decision["state"]["issues"]
    assert "unexpected_observation" not in {i["code"] for i in decision["issues"]}


class RejectingCoordinator(Coordinator):
    """A coordinator whose lifecycle rejects the journaled observation."""

    def step(self):
        decision = super().step()
        decision["issues"] = [*decision["issues"], {"code": "unexpected_observation"}]
        return decision


def test_hand_off_raises_when_the_lifecycle_rejects_the_observation():
    coordinator = RejectingCoordinator(walk_to_running(T0 + 10), Clock(T0 + 100))
    with pytest.raises(RuntimeError, match="unexpected_observation"):
        watchdog.hand_off(coordinator, completed_report())


def test_hand_off_before_bootstrapping_raises():
    state = walk_to_bootstrapping(T0 + 10)
    state["state"] = "armed"
    with pytest.raises(RuntimeError, match="armed"):
        watchdog.hand_off(Coordinator(state, Clock(T0 + 200)), completed_report())


# --- review B3-09: the real adapter signals only groups whose leader identity is unchanged ---


STAT = b"4242 (py (x) thing) S 1 4242 4242 0 -1 4194560 0 0 0 0 1 2 0 0 20 0 1 0 98765 0 0\n"


def test_proc_stat_starttime_parser_known_answer():
    # field 22 (starttime) follows the last ')' as the 20th whitespace-separated field
    assert watchdog.parse_proc_stat(STAT) == {"pgrp": 4242, "starttime": 98765}
    with pytest.raises(ValueError):
        watchdog.parse_proc_stat(b"garbage")


class FakeProc:
    def __init__(self, table):
        self.table = dict(table)  # pid -> (pgrp, starttime)
        self.sent = []

    def stat(self, pid):
        if pid not in self.table:
            return None
        pgrp, start = self.table[pid]
        return {"pgrp": pgrp, "starttime": start}

    def killpg(self, pgid, sig):
        self.sent.append((pgid, sig))


def system(fake, roles=None):
    return watchdog.SystemProcesses(roles or spec()["roles"], read_stat=fake.stat,
                                    killpg=fake.killpg, getpgrp=lambda: 1)


def test_system_adapter_signals_a_verified_group():
    fake = FakeProc({p: (p, 1000 + p) for p in (101, 102, 103, 104, 105)})
    adapter = system(fake)
    assert adapter.signal_group(104, signal.SIGTERM) == "sent"
    assert fake.sent == [(104, signal.SIGTERM)]


def test_system_adapter_refuses_a_reused_group_id():
    fake = FakeProc({p: (p, 1000 + p) for p in (101, 102, 103, 104, 105)})
    adapter = system(fake)
    fake.table[104] = (104, 999_999)  # group died; an unrelated leader reused the id
    assert adapter.signal_group(104, signal.SIGTERM) == "refused_identity_changed"
    fake.table[105] = (77, 1105)  # pid 105 now belongs to another group
    assert adapter.signal_group(105, signal.SIGKILL) == "refused_identity_changed"
    assert adapter.signal_group(999, signal.SIGKILL) == "refused_unbound_group"
    assert fake.sent == []


def test_system_adapter_leaderless_group_stays_signalable_until_the_id_reappears():
    fake = FakeProc({p: (p, 1000 + p) for p in (101, 102, 103, 105)})  # leader 104 gone
    adapter = system(fake)
    assert adapter.signal_group(104, signal.SIGTERM) == "sent"  # members keep the id busy
    fake.table[104] = (104, 5)  # a new process now owns the id: the group is not ours
    assert adapter.signal_group(104, signal.SIGKILL) == "refused_identity_changed"


def test_system_adapter_refuses_a_role_outside_its_declared_group():
    fake = FakeProc({101: (101, 1), 102: (102, 2), 103: (103, 3), 104: (55, 4), 105: (105, 5)})
    adapter = system(fake)
    assert adapter.signal_group(104, signal.SIGTERM) == "refused_identity_changed"
    assert adapter.signal_group(101, signal.SIGTERM) == "sent"


def test_system_adapter_never_signals_its_own_group():
    fake = FakeProc({p: (p, p) for p in (101, 102, 103, 104, 105)})
    adapter = watchdog.SystemProcesses(spec()["roles"], read_stat=fake.stat,
                                       killpg=fake.killpg, getpgrp=lambda: 103)
    assert adapter.signal_group(103, signal.SIGTERM) == "refused_own_group"
    assert fake.sent == []


# ------------------------------------------------------------------- G1: watch-spec producer

LIMITS = {"stall_seconds": 600, "poll_seconds": 30, "deadline_epoch": T0 + 30_000,
          "provider_cutoff_epoch": T0 + 36_000, "export_reserve_seconds": 3600,
          "term_grace_seconds": 10}
LAUNCH = {"artifact_kind": "s4_launch_spec", "status": "passed", "scope": "synthetic",
          "launch_hash": LAUNCH_HASH}


def produce(**overrides):
    kwargs = {"launch_spec": LAUNCH, "run_id": "run-a",
              "role_records": copy.deepcopy(spec()["roles"]),
              "container_record": {"compose_project": PROJECT, "container_ids": [RECORDED]},
              "limits": dict(LIMITS)}
    kwargs.update(overrides)
    return watchdog.build_watch_spec(**kwargs)


def test_g1_producer_composes_exactly_the_v2_spec():
    assert produce() == spec()
    assert produce(container_record={"compose_project": None, "container_ids": [RECORDED]})[
        "compose_project"] is None


@pytest.mark.parametrize("overrides", [
    {"launch_spec": dict(LAUNCH, status="blocked")},
    {"launch_spec": dict(LAUNCH, artifact_kind="s4_watch_spec")},
    {"launch_spec": dict(LAUNCH, launch_hash=None)},
    {"container_record": {"compose_project": PROJECT}},
    {"container_record": {"compose_project": None, "container_ids": []}},
    {"limits": {k: v for k, v in LIMITS.items() if k != "stall_seconds"}},  # no default
    {"limits": dict(LIMITS, extra=1)},
    {"role_records": "host:101"},
    {"role_records": [{"name": "host", "role": "host", "pid": 101}]},
    {"run_id": ""},
])
def test_g1_producer_negative_controls(overrides):
    with pytest.raises(watchdog.SpecError):
        produce(**overrides)


def test_g1_directory_progress_counts_regular_bytes_and_fails_closed(tmp_path):
    root = tmp_path / "physloop_capture"
    root.mkdir()
    progress = watchdog.directory_progress(root)
    assert progress() == 0
    (root / "rollout-1.jsonl").write_bytes(b"x" * 7)
    (root / "blobs").mkdir()
    (root / "blobs" / "b1").write_bytes(b"y" * 5)
    (root / "link").symlink_to(root / "rollout-1.jsonl")  # never followed or counted
    assert progress() == 12
    assert watchdog.directory_progress(tmp_path / "absent")() is None
    (tmp_path / "alias").symlink_to(root)
    assert watchdog.directory_progress(tmp_path / "alias")() is None
