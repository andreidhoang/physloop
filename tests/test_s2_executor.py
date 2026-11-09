"""Literal known answers for the pure S2 unit plan and recorder events (build-spec B1)."""

import ast
import copy
import logging
import re
from pathlib import Path

import pytest
from s2_packet_fixture import diagnostic_unit, full_sweep_request, packet_request

from gate.s2_capture import reconcile_unit_attempts
from gate.s2_executor import (
    CAPTURE_IDS,
    UNIT_STEPS,
    build_execution_plan,
    cancelled_event,
    container_ids,
    driver_ready_line,
    has_upstream_attempt_evidence,
    inspect_argv,
    shm_override,
    started_event,
    terminal_established,
    unit_terminal_events,
    worst_case_unit_seconds,
)
from gate.s2_identity import identity_hash
from gate.s2_ledger import reconcile_attempts
from gate.s2_packet import build_full_sweep_packet, build_pilot_packet

DRIVER_MAIN = (Path(__file__).parent / "fixtures" / "alpasim_launch_upstream" / "alpasim"
               / "src" / "driver" / "src" / "alpasim_driver" / "main.py.txt")
RUN_CONFIG = {"schema_version": 1, "wizard_prepare_timeout_s": 600,
              "compose_config_timeout_s": 60, "driver_ready_timeout_s": 900,
              "driver_poll_interval_s": 5, "compose_up_timeout_s": 3600,
              "capture_timeout_s": 120, "teardown_timeout_s": 60, "port_release_timeout_s": 30}
SCENE = "clip-1"
RUN_UUID = "9b2e5c1a-7f3d-4e8b-a1c2-3d4e5f6a7b8c"
UUID_A = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
UUID_B = "6fa459ea-ee8a-3ca4-894e-db77e160355e"
JOB = "j" * 64


@pytest.fixture(scope="module")
def packet():
    built = build_pilot_packet(packet_request())
    assert built["prepared"]
    return built


def codes(report):
    return {item["code"] for item in report["issues"]}


# ------------------------------------------------------------------ plan

def test_step_order_is_the_literal_runbook_sequence():
    assert UNIT_STEPS == (
        "budget_check", "ledger_started", "run_dirs_absent", "wizard_prepare",
        "generated_config_check", "compose_shm_override", "compose_config_check",
        "driver_port_free", "driver_start", "driver_ready", "compose_up",
        "capture:compose_up_stdout", "capture:compose_logs", "capture:compose_ps",
        "capture:compose_container_ids", "capture:container_inspect",
        "run_metadata_capture", "run_dir_hash", "compose_down", "driver_stop",
        "driver_port_released", "reconcile", "ledger_terminal")


def test_plan_binds_packet_argv_verbatim_in_registered_order(packet):
    plan = build_execution_plan(packet, RUN_CONFIG)
    assert plan["status"] == "passed", plan["issues"]
    assert plan["scope"] == "synthetic" and plan["commands_executed"] is False
    assert plan["input_hashes"] == {"packet": identity_hash(packet),
                                    "run_config": identity_hash(RUN_CONFIG)}
    assert [unit["job_id"] for unit in plan["unit_plans"]] == \
        packet["inputs"]["execution_order"]["job_ids"]
    unit, first = packet["units"][0], plan["unit_plans"][0]
    steps = {step["id"]: step for step in first["steps"]}
    assert [step["id"] for step in first["steps"]] == list(UNIT_STEPS)
    assert steps["wizard_prepare"]["command"] == unit["wizard_prepare"]
    assert steps["driver_start"]["command"] == unit["driver"]
    assert steps["compose_up"]["command"] == unit["compose"]
    assert steps["compose_up"]["stdout_path"] == unit["capture_dir"] + "/compose-up.stdout.txt"
    assert steps["compose_down"]["command"] == unit["teardown"]
    assert [steps[f"capture:{c}"]["capture"] for c in CAPTURE_IDS] == \
        unit["required_captures_before_teardown"]
    assert steps["ledger_started"]["attempt_id"] == f"{unit['job']['job_id']}/attempt-1"
    assert steps["driver_ready"]["port"] == 6789
    assert steps["driver_ready"]["timeout_s"] == 900
    assert steps["compose_shm_override"]["content"] is None
    assert steps["run_dirs_absent"]["paths"] == [
        unit["compose"]["cwd"], unit["driver_output_path"].rsplit("/", 1)[0]]
    teardown = [s["id"] for s in first["steps"] if s["always_after"]]
    assert teardown == ["compose_down", "driver_stop", "driver_port_released"]
    assert {s["on_failure"] for s in first["steps"] if s["stage"] == "capture"} == {"record"}
    assert steps["generated_config_check"]["on_failure"] == "stop_run"


def test_shm_override_known_answers():
    default = dict.fromkeys(("controller-0", "physics-0", "renderer-0", "runtime-0"),
                            "docker_default")
    assert shm_override(default) is None
    assert shm_override({**default, "renderer-0": "64g", "runtime-0": "1g"}) == {
        "services": {"renderer-0": {"shm_size": "64g"}, "runtime-0": {"shm_size": "1g"}}}
    with pytest.raises(ValueError):
        shm_override(["renderer-0"])
    with pytest.raises(ValueError):
        shm_override({"renderer-0": 64})


def mutate(packet, change):
    copied = copy.deepcopy(packet)
    change(copied)
    return copied


@pytest.mark.parametrize("change, code", [
    (lambda p: p["units"][0]["wizard_prepare"]["argv"].append("x=1"), "commands_hash_mismatch"),
    (lambda p: p["units"][0]["compose"].update(argv=["docker", "compose", "up"]),
     "commands_hash_mismatch"),
    (lambda p: p["units"][0]["required_captures_before_teardown"].reverse(),
     "unexpected_capture_plan"),
    (lambda p: p["units"][0]["teardown"].update(requires=[]), "unexpected_teardown"),
    (lambda p: p["units"][0].update(index=5), "unit_order_mismatch"),
    (lambda p: p["units"][1].update(capture_dir="/elsewhere"), "unexpected_capture_dir"),
    (lambda p: p["units"][0]["job"].update(scene_id="other"), "unit_job_unscheduled"),
    (lambda p: p["units"][0]["compose_expectations"].update(scope="actual"), "scope_mismatch"),
    (lambda p: p.update(prepared=False), "packet_not_prepared"),
    (lambda p: p.update(schema_version=1), "unsupported_packet"),
    (lambda p: p.update(scope=None), "invalid_scope"),
    (lambda p: p.update(units=[]), "invalid_packet"),
    (lambda p: p["issues"].append({"code": "x", "severity": "blocking"}), "packet_not_prepared"),
])
def test_tampered_or_unprepared_packets_block(packet, change, code):
    plan = build_execution_plan(mutate(packet, change), RUN_CONFIG)
    assert plan["status"] == "blocked" and code in codes(plan) and plan["unit_plans"] == []


def test_compose_argv_without_pull_never_is_refused_even_with_a_matching_hash(packet):
    def change(copied):
        unit = copied["units"][0]
        unit["compose"]["argv"] = ["docker", "compose", "up", "--exit-code-from", "runtime-0"]
        unit["commands_sha256"] = identity_hash(
            {key: unit[key] for key in ("wizard_prepare", "driver", "compose")})
    plan = build_execution_plan(mutate(packet, change), RUN_CONFIG)
    # B1-F1: the edit no longer matches the packet rebuilt from its inputs either, so the
    # rebuild check fires beside the literal Compose check (this set was exact before).
    assert codes(plan) == {"unexpected_compose_command", "packet_rebuild_mismatch"}


def _recompute_commands(unit):
    unit["commands_sha256"] = identity_hash(
        {key: unit[key] for key in ("wizard_prepare", "driver", "compose")})


def _wizard_autoresume(copied):
    unit = copied["units"][0]
    unit["wizard_prepare"]["argv"] = [
        "runtime.enable_autoresume=true" if item == "runtime.enable_autoresume=false" else item
        for item in unit["wizard_prepare"]["argv"]]
    assert "runtime.enable_autoresume=true" in unit["wizard_prepare"]["argv"]
    _recompute_commands(unit)


def _inputs_and_request_hash(copied):
    copied["inputs"]["launch_request"]["requested_steps"] = 999
    copied["request_sha256"] = identity_hash(copied["inputs"])


@pytest.mark.parametrize("change, expected", [
    # Wizard argv edited and commands_sha256 recomputed in the same file.
    (_wizard_autoresume, {"packet_rebuild_mismatch"}),
    # argv/env outside commands_sha256, no hash touched.
    (lambda p: p["units"][0]["teardown"].update(argv=["sh", "-c", 'rm -rf "$HOME"/runs']),
     {"packet_rebuild_mismatch"}),
    (lambda p: p["units"][0]["compose_config"].update(argv=["sh", "-c", "true"]),
     {"packet_rebuild_mismatch"}),
    (lambda p: p["units"][0]["required_captures_before_teardown"][1].update(
        argv=["docker", "compose", "down", "-v"]), {"packet_rebuild_mismatch"}),
    (lambda p: p["units"][0]["required_captures_before_teardown"][4].update(
        argv_template=["sh", "-c", "{container_ids}"]), {"packet_rebuild_mismatch"}),
    (lambda p: p["units"][0]["required_captures_before_teardown"][1]["env_set"].update(
        DOCKER_HOST="tcp://203.0.113.9:2375"), {"packet_rebuild_mismatch"}),
    (lambda p: p["units"][0]["teardown"]["env_allowlist"].append("DOCKER_HOST"),
     {"packet_rebuild_mismatch"}),
    (lambda p: p["schedule"][0].update(replicate_id=7),
     {"unit_job_unscheduled", "packet_rebuild_mismatch"}),
    # Inputs edited with the recorded request hash left alone.
    (lambda p: p["inputs"]["launch_request"].update(requested_steps=999),
     {"request_hash_mismatch"}),
    # Inputs edited and the request hash recomputed: the units no longer follow from them.
    (_inputs_and_request_hash, {"packet_rebuild_mismatch"}),
])
def test_packet_edited_after_preparation_is_refused(packet, change, expected):
    """B1-F1: everything that runs is bound to the inputs the authority's hash names."""
    plan = build_execution_plan(mutate(packet, change), RUN_CONFIG)
    assert plan["status"] == "blocked" and plan["unit_plans"] == []
    assert codes(plan) == expected


def test_untouched_packet_rebuilds_identically(packet):
    assert identity_hash(packet["inputs"]) == packet["request_sha256"]
    rebuilt = build_pilot_packet(packet["inputs"])
    assert identity_hash(rebuilt["units"]) == identity_hash(packet["units"])
    assert build_execution_plan(packet, RUN_CONFIG)["status"] == "passed"


@pytest.mark.parametrize("config", [
    None, {}, {**RUN_CONFIG, "extra": 1}, {**RUN_CONFIG, "schema_version": 2},
    {**RUN_CONFIG, "compose_up_timeout_s": 0}, {**RUN_CONFIG, "capture_timeout_s": True},
    {**RUN_CONFIG, "driver_poll_interval_s": 900},
    {**RUN_CONFIG, "teardown_timeout_s": float("nan")},
])
def test_run_config_is_required_without_defaults(packet, config):
    plan = build_execution_plan(packet, config)
    assert plan["status"] == "blocked" and "invalid_run_config" in codes(plan)


# ------------------------------------------------------------------ packet kinds

# Regression locks on this fixture, not a specification: identity hashes of the pilot plan
# without ``producer`` (this module's source hash) and ``input_hashes`` (the packet hash
# carries gate/s2_packet.py's source hash; asserted directly below). The broader pins
# (``producer`` only excluded) were computed before the full-sweep dispatch was added and
# held after it; these narrower ones were then taken, so pilot plan output is locked.
PILOT_PLAN_SHA256 = "901c0b9fa185a73e6049386b33e2e2ed7abefc25a33552923d4687cee9ede6af"
PILOT_REBUILT_INPUTS_PLAN_SHA256 = (
    "eb7cdfcb82a045e736911655dd08c657e21f1cbc98853b010accdb45a6c8c590")


def _without_source_hashes(plan):
    return {key: value for key, value in plan.items()
            if key not in ("producer", "input_hashes")}


def test_pilot_plan_output_is_unchanged_by_the_packet_kind_dispatch(packet):
    plan = build_execution_plan(packet, RUN_CONFIG)
    assert identity_hash(_without_source_hashes(plan)) == PILOT_PLAN_SHA256
    assert plan["input_hashes"] == {"packet": identity_hash(packet),
                                    "run_config": identity_hash(RUN_CONFIG)}
    tampered = mutate(packet, _inputs_and_request_hash)
    blocked = build_execution_plan(tampered, RUN_CONFIG)
    assert identity_hash(_without_source_hashes(blocked)) == PILOT_REBUILT_INPUTS_PLAN_SHA256
    assert blocked["issues"][0]["message"] == (
        "units/schedule differ from build_pilot_packet(packet.inputs): the packet was "
        "edited after preparation")
    old = build_execution_plan(mutate(packet, lambda p: p.update(schema_version=1)), RUN_CONFIG)
    assert [(i["code"], i["message"]) for i in old["issues"]] == [
        ("unsupported_packet", "Require an s2_pilot_packet of schema version 2")]


@pytest.fixture(scope="module")
def sweep():
    built = build_full_sweep_packet(full_sweep_request())
    assert built["prepared"] and built["artifact_kind"] == "s2_full_sweep_packet"
    return built


def test_full_sweep_packet_plans_every_unit_with_its_own_port_and_dirs(sweep):
    plan = build_execution_plan(sweep, RUN_CONFIG)
    assert plan["status"] == "passed", plan["issues"]
    assert plan["scope"] == "synthetic" and plan["commands_executed"] is False
    assert plan["input_hashes"]["packet"] == identity_hash(sweep)
    assert len(plan["unit_plans"]) == 2646
    assert [unit["job_id"] for unit in plan["unit_plans"]] == \
        sweep["inputs"]["execution_order"]["job_ids"]
    assert [unit["port"] for unit in plan["unit_plans"]] == list(range(20000, 22646))
    assert len({unit["run_dir"] for unit in plan["unit_plans"]}) == 2646
    for unit, unit_plan in zip(sweep["units"][:3], plan["unit_plans"][:3], strict=True):
        steps = {step["id"]: step for step in unit_plan["steps"]}
        assert [step["id"] for step in unit_plan["steps"]] == list(UNIT_STEPS)
        assert steps["wizard_prepare"]["command"] == unit["wizard_prepare"]
        assert steps["driver_start"]["command"] == unit["driver"]
        assert steps["compose_up"]["command"] == unit["compose"]
        port = unit["generated_config_expectations"]["driver_port"]
        assert steps["driver_port_free"]["port"] == steps["driver_ready"]["port"] == port
        assert steps["driver_port_released"]["port"] == port
        assert unit_plan["run_dir"] == (f"/synthetic/runs/{unit['policy_arm']}/"
                                        f"{unit['job']['job_id']}")


def _sweep_inputs_and_request_hash(copied):
    copied["inputs"]["launch"]["driver_port_base"] = 20001
    copied["request_sha256"] = identity_hash(copied["inputs"])


def _sweep_wizard_edit(copied):
    unit = copied["units"][0]
    unit["wizard_prepare"]["argv"] = [
        "runtime.enable_autoresume=true" if item == "runtime.enable_autoresume=false" else item
        for item in unit["wizard_prepare"]["argv"]]
    assert "runtime.enable_autoresume=true" in unit["wizard_prepare"]["argv"]
    _recompute_commands(unit)


@pytest.mark.parametrize("change, expected", [
    # Inputs edited, request hash left alone.
    (lambda p: p["inputs"]["launch"].update(driver_port_base=20001), {"request_hash_mismatch"}),
    # Inputs edited and request hash recomputed: units no longer follow from the inputs.
    (_sweep_inputs_and_request_hash, {"packet_rebuild_mismatch"}),
    # Unit argv edited and commands_sha256 recomputed.
    (_sweep_wizard_edit, {"packet_rebuild_mismatch"}),
    # Fields outside commands_sha256.
    (lambda p: p["units"][0].update(policy_arm="B" if p["units"][0]["policy_arm"] == "A"
                                    else "A"), {"packet_rebuild_mismatch"}),
    (lambda p: p["units"][1]["teardown"].update(argv=["sh", "-c", "true"]),
     {"packet_rebuild_mismatch"}),
    # A sweep relabelled as a pilot is rebuilt with the pilot builder and refused.
    (lambda p: p.update(artifact_kind="s2_pilot_packet", schema_version=2),
     {"packet_rebuild_mismatch"}),
])
def test_tampered_full_sweep_packet_is_refused(sweep, change, expected):
    plan = build_execution_plan(mutate(sweep, change), RUN_CONFIG)
    assert plan["status"] == "blocked" and plan["unit_plans"] == []
    assert codes(plan) == expected


@pytest.mark.parametrize("change", [
    lambda p: p.update(artifact_kind="s2_other_packet"),
    lambda p: p.update(artifact_kind="s2_full_sweep_request"),
    lambda p: p.update(artifact_kind="S2_FULL_SWEEP_PACKET"),
    lambda p: p.update(artifact_kind=None),
    lambda p: p.pop("artifact_kind"),
    lambda p: p.update(schema_version=2),
    lambda p: p.update(schema_version=True),
])
def test_unknown_packet_kind_or_version_is_refused(sweep, change):
    plan = build_execution_plan(mutate(sweep, change), RUN_CONFIG)
    assert plan["status"] == "blocked" and plan["unit_plans"] == []
    assert codes(plan) == {"unsupported_packet"}


# ------------------------------------------------------------------ step helpers

def pinned_driver_line(port):
    """Render the start line with the pinned format strings taken from driver main.py."""
    source = DRIVER_MAIN.read_text()
    tree = ast.parse(source)
    strings = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    message = "Starting %s driver on %s (external IP: %s:%d)"
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s"
    assert message in strings and fmt in strings and 'datefmt="%H:%M:%S"' in source
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
    formatter.converter = __import__("time").gmtime
    record = logging.LogRecord("alpasim_driver", logging.INFO, __file__, 1, message,
                               ("alpamayo1_5", f"0.0.0.0:{port}", "10.1.2.3", port), None)
    record.created, record.msecs = 36005.0, 7.0
    return formatter.format(record)


def test_driver_ready_line_matches_the_pinned_formatter_output():
    line = pinned_driver_line(6789)
    assert line == ("10:00:05.007 INFO:\tStarting alpamayo1_5 driver on 0.0.0.0:6789 "
                    "(external IP: 10.1.2.3:6789)")
    assert driver_ready_line(f"loading\n{line}\n", 6789)
    assert driver_ready_line(f"{line}\r\n", 6789)
    assert driver_ready_line(line.replace("10.1.2.3", "unknown"), 6789)


@pytest.mark.parametrize("text", [
    "", "10:00:05.007 INFO:\tLoading model",
    pinned_driver_line(6790),
    pinned_driver_line(6789).replace("INFO", "DEBUG"),
    pinned_driver_line(6789).replace("0.0.0.0:6789 ", "127.0.0.1:6789 "),
    pinned_driver_line(6789).replace("alpamayo1_5", "vam"),
    "garbage " + pinned_driver_line(6789),
    pinned_driver_line(6789).split(" (external")[0],
])
def test_driver_ready_negative_controls(text):
    assert not driver_ready_line(text, 6789)


# Hydra's default job_logging formatter (hydra-core 1.3.8 hydra/conf/hydra/job_logging/
# default.yaml, sha256 edf5b4354defc88295e876ed88bb822c24f482d937cbe41783006c08f5b6a280).
# driver main.py is @hydra.main, and run_job configures this logging before the task
# function, so main.py's logging.basicConfig (no force=True) adds nothing.
HYDRA_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


def hydra_driver_line(port, name="__main__", level=logging.INFO):
    formatter = logging.Formatter(HYDRA_FORMAT)
    formatter.converter = __import__("time").gmtime
    record = logging.LogRecord(name, level, __file__, 1,
                               "Starting %s driver on %s (external IP: %s:%d)",
                               ("alpamayo1_5", f"0.0.0.0:{port}", "10.0.0.2", port), None)
    record.created = 1_794_197_610.157  # 2026-11-09 04:13:30 UTC
    record.msecs = 157.0
    return formatter.format(record)


def test_driver_ready_line_matches_hydra_job_logging():
    """B1-F2: the line the driver actually prints under Hydra's default job logging."""
    assert "@hydra.main(" in DRIVER_MAIN.read_text()
    line = hydra_driver_line(6789)
    assert line == ("[2026-11-09 04:13:30,157][__main__][INFO] - Starting alpamayo1_5 driver "
                    "on 0.0.0.0:6789 (external IP: 10.0.0.2:6789)")
    assert driver_ready_line(f"[2026-11-09 04:13:01,000][__main__][INFO] - loading\n{line}\n",
                             6789)
    assert driver_ready_line(hydra_driver_line(6789, name="alpasim_driver.main"), 6789)


@pytest.mark.parametrize("text", [
    hydra_driver_line(6790),
    hydra_driver_line(6789, level=logging.WARNING),
    "garbage " + hydra_driver_line(6789),
    hydra_driver_line(6789).replace("0.0.0.0:6789 ", "127.0.0.1:6789 "),
    hydra_driver_line(6789).replace("][INFO] - ", "][INFO] ", 1),
    hydra_driver_line(6789).replace("2026-11-09 ", "", 1),
    hydra_driver_line(6789) + " trailing",
])
def test_driver_ready_hydra_negative_controls(text):
    assert not driver_ready_line(text, 6789)


def test_worst_case_unit_seconds_known_answer():
    """B1-F7: hand-derived upper bound of one unit under RUN_CONFIG, kill grace 10 s.

    eight runner.run calls (wizard 600 + compose config 60 + compose up 3600 + four
    captures 4 x 120 + compose down 60 = 4800) each escalating SIGINT/TERM/KILL with a
    10 s wait (8 x 30 = 240) + the Compose output pump join (10) + readiness 900 plus one
    poll overshoot 5 + driver stop 60 plus TERM/KILL waits 20 + port release 30 plus one
    poll 5 = 6070.
    """
    assert worst_case_unit_seconds(RUN_CONFIG, kill_grace_s=10) == 6070
    assert worst_case_unit_seconds(RUN_CONFIG, kill_grace_s=0) == 5800
    with pytest.raises(ValueError):
        worst_case_unit_seconds({**RUN_CONFIG, "capture_timeout_s": -1}, kill_grace_s=10)
    with pytest.raises(ValueError):
        worst_case_unit_seconds(RUN_CONFIG, kill_grace_s=-1)


def test_container_ids_and_inspect_expansion(packet):
    ids = container_ids("0123456789ab\n" + "f" * 64 + "\n")
    assert ids == ["0123456789ab", "f" * 64]
    capture = packet["units"][0]["required_captures_before_teardown"][-1]
    assert inspect_argv(capture, ids) == ["docker", "inspect", "0123456789ab", "f" * 64]
    for bad in ("", "XYZ\n", "0123456789ab\n0123456789ab\n", "abc\n", "0123456789ab; rm -rf"):
        with pytest.raises(ValueError):
            container_ids(bad)
    with pytest.raises(ValueError):
        inspect_argv({"argv_template": ["docker", "inspect"], "placeholder": "{x}"}, ids)


# ------------------------------------------------------------------ recorder events

def line(message, level="INFO", worker=True):
    prefix = "[W0] " if worker else ""
    return f"runtime-0-1  | 10:00:00.123 {prefix}{level}:\t{message}"


def evidence(lines, files, rows, exit_code=0, steps=3):
    return {"scope": "synthetic", "scene_id": SCENE, "requested_steps": steps,
            "logs": [{"source": "compose_up_stdout", "text": "\n".join(lines)}],
            "rollout_files": files, "run_metadata": f"run_uuid: {RUN_UUID}\n",
            "summary": None if rows is None else {"schema_version": 1, "rollouts": rows},
            "compose_exit_code": exit_code}


def scored(uuid):
    return {"run_uuid": RUN_UUID, "clipgt_id": SCENE, "rollout_id": uuid, "status": "pass",
            "metrics": {}}


def failure(rollout):
    return {"run_uuid": None, "clipgt_id": SCENE, "rollout_id": rollout, "status": "fail",
            "metrics": {"run_name": "r", "run_uuid": None, "clipgt_id": SCENE,
                        "rollout_id": rollout, "error": "boom"}}


def start(uuid, steps=3):
    return line(f"Session STARTING: uuid={uuid} scene={SCENE} steps={steps}")


def done(uuid):
    return line(f"Session COMPLETED: uuid={uuid} scene={SCENE} simulated 1.00 sim seconds")


def fail(job, uuid, retryable, scene=SCENE):
    return line(f"Rollout FAILED: job={job} scene={scene} uuid={uuid} retryable={retryable} "
                "error=boom", "WARNING")


def complete(uuid):
    return [f"rollouts/{SCENE}/{uuid}/{name}" for name in ("_complete", "metrics.parquet")]


def recorded(report, interruption=None):
    """started (attempt 1) + terminal events, then the W2 consumer over them."""
    events = [started_event(JOB, 1, 10, ["packet_unit:x"])]
    events += unit_terminal_events(JOB, report, 20, "unit_attempt_evidence:abc", log_count=1,
                                   interruption=interruption)
    job = {"job_id": JOB, "phase": "harmonizer_pilot", "arm": "A", "policy_id": "p",
           "scene_id": SCENE, "road_id": "r", "stratum": "s", "replicate_id": 0,
           "protocol_hash": "a" * 64, "requested_steps": 3}
    ledger = reconcile_attempts([job], events)
    assert "invalid_event" not in codes(ledger) and "invalid_transition" not in codes(ledger)
    return [(e["attempt_id"][-9:], e["type"], e.get("reason"),
             e["upstream_identity"] is not None) for e in events], ledger


def test_complete_attempt_is_process_completed_and_admitted():
    report = reconcile_unit_attempts(evidence([start(UUID_A), done(UUID_A)],
                                              complete(UUID_A), [scored(UUID_A)]))
    events, ledger = recorded(report)
    assert events == [("attempt-1", "started", None, False),
                      ("attempt-1", "completed", "process_completed", True)]
    assert ledger["admitted"]
    assert ledger["attempts"][0]["upstream_identity"] == {
        "run_uuid": RUN_UUID, "clipgt_id": SCENE, "rollout_id": UUID_A}


def test_internal_retry_yields_two_attempts_ordered_by_log_line():
    retry = line(f"Retrying rollout after retryable infrastructure failure: scene={SCENE} "
                 f"failed_job={'a' * 32} retry_job={'b' * 32} attempt=1/1 error=boom",
                 "WARNING", worker=False)
    # UUID_B sorts before UUID_A as text, so log order (not UUID order) must win.
    lines = [start(UUID_B), fail("a" * 32, UUID_B, True), retry, start(UUID_A), done(UUID_A)]
    report = reconcile_unit_attempts(evidence(
        lines, [f"rollouts/{SCENE}/{UUID_B}/rollout.asl", *complete(UUID_A)],
        [scored(UUID_A)]))
    events, ledger = recorded(report)
    assert events == [("attempt-1", "started", None, False),
                      ("attempt-1", "failed", "upstream_rollout_failed:retryable=True", True),
                      ("attempt-2", "started", None, False),
                      ("attempt-2", "completed", "process_completed", True)]
    assert codes(ledger) == {"retry_selection_unregistered"}
    assert ledger["attempts"][1]["retry_reason"] == "runtime_internal_retry"
    assert ledger["attempts"][0]["upstream_identity"]["rollout_id"] == UUID_B


def test_pre_uuid_crash_is_an_attempt_without_identity():
    report = reconcile_unit_attempts(evidence([fail("c" * 32, "N/A", False, "N/A")], [],
                                              [failure("failed-0")]))
    events, ledger = recorded(report)
    assert events[-1] == ("attempt-1", "failed",
                          "upstream_failure_before_rollout_uuid:retryable=False", False)
    assert codes(ledger) == {"job_not_completed"}


def test_failed_rollout_with_exit_zero_is_failed():
    report = reconcile_unit_attempts(evidence(
        [start(UUID_A), fail("d" * 32, UUID_A, False)],
        [f"rollouts/{SCENE}/{UUID_A}/rollout.asl"], [failure(UUID_A)], exit_code=0))
    assert report["compose_exit_code"] == {"value": 0, "used_for_success": False}
    events, _ = recorded(report)
    assert events[-1] == ("attempt-1", "failed", "upstream_rollout_failed:retryable=False", True)


def test_interrupted_incomplete_attempt_is_cancelled_with_the_recorder_reason():
    report = reconcile_unit_attempts(evidence([start(UUID_A)],
                                              [f"rollouts/{SCENE}/{UUID_A}/rollout.asl"], None,
                                              exit_code=None))
    events, _ = recorded(report, interruption="compose_up_timeout")
    assert events[-1] == ("attempt-1", "cancelled", "compose_up_timeout", True)
    events, _ = recorded(report)
    assert events[-1] == ("attempt-1", "failed", "upstream_attempt_incomplete", True)


def test_complete_attempt_with_conflicting_evidence_is_not_completed():
    report = reconcile_unit_attempts(evidence(
        [start(UUID_A), done(UUID_A)], complete(UUID_A),
        [{**scored(UUID_A), "run_uuid": "other"}]))
    assert "run_uuid_mismatch" in codes(report)
    events, _ = recorded(report)
    assert events[-1] == ("attempt-1", "failed", "completion_not_established:run_uuid_mismatch",
                          True)


def test_no_evidence_and_summary_only_failure():
    report = reconcile_unit_attempts(evidence([line("nothing")], [], None))
    events, _ = recorded(report)
    assert events[-1] == ("attempt-1", "failed", "no_upstream_attempt_evidence", False)
    report = reconcile_unit_attempts(evidence([line("nothing")], [], [failure("failed-0")]))
    events, _ = recorded(report)
    assert events[-1] == ("attempt-1", "failed", "upstream_failure_row_without_log", False)


def test_event_shapes_and_cancellation():
    event = cancelled_event(JOB, 1, 5, "driver_not_ready", ["r", "r"])
    assert event == {"schema_version": 2, "event_id": f"{JOB}/attempt-1:cancelled",
                     "job_id": JOB, "attempt_id": f"{JOB}/attempt-1", "type": "cancelled",
                     "timestamp_us": 5, "evidence_refs": ["r"], "reason": "driver_not_ready",
                     "upstream_identity": None}
    with pytest.raises(ValueError):
        cancelled_event(JOB, 1, 5, " ", ["r"])
    with pytest.raises(ValueError):
        unit_terminal_events(JOB, {"artifact_kind": "other"}, 1, "r", log_count=1)
    assert re.fullmatch(r"j{64}/attempt-1:started",
                        started_event(JOB, 1, 0, ["r"])["event_id"])


# ------------------------------------------------------------------ B1 residuals (B1R)

def test_upstream_attempt_evidence_known_answers():
    silent = reconcile_unit_attempts(evidence(["Attaching to runtime-0-1"], [], None,
                                              exit_code=None))
    assert has_upstream_attempt_evidence(silent) is False
    started_only = reconcile_unit_attempts(evidence(
        [start(UUID_A)], [f"rollouts/{SCENE}/{UUID_A}/rollout.asl"], None, exit_code=None))
    assert has_upstream_attempt_evidence(started_only) is True
    pre_uuid = reconcile_unit_attempts(evidence([fail("c" * 32, "N/A", False, "N/A")], [],
                                                [failure("failed-0")]))
    assert has_upstream_attempt_evidence(pre_uuid) is True
    row_only = reconcile_unit_attempts(evidence([line("nothing")], [], [failure("failed-0")]))
    assert has_upstream_attempt_evidence(row_only) is True
    with pytest.raises(ValueError):
        has_upstream_attempt_evidence({"artifact_kind": "other"})


def _terminal(kind, reason):
    return {"type": kind, "reason": reason}


@pytest.mark.parametrize("event, established", [
    (_terminal("completed", "process_completed"), True),
    (_terminal("failed", "upstream_rollout_failed:retryable=False"), True),
    (_terminal("failed", "upstream_failure_before_rollout_uuid:retryable=True"), True),
    (_terminal("completed", "other"), False),
    (_terminal("failed", "completion_not_established:run_uuid_mismatch"), False),
    (_terminal("failed", "upstream_attempt_incomplete"), False),
    (_terminal("failed", "upstream_failure_row_without_log"), False),
    (_terminal("failed", "no_upstream_attempt_evidence"), False),
    (_terminal("cancelled", "compose_up_timeout"), False),
    (_terminal("cancelled", "upstream_rollout_failed:retryable=False"), False),
    (_terminal("started", None), False),
    (None, False),
])
def test_terminal_established_known_answers(event, established):
    assert terminal_established(event) is established


# ------------------------------------------------------------ diagnostic qualification units

@pytest.fixture(scope="module")
def diagnostic_packet(packet):
    request = packet_request()
    request["diagnostic_units"] = [
        diagnostic_unit(packet["units"][0]["job"]["job_id"]),
        diagnostic_unit(packet["units"][1]["job"]["job_id"], diagnostic_id="harmonizer-on")]
    built = build_pilot_packet(request)
    assert built["prepared"], built["issues"]
    return built


def test_packet_without_diagnostics_plans_no_diagnostic_keys(packet):
    plan = build_execution_plan(packet, RUN_CONFIG)
    assert "diagnostic_unit_plans" not in plan and "diagnostic_schedule" not in plan
    assert identity_hash(_without_source_hashes(plan)) == PILOT_PLAN_SHA256


def test_diagnostic_units_are_planned_separately_from_the_pilot_jobs(packet, diagnostic_packet):
    plain = build_execution_plan(packet, RUN_CONFIG)
    plan = build_execution_plan(diagnostic_packet, RUN_CONFIG)
    assert plan["status"] == "passed", plan["issues"]
    # The ten pilot unit plans are exactly those of the packet without diagnostics.
    assert identity_hash(plan["unit_plans"]) == identity_hash(plain["unit_plans"])
    assert [item["job_id"] for item in plan["diagnostic_unit_plans"]] == [
        "diagnostic-camera-check", "diagnostic-harmonizer-on"]
    first = plan["diagnostic_unit_plans"][0]
    steps = {step["id"]: step for step in first["steps"]}
    assert [step["id"] for step in first["steps"]] == list(UNIT_STEPS)
    assert steps["wizard_prepare"]["stdout_journal_path"] == (
        "diagnostics/000/wizard-prepare.stdout.txt")
    assert steps["ledger_started"]["attempt_id"] == "diagnostic-camera-check/attempt-1"
    assert first["diagnostic"]["role"] == "camera_config_qualification"
    assert first["diagnostic"]["budget_line"] == {"line_id": "pilot_diagnostics",
                                                  "cap_usd": "5"}
    assert first["run_dir"] not in {item["run_dir"] for item in plan["unit_plans"]}
    schedule = plan["diagnostic_schedule"]
    assert [job["job_id"] for job in schedule] == [
        "diagnostic-camera-check", "diagnostic-harmonizer-on"]
    assert not {job["job_id"] for job in schedule} & {
        job["job_id"] for job in diagnostic_packet["schedule"]}
    assert {job["phase"] for job in schedule} == {"diagnostic_qualification"}  # review G2-04
    # The diagnostic schedule is a valid ledger schedule of its own.
    report = reconcile_attempts(schedule, [])
    assert {item["code"] for item in report["issues"]} == {"missing_job"}


def _diag_scheduled_job(copied):
    unit = copied["diagnostic_units"][0]
    unit["job"] = dict(copied["schedule"][0])


def _diag_counts(copied):
    copied["diagnostic_units"][0]["diagnostic"]["counts_toward_pilot_jobs"] = True


def _diag_drop_block(copied):
    copied["diagnostic_units"][0].pop("diagnostic")


def _diag_dropped(copied):
    copied.pop("diagnostic_units")


def _diag_pilot_phase(copied):
    copied["diagnostic_units"][0]["job"]["phase"] = "harmonizer_pilot"


def _diag_shared_run_dir(copied):
    copied["diagnostic_units"][0]["compose"]["cwd"] = copied["units"][0]["compose"]["cwd"]


@pytest.mark.parametrize("change, expected", [
    (_diag_scheduled_job, {"diagnostic_unit_scheduled", "invalid_diagnostic_unit",
                           "duplicate_or_missing_job", "packet_rebuild_mismatch"}),
    (_diag_counts, {"invalid_diagnostic_unit", "packet_rebuild_mismatch"}),
    (_diag_drop_block, {"invalid_diagnostic_unit", "packet_rebuild_mismatch"}),
    # Review G2-04: a diagnostic job relabelled as a pilot-phase job.
    (_diag_pilot_phase, {"invalid_diagnostic_unit", "packet_rebuild_mismatch"}),
    # The inputs still declare diagnostics, so dropping them from the packet is an edit.
    (_diag_dropped, {"packet_rebuild_mismatch"}),
    (lambda p: p.update(diagnostic_units=[]), {"invalid_diagnostic_unit",
                                               "packet_rebuild_mismatch"}),
    (_diag_shared_run_dir, {"duplicate_run_dir", "commands_hash_mismatch",
                            "unexpected_capture_dir", "unexpected_capture_plan",
                            "unexpected_teardown", "unexpected_compose_config",
                            "unexpected_driver_command", "packet_rebuild_mismatch"}),
])
def test_tampered_diagnostic_units_are_refused(diagnostic_packet, change, expected):
    plan = build_execution_plan(mutate(diagnostic_packet, change), RUN_CONFIG)
    assert plan["status"] == "blocked" and plan["unit_plans"] == []
    assert "diagnostic_unit_plans" not in plan
    assert codes(plan) == expected


def test_full_sweep_packet_cannot_carry_diagnostic_units(sweep, diagnostic_packet):
    tampered = mutate(sweep, lambda p: p.update(
        diagnostic_units=copy.deepcopy(diagnostic_packet["diagnostic_units"])))
    plan = build_execution_plan(tampered, RUN_CONFIG)
    assert plan["status"] == "blocked"
    assert "invalid_diagnostic_unit" in codes(plan)
