"""Fake-host end-to-end known answers for the S2 guest unit runner (build-spec B1).

No test runs uv, docker, nvidia-smi or a driver: every process goes through ``FakeHost``,
which emulates the pinned wizard outputs, the driver start line, Compose's effective
config/override merge and the pinned runtime log lines (formats as in
tests/test_s2_capture.py, which checks them against the vendored AlpaSim source).
``SubprocessRunner`` is exercised only against the local Python interpreter.
"""

import io
import itertools
import json
import math
import os
import shlex
import sys
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest
import yaml
from s2_packet_fixture import diagnostic_unit, full_sweep_request, packet_request

from gate.s2_diagnostics import _check_ledger
from gate.s2_identity import identity_hash
from gate.s2_launch import filter_environment
from gate.s2_packet import build_full_sweep_packet, build_pilot_packet
from gate.s2_telemetry_sources import useful_attempt_rows
from scripts.s2_run_units import (
    BudgetMeter,
    SubprocessRunner,
    main,
    readiness_issues,
    run_units,
    validate_authority,
    validate_budget,
)

CANARY = "hf_CANARYcanaryCANARYcanary0123456789"
HOST_ENV = {"HOME": "/home/guest", "PATH": "/usr/bin:/bin", "HF_TOKEN": CANARY,
            "LANG": "en_US.UTF-8"}
RUN_CONFIG = {"schema_version": 1, "wizard_prepare_timeout_s": 600,
              "compose_config_timeout_s": 60, "driver_ready_timeout_s": 900,
              "driver_poll_interval_s": 5, "compose_up_timeout_s": 3600,
              "capture_timeout_s": 120, "teardown_timeout_s": 60, "port_release_timeout_s": 30}
JOB_HEX = "0123456789abcdef0123456789abcde"


def uuid(index, n):
    return f"{index:08x}-0000-4000-8000-{n:012x}"


def run_uuid(index):
    return f"{index:08x}-1111-4111-8111-111111111111"


def stamp(seconds):
    return f"10:00:{seconds:02d}.123"


def worker(message, seconds=0, level="INFO"):
    return f"{stamp(seconds)} [W0] {level}:\t{message}"


def scheduler(message, seconds=0, level="WARNING"):
    return f"{stamp(seconds)} {level}:\t{message}"


def scored_row(rollout, scene, run):
    return {"run_uuid": run, "run_name": "e2e_challenge_dev", "clipgt_id": scene,
            "rollout_id": rollout, "status": "pass", "passed": True, "score": 0.8,
            "failure_reason": None,
            "metrics": {"run_uuid": run, "clipgt_id": scene, "rollout_id": rollout},
            "score_metrics": {}}


def failure_row(rollout, scene):
    metrics = {"run_name": "e2e_challenge_dev", "run_uuid": None, "clipgt_id": scene,
               "rollout_id": rollout, "error": "boom"}
    return {"run_uuid": None, "run_name": "e2e_challenge_dev", "clipgt_id": scene,
            "rollout_id": rollout, "status": "fail", "passed": False, "failure_reason": "boom",
            "metrics": metrics, "score": 0.0, "score_metrics": {}}


def started(u, scene, steps=3, s=1):
    return worker(f"Session STARTING: uuid={u} scene={scene} steps={steps}", s)


def completed(u, scene, s=2):
    return worker(f"Session COMPLETED: uuid={u} scene={scene} simulated 0.30 sim seconds in "
                  "1.00 wall clock seconds for 0.30x real time (total rollout 2.00s incl. "
                  "setup/warmup)", s)


def failed(job, scene, u, retryable, error, s=2):
    return worker(f"Rollout FAILED: job={job} scene={scene} uuid={u} retryable={retryable} "
                  f"error={error}\nTraceback (most recent call last):\n  raise Boom", s,
                  "WARNING")


def scenario_output(kind, index, scene):
    """(runtime lines, {rollout uuid: files}, summary rows) per scenario."""
    a, b, run = uuid(index, 1), uuid(index, 2), run_uuid(index)
    complete = ["_complete", "metrics.parquet", "rollout.asl"]
    if kind == "complete":
        return ([started(a, scene), completed(a, scene)], {a: complete},
                [scored_row(a, scene, run)])
    if kind == "failed_exit0":
        return ([started(a, scene), failed(JOB_HEX + "a", scene, a, False,
                                           "Camera camera_rear_left_70fov not in desired")],
                {a: ["rollout.asl"]}, [failure_row(a, scene)])
    if kind == "pre_uuid_crash":
        return ([failed(JOB_HEX + "b", "N/A", "N/A", False, "scene load failed")], {},
                [failure_row("failed-0", scene)])
    if kind == "internal_retry":
        retry = scheduler("Retrying rollout after retryable infrastructure failure: "
                          f"scene={scene} failed_job={JOB_HEX}c retry_job={JOB_HEX}d "
                          "attempt=1/1 error=renderer died", 3)
        return ([started(a, scene), failed(JOB_HEX + "c", scene, a, True, "renderer died"),
                 retry, started(b, scene, s=4), completed(b, scene, s=5)],
                {a: ["rollout.asl"], b: complete}, [scored_row(b, scene, run)])
    if kind == "timeout":
        return ([started(a, scene)], {a: ["rollout.asl"]}, None)
    raise AssertionError(kind)


def value(argv, prefix):
    token = next(item for item in argv if item.startswith(prefix))
    text = token[len(prefix):]
    return json.loads(text) if text.startswith(("[", '"')) else text


class Handle:
    def __init__(self, port, ready):
        self.port, self.ready, self.running, self.stopped = port, ready, True, False


class Clock:
    def __init__(self):
        self.now = 1_000_000

    def monotonic_us(self):
        self.now += 1
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1_000_000)

    def epoch(self):
        return 1_800_000_000 + self.now / 1_000_000


class FakeHost:
    """Emulates wizard, driver, Compose and Docker CLI effects in local directories."""

    fixture_only = True  # B1-F14: declares itself a fake; anything else is a real runner

    def __init__(self, journal, scenarios=None, *, cameras=None, never_ready=False,
                 ignore_override=False, wizard_rc=0, clock=None, rollout_s=0):
        self.journal = journal
        self.clock, self.rollout_s = clock, rollout_s  # Compose up advances the fake clock
        self.scenarios = scenarios or {}
        self.cameras = cameras
        self.never_ready = never_ready
        self.ignore_override = ignore_override
        self.wizard_rc = wizard_rc
        self.calls = []
        self.driver = None
        self.unit = -1
        self.ledger_at_wizard = []

    # helpers ---------------------------------------------------------------
    def _write(self, path, text):
        with open(path, "x") as stream:
            stream.write(text)

    def port_open(self, port):
        driver = self.driver
        return bool(driver and driver.running and driver.ready and driver.port == port)

    # runner protocol -------------------------------------------------------
    def run(self, argv, *, cwd, env, stdout_path, timeout_s, merge_stderr=False):
        self.calls.append(("run", list(argv), cwd, dict(env)))
        if argv[:4] == ["uv", "run", "--no-sync", "alpasim_wizard"]:
            return self._wizard(argv, stdout_path)
        if argv[:3] == ["docker", "compose", "config"]:
            return self._config(cwd, stdout_path)
        if argv[:3] == ["docker", "compose", "up"]:
            return self._up(cwd, stdout_path)
        if argv[:3] == ["docker", "compose", "logs"]:
            text = "\n".join(f"runtime-0  | 2026-11-08T10:00:00.000000001Z {first}"
                             for first in (line.split("\n")[0] for line in self.lines))
            self._write(stdout_path, text + "\n")
            return {"returncode": 0, "timed_out": False}
        if argv[:5] == ["docker", "compose", "ps", "-a", "--format"]:
            self._write(stdout_path, "[]\n")
            return {"returncode": 0, "timed_out": False}
        if argv[:5] == ["docker", "compose", "ps", "-a", "-q"]:
            self._write(stdout_path, "\n".join(f"{i:012x}" for i in range(1, 5)) + "\n")
            return {"returncode": 0, "timed_out": False}
        if argv[:2] == ["docker", "inspect"]:
            self._write(stdout_path, json.dumps([{"Id": item} for item in argv[2:]]))
            return {"returncode": 0, "timed_out": False}
        if argv[:3] == ["docker", "compose", "down"]:
            self._write(stdout_path, "")
            return {"returncode": 0, "timed_out": False}
        raise AssertionError(f"unexpected argv {argv}")

    def _wizard(self, argv, stdout_path):
        self.unit += 1
        name = "diagnostic-ledger-events.jsonl" if "/diagnostic-" in value(
            argv, "wizard.log_dir=") else "ledger-events.jsonl"
        ledger = (self.journal / name).read_text().splitlines()
        self.ledger_at_wizard.append(json.loads(ledger[-1]))
        self._write(stdout_path, "wizard ok\n")
        if self.wizard_rc:
            return {"returncode": self.wizard_rc, "timed_out": False}
        run_dir = value(argv, "wizard.log_dir=")
        scene = value(argv, "scenes.scene_ids=")[0]
        steps = int(value(argv, "runtime.simulation_config.n_sim_steps="))
        port = int(value(argv, "driver.port="))
        override = next((item for item in argv if item.startswith("cameras=")), None)
        preset = self.cameras or (4 if override else 6)
        names = ["camera_front_wide_120fov", "camera_front_tele_30fov",
                 "camera_cross_left_120fov", "camera_cross_right_120fov",
                 "camera_rear_left_70fov", "camera_rear_right_70fov"][:preset]
        os.makedirs(run_dir)
        self._write(f"{run_dir}/generated-user-config-0.yaml", yaml.safe_dump({
            "scenes": [{"scene_id": scene}],
            "simulation_config": {"cameras": [{"logical_id": n} for n in names],
                                  "n_sim_steps": steps, "n_rollouts": 1}}))
        self._write(f"{run_dir}/driver-config.yaml", yaml.safe_dump({
            "port": port,
            "model": {"checkpoint_path": value(argv, "driver.model.checkpoint_path=")},
            "inference": {"use_cameras": names[:4][::-1], "max_batch_size": 1}}))
        renderer = value(argv, "services.renderer.command=")
        ego = value(argv, "defines.filesystem=") + "/nre-artifacts/ego-hoods"
        base = value(argv, "defines.base_image=")
        services = {name: {"image": base, "network_mode": "host"}
                    for name in ("controller-0", "physics-0", "runtime-0")}
        services["renderer-0"] = {
            "image": value(argv, "services.renderer.image="), "network_mode": "host",
            "command": ["-c", " ".join(shlex.quote(token) for token in renderer)],
            "volumes": [f"{ego}:/mnt/ego-hoods"]}
        self._write(f"{run_dir}/docker-compose.yaml", yaml.safe_dump({"services": services}))
        self._write(f"{run_dir}/run_metadata.yaml",
                    f"run_name: e2e_challenge_dev\nrun_uuid: {run_uuid(self.unit)}\n")
        self.scene, self.run_dir = scene, run_dir
        return {"returncode": 0, "timed_out": False}

    def _config(self, cwd, stdout_path):
        config = yaml.safe_load(Path(f"{cwd}/docker-compose.yaml").read_text())
        override = f"{cwd}/docker-compose.override.yaml"
        if os.path.exists(override) and not self.ignore_override:
            for name, fields in yaml.safe_load(Path(override).read_text())["services"].items():
                config["services"][name].update(fields)
        for service in config["services"].values():
            service["pull_policy"] = "never"
        self._write(stdout_path, json.dumps(config))
        return {"returncode": 0, "timed_out": False}

    def _up(self, cwd, stdout_path):
        if self.clock is not None:
            self.clock.now += int(self.rollout_s * 1_000_000)
        kind = self.scenarios.get(self.unit, "complete")
        lines, folders, rows = scenario_output(kind, self.unit, self.scene)
        self.lines = lines
        text = "\n".join(f"runtime-0-1  | {part}" for line in lines for part in line.split("\n"))
        self._write(stdout_path, "Attaching to runtime-0-1\n" + text + "\n")
        for rollout, files in folders.items():
            os.makedirs(f"{cwd}/rollouts/{self.scene}/{rollout}")
            for name in files:
                self._write(f"{cwd}/rollouts/{self.scene}/{rollout}/{name}", name)
        if rows is not None:
            os.makedirs(f"{cwd}/aggregate/videos/all")
            self._write(f"{cwd}/aggregate/results-summary.json",
                        json.dumps({"schema_version": 1, "rollouts": rows}))
            if folders:
                first = min(folders)
                os.symlink(f"../../../rollouts/{self.scene}/{first}/rollout.asl",
                           f"{cwd}/aggregate/videos/all/link.mp4")
        if kind == "timeout":
            return {"returncode": None, "timed_out": True}
        return {"returncode": 0, "timed_out": False}

    def start(self, argv, *, cwd, env, stdout_path):
        self.calls.append(("start", list(argv), cwd, dict(env)))
        # The driver listens on the port of the driver-config.yaml it is pointed at (the
        # pilot's is DRIVER_PORT for every unit; a full-sweep unit has its own).
        config_dir = next(item for item in argv if item.startswith("--config-path="))
        config = yaml.safe_load(
            Path(config_dir.split("=", 1)[1], "driver-config.yaml").read_text())
        port = config["port"]
        # B1-F2: driver main.py is @hydra.main, so its lines carry Hydra's default
        # job_logging prefix (tests/test_s2_executor.py derives it), not basicConfig's.
        lines = "[2026-11-08 10:00:00,000][__main__][INFO] - Loading model\n"
        if not self.never_ready:
            lines += (f"[2026-11-08 10:00:05,000][__main__][INFO] - Starting alpamayo1_5 "
                      f"driver on 0.0.0.0:{port} (external IP: 10.0.0.2:{port})\n")
        self._write(stdout_path, lines)
        self.driver = Handle(port, not self.never_ready)
        return self.driver

    def poll(self, handle):
        return None if handle.running else -2

    def stop(self, handle, *, timeout_s):
        self.calls.append(("stop", None, None, None))
        handle.running, handle.stopped = False, True
        return {"returncode": -2, "escalated": False}


DRIVER_PORT = 6789  # tests/s2_launch_fixture.py launch_request()["driver_port"]


WORST_CASE_UNIT_S = 6070  # tests/test_s2_executor.py hand-derived bound for RUN_CONFIG


QUOTE = {"artifact_kind": "s2_cost_quote", "schema_version": 1, "scope": "synthetic",
         "currency": "USD", "provider": "vast", "rate_basis": "per_gpu",
         "compute_usd_per_hour": "1.20", "storage_usd_per_hour": "0",
         "transfer_usd_per_gb": {"download": "0", "upload": "0"},
         "billing_granularity_seconds": 1,
         "fees_and_tax": {"bound_usd": "0", "basis": "synthetic fixture: no fees"}}
ACCOUNTED_UNTIL = 1_800_000_000  # Clock.epoch() starts just after this
CLOCK_UNCERTAINTY_US = 250  # an invented operator bound for these fakes


def budget_ok(packet=None, *, prior=(), **changes):
    """Synthetic starting snapshot: caps 100/1200/2500 (D-018's recorded values, used
    here only as fixture numbers), $1.20/GPU-h, no prior usage unless ``prior``. The quote
    takes the packet's scope (review G2-03) and every declared diagnostic budget line starts
    with no prior spend (review G2-02).

    Exposure per unit = max(2.03, 1.20 x (6070 + 60) s / 3600) = 2.0433... USD.
    """
    sweep = type(packet) is dict and packet.get("artifact_kind") == "s2_full_sweep_packet"
    scope = packet["scope"] if type(packet) is dict else "synthetic"
    lines = packet.get("diagnostics", {}).get("budget_lines", {}) if type(
        packet) is dict else {}
    value = {"artifact_kind": "s2_unit_budget", "schema_version": 2, "scope": scope,
             "phase": "full_sweep" if sweep else "pilot", "prior_events": list(prior),
             "quote": {**QUOTE, "scope": scope},
             "prior_line_spend_usd": {line_id: "0" for line_id in lines},
             "caps_usd": {"pilot": "100", "stage": "1200", "project": "2500"},
             "storage_retention_until_epoch": None,
             "limits": {"task_allowance_usd": None, "reserves_usd": {"cleanup": "0"},
                        "deadline_epoch": ACCOUNTED_UNTIL + 10 * 86_400},
             "instance_resource_id": "58628722", "gpu_count": 1,
             "accounted_until_epoch": ACCOUNTED_UNTIL, "cleanup_seconds": 60,
             "next_unit_upper_bound_usd": "2.03"}
    value.update(changes)
    return value


def prior_usage(amount, tier, event_id="prior-usage-1"):
    return {"event_id": event_id, "kind": "prior_usage", "amount_usd": amount,
            "decision_ref": "SYNTHETIC-prior", "counts_toward": tier}


def decisions_md(*entries):
    """Invented DECISIONS.md bytes (gate.s2_authority grammar); not a real decision log."""
    lines = ["# Decision log (synthetic fixture for scripts/s2_run_units tests)", ""]
    for decision_id, grants, extra in entries:
        lines.append(f"- **{decision_id} · 2026-11-09 · Synthetic approval.** Invented.")
        lines += [f"  - authorizes: {kind} sha256:{sha}" for kind, sha in grants]
        lines += [f"  - {line}" for line in extra]
        lines.append("")
    return "\n".join(lines).encode("utf-8")


RELEASE = {"artifact_kind": "s2_full_release", "note": "SYNTHETIC release record"}
RELEASE_ID = "D-950"


def run_kw(packet, **changes):
    """Required run inputs; a full sweep also gets a recorded synthetic release."""
    value = {"unit_selection": "all", "clock_uncertainty_us": CLOCK_UNCERTAINTY_US}
    if type(packet) is dict and packet.get("artifact_kind") == "s2_full_sweep_packet":
        value.update(decisions_md=decisions_md(
            (RELEASE_ID, [("s2_full_release", identity_hash(RELEASE))], [])),
            release_record=RELEASE, release_decision_id=RELEASE_ID)
    value.update(changes)
    return value


@pytest.fixture
def base(tmp_path_factory):
    return tmp_path_factory.mktemp("units")


def make_packet(base, *, camera_override="4cam_1080", shm=None, scope="synthetic",
                diagnostics=None):
    request = packet_request()
    launch = request["launch_request"]
    launch["renderer_camera_override"] = camera_override
    if scope != "synthetic":
        request["scope"] = launch["scope"] = request["ego_hood_staging"]["scope"] = scope
    if diagnostics is not None:
        jobs = request["execution_order"]["job_ids"]
        request["diagnostic_units"] = [diagnostic_unit(jobs[source], **changes)
                                       for source, changes in diagnostics]
    for arm in launch["arms"]:
        mode = "on" if arm["harmonizer"] else "off"
        arm["run_dir"] = f"{base}/runs/{mode}"
        arm["driver_log_dir"] = f"{base}/driver-logs/{mode}"
    if shm is not None:
        request["shm_size"] = shm
    packet = build_pilot_packet(request)
    assert packet["prepared"], packet["issues"]
    return packet


def authority(packet, **changes):
    return {"artifact_kind": "s2_unit_run_authority", "schema_version": 1,
            "scope": packet["scope"], "action_record_ref": "SYNTHETIC-action-record",
            "packet_request_sha256": packet["request_sha256"],
            "job_ids": [unit["job"]["job_id"] for unit in packet["units"]], **changes}


def execute(base, packet, host=None, budget=None, *, clock=None, run=None, auth=None,
            **host_kwargs):
    journal = base / "journal"
    clock = clock or Clock()
    host = host or FakeHost(journal, clock=clock, **host_kwargs)
    summary = run_units(packet, authority=auth or authority(packet), run_config=RUN_CONFIG,
                        journal_dir=journal, runner=host, port_open_fn=host.port_open,
                        clock=clock, budget=budget if budget is not None else budget_ok(packet),
                        host_env=HOST_ENV, **(run or run_kw(packet)))
    events = [json.loads(line) for line in
              (journal / "ledger-events.jsonl").read_text().splitlines()]
    return summary, events, host


def by_job(events, job_id):
    return [(e["attempt_id"].rsplit("/", 1)[1], e["type"], e.get("reason"),
             e["upstream_identity"] is not None) for e in events if e["job_id"] == job_id]


def codes(report):
    return {item["code"] for item in report["issues"]}


# --------------------------------------------------------------------- end to end

def test_ten_unit_fake_run_binds_every_upstream_attempt(base):
    packet = make_packet(base)
    scenarios = {1: "pre_uuid_crash", 2: "internal_retry", 3: "failed_exit0"}
    summary, events, _ = execute(base, packet, scenarios=scenarios)
    jobs = [unit["job"]["job_id"] for unit in packet["units"]]
    assert summary["status"] == "passed" and summary["run_outcome"] == "completed"
    assert summary["stop_reason"] is None
    assert [r["outcome"] for r in summary["unit_records"]] == ["executed"] * 10
    assert by_job(events, jobs[0]) == [("attempt-1", "started", None, False),
                                       ("attempt-1", "completed", "process_completed", True)]
    # Pre-UUID crash: one failed attempt without upstream identity.
    assert by_job(events, jobs[1]) == [
        ("attempt-1", "started", None, False),
        ("attempt-1", "failed", "upstream_failure_before_rollout_uuid:retryable=False", False)]
    # Internal retry: two attempts, parent chain, retry reason from the runtime.
    assert by_job(events, jobs[2]) == [
        ("attempt-1", "started", None, False),
        ("attempt-1", "failed", "upstream_rollout_failed:retryable=True", True),
        ("attempt-2", "started", None, False),
        ("attempt-2", "completed", "process_completed", True)]
    retry_start = [e for e in events if e["job_id"] == jobs[2] and e["type"] == "started"][1]
    assert retry_start["parent_attempt_id"] == f"{jobs[2]}/attempt-1"
    assert retry_start["retry_reason"] == "runtime_internal_retry"
    # Failed rollout although Compose exited 0.
    assert summary["unit_records"][3]["compose_exit_code"] == 0
    assert by_job(events, jobs[3])[-1] == (
        "attempt-1", "failed", "upstream_rollout_failed:retryable=False", True)
    identity = next(e for e in events if e["job_id"] == jobs[2] and e["type"] == "completed")
    assert identity["upstream_identity"] == {"run_uuid": run_uuid(2),
                                             "clipgt_id": packet["units"][2]["job"]["scene_id"],
                                             "rollout_id": uuid(2, 2)}
    assert all(e["schema_version"] == 2 and e["evidence_refs"] for e in events)
    ledger = summary["ledger"]
    assert not ledger["admitted"]
    assert codes(ledger) == {"retry_selection_unregistered", "job_not_completed"}
    assert {i["path"] for i in ledger["issues"] if i["code"] == "job_not_completed"} == {
        f"job:{jobs[1]}", f"job:{jobs[3]}"}
    assert ledger["counts"]["attempts"] == 11
    assert ledger["counts"]["completed_attempts"] == 8
    assert ledger["counts"]["failed_attempts"] == 3
    assert summary["not_run_job_ids"] == []
    _check_ledger(ledger)  # the W2 consumer's exact key-set contract still holds
    saved = json.loads((base / "journal" / "run-summary.json").read_text())
    assert saved == summary


def test_started_is_durable_before_the_first_side_effect(base):
    packet = make_packet(base)
    _, _, host = execute(base, packet)
    jobs = [unit["job"]["job_id"] for unit in packet["units"]]
    assert [(e["job_id"], e["type"]) for e in host.ledger_at_wizard] == [
        (job, "started") for job in jobs]


def test_unit_call_order_argv_and_environment_are_the_packets(base):
    packet = make_packet(base)
    _, _, host = execute(base, packet)
    unit = packet["units"][0]
    calls = host.calls[:10]
    assert [c[0] for c in calls] == ["run", "run", "start", "run", "run", "run", "run", "run",
                                     "run", "stop"]
    assert calls[0][1] == unit["wizard_prepare"]["argv"]
    assert calls[1][1] == unit["compose_config"]["argv"]
    assert calls[2][1] == unit["driver"]["argv"]
    assert calls[3][1] == ["docker", "compose", "up", "--pull", "never", "--exit-code-from",
                           "runtime-0"]
    assert [c[1][:4] for c in calls[4:7]] == [["docker", "compose", "logs", "--no-color"],
                                              ["docker", "compose", "ps", "-a"],
                                              ["docker", "compose", "ps", "-a"]]
    assert calls[7][1] == ["docker", "inspect", *(f"{i:012x}" for i in range(1, 5))]
    assert calls[8][1] == ["docker", "compose", "down", "--timeout", "10"]
    assert calls[0][3] == filter_environment(unit["wizard_prepare"], HOST_ENV)
    assert calls[2][3] == filter_environment(unit["driver"], HOST_ENV)
    assert calls[3][3] == filter_environment(unit["compose"], HOST_ENV)
    assert all(CANARY not in json.dumps(c[3]) for c in host.calls if c[3] is not None)
    assert all("HF_TOKEN" not in (c[3] or {}) for c in host.calls)


def test_capture_artifacts_are_written_before_teardown_and_never_score(base):
    packet = make_packet(base)
    summary, _, _ = execute(base, packet)
    capture = packet["units"][0]["capture_dir"]
    names = sorted(path.name for path in Path(capture).iterdir())
    for expected in ("compose-check.json", "compose-config.json", "compose-logs.txt",
                     "compose-ps.json", "compose-up.stdout.txt", "docker-inspect.json",
                     "generated-config-check.json", "run-dir-manifest.json",
                     "unit-attempt-evidence.json", "compose-down.txt"):
        assert expected in names
    manifest = json.loads(Path(capture, "run-dir-manifest.json").read_text())
    assert manifest["status"] == "passed" and manifest["symlink_policy"] == "record"
    assert manifest["symlinks"][0]["path"] == "aggregate/videos/all/link.mp4"
    assert "run_metadata.yaml" in {item["path"] for item in manifest["files"]}
    evidence = json.loads(Path(capture, "unit-attempt-evidence.json").read_text())
    assert evidence["unit_outcome"] == "complete" and evidence["score_admitted"] is False
    steps = [step["id"] for step in summary["unit_records"][0]["steps"]]
    assert steps.index("run_dir_hash") < steps.index("compose_down")
    assert steps.index("capture:container_inspect") < steps.index("compose_down")
    assert steps.index("compose_down") < steps.index("reconcile")


# --------------------------------------------------------------------- stops

def test_camera_mismatch_blocks_before_driver_and_compose(base):
    packet = make_packet(base, camera_override=None)
    summary, events, host = execute(base, packet)
    job = packet["units"][0]["job"]["job_id"]
    assert summary["status"] == "blocked" and summary["run_outcome"] == "stopped"
    assert summary["stop_reason"]["reason"].startswith("generated_config_blocked:")
    assert "renderer_driver_camera_mismatch" in summary["stop_reason"]["reason"]
    assert [c[0] for c in host.calls] == ["run"]  # wizard only: no driver, no compose
    assert by_job(events, job) == [
        ("attempt-1", "started", None, False),
        ("attempt-1", "cancelled", summary["stop_reason"]["reason"], False)]
    assert len(summary["not_run_job_ids"]) == 9
    report = json.loads(Path(packet["units"][0]["capture_dir"],
                             "generated-config-check.json").read_text())
    assert "renderer_driver_camera_mismatch" in codes(report)


def test_ignored_camera_override_is_also_blocked(base):
    packet = make_packet(base)
    summary, _, _ = execute(base, packet, cameras=6)
    reason = summary["stop_reason"]["reason"]
    assert "renderer_cameras_unexpected" in reason and "renderer_driver_camera_mismatch" in reason


def test_driver_that_never_becomes_ready_is_stopped_and_stops_the_run(base):
    packet = make_packet(base)
    summary, events, host = execute(base, packet, never_ready=True)
    job = packet["units"][0]["job"]["job_id"]
    assert summary["stop_reason"]["reason"] == "driver_not_ready"
    assert host.driver.stopped
    assert not any(c[1] and c[1][:3] == ["docker", "compose", "up"] for c in host.calls)
    assert by_job(events, job)[-1] == ("attempt-1", "cancelled", "driver_not_ready", False)
    steps = {s["id"]: s["status"] for s in summary["unit_records"][0]["steps"]}
    assert steps["driver_ready"] == "blocked" and steps["driver_stop"] == "passed"
    assert steps["driver_port_released"] == "passed"


def segment_usd(event):
    """Independent oracle: 1.20 USD per GPU-hour, whole seconds rounded up."""
    seconds = math.ceil(Fraction(str(event["end_epoch"])) - Fraction(str(event["start_epoch"])))
    return Fraction(120, 100) * event["gpu_count"] * seconds / 3600


def cost_events(base):
    return [json.loads(line) for line in
            (base / "journal" / "cost-events.jsonl").read_text().splitlines()]


def test_headroom_shrinks_unit_by_unit_and_stops_before_a_unit(base):
    """CPU-08 to runner: each check reconciles the snapshot plus this run's own segments.

    Prior pilot usage $95 of the $100 pilot cap; each Compose run takes 3,600 s at
    $1.20/GPU-h; exposure per unit is 2.0433 USD. Remaining pilot headroom before units
    0..3 is about 5, 3.8, 2.6 and 1.4 USD, so unit 3 is refused.
    """
    packet = make_packet(base)
    clock = Clock()
    summary, events, _ = execute(base, packet, budget=budget_ok(
        packet, prior=[prior_usage("95", "pilot")]), clock=clock, rollout_s=3600)
    jobs = [unit["job"]["job_id"] for unit in packet["units"]]
    assert summary["stop_reason"] == {"index": 3, "job_id": jobs[3], "role": "pilot_job",
                                      "reason": "budget_stop"}
    assert summary["unit_records"][3]["outcome"] == "not_started"
    assert summary["unit_records"][3]["steps"][0]["issues"] == ["pilot_headroom_exhausted"]
    assert {e["job_id"] for e in events} == set(jobs[:3])
    assert summary["not_run_job_ids"] == jobs[3:]
    missing = {i["path"] for i in summary["ledger"]["issues"] if i["code"] == "missing_job"}
    assert missing == {f"job:{job}" for job in jobs[3:]}
    # Oracle: remaining = 100 - 95 - (closed segments + the open one accrued to as_of).
    segments = cost_events(base)
    assert [event["event_id"].rsplit(":", 2)[1:] for event in segments] == [
        ["segment", "0000"], ["segment", "0001"], ["segment", "0002"], ["segment", "0003"]]
    assert segments[0]["start_epoch"] == ACCOUNTED_UNTIL
    for left, right in itertools.pairwise(segments):
        assert left["end_epoch"] == right["start_epoch"]  # contiguous, never double counted
    assert all(event["resource_id"] == summary["budget"]["run_resource_id"] ==
               f"58628722#{summary['clock_anchor']['clock_id'].split(':')[1]}"
               for event in segments)
    remaining = []
    for index in range(4):
        decision = summary["unit_records"][index]["steps"][0]["decision"]
        closed = sum((segment_usd(event) for event in segments[:index]), Fraction(0))
        open_part = segment_usd({"start_epoch": segments[index]["start_epoch"],
                                 "end_epoch": decision["as_of_epoch"], "gpu_count": 1})
        expected = 100 - 95 - closed - open_part
        # s2_cost rounds to 60 significant digits; equal far beyond a cent.
        assert abs(Fraction(decision["remaining_usd"]["pilot"]) - expected) < Fraction(
            1, 10**50)
        assert decision["tier"] == "pilot" and decision["ledger_status"] == "passed"
        assert set(decision["remaining_usd"]) == {"pilot", "stage", "project"}
        remaining.append(expected)
    assert remaining == sorted(remaining, reverse=True) and len(set(remaining)) == 4
    assert remaining[2] > Fraction(7356, 3600) > remaining[3]
    assert summary["budget"]["run_segments"] == 4  # three units plus the tail


def test_exposure_covers_the_configured_worst_case_not_a_smaller_declared_bound(base):
    """Prior $98: a declared bound of $0 would fit, but 1.20 x 6,130 s / 3,600 does not."""
    packet = make_packet(base)
    budget = budget_ok(packet, prior=[prior_usage("98", "pilot")],
                       next_unit_upper_bound_usd="0")
    summary, events, host = execute(base, packet, budget=budget)
    step = summary["unit_records"][0]["steps"][0]
    assert step["issues"] == ["pilot_headroom_exhausted"]
    assert abs(Fraction(step["decision"]["exposure_usd"]) - Fraction(7356, 3600)) < Fraction(
        1, 10**40)
    assert step["worst_case_unit_seconds"] == WORST_CASE_UNIT_S
    assert events == [] and host.calls == []


def test_deadline_without_room_for_the_worst_case_unit_does_not_start(base):
    """B1-F7 repro on the ledger path: 6,000 s before the deadline < 6,070 + 60 s."""
    packet = make_packet(base)
    budget = budget_ok(packet)
    budget["limits"]["deadline_epoch"] = ACCOUNTED_UNTIL + 6000
    summary, events, _ = execute(base, packet, budget=budget)
    assert summary["unit_records"][0]["outcome"] == "not_started" and events == []
    assert summary["unit_records"][0]["steps"][0]["issues"] == ["deadline_headroom_exhausted"]


def _open_instance_interval(budget):
    budget["prior_events"] = [{"event_id": "c", "kind": "compute_interval",
                               "resource_id": "58628722", "start_epoch": 0, "end_epoch": None,
                               "gpu_count": 1}]


def _late_instance_interval(budget):
    budget["prior_events"] = [{"event_id": "c", "kind": "compute_interval",
                               "resource_id": "58628722", "start_epoch": 0,
                               "end_epoch": ACCOUNTED_UNTIL + 1, "gpu_count": 1}]


@pytest.mark.parametrize("mutate, code", [
    (lambda b: b.pop("limits"), "invalid_budget"),
    (lambda b: b.update(extra=1), "invalid_budget"),
    (lambda b: b.update(schema_version=1), "invalid_budget"),  # v2 added line spend (G2-02)
    (lambda b: b.update(schema_version=3), "invalid_budget"),
    (lambda b: b.update(phase="full_sweep"), "budget_phase_mismatch"),
    (lambda b: b.update(scope="actual"), "scope_mismatch"),
    (lambda b: b.update(gpu_count=0), "invalid_budget"),
    (lambda b: b.update(accounted_until_epoch=None), "invalid_budget"),
    (lambda b: b.update(next_unit_upper_bound_usd=2.03), "invalid_budget"),
    (lambda b: b.update(instance_resource_id=""), "invalid_budget"),
    (_open_instance_interval, "prior_compute_overlaps_run"),
    (_late_instance_interval, "prior_compute_overlaps_run"),
    (lambda b: b.update(prior_events=[{**prior_usage("1", "pilot"),
                                       "event_id": "s2_run_units:x"}]), "reserved_event_id"),
    (lambda b: b["quote"].update(compute_usd_per_hour=1.2), "budget_snapshot_invalid"),
    (lambda b: b.update(caps_usd={"pilot": "100"}), "budget_snapshot_invalid"),
    (lambda b: b.update(limits={"task_allowance_usd": None, "reserves_usd": {},
                                "deadline_epoch": 1}), "invalid_budget"),
])
def test_invalid_budget_snapshot_is_refused_before_any_effect(base, mutate, code):
    packet = make_packet(base)
    budget = budget_ok(packet)
    mutate(budget)
    host = FakeHost(base / "journal")
    summary = run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                        clock=Clock(), budget=budget, host_env=HOST_ENV, **run_kw(packet))
    assert summary["run_outcome"] == "refused" and code in codes(summary), summary["issues"]
    assert host.calls == [] and not (base / "journal").exists()


def test_open_compute_of_another_resource_accrues_to_now(base):
    """A second instance's open interval is the snapshot's own and accrues to as_of."""
    packet = make_packet(base)
    other = {"event_id": "other", "kind": "compute_interval", "resource_id": "other-vm",
             "start_epoch": ACCOUNTED_UNTIL - 3600, "end_epoch": None, "gpu_count": 1}
    summary, _, _ = execute(base, packet, budget=budget_ok(packet, prior=[other]))
    decision = summary["unit_records"][0]["steps"][0]["decision"]
    accrued = segment_usd({"start_epoch": ACCOUNTED_UNTIL - 3600,
                           "end_epoch": decision["as_of_epoch"], "gpu_count": 1})
    own = segment_usd({"start_epoch": ACCOUNTED_UNTIL, "end_epoch": decision["as_of_epoch"],
                       "gpu_count": 1})
    assert abs(Fraction(decision["remaining_usd"]["pilot"]) - (100 - accrued - own)) < Fraction(
        1, 10**50)


def test_existing_run_dir_is_never_overwritten(base):
    packet = make_packet(base)
    os.makedirs(packet["units"][0]["compose"]["cwd"])
    summary, events, host = execute(base, packet)
    assert summary["stop_reason"]["reason"] == "run_dir_exists"
    assert host.calls == []
    assert [e["type"] for e in events] == ["started", "cancelled"]


def test_wizard_failure_cancels_and_stops(base):
    summary, events, _ = execute(base, make_packet(base), wizard_rc=1)
    assert summary["stop_reason"]["reason"] == "wizard_prepare_failed"
    assert events[-1]["reason"] == "wizard_prepare_failed"


def test_shm_override_is_written_and_verified_on_the_effective_config(base):
    shm = {"controller-0": "docker_default", "physics-0": "docker_default",
           "renderer-0": "64g", "runtime-0": "docker_default"}
    packet = make_packet(base, shm=shm)
    summary, _, _ = execute(base, packet)
    run_dir = packet["units"][0]["compose"]["cwd"]
    assert Path(run_dir, "docker-compose.override.yaml").read_text() == (
        "services:\n  renderer-0:\n    shm_size: 64g\n")
    assert summary["status"] == "passed" and summary["run_outcome"] == "completed"


def test_override_not_applied_by_compose_blocks_before_gpu(base):
    shm = {"controller-0": "docker_default", "physics-0": "docker_default",
           "renderer-0": "64g", "runtime-0": "docker_default"}
    packet = make_packet(base, shm=shm)
    summary, _, host = execute(base, packet, ignore_override=True)
    assert summary["stop_reason"]["reason"] == "compose_config_blocked:shm_size_mismatch"
    assert not any(c[0] == "start" for c in host.calls)


def test_compose_timeout_cancels_the_incomplete_attempt(base):
    summary, events, _ = execute(base, make_packet(base), scenarios={0: "timeout"})
    assert summary["unit_records"][0]["compose_exit_code"] is None
    assert events[1]["type"] == "cancelled" and events[1]["reason"] == "compose_up_timeout"
    assert events[1]["upstream_identity"] is not None


# --------------------------------------------------------------------- authority

@pytest.mark.parametrize("changes, code", [
    ({"packet_request_sha256": "0" * 64}, "authority_packet_mismatch"),
    ({"job_ids": []}, "authority_jobs_mismatch"),
    ({"action_record_ref": "TBD"}, "authority_unrecorded"),
    ({"action_record_ref": CANARY}, "authority_unrecorded"),
    ({"scope": "actual"}, "scope_mismatch"),
    ({"schema_version": 2}, "invalid_authority"),
])
def test_authority_negative_controls_refuse_before_any_effect(base, changes, code):
    packet = make_packet(base)
    host = FakeHost(base / "journal")
    summary = run_units(packet, authority=authority(packet, **changes), run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                        clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    assert summary["status"] == "blocked" and summary["run_outcome"] == "refused"
    assert code in codes(summary)
    assert host.calls == [] and not (base / "journal").exists()


def test_real_runner_refuses_fixture_scope_and_synthetic_references(base):
    packet = make_packet(base)
    issues = validate_authority(authority(packet), packet, real_runner=True)
    assert {i["code"] for i in issues} == {"fixture_scope_refused",
                                           "synthetic_authority_refused"}
    assert validate_authority(None, packet, real_runner=False)[0]["code"] == "invalid_authority"


@pytest.mark.parametrize("change", [
    {"compose_up_timeout_s": None}, {"driver_poll_interval_s": 900}, {"extra": 1},
    {"schema_version": 2}, {"teardown_timeout_s": float("inf")},
])
def test_run_config_has_no_defaults(base, change):
    packet = make_packet(base)
    config = {**RUN_CONFIG, **change}
    if change.get("compose_up_timeout_s", 0) is None:
        del config["compose_up_timeout_s"]
    host = FakeHost(base / "journal")
    summary = run_units(packet, authority=authority(packet), run_config=config,
                        journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                        clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    assert summary["run_outcome"] == "refused" and "invalid_run_config" in codes(summary)
    assert host.calls == []


def test_cli_validates_without_executing_and_refuses_bad_authority(base, capsys):
    packet = make_packet(base)
    paths = {}
    for name, payload in (("packet", packet), ("authority", authority(packet)),
                          ("config", RUN_CONFIG), ("budget", budget_ok(packet))):
        paths[name] = base / f"{name}.json"
        paths[name].write_text(json.dumps(payload))
    common = [str(paths["packet"]), "--authority", str(paths["authority"]), "--run-config",
              str(paths["config"]), "--budget-file", str(paths["budget"]), "--journal-dir",
              str(base / "journal"), "--unit-selection", "all", "--clock-uncertainty-us",
              str(CLOCK_UNCERTAINTY_US)]
    host = FakeHost(base / "journal")
    assert main(common, runner=host, port_open_fn=host.port_open, clock=Clock(),
                environ=HOST_ENV) == 0
    assert host.calls == [] and "nothing executed" in capsys.readouterr().out
    # B1-F14: a runner that does not declare itself fixture-only is treated as real, so
    # synthetic authority and a missing decision log are refused before anything is
    # created. The sentinel raises on any use, so a refusal regression fails loudly.
    # Review G2-05: the dry run used to skip the real-runner checks here and exit 0; it now
    # applies run_units' checks and refuses the synthetic packet and authority as well.
    sentinel = RefusingRunner()
    assert main(common, runner=sentinel, port_open_fn=sentinel.port_open) == 2
    out = capsys.readouterr().out
    assert "refused: fixture_scope_refused" in out and "validated" not in out
    assert main(common + ["--execute"], runner=sentinel, port_open_fn=sentinel.port_open,
                clock=Clock(), environ=HOST_ENV) == 2
    out = capsys.readouterr().out
    assert "refused: decision_record_missing" in out
    assert sentinel.used == [] and not (base / "journal").exists()
    # An invalid budget snapshot is refused in the dry run too.
    paths["budget"].write_text(json.dumps({**budget_ok(packet), "phase": "full_sweep"}))
    assert main(common, runner=host, port_open_fn=host.port_open) == 2
    assert "refused: budget_phase_mismatch" in capsys.readouterr().out
    paths["budget"].write_text(json.dumps(budget_ok(packet)))
    assert main(common + ["--execute"], runner=host, port_open_fn=host.port_open,
                clock=Clock(), environ=HOST_ENV) == 0
    assert len(host.calls) > 0
    saved = json.loads((base / "journal" / "run-summary.json").read_text())
    assert saved["clock_anchor"]["uncertainty_us"] == CLOCK_UNCERTAINTY_US
    with pytest.raises(SystemExit):  # the selection and uncertainty have no default
        main(common[:-4], runner=host)


# --------------------------------------------------------------------- real runner, local only

def test_subprocess_runner_separates_and_merges_streams(tmp_path):
    runner = SubprocessRunner(tee=io.BytesIO())
    script = "import sys; print('out'); print('err', file=sys.stderr)"
    env = {"PATH": "/usr/bin:/bin"}
    result = runner.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                        stdout_path=str(tmp_path / "a.txt"), timeout_s=30)
    assert result == {"returncode": 0, "timed_out": False}
    assert (tmp_path / "a.txt").read_text() == "out\n"
    assert (tmp_path / "a.txt.stderr").read_text() == "err\n"
    result = runner.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                        stdout_path=str(tmp_path / "b.txt"), timeout_s=30, merge_stderr=True)
    assert result["returncode"] == 0
    assert sorted((tmp_path / "b.txt").read_text().split()) == ["err", "out"]
    with pytest.raises(FileExistsError):
        runner.run([sys.executable, "-c", ""], cwd=str(tmp_path), env=env,
                   stdout_path=str(tmp_path / "a.txt"), timeout_s=30)


def test_subprocess_runner_timeout_and_stop(tmp_path):
    runner = SubprocessRunner(tee=io.BytesIO())
    env = {"PATH": "/usr/bin:/bin"}
    result = runner.run([sys.executable, "-c", "import time; time.sleep(60)"],
                        cwd=str(tmp_path), env=env, stdout_path=str(tmp_path / "t.txt"),
                        timeout_s=0.5)
    assert result == {"returncode": None, "timed_out": True}
    script = "import time; print('up', flush=True); time.sleep(60)"
    handle = runner.start([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                          stdout_path=str(tmp_path / "d.txt"))
    assert runner.poll(handle) is None
    stopped = runner.stop(handle, timeout_s=10)
    assert stopped["returncode"] is not None and runner.poll(handle) is not None


# --------------------------------------------------------------------- review regressions

class RefusingRunner(SubprocessRunner):
    """A real-runner subclass that records and refuses every effect (B1-F14 sentinel)."""

    def __init__(self):
        super().__init__(tee=io.BytesIO())
        self.used = []

    def run(self, argv, **_kwargs):
        self.used.append(("run", argv))
        raise AssertionError(f"sentinel runner must never run {argv}")

    def start(self, argv, **_kwargs):
        self.used.append(("start", argv))
        raise AssertionError(f"sentinel runner must never start {argv}")

    def port_open(self, port):
        self.used.append(("port_open", port))
        raise AssertionError("sentinel port probe must never be used")


def test_runner_without_fixture_declaration_refuses_synthetic_packets(base):
    """B1-F14: the refusal follows the runner object, not a caller-asserted flag."""
    packet = make_packet(base)
    sentinel = RefusingRunner()
    summary = run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=sentinel,
                        port_open_fn=sentinel.port_open, clock=Clock(), budget=budget_ok(packet),
                        host_env=HOST_ENV, **run_kw(packet))
    assert summary["run_outcome"] == "refused"
    assert codes(summary) == {"fixture_scope_refused", "synthetic_authority_refused",
                              "decision_record_missing"}
    assert sentinel.used == [] and not (base / "journal").exists()

    class Undeclared:  # duck-typed fake without the declaration is also "real"
        pass
    summary = run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=Undeclared(),
                        port_open_fn=None, clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    assert "fixture_scope_refused" in codes(summary)


def journal_events(base):
    return [json.loads(line) for line in
            (base / "journal" / "ledger-events.jsonl").read_text().splitlines()]


def after_up(host):
    calls = [(c[0], (c[1] or [])[:3]) for c in host.calls]
    return calls[calls.index(("run", ["docker", "compose", "up"])) + 1:]


CAPTURES_THEN_TEARDOWN = [("run", ["docker", "compose", "logs"]),
                          ("run", ["docker", "compose", "ps"]),
                          ("run", ["docker", "compose", "ps"]),
                          ("run", ["docker", "inspect", "000000000001"]),
                          ("run", ["docker", "compose", "down"]), ("stop", [])]


class InterruptHost(FakeHost):
    """Compose writes a started-only attempt, then the operator presses Ctrl-C."""

    def __init__(self, journal, error=KeyboardInterrupt, **kwargs):
        super().__init__(journal, scenarios={0: "timeout"}, **kwargs)
        self.error = error

    def _up(self, cwd, stdout_path):
        super()._up(cwd, stdout_path)
        raise self.error


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interrupt_after_compose_captures_before_teardown_and_records(base, error):
    """B1-F5: Ctrl-C after Compose starts keeps logs/ps/inspect, closes the attempt and
    writes a partial run summary before re-raising."""
    packet = make_packet(base)
    host = InterruptHost(base / "journal", error)
    with pytest.raises(error):
        run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                  journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                  clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    assert after_up(host) == CAPTURES_THEN_TEARDOWN
    job = packet["units"][0]["job"]["job_id"]
    reason = f"interrupted:{error.__name__}"
    assert by_job(journal_events(base), job) == [("attempt-1", "started", None, False),
                                                 ("attempt-1", "cancelled", reason, True)]
    summary = json.loads((base / "journal" / "run-summary.json").read_text())
    assert summary["run_outcome"] == "interrupted" and summary["status"] == "blocked"
    assert summary["stop_reason"] == {"index": 0, "job_id": job, "role": "pilot_job",
                                      "reason": reason}
    assert summary["attempt_ids"] == [f"{job}/attempt-1"]
    assert len(summary["not_run_job_ids"]) == 9
    assert Path(packet["units"][0]["capture_dir"], "run-dir-manifest.json").exists()
    record = json.loads((base / "journal/units/000/unit-record.json").read_text())
    assert record["outcome"] == "interrupted"


def test_exception_after_compose_still_captures_before_teardown(base):
    """B1-F5: an ordinary error escaping the GPU phase does not skip the captures."""
    packet = make_packet(base)
    host = InterruptHost(base / "journal", RuntimeError("boom"))
    summary = run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                        clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    assert after_up(host) == CAPTURES_THEN_TEARDOWN
    assert summary["stop_reason"]["reason"] == "internal_error:RuntimeError"
    job = packet["units"][0]["job"]["job_id"]
    assert by_job(journal_events(base), job)[-1] == (
        "attempt-1", "cancelled", "internal_error:RuntimeError", True)


FORGED = "ffffffff-ffff-4fff-8fff-ffffffffffff"


class MutateAfterHash(FakeHost):
    """Rewrites run_metadata.yaml and the summary after the run-dir manifest was hashed."""

    def run(self, argv, *, cwd, env, stdout_path, timeout_s, merge_stderr=False):
        result = super().run(argv, cwd=cwd, env=env, stdout_path=stdout_path,
                             timeout_s=timeout_s, merge_stderr=merge_stderr)
        if argv[:3] == ["docker", "compose", "down"] and self.unit == 0:
            Path(cwd, "run_metadata.yaml").write_text(
                f"run_name: e2e_challenge_dev\nrun_uuid: {FORGED}\n")
            path = Path(cwd, "aggregate/results-summary.json")
            data = json.loads(path.read_text())
            for row in data["rollouts"]:
                row["run_uuid"] = row["metrics"]["run_uuid"] = FORGED
            path.write_text(json.dumps(data))
            Path(cwd, "physloop-capture/compose-logs.txt").write_text("forged\n")
        return result


def test_reconcile_reads_only_bytes_the_pre_teardown_manifest_hashed(base):
    """B1-F6: bytes changed after hashing are never reconciled; the mismatch is an issue."""
    packet = make_packet(base)
    summary, events, _ = execute(base, packet, host=MutateAfterHash(base / "journal"))
    job = packet["units"][0]["job"]["job_id"]
    unit_events = [e for e in events if e["job_id"] == job]
    assert all((e["upstream_identity"] or {}).get("run_uuid") != FORGED for e in unit_events)
    assert unit_events[-1]["type"] != "completed"
    record = summary["unit_records"][0]
    assert {(i["code"], i["path"]) for i in record["issues"]} == {
        ("capture_receipt_mismatch", "run_metadata.yaml"),
        ("capture_receipt_mismatch", "aggregate/results-summary.json"),
        ("capture_receipt_mismatch", "physloop-capture/compose-logs.txt")}
    reconcile = next(s for s in record["steps"] if s["id"] == "reconcile")
    assert reconcile["status"] == "unverified"
    # The other units are untouched and still complete.
    assert by_job(events, packet["units"][1]["job"]["job_id"])[-1][1] == "completed"


class NoManifestHost(FakeHost):
    def _up(self, cwd, stdout_path):
        result = super()._up(cwd, stdout_path)
        os.symlink("/nonexistent", f"{cwd}/physloop-capture/run-dir-manifest.json")
        return result


def test_unwritable_manifest_is_an_issue_not_an_empty_file_list(base):
    """B1-F6: an absent run-dir manifest is recorded, never read as 'no files'."""
    packet = make_packet(base)
    summary, events, _ = execute(base, packet, host=NoManifestHost(base / "journal"))
    record = summary["unit_records"][0]
    assert "run_dir_manifest_absent" in codes(record)
    assert next(s for s in record["steps"] if s["id"] == "reconcile")["status"] == "unverified"
    job = packet["units"][0]["job"]["job_id"]
    assert by_job(events, job)[-1][1] != "completed"


class StubbornDriver(FakeHost):
    def stop(self, handle, *, timeout_s):
        self.calls.append(("stop", None, None, None))
        handle.running = False  # port closes, the process survives every signal
        return {"returncode": None, "escalated": True}


def test_driver_surviving_every_signal_stops_the_run(base):
    """B1-F8: returncode None after escalation is an unverified teardown."""
    summary, _, host = execute(base, make_packet(base), host=StubbornDriver(base / "journal"))
    assert summary["stop_reason"]["reason"] == "teardown_unverified"
    assert sum(1 for c in host.calls if c[0] == "start") == 1
    steps = {s["id"]: s for s in summary["unit_records"][0]["steps"]}
    assert steps["driver_stop"]["status"] == "blocked"
    assert steps["gpu_memory_released"]["status"] == "qualify_on_host"


def test_gpu_release_is_labelled_qualify_on_host_on_a_normal_stop(base):
    summary, _, _ = execute(base, make_packet(base))
    steps = {s["id"]: s for s in summary["unit_records"][0]["steps"]}
    assert steps["driver_stop"]["status"] == "passed"
    assert steps["gpu_memory_released"]["status"] == "qualify_on_host"


CONTRACT_STATUSES = {"passed", "blocked", "not_run", "not_estimable", "unverified",
                     "not_applicable", "qualify_on_host"}
ENVELOPE = {"artifact_kind", "schema_version", "scope", "producer", "input_hashes", "units",
            "issues", "status"}


def test_unit_records_and_summaries_use_the_shared_envelope_and_vocabulary(base):
    """B1-F12: contract 2.1 envelope, 2.2 statuses, domain outcomes kept separately."""
    packet = make_packet(base)
    summary, _, _ = execute(base, packet, scenarios={1: "pre_uuid_crash"})
    record = json.loads((base / "journal/units/000/unit-record.json").read_text())
    assert ENVELOPE <= set(record)
    assert record["artifact_kind"] == "s2_unit_record" and record["schema_version"] == 1
    assert record["scope"] == "synthetic" and record["status"] == "passed"
    assert record["producer"]["module"] == "scripts.s2_run_units"
    assert set(record["input_hashes"]) == {"packet", "execution_plan"}
    statuses = {s["status"] for r in summary["unit_records"] for s in r["steps"]}
    assert statuses <= CONTRACT_STATUSES, statuses
    assert summary["status"] in CONTRACT_STATUSES and ENVELOPE <= set(summary)
    refused = run_units(packet, authority=authority(packet, job_ids=[]),
                        run_config=RUN_CONFIG, journal_dir=base / "other",
                        runner=FakeHost(base / "other"), port_open_fn=None, clock=Clock(),
                        budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    assert set(refused) == set(summary) and refused["attempt_ids"] == []
    assert refused["status"] == "blocked"


def test_capture_failures_use_contract_statuses(base):
    summary, _, _ = execute(base, make_packet(base), scenarios={0: "timeout"})
    steps = {s["id"]: s for s in summary["unit_records"][0]["steps"]}
    assert steps["compose_up"]["status"] == "unverified"
    assert steps["compose_up"]["outcome"] == "recorded"


class NoPull(FakeHost):
    """Compose rejects '--pull never' (its support is qualify_on_host)."""

    def _up(self, cwd, stdout_path):
        self.lines = []
        Path(stdout_path).write_text("unknown flag: --pull\n")
        return {"returncode": 125, "timed_out": False}


def test_systemic_compose_failure_stops_after_the_first_unit(base):
    """B1-F13: no upstream attempt evidence stops the run instead of repeating ten times."""
    summary, events, host = execute(base, make_packet(base), host=NoPull(base / "journal"))
    assert summary["stop_reason"]["reason"] == "no_upstream_attempt_evidence"
    assert sum(1 for c in host.calls if c[0] == "start") == 1
    assert len(summary["not_run_job_ids"]) == 9
    assert [e["reason"] for e in events if e["type"] != "started"] == [
        "no_upstream_attempt_evidence"]


# --------------------------------------------------------------------- B1 residuals (B1R)

def unit_steps(record):
    return {step["id"]: step for step in record["steps"]}


def step_order(record):
    return [step["id"] for step in record["steps"]]


class CaptureInterrupt(FakeHost):
    """Compose completes normally; one interrupt lands inside one docker capture."""

    def __init__(self, journal, prefix, error, **kwargs):
        super().__init__(journal, **kwargs)
        self.prefix, self.error = prefix, error

    def run(self, argv, **kwargs):
        if argv[:len(self.prefix)] == self.prefix:
            self.calls.append(("run", list(argv), kwargs.get("cwd"), dict(kwargs["env"])))
            raise self.error
        return super().run(argv, **kwargs)


DOCKER_CAPTURES = [("capture:compose_logs", ["docker", "compose", "logs"]),
                   ("capture:compose_ps", ["docker", "compose", "ps", "-a", "--format"]),
                   ("capture:compose_container_ids", ["docker", "compose", "ps", "-a", "-q"]),
                   ("capture:container_inspect", ["docker", "inspect"])]


def run_interrupted(base, packet, host, error):
    with pytest.raises(error):
        run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                  journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                  clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    summary = json.loads((base / "journal" / "run-summary.json").read_text())
    return summary, journal_events(base)


def assert_recorded_interrupt(packet, summary, events, error):
    """The completed upstream attempt stays completed; the unit and run record the stop."""
    job = packet["units"][0]["job"]["job_id"]
    reason = f"interrupted:{error.__name__}"
    assert by_job(events, job) == [("attempt-1", "started", None, False),
                                   ("attempt-1", "completed", "process_completed", True)]
    assert summary["run_outcome"] == "interrupted" and summary["status"] == "blocked"
    assert summary["stop_reason"] == {"index": 0, "job_id": job, "role": "pilot_job",
                                      "reason": reason}
    record = summary["unit_records"][0]
    assert record["outcome"] == "interrupted" and record["status"] == "blocked"
    order = step_order(record)
    assert order.index("run_dir_hash") < order.index("compose_down")
    steps = unit_steps(record)
    assert steps["compose_down"]["status"] == "passed"
    assert steps["driver_stop"]["status"] == "passed"
    assert steps["driver_port_released"]["status"] == "passed"
    return steps


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("position", range(len(DOCKER_CAPTURES)))
def test_interrupt_inside_a_docker_capture_still_hashes_and_tears_down(base, position, error):
    """B1R-1: a BaseException inside one capture no longer skips the later captures, the
    run-dir hash or the run metadata; remaining docker captures are recorded not_run."""
    packet = make_packet(base)
    step_id, prefix = DOCKER_CAPTURES[position]
    host = CaptureInterrupt(base / "journal", prefix, error)
    summary, events = run_interrupted(base, packet, host, error)
    assert after_up(host) == [*CAPTURES_THEN_TEARDOWN[:position + 1],
                              *CAPTURES_THEN_TEARDOWN[-2:]]
    assert Path(packet["units"][0]["capture_dir"], "run-dir-manifest.json").exists()
    steps = assert_recorded_interrupt(packet, summary, events, error)
    assert steps[step_id]["status"] == "blocked"
    assert steps[step_id]["outcome"] == "interrupted"
    assert steps[step_id]["error"] == error.__name__
    for later, _ in DOCKER_CAPTURES[position + 1:]:
        assert steps[later]["status"] == "not_run" and steps[later]["reason"] == "interrupted"
    assert steps["run_metadata_capture"]["status"] == "passed"
    assert steps["run_dir_hash"]["status"] == "passed"
    assert sum(1 for c in host.calls if c[0] == "start") == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interrupt_in_the_compose_up_stdout_check_is_recorded(base, monkeypatch, error):
    packet = make_packet(base)
    target = packet["units"][0]["required_captures_before_teardown"][0]["stdout_path"]
    original = os.path.isfile

    def isfile(path):
        if str(path) == target:
            raise error
        return original(path)
    monkeypatch.setattr("scripts.s2_run_units.os.path.isfile", isfile)
    host = FakeHost(base / "journal")
    summary, events = run_interrupted(base, packet, host, error)
    assert after_up(host) == CAPTURES_THEN_TEARDOWN[-2:]
    steps = assert_recorded_interrupt(packet, summary, events, error)
    assert steps["capture:compose_up_stdout"]["status"] == "blocked"
    assert all(steps[later]["status"] == "not_run" for later, _ in DOCKER_CAPTURES)
    assert steps["run_dir_hash"]["status"] == "passed"


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interrupt_while_reading_run_metadata_still_hashes(base, monkeypatch, error):
    import scripts.s2_run_units as module
    original = module._read_file

    def read(path, max_bytes):
        if str(path).endswith("/run_metadata.yaml"):
            raise error
        return original(path, max_bytes)
    monkeypatch.setattr(module, "_read_file", read)
    packet = make_packet(base)
    host = FakeHost(base / "journal")
    summary, events = run_interrupted(base, packet, host, error)
    assert after_up(host) == CAPTURES_THEN_TEARDOWN
    steps = assert_recorded_interrupt(packet, summary, events, error)
    assert steps["run_metadata_capture"]["status"] == "blocked"
    assert steps["run_metadata_capture"]["error"] == error.__name__
    assert steps["run_dir_hash"]["status"] == "passed"


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interrupt_while_hashing_the_run_dir_is_recorded_and_torn_down(base, monkeypatch,
                                                                        error):
    def boom(*_args, **_kwargs):
        raise error
    monkeypatch.setattr("scripts.s2_run_units.build_manifest", boom)
    packet = make_packet(base)
    host = FakeHost(base / "journal")
    summary, _ = run_interrupted(base, packet, host, error)
    assert after_up(host) == CAPTURES_THEN_TEARDOWN
    record = summary["unit_records"][0]
    steps = unit_steps(record)
    assert steps["run_dir_hash"]["status"] == "blocked"
    assert steps["run_dir_hash"]["error"] == error.__name__
    assert step_order(record).index("run_dir_hash") < step_order(record).index("compose_down")
    assert steps["driver_stop"]["status"] == "passed"
    assert summary["run_outcome"] == "interrupted" and record["status"] == "blocked"


class DownRaises(FakeHost):
    def __init__(self, journal, error, **kwargs):
        super().__init__(journal, **kwargs)
        self.error = error

    def run(self, argv, **kwargs):
        if argv[:3] == ["docker", "compose", "down"]:
            self.calls.append(("run", list(argv), kwargs.get("cwd"), dict(kwargs["env"])))
            raise self.error
        return super().run(argv, **kwargs)


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interrupt_during_compose_down_still_stops_the_driver(base, error):
    """B1R-3: the 10B driver is stopped and its port release verified, then re-raised."""
    packet = make_packet(base)
    host = DownRaises(base / "journal", error)
    summary, events = run_interrupted(base, packet, host, error)
    assert host.driver.stopped and after_up(host) == CAPTURES_THEN_TEARDOWN
    record = summary["unit_records"][0]
    steps = unit_steps(record)
    assert steps["compose_down"]["status"] == "blocked"
    assert steps["compose_down"]["error"] == error.__name__
    assert steps["driver_stop"]["status"] == "passed"
    assert steps["driver_port_released"]["status"] == "passed"
    assert summary["run_outcome"] == "interrupted" and record["status"] == "blocked"
    job = packet["units"][0]["job"]["job_id"]
    assert summary["stop_reason"]["reason"] == f"interrupted:{error.__name__}"
    assert by_job(events, job)[-1] == ("attempt-1", "completed", "process_completed", True)


def test_runner_error_during_compose_down_still_stops_the_driver(base):
    packet = make_packet(base)
    summary, _, host = execute(base, packet, host=DownRaises(base / "journal", ValueError("x")))
    assert host.driver.stopped and after_up(host) == CAPTURES_THEN_TEARDOWN
    steps = unit_steps(summary["unit_records"][0])
    assert steps["compose_down"]["status"] == "blocked"
    assert steps["compose_down"]["error"] == "ValueError"
    assert steps["driver_stop"]["status"] == "passed"
    assert steps["driver_port_released"]["status"] == "passed"
    assert summary["stop_reason"]["reason"] == "teardown_unverified"
    assert summary["status"] == "blocked" and len(summary["not_run_job_ids"]) == 9


class FailingCaptures(FakeHost):
    """Every docker CLI capture fails and run_metadata.yaml disappears before captures."""

    def run(self, argv, **kwargs):
        if argv[:3] in (["docker", "compose", "logs"], ["docker", "compose", "ps"]):
            self.calls.append(("run", list(argv), kwargs.get("cwd"), dict(kwargs["env"])))
            Path(kwargs["stdout_path"]).write_text("")
            return {"returncode": 1, "timed_out": False}
        return super().run(argv, **kwargs)

    def _up(self, cwd, stdout_path):
        result = super()._up(cwd, stdout_path)
        os.unlink(f"{cwd}/run_metadata.yaml")
        return result


def test_blocked_captures_never_read_as_a_passed_unit_or_run(base):
    """B1R-2 probe 1: absent or failed evidence is never 'passed' (contract 2.2)."""
    summary, _, _ = execute(base, make_packet(base), host=FailingCaptures(base / "journal"))
    assert summary["run_outcome"] == "completed"
    record = summary["unit_records"][0]
    assert record["status"] == "blocked"
    assert {"step_blocked:capture:compose_logs", "step_not_run:capture:container_inspect",
            "step_unverified:run_metadata_capture"} <= set(record["status_reasons"])
    assert any(reason.startswith("terminal_not_established:")
               for reason in record["status_reasons"])
    assert {r["status"] for r in summary["unit_records"]} == {"blocked"}
    assert summary["status"] == "blocked"
    assert "unit:0:blocked" in summary["status_reasons"]


def test_receipt_mismatch_makes_the_run_unverified_not_passed(base):
    """B1R-2 probe 2: capture_receipt_mismatch on one unit."""
    packet = make_packet(base)
    summary, _, _ = execute(base, packet, host=MutateAfterHash(base / "journal"))
    assert summary["run_outcome"] == "completed"
    assert summary["unit_records"][0]["status"] == "unverified"
    assert "issue:capture_receipt_mismatch" in summary["unit_records"][0]["status_reasons"]
    assert [r["status"] for r in summary["unit_records"][1:]] == ["passed"] * 9
    assert summary["status"] == "unverified"
    assert summary["status_reasons"] == ["unit:0:unverified"]


def test_cancelled_terminal_is_not_a_passed_unit(base):
    """A runner-stopped Compose (cancelled attempt) never establishes completion."""
    summary, _, _ = execute(base, make_packet(base), scenarios={0: "timeout"})
    record = summary["unit_records"][0]
    assert record["status"] == "unverified"
    assert record["status_reasons"] == [
        f"terminal_not_established:{record['job_id']}/attempt-1:cancelled:compose_up_timeout"]
    assert summary["status"] == "unverified" and summary["run_outcome"] == "completed"


def test_passed_unit_and_run_carry_no_status_reasons(base):
    summary, _, _ = execute(base, make_packet(base))
    assert {r["status"] for r in summary["unit_records"]} == {"passed"}
    assert all(r["status_reasons"] == [] for r in summary["unit_records"])
    assert summary["status"] == "passed" and summary["status_reasons"] == []


class SilentTimeout(FakeHost):
    """Compose times out without printing a single runtime line."""

    def _up(self, cwd, stdout_path):
        self.lines = []
        Path(stdout_path).write_text("Attaching to runtime-0-1\n")
        return {"returncode": None, "timed_out": True}


def test_compose_timeout_without_upstream_evidence_is_systemic(base):
    """B1R-4 (and B1R-2 probe 3): one driver load and one Compose timeout, not ten."""
    summary, events, host = execute(base, make_packet(base), host=SilentTimeout(base / "journal"))
    assert summary["stop_reason"]["reason"] == "no_upstream_attempt_evidence"
    assert sum(1 for c in host.calls if c[0] == "start") == 1
    assert sum(1 for c in host.calls if c[1] and c[1][:3] == ["docker", "compose", "up"]) == 1
    assert len(summary["not_run_job_ids"]) == 9
    # The ledger event stays the truthful recorder reason (P4b).
    assert [(e["type"], e["reason"]) for e in events if e["type"] != "started"] == [
        ("cancelled", "compose_up_timeout")]
    assert summary["unit_records"][0]["status"] == "blocked"
    assert summary["status"] == "blocked" and summary["run_outcome"] == "stopped"


# --------------------------------------------------------------------- full-sweep packets

def make_sweep_packet(base):
    """The synthetic 2 x 441 x 3 sweep (``gate.s2_schedule`` fixes 441 scenes and at least
    three replicates, so 2,646 units is the smallest full sweep); outputs under ``base``."""
    request = full_sweep_request()
    request["launch"]["renderer_camera_override"] = "4cam_1080"
    for arm, policy in request["launch"]["policies"].items():
        policy["run_dir"] = f"{base}/runs/{arm}"
        policy["driver_log_dir"] = f"{base}/driver-logs/{arm}"
    packet = build_full_sweep_packet(request)
    assert packet["prepared"] and packet["artifact_kind"] == "s2_full_sweep_packet", \
        packet["issues"]
    return packet


def test_full_sweep_packet_runs_units_on_their_own_ports_through_the_fake_host(base):
    """Execution is bounded by the stage cap after three of the 2,646 units.

    D-018 nesting: the pilot tier is $0.50 from its cap, which would refuse the first unit,
    but a full-sweep unit is metered from the stage tier ($5 of headroom), so three
    3,600 s units run and the fourth stops on the stage cap, never the pilot cap.
    """
    packet = make_sweep_packet(base)
    budget = budget_ok(packet, prior=[prior_usage("99.5", "pilot"),
                                      prior_usage("1095.5", "stage", "prior-usage-2")])
    clock = Clock()
    summary, events, host = execute(base, packet, budget=budget, clock=clock, rollout_s=3600)
    jobs = [unit["job"]["job_id"] for unit in packet["units"]]
    assert summary["scope"] == "synthetic"
    assert summary["stop_reason"] == {"index": 3, "job_id": jobs[3], "role": "pilot_job",
                                      "reason": "budget_stop"}
    assert summary["budget"]["tier"] == "stage" and summary["budget"]["phase"] == "full_sweep"
    checks = [record["steps"][0] for record in summary["unit_records"]]
    assert [check["issues"] for check in checks] == [[], [], [], ["stage_headroom_exhausted"]]
    assert all(set(check["decision"]["remaining_usd"]) == {"stage", "project"}
               for check in checks)
    assert summary["authority"]["release"]["authorized"] is True
    assert [r["outcome"] for r in summary["unit_records"][:3]] == ["executed"] * 3
    assert {r["status"] for r in summary["unit_records"][:3]} == {"passed"}
    assert summary["unit_records"][3]["outcome"] == "not_started"
    assert summary["not_run_job_ids"] == jobs[3:] and len(jobs[3:]) == 2643
    for index in range(3):
        assert by_job(events, jobs[index]) == [
            ("attempt-1", "started", None, False),
            ("attempt-1", "completed", "process_completed", True)]
    assert {e["job_id"] for e in events} == set(jobs[:3])
    # Every executed argv is the unit's own; each driver came up on its unit's port.
    starts = [c for c in host.calls if c[0] == "start"]
    assert [c[1] for c in starts] == [unit["driver"]["argv"] for unit in packet["units"][:3]]
    wizards = [c[1] for c in host.calls if c[1] and c[1][:4] == [
        "uv", "run", "--no-sync", "alpasim_wizard"]]
    assert wizards == [unit["wizard_prepare"]["argv"] for unit in packet["units"][:3]]
    for index, record in enumerate(summary["unit_records"][:3]):
        steps = {step["id"]: step["status"] for step in record["steps"]}
        assert steps["driver_ready"] == steps["driver_port_released"] == "passed"
        assert f"driver.port={20000 + index}" in wizards[index]
    # driver_ready passed only if port_open saw the driver on the plan's port for that unit.
    assert host.driver.port == 20002 and host.driver.stopped
    missing = {i["path"] for i in summary["ledger"]["issues"] if i["code"] == "missing_job"}
    assert missing == {f"job:{job}" for job in jobs[3:]}
    _check_ledger(summary["ledger"])


def _refused(base, packet):
    host = FakeHost(base / "journal")
    summary = run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                        clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run_kw(packet))
    assert summary["status"] == "blocked" and summary["run_outcome"] == "refused"
    assert host.calls == [] and not (base / "journal").exists()
    return codes(summary)


def test_tampered_full_sweep_inputs_are_refused_before_any_effect(base):
    packet = make_sweep_packet(base)
    packet["inputs"]["launch"]["driver_port_base"] = 30000
    assert _refused(base, packet) == {"request_hash_mismatch"}
    packet["request_sha256"] = identity_hash(packet["inputs"])
    # The authority names the new hash too, so only the rebuild binding refuses.
    assert _refused(base, packet) == {"packet_rebuild_mismatch"}


def test_unknown_packet_kind_is_refused_before_any_effect(base):
    packet = make_packet(base)
    packet["artifact_kind"] = "s2_other_packet"
    assert _refused(base, packet) == {"unsupported_packet"}


# --------------------------------------------------------------------- hash-bound authority

class RealHost(FakeHost):
    """The FakeHost without the fixture declaration: the runner treats it as real."""

    fixture_only = False


def readiness_for(packet, **changes):
    """A stand-in shaped like a passed inspect_readiness report for ``packet`` (synthetic;
    ``test_a_real_inspect_readiness_report_is_accepted_for_its_own_packet`` covers the
    real producer's output)."""
    report = {"artifact_kind": "s2_readiness_report", "schema_version": 1,
              "scope": packet["scope"], "phase": "harmonizer_pilot", "status": "passed",
              "sections": {"planning_consistency": {
                  "status": "passed",
                  "packet": {"request_sha256": packet["request_sha256"]}}}}
    report.update(changes)
    return report


def real_case(base, *, diagnostics=None):
    packet = make_packet(base, scope="actual", diagnostics=diagnostics)
    readiness = readiness_for(packet)
    grants = [("s2_pilot_packet", identity_hash(packet)),
              ("s2_readiness_report", identity_hash(readiness))]
    auth = authority(packet, scope="actual", action_record_ref="D-901 unit-run approval")
    run = run_kw(packet, decisions_md=decisions_md(("D-901", grants, [])),
                 decision_id="D-901", readiness_report=readiness)
    return packet, readiness, grants, auth, run


def refused_codes(base, packet, auth, run, host_class=RealHost):
    host = host_class(base / "journal")
    summary = run_units(packet, authority=auth, run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                        clock=Clock(), budget=budget_ok(packet), host_env=HOST_ENV, **run)
    assert summary["run_outcome"] == "refused" and summary["status"] == "blocked"
    assert host.calls == [] and not (base / "journal").exists()
    return codes(summary), summary


def test_real_runner_executes_only_under_a_hash_bound_decision(base):
    packet, readiness, _grants, auth, run = real_case(base)
    clock = Clock()
    host = RealHost(base / "journal", clock=clock)
    summary, _events, _ = execute(base, packet, host=host, clock=clock, run=run, auth=auth)
    assert summary["run_outcome"] == "completed" and summary["status"] == "passed"
    assert len(summary["unit_records"]) == 10
    decision = summary["authority"]["decision"]
    assert summary["authority"]["mode"] == "recorded_decision"
    assert decision["authorized"] is True and decision["decision_id"] == "D-901"
    assert decision["authenticity"] == "supplied_not_authenticated"
    assert summary["authority"]["packet_sha256"] == identity_hash(packet)
    assert summary["authority"]["readiness_report_sha256"] == identity_hash(readiness)
    assert summary["authority"]["readiness_waived"] is False
    assert summary["input_hashes"]["readiness_report"] == identity_hash(readiness)


def _other_packet_grant(case):
    _packet, _readiness, grants, _auth, run = case
    run["decisions_md"] = decisions_md(("D-901", [("s2_pilot_packet", "0" * 64), grants[1]],
                                        []))


def _readiness_grant_missing(case):
    case[4]["decisions_md"] = decisions_md(("D-901", [case[2][0]], []))


def _closed_later(case):
    case[4]["decisions_md"] = decisions_md(("D-901", case[2], []),
                                           ("D-902", [], ["closes_authority: D-901"]))


def _readiness_for_other_request(case):
    packet, _readiness, grants, _auth, run = case
    other = readiness_for({**packet, "request_sha256": "1" * 64})
    run["readiness_report"] = other
    run["decisions_md"] = decisions_md(("D-901", [grants[0], ("s2_readiness_report",
                                                              identity_hash(other))], []))


def _readiness_blocked(case):
    packet, _readiness, grants, _auth, run = case
    blocked = readiness_for(packet, status="blocked")
    run["readiness_report"] = blocked
    run["decisions_md"] = decisions_md(("D-901", [grants[0], ("s2_readiness_report",
                                                              identity_hash(blocked))], []))


@pytest.mark.parametrize("mutate, expected", [
    (lambda c: c[4].update(decision_id=None), {"decision_record_missing"}),
    (lambda c: c[4].update(decisions_md=None), {"decision_record_missing"}),
    (lambda c: c[3].update(action_record_ref="approved by Huy in chat"),
     {"authority_decision_mismatch"}),
    (lambda c: c[3].update(action_record_ref="D-9010 approval"),
     {"authority_decision_mismatch"}),
    (_other_packet_grant, {"packet_not_authorized"}),
    (_readiness_grant_missing, {"readiness_not_authorized"}),
    (_closed_later, {"packet_not_authorized"}),
    (lambda c: c[4].update(decision_id="D-777"), {"authority_decision_mismatch",
                                                  "packet_not_authorized"}),
    (lambda c: c[4].update(readiness_report=None), {"invalid_readiness_report"}),
    (_readiness_for_other_request, {"readiness_packet_mismatch"}),
    (_readiness_blocked, {"readiness_not_passed"}),
    (lambda c: c[4].update(readiness_report={**c[1], "phase": "full_sweep"}),
     {"readiness_phase_mismatch"}),
    (lambda c: c[4].update(readiness_report={**c[1], "scope": "synthetic"}),
     {"readiness_scope_mismatch"}),
])
def test_real_runner_negative_controls_refuse_before_any_effect(base, mutate, expected):
    case = list(real_case(base))
    mutate(case)
    packet, _readiness, _grants, auth, run = case
    found, _summary = refused_codes(base, packet, auth, run)
    assert found == expected


def test_closed_decision_is_named_in_the_refusal(base):
    case = list(real_case(base))
    _closed_later(case)
    _found, summary = refused_codes(base, case[0], case[3], case[4])
    assert summary["authority"]["decision"]["closed_by"] == "D-902"


def test_engineering_only_decision_never_authorizes_pilot_units(base):
    packet, _readiness, grants, auth, run = real_case(base, diagnostics=[(0, {})])
    run["decisions_md"] = decisions_md(("D-901", [grants[0]],
                                        ["engineering_only_diagnostic: true"]))
    run["readiness_report"] = None
    found, _summary = refused_codes(base, packet, auth, run)
    assert found == {"engineering_only_decision_for_scientific_units"}


def test_engineering_only_decision_runs_declared_diagnostics_without_readiness(base):
    packet, _readiness, grants, auth, run = real_case(base, diagnostics=[(0, {})])
    run.update(decisions_md=decisions_md(("D-901", [grants[0]],
                                          ["engineering_only_diagnostic: true"])),
               readiness_report=None, unit_selection="diagnostics_only")
    clock = Clock()
    host = RealHost(base / "journal", clock=clock)
    summary, events, _ = execute(base, packet, host=host, clock=clock, run=run, auth=auth)
    assert summary["run_outcome"] == "completed"
    assert summary["authority"]["readiness_waived"] is True
    assert summary["authority"]["decision"]["engineering_only_diagnostic"] is True
    assert [r["job_id"] for r in summary["diagnostic_unit_records"]] == [
        "diagnostic-camera-check"]
    assert summary["unit_records"] == [] and events == []  # the pilot ledger stays empty
    assert len(summary["not_run_job_ids"]) == 10


def test_a_real_inspect_readiness_report_is_accepted_for_its_own_packet(tmp_path):
    from test_s2_readiness import complete_case

    from gate.s2_readiness import inspect_readiness

    plan, index, schedule, record = complete_case(tmp_path)
    report = inspect_readiness(plan, index, schedule, record, root=tmp_path)
    assert report["status"] == "passed"
    assert readiness_issues(report, plan["packet"]) == []
    other = make_packet(tmp_path / "other")
    assert {item["code"] for item in readiness_issues(report, other)} == {
        "readiness_packet_mismatch"}
    assert readiness_issues(None, other)[0]["code"] == "invalid_readiness_report"


# --------------------------------------------------------------------- full-sweep release

def test_full_sweep_without_a_recorded_release_is_refused(base):
    packet = make_sweep_packet(base)
    run = run_kw(packet)
    for key in ("decisions_md", "release_record", "release_decision_id"):
        run.pop(key)
    found, _ = refused_codes(base, packet, authority(packet), run, host_class=FakeHost)
    assert found == {"full_release_unrecorded"}
    other = {**RELEASE, "note": "another release"}
    found, summary = refused_codes(base, packet, authority(packet),
                                   run_kw(packet, release_record=other), host_class=FakeHost)
    assert found == {"full_release_unrecorded"}
    assert summary["authority"]["release"]["authorized"] is False
    assert "artifact_not_authorized" in summary["authority"]["release"]["issue_codes"]


def test_pilot_packet_takes_no_release_record(base):
    packet = make_packet(base)
    found, _ = refused_codes(base, packet, authority(packet),
                             run_kw(packet, release_record=RELEASE), host_class=FakeHost)
    assert found == {"release_not_applicable"}


# --------------------------------------------------------------------- diagnostic units

def test_declared_diagnostic_runs_first_with_its_own_ledger_and_budget_line(base):
    packet = make_packet(base, diagnostics=[(0, {})])
    summary, events, host = execute(base, packet)
    assert summary["run_outcome"] == "completed" and summary["status"] == "passed"
    [diagnostic] = summary["diagnostic_unit_records"]
    assert diagnostic["job_id"] == "diagnostic-camera-check" and diagnostic["index"] == 0
    check = diagnostic["steps"][0]
    assert check["role"] == "diagnostic_qualification" and check["issues"] == []
    # Review G2-02: the line detail separates carried-forward and this run's spend.
    assert check["decision"]["budget_line"] == {"line_id": "pilot_diagnostics", "cap_usd": "5",
                                                "prior_spent_usd": "0", "run_spent_usd": "0",
                                                "spent_usd": "0", "remaining_usd": "5"}
    assert summary["unit_records"][0]["steps"][0]["role"] == "pilot_job"
    # It ran before the pilot units, in its own directories.
    wizards = [c[1] for c in host.calls if c[1] and c[1][:4] == [
        "uv", "run", "--no-sync", "alpasim_wizard"]]
    assert wizards[0] == packet["diagnostic_units"][0]["wizard_prepare"]["argv"]
    assert wizards[1:] == [unit["wizard_prepare"]["argv"] for unit in packet["units"]]
    # Never one of the ten pilot jobs: its events are only in the diagnostic ledger.
    assert "diagnostic-camera-check" not in {e["job_id"] for e in events}
    assert summary["ledger"]["counts"]["scheduled_jobs"] == 10
    assert summary["ledger"]["admitted"] is True
    diag_events = [json.loads(line) for line in (
        base / "journal" / "diagnostic-ledger-events.jsonl").read_text().splitlines()]
    assert {e["job_id"] for e in diag_events} == {"diagnostic-camera-check"}
    assert summary["diagnostic_ledger"]["admitted"] is True
    assert summary["diagnostic_ledger"]["counts"]["scheduled_jobs"] == 1
    assert (base / "journal/diagnostics/000/unit-record.json").exists()
    assert (base / "journal/units/000/unit-record.json").exists()
    started = next(e for e in diag_events if e["type"] == "started")
    assert started["evidence_refs"] == [
        f"packet_unit:{packet['diagnostic_units'][0]['commands_sha256']}"]
    terminal = next(e for e in diag_events if e["type"] != "started")
    assert terminal["upstream_identity"]["clipgt_id"] == packet["units"][0]["job"]["scene_id"]


def test_diagnostics_only_runs_no_pilot_job(base):
    packet = make_packet(base, diagnostics=[(0, {}), (1, {"diagnostic_id": "harmonizer-on"})])
    summary, events, _ = execute(base, packet, run=run_kw(packet,
                                                          unit_selection="diagnostics_only"))
    assert summary["run_outcome"] == "completed"
    assert [r["job_id"] for r in summary["diagnostic_unit_records"]] == [
        "diagnostic-camera-check", "diagnostic-harmonizer-on"]
    assert summary["unit_records"] == [] and events == []
    assert summary["not_run_job_ids"] == [u["job"]["job_id"] for u in packet["units"]]
    assert summary["not_run_diagnostic_job_ids"] == []


def test_diagnostic_budget_line_is_metered_separately(base):
    """Line cap $3: the first 3,600 s diagnostic spends 1.2 USD, leaving 1.8 < 2.0433."""
    line = {"line_id": "pilot_diagnostics", "cap_usd": "3"}
    packet = make_packet(base, diagnostics=[(0, {"budget_line": line}),
                                            (1, {"diagnostic_id": "second", "budget_line": line})])
    clock = Clock()
    summary, events, _ = execute(base, packet, clock=clock, rollout_s=3600)
    assert summary["stop_reason"] == {"index": 1, "job_id": "diagnostic-second",
                                      "role": "diagnostic_qualification",
                                      "reason": "budget_stop"}
    first, second = summary["diagnostic_unit_records"]
    assert first["outcome"] == "executed" and second["outcome"] == "not_started"
    assert second["steps"][0]["issues"] == ["diagnostic_budget_line_exhausted"]
    spent = Fraction(second["steps"][0]["decision"]["budget_line"]["spent_usd"])
    exact = segment_usd(cost_events(base)[0])
    assert 0 <= spent - exact < Fraction(1, 10**50)  # ceiling rounding at 60 digits
    assert Fraction(3) - spent < Fraction(7356, 3600)
    assert summary["unit_records"] == [] and events == []  # no pilot unit ran


def test_line_cap_below_one_unit_refuses_the_diagnostic(base):
    packet = make_packet(base, diagnostics=[(0, {"budget_line": {
        "line_id": "pilot_diagnostics", "cap_usd": "2"}})])
    summary, _, host = execute(base, packet)
    assert summary["diagnostic_unit_records"][0]["steps"][0]["issues"] == [
        "diagnostic_budget_line_exhausted"]
    assert host.calls == []


@pytest.mark.parametrize("changes, code", [
    ({"unit_selection": "diagnostics_only"}, "no_declared_diagnostics"),
    ({"unit_selection": None}, "invalid_unit_selection"),
    ({"unit_selection": "pilot"}, "invalid_unit_selection"),
    ({"clock_uncertainty_us": None}, "clock_uncertainty_unset"),
    ({"clock_uncertainty_us": True}, "clock_uncertainty_unset"),
    ({"clock_uncertainty_us": -1}, "clock_uncertainty_unset"),
    ({"clock_uncertainty_us": 1.5}, "clock_uncertainty_unset"),
])
def test_run_inputs_have_no_defaults(base, changes, code):
    packet = make_packet(base)
    found, _ = refused_codes(base, packet, authority(packet), run_kw(packet, **changes),
                             host_class=FakeHost)
    assert found == {code}


# --------------------------------------------------------------------- clock anchor (F6)

def test_run_summary_records_a_clock_anchor_and_every_step_is_timed(base):
    packet = make_packet(base)
    summary, events, _ = execute(base, packet)
    anchor = summary["clock_anchor"]
    assert set(anchor) == {"clock_id", "monotonic_us", "utc", "uncertainty_us", "anchor_ref"}
    assert anchor["uncertainty_us"] == CLOCK_UNCERTAINTY_US
    assert anchor["anchor_ref"] == "run-summary.json#clock_anchor"
    assert anchor["clock_id"].startswith("s2_run_units:") and len(anchor["clock_id"]) == 29
    moment = datetime.fromisoformat(anchor["utc"])
    assert moment.utcoffset() == timedelta(0)
    # Clock.epoch() is 1.8e9 + monotonic seconds, sampled right after the monotonic read.
    assert abs(moment.timestamp() - (1_800_000_000 + anchor["monotonic_us"] / 1e6)) < 1e-5
    assert anchor["monotonic_us"] < events[0]["timestamp_us"]
    previous = anchor["monotonic_us"]
    for record in summary["unit_records"]:
        for step in record["steps"]:
            assert type(step["start_us"]) is int and type(step["end_us"]) is int
            assert previous <= step["start_us"] <= step["end_us"]
            previous = step["end_us"]
        pairs = list(itertools.pairwise(record["steps"]))
        assert all(left["end_us"] == right["start_us"] for left, right in pairs)
    # The runner's records feed the telemetry producers on the anchor's clock.
    rows = useful_attempt_rows(summary["ledger"], summary["unit_records"],
                               clock_id=anchor["clock_id"], scope="synthetic")
    assert rows["counts"]["attempts"] == 10 and rows["counts"]["useful_true"] == 10


def test_budget_meter_known_answers_without_a_run(tmp_path):
    """Direct meter oracle: $2/GPU-h, two GPUs, 1,800 s billed segment = $2."""
    packet = make_packet(tmp_path)
    budget = budget_ok(packet, gpu_count=2)
    budget["quote"]["compute_usd_per_hour"] = "2"
    issues, context = validate_budget(budget, packet, worst_case_s=0)
    assert issues == []
    meter = BudgetMeter(context, "k", tmp_path / "cost.jsonl")
    try:
        first = meter.check(ACCOUNTED_UNTIL)
        assert first["can_start"] and first["remaining_usd"] == {
            "pilot": "100", "stage": "1200", "project": "2500"}
        event = meter.close_segment(ACCOUNTED_UNTIL + 1800)
        assert event == {"event_id": "s2_run_units:k:segment:0000", "kind": "compute_interval",
                         "resource_id": "58628722#k", "start_epoch": ACCOUNTED_UNTIL,
                         "end_epoch": ACCOUNTED_UNTIL + 1800, "gpu_count": 2}
        after = meter.check(ACCOUNTED_UNTIL + 1800)
        assert after["remaining_usd"] == {"pilot": "98", "stage": "1198", "project": "2498"}
        regressed = meter.check(ACCOUNTED_UNTIL + 1)
        assert regressed["issues"] == ["wall_clock_regressed"] and not regressed["can_start"]
        assert meter.check(ACCOUNTED_UNTIL + 5000)["issues"] == ["wall_clock_regressed"]
    finally:
        meter.close()
    # Review G2-02: the journal line also names the budget line (null outside diagnostics);
    # the in-memory event stays gate.s2_cost's exact compute_interval.
    assert [json.loads(line) for line in (tmp_path / "cost.jsonl").read_text().splitlines()] == [
        {**event, "line_id": None}]


# --------------------------------------------------------------------- review G2 regressions

CREDIT_95 = [{"event_id": "open", "kind": "opening_credit", "amount_usd": "100",
              "as_of_epoch": ACCOUNTED_UNTIL - 86_400},
             {"event_id": "snap", "kind": "credit_snapshot", "credit_usd": "5",
              "at_epoch": ACCOUNTED_UNTIL}]
SETTLED_95 = [{"event_id": "settle", "kind": "settlement_final", "amount_usd": "95",
               "at_epoch": ACCOUNTED_UNTIL, "evidence_ref": "invoice-ref"}]


@pytest.mark.parametrize("prior", [CREDIT_95, SETTLED_95], ids=["credit", "settlement"])
def test_snapshot_whose_usage_absorbs_run_segments_is_refused(base, prior):
    """G2-01: s2_cost takes max(per-resource, credit consumption, settlement), so $95 of
    pre-run usage encoded as credit consumption or a final settlement would absorb this
    run's own segments (repro: all ten 3,600 s units ran, true pilot usage 107 > cap 100).
    Such a snapshot is refused before any effect; ``prior_usage`` is the additive form
    (``test_headroom_shrinks_unit_by_unit_and_stops_before_a_unit`` is the control)."""
    packet = make_packet(base)
    host = FakeHost(base / "journal")
    summary = run_units(packet, authority=authority(packet), run_config=RUN_CONFIG,
                        journal_dir=base / "journal", runner=host, port_open_fn=host.port_open,
                        clock=Clock(), budget=budget_ok(packet, prior=prior), host_env=HOST_ENV,
                        **run_kw(packet))
    assert summary["run_outcome"] == "refused"
    assert codes(summary) == {"non_additive_prior_event"}
    paths = {item["path"] for item in summary["issues"]}
    assert paths == {f"budget.prior_events[{len(prior) - 1}]"}
    assert host.calls == [] and not (base / "journal").exists()


def test_opening_credit_alone_stays_additive(base):
    """Control for G2-01: an opening credit without a credit snapshot adds no usage."""
    packet = make_packet(base)
    issues, context = validate_budget(budget_ok(packet, prior=CREDIT_95[:1]), packet,
                                      worst_case_s=WORST_CASE_UNIT_S)
    assert issues == [] and context is not None


def test_actual_budget_needs_an_actual_quote_and_recorded_prior_references(base):
    """G2-03: the nested quote's scope must be the packet's; an actual budget refuses a
    SYNTHETIC prior-usage reference."""
    packet = make_packet(base / "a", scope="actual")
    budget = budget_ok(packet)
    assert validate_budget(budget, packet, worst_case_s=WORST_CASE_UNIT_S)[0] == []
    budget["quote"]["scope"] = "synthetic"
    issues, context = validate_budget(budget, packet, worst_case_s=WORST_CASE_UNIT_S)
    assert {item["code"] for item in issues} == {"quote_scope_mismatch"} and context is None
    synthetic_ref = budget_ok(packet, prior=[prior_usage("1", "pilot")])
    issues, _ = validate_budget(synthetic_ref, packet, worst_case_s=WORST_CASE_UNIT_S)
    assert {item["code"] for item in issues} == {"synthetic_reference_in_actual_budget"}
    recorded = budget_ok(packet, prior=[{**prior_usage("1", "pilot"), "decision_ref": "D-020"}])
    assert validate_budget(recorded, packet, worst_case_s=WORST_CASE_UNIT_S)[0] == []
    # A synthetic packet still meters on a synthetic quote; an actual quote is refused there.
    plain = make_packet(base / "s")
    mixed = budget_ok(plain)
    mixed["quote"]["scope"] = "actual"
    issues, _ = validate_budget(mixed, plain, worst_case_s=WORST_CASE_UNIT_S)
    assert {item["code"] for item in issues} == {"quote_scope_mismatch"}


def test_real_run_meters_on_a_quote_of_its_own_scope(base):
    """G2-03: the suite's real-runner positive case now carries an actual quote."""
    packet, _readiness, _grants, auth, run = real_case(base)
    clock = Clock()
    host = RealHost(base / "journal", clock=clock)
    summary, _events, _ = execute(base, packet, host=host, clock=clock, run=run, auth=auth)
    assert summary["run_outcome"] == "completed"
    decision = summary["unit_records"][0]["steps"][0]["decision"]
    assert decision["ledger_status"] == "passed"
    budget = budget_ok(packet)
    assert budget["quote"]["scope"] == "actual" == summary["scope"]


def _diagnostic_run(base, line, prior_spend):
    packet = make_packet(base, diagnostics=[(0, {"budget_line": line})])
    budget = budget_ok(packet, prior_line_spend_usd={line["line_id"]: prior_spend})
    summary, _events, _ = execute(base, packet, budget=budget, clock=Clock(), rollout_s=6000,
                                  run=run_kw(packet, unit_selection="diagnostics_only"))
    return summary


def test_diagnostic_line_cap_is_cumulative_across_runs(base):
    """G2-02: line cap $3, each diagnostic Compose run 6,000 s. Run 1 spends about 2.0007 USD
    of the line; carried into run 2 as ``prior_line_spend_usd``, 3 - 2.0007 < 2.0433 refuses
    the second diagnostic (before the fix each run restarted the line at $0)."""
    line = {"line_id": "pilot_diagnostics", "cap_usd": "3"}
    first = _diagnostic_run(base / "run1", line, "0")
    [record] = first["diagnostic_unit_records"]
    assert record["outcome"] == "executed"
    detail = record["steps"][0]["decision"]["budget_line"]
    assert detail == {"line_id": "pilot_diagnostics", "cap_usd": "3", "prior_spent_usd": "0",
                      "run_spent_usd": "0", "spent_usd": "0", "remaining_usd": "3"}
    segments = [json.loads(line) for line in (
        base / "run1" / "journal" / "cost-events.jsonl").read_text().splitlines()]
    assert [event["line_id"] for event in segments] == ["pilot_diagnostics", None]
    exact = segment_usd(segments[0])
    spend = first["budget"]["line_spend"]["pilot_diagnostics"]
    assert spend["cap_usd"] == "3" and spend["prior_spent_usd"] == "0"
    assert 0 <= Fraction(spend["run_spent_usd"]) - exact < Fraction(1, 10**50)
    assert spend["cumulative_spent_usd"] == spend["run_spent_usd"]
    assert Fraction(2) < exact < Fraction(3) - Fraction(7356, 3600) + Fraction(2)
    second = _diagnostic_run(base / "run2", line, spend["cumulative_spent_usd"])
    [record] = second["diagnostic_unit_records"]
    assert record["outcome"] == "not_started"
    assert record["steps"][0]["issues"] == ["diagnostic_budget_line_exhausted"]
    detail = record["steps"][0]["decision"]["budget_line"]
    assert detail["prior_spent_usd"] == spend["cumulative_spent_usd"]
    assert Fraction(detail["remaining_usd"]) == 3 - Fraction(spend["cumulative_spent_usd"])
    assert second["budget"]["line_spend"]["pilot_diagnostics"]["cumulative_spent_usd"] == (
        spend["cumulative_spent_usd"])


def test_escaping_recorder_failure_keeps_the_diagnostic_segment_on_its_line(base,
                                                                           monkeypatch):
    """G2R: after Compose ran (3,600 s), the diagnostic ledger's terminal write fails and the
    recorder's fallback cancelled write fails again, so OSError escapes ``_Run.unit``. The
    unit's segment must stay on its budget line (repro: closed by the tail with line_id null,
    so ``cumulative_spent_usd`` read '0' and under-counted the cap in the next run)."""
    line = {"line_id": "pilot_diagnostics", "cap_usd": "5"}
    packet = make_packet(base, diagnostics=[(0, {"budget_line": line})])
    module = sys.modules["scripts.s2_run_units"]
    original = module._Run.emit

    def emit(self, event):
        if self.target == "diagnostic" and event["type"] != "started":
            raise OSError("ledger write failed")
        return original(self, event)

    monkeypatch.setattr(module._Run, "emit", emit)
    with pytest.raises(OSError, match="ledger write failed"):
        execute(base, packet, rollout_s=3600)
    summary = json.loads((base / "journal" / "run-summary.json").read_text())
    segments = cost_events(base)
    assert segments[0]["line_id"] == "pilot_diagnostics"
    assert [event["line_id"] for event in segments[1:]] == [None]
    exact = segment_usd(segments[0])
    assert exact >= Fraction(120, 100)  # the 3,600 s Compose run is in this segment
    spend = summary["budget"]["line_spend"]["pilot_diagnostics"]
    assert 0 <= Fraction(spend["cumulative_spent_usd"]) - exact < Fraction(1, 10**50)
    assert spend["cumulative_spent_usd"] == spend["run_spent_usd"]
    assert summary["status"] != "passed" and summary["status"] == "blocked"
    assert summary["run_outcome"] == "interrupted"
    assert "run_outcome:interrupted" in summary["status_reasons"]


@pytest.mark.parametrize("spend, code", [
    ({}, "invalid_budget"),  # the declared line has no carried-forward spend
    ({"pilot_diagnostics": "0", "other": "0"}, "invalid_budget"),
    ({"pilot_diagnostics": 0}, "invalid_budget"),
    ({"pilot_diagnostics": "-1"}, "invalid_budget"),
    (None, "invalid_budget"),
])
def test_prior_line_spend_has_no_default(base, spend, code):
    packet = make_packet(base, diagnostics=[(0, {})])
    budget = budget_ok(packet, prior_line_spend_usd=spend)
    issues, context = validate_budget(budget, packet, worst_case_s=WORST_CASE_UNIT_S)
    assert {item["code"] for item in issues} == {code} and context is None
    assert any(item["path"] == "budget.prior_line_spend_usd" for item in issues)


def test_packet_without_diagnostics_takes_an_empty_line_spend(base):
    packet = make_packet(base)
    issues, _ = validate_budget(budget_ok(packet, prior_line_spend_usd={"x": "0"}), packet,
                                worst_case_s=WORST_CASE_UNIT_S)
    assert {item["code"] for item in issues} == {"invalid_budget"}
    old = budget_ok(packet, schema_version=1)
    old.pop("prior_line_spend_usd")
    issues, _ = validate_budget(old, packet, worst_case_s=WORST_CASE_UNIT_S)
    assert {item["code"] for item in issues} == {"invalid_budget"}


def test_diagnostic_outputs_are_marked_and_refused_by_actual_extraction(base):
    """G2-04: a diagnostic unit record has its own kind and role, the diagnostic ledger job
    its own phase, and gate.s2_actual_extract.extract_unit refuses the record structurally
    (repro: it accepted it and emitted an observation row)."""
    from gate.s2_actual_extract import extract_unit

    packet, _readiness, grants, auth, run = real_case(base, diagnostics=[(0, {})])
    run.update(decisions_md=decisions_md(("D-901", [grants[0]],
                                          ["engineering_only_diagnostic: true"])),
               readiness_report=None, unit_selection="diagnostics_only")
    clock = Clock()
    host = RealHost(base / "journal", clock=clock)
    summary, _events, _ = execute(base, packet, host=host, clock=clock, run=run, auth=auth)
    assert summary["run_outcome"] == "completed"
    record = json.loads((base / "journal/diagnostics/000/unit-record.json").read_text())
    assert record["artifact_kind"] == "s2_diagnostic_unit_record"
    assert record["role"] == "diagnostic_qualification"
    assert record["inference_input"] is False and record["diagnostic_id"] == "camera-check"
    [job] = summary["diagnostic_ledger"]["jobs"]
    assert job["phase"] == "diagnostic_qualification"
    unit = packet["diagnostic_units"][0]
    manifest = json.loads(Path(unit["capture_dir"], "run-dir-manifest.json").read_text())
    receipts = [{k: item[k] for k in ("path", "sha256", "size_bytes")}
                for item in manifest["files"]]
    with pytest.raises(ValueError, match="unit_record"):
        extract_unit(unit["compose"]["cwd"], receipts, record, summary["diagnostic_ledger"],
                     scope="actual")
    # Pilot unit records keep the extractor's exact shape (no role key added).
    pilot = make_packet(base / "pilot")
    pilot_summary, _, _ = execute(base / "pilot", pilot)
    assert pilot_summary["unit_records"][0]["artifact_kind"] == "s2_unit_record"
    assert "role" not in pilot_summary["unit_records"][0]


def test_fixture_runner_refuses_an_actual_scope_packet(base):
    """G2-06: a fixture runner could mint actual-scope unit records from a SYNTHETIC
    authority without any decision log."""
    packet = make_packet(base, scope="actual")
    found, summary = refused_codes(base, packet, authority(packet), run_kw(packet),
                                   host_class=FakeHost)
    assert found == {"fixture_runner_actual_scope_refused"}
    assert summary["authority"]["mode"] == "fixture_runner"


def _cli_paths(base, packet):
    paths = {}
    for name, payload in (("packet", packet), ("authority", authority(packet)),
                          ("config", RUN_CONFIG), ("budget", budget_ok(packet))):
        paths[name] = base / f"{name}.json"
        paths[name].write_text(json.dumps(payload))
    return [str(paths["packet"]), "--authority", str(paths["authority"]), "--run-config",
            str(paths["config"]), "--budget-file", str(paths["budget"]), "--journal-dir",
            str(base / "journal")]


@pytest.mark.parametrize("extra, code", [
    (["--unit-selection", "diagnostics_only", "--clock-uncertainty-us", "250"],
     "no_declared_diagnostics"),
    (["--unit-selection", "all", "--clock-uncertainty-us", "-5"], "clock_uncertainty_unset"),
])
def test_dry_run_applies_the_execute_input_checks(base, capsys, extra, code):
    """G2-05: the dry run said 'validated' (exit 0) for inputs --execute refuses."""
    packet = make_packet(base)
    args = _cli_paths(base, packet) + extra
    host = FakeHost(base / "journal")
    assert main(args, runner=host, port_open_fn=host.port_open, clock=Clock(),
                environ=HOST_ENV) == 2
    out = capsys.readouterr().out
    assert f"refused: {code}" in out and "validated" not in out
    assert main(args + ["--execute"], runner=host, port_open_fn=host.port_open, clock=Clock(),
                environ=HOST_ENV) == 2
    assert host.calls == [] and not (base / "journal").exists()


def test_dry_run_of_a_real_runner_says_when_authority_was_not_verified(base, capsys):
    """G2-05: without --decisions a real-runner dry run lists the plan but never says
    'validated'; every other refusal (here none) would still exit 2."""
    packet, _readiness, _grants, auth, _run = real_case(base)
    args = _cli_paths(base, packet) + ["--unit-selection", "all",
                                       "--clock-uncertainty-us", "250"]
    Path(args[2]).write_text(json.dumps(auth))
    sentinel = RefusingRunner()
    assert main(args, runner=sentinel, port_open_fn=sentinel.port_open) == 2
    out = capsys.readouterr().out
    assert "authority NOT verified" in out and "(pass --execute)" not in out
    assert "pilot_job 0:" in out
    assert sentinel.used == [] and not (base / "journal").exists()
