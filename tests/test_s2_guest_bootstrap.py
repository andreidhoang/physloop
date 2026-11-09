"""Agent-authored known answers for the guest bootstrap runner (CPU-09; R2 phases A-E).

Every command goes through a fake runner; the only real subprocess is the Python interpreter in
the ``subprocess_runner`` tests. No docker, hf, uv, nvidia tool, provider or network is used.
Secrets are synthetic canaries in temporary files.

Runner-mechanics tests use hand-built stages, so they inject a ``compiler`` that returns the plan
under test; the CLI never exposes that seam and the authority tests recompile for real.
"""

import errno
import inspect
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_s2_bootstrap import ALL_KEYS, action
from test_s2_runtime import fixture_inventory

from gate.s2_bootstrap import RunResult, build_execution_plan
from gate.s2_identity import identity_hash
from scripts import s2_guest_bootstrap as guest

ENV_DUMP = ("import os,sys; sys.stdout.write(','.join(sorted(os.environ))); "
            "sys.stdout.write('|' + sys.stdin.read())")
HF_CANARY = "synthetic-hf-canary-0123456789abcdef"
NGC_CANARY = "synthetic-ngc-canary-fedcba9876543210"
PARENT_CANARY = "parent-env-canary-should-never-reach-a-stage"
ENVIRON = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HF_TOKEN": PARENT_CANARY,
           "AWS_SECRET_ACCESS_KEY": PARENT_CANARY, "HOME": "/home/operator"}


class FakeRunner:
    """Scripted runner: maps argv[0] to (returncode, stdout, side effect)."""

    def __init__(self, script=None):
        self.script = script or {}
        self.calls = []

    def __call__(self, argv, *, cwd, env, stdin, timeout_s):
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(env), "stdin": stdin,
                           "timeout_s": timeout_s})
        action_ = self.script.get(argv[0], (0, b"ok\n"))
        if callable(action_):
            return action_(argv, env, stdin)
        returncode, stdout = action_[:2]
        if isinstance(returncode, BaseException):
            raise returncode
        return RunResult(returncode, stdout, b"", False)


def stage(stage_id, argv, **overrides):
    base = {
        "id": stage_id, "phase": "A", "purpose": "test", "source_anchor": "R2 A1 test",
        "argv": argv, "cwd": "/", "env_allowlist": ["LANG", "LC_ALL"],
        "env_set": {"PATH": "/usr/bin:/bin", "HOME": "/root"}, "secret_env": {},
        "stdin_secret": None, "requires": ["host_setup"], "expected_outputs": [],
        "idempotency_check": None, "cleanup_argv": None, "timeout_s": 30, "level": "L2",
        "records_first_delivery": False, "reviewed_script": False, "attempt_scoped": False,
    }
    if argv is None:
        base.update(cwd=None, env_allowlist=[], env_set={})
    base.update(overrides)
    return base


# Stand-in inventory for hand-built plans; its hash binds the plan like a real inventory and its
# kind is the registered s2_runtime inventory kind the hand-built plans name (A2X).
STUB_INVENTORY = {"artifact_kind": "s2_runtime_inventory", "stub": "hand-built plans"}
UNITS = {"timeout_s": "seconds", "size_bytes": "bytes", "bytes": "bytes",
         "filesystem_floor_bytes": "bytes"}


def seal(plan):
    plan = dict(plan)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = identity_hash(plan)
    return plan


def make_plan(tmp_path, stages, secrets=(), authorizations=None):
    root = tmp_path / "work"
    root.mkdir(exist_ok=True)
    record = action("A")
    plan = {
        "artifact_kind": "s2_bootstrap_plan", "schema_version": 1, "scope": "synthetic",
        "compiled": True, "producer": {"module": "test", "source_sha256": None},
        "inputs": {"inventory_sha256": identity_hash(STUB_INVENTORY),
                   "action_record_sha256": identity_hash(record)},
        "phases": ["A"], "roots": {"work": str(root)},
        "authorizations": authorizations if authorizations is not None else {
            "host_setup": ALL_KEYS["host_setup"], "acquire:teacher": ALL_KEYS["acquire:teacher"]},
        "secrets_required": sorted(secrets), "stages": stages, "issues": [],
        "execution_status": "qualify_on_host", "level": "L2", "units": dict(UNITS),
        "compiler": {"inventory_kind": "s2_runtime",
                     "function": "gate.s2_bootstrap.build_execution_plan"},
    }
    return seal(plan), record, root


def secret_file(tmp_path, name, value, mode=0o600):
    path = tmp_path / f"{name}.secret"
    path.write_text(value + "\n")
    path.chmod(mode)
    return str(path)


def run(plan, record, tmp_path, runner, secrets=None, resume=False, journal="journal",
        inventory=None):
    """Hand-built plans: inject a compiler that reproduces the plan; compiled plans: real."""
    extra = {} if inventory is not None else {
        "compiler": lambda inventory, record, *, phases: plan}
    return guest.run_plan(plan, action_record=record,
                          inventory=STUB_INVENTORY if inventory is None else inventory,
                          runner=runner, journal_dir=str(tmp_path / journal),
                          secret_files=secrets or {}, environ=dict(ENVIRON),
                          clock=lambda: "2026-11-09T00:00:00Z", resume=resume, **extra)


def journal(tmp_path, name="journal"):
    lines = (tmp_path / name / "journal.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_runs_in_order_and_records_outputs(tmp_path):
    root = tmp_path / "work"

    def writer(argv, env, stdin):
        (root / "out.txt").write_bytes(b"abc")
        return RunResult(0, b"written\n", b"", False)

    rule = {"kind": "file", "root": "work", "path": str(root / "out.txt"), "size_bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_check", ["checker"], expected_outputs=[
            {"kind": "stdout_regex", "pattern": "^ok", "required": True}]),
        stage("S2_write", ["writer"], expected_outputs=[rule]),
        stage("S3_archive", None, expected_outputs=[
            {"kind": "archive_file", "root": "work", "path": str(root / "out.txt"),
             "name": "out.txt"}]),
    ])
    runner = FakeRunner({"writer": writer})
    result = run(plan, record, tmp_path, runner)
    assert result["status"] == "completed", result
    assert [c["argv"] for c in runner.calls] == [["checker"], ["writer"]]
    assert [s["outcome"] for s in result["stages"]] == ["completed"] * 3
    events = [(e["event"], e.get("stage_id")) for e in journal(tmp_path)]
    assert events == [("run_started", None), ("stage_started", "S1_check"),
                      ("stage_completed", "S1_check"), ("stage_started", "S2_write"),
                      ("stage_completed", "S2_write"), ("stage_completed", "S3_archive"),
                      ("run_finished", None)]
    assert (tmp_path / "journal" / "archive" / "out.txt").read_bytes() == b"abc"
    assert (tmp_path / "journal" / "stages" / "S1_check.1.stdout").read_bytes() == b"ok\n"
    completed = [e for e in journal(tmp_path) if e["event"] == "stage_completed"]
    assert completed[1]["rules"][0]["recorded"]["size_bytes"] == 3
    assert all(e["plan_sha256"] == plan["plan_sha256"] for e in journal(tmp_path))


def test_environment_is_allowlisted_and_secrets_come_only_from_files(tmp_path):
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_plain", ["plain"]),
        stage("S2_download", ["downloader"], secret_env={"HF_TOKEN": "hf_token"},
              requires=["host_setup", "acquire:teacher"]),
    ], secrets=["hf_token"])
    runner = FakeRunner()
    secrets = {"hf_token": secret_file(tmp_path, "hf", HF_CANARY)}
    assert run(plan, record, tmp_path, runner, secrets)["status"] == "completed"
    plain, download = runner.calls
    assert plain["env"] == {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin", "HOME": "/root"}
    assert download["env"] == {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin", "HOME": "/root",
                               "HF_TOKEN": HF_CANARY}
    for call in runner.calls:
        assert PARENT_CANARY not in json.dumps(call)
    text = (tmp_path / "journal" / "journal.jsonl").read_text()
    assert HF_CANARY not in text and PARENT_CANARY not in text


def test_stdin_secret_redaction_and_cleanup_after_later_failure(tmp_path):
    def login(argv, env, stdin):
        return RunResult(0, b"echo " + stdin + b"\nLogin Succeeded\n", stdin, False)

    plan, record, _ = make_plan(tmp_path, [
        stage("S1_login", ["login"], stdin_secret="ngc_api_key", cleanup_argv=["logout"],
              expected_outputs=[{"kind": "stdout_regex", "pattern": "Login Succeeded",
                                 "required": True}]),
        stage("S2_pull", ["puller"]),
        stage("S3_never", ["never"]),
    ], secrets=["ngc_api_key"])
    runner = FakeRunner({"login": login, "puller": (1, b"denied\n")})
    secrets = {"ngc_api_key": secret_file(tmp_path, "ngc", NGC_CANARY)}
    result = run(plan, record, tmp_path, runner, secrets)
    assert result["status"] == "failed"
    assert result["failed_stage"] == "S2_pull" and result["reason"] == "exit_status"
    assert [c["argv"] for c in runner.calls] == [["login"], ["puller"], ["logout"]]
    assert runner.calls[0]["stdin"] == NGC_CANARY.encode()
    assert "NGC" not in json.dumps(runner.calls[0]["env"])
    stdout = (tmp_path / "journal" / "stages" / "S1_login.1.stdout").read_bytes()
    stderr = (tmp_path / "journal" / "stages" / "S1_login.1.stderr").read_bytes()
    assert NGC_CANARY.encode() not in stdout + stderr
    assert b"[REDACTED:ngc_api_key]" in stdout
    events = [e["event"] for e in journal(tmp_path)]
    assert events[-3:] == ["stage_failed", "cleanup", "run_finished"]
    assert NGC_CANARY not in (tmp_path / "journal" / "journal.jsonl").read_text()


def test_stop_at_first_failure_on_verification_timeout_and_missing_tool(tmp_path):
    for name, script, reason in (
        ("verify", {"checker": (0, b"nope\n")}, "verification_failed"),
        ("timeout", {"checker": lambda a, e, s: RunResult(None, b"", b"", True)}, "timeout"),
        ("missing", {"checker": (FileNotFoundError("checker"), b"")}, "runner_error"),
    ):
        plan, record, _ = make_plan(tmp_path, [
            stage("S1_check", ["checker"], expected_outputs=[
                {"kind": "stdout_regex", "pattern": "^ok", "required": True}]),
            stage("S2_after", ["after"]),
        ])
        runner = FakeRunner(script)
        result = run(plan, record, tmp_path, runner, journal=f"j-{name}")
        assert result["status"] == "failed" and result["reason"] == reason, (name, result)
        assert [c["argv"] for c in runner.calls] == [["checker"]]


def test_resume_skips_only_stages_whose_outputs_reverify(tmp_path):
    root = tmp_path / "work"

    def writer(argv, env, stdin):
        (root / "out.txt").write_bytes(b"abc")
        return RunResult(0, b"", b"", False)

    rule = {"kind": "file", "root": "work", "path": str(root / "out.txt"), "size_bytes": 3,
            "sha256": None}
    stages = [
        stage("S1_check", ["checker"]),
        stage("S2_write", ["writer"], expected_outputs=[rule], idempotency_check=[rule]),
        stage("S3_flaky", ["flaky"]),
    ]
    plan, record, _ = make_plan(tmp_path, stages)
    first = FakeRunner({"writer": writer, "flaky": (1, b"")})
    assert run(plan, record, tmp_path, first)["failed_stage"] == "S3_flaky"

    # Without --resume an existing journal is never overwritten.
    refused = run(plan, record, tmp_path, FakeRunner())
    assert refused["status"] == "refused" and refused["issues"][0]["code"] == "journal_exists"

    second = FakeRunner({"writer": writer})
    result = run(plan, record, tmp_path, second, resume=True)
    assert result["status"] == "completed", result
    # S1 has no idempotency check, so it reruns; S2's file re-verifies, so it is skipped.
    assert [c["argv"] for c in second.calls] == [["checker"], ["flaky"]]
    assert [s["outcome"] for s in result["stages"]] == ["completed", "skipped", "completed"]
    attempts = sorted(p.name for p in (tmp_path / "journal" / "stages").glob("S1_check.*.stdout"))
    assert attempts == ["S1_check.1.stdout", "S1_check.2.stdout"]

    # Review A2-F3 (CPU-09): a changed output blocks instead of being re-run or overwritten.
    (root / "out.txt").write_bytes(b"abcd")
    third = FakeRunner({"writer": writer})
    blocked = run(plan, record, tmp_path, third, resume=True)
    assert blocked["status"] == "failed" and blocked["reason"] == "receipt_mismatch"
    assert ["writer"] not in [c["argv"] for c in third.calls]
    assert (root / "out.txt").read_bytes() == b"abcd"


def test_resume_refuses_a_journal_from_another_plan(tmp_path):
    plan, record, _ = make_plan(tmp_path, [stage("S1_check", ["checker"])])
    run(plan, record, tmp_path, FakeRunner({"checker": (1, b"")}))
    other, record2, _ = make_plan(tmp_path, [stage("S1_check", ["checker", "--other"])])
    result = run(other, record2, tmp_path, FakeRunner(), resume=True)
    assert result["status"] == "refused"
    assert result["issues"][0]["code"] == "journal_plan_mismatch"


def test_stage_without_recorded_authority_never_runs(tmp_path):
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_check", ["checker"]),
        stage("S2_gated", ["gated"], requires=["host_setup", "acquire:vlm_config"]),
    ])
    runner = FakeRunner()
    result = run(plan, record, tmp_path, runner)
    assert result["status"] == "refused"
    assert "authorization_missing" in {i["code"] for i in result["issues"]}
    assert runner.calls == []
    assert not (tmp_path / "journal").exists()


def test_refuses_tampered_plan_and_mismatched_action_record(tmp_path):
    plan, record, _ = make_plan(tmp_path, [stage("S1_check", ["checker"])])
    tampered = dict(plan, stages=[stage("S1_check", ["rm", "-rf", "/"])])
    runner = FakeRunner()
    result = run(tampered, record, tmp_path, runner)
    assert result["status"] == "refused" and runner.calls == []
    assert "plan_hash_mismatch" in {i["code"] for i in result["issues"]}
    other_record = dict(record, action_record_ref="SYNTHETIC-other")
    result = run(plan, other_record, tmp_path, runner)
    assert result["status"] == "refused" and runner.calls == []
    assert result["issues"][0]["code"] == "action_record_mismatch"


def test_secret_file_checks(tmp_path):
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_dl", ["downloader"], secret_env={"HF_TOKEN": "hf_token"})],
        secrets=["hf_token"])
    runner = FakeRunner()
    assert run(plan, record, tmp_path, runner)["issues"][0]["code"] == "secret_file_missing"
    loose = {"hf_token": secret_file(tmp_path, "loose", HF_CANARY, mode=0o644)}
    assert run(plan, record, tmp_path, runner, loose)["issues"][0]["code"] == (
        "secret_file_permissions")
    short = {"hf_token": secret_file(tmp_path, "short", "abc")}
    assert run(plan, record, tmp_path, runner, short)["issues"][0]["code"] == "secret_invalid"
    target = tmp_path / "real.secret"
    target.write_text(HF_CANARY)
    target.chmod(0o600)
    os.symlink(target, tmp_path / "link.secret")
    linked = {"hf_token": str(tmp_path / "link.secret")}
    assert run(plan, record, tmp_path, runner, linked)["issues"][0]["code"] == (
        "secret_file_permissions")
    assert runner.calls == []


def test_secret_value_inside_plan_is_refused(tmp_path):
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_dl", ["downloader", HF_CANARY], secret_env={"HF_TOKEN": "hf_token"})],
        secrets=["hf_token"])
    runner = FakeRunner()
    result = run(plan, record, tmp_path, runner,
                 {"hf_token": secret_file(tmp_path, "hf", HF_CANARY)})
    assert result["status"] == "refused" and runner.calls == []
    assert result["issues"][0]["code"] == "secret_in_plan"


def test_compiled_fixture_plan_stops_at_first_unverified_output(tmp_path):
    plan = build_execution_plan(fixture_inventory(), action(), phases=list("ABCDE"))
    assert plan["compiled"] is True
    runner = FakeRunner()
    secrets = {"hf_token": secret_file(tmp_path, "hf", HF_CANARY),
               "ngc_api_key": secret_file(tmp_path, "ngc", NGC_CANARY)}
    result = run(plan, action(), tmp_path, runner, secrets, inventory=fixture_inventory())
    # A0 "succeeds" in the fake; A1's probe result file does not exist on this machine.
    assert result["status"] == "failed" and result["failed_stage"] == "A1_host_probe"
    assert result["reason"] == "verification_failed"
    assert [c["argv"][0] for c in runner.calls] == ["install", "bash"]
    assert runner.calls[1]["argv"][2] == "/data/evidence/s2-bootstrap/host-probe-attempt-1"
    assert runner.calls[1]["env"]["PATH"].startswith("/root/.local/bin:/root/.cargo/bin:")
    assert all(HF_CANARY not in json.dumps(c) for c in runner.calls)
    assert set(ALL_KEYS) <= set(plan["authorizations"])


WIRING = Path(__file__).resolve().parent / "fixtures" / "s2_authority_wiring"


def render_decisions(path, grants, records=()):
    """Render the G1 decision template for one artifact (a fixture file, never the real log)."""
    from string import Template

    lines = [f"  - authorizes: {kind} sha256:{sha}" for kind, sha in grants]
    wrong = [f"  - authorizes: {kind} sha256:{'0' * 64}" for kind, _ in grants]
    text = Template((WIRING / "decisions_template.md").read_text()).substitute(
        grants="\n".join(lines), wrong_grants="\n".join(wrong),
        records="\n".join(f"  - records: {ref}" for ref in records))
    path.write_text(text)
    return path


def plan_decisions(tmp_path, plan, records=None):
    kind = plan["artifact_kind"]
    wanted = guest.authority_request(plan, plan["compiler"]["inventory_kind"])
    assert wanted["artifact_kind"] == kind and wanted["sha256"] == identity_hash(plan)
    return render_decisions(tmp_path / "DECISIONS.md", [(kind, identity_hash(plan))],
                            wanted["records"] if records is None else records)


def write_inputs(tmp_path, inventory, record, phases):
    plan = build_execution_plan(inventory, record, phases=list(phases))
    assert plan["compiled"] is True, plan["issues"]
    for name, value in (("plan", plan), ("record", record), ("inventory", inventory)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    return plan, [str(tmp_path / "plan.json"), "--action-record", str(tmp_path / "record.json"),
                  "--inventory", str(tmp_path / "inventory.json"), "--journal-dir",
                  str(tmp_path / "j")]


def test_main_dry_run_validates_without_running(tmp_path, capsys):
    record = action("B", {"host_setup": ALL_KEYS["host_setup"]})
    plan, args = write_inputs(tmp_path, fixture_inventory(), record, "B")
    runner = FakeRunner()
    code = guest.main(args, runner=runner, environ=dict(ENVIRON))
    assert code == 0 and runner.calls == []
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "validated" and out["stages"] == [s["id"] for s in plan["stages"]]
    assert not (tmp_path / "j").exists()
    assert out["authority_request"] == {"artifact_kind": "s2_bootstrap_plan",
                                        "sha256": identity_hash(plan), "records": []}
    # G1: executing needs a decision entry that binds this plan's hash.
    assert guest.main(args + ["--execute"], runner=runner, environ=dict(ENVIRON)) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "authority_required"
    assert runner.calls == [] and not (tmp_path / "j").exists()
    decisions = plan_decisions(tmp_path, plan)
    code = guest.main(args + ["--execute", "--decision-id", "D-032", "--decisions",
                              str(decisions)], runner=runner, environ=dict(ENVIRON))
    # The fake clone creates nothing, so B1b's HEAD rule fails: the run stops there.
    assert code == 2 and [c["argv"][:2] for c in runner.calls[:2]] == [
        ["git", "clone"], ["git", "-C"]]
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "failed" and out["failed_stage"] == "B1b_checkout"


def test_main_rejects_duplicate_keys_and_relative_journal(tmp_path, capsys):
    record = action("B", {"host_setup": ALL_KEYS["host_setup"]})
    plan, args = write_inputs(tmp_path, fixture_inventory(), record, "B")
    text = json.dumps(plan)
    (tmp_path / "plan.json").write_text(text[:-1] + ', "compiled": true}')
    assert guest.main(args, runner=FakeRunner(), environ={}) == 2
    (tmp_path / "plan.json").write_text(text)
    args[-1] = "relative/journal"
    runner = FakeRunner()
    assert guest.main(args + ["--execute"], runner=runner, environ={}) == 2
    assert runner.calls == []


def test_subprocess_runner_isolates_environment_and_kills_on_timeout():
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PHYSLOOP_X": "1"}
    result = guest.subprocess_runner(
        [sys.executable, "-I", "-c", ENV_DUMP],
        cwd="/", env=env, stdin=b"piped", timeout_s=30)
    assert result.returncode == 0 and result.timed_out is False
    names, piped = result.stdout.decode().split("|")
    assert "PHYSLOOP_X" in names.split(",") and "HF_TOKEN" not in names.split(",")
    assert piped == "piped"
    slow = guest.subprocess_runner([sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                                   cwd="/", env=env, stdin=None, timeout_s=0.5)
    assert slow.timed_out is True and slow.returncode is None


def test_real_runner_has_no_shell_and_is_reached_only_without_an_injected_runner():
    source = Path(guest.__file__).read_text()
    # The real runner is only reachable from main(..., --execute) without an injected runner.
    assert source.count("subprocess_runner") >= 2
    assert "shell=True" not in source


# --- review A2 regressions --------------------------------------------------------------


def test_resealed_plan_cannot_widen_or_rewrite_the_records_authority(tmp_path):
    """Review A2-F1: the runner recompiles from the inventory and action record; an edited,
    re-sealed plan (added authority, dropped requires, injected argv) never runs."""
    record = action("B", {"host_setup": ALL_KEYS["host_setup"]})
    inventory = fixture_inventory()
    plan = build_execution_plan(inventory, record, phases=["B"])
    assert plan["compiled"] is True
    gated = dict(deepcopy(plan["stages"][0]), id="E1_teacher_download", phase="E", argv=[
        "uv", "run", "--no-sync", "hf", "download", "nvidia/Alpamayo-1.5-10B"],
        requires=["acquire:teacher", "host_setup", "terms:teacher"], expected_outputs=[],
        idempotency_check=None)
    injected = deepcopy(plan)
    injected["stages"][0]["argv"] = ["rm", "-rf", "/data"]
    variants = {
        "added_authority": (seal(dict(plan, stages=[gated], phases=["B", "E"], authorizations={
            **plan["authorizations"], "acquire:teacher": "forged", "terms:teacher": "forged"})),
            {"phase_not_authorized", "authorization_not_recorded", "plan_not_reproducible"}),
        "requires_dropped": (seal(dict(plan, stages=[dict(gated, requires=["host_setup"])])),
                             {"stage_phase_not_in_plan", "plan_not_reproducible"}),
        "argv_injected": (seal(injected), {"plan_not_reproducible"}),
    }
    for label, (forged, expected) in variants.items():
        assert guest.validate_plan(forged) == [], label
        runner = FakeRunner()
        result = guest.run_plan(forged, action_record=record, inventory=inventory, runner=runner,
                                journal_dir=str(tmp_path / label), secret_files={},
                                environ=dict(ENVIRON), clock=lambda: "t")
        assert result["status"] == "refused", (label, result)
        assert runner.calls == [] and not (tmp_path / label).exists(), label
        assert {i["code"] for i in result["issues"]} <= expected, (label, result["issues"])
        assert result["reason"] in expected, label
    # A different inventory than the one the plan was compiled from is refused before recompile.
    other = deepcopy(inventory)
    other["toolchain"]["rust_toolchain"] = "1.90.0"
    result = guest.run_plan(plan, action_record=record, inventory=other, runner=FakeRunner(),
                            journal_dir=str(tmp_path / "other"), secret_files={},
                            environ=dict(ENVIRON), clock=lambda: "t")
    assert result["reason"] == "inventory_mismatch"
    # The genuine plan runs.
    runner = FakeRunner()
    result = guest.run_plan(plan, action_record=record, inventory=inventory, runner=runner,
                            journal_dir=str(tmp_path / "genuine"), secret_files={},
                            environ=dict(ENVIRON), clock=lambda: "t")
    assert result["status"] == "failed" and runner.calls[0]["argv"][:2] == ["git", "clone"]


def test_compiler_seam_is_not_reachable_from_the_cli():
    assert "compiler" not in inspect.signature(guest.main).parameters
    assert "compiler" not in inspect.getsource(guest.main)


def actual_inputs():
    """Fixture with non-synthetic scope and refs (test values; no real decision is named)."""
    inventory = fixture_inventory()
    inventory["scope"] = "actual"
    for item in inventory["images"] + inventory["assets"]:
        if item["authorization_ref"]:
            item["authorization_ref"] = item["authorization_ref"].replace("SYNTHETIC", "TEST")
    keys = {k: v.replace("SYNTHETIC", "TEST") for k, v in ALL_KEYS.items()}
    record = action("A", {"host_setup": keys["host_setup"]})
    record["action_record_ref"] = "TEST-action-record"
    return inventory, record


def test_real_runner_refuses_fixture_scope_and_synthetic_authority(tmp_path, monkeypatch, capsys):
    """Review A2-F2: without an injected runner, only an actual-scope plan whose authority
    names no SYNTHETIC reference may reach the real subprocess runner."""
    calls = []

    def recorder(argv, **kwargs):
        calls.append(list(argv))
        return RunResult(0, b"", b"", False)

    monkeypatch.setattr(guest, "subprocess_runner", recorder)
    record = action("A", {"host_setup": ALL_KEYS["host_setup"]})
    for scope in ("synthetic", "public_source_fixture"):
        inventory = dict(fixture_inventory(), scope=scope)
        sub = tmp_path / scope
        sub.mkdir()
        _, args = write_inputs(sub, inventory, record, "A")
        assert guest.main(args + ["--execute"], environ=dict(ENVIRON)) == 2
        out = json.loads(capsys.readouterr().out)
        assert out["status"] == "refused" and out["reason"] == "fixture_scope_refused"
    assert calls == []
    inventory, actual_record = actual_inputs()
    sub = tmp_path / "synthetic-ref"
    sub.mkdir()
    synthetic_ref = dict(actual_record, authorizations={"host_setup": ALL_KEYS["host_setup"]})
    _, args = write_inputs(sub, inventory, synthetic_ref, "A")
    assert guest.main(args + ["--execute"], environ=dict(ENVIRON)) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "synthetic_authority_refused"
    assert calls == []
    # An actual plan reaches the runner only under a decision that binds its hash (G1).
    sub = tmp_path / "actual"
    sub.mkdir()
    plan, args = write_inputs(sub, inventory, actual_record, "A")
    assert guest.main(args + ["--execute"], environ=dict(ENVIRON)) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "authority_required"
    assert calls == []
    decisions = ["--decision-id", "D-032", "--decisions", str(plan_decisions(sub, plan))]
    assert guest.main(args + ["--execute", *decisions], environ=dict(ENVIRON)) == 2
    assert json.loads(capsys.readouterr().out)["failed_stage"] == "A1_host_probe"
    assert [c[0] for c in calls] == ["install", "bash"]


def test_resume_blocks_when_unverified_outputs_changed_since_their_receipt(tmp_path):
    """Review A2-F3: rules with unknown size/sha pass statically; the journal receipt does not."""
    root = tmp_path / "work"

    def writer(argv, env, stdin):
        (root / "out.txt").write_bytes(b"abc")
        return RunResult(0, b"", b"", False)

    rule = {"kind": "file", "root": "work", "path": str(root / "out.txt"), "size_bytes": None,
            "sha256": None}
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_write", ["writer"], expected_outputs=[rule], idempotency_check=[rule]),
        stage("S2_flaky", ["flaky"]),
    ])
    assert run(plan, record, tmp_path, FakeRunner({"writer": writer, "flaky": (1, b"")}))[
        "failed_stage"] == "S2_flaky"
    (root / "out.txt").write_bytes(b"xyz")  # same size, different bytes
    second = FakeRunner({"writer": writer})
    result = run(plan, record, tmp_path, second, resume=True)
    assert result["status"] == "failed" and result["failed_stage"] == "S1_write"
    assert result["reason"] == "receipt_mismatch"
    assert second.calls == [] and (root / "out.txt").read_bytes() == b"xyz"
    failed = [e for e in journal(tmp_path) if e["event"] == "stage_failed"][-1]
    assert failed["conflicts"][0]["path"] == str(root / "out.txt")
    assert failed["conflicts"][0]["reason"] == "sha256_mismatch"
    # Still blocked on the next resume: the stage never overwrites the changed bytes.
    third = FakeRunner({"writer": writer})
    assert run(plan, record, tmp_path, third, resume=True)["reason"] == "receipt_mismatch"
    assert third.calls == []
    # A deleted artifact has nothing to overwrite: the stage runs again.
    (root / "out.txt").unlink()
    fourth = FakeRunner({"writer": writer})
    assert run(plan, record, tmp_path, fourth, resume=True)["status"] == "completed"
    assert [c["argv"] for c in fourth.calls] == [["writer"], ["flaky"]]


@pytest.mark.parametrize("failure", ["enospc", "stale_output"])
def test_journal_errors_still_run_cleanup_and_return_a_structured_failure(
        tmp_path, monkeypatch, failure):
    """Review A2-F4: run_plan never raises; queued logout and run_finished always happen."""
    def login(argv, env, stdin):
        return RunResult(0, b"Login Succeeded\n", b"", False)

    plan, record, _ = make_plan(tmp_path, [
        stage("S1_login", ["login"], stdin_secret="ngc_api_key", cleanup_argv=["logout"]),
        stage("S2_pull", ["puller"]),
        stage("S3_never", ["never"]),
    ], secrets=["ngc_api_key"])
    if failure == "enospc":
        original = guest._Journal.write_output

        def write_output(self, stage_id, attempt, suffix, data):
            if stage_id == "S2_pull":
                raise OSError(errno.ENOSPC, "No space left on device")
            return original(self, stage_id, attempt, suffix, data)

        monkeypatch.setattr(guest._Journal, "write_output", write_output)
    else:
        (tmp_path / "journal" / "stages").mkdir(parents=True)
        (tmp_path / "journal" / "stages" / "S2_pull.1.stdout").write_bytes(b"old\n")
    runner = FakeRunner({"login": login})
    secrets = {"ngc_api_key": secret_file(tmp_path, "ngc", NGC_CANARY)}
    result = run(plan, record, tmp_path, runner, secrets)
    assert result["status"] == "failed" and result["failed_stage"] == "S2_pull"
    assert result["reason"] == "internal_error"
    assert [c["argv"] for c in runner.calls] == [["login"], ["puller"], ["logout"]]
    events = [e["event"] for e in journal(tmp_path)]
    assert events[-3:] == ["stage_failed", "cleanup", "run_finished"]
    assert (tmp_path / "journal" / "stages" / "S2_pull.1.stdout").exists() is (
        failure == "stale_output")


def test_journal_append_failure_still_runs_cleanup(tmp_path, monkeypatch):
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_login", ["login"], cleanup_argv=["logout"]), stage("S2_next", ["next"])])
    original = guest._Journal.append

    def append(self, event, **fields):
        if event == "stage_completed":
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(self, event, **fields)

    monkeypatch.setattr(guest._Journal, "append", append)
    runner = FakeRunner()
    result = run(plan, record, tmp_path, runner)
    assert result["status"] == "failed" and result["reason"] == "internal_error"
    assert [c["argv"] for c in runner.calls] == [["login"], ["logout"]]


def test_attempt_scoped_stage_gets_a_fresh_directory_per_attempt(tmp_path):
    """Review A2-F9: a probe that refuses an existing output directory can be retried."""
    root = tmp_path / "work"
    token = "__ATTEMPT__"

    def probe(argv, env, stdin):
        out = Path(argv[1])
        out.mkdir()  # like the real probe: an existing directory would be exit 73
        verdict = "FAIL" if out.name.endswith("-1") else "PASS"
        (out / "result.txt").write_text(verdict + "\n")
        return RunResult(0, b"", b"", False)

    rule = {"kind": "file_regex", "root": "work", "path": f"{root}/probe-{token}/result.txt",
            "pattern": r"\APASS\n\Z"}
    plan, record, _ = make_plan(tmp_path, [
        stage("S1_probe", ["probe", f"{root}/probe-{token}"], expected_outputs=[rule],
              idempotency_check=[rule], attempt_scoped=True)])
    first = FakeRunner({"probe": probe})
    assert run(plan, record, tmp_path, first)["reason"] == "verification_failed"
    second = FakeRunner({"probe": probe})
    assert run(plan, record, tmp_path, second, resume=True)["status"] == "completed"
    assert [c["argv"][1] for c in first.calls + second.calls] == [
        f"{root}/probe-1", f"{root}/probe-2"]
    started = [e for e in journal(tmp_path) if e["event"] == "stage_started"]
    assert [e["argv"][1] for e in started] == [f"{root}/probe-1", f"{root}/probe-2"]
    # A later resume re-verifies the completed attempt's directory and skips the stage.
    third = FakeRunner({"probe": probe})
    result = run(plan, record, tmp_path, third, resume=True)
    assert result["stages"] == [{"id": "S1_probe", "outcome": "skipped"}] and third.calls == []


def test_journal_and_summary_carry_artifact_headers(tmp_path):
    """Review A2-F11."""
    plan, record, _ = make_plan(tmp_path, [stage("S1_check", ["checker"])])
    result = run(plan, record, tmp_path, FakeRunner())
    assert result["artifact_kind"] == "s2_bootstrap_run_summary"
    assert result["schema_version"] == 1 and result["scope"] == "synthetic"
    assert result["producer"]["module"] == "scripts.s2_guest_bootstrap"
    assert len(result["producer"]["source_sha256"]) == 64
    assert result["inputs"]["plan_sha256"] == plan["plan_sha256"]
    started = journal(tmp_path)[0]
    assert started["event"] == "run_started"
    assert started["artifact_kind"] == "s2_bootstrap_journal"
    assert started["schema_version"] == 1 and started["scope"] == "synthetic"
    assert started["producer"] == result["producer"]
    assert started["inputs"] == result["inputs"]
    refused = run(dict(plan, scope="x"), record, tmp_path, FakeRunner(), journal="j2")
    assert refused["artifact_kind"] == "s2_bootstrap_run_summary"


def test_metacharacter_file_name_arrives_as_one_literal_argv_element(tmp_path):
    """Review A2-F10 (CPU-09): no shell evaluation, no side effect."""
    name = "odd name;$(touch pwned)'.json"
    inventory = fixture_inventory()
    inventory["assets"][2]["files"][1]["path"] = name
    plan = build_execution_plan(inventory, action(), phases=list("ABCDE"))
    argv = next(s for s in plan["stages"] if s["id"] == "E2b_processor_cache")["argv"]
    assert name in argv
    result = guest.subprocess_runner(
        [sys.executable, "-I", "-c", "import sys; sys.stdout.write(sys.argv[1])", name],
        cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"}, stdin=None, timeout_s=30)
    assert result.stdout.decode() == name
    assert not (tmp_path / "pwned").exists()


# --- A2X: the one runner executes S4 plans through their registered compiler ---------------


S4_PHASES = ["alpasim", "images", "assets", "convert", "nccl", "record"]
S4_EXECUTED_IDS = [
    "S1_cache_dir", "S2_cache_fresh", "I1_ngc_login", "I2_renderer_pull",
    "I3_renderer_inspect", "I4_ngc_logout", "M1_teacher_download", "M2_cosmos_reason2_cache",
    "C1_convert_release", "N1_nccl_p2p", "R1_alpasim_head", "R2_alpasim_tree",
    "R3_alpasim_venv_freeze",
]


def s4_inputs(tmp_path):
    """The rehearsal's S4 fixture with host paths under tmp_path and file hashes unverified.

    ``host`` (285 GB filesystem floor on this machine) and ``alpagym`` (bytes hashing to the
    real AlpaGym uv.lock pin) cannot be satisfied by a fake, so the executed phases are the six
    others; every one of their stages runs through the fake runner.
    """
    from scripts.rehearse_s4 import (
        synthetic_bootstrap_record,
        synthetic_inventory,
        synthetic_rental_record,
    )

    inventory = synthetic_inventory()
    base = tmp_path / "host"
    inventory["host_paths"] = {name: str(base / name) for name in inventory["host_paths"]}
    for asset in inventory["assets"]:
        for row in asset["files"]:
            row["sha256"] = "unverified"
    return inventory, synthetic_bootstrap_record(), synthetic_rental_record()


def s4_plan(tmp_path, phases=S4_PHASES):
    from teacher_rl import ops

    inventory, record, rental = s4_inputs(tmp_path)
    compiled = ops.build_s4_bootstrap_plan(inventory, record, rental, phases=list(phases))
    assert compiled["compiled"] is True, compiled["issues"]
    return compiled, inventory, record, rental


class S4Host:
    """Fake guest for the S4 stages: dispatches on the whole argv, writes the outputs."""

    def __init__(self, inventory, fail=()):
        from teacher_rl import ops, pins

        self.ops, self.pins = ops, pins
        self.inventory, self.fail = inventory, set(fail)
        self.calls = []
        self.sizes = {}
        for asset in inventory["assets"]:
            for row in asset["files"]:
                size = row["size_bytes"]
                self.sizes[(asset["repository"], row["path"])] = size if type(size) is int else 3

    def __call__(self, argv, *, cwd, env, stdin, timeout_s):
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(env), "stdin": stdin})
        if self.fail & set(argv):
            return RunResult(1, b"", b"failed\n", False)
        return RunResult(0, self.effect(argv), b"", False)

    def effect(self, argv):
        renderer = self.inventory["images"][0]["reference"]
        if argv[0] == "install":
            for path in argv[2:]:
                os.makedirs(path, exist_ok=True)
            return b""
        if argv[0] == "find":
            return b""
        if argv[:2] == ["docker", "login"]:
            return b"Login Succeeded\n"
        if argv[:3] == ["docker", "image", "inspect"]:
            return f'sha256:{"a" * 64} ["{renderer}"] linux/amd64\n'.encode()
        if argv[0] == "docker":
            return b""
        if argv[:5] == ["uv", "run", "--no-sync", "hf", "download"]:
            return self.download(argv[5:])
        if argv[:4] == ["uv", "run", "--no-sync", "python"] and "--output" in argv:
            out = Path(argv[argv.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(
                '{"vlm_name_or_path": "' + self.pins.COSMOS_REASON2["repository"] + '"}')
            return b""
        if argv[:2] == ["git", "-C"]:
            pin = self.pins.UPSTREAM_PINS["training_alpasim"]
            return (pin["tree"] if argv[-1] == "HEAD^{tree}" else pin["commit"]).encode() + b"\n"
        if argv[:3] == ["uv", "pip", "freeze"]:
            return b"torch==2.8.0\n"
        return b"ok\n"

    def download(self, rest):
        repository = rest[0]
        names = rest[1:rest.index("--revision")]
        revision = rest[rest.index("--revision") + 1]
        hf_home = self.inventory["host_paths"]["hf_home"]
        for name in names:
            if "--local-dir" in rest:
                target = Path(rest[rest.index("--local-dir") + 1]) / name
            else:
                target = Path(self.ops._hf_cache_path(hf_home, repository, revision, name))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"A" * self.sizes[(repository, name)])
        return b""


def run_s4(compiled, inventory, record, rental, tmp_path, runner, resume=False,
           journal="journal"):
    secrets = {"hf_token": secret_file(tmp_path, "hf", HF_CANARY),
               "ngc_api_key": secret_file(tmp_path, "ngc", NGC_CANARY)}
    return guest.run_plan(compiled, action_record=record, inventory=inventory,
                          rental_record=rental, runner=runner,
                          journal_dir=str(tmp_path / journal), secret_files=secrets,
                          environ=dict(ENVIRON), clock=lambda: "2026-11-09T00:00:00Z",
                          resume=resume)


def test_s4_plan_executes_end_to_end_with_resume_and_receipt_mismatch(tmp_path):
    compiled, inventory, record, rental = s4_plan(tmp_path)
    assert [s["id"] for s in compiled["stages"]] == S4_EXECUTED_IDS

    # First run: the NCCL check fails; everything before it completed, the logout ran.
    first_runner = S4Host(inventory, fail={"torchrun"})
    result = run_s4(compiled, inventory, record, rental, tmp_path, first_runner)
    assert result["status"] == "failed", result
    assert result["failed_stage"] == "N1_nccl_p2p" and result["reason"] == "exit_status"
    assert [s["outcome"] for s in result["stages"]] == ["completed"] * 9 + ["failed"]
    assert first_runner.calls[-1]["argv"] == ["docker", "logout", "nvcr.io"]  # I1 cleanup
    login = next(c for c in first_runner.calls if c["argv"][:2] == ["docker", "login"])
    assert login["stdin"] == NGC_CANARY.encode()
    hf = next(c for c in first_runner.calls if c["argv"][3:5] == ["hf", "download"]
              and "nvidia/Cosmos-Reason2-8B" in c["argv"])
    assert hf["env"]["HF_TOKEN"] == HF_CANARY
    text = (tmp_path / "journal" / "journal.jsonl").read_text()
    assert HF_CANARY not in text and NGC_CANARY not in text
    started = journal(tmp_path)[0]
    assert started["inputs"]["rental_record_sha256"] == identity_hash(rental)

    # Receipt mismatch: changed teacher bytes (same size) block instead of being re-downloaded.
    snapshot = Path(inventory["host_paths"]["teacher_snapshot"]) / "config.json"
    original = snapshot.read_bytes()
    snapshot.write_bytes(b"B" * len(original))
    blocked_runner = S4Host(inventory)
    blocked = run_s4(compiled, inventory, record, rental, tmp_path, blocked_runner, resume=True)
    assert blocked["status"] == "failed" and blocked["failed_stage"] == "M1_teacher_download"
    assert blocked["reason"] == "receipt_mismatch"
    assert not any(c["argv"][3:5] == ["hf", "download"] for c in blocked_runner.calls)
    assert snapshot.read_bytes() == b"B" * len(original)

    # Restored bytes match the receipt again: resume completes and skips re-verified stages
    # (M1 failed on the blocked resume, so it is no longer complete and runs again).
    snapshot.write_bytes(original)
    second = S4Host(inventory)
    result = run_s4(compiled, inventory, record, rental, tmp_path, second, resume=True)
    assert result["status"] == "completed", result
    outcomes = {s["id"]: s["outcome"] for s in result["stages"]}
    assert list(outcomes) == S4_EXECUTED_IDS
    assert {k for k, v in outcomes.items() if v == "skipped"} == {
        "I1_ngc_login", "I2_renderer_pull", "M2_cosmos_reason2_cache"}
    downloads = [c["argv"][5] for c in second.calls if c["argv"][3:5] == ["hf", "download"]]
    assert downloads == ["nvidia/Alpamayo-1.5-10B"]
    assert snapshot.read_bytes() == original
    assert ["uv", "run", "--no-sync", "torchrun", "--nproc-per-node=2",
            "install/check_nccl_p2p.py"] in [c["argv"] for c in second.calls]


def test_s4_plan_through_the_cli_needs_its_rental_record(tmp_path, capsys):
    compiled, inventory, record, rental = s4_plan(tmp_path)
    for name, value in (("plan", compiled), ("record", record), ("inventory", inventory),
                        ("rental", rental)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    args = [str(tmp_path / "plan.json"), "--action-record", str(tmp_path / "record.json"),
            "--inventory", str(tmp_path / "inventory.json"), "--journal-dir",
            str(tmp_path / "j")]
    runner = FakeRunner()
    assert guest.main(args, runner=runner, environ=dict(ENVIRON)) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "refused" and out["issues"][0]["code"] == "input_missing"
    args += ["--rental-record", str(tmp_path / "rental.json")]
    assert guest.main(args, runner=runner, environ=dict(ENVIRON)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "validated" and out["stages"] == S4_EXECUTED_IDS
    assert runner.calls == []


def reseal(plan, **changes):
    return seal(dict(deepcopy(plan), **changes))


def refused_codes(compiled, tmp_path, label, *, inventory, record, rental):
    runner = FakeRunner()
    result = guest.run_plan(compiled, action_record=record, inventory=inventory,
                            rental_record=rental, runner=runner,
                            journal_dir=str(tmp_path / label), secret_files={},
                            environ=dict(ENVIRON), clock=lambda: "t")
    assert result["status"] == "refused", (label, result)
    assert runner.calls == [] and not (tmp_path / label).exists(), label
    return result["reason"], {i["code"] for i in result["issues"]}


def test_s4_negative_controls_unknown_kind_compiler_mismatch_and_wrong_inventory(tmp_path):
    compiled, inventory, record, rental = s4_plan(tmp_path)
    s4 = {"inventory": inventory, "record": record, "rental": rental}
    s2_record = action("B", {"host_setup": ALL_KEYS["host_setup"]})
    s2_plan = build_execution_plan(fixture_inventory(), s2_record, phases=["B"])
    s2 = {"inventory": fixture_inventory(), "record": s2_record, "rental": None}
    unknown = dict(compiled["compiler"], inventory_kind="s9_runtime")
    swapped = dict(compiled["compiler"], function=s2_plan["compiler"]["function"])
    cases = {
        "unknown_kind": (reseal(compiled, compiler=unknown), s4, "unknown_plan_kind"),
        "compiler_swapped": (reseal(compiled, compiler=swapped), s4, "compiler_mismatch"),
        "kind_relabelled": (reseal(s2_plan, compiler=compiled["compiler"]), s2,
                            "compiler_mismatch"),
        "s4_plan_s2_inventory": (compiled, dict(s4, inventory=fixture_inventory()),
                                 "inventory_kind_mismatch"),
        "s2_plan_s4_inventory": (s2_plan, dict(s2, inventory=inventory),
                                 "inventory_kind_mismatch"),
        "s4_without_rental": (compiled, dict(s4, rental=None), "input_missing"),
        "s2_with_rental": (s2_plan, dict(s2, rental=rental), "unexpected_input"),
        "other_rental": (compiled, dict(s4, rental=dict(rental, region="OTHER")),
                         "rental_record_mismatch"),
        "producer_swapped": (reseal(compiled, producer=s2_plan["producer"]), s4,
                             "compiler_mismatch"),
    }
    for label, (forged, inputs, code) in cases.items():
        reason, _ = refused_codes(forged, tmp_path, label, inventory=inputs["inventory"],
                                  record=inputs["record"], rental=inputs["rental"])
        assert reason == code, (label, reason)
    # An edited, re-sealed S4 plan is caught by recompiling through teacher_rl.ops.
    injected = deepcopy(compiled)
    injected["stages"][0]["argv"] = ["rm", "-rf", "/data"]
    reason, _ = refused_codes(seal(injected), tmp_path, "argv_injected", inventory=inventory,
                              record=record, rental=rental)
    assert reason == "plan_not_reproducible"


# --- G1: hash-bound authority and recorded terms/acquisition refs --------------------------


def gated_inputs(tmp_path):
    """Phase D (NRE login and pull): gated by terms:renderer and acquire:renderer."""
    record = action("D", ALL_KEYS)
    plan, args = write_inputs(tmp_path, fixture_inventory(), record, "D")
    secrets = ["--secret-file", f"ngc_api_key={secret_file(tmp_path, 'ngc', NGC_CANARY)}"]
    return plan, args + secrets


def test_g1_authority_request_names_the_gated_keys_in_key_form(tmp_path, capsys):
    plan, args = gated_inputs(tmp_path)
    assert guest.main(args, runner=FakeRunner(), environ=dict(ENVIRON)) == 0
    request = json.loads(capsys.readouterr().out)["authority_request"]
    assert request == {"artifact_kind": "s2_bootstrap_plan", "sha256": identity_hash(plan),
                       "records": ["acquire:renderer", "terms:renderer"]}
    # The free-text values the plan carries are not what a decision records.
    assert plan["authorizations"]["acquire:renderer"] == "SYNTHETIC-D-0xx NRE pull"


@pytest.mark.parametrize("decision_id, records, reason", [
    ("D-032", None, None),                         # exact hash line and both records
    ("D-019", None, "plan_not_authorized"),        # closed authority (D-020)
    ("D-024", None, "plan_not_authorized"),        # records only, no authorizes line
    ("D-033", None, "plan_not_authorized"),        # wrong hash
    ("D-034", None, "records_missing"),            # hash bound, terms/acquire not recorded
    ("D-032", ["terms:renderer"], "records_missing"),
    ("D-040", None, "plan_not_authorized"),        # no such entry
])
def test_g1_execute_requires_the_plan_hash_and_every_gated_record(tmp_path, capsys,
                                                                   decision_id, records, reason):
    plan, args = gated_inputs(tmp_path)
    decisions = plan_decisions(tmp_path, plan, records)
    runner = FakeRunner()
    code = guest.main(args + ["--execute", "--decision-id", decision_id, "--decisions",
                              str(decisions)], runner=runner, environ=dict(ENVIRON))
    out = json.loads(capsys.readouterr().out)
    if reason is None:
        assert [c["argv"][:2] for c in runner.calls][:1] == [["docker", "login"]]
        assert out["status"] in ("completed", "failed") and out["reason"] != "authority_required"
        return
    assert code == 2 and out["status"] == "refused" and out["reason"] == reason
    assert runner.calls == [] and not (tmp_path / "j").exists()


def test_g1_free_text_refs_and_unreadable_log_never_suffice(tmp_path, capsys):
    plan, args = gated_inputs(tmp_path)
    # A decision that records the free-text values instead of the keys is not a record.
    text = render_decisions(tmp_path / "free.md", [("s2_bootstrap_plan", identity_hash(plan))],
                            ()).read_text().replace(
        "- **D-034 · 2026-11-12 · Synthetic approval without its terms and acquisition "
        "records.**", "- **D-034 · 2026-11-12 · Synthetic approval.** records: SYNTHETIC-D-0xx "
        "NRE pull; terms SYNTHETIC-terms-renderer")
    assert "records: SYNTHETIC-D-0xx NRE pull" in text
    (tmp_path / "free.md").write_text(text)
    for path, reason in ((tmp_path / "free.md", "records_missing"),
                         (tmp_path / "absent.md", "decisions_unreadable")):
        runner = FakeRunner()
        assert guest.main(args + ["--execute", "--decision-id", "D-034", "--decisions",
                                  str(path)], runner=runner, environ=dict(ENVIRON)) == 2
        assert json.loads(capsys.readouterr().out)["reason"] == reason
        assert runner.calls == []


def test_g1_real_decision_log_authorizes_no_bootstrap_plan(tmp_path, capsys):
    _plan, args = gated_inputs(tmp_path)
    for decision_id in ("D-019", "D-024"):
        runner = FakeRunner()
        assert guest.main(args + ["--execute", "--decision-id", decision_id], runner=runner,
                          environ=dict(ENVIRON)) == 2  # default path: docs/DECISIONS.md
        assert json.loads(capsys.readouterr().out)["reason"] == "plan_not_authorized"
        assert runner.calls == []


def test_g1_run_plan_checks_authority_whenever_given_and_for_the_real_runner(tmp_path,
                                                                            monkeypatch):
    # Review G1-11: the "real" runner identity is a refusing fake, so a regression in the
    # scope or secret refusals fails the test instead of shelling out to docker.
    def refuse(*_args, **_kwargs):
        raise AssertionError("real subprocess runner reached")

    monkeypatch.setattr(guest, "subprocess_runner", refuse)
    plan, _ = gated_inputs(tmp_path)
    record, inventory = action("D", ALL_KEYS), fixture_inventory()
    wrong = {"decisions_md": plan_decisions(tmp_path, plan).read_bytes(), "decision_id": "D-033"}
    result = guest.run_plan(plan, action_record=record, inventory=inventory, runner=FakeRunner(),
                            journal_dir=str(tmp_path / "x"), secret_files={},
                            environ=dict(ENVIRON), clock=lambda: "t", authority=wrong)
    assert result["reason"] == "plan_not_authorized"
    result = guest.run_plan(plan, action_record=record, inventory=inventory,
                            runner=guest.subprocess_runner, journal_dir=str(tmp_path / "y"),
                            secret_files={}, environ=dict(ENVIRON), clock=lambda: "t")
    # The real runner is refused before any authority or process: fixture scope first.
    assert result["reason"] == "fixture_scope_refused"
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_g1_s4_plan_needs_its_s4_bootstrap_plan_authority(tmp_path, capsys):
    compiled, inventory, record, rental = s4_plan(tmp_path)
    request = guest.authority_request(compiled, "s4_runtime")
    assert request["artifact_kind"] == "s4_bootstrap_plan"
    assert request["records"] == sorted({k for s in compiled["stages"] for k in s["requires"]
                                         if k.startswith(("terms:", "acquire:"))})
    assert "acquire:cosmos_reason2" in request["records"]
    s2_style = render_decisions(tmp_path / "s2.md", [("s2_bootstrap_plan",
                                                      identity_hash(compiled))],
                                request["records"])
    good = plan_decisions(tmp_path, compiled)
    for path, expected in ((s2_style, "plan_not_authorized"), (good, None)):
        result = guest.run_plan(
            compiled, action_record=record, inventory=inventory, rental_record=rental,
            runner=FakeRunner(), journal_dir=str(tmp_path / f"j-{expected}"),
            secret_files={"hf_token": secret_file(tmp_path, "hf", HF_CANARY),
                          "ngc_api_key": secret_file(tmp_path, "ngc", NGC_CANARY)},
            environ=dict(ENVIRON), clock=lambda: "t",
            authority={"decisions_md": path.read_bytes(), "decision_id": "D-032"})
        if expected is None:
            assert result["status"] != "refused", result
        else:
            assert (result["status"], result["reason"]) == ("refused", expected)
