"""CLI tests for ``python -m teacher_rl`` (B3 item 7): every command is pure or a dry run.

Inputs are synthetic JSON files under ``tmp_path``; ``main`` is called in-process with a
captured stdout. Nothing provisions, launches, signals or touches the network.
"""

import io
import json
import os
import struct
from pathlib import Path

import pytest

from gate.s2_identity import canonical_json_bytes
from gate.tensor_manifest import build_manifest
from scripts.rehearse_s4 import (
    RUN_DIR_NAME,
    synthetic_base_cosmos_toml,
    synthetic_bootstrap_record,
    synthetic_export_chain,
    synthetic_forecast_input,
    synthetic_inventory,
    synthetic_registration,
    synthetic_rental_record,
    synthetic_run_registry,
)
from teacher_rl import __main__ as cli
from teacher_rl.__main__ import main

_HEADER = (b'{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
           b'"b":{"dtype":"U8","shape":[],"data_offsets":[8,9]}}')


def tiny(last=7):
    return struct.pack("<Q", 105) + _HEADER + bytes.fromhex("0000803f000000c0") + bytes([last])


def write(tmp_path, name, value):
    path = tmp_path / name
    path.write_bytes(value if type(value) is bytes else canonical_json_bytes(value))
    return str(path)


def run(argv):
    out = io.StringIO()
    code = main(argv, stdout=out)
    return code, json.loads(out.getvalue())


def test_validate_and_compile(tmp_path):
    reg = write(tmp_path, "reg.json", synthetic_registration())
    code, report = run(["validate-registration", reg])
    assert code == 0 and report["status"] == "passed"
    code, compiled = run(["compile", reg])
    assert code == 0 and compiled["cosmos_toml_overrides"] == {"train.seed": 7}
    bad = synthetic_registration()
    bad["owner"].pop("reward_arm")
    code, report = run(["validate-registration", write(tmp_path, "bad.json", bad)])
    assert code == 1 and report["status"] == "blocked"


def test_bootstrap_plan(tmp_path):
    paths = [write(tmp_path, "inv.json", synthetic_inventory()),
             write(tmp_path, "rec.json", synthetic_bootstrap_record(phases=("host",))),
             write(tmp_path, "rent.json", synthetic_rental_record())]
    args = ["bootstrap-plan", "--inventory", paths[0], "--action-record", paths[1],
            "--rental-record", paths[2]]
    code, plan = run([*args, "--phases", "host"])
    assert code == 0 and plan["compiled"] is True and plan["gpu_count"] == 2
    code, plan = run([*args, "--phases", "host,alpagym"])
    assert code == 1 and plan["compiled"] is False


def test_forecast(tmp_path):
    code, report = run(["forecast", write(tmp_path, "f.json", synthetic_forecast_input())])
    assert code == 0 and report["total_usd"] == "47.7"


def test_launch_plan_routes(tmp_path):
    common = ["--alpagym-checkout", "/data/src/alpagym", "--capture-tree", "/data/tree",
              "--capture-manifest", "/data/tree.json", "--capture-config", "/data/cap.json",
              "--worker-credentials", "inherit_disclosed",
              "--action-authority-ref", "D-999 synthetic",
              "--alpasim-cache-dir", "/data/cache/alpagym-alpasim",
              "--nurec-terms-ref", "SYNTHETIC-terms-nurec",
              "--nurec-acquire-ref", "SYNTHETIC-acquire-nurec"]
    hydra = write(tmp_path, "hydra.json", synthetic_registration(route="hydra_only"))
    code, spec = run(["launch-plan", "--registration", hydra, *common])
    assert code == 0 and spec["argv"][:8] == ["uv", "run", "--no-sync", "--all-packages",
                                              "python", "-B", "-m",
                                              "teacher_rl.capture.host_launch"]
    assert spec["executes"] is False
    h1 = write(tmp_path, "h1.json", synthetic_registration())
    toml = write(tmp_path, "cosmos_config.toml", synthetic_base_cosmos_toml())
    code, spec = run(["launch-plan", "--registration", h1, *common, "--run-dir",
                      f"/data/alpagym-runs/{RUN_DIR_NAME}", "--base-cosmos-toml", toml,
                      "--resolved-config-sha256", "5" * 64])
    assert code == 0 and spec["route"] == "h1_resolved_toml"
    code, spec = run(["launch-plan", "--registration", h1, *common])
    assert code == 1 and spec["argv"] is None
    # no NuRec references -> blocked before HF_TOKEN could reach the host CLI
    code, spec = run(["launch-plan", "--registration", hydra, *common[:-4]])
    assert code == 1 and "nurec_authorization_missing" in {i["code"] for i in spec["issues"]}


def test_evaluation_request_and_aggregation_spec(tmp_path):
    reg = synthetic_registration()
    registry = synthetic_run_registry(reg)
    from scripts.rehearse_s4 import _s2_registration
    shared = ["--s4-registration", write(tmp_path, "reg.json", reg),
              "--run-registry", write(tmp_path, "rr.json", registry), "--scope", "synthetic"]
    requests = []
    for index, run_id in enumerate(registry["run_ids"]):
        audit_inputs, manifest = synthetic_export_chain(index)
        s2 = _s2_registration("tensor_set_sha256:" + manifest["tensor_set_sha256"])
        code, req = run(["evaluation-request", "--manifest",
                         write(tmp_path, f"m{index}.json", manifest), "--run-id", run_id,
                         "--checkpoint-step", "3", "--s2-registration",
                         write(tmp_path, f"s2{index}.json", s2), "--export-audit-inputs",
                         write(tmp_path, f"ea{index}.json", audit_inputs), *shared])
        assert code == 0 and req["status"] == "passed", req
        requests += ["--request", write(tmp_path, f"q{index}.json", req)]
    code, spec = run(["aggregation-spec", *requests, *shared])
    assert code == 0 and spec["registered_run_ids"] == ["run-a", "run-b", "run-c"]
    # read from the registration's owner.run_aggregation slot (B3 re-verification)
    assert (spec["rule"], spec["k"]) == ("all", None) == (
        reg["owner"]["run_aggregation"]["rule"], reg["owner"]["run_aggregation"]["k"])
    # review B3-20: the rule is not a flag; a different registry is refused by the requests
    other = write(tmp_path, "rr2.json", synthetic_run_registry(reg, rule="at_least_k", k=2))
    code, spec = run(["aggregation-spec", *requests, "--s4-registration", shared[1],
                      "--run-registry", other, "--scope", "synthetic"])
    assert code == 1 and {"aggregation_mismatch", "request_registry_mismatch"} <= {
        i["code"] for i in spec["issues"]}
    with pytest.raises(SystemExit):
        main(["aggregation-spec", *requests, *shared, "--rule", "all"], stdout=io.StringIO())


def test_parity_and_export_audit_run_torch_free(tmp_path):
    a = write(tmp_path, "a.json", build_manifest({"m.safetensors": tiny()}, scope="synthetic"))
    b = write(tmp_path, "b.json", build_manifest({"m.safetensors": tiny(8)}, scope="synthetic"))
    tol = write(tmp_path, "t.json", {
        "registration_ref": "experiments/2026-11-09-synthetic-s4.md#parity",
        "atol": 0.0, "rtol": 0.0, "max_ulp": None})
    code, same = run(["parity", "--reference", a, "--candidate", a, "--tolerance", tol])
    assert code == 0 and same["artifact_kind"] == "s4_parity_ladder"
    code, differ = run(["parity", "--reference", a, "--candidate", b, "--tolerance", tol])
    assert differ["rungs"]["bitwise"]["differing"] == ["b"]
    code, audit = run(["export-audit", "--release", a, "--export", b,
                       "--hf-config-is-vlm", "true"])
    # blocked export audits exit 1 (here: release tensors absent from the export unclassified)
    assert audit["artifact_kind"] == "s4_export_audit"
    assert (code, audit["status"]) == (1, "blocked")


def test_input_errors_exit_2(tmp_path):
    dup = write(tmp_path, "dup.json", b'{"a": 1, "a": 2}')
    code, error = run(["validate-registration", dup])
    assert code == 2 and error["artifact_kind"] == "s4_cli_error"
    link = tmp_path / "link.json"
    os.symlink(write(tmp_path, "real.json", synthetic_registration()), link)
    assert run(["compile", str(link)])[0] == 2
    assert run(["forecast", str(tmp_path / "missing.json")])[0] == 2
    with pytest.raises(SystemExit) as exit_info:
        main(["export-audit", "--release", dup, "--export", dup, "--hf-config-is-vlm", "yes"],
             stdout=io.StringIO())
    assert exit_info.value.code == 2


def test_review_b3_16_null_artifact_is_an_input_error_not_a_traceback(tmp_path):
    null = write(tmp_path, "null.json", b"null")
    a = write(tmp_path, "a.json", build_manifest({"m.safetensors": tiny()}, scope="synthetic"))
    code, error = run(["export-audit", "--release", null, "--export", a,
                       "--hf-config-is-vlm", "true"])
    assert code == 2 and error["artifact_kind"] == "s4_cli_error"
    assert "expected a JSON object" in error["error"]
    code, error = run(["forecast", write(tmp_path, "list.json", b"[1, 2]")])
    assert code == 2 and error["status"] == "blocked"


def test_review_b3_16_any_command_exception_exits_2(tmp_path, monkeypatch):
    def boom(args):
        raise KeyError("release")

    monkeypatch.setitem(cli.COMMANDS, "forecast", boom)
    code, error = run(["forecast", write(tmp_path, "f.json", synthetic_forecast_input())])
    assert code == 2 and error["error"] == "KeyError: 'release'"


def test_review_b3_16_non_utf8_argv_is_an_error_json_not_a_traceback(tmp_path):
    # A surrogate-escaped byte (os.fsdecode of b"\xff") reaches the artifact unchanged and
    # cannot be serialized as strict UTF-8: still one error JSON and exit 2.
    reg = synthetic_registration()
    audit_inputs, manifest = synthetic_export_chain(0)
    code, error = run([
        "evaluation-request", "--manifest", write(tmp_path, "m.json", manifest),
        "--s4-registration", write(tmp_path, "reg.json", reg),
        "--run-registry", write(tmp_path, "rr.json", synthetic_run_registry(reg)),
        "--run-id", "run-a\udcff", "--checkpoint-step", "3",
        "--s2-registration", write(tmp_path, "s2.json", {"schema_version": 1}),
        "--scope", "synthetic",
        "--export-audit-inputs", write(tmp_path, "ea.json", audit_inputs)])
    assert code == 2 and error["artifact_kind"] == "s4_cli_error"
    assert error["command"] == "evaluation-request"


def test_review_b3_16_error_text_with_a_surrogate_is_escaped(tmp_path, monkeypatch):
    def boom(args):
        raise ValueError("bad byte \udcff")

    monkeypatch.setitem(cli.COMMANDS, "forecast", boom)
    code, error = run(["forecast", write(tmp_path, "f.json", synthetic_forecast_input())])
    assert code == 2 and error["error"] == "ValueError: bad byte \\udcff"


def test_rehearse_command(tmp_path):
    code, summary = run(["rehearse", "--workdir", str(tmp_path / "rehearsal")])
    assert code == 0 and summary["status"] == "passed"
    assert run(["rehearse", "--workdir", str(tmp_path / "rehearsal")])[0] == 2
    assert run(["rehearse", "--workdir", "relative"])[0] == 2


# ------------------------------------------------------------------- G1 commands


def test_rehearse_passes_only_with_a_passed_decision(tmp_path, monkeypatch):
    import scripts.rehearse_s4 as rehearsal

    real = rehearsal.run_rehearsal

    def blocked_decision(workdir):
        summary = real(workdir)
        assert summary["decision"]["status"] == "passed"
        return dict(summary, decision=dict(summary["decision"], status="blocked"))

    monkeypatch.setattr(rehearsal, "run_rehearsal", blocked_decision)
    code, summary = run(["rehearse", "--workdir", str(tmp_path / "r")])
    assert code == 1 and summary["status"] == "blocked"


def _aggregation(tmp_path):
    from scripts.rehearse_s4 import RUN_IDS, _evaluations, _s2_registration
    from teacher_rl import evaluate

    registration = synthetic_registration()
    registry = synthetic_run_registry(registration)
    requests = []
    for index, run_id in enumerate(RUN_IDS):
        audit_inputs, manifest = synthetic_export_chain(index)
        requests.append(evaluate.build_evaluation_request(
            inference_manifest=manifest, s4_registration=registration, run_registry=registry,
            run_id=run_id, checkpoint_step=3,
            s2_registration=_s2_registration("tensor_set_sha256:"
                                             + manifest["tensor_set_sha256"]),
            scope="synthetic", export_audit_inputs=audit_inputs))
    spec = evaluate.aggregation_spec(requests=requests, s4_registration=registration,
                                     run_registry=registry, scope="synthetic")
    assert spec["status"] == "passed"
    return spec, _evaluations(spec)


def test_decide_and_aggregate_command_matches_the_library_call(tmp_path):
    from teacher_rl import evaluate

    spec, evaluations = _aggregation(tmp_path)
    args = ["decide-and-aggregate", "--spec", write(tmp_path, "agg.json", spec),
            "--evaluations", write(tmp_path, "evals.json", evaluations)]
    code, decided = run(args)
    expected = json.loads(canonical_json_bytes(evaluate.decide_and_aggregate(evaluations,
                                                                             spec)))
    assert code == 0 and decided == expected and decided["status"] == "passed"
    assert sorted(decided["outcomes"]) == ["run-a", "run-b", "run-c"]
    missing = {k: v for k, v in evaluations.items() if k != "run-c"}
    code, decided = run(["decide-and-aggregate", "--spec", args[2], "--evaluations",
                         write(tmp_path, "missing.json", missing)])
    assert code == 1 and decided["status"] == "blocked"
    assert decided["issues"][0]["code"] == "decision_binding_failed"
    code, error = run(["decide-and-aggregate", "--spec", args[2], "--evaluations",
                       write(tmp_path, "list.json", [1])])
    assert code == 2 and error["artifact_kind"] == "s4_cli_error"


def _watch_inputs(tmp_path, scope="actual"):
    from tests.test_s4_capture import launch_decisions, launch_inputs
    from tests.test_s4_watchdog import LIMITS
    from tests.test_s4_watchdog import spec as watch_spec

    launch_spec, registration = launch_inputs(tmp_path, tmp_path / "tree", tmp_path / "m.json",
                                              scope=scope)
    files = {"launch": write(tmp_path, "launch.json", launch_spec),
             "registration": write(tmp_path, "registration.json", registration),
             "decisions": str(launch_decisions(tmp_path, launch_spec, registration))}
    code, spec = run(["watch-spec", "--launch-spec", files["launch"], "--run-id", "run-a",
                      "--roles", write(tmp_path, "roles.json", watch_spec()["roles"]),
                      "--containers", write(tmp_path, "c.json", {
                          "compose_project": "wizard_0", "container_ids": []}),
                      "--limits", write(tmp_path, "limits.json", LIMITS)])
    assert code == 0 and spec["launch_hash"] == launch_spec["launch_hash"]
    assert spec["scope"] == scope and spec["schema_version"] == 2
    files["spec"] = write(tmp_path, "watch.json", spec)
    files["decisions"] = str(watch_decisions(tmp_path, launch_spec, registration, spec))
    return files, spec


def watch_decisions(tmp_path, launch_spec, registration, spec):
    """The launch grants plus the G1-02 line binding this exact watch spec."""
    from gate.s2_identity import identity_hash
    from tests.test_s4_capture import launch_decisions

    return launch_decisions(tmp_path, launch_spec, registration, grants=[
        ("s4_launch_plan", launch_spec["launch_hash"]),
        ("s4_registration", identity_hash(registration)),
        ("s4_watch_spec", identity_hash(spec))])


def _execute(files, decision_id="D-032"):
    return ["watch", "--spec", files["spec"], "--execute", "--launch-spec", files["launch"],
            "--registration", files["registration"], "--decisions", files["decisions"],
            "--decision-id", decision_id, "--progress-dir", "/data/run/physloop_capture"]


def _runtime(calls):
    from tests.test_s4_watchdog import OK_EMPTY, Clock, Processes, Runner

    def runtime(spec, progress_dir):
        calls.append(progress_dir)
        runner = Runner([OK_EMPTY, OK_EMPTY])
        return Processes(dead={104}), (lambda: 5), runner, Clock()
    return runtime


def run_watch(argv, calls):
    out = io.StringIO()
    code = main(argv, stdout=out, watch_runtime=_runtime(calls))
    return code, json.loads(out.getvalue())


def test_watch_dry_run_prints_the_scoped_plan_and_runs_nothing(tmp_path):
    files, spec = _watch_inputs(tmp_path)
    calls = []
    code, plan = run_watch(["watch", "--spec", files["spec"]], calls)
    assert code == 0 and calls == []
    assert plan["artifact_kind"] == "s4_watch_plan" and plan["executes"] is False
    assert plan["container_scope"] == {"compose_project": "wizard_0", "container_ids": []}
    assert plan["process_groups"] == [101, 102, 103, 104, 105]
    assert not any("name=" in " ".join(c) for c in plan["teardown_commands"])
    from gate.s2_identity import identity_hash
    assert plan["execute_requires"]["decision_lines"] == [
        f"authorizes: s4_launch_plan sha256:{spec['launch_hash']}",
        "authorizes: s4_registration sha256:<identity_hash of --registration>",
        f"authorizes: s4_watch_spec sha256:{identity_hash(spec)}",
        "records: acquire:nurec_scenes", "records: terms:nurec_scenes"]
    v1 = dict(spec, schema_version=1)
    code, error = run_watch(["watch", "--spec", write(tmp_path, "v1.json", v1)], calls)
    assert code == 2 and "name pattern" in error["error"]


def test_watch_execute_needs_the_launch_authority_then_tears_down(tmp_path):
    files, _ = _watch_inputs(tmp_path)
    calls = []
    code, report = run_watch(_execute(files), calls)
    assert code == 0 and calls == ["/data/run/physloop_capture"]
    assert report["decision"] == "terminate" and report["reasons"] == ["role_dead:rollout_0"]
    assert report["cleanup_status"] == "verified_clean"
    assert report["handoff_observation"]["kind"] == "work_failed"
    args = _execute(files)
    index = args.index("--registration")
    code, error = run_watch(args[:index] + args[index + 2:], calls)
    assert code == 2 and "--registration" in error["error"] and len(calls) == 1


@pytest.mark.parametrize("decision_id, code_expected", [
    ("D-019", "launch_not_authorized"), ("D-024", "launch_not_authorized"),
    ("D-033", "launch_not_authorized"), ("D-034", "nurec_records_missing")])
def test_watch_execute_refusals_construct_no_runtime(tmp_path, decision_id, code_expected):
    files, _ = _watch_inputs(tmp_path)
    calls = []
    code, refusal = run_watch(_execute(files, decision_id), calls)
    assert code == 1 and calls == []
    assert refusal["artifact_kind"] == "s4_watch_refusal" and refusal["executes"] is False
    assert code_expected in refusal["issues"]


def test_watch_execute_refuses_another_launch_synthetic_scope_and_the_real_log(tmp_path):
    files, spec = _watch_inputs(tmp_path)
    calls = []
    other = dict(spec, launch_hash="f" * 64)
    args = _execute(dict(files, spec=write(tmp_path, "other.json", other)))
    code, refusal = run_watch(args, calls)
    # the edited spec is another launch's and (its hash changed) no longer the authorized one
    assert code == 1 and refusal["issues"] == ["watch_spec_not_for_this_launch",
                                               "watch_spec_not_authorized"]
    real = os.path.join(os.path.dirname(__file__), "..", "docs", "DECISIONS.md")
    code, refusal = run_watch(_execute(dict(files, decisions=real), "D-024"), calls)
    assert code == 1 and "launch_not_authorized" in refusal["issues"]
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    files, _ = _watch_inputs(synthetic, scope="synthetic")
    code, refusal = run_watch(_execute(files), calls)
    assert code == 1 and {"launch_scope_not_actual", "watch_scope_not_actual"} <= set(
        refusal["issues"])
    assert calls == []


# ------------------------------------------------------------------- G1 review regressions


def test_g1_02_watch_spec_naming_another_runs_project_is_refused(tmp_path):
    """Review G1-02: same launch_hash, another run's Compose project -> nothing is removed."""
    from tests.test_s4_watchdog import OK_EMPTY, Clock, Processes, Runner, listing

    files, spec = _watch_inputs(tmp_path)
    other = dict(spec, compose_project="wizard_1")
    runner = Runner([listing(("c" * 64, "wizard_1")), OK_EMPTY, OK_EMPTY, OK_EMPTY])
    built = []

    def runtime(spec_, progress_dir):
        built.append(progress_dir)
        return Processes(dead={104}), (lambda: 5), runner, Clock()

    out = io.StringIO()
    code = main(_execute(dict(files, spec=write(tmp_path, "other.json", other))), stdout=out,
                watch_runtime=runtime)
    refusal = json.loads(out.getvalue())
    assert code == 1 and refusal["artifact_kind"] == "s4_watch_refusal"
    assert refusal["issues"] == ["watch_spec_not_authorized"]
    assert built == [] and runner.calls == []
    ids = dict(spec, compose_project=None, container_ids=["c" * 64])  # IDs are records too
    out = io.StringIO()
    code = main(_execute(dict(files, spec=write(tmp_path, "ids.json", ids))), stdout=out,
                watch_runtime=runtime)
    assert code == 1 and json.loads(out.getvalue())["issues"] == ["watch_spec_not_authorized"]
    assert built == [] and runner.calls == []
    no_line = tmp_path / "no-line"  # the launch grants alone no longer suffice
    no_line.mkdir()
    from tests.test_s4_capture import launch_decisions
    launch = json.loads(Path(files["launch"]).read_text())
    registration = json.loads(Path(files["registration"]).read_text())
    code, refusal = run_watch(_execute(dict(files, decisions=str(launch_decisions(
        no_line, launch, registration)))), built)
    assert code == 1 and refusal["issues"] == ["watch_spec_not_authorized"] and built == []


def test_g1_09_watch_execute_revalidates_the_registration(tmp_path):
    """Review G1-09: a registration validate_registration blocks never authorizes a watch."""
    import copy

    from gate.s2_identity import identity_hash
    from teacher_rl.registration import validate_registration

    files, spec = _watch_inputs(tmp_path)
    launch = json.loads(Path(files["launch"]).read_text())
    bad = dict(json.loads(Path(files["registration"]).read_text()), registration_id="Not A Slug")
    assert validate_registration(bad)["status"] == "blocked"
    edited = copy.deepcopy(launch)
    edited["input_hashes"]["registration"] = identity_hash(bad)  # outside launch_hash
    sub = tmp_path / "bad"
    sub.mkdir()
    files = dict(files, launch=write(sub, "launch.json", edited),
                 registration=write(sub, "registration.json", bad),
                 decisions=str(watch_decisions(sub, edited, bad, spec)))
    calls = []
    code, refusal = run_watch(_execute(files), calls)
    assert code == 1 and refusal["issues"] == ["registration_not_validated"] and calls == []
