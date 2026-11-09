"""Offline CLI contract tests; input files contain invented fixtures only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_alpasim_adapter import SHA, fixture
from test_s2_preflight import manifest

ROOT = Path(__file__).parents[1]


def command(*args):
    result = subprocess.run(
        [sys.executable, "-m", "gate.s2", *map(str, args)],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.fail(f"CLI did not return JSON: stdout={result.stdout!r}; stderr={result.stderr!r}")
    return result, payload


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_preflight_draft_is_json_and_exit_two(tmp_path):
    document = manifest()
    document["status"] = "draft"
    path = write_json(tmp_path / "draft.json", document)
    result, payload = command("preflight", path)
    assert result.returncode == 2
    assert payload["ready_for_planning"] is False
    assert payload["cloud_launch_authorized"] is False
    assert payload["blocking_reasons"]


@pytest.mark.parametrize("admitted", [True, False])
def test_normalize_admits_or_quarantines_entire_fixture(tmp_path, admitted):
    summary, identities, observations = fixture()
    if not admitted:
        observations[0]["completion_validated"] = False
    paths = [
        write_json(tmp_path / f"{name}.json", value)
        for name, value in (
            ("summary", summary),
            ("identities", identities),
            ("observations", observations),
        )
    ]
    result, payload = command(
        "normalize",
        "--summary",
        paths[0],
        "--identities",
        paths[1],
        "--observations",
        paths[2],
        "--arm",
        "A",
        "--policy",
        "teacher-bf16",
        "--observed-evaluator-sha",
        SHA,
    )
    assert result.returncode == (0 if admitted else 2)
    assert payload["admitted"] is admitted
    assert len(payload["records"]) == (1 if admitted else 0)
    assert payload["sidecar_authenticity"] == "not_verified"


@pytest.mark.parametrize("kind", ["malformed", "missing"])
def test_file_errors_are_structured_json(tmp_path, kind):
    path = tmp_path / "input.json"
    if kind == "malformed":
        path.write_text("{this is not json", encoding="utf-8")
    result, payload = command("preflight", path)
    assert result.returncode == 2
    assert payload["error"]
    assert "Traceback" not in result.stderr


def test_cost_table_preserves_historical_arithmetic_without_spend_authorization():
    result, payload = command("cost-table")
    assert result.returncode == 0
    assert payload["spend_authorized"] is False
    pairs = {
        (row["gpu_hours"], row["rate_per_gpu_hour_usd"]): row["compute_cost_usd"]
        for row in payload["rows"]
    }
    assert pairs[(136, 1.80)] == pytest.approx(244.8)
    assert pairs[(340, 2.49)] == pytest.approx(846.6)


# ---------------------------------------------------------------- B2 subcommands
# In-process calls of gate.s2.main (same parser, handlers and exit codes as the module
# entry point) keep these fast; the subprocess tests above cover the process contract.


def run_main(capsys, *args):
    from gate.s2 import main

    code = main([str(arg) for arg in args])
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_max_asl_bytes_default_is_the_capture_bound_with_literal_help():
    from gate.s2 import MAX_ASL_BYTES_HELP, build_parser
    from gate.s2_capture import MAX_ASL_BYTES

    args = build_parser().parse_args(["inspect-capture", "r.json", "--root", "x"])
    assert args.max_asl_bytes == MAX_ASL_BYTES == 2 * 1024 * 1024 * 1024
    assert MAX_ASL_BYTES_HELP == ("streamed, hash-verified; agent technical default (D-018), "
                                  "provisional until the first measured ASL")
    result = subprocess.run([sys.executable, "-B", "-m", "gate.s2", "inspect-capture", "--help"],
                            cwd=ROOT, capture_output=True, text=True, check=False, timeout=10,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert " ".join(MAX_ASL_BYTES_HELP.split()) in " ".join(result.stdout.split())


def test_validate_inventory_accepts_fixture_and_rejects_draft(tmp_path, capsys):
    from test_s2_runtime import fixture_inventory

    path = write_json(tmp_path / "inventory.json", fixture_inventory())
    code, payload = run_main(capsys, "validate-inventory", path)
    assert code == 0 and payload["status"] == "passed"
    draft = json.loads((ROOT / "experiments/S2_SETUP_DRAFT.json").read_text())
    path = write_json(tmp_path / "draft.json", draft["runtime_inventory"])
    code, payload = run_main(capsys, "validate-inventory", path)
    assert code == 2 and payload["status"] == "blocked"


def test_compile_bootstrap_is_a_plan_and_refuses_without_authorization(tmp_path, capsys):
    from test_s2_bootstrap import action
    from test_s2_runtime import fixture_inventory

    inventory = write_json(tmp_path / "inventory.json", fixture_inventory())
    record = write_json(tmp_path / "action.json", action())
    code, payload = run_main(capsys, "compile-bootstrap", inventory, record, "--phases", "A")
    assert code == 0 and payload["compiled"] and payload["stages"]
    assert payload["execution_status"] == "qualify_on_host"
    unauthorized = action(keys=())
    record = write_json(tmp_path / "unauthorized.json", unauthorized)
    code, payload = run_main(capsys, "compile-bootstrap", inventory, record, "--phases", "A",
                             "B", "C", "D", "E")
    assert code == 2 and not payload["compiled"] and payload["stages"] == []


def test_describe_validity_reports_bins_without_admission(tmp_path, capsys):
    from test_s2_diagnostics import single_arm

    ledger, observations = single_arm([0.5, 1.0, 2.0, 4.0])
    path = write_json(tmp_path / "validity.json", {"ledger_report": ledger,
                                                    "observations": observations})
    code, payload = run_main(capsys, "describe-validity", path)
    assert payload["artifact_kind"] == "s2_validity_report"
    assert code == (0 if payload["status"] == "passed" else 2)
    path = write_json(tmp_path / "bad.json", {"ledger_report": ledger})
    code, payload = run_main(capsys, "describe-validity", path)
    assert code == 2 and "observations" in payload["error"]


def test_audit_population_without_rule_blocks_decisions_only(tmp_path, capsys):
    from test_s2_population import manifest

    path = write_json(tmp_path / "population.json", {"manifest": manifest()})
    code, payload = run_main(capsys, "audit-population", path)
    assert payload["artifact_kind"] == "s2_population_audit"
    assert code == (0 if payload["status"] == "passed" else 2)
    assert payload["decisions"]["code"] == "overlap_rule_unregistered"


def test_describe_repeats_is_blocked_until_a_method_is_registered(tmp_path, capsys):
    from test_s2_probes import obs

    rows = [obs(0, rep="r0"), obs(1, rep="r1"), obs(0, rep="r2")]
    path = write_json(tmp_path / "repeats.json", {"observations": rows, "scope": "synthetic"})
    code, payload = run_main(capsys, "describe-repeats", path)
    assert code == 2 and payload["status"] == "blocked"
    assert payload["descriptive_status"] == "passed"


def test_compare_probes_reads_receipted_payloads(tmp_path, capsys):
    import hashlib

    from test_s2_probes import F32, probe

    layout = {"action": {"step": 0, "role": "x", "units": "m"}}
    first = probe([("action", "F32", [2], F32(1.0, 2.0))], layout)
    sides = {}
    for name, artifact in (("a", first), ("b", first)):
        (tmp_path / f"{name}.bin").write_bytes(artifact["payload"])
        sides[name] = {"receipt": artifact["receipt"], "payload": {
            "path": f"{name}.bin", "size_bytes": len(artifact["payload"]),
            "sha256": hashlib.sha256(artifact["payload"]).hexdigest()}}
    path = write_json(tmp_path / "probes.json", {**sides, "kind": "fixed_input_policy"})
    code, payload = run_main(capsys, "compare-probes", path, "--root", tmp_path)
    assert payload["outcome"] == "identical"
    assert code == (0 if payload["status"] == "passed" else 2)
    (tmp_path / "b.bin").write_bytes(b"changed")
    code, payload = run_main(capsys, "compare-probes", path, "--root", tmp_path)
    assert code == 2 and "error" in payload


def test_summarize_telemetry_known_answer(tmp_path, capsys):
    from test_s2_telemetry import CLOCK, S, allocation, rollout

    request = {"samples": [], "allocation": allocation(),
               "rollouts": [rollout(i) for i in range(4)], "clock_metadata": CLOCK,
               "max_sample_gap_us": 2 * S, "scope": "synthetic"}
    code, payload = run_main(capsys, "summarize-telemetry",
                             write_json(tmp_path / "telemetry.json", request))
    assert payload["artifact_kind"] == "s2_telemetry_summary"
    assert code == (0 if payload["status"] == "passed" else 2)


def test_run_sensitivity_and_bad_design(tmp_path, capsys):
    from test_s2_power import continuous, design, roads

    request = {"design": design({"urban": roads(3)}), "scenarios": [continuous()],
               "n_sim": 5, "seed": 7}
    code, payload = run_main(capsys, "run-sensitivity", write_json(tmp_path / "p.json", request))
    assert code == 0 and payload["artifact_kind"] == "s2_power_sensitivity"
    request["n_sim"] = 0
    code, payload = run_main(capsys, "run-sensitivity", write_json(tmp_path / "q.json", request))
    assert code == 2 and "error" in payload


def test_evaluate_registered_synthetic_records(tmp_path, capsys):
    from test_s2_gate import SCORE_METRIC, paired, registration

    request = {"records": paired(40), "registration": registration(), "scope": "synthetic",
               "adapter_metric": SCORE_METRIC}
    code, payload = run_main(capsys, "evaluate-registered", write_json(tmp_path / "g.json",
                                                                       request))
    assert payload["artifact_kind"] == "s2_registered_gate_evaluation"
    assert code == (0 if payload["status"] == "passed" else 2)
    request["registration"] = registration(margin=None)
    code, payload = run_main(capsys, "evaluate-registered", write_json(tmp_path / "h.json",
                                                                       request))
    assert code == 2


def test_inspect_actual_never_admits_scores(tmp_path, capsys):
    from test_s2_actual import BINDING, make_run

    run, receipts = make_run(tmp_path)
    request = {"receipts": receipts, "binding": BINDING, "scope": "synthetic"}
    code, payload = run_main(capsys, "inspect-actual", write_json(tmp_path / "a.json", request),
                             "--root", run)
    assert payload["score_admitted"] is False
    assert code == (0 if payload["status"] == "passed" else 2)


def test_tensor_manifest_build_and_compare(tmp_path, capsys):
    from test_tensor_manifest import FILE, SHA_A

    (tmp_path / "model.safetensors").write_bytes(FILE)
    code, first = run_main(capsys, "build-tensor-manifest", "--root", tmp_path, "--scope",
                           "synthetic", "model.safetensors")
    assert code == 0 and first["status"] == "passed"
    assert SHA_A in json.dumps(first)
    a = write_json(tmp_path / "a.json", first)
    code, comparison = run_main(capsys, "compare-tensor-manifests", a, a)
    assert code == 0 and comparison["tensors_identical"] is True


def test_readiness_report_seal_and_verify_round_trip(tmp_path, capsys):
    from test_s2_readiness import complete_case

    evidence_root = tmp_path / "evidence-root"
    evidence_root.mkdir()
    plan, index, schedule, record = complete_case(evidence_root)
    request = write_json(tmp_path / "readiness.json", {"plan": plan, "evidence_index": index,
                                                        "schedule": schedule,
                                                        "action_record": record})
    code, readiness = run_main(capsys, "readiness", request, "--root", evidence_root)
    assert code == 0 and readiness["cloud_launch_authorized"] is False
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(bundle / "readiness.json", readiness)
    index = {"artifact_kind": "s2_bundle_index", "schema_version": 1, "scope": "synthetic",
             "bundle_id": "cli-test", "required_roles": ["report"],
             "entries": [{"path": "readiness.json", "role": "report",
                          "restriction_class": "public_safe", "licensed_lineage": False,
                          "replay": None}]}
    code, sealed = run_main(capsys, "seal", write_json(tmp_path / "index.json", index),
                            "--root", bundle)
    assert code == 0 and sealed["sealed"]
    code, verified = run_main(capsys, "verify-bundle", "--root", bundle)
    assert code == 0 and verified["verified"]
    code, verified = run_main(capsys, "verify-bundle", "--root", bundle, "--manifest",
                              write_json(tmp_path / "other.json", {"x": 1}))
    assert code == 2 and "manifest_mismatch" in {i["code"] for i in verified["issues"]}
    report_request = {"scope": "synthetic", "evidence": {}, "readiness": readiness}
    code, report = run_main(capsys, "report", write_json(tmp_path / "r.json", report_request))
    assert code == 0 and report["milestones"]["S2_ACCEPTED"]["status"] == "not_run"


def test_prepare_full_sweep_cli(tmp_path, capsys):
    from s2_packet_fixture import full_sweep_request

    path = write_json(tmp_path / "sweep.json", full_sweep_request())
    code, payload = run_main(capsys, "prepare-full-sweep", path)
    assert code == 0 and len(payload["units"]) == 2646
    assert payload["commands_executed"] is False
    request = full_sweep_request()
    request["execution_order"]["job_ids"] = []
    code, payload = run_main(capsys, "prepare-full-sweep",
                             write_json(tmp_path / "bad.json", request))
    assert code == 2 and not payload["prepared"]


@pytest.mark.parametrize("command", ["describe-validity", "audit-population", "run-sensitivity",
                                     "report", "evaluate-registered", "summarize-telemetry"])
def test_unknown_request_keys_are_structured_errors(tmp_path, capsys, command):
    code, payload = run_main(capsys, command,
                             write_json(tmp_path / "x.json", {"unexpected": 1}))
    assert code == 2 and payload["spend_authorized"] is False and "unexpected" in payload["error"]


@pytest.mark.parametrize("error", [AttributeError, KeyError, IndexError])
def test_structural_exceptions_become_error_json(tmp_path, capsys, monkeypatch, error):
    """Review B2-07: a structural exception prints the documented error JSON, exit 2."""
    from gate import s2

    def broken(_args):
        raise error("synthetic structural defect")

    text, _handler, key, arguments = s2.COMMANDS["report"]
    monkeypatch.setitem(s2.COMMANDS, "report", (text, broken, key, arguments))
    code, payload = run_main(capsys, "report", write_json(tmp_path / "r.json", {}))
    assert code == 2 and payload["spend_authorized"] is False
    assert "synthetic structural defect" in payload["error"]


def test_malformed_report_and_seal_inputs_are_structured(tmp_path, capsys):
    """Review B2-07: list-shaped sections and roles are issues, not tracebacks."""
    readiness = {"artifact_kind": "s2_readiness_report", "schema_version": 1,
                 "scope": "synthetic", "status": "passed", "sections": [1],
                 "next_blocking_checks": []}
    code, report = run_main(capsys, "report", write_json(
        tmp_path / "r.json", {"scope": "synthetic", "evidence": {}, "readiness": readiness}))
    assert code == 2 and "invalid_readiness" in {i["code"] for i in report["issues"]}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.txt").write_text("hello\n")
    index = {"artifact_kind": "s2_bundle_index", "schema_version": 1, "scope": "synthetic",
             "bundle_id": "cli-test", "required_roles": [],
             "entries": [{"path": "a.txt", "role": ["report"],
                          "restriction_class": "public_safe", "licensed_lineage": False,
                          "replay": None}]}
    code, sealed = run_main(capsys, "seal", write_json(tmp_path / "index.json", index),
                            "--root", bundle)
    assert code == 2 and not sealed["sealed"] and "unknown_role" in {
        i["code"] for i in sealed["issues"]}


def test_authority_line_matches_what_the_verifier_accepts(tmp_path, capsys):
    """The printed line authorizes exactly that artifact and nothing else (D-024 G-round)."""
    from gate.s2 import main
    from gate.s2_authority import verify_authority
    from gate.s2_identity import identity_hash

    artifact = {"artifact_kind": "s2_action_record", "schema_version": 1, "x": [1, 2]}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(artifact))
    assert main(["authority-line", "--kind", "s2_action_record", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    digest = identity_hash(artifact)
    assert result["line"] == f"authorizes: s2_action_record sha256:{digest}"
    assert result["grants_authority"] is False
    decisions = ("# Decision log\n\n- **D-901 · 2026-11-09 · Synthetic approval.** Text.\n"
                 f"  - {result['line']}\n").encode()
    granted = verify_authority(decisions, decision_id="D-901",
                               artifact_kind="s2_action_record", artifact_sha256=digest)
    assert granted["authorized"] is True
    other = verify_authority(decisions, decision_id="D-901",
                             artifact_kind="s2_action_record", artifact_sha256="0" * 64)
    assert other["authorized"] is False


def test_authority_line_refuses_unknown_kinds(tmp_path):
    from gate.s2 import main

    path = tmp_path / "x.json"
    path.write_text("{}")
    with pytest.raises(SystemExit):
        main(["authority-line", "--kind", "anything", str(path)])
